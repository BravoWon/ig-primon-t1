"""PROPER dimensionalized test: is there a coupling between the GEOLOGY block and the PRODUCTION-DYNAMICS
block (decline-curve components from the time series), via Canonical Correlation Analysis?

  GEOLOGY block G   : [SA residual, multi-formation common-mode, Grayburg residual, Wolfcamp residual, SA subsea(TVD)]
  PRODUCTION block P: [log peak-rate qi, decline Di, log cum/EUR, months-on, flatness(b-proxy), time-to-peak]
                       -- the "embedded knowledge in the series", per well.

CCA finds the geology- and production-combinations that are maximally correlated (the shared latent
component). GUARDRAILS (the proper statistics): (1) regularized CCA; (2) PERMUTATION TEST on the top
canonical correlation (CCA overfits at small n -> must beat its own shuffled null); (3) OUT-OF-FOLD
canonical correlation under spatial-block CV (learn directions on train blocks, test the coupling on
held-out blocks). A component counts only if it survives permutation AND generalizes out-of-fold.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_test import DATA
from value_test_tensor import common_mode, detrended_residuals

np.random.seed(0)
GNAMES = ["SA_resid", "SA_subsea", "common_mode"]
PNAMES = ["log_qi", "Di", "log_cum", "months_on", "flat", "t_peak"]


def sa_subsea(coords):
    d = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(DATA, "andrews_tops.csv"), encoding="utf-8")):
        nm = r["formation_name"].upper()
        if "SAN ANDRES" in nm and "DISPOSAL" not in nm:
            try:
                s = float(r["subsea"])
            except ValueError:
                continue
            if -8000 <= s <= 2000:
                d[r["api8"]].append(s)
    return {a: float(np.mean(v)) for a, v in d.items()}


def decline_features(ser):
    oil = np.array([v for _, v in sorted(ser.items())], float)
    if (oil > 0).sum() < 12 or oil.max() <= 0:
        return None
    qi = oil.max(); tpk = int(np.argmax(oil)); post = oil[tpk:]; t = np.arange(len(post)); mask = post > 0
    Di = -np.polyfit(t[mask], np.log(post[mask]), 1)[0] if mask.sum() >= 6 else 0.0
    cum = oil.sum(); months_on = int((oil > 0).sum())
    best12 = max((oil[i:i + 12].sum() for i in range(max(1, len(oil) - 11))), default=0.0)
    flat = best12 / (12 * qi + 1e-9)
    return [float(np.log(qi)), float(Di), float(np.log(cum + 1)), float(months_on), flat, float(tpk)]


def _cca(Gc, Pc, reg=1e-2):
    n = len(Gc)
    Sxx = np.cov(Gc, rowvar=False) + reg * np.eye(Gc.shape[1])
    Syy = np.cov(Pc, rowvar=False) + reg * np.eye(Pc.shape[1])
    Sxy = Gc.T @ Pc / (n - 1)
    def isq(S):
        w, V = np.linalg.eigh(S); return V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-8))) @ V.T
    Kx, Ky = isq(Sxx), isq(Syy)
    U, s, Vt = np.linalg.svd(Kx @ Sxy @ Ky)
    return s, Kx @ U[:, 0], Ky @ Vt[0]


def cca(G, P):
    return _cca((G - G.mean(0)) / (G.std(0) + 1e-9), (P - P.mean(0)) / (P.std(0) + 1e-9))


def run():
    print("[PROPER CCA: geology block  <->  production-dynamics block]\n")
    coords = {r["api8"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(open(os.path.join(DATA, "andrews_coords.csv")))}
    resid = detrended_residuals(coords); cf = common_mode(resid, coords); sas = sa_subsea(coords)
    SA, GB, WC = resid.get("SAN ANDRES", {}), resid.get("GRAYBURG", {}), resid.get("WOLFCAMP", {})
    series = json.load(open(os.path.join(DATA, "andrews_lease_series.json")))
    link = json.load(open(os.path.join(DATA, "andrews_linkage_all.json")))
    api_oil = {a: [tuple(k.split("}")) for k in ks] for a, ks in link["api_oil"].items()}
    nwells = link["lease_nwells"]
    hz = {r["api8"] for r in csv.DictReader(open(os.path.join(DATA, "andrews_horiz.csv")))}

    rows = []
    for a in (set(SA) & set(sas)) - hz:
        if a not in api_oil or a not in coords:
            continue
        for k in api_oil[a]:
            ks = "}".join(k)
            if nwells.get(ks, 9) != 1 or ks not in series:
                continue
            df = decline_features({m: v for m, v in series[ks].items()})
            if df is None:
                continue
            g = [SA[a], sas[a], cf.get(a, 0.0)]                  # robust geology block (common-mode imputed 0 if absent)
            rows.append((coords[a][0], coords[a][1], g, df)); break
    print("\n  clean sample (single-well vertical, full geology + decline series): %d wells" % len(rows))
    if len(rows) < 60:
        print("  too few; stopping"); return
    la = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows])
    G = np.array([r[2] for r in rows]); P = np.array([r[3] for r in rows])

    s, wx, wy = cca(G, P)
    print("\n  in-sample canonical correlations:", np.round(s, 3))
    print("  geology loadings   (comp 1):", dict(zip(GNAMES, np.round(wx, 2))))
    print("  production loadings(comp 1):", dict(zip(PNAMES, np.round(wy, 2))))

    obs = s[0]; null = np.array([cca(G, P[np.random.permutation(len(P))])[0][0] for _ in range(500)])
    p = (np.sum(null >= obs) + 1) / 501
    print("\n  [PERMUTATION TEST] top canonical corr = %.3f | shuffled null mean %.3f, 95th %.3f | p = %.3f"
          % (obs, null.mean(), np.percentile(null, 95), p))

    blocks = [(int(la[i] / 0.1), int(lo[i] / 0.1)) for i in range(len(rows))]
    foldc = []
    for b in set(blocks):
        te = np.array([blocks[i] == b for i in range(len(blocks))]); tr = ~te
        if tr.sum() < 25 or te.sum() < 3:
            continue
        mx, sx = G[tr].mean(0), G[tr].std(0) + 1e-9; my, sy = P[tr].mean(0), P[tr].std(0) + 1e-9
        _, wx2, wy2 = _cca((G[tr] - mx) / sx, (P[tr] - my) / sy)
        gv = ((G[te] - mx) / sx) @ wx2; pv = ((P[te] - my) / sy) @ wy2
        if np.std(gv) > 0 and np.std(pv) > 0:
            foldc.append((np.corrcoef(gv, pv)[0, 1], te.sum()))
    oos = np.average([c for c, _ in foldc], weights=[w for _, w in foldc]) if foldc else float("nan")
    print("  [OUT-OF-FOLD] mean per-block test canonical corr (spatial CV, %d folds) = %.3f" % (len(foldc), oos))

    real = p < 0.05 and oos > 0.15
    print("\n  [VERDICT] %s" % ("REAL geology<->production-dynamics coupling (survives permutation AND generalizes)"
          if real else "NULL: in-sample CCA is overfit; permutation/out-of-fold do not support a real coupling"))
    json.dump({"n": len(rows), "canon_corrs": s.tolist(), "perm_p": float(p), "oos_canon_corr": float(oos),
               "geology_loadings": dict(zip(GNAMES, wx.tolist())), "prod_loadings": dict(zip(PNAMES, wy.tolist())),
               "verdict_real": bool(real)}, open(os.path.join(DATA, "cca_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    run()
