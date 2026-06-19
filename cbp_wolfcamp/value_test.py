"""Stage D: the value test + calibrated probability model.

Does sitting on a corroborated structural high LIFT productivity, OVER a structure-blind baseline,
on spatially-held-out wells?  Primary sample: vertical Wolfcamp wells on single-well oil leases,
post-1993 first production (clean per-well IP = best 12-month oil).

  baseline B0      : smooth spatial trend (lon,lat,lon^2,lat^2,lon*lat) + subsea depth + vintage year
  B0 + structure   : + detrended structural residual (the local closure signal)
  test             : out-of-fold AUC / Brier uplift under SPATIAL-BLOCK CV (leakage defense)
  decision metric  : does structure rank top-quartile producers into the top quintile? on/off-high IP ratio
  probability      : calibrated P(top-quartile | features); reliability diagram
Structure-real != structure-pays: this is the arbiter.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
CTY = "003"; CELL = 0.02; IP_COL = "best12_oil"


def design2d(lon, lat):
    return np.c_[lon, lat, lon * lon, lon * lat, lat * lat]


def load():
    coords = {r["api8"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(open(os.path.join(DATA, "andrews_coords.csv")))}
    sub = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(DATA, "wolfcamp_tops.csv"), encoding="utf-8")):
        if r["county"] != CTY:
            continue
        nm = r["formation_name"].upper()
        if "WOLFCAMP" not in nm or "DISPOSAL" in nm:
            continue
        try:
            d = float(r["depth_md"]); s = float(r["subsea_top"])
        except ValueError:
            continue
        if 2000 <= d <= 16000 and -12000 <= s <= -1000:
            sub[r["api8"]].append(s)
    subsea = {a: float(np.mean(v)) for a, v in sub.items()}
    link = json.load(open(os.path.join(DATA, "andrews_linkage.json")))
    prod = {(r["ogc"], r["dist"], r["lease"]): r for r in csv.DictReader(open(os.path.join(DATA, "andrews_lease_production.csv")))}
    return coords, subsea, link, prod


def residual_field(verts, coords, subsea):
    pts = [(coords[a][0], coords[a][1], subsea[a]) for a in verts if a in coords and a in subsea]
    lat = np.array([p[0] for p in pts]); lon = np.array([p[1] for p in pts]); ss = np.array([p[2] for p in pts])
    cellv = defaultdict(list)
    for k in range(len(ss)):
        cellv[(int((lon[k] - lon.min()) / CELL), int((lat[k] - lat.min()) / CELL))].append(ss[k])
    cx = np.array([lon.min() + (i + .5) * CELL for (i, j) in cellv]); cy = np.array([lat.min() + (j + .5) * CELL for (i, j) in cellv])
    cz = np.array([np.median(v) for v in cellv.values()])
    A = np.c_[np.ones_like(cx), design2d(cx, cy)]
    coef, *_ = np.linalg.lstsq(A, cz, rcond=None)
    return coef


def logit_fit(X, y, l2=2.0, iters=4000, lr=0.3):
    Xb = np.c_[np.ones(len(X)), X]; w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y)
        w -= lr * g
    return w


def logit_pred(w, X):
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), X] @ w, -30, 30)))


def auc(y, p):
    y = np.asarray(y); order = np.argsort(p, kind="mergesort"); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run():
    print("[Value test + calibrated probability — Andrews Wolfcamp verticals]\n")
    coords, subsea, link, prod = load()
    verts = set(link["verticals"])
    coef = residual_field(verts, coords, subsea)
    api_oil = {a: [tuple(k.split("}")) for k in ks] for a, ks in link["api_oil"].items()}
    nwells = link["lease_nwells"]

    rows = []
    for a in verts:
        if a not in coords or a not in subsea or a not in api_oil:
            continue
        for k in api_oil[a]:
            if nwells.get("}".join(k), 9) != 1 or k not in prod:
                continue
            pr = prod[k]
            if pr["first_ym"] < "199301":
                continue
            ip = float(pr[IP_COL])
            if ip <= 0:
                continue
            la, lo = coords[a]; resid = subsea[a] - (np.r_[1, design2d(np.array([lo]), np.array([la]))[0]] @ coef)
            rows.append((a, la, lo, subsea[a], float(resid), ip, int(pr["first_ym"][:4])))
            break
    print("  clean sample (single-well oil lease, post-93, IP>0): %d wells" % len(rows))
    if len(rows) < 80:
        print("  sample too small for CV; stopping"); return

    la = np.array([r[1] for r in rows]); lo = np.array([r[2] for r in rows]); ss = np.array([r[3] for r in rows])
    res = np.array([r[4] for r in rows]); ip = np.array([r[5] for r in rows]); yr = np.array([r[6] for r in rows])
    logip = np.log1p(ip); y = (ip >= np.quantile(ip, 0.75)).astype(float)

    print("  IP (best 12-mo oil, bbl): median %.0f | p90 %.0f | top-quartile cutoff %.0f" % (np.median(ip), np.quantile(ip, .9), np.quantile(ip, .75)))
    print("\n  [HEADLINE associations]")
    print("    Spearman(structural residual, log IP): %+.3f" % spearman(res, logip))
    hi = ip[res > 150]; loo = ip[res < -150]
    if len(hi) > 5 and len(loo) > 5:
        print("    median IP on highs(>+150ft): %.0f (n=%d) vs lows(<-150ft): %.0f (n=%d) -> ratio %.2fx"
              % (np.median(hi), len(hi), np.median(loo), len(loo), np.median(hi) / max(np.median(loo), 1)))

    # features
    Xs = np.c_[design2d(lo, la), ss, yr]                  # structure-blind baseline B0
    Xf = np.c_[Xs, res]                                    # B0 + structure
    blocks = [(int(la[i] / 0.1), int(lo[i] / 0.1)) for i in range(len(rows))]
    ub = list(set(blocks))

    def cv(X):
        oof = np.full(len(y), np.nan)
        for b in ub:
            te = np.array([blocks[i] == b for i in range(len(blocks))]); tr = ~te
            if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2:
                continue
            mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
            w = logit_fit((X[tr] - mu) / sd, y[tr])
            oof[te] = logit_pred(w, (X[te] - mu) / sd)
        return oof

    ob, of = cv(Xs), cv(Xf); m = ~np.isnan(ob) & ~np.isnan(of)
    print("\n  [SPATIAL-BLOCK CV — %d blocks, %d wells scored]" % (len(ub), m.sum()))
    ab, af = auc(y[m], ob[m]), auc(y[m], of[m]); bb, bf = np.mean((ob[m] - y[m]) ** 2), np.mean((of[m] - y[m]) ** 2)
    print("    out-of-fold AUC : baseline %.3f -> +structure %.3f  (lift %+.3f)" % (ab, af, af - ab))
    print("    out-of-fold Brier: baseline %.4f -> +structure %.4f  (%+.4f, lower better)" % (bb, bf, bf - bb))

    # decision lift: rank by structure score, top quintile capture of true top-quartile producers
    sidx = np.argsort(-of[m]); topq = int(0.2 * m.sum()); cap = y[m][sidx][:topq].sum() / max(y[m].sum(), 1)
    print("    decision lift: top-quintile by P(struct) captures %.0f%% of true top-quartile producers (base rate 25%%)" % (100 * cap))

    verdict = "STRUCTURE PAYS" if (af - ab) > 0.02 and spearman(res, logip) > 0 else "NULL / structure does not add lift"
    print("\n  [VERDICT vs pre-registration] %s" % verdict)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        ax[0].scatter(res, ip, s=10, alpha=.5); ax[0].set_xlabel("structural residual (ft)"); ax[0].set_ylabel("IP best-12mo oil (bbl)")
        ax[0].set_title("IP vs structure  (Spearman %+.2f)" % spearman(res, logip)); ax[0].set_yscale("log")
        # reliability
        pr = of[m]; yy = y[m]; bins = np.linspace(0, 1, 6); mids = []; obs = []
        for i in range(5):
            sel = (pr >= bins[i]) & (pr < bins[i + 1] + (1 if i == 4 else 0))
            if sel.sum() > 3:
                mids.append(pr[sel].mean()); obs.append(yy[sel].mean())
        ax[1].plot([0, 1], [0, 1], "k--"); ax[1].plot(mids, obs, "o-"); ax[1].set_xlabel("predicted P"); ax[1].set_ylabel("observed freq")
        ax[1].set_title("Calibration (out-of-fold)")
        # lift curve
        order = np.argsort(-of[m]); cumy = np.cumsum(yy[order]) / yy.sum(); frac = np.arange(1, m.sum() + 1) / m.sum()
        ax[2].plot(frac, cumy); ax[2].plot([0, 1], [0, 1], "k--"); ax[2].set_xlabel("frac wells drilled (ranked by P)"); ax[2].set_ylabel("frac of top producers captured")
        ax[2].set_title("Lift curve (AUC %.2f)" % af)
        png = os.path.join(DATA, "andrews_value_test.png"); plt.tight_layout(); plt.savefig(png, dpi=110); print("  wrote %s" % png)
    except Exception as e:                                       # noqa: BLE001
        print("  (plot skipped: %r)" % e)

    json.dump({"n": len(rows), "spearman_resid_logip": spearman(res, logip), "auc_base": ab, "auc_struct": af,
               "auc_lift": af - ab, "brier_base": bb, "brier_struct": bf, "top_quintile_capture": cap,
               "verdict": verdict}, open(os.path.join(DATA, "value_test_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    run()
