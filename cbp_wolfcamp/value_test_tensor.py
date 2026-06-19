"""TENSOR vs STANDARD: does a multi-formation common-mode structural factor predict production better
than the single-horizon residual?  (Operationalizes "tensorize -> new knowledge vs standard methods.")

PRE-REG (honest): build a wells x formations detrended-residual matrix over the column (San Andres ..
Strawn), jointly decompose (SVD) -> PC1 = the covariant common-mode structure (the deformation coherent
across all layers, denoised of single-horizon pick error). Compare, under spatial-block CV against
cumulative oil:
  baseline (location+depth)  vs  +single-horizon residual (STANDARD)  vs  +tensor common-mode (TENSOR).
Falsifier: the tensor common-mode adds no out-of-fold lift over the single-horizon residual.
Expectation: cleaner structure measure, but bounded by ground truth -> small-to-no lift (method != signal).
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_test import DATA, auc, design2d, logit_fit, logit_pred, spearman

MARKERS = ["SAN ANDRES", "GRAYBURG", "GLORIETA", "SPRABERRY", "WOLFCAMP", "STRAWN"]
STD_HORIZON = "SAN ANDRES"


def detrended_residuals(coords):
    by_form = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(os.path.join(DATA, "andrews_tops.csv"), encoding="utf-8")):
        nm = r["formation_name"].upper()
        if "DISPOSAL" in nm:
            continue
        mk = next((m for m in MARKERS if m in nm), None)
        if not mk:
            continue
        try:
            d = float(r["depth_md"]); s = float(r["subsea"])
        except ValueError:
            continue
        if 1000 <= d <= 16000 and -12000 <= s <= 2000:
            by_form[mk][r["api8"]].append(s)
    resid = {}
    for mk in MARKERS:
        apis = [a for a in by_form[mk] if a in coords]
        if len(apis) < 50:
            continue
        sub = np.array([np.mean(by_form[mk][a]) for a in apis])
        lon = np.array([coords[a][1] for a in apis]); lat = np.array([coords[a][0] for a in apis])
        A = np.c_[np.ones_like(lon), design2d(lon, lat)]
        coef, *_ = np.linalg.lstsq(A, sub, rcond=None)
        r_ = sub - A @ coef
        resid[mk] = {apis[i]: float(r_[i]) for i in range(len(apis))}
        print("  %-12s residual on %d wells" % (mk, len(apis)))
    return resid


def common_mode(resid, coords):
    forms = [m for m in MARKERS if m in resid]
    apis = [a for a in coords if sum(a in resid[m] for m in forms) >= 3]
    M = np.array([[resid[m].get(a, np.nan) for m in forms] for a in apis])
    colmean = np.nanmean(M, 0); idx = np.where(np.isnan(M)); M[idx] = np.take(colmean, idx[1])
    Mc = (M - M.mean(0)) / (M.std(0) + 1e-9)
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    pc1 = U[:, 0] * S[0]
    if np.corrcoef(pc1, np.nanmean(M, 1))[0, 1] < 0:
        pc1 = -pc1
    var = (S ** 2 / (S ** 2).sum())[0]
    print("\n  tensor common-mode (PC1) over %d wells x %d formations; PC1 explains %.0f%% of joint variance"
          % (len(apis), len(forms), 100 * var))
    print("  PC1 loadings:", {forms[i]: round(float(Vt[0, i]), 2) for i in range(len(forms))})
    return {apis[i]: float(pc1[i]) for i in range(len(apis))}


def run():
    print("[TENSOR vs STANDARD — multi-formation common-mode for value prediction]\n")
    coords = {r["api8"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(open(os.path.join(DATA, "andrews_coords.csv")))}
    resid = detrended_residuals(coords)
    cf = common_mode(resid, coords)
    sa = resid.get(STD_HORIZON, {})

    hz = {r["api8"] for r in csv.DictReader(open(os.path.join(DATA, "andrews_horiz.csv")))}
    link = json.load(open(os.path.join(DATA, "andrews_linkage_all.json")))
    api_oil = {a: [tuple(k.split("}")) for k in ks] for a, ks in link["api_oil"].items()}
    nwells = link["lease_nwells"]
    prod = {(r["ogc"], r["dist"], r["lease"]): r for r in csv.DictReader(open(os.path.join(DATA, "andrews_lease_production_all.csv")))}

    rows = []
    for a in (set(sa) & set(cf)) - hz:
        if a not in api_oil:
            continue
        for k in api_oil[a]:
            if nwells.get("}".join(k), 9) != 1 or k not in prod:
                continue
            cum = float(prod[k]["cum_oil"])
            if cum <= 0:
                continue
            la, lo = coords[a]
            rows.append((la, lo, sa[a], cf[a], cum))
            break
    print("\n  clean single-well-lease verticals (have SA residual + common-mode + production): %d" % len(rows))
    if len(rows) < 80:
        print("  too few; stopping"); return

    la = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows])
    sar = np.array([r[2] for r in rows]); cfac = np.array([r[3] for r in rows]); cum = np.array([r[4] for r in rows])
    print("\n  [HEADLINE] Spearman vs log cumulative oil:")
    print("    single-horizon San Andres residual (STANDARD): %+.3f" % spearman(sar, np.log1p(cum)))
    print("    tensor common-mode factor        (TENSOR)    : %+.3f" % spearman(cfac, np.log1p(cum)))
    print("    (cross-check: corr(common-mode, single-horizon) = %.2f)" % np.corrcoef(cfac, sar)[0, 1])

    y = (cum >= np.quantile(cum, 0.75)).astype(float)
    base = np.c_[design2d(lo, la), sar * 0 + np.interp(la, [la.min(), la.max()], [0, 0])]  # placeholder depth-less base
    base = design2d(lo, la)
    blocks = [(int(la[i] / 0.1), int(lo[i] / 0.1)) for i in range(len(rows))]; ub = list(set(blocks))

    def cv(X):
        oof = np.full(len(y), np.nan)
        for b in ub:
            te = np.array([blocks[i] == b for i in range(len(blocks))]); tr = ~te
            if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2:
                continue
            mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
            w = logit_fit((X[tr] - mu) / sd, y[tr]); oof[te] = logit_pred(w, (X[te] - mu) / sd)
        return oof

    models = {"baseline (location)": base,
              "+ STANDARD (single-horizon)": np.c_[base, sar],
              "+ TENSOR (common-mode)": np.c_[base, cfac]}
    oofs = {k: cv(X) for k, X in models.items()}
    m = np.all([~np.isnan(v) for v in oofs.values()], 0)
    print("\n  [SPATIAL-BLOCK CV — %d blocks, %d scored] out-of-fold AUC:" % (len(ub), m.sum()))
    res = {}
    for k, v in oofs.items():
        res[k] = auc(y[m], v[m]); print("    %-32s %.3f" % (k, res[k]))
    lift_std = res["+ STANDARD (single-horizon)"] - res["baseline (location)"]
    lift_ten = res["+ TENSOR (common-mode)"] - res["baseline (location)"]
    print("\n    STANDARD lift over baseline: %+.3f" % lift_std)
    print("    TENSOR   lift over baseline: %+.3f" % lift_ten)
    print("    TENSOR vs STANDARD: %+.3f" % (lift_ten - lift_std))
    verdict = ("TENSOR adds new predictive knowledge over standard" if lift_ten - lift_std > 0.02
               else "NO new knowledge: tensor ~ standard (method does not mint signal beyond ground truth)")
    print("\n  [VERDICT] %s" % verdict)
    json.dump({"spearman_standard": spearman(sar, np.log1p(cum)), "spearman_tensor": spearman(cfac, np.log1p(cum)),
               "auc": res, "lift_standard": lift_std, "lift_tensor": lift_ten, "verdict": verdict},
              open(os.path.join(DATA, "tensor_vs_standard_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    run()
