"""FULL multivariate model (the maximal tensorization): strength = where (location, structure, depth) +
how-made (perf net-pay, #perfs, position) + when (vintage) -> merit (clean IP). Powered n=908, 20 counties.
Staged so precision is watched as dimensions stack; bootstrap CI on the full-model lift over location;
standardized feature importance so we see what actually drives strength. Measures the precision CEILING
honestly instead of asserting it.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from andrews_perfs_tops import marker_of
from cbp_value_test import sa_residuals
from merit_map import sa_details
from perf_indexed_test import clean_ip
from value_test import DATA, auc, design2d, logit_fit, logit_pred


def cv_oof(X, y, blocks, ub):
    oof = np.full(len(y), np.nan)
    for b in ub:
        te = np.array([blk == b for blk in blocks]); tr = ~te
        if y[tr].sum() < 3 or (1 - y[tr]).sum() < 3:
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
        w = logit_fit((X[tr] - mu) / sd, y[tr]); oof[te] = logit_pred(w, (X[te] - mu) / sd)
    return oof


def boot(y, a, b, n=1000):
    rng = np.random.default_rng(0); d = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y)); d.append(auc(y[s], b[s]) - auc(y[s], a[s]))
    d = np.array(d); return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run():
    print("[FULL multivariate model — powered CBP San Andres, n~900]\n")
    pt = json.load(open(os.path.join(DATA, "cbp_perfs_tops.json")))
    coords = {r["api8"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(open(os.path.join(DATA, "cbp_coords.csv")))}
    horiz = set(json.load(open(os.path.join(DATA, "cbp_horiz.json"))))
    series = json.load(open(os.path.join(DATA, "cbp_series.json")))
    setup = json.load(open(os.path.join(DATA, "cbp_setup.json")))
    api_oil = {a: [tuple(k.split("}")) for k in ks] for a, ks in setup["api_oil"].items()}
    nwells = setup["lease_nwells"]
    resid = sa_residuals(pt, coords)

    rows = []
    for api, v in pt.items():
        if "SAN ANDRES" not in v["perf_forms"] or api in horiz or api not in coords or api not in resid or api not in api_oil:
            continue
        det = sa_details(v)
        if det is None:
            continue
        sa = [d for d, m in ((dd, marker_of(nn)) for nn, dd in v["tops"]) if m == "SAN ANDRES"]
        subsea = v["elev"] - min(sa)
        for k in api_oil[api]:
            ks = "}".join(k)
            if nwells.get(ks, 9) != 1 or ks not in series:
                continue
            ip = clean_ip(series[ks])
            if not ip or ip <= 0:
                continue
            first = min(series[ks]) if series[ks] else "199301"
            yr = int(first[:4]) if first[:4].isdigit() else 1993
            la, lo = coords[api]
            rows.append((la, lo, resid[api], subsea, np.log1p(det["thick"]), det["npf"], det["pos"], yr, ip)); break
    print("  full-model sample: %d\n" % len(rows))
    if len(rows) < 200:
        print("  too few; stopping"); return

    A = np.array(rows, float)
    la, lo, res, sub, lth, npf, pos, yr, ip = [A[:, i] for i in range(9)]
    y = (ip >= np.quantile(ip, 0.75)).astype(float)
    loc = design2d(lo, la)
    blocks = [(int(la[i] / 0.1), int(lo[i] / 0.1)) for i in range(len(rows))]; ub = list(set(blocks))

    stages = [("M0 location (where, coarse)", loc),
              ("M1 + depth", np.c_[loc, sub]),
              ("M2 + structure", np.c_[loc, sub, res]),
              ("M3 + completion (thick,#perf,pos)", np.c_[loc, sub, res, lth, npf, pos]),
              ("M4 + vintage  = FULL TENSOR", np.c_[loc, sub, res, lth, npf, pos, yr])]
    base_oof = cv_oof(loc, y, blocks, ub); m0 = ~np.isnan(base_oof)
    print("  [SPATIAL-BLOCK CV] out-of-fold AUC (precision), staged:")
    full_oof = None
    for name, X in stages:
        oof = cv_oof(X, y, blocks, ub); m = ~np.isnan(oof) & m0
        print("    %-36s %.3f" % (name, auc(y[m], oof[m])))
        full_oof = oof
    m = ~np.isnan(full_oof) & m0
    bm, blo, bhi = boot(y[m], base_oof[m], full_oof[m])
    print("\n  FULL TENSOR vs location-only: lift %+.3f  95%% CI [%+.3f, %+.3f]  -> %s"
          % (bm, blo, bhi, "real gain" if blo > 0 else "no real gain"))

    # feature importance (standardized full model, all data)
    X = np.c_[lo, la, sub, res, lth, npf, pos, yr]
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    w = logit_fit(Xs, y)
    names = ["lon", "lat", "depth", "structure", "log_netpay", "n_perfs", "perf_pos", "vintage"]
    imp = sorted(zip(names, w[1:]), key=lambda t: -abs(t[1]))
    print("\n  [what drives strength] standardized logistic weights:")
    for n_, c_ in imp:
        print("    %-12s %+.2f" % (n_, c_))
    print("\n  full-model AUC %.3f -> top-quartile precision/recall at that AUC is MODERATE; the rest is"
          % auc(y[m], full_oof[m]))
    print("  irreducible variance (sub-seismic heterogeneity, completion execution, allocation noise).")
    json.dump({"n": len(rows), "full_auc": auc(y[m], full_oof[m]), "full_vs_loc_lift": bm,
               "ci": [blo, bhi], "importance": dict(imp)}, open(os.path.join(DATA, "cbp_full_model_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    run()
