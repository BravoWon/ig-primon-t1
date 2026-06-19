"""Powered re-run with PER-WELL W-10 test rate (no lease-allocation, no single-well restriction).
Crosswalk W-10 (district+lease+well) -> API via OG_WELL_COMPLETION; join to San Andres perf-indexed
verticals; re-run structure->production with bootstrap-CI'd spatial-block CV. Breaks the n-wall.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from andrews_perfs_tops import marker_of
from cbp_value_test import sa_residuals
from pdq_production import DATA, dsv
from value_test import auc, design2d, logit_fit, logit_pred, spearman


def norm(d, l, w):
    return (d.strip(), l.strip().lstrip("0"), w.strip().lstrip("0"))


def crosswalk():
    fh, ix = dsv("OG_WELL_COMPLETION_DATA_TABLE.dsv")
    xw = {}
    for line in fh:
        f = line.rstrip("\n").split("}")
        if len(f) <= ix["API_UNIQUE_NO"] or f[ix["OIL_GAS_CODE"]].strip() != "O":
            continue
        api8 = f[ix["API_COUNTY_CODE"]].strip().zfill(3) + f[ix["API_UNIQUE_NO"]].strip().zfill(5)
        xw[norm(f[ix["DISTRICT_NO"]], f[ix["LEASE_NO"]], f[ix["WELL_NO"]])] = api8
    return xw


def run():
    print("[W-10 per-well powered test — San Andres]\n")
    xw = crosswalk(); print("  crosswalk keys (oil completions): %d" % len(xw))
    rate = {}
    for r in csv.DictReader(open(os.path.join(DATA, "w10_well_rates.csv"))):
        a = xw.get(norm(r["dst"], r["lease"], r["wellno"]))
        if a and float(r["oil_test_bbld"]) > 0:
            rate[a] = float(r["oil_test_bbld"])
    print("  W-10 rates joined to API: %d" % len(rate))

    pt = json.load(open(os.path.join(DATA, "cbp_perfs_tops.json")))
    coords = {r["api8"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(open(os.path.join(DATA, "cbp_coords.csv")))}
    horiz = set(json.load(open(os.path.join(DATA, "cbp_horiz.json"))))
    resid = sa_residuals(pt, coords)

    rows = []
    for api, v in pt.items():
        if "SAN ANDRES" not in v["perf_forms"] or api in horiz or api not in coords or api not in resid or api not in rate:
            continue
        la, lo = coords[api]
        rows.append((la, lo, resid[api], rate[api]))
    print("  POWERED sample (SA perf-indexed verticals w/ W-10 rate): %d\n" % len(rows))
    if len(rows) < 200:
        print("  join too thin (W-10 = currently-active wells only); n=%d" % len(rows))
        if len(rows) < 80:
            return

    la = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows])
    res = np.array([r[2] for r in rows]); ip = np.array([r[3] for r in rows])
    y = (ip >= np.quantile(ip, 0.75)).astype(float)
    sp = spearman(res, np.log1p(ip))
    hi = ip[res > 150]; loo = ip[res < -150]
    print("  Spearman(resid, log test-rate) = %+.3f | hi/lo = %.2fx (n_hi=%d n_lo=%d)"
          % (sp, np.median(hi) / max(np.median(loo), 1) if len(hi) > 5 and len(loo) > 5 else float("nan"), len(hi), len(loo)))

    blocks = [(int(la[i] / 0.1), int(lo[i] / 0.1)) for i in range(len(rows))]; ub = list(set(blocks))

    def cv(X):
        oof = np.full(len(y), np.nan)
        for b in ub:
            te = np.array([z == b for z in blocks]); tr = ~te
            if y[tr].sum() < 3 or (1 - y[tr]).sum() < 3:
                continue
            mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
            w = logit_fit((X[tr] - mu) / sd, y[tr]); oof[te] = logit_pred(w, (X[te] - mu) / sd)
        return oof
    ob = cv(design2d(lo, la)); of = cv(np.c_[design2d(lo, la), res]); m = ~np.isnan(ob) & ~np.isnan(of)
    ab, af = auc(y[m], ob[m]), auc(y[m], of[m])
    rng = np.random.default_rng(0); d = []
    for _ in range(1000):
        s = rng.integers(0, m.sum(), m.sum()); d.append(auc(y[m][s], of[m][s]) - auc(y[m][s], ob[m][s]))
    d = np.array(d)
    print("\n  [SPATIAL-BLOCK CV] baseline %.3f -> +structure %.3f | lift %+.3f CI [%+.3f, %+.3f] -> %s"
          % (ab, af, d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5),
             "significant" if (np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0) else "n.s."))
    print("\n  -> per-well test rate (no allocation) at n=%d: %s" % (len(rows),
          "confirms weak/inverted structure" if sp < 0.1 else "structure signal present"))


if __name__ == "__main__":
    run()
