"""POWERED test: pooled across 20 CBP counties. Per-county detrended San Andres structure, perf-indexed
to wells completed in San Andres, clean type-curve IP, single-well oil leases. Spatial-block CV with a
BOOTSTRAP CI on the AUC lift (significance earned, not eyeballed), per-county consistency, and the
depletion-inversion (vintage) split. n -> ~1000+ resolves the Andrews +0.057 trend.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from andrews_perfs_tops import marker_of
from perf_indexed_test import clean_ip
from value_test import DATA, auc, design2d, logit_fit, logit_pred, spearman


def sa_residuals(pt, coords):
    by_cc = defaultdict(list)
    for api, v in pt.items():
        if api not in coords or v["elev"] is None:
            continue
        sa = [d for d, m in ((dd, marker_of(nn)) for nn, dd in v["tops"]) if m == "SAN ANDRES"]
        if not sa:
            continue
        subsea = v["elev"] - min(sa)
        if -8000 <= subsea <= 2000:
            la, lo = coords[api]; by_cc[v["cc"]].append((api, subsea, la, lo))
    resid = {}
    for cc, wells in by_cc.items():
        sub = np.array([w[1] for w in wells]); lat = np.array([w[2] for w in wells]); lon = np.array([w[3] for w in wells])
        if len(wells) >= 40:
            A = np.c_[np.ones_like(lon), design2d(lon, lat)]
            coef, *_ = np.linalg.lstsq(A, sub, rcond=None); r = sub - A @ coef
        else:
            r = sub - sub.mean()
        for i, w in enumerate(wells):
            resid[w[0]] = float(r[i])
    return resid


def cv_oof(X, y, blocks, ub):
    oof = np.full(len(y), np.nan)
    for b in ub:
        te = np.array([blk == b for blk in blocks]); tr = ~te
        if y[tr].sum() < 3 or (1 - y[tr]).sum() < 3:
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
        w = logit_fit((X[tr] - mu) / sd, y[tr]); oof[te] = logit_pred(w, (X[te] - mu) / sd)
    return oof


def boot_lift(y, ob, of, n=1000):
    rng = np.random.default_rng(0); lifts = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        lifts.append(auc(y[s], of[s]) - auc(y[s], ob[s]))
    lifts = np.array(lifts)
    return float(lifts.mean()), float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))


def run():
    print("[POWERED CBP San Andres structure->value test]\n")
    pt = json.load(open(os.path.join(DATA, "cbp_perfs_tops.json")))
    coords = {r["api8"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(open(os.path.join(DATA, "cbp_coords.csv")))}
    horiz = set(json.load(open(os.path.join(DATA, "cbp_horiz.json"))))
    series = json.load(open(os.path.join(DATA, "cbp_series.json")))
    setup = json.load(open(os.path.join(DATA, "cbp_setup.json")))
    api_oil = {a: [tuple(k.split("}")) for k in ks] for a, ks in setup["api_oil"].items()}
    nwells = setup["lease_nwells"]
    resid = sa_residuals(pt, coords)
    print("  San Andres structural residuals (per-county detrended): %d wells" % len(resid))

    rows = []
    for api, v in pt.items():
        if "SAN ANDRES" not in v["perf_forms"] or api in horiz or api not in coords or api not in resid or api not in api_oil:
            continue
        for k in api_oil[api]:
            ks = "}".join(k)
            if nwells.get(ks, 9) != 1 or ks not in series:
                continue
            ip = clean_ip(series[ks])
            if not ip or ip <= 0:
                continue
            first = min(series[ks]) if series[ks] else "999999"
            la, lo = coords[api]
            rows.append((la, lo, resid[api], float(ip), v["cc"], first)); break
    print("  POWERED sample (perf-indexed SA single-well verticals): %d\n" % len(rows))
    if len(rows) < 200:
        print("  unexpectedly small; stopping"); return

    la = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows])
    res = np.array([r[2] for r in rows]); ip = np.array([r[3] for r in rows])
    cc = np.array([r[4] for r in rows]); first = np.array([r[5] for r in rows])
    y = (ip >= np.quantile(ip, 0.75)).astype(float)

    sp = spearman(res, np.log1p(ip))
    hi = ip[res > 150]; loo = ip[res < -150]
    ratio = np.median(hi) / max(np.median(loo), 1)
    print("  [HEADLINE] Spearman(resid, log IP) = %+.3f | hi/lo ratio = %.2fx (n_hi=%d n_lo=%d)" % (sp, ratio, len(hi), len(loo)))

    blocks = [(int(la[i] / 0.1), int(lo[i] / 0.1)) for i in range(len(rows))]; ub = list(set(blocks))
    ob = cv_oof(design2d(lo, la), y, blocks, ub)
    of = cv_oof(np.c_[design2d(lo, la), res], y, blocks, ub)
    m = ~np.isnan(ob) & ~np.isnan(of)
    ab, af = auc(y[m], ob[m]), auc(y[m], of[m])
    bm, blo, bhi = boot_lift(y[m], ob[m], of[m])
    print("\n  [SPATIAL-BLOCK CV — %d blocks, %d scored]" % (len(ub), m.sum()))
    print("    baseline AUC %.3f -> +structure %.3f" % (ab, af))
    print("    LIFT %+.3f  95%% bootstrap CI [%+.3f, %+.3f]  -> %s" % (bm, blo, bhi,
          "SIGNIFICANT (CI excludes 0)" if (blo > 0 or bhi < 0) else "not significant (CI spans 0)"))

    print("\n  [DEPLETION-INVERSION — vintage split]")
    for lab, sel in [("OLD (pre-93, depleted)", first == "199301"), ("NEW (post-93)", first > "199301")]:
        if sel.sum() > 40:
            r2 = ip[sel][res[sel] > 150]; r3 = ip[sel][res[sel] < -150]
            rr = np.median(r2) / max(np.median(r3), 1) if (len(r2) > 5 and len(r3) > 5) else float("nan")
            print("    %-26s n=%-4d Spearman %+.3f  hi/lo %.2fx" % (lab, sel.sum(), spearman(res[sel], np.log1p(ip[sel])), rr))

    print("\n  [PER-COUNTY consistency] Spearman(resid, log IP) by county (n>=40):")
    for c in sorted(set(cc)):
        sc = cc == c
        if sc.sum() >= 40:
            print("    %s  n=%-4d  %+.3f" % (c, sc.sum(), spearman(res[sc], np.log1p(ip[sc]))))

    json.dump({"n": len(rows), "spearman": sp, "hi_lo_ratio": ratio, "auc_base": ab, "auc_struct": af,
               "lift": bm, "lift_ci": [blo, bhi], "significant": bool(blo > 0 or bhi < 0)},
              open(os.path.join(DATA, "cbp_powered_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    run()
