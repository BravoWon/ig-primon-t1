"""The one map ground truth voted YES on: bypassed-oil targeting in depleted CBP San Andres.
Coherent synthesis: structure (confirmed inversion -> lows hold bypassed oil) x maturity (depletion) x
proven production. Validated FIRST (bootstrap CI: do structural lows outproduce highs in mature fields,
at power?) THEN rendered. Honest: the edge is real & confirmed but MODEST -- a redevelopment high-grader.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbp_value_test import sa_residuals
from perf_indexed_test import clean_ip
from value_test import DATA, spearman


def ratio_ci(res, prod, mask, n=3000):
    r = res[mask]; p = prod[mask]
    lows = p[r < -150]; highs = p[r > 150]
    if len(lows) < 10 or len(highs) < 10:
        return None
    obs = np.median(lows) / max(np.median(highs), 1.0)
    rng = np.random.default_rng(0); bs = []
    for _ in range(n):
        bs.append(np.median(lows[rng.integers(0, len(lows), len(lows))]) /
                  max(np.median(highs[rng.integers(0, len(highs), len(highs))]), 1.0))
    return obs, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(lows), len(highs)


def run():
    print("[Bypassed-oil targeting map — depleted CBP San Andres]\n")
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
        for k in api_oil[api]:
            ks = "}".join(k)
            if nwells.get(ks, 9) != 1 or ks not in series:
                continue
            ip = clean_ip(series[ks])
            if not ip or ip <= 0:
                continue
            old = (min(series[ks]) == "199301")
            rows.append((coords[api][0], coords[api][1], resid[api], ip, old)); break
    la = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows])
    res = np.array([r[2] for r in rows]); prod = np.array([r[3] for r in rows]); old = np.array([r[4] for r in rows])
    print("  sample: %d  (mature/old: %d)\n" % (len(rows), old.sum()))

    print("  [VALIDATION FIRST — do structural LOWS outproduce HIGHS? (lows/highs median, bootstrap CI)]")
    for lab, mask in [("ALL San Andres verticals", np.ones(len(rows), bool)), ("MATURE/depleted only", old)]:
        rc = ratio_ci(res, prod, mask)
        if rc:
            obs, clo, chi, nl, nh = rc
            sig = "SIGNIFICANT (CI excludes 1.0)" if (clo > 1 or chi < 1) else "not significant"
            print("    %-26s lows/highs = %.2fx  95%% CI [%.2f, %.2f]  (n_lo=%d n_hi=%d) -> %s" % (lab, obs, clo, chi, nl, nh, sig))
    print("    Spearman(-residual, log prod) all=%+.3f  mature=%+.3f  (positive = lows produce more = bypassed)"
          % (spearman(-res, np.log1p(prod)), spearman(-res[old], np.log1p(prod[old]))))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        # bypassed potential field: structural low (-resid) IDW over the producing footprint, mature-weighted
        gx = np.linspace(lo.min(), lo.max(), 90); gy = np.linspace(la.min(), la.max(), 90)
        GX, GY = np.meshgrid(gx, gy); Z = np.full_like(GX, np.nan); near = np.zeros_like(GX)
        wlo, wla, wsc = lo[old], la[old], (-res[old])                 # mature wells, bypassed score = structural low
        for i in range(GX.shape[0]):
            for j in range(GX.shape[1]):
                d2 = (wlo - GX[i, j]) ** 2 + (wla - GY[i, j]) ** 2
                near[i, j] = np.sqrt(d2.min())
                w = 1 / (d2 + 1e-4) ** 1.5; Z[i, j] = (w * wsc).sum() / w.sum()
        Z[near > 0.06] = np.nan                                       # mask to the producing footprint only
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        c = ax[0].contourf(GX, GY, Z, 18, cmap="YlOrRd"); plt.colorbar(c, ax=ax[0], label="bypassed-oil potential (structural low)")
        ax[0].scatter(wlo, wla, s=5, c="k", alpha=.3)
        ax[0].set_title("BYPASSED-OIL TARGET MAP — mature San Andres\n(hot = drained-crest flanks holding remaining oil)")
        sc = ax[1].scatter(lo, la, c=res, s=10, cmap="RdBu_r", vmin=-np.percentile(np.abs(res), 90), vmax=np.percentile(np.abs(res), 90))
        ax[1].set_title("structural residual (blue lows = targets, red highs = drained)"); plt.colorbar(sc, ax=ax[1])
        for a in ax:
            a.set_xlabel("lon"); a.set_ylabel("lat")
        png = os.path.join(DATA, "bypassed_oil_map.png"); plt.tight_layout(); plt.savefig(png, dpi=110)
        print("\n  wrote map: %s" % png)
    except Exception as e:                                            # noqa: BLE001
        print("\n  (map skipped: %r)" % e)


if __name__ == "__main__":
    run()
