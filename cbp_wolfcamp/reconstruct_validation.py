"""Leave-out RECONSTRUCTION validation -> the 'future mapper'. Delete wells (by spatial block so they
can't cheat off near neighbors), infill them from the rest (offset-production field) + structure, compare
infill to the hidden truth. If it reconstructs held-out wells, it predicts undrilled locations.
Honest baseline: spatial-offset interpolation alone vs + structure (the bypassed signal). Then render the map.
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


def run():
    print("[Reconstruction validation -> future mapper — CBP San Andres]\n")
    pt = json.load(open(os.path.join(DATA, "cbp_perfs_tops.json")))
    coords = {r["api8"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(open(os.path.join(DATA, "cbp_coords.csv")))}
    horiz = set(json.load(open(os.path.join(DATA, "cbp_horiz.json"))))
    series = json.load(open(os.path.join(DATA, "cbp_series.json")))
    setup = json.load(open(os.path.join(DATA, "cbp_setup.json")))
    api_oil = {a: [tuple(k.split("}")) for k in ks] for a, ks in setup["api_oil"].items()}
    nwells = setup["lease_nwells"]; resid = sa_residuals(pt, coords)
    rows = []
    for api, v in pt.items():
        if "SAN ANDRES" not in v["perf_forms"] or api in horiz or api not in coords or api not in resid or api not in api_oil:
            continue
        for k in api_oil[api]:
            ks = "}".join(k)
            if nwells.get(ks, 9) == 1 and ks in series:
                ip = clean_ip(series[ks])
                if ip and ip > 0:
                    rows.append((coords[api][0], coords[api][1], resid[api], ip))
                break
    la = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows])
    res = np.array([r[2] for r in rows]); lp = np.log1p(np.array([r[3] for r in rows]))
    n = len(rows); print("  wells: %d" % n)
    blocks = np.array([hash((int(la[i] / 0.15), int(lo[i] / 0.15))) for i in range(n)]); ub = list(set(blocks))

    # leave-BLOCK-out spatial infill (a deleted well predicted ONLY from other blocks -> no neighbor cheating)
    idw = np.full(n, np.nan)
    for b in ub:
        te = blocks == b; tr = ~te
        for i in np.where(te)[0]:
            d2 = (lo[tr] - lo[i]) ** 2 + (la[tr] - la[i]) ** 2 + 1e-6
            w = 1 / d2 ** 1.5; idw[i] = (w * lp[tr]).sum() / w.sum()

    def blockcv(X):
        pred = np.full(n, np.nan)
        for b in ub:
            te = blocks == b; tr = ~te
            A = np.c_[np.ones(tr.sum()), X[tr]]; coef, *_ = np.linalg.lstsq(A, lp[tr], rcond=None)
            pred[te] = np.c_[np.ones(te.sum()), X[te]] @ coef
        return pred
    p_sp = blockcv(idw[:, None]); p_fu = blockcv(np.c_[idw, -res]); m = ~np.isnan(p_sp)
    s_sp = spearman(p_sp[m], lp[m]); s_fu = spearman(p_fu[m], lp[m])
    rng = np.random.default_rng(0); d = []
    for _ in range(2000):
        s = rng.integers(0, m.sum(), m.sum())
        d.append(spearman(p_fu[m][s], lp[m][s]) - spearman(p_sp[m][s], lp[m][s]))
    d = np.array(d)
    print("\n  [RECONSTRUCTION — infill held-out wells from the rest]")
    print("    FUTURE MAPPER (offset-interpolation only):   Spearman(infill, truth) = %+.3f" % s_sp)
    print("    + structure (bypassed signal):               Spearman(infill, truth) = %+.3f" % s_fu)
    print("    structure increment: %+.3f  95%% CI [%+.3f, %+.3f] -> %s"
          % (d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5),
             "structure adds (CI excludes 0)" if (np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0) else "structure adds n.s. (offsets carry it)"))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        A = np.c_[np.ones(n), idw, -res]; coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
        gx = np.linspace(lo.min(), lo.max(), 90); gy = np.linspace(la.min(), la.max(), 90); GX, GY = np.meshgrid(gx, gy)
        Z = np.full_like(GX, np.nan); near = np.zeros_like(GX)
        for i in range(GX.shape[0]):
            for j in range(GX.shape[1]):
                d2 = (lo - GX[i, j]) ** 2 + (la - GY[i, j]) ** 2; near[i, j] = np.sqrt(d2.min())
                w = 1 / (d2 + 1e-4) ** 1.5; gi = (w * lp).sum() / w.sum()
                gr = (w * (-res)).sum() / w.sum()
                Z[i, j] = coef[0] + coef[1] * gi + coef[2] * gr
        Z[near > 0.06] = np.nan
        fig, ax = plt.subplots(figsize=(9, 7))
        c = ax.contourf(GX, GY, np.expm1(Z), 18, cmap="viridis"); plt.colorbar(c, ax=ax, label="predicted clean-IP oil (bbl)")
        ax.scatter(lo, la, s=5, c="k", alpha=.25); ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_title("FUTURE MAPPER — predicted San Andres productivity\n(offset field + bypassed-structure, reconstruction-validated)")
        png = os.path.join(DATA, "future_mapper.png"); plt.tight_layout(); plt.savefig(png, dpi=110); print("\n  wrote %s" % png)
    except Exception as e:                                            # noqa: BLE001
        print("\n  (map skipped: %r)" % e)


if __name__ == "__main__":
    run()
