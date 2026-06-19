"""Layer merit onto the details: does the completion-detail reservoir-quality proxy (perforated net pay,
count, depth, position-in-zone) predict MERIT (production) where structure did not? Build the map.

Tests under spatial-block CV: baseline(location) vs +STRUCTURE vs +DETAILS(own) vs +OFFSET(pre-drill).
Honest: own-perf is known post-completion & partly an operator choice (endogenous); the pre-drill map is
the OFFSET net-pay field (k-NN of neighbors), which you'd have before drilling. Produces the heat map.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from andrews_perfs_tops import marker_of
from perf_indexed_test import clean_ip
from value_test import DATA, auc, design2d, logit_fit, logit_pred, spearman
from value_test_tensor import detrended_residuals


def sa_details(v):
    elev = v["elev"]
    mk = sorted([(d, marker_of(n)) for n, d in v["tops"] if marker_of(n)], key=lambda x: x[0])
    sa = [d for d, m in mk if m == "SAN ANDRES"]
    if not sa or elev is None:
        return None
    sa_top = min(sa); below = [d for d, m in mk if d > sa_top]
    sa_base = min((min(below) if below else 10 ** 9), sa_top + 700)   # San Andres ~few-hundred ft, not 1500+ (domain fix)
    pin = [(fr, to) for fr, to in v["perfs"] if sa_top <= (fr + to) / 2 < sa_base and 0 < to - fr < 300]
    if not pin:
        return None
    thick = sum(to - fr for fr, to in pin); mid = float(np.mean([(fr + to) / 2 for fr, to in pin]))
    return {"thick": float(thick), "npf": len(pin), "perf_subsea": elev - mid,
            "pos": (mid - sa_top) / max(sa_base - sa_top, 1)}


def cv_auc(X, y, blocks, ub):
    oof = np.full(len(y), np.nan)
    for b in ub:
        te = np.array([blocks[i] == b for i in range(len(blocks))]); tr = ~te
        if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2:
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
        w = logit_fit((X[tr] - mu) / sd, y[tr]); oof[te] = logit_pred(w, (X[te] - mu) / sd)
    m = ~np.isnan(oof)
    return auc(y[m], oof[m]), m


def run():
    print("[MERIT MAP — completion-detail reservoir quality vs structure — San Andres]\n")
    coords = {r["api8"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(open(os.path.join(DATA, "andrews_coords.csv")))}
    SA = detrended_residuals(coords).get("SAN ANDRES", {})
    pt = json.load(open(os.path.join(DATA, "andrews_perfs_tops.json")))
    series = json.load(open(os.path.join(DATA, "andrews_lease_series.json")))
    link = json.load(open(os.path.join(DATA, "andrews_linkage_all.json")))
    api_oil = {a: [tuple(k.split("}")) for k in ks] for a, ks in link["api_oil"].items()}
    nwells = link["lease_nwells"]
    hz = {r["api8"] for r in csv.DictReader(open(os.path.join(DATA, "andrews_horiz.csv")))}

    rows = []
    for a, v in pt.items():
        if a in hz or a not in coords or a not in SA or a not in api_oil or "SAN ANDRES" not in v["perf_forms"]:
            continue
        det = sa_details(v)
        if det is None:
            continue
        ip = None
        for k in api_oil[a]:
            ks = "}".join(k)
            if nwells.get(ks, 9) == 1 and ks in series:
                ip = clean_ip(series[ks]); break
        if not ip or ip <= 0:
            continue
        la, lo = coords[a]
        rows.append((la, lo, SA[a], det["perf_subsea"], det["thick"], det["npf"], det["pos"], ip))
    print("  perf-indexed San Andres wells w/ details + merit: %d" % len(rows))
    if len(rows) < 80:
        print("  too few; stopping"); return

    la = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows])
    res = np.array([r[2] for r in rows]); psub = np.array([r[3] for r in rows])
    thick = np.array([r[4] for r in rows]); npf = np.array([r[5] for r in rows])
    pos = np.array([r[6] for r in rows]); ip = np.array([r[7] for r in rows])
    y = (ip >= np.quantile(ip, 0.75)).astype(float)
    blocks = [(int(la[i] / 0.1), int(lo[i] / 0.1)) for i in range(len(rows))]; ub = list(set(blocks))

    print("\n  Spearman vs log merit:  perf_thick %+.3f | n_perfs %+.3f | structure(resid) %+.3f"
          % (spearman(thick, np.log1p(ip)), spearman(npf, np.log1p(ip)), spearman(res, np.log1p(ip))))

    # offset (pre-drill) net-pay field: k-NN mean log-thick of OTHER wells
    lt = np.log1p(thick); knn = np.zeros(len(rows))
    for i in range(len(rows)):
        d2 = (la - la[i]) ** 2 + (lo - lo[i]) ** 2; d2[i] = 1e9
        nn = np.argsort(d2)[:8]; knn[i] = lt[nn].mean()

    base = np.c_[design2d(lo, la), psub]
    models = {"baseline (loc+depth)": base,
              "+ STRUCTURE": np.c_[base, res],
              "+ DETAILS own (thick,npf,pos)": np.c_[base, lt, npf, pos],
              "+ OFFSET net-pay (pre-drill)": np.c_[base, knn]}
    print("\n  [SPATIAL-BLOCK CV] out-of-fold AUC:")
    a0 = None
    for k, X in models.items():
        au, _ = cv_auc(X, y, blocks, ub)
        if a0 is None:
            a0 = au
        print("    %-34s %.3f  (lift %+.3f)" % (k, au, au - a0))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        gx = np.linspace(lo.min(), lo.max(), 60); gy = np.linspace(la.min(), la.max(), 60)
        GX, GY = np.meshgrid(gx, gy); Z = np.zeros_like(GX)
        for ii in range(GX.shape[0]):
            for jj in range(GX.shape[1]):
                d2 = (lo - GX[ii, jj]) ** 2 + (la - GY[ii, jj]) ** 2 + 1e-6
                w_ = 1 / d2 ** 1.5; Z[ii, jj] = (w_ * lt).sum() / w_.sum()      # IDW net-pay proxy
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        c = ax[0].contourf(GX, GY, Z, 20, cmap="YlOrRd")
        ax[0].set_title("MERIT MAP — San Andres net-pay proxy (IDW log perf-thickness)"); plt.colorbar(c, ax=ax[0])
        ax[0].scatter(lo, la, s=6, c="k", alpha=.3)
        sc = ax[1].scatter(lo, la, c=np.log1p(ip), s=14, cmap="viridis")
        ax[1].set_title("actual merit (log clean-IP oil)"); plt.colorbar(sc, ax=ax[1])
        for a in ax:
            a.set_xlabel("lon"); a.set_ylabel("lat")
        png = os.path.join(DATA, "san_andres_merit_map.png"); plt.tight_layout(); plt.savefig(png, dpi=110)
        print("\n  wrote map: %s" % png)
    except Exception as e:                                       # noqa: BLE001
        print("\n  (map skipped: %r)" % e)


if __name__ == "__main__":
    run()
