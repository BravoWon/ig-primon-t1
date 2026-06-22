#!/usr/bin/env python
"""Make the SLICE recommendations DRILLABLE: cross-check the UCB candidate sites against actual
existing well density (ALL leases, not just the modeled subset). UCB chases high v_hat, which sits
where producers already are -> likely saturated infill. So we (a) measure how drilled-up each
candidate is, and (b) re-rank for OPEN step-out acreage: high v_hat+sigma AND low local density.

    python slice_sanity.py
"""
from __future__ import annotations
import csv, re
import numpy as np
from scipy.spatial import cKDTree
import value_geometry as vg
import slice_acquisition as sa

PROD = vg.PROD
KM = 111.0  # km per degree latitude


def density_field():
    """ALL CKU leases with coordinates -> a well-count tree (weighted by max_wells)."""
    lat, lon, nw = [], [], []
    for r in csv.DictReader(open(PROD, encoding="utf-8")):
        if r["county"].upper() not in vg.CKU:
            continue
        try:
            la, lo = float(r["lat"]), float(r["lon"])
            mw = int(r["max_wells"] or 1)
        except (ValueError, KeyError):
            continue
        if -103 < lo < -97 and 37 < la < 41:
            lat.append(la); lon.append(lo); nw.append(max(mw, 1))
    lat, lon, nw = np.array(lat), np.array(lon), np.array(nw)
    mlat = lat.mean()
    XY = np.c_[lat, lon * np.cos(np.radians(mlat))]
    return cKDTree(XY), nw, mlat


def counts(tree, nw, mlat, a, o, km):
    r = km / KM
    j = tree.query_ball_point([a, o * np.cos(np.radians(mlat))], r)
    return len(j), int(nw[j].sum()) if j else 0


def main():
    W = vg.load()
    Xfull, names = vg.featurize(W)
    shelf = sa.shelf_mask(W, Xfull, names)
    cands = sa.grid_sites(W, shelf, Xfull, names, top=400)   # (a,o,vh,sg,ucb,dmin)
    tree, nw, mlat = density_field()
    print(f"[SLICE sanity] {len(cands)} shelf grid candidates; density from "
          f"{len(nw)} CKU leases ({nw.sum()} wells)")

    # enrich each candidate with local well density (within 3 km and 8 km)
    rich = []
    for a, o, vh, sg, u, dmin in cands:
        l3, w3 = counts(tree, nw, mlat, a, o, 3.0)
        l8, w8 = counts(tree, nw, mlat, a, o, 8.0)
        rich.append(dict(a=a, o=o, vh=vh, sg=sg, u=u, dmin=dmin, w3=w3, w8=w8))
    w8arr = np.array([r["w8"] for r in rich])
    print(f"  candidate local density (wells within 8 km): "
          f"p10={np.percentile(w8arr,10):.0f} p50={np.median(w8arr):.0f} p90={np.percentile(w8arr,90):.0f}")

    print("\n[A] ORIGINAL top-10 by UCB -- how saturated are they?")
    print("   lat      lon     ucb   wells<3km  wells<8km  verdict")
    for r in sorted(rich, key=lambda r: -r["u"])[:10]:
        v = "MATURE infill" if r["w8"] >= np.median(w8arr) else "step-out"
        print(f"  {r['a']:7.3f} {r['o']:8.3f}  {r['u']:+.2f}    {r['w3']:3}        {r['w8']:3}     {v}")
    top10 = sorted(rich, key=lambda r: -r["u"])[:10]
    print(f"  --> {sum(r['w8']>=np.median(w8arr) for r in top10)}/10 of the UCB top-10 sit in "
          f"above-median-density (saturated) acreage")

    # [B] OPEN step-out re-rank: keep only low-density cells, then rank by UCB
    open_thresh = np.percentile(w8arr, 35)
    openc = [r for r in rich if r["w8"] <= open_thresh]
    print(f"\n[B] OPEN step-out re-rank (wells<8km <= p35={open_thresh:.0f}): "
          f"{len(openc)} open candidates; top-10 by UCB among open acreage:")
    print("   lat      lon     v_hat  sigma  ucb   wells<8km  nearest(km)")
    for r in sorted(openc, key=lambda r: -r["u"])[:10]:
        print(f"  {r['a']:7.3f} {r['o']:8.3f}  {r['vh']:+.2f}  {r['sg']:.2f}  {r['u']:+.2f}    "
              f"{r['w8']:3}      {r['dmin']*KM:.1f}")

    # honest summary
    best_open = max(openc, key=lambda r: r["u"]) if openc else None
    best_inf = max(rich, key=lambda r: r["u"])
    print(f"\n[SUMMARY] best infill UCB={best_inf['u']:+.2f} (wells<8km={best_inf['w8']}) vs "
          f"best OPEN UCB={best_open['u']:+.2f} (wells<8km={best_open['w8']})"
          if best_open else "  no open candidates")
    print(f"  predicted-value give-up for going open: "
          f"{best_inf['vh']-best_open['vh']:+.2f} log-bbl "
          f"({100*(np.exp(best_inf['vh']-best_open['vh'])-1):+.0f}% best12)" if best_open else "")


if __name__ == "__main__":
    main()
