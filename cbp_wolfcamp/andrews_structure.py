"""Derive Wolfcamp structural closures (anticlines) on the CBP from well control -- Andrews County first.

Valid technique, not recited names:
  1. join Wolfcamp subsea tops (wellbore parse) to GIS surface coords by API
  2. decluster: cell-median subsea per grid cell (wells are clustered)
  3. regional-residual separation: subtract a quadratic regional trend -> residual = LOCAL structure
  4. closure detection: positive residual highs (candidate anticlines) with well support
  5. CONFOUNDER-#1 check: is there control on flanks/lows (residual spread both signs), or only crests?
Outputs: residual map PNG (if matplotlib), closure inventory CSV, summary JSON.  Phase 1 = verticals (MD~=TVD).
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rrc_gis as G

CTY = "003"                                   # Andrews County RRC code
BOX = G.CBP_SUBAREAS["andrews_co"]
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CELL = 0.02                                    # ~2 km grid cell


def load_tops():
    agg = defaultdict(list)
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
            agg[r["api8"]].append(s)
    return {api: float(np.mean(v)) for api, v in agg.items()}    # mean over benches/picks per well


def pull_coords():
    cache = os.path.join(DATA, "andrews_coords.csv")
    if os.path.exists(cache):
        coord = {r["api8"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(open(cache))}
        print("  loaded %d Andrews coords (attribute-query API LIKE '%s%%', 100%% join)" % (len(coord), CTY))
        return coord
    rows = G.fetch_where("API LIKE '%s%%'" % CTY, "API,GIS_LAT83,GIS_LONG83")   # attribute query, no bbox clip
    coord = {}
    for a in rows:
        api = (a.get("API") or "").strip()
        if len(api) >= 8:
            try:
                coord[api[:8]] = (float(a["GIS_LAT83"]), float(a["GIS_LONG83"]))
            except (TypeError, ValueError):
                pass
    print("  GIS attribute pull: %d rows -> %d usable APIs" % (len(rows), len(coord)))
    return coord


def run():
    print("[Andrews Wolfcamp structure from well control]\n")
    tops = load_tops()
    coord = pull_coords()
    pts = [(coord[api][0], coord[api][1], ss, api) for api, ss in tops.items() if api in coord]
    print("  Wolfcamp-top wells: %d | joined to coords: %d (%.0f%%)\n"
          % (len(tops), len(pts), 100 * len(pts) / max(len(tops), 1)))
    if len(pts) < 100:
        print("  too few joined wells to map; stopping"); return
    lat = np.array([p[0] for p in pts]); lon = np.array([p[1] for p in pts]); ss = np.array([p[2] for p in pts])

    # decluster: cell-median subsea
    ix = ((lon - lon.min()) / CELL).astype(int); iy = ((lat - lat.min()) / CELL).astype(int)
    cellv = defaultdict(list)
    for k in range(len(ss)):
        cellv[(ix[k], iy[k])].append(ss[k])
    cx = np.array([lon.min() + (i + .5) * CELL for (i, j) in cellv])
    cy = np.array([lat.min() + (j + .5) * CELL for (i, j) in cellv])
    cz = np.array([np.median(v) for v in cellv.values()])
    print("  declustered cells: %d (median %d wells/cell)" % (len(cz), int(np.median([len(v) for v in cellv.values()]))))

    # regional-residual separation: quadratic regional trend
    def design(x, y):
        return np.c_[np.ones_like(x), x, y, x * x, x * y, y * y]
    coef, *_ = np.linalg.lstsq(design(cx, cy), cz, rcond=None)
    cresid = cz - design(cx, cy) @ coef
    wresid = ss - design(lon, lat) @ coef            # well-level residual

    # CONFOUNDER #1: structural variation in the drilled population
    pos = int((wresid > 150).sum()); neg = int((wresid < -150).sum())
    print("\n  [CONFOUNDER-#1 CHECK] regional trend removed; well residual spread:")
    print("    residual stdev: %.0f ft | range: %.0f..%.0f ft" % (wresid.std(), wresid.min(), wresid.max()))
    print("    wells on highs (>+150ft): %d (%.0f%%) | on lows (<-150ft): %d (%.0f%%) | mid: %d"
          % (pos, 100 * pos / len(wresid), neg, 100 * neg / len(wresid), len(wresid) - pos - neg))
    testable = pos > 0.1 * len(wresid) and neg > 0.1 * len(wresid)
    print("    -> %s: %s" % ("TESTABLE" if testable else "RANGE-RESTRICTED",
          "control on both highs and flanks/lows" if testable else "drilled population lacks low-side control"))

    # closure detection: positive-residual cells that are local maxima with support
    cells = list(cellv.keys()); rmap = {cells[i]: cresid[i] for i in range(len(cells))}
    closures = []
    for n, (i, j) in enumerate(cells):
        r = cresid[n]
        if r < 200 or len(cellv[(i, j)]) < 2:
            continue
        nbrs = [rmap[(i + di, j + dj)] for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if (di or dj) and (i + di, j + dj) in rmap]
        if nbrs and r >= max(nbrs):
            closures.append((cx[n], cy[n], r, len(cellv[(i, j)])))
    closures.sort(key=lambda c: -c[2])
    print("\n  candidate anticlinal closures (local residual highs, >=200 ft, >=2 wells): %d" % len(closures))
    print("    %-9s %-9s %8s %6s" % ("lon", "lat", "relief_ft", "wells"))
    for lo, la, r, nw in closures[:10]:
        print("    %-9.3f %-9.3f %8.0f %6d" % (lo, la, r, nw))

    # artifacts
    with open(os.path.join(DATA, "andrews_closures.csv"), "w") as f:
        f.write("lon,lat,residual_ft,wells\n")
        for lo, la, r, nw in closures:
            f.write("%.4f,%.4f,%.0f,%d\n" % (lo, la, r, nw))
    json.dump({"wells_joined": len(pts), "cells": len(cz), "resid_std": round(float(wresid.std()), 1),
               "pct_high": round(100 * pos / len(wresid), 1), "pct_low": round(100 * neg / len(wresid), 1),
               "testable": bool(testable), "closures": len(closures)},
              open(os.path.join(DATA, "andrews_structure_summary.json"), "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        s0 = ax[0].scatter(lon, lat, c=ss, s=8, cmap="viridis")
        ax[0].set_title("Wolfcamp subsea top (ft) — Andrews Co. well control"); plt.colorbar(s0, ax=ax[0])
        vmax = np.percentile(np.abs(wresid), 95)
        s1 = ax[1].scatter(lon, lat, c=wresid, s=8, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for lo, la, r, nw in closures:
            ax[1].plot(lo, la, "k^", ms=6 + r / 200)
        ax[1].set_title("Residual (regional trend removed) — ▲ = candidate anticlinal closures")
        plt.colorbar(s1, ax=ax[1])
        for a in ax:
            a.set_xlabel("lon"); a.set_ylabel("lat")
        png = os.path.join(DATA, "andrews_structure.png")
        plt.tight_layout(); plt.savefig(png, dpi=110); print("\n  wrote map: %s" % png)
    except Exception as e:                                       # noqa: BLE001
        print("\n  (matplotlib unavailable: %r -- numeric artifacts written)" % e)
    return closures


if __name__ == "__main__":
    run()
