"""Multi-horizon corroboration of the Andrews Wolfcamp closures (internal triangulation, data in hand).

A real anticline STACKS: a closure on Wolfcamp should also show as a residual high on a shallower marker
above and a deeper marker below. A "closure" that appears only on Wolfcamp is likely a pick artifact.
Re-streams the cached wellbore file (Andrews only), builds detrended residual surfaces on 3 horizons,
measures spatial concordance, and reports which Wolfcamp closures are corroborated above AND below.
Phase 1: verticals (MD~=TVD).
"""
import csv
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
GZ = os.path.join(DATA, "dbf900.txt.gz")
CTY = "003"; CELL = 0.02
SHALLOW = ["SAN ANDRES", "GRAYBURG", "CLEAR FORK", "GLORIETA", "SPRABERRY", "DEAN"]
DEEP = ["ELLENBURGER", "DEVONIAN", "FUSSELMAN", "MISSISSIPPIAN", "STRAWN", "CANYON", "ATOKA"]


def parse_andrews():
    """{api8: {marker: subsea_ft}} for Andrews wells with a valid datum elevation."""
    wells = defaultdict(dict); cnt = Counter()
    api = elev = None; is3 = False; tops = []

    def flush():
        if is3 and elev:
            seen = {}
            for nm, dep in tops:
                for kw in [w for w in ([("WOLFCAMP", "WOLFCAMP")] + [(s, s) for s in SHALLOW] + [(d, d) for d in DEEP])]:
                    key, lab = kw
                    if key in nm and lab not in seen:
                        seen[lab] = elev - dep
            for lab, ss in seen.items():
                if -14000 <= ss <= 2000:
                    wells[api][lab] = ss; cnt[lab] += 1

    with gzip.open(GZ, "rt", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            c = line[:2]
            if c == "01":
                flush()
                api = line[2:10]; is3 = (line[2:5] == CTY); elev = None; tops = []
            elif c == "03" and is3:
                if elev is None:
                    e = line[85:89].strip()
                    if e.isdigit() and int(e) > 0:
                        elev = int(e)
            elif c == "09" and is3:
                nm = line[5:37].strip().upper(); dep = line[37:42].strip()
                if dep.isdigit() and 1000 < int(dep) < 30000:
                    tops.append((nm, int(dep)))
        flush()
    return wells, cnt


def residual_field(pts, lon0, lat0):
    lat = np.array([p[0] for p in pts]); lon = np.array([p[1] for p in pts]); ss = np.array([p[2] for p in pts])
    cellv = defaultdict(list)
    for k in range(len(ss)):
        cellv[(int((lon[k] - lon0) / CELL), int((lat[k] - lat0) / CELL))].append(ss[k])
    keys = list(cellv)
    cx = np.array([lon0 + (i + .5) * CELL for (i, j) in keys])
    cy = np.array([lat0 + (j + .5) * CELL for (i, j) in keys])
    cz = np.array([np.median(v) for v in cellv.values()])
    A = np.c_[np.ones_like(cx), cx, cy, cx * cx, cx * cy, cy * cy]
    coef, *_ = np.linalg.lstsq(A, cz, rcond=None)
    cres = cz - A @ coef
    return {keys[i]: (cx[i], cy[i], cres[i]) for i in range(len(keys))}


def concord(fa, fb):
    common = sorted(set(fa) & set(fb))
    if len(common) < 10:
        return None, len(common)
    a = np.array([fa[k][2] for k in common]); b = np.array([fb[k][2] for k in common])
    return float(np.corrcoef(a, b)[0, 1]), len(common)


def run():
    print("[Multi-horizon corroboration — Andrews]\n")
    coord = {r["api8"]: (float(r["lat"]), float(r["lon"]))
             for r in csv.DictReader(open(os.path.join(DATA, "andrews_coords.csv")))}
    wells, cnt = parse_andrews()
    print("  Andrews wells with tops+datum: %d" % len(wells))
    print("  marker coverage:", cnt.most_common(12))
    shallow = max(SHALLOW, key=lambda m: cnt.get(m, 0)); deep = max(DEEP, key=lambda m: cnt.get(m, 0))
    print("\n  chosen bracket:  SHALLOW=%s (%d)  |  WOLFCAMP (%d)  |  DEEP=%s (%d)"
          % (shallow, cnt.get(shallow, 0), cnt.get("WOLFCAMP", 0), deep, cnt.get(deep, 0)))

    def pts(marker):
        return [(coord[a][0], coord[a][1], w[marker]) for a, w in wells.items() if marker in w and a in coord]
    P = {"shallow": pts(shallow), "wolfcamp": pts("WOLFCAMP"), "deep": pts(deep)}
    allp = P["shallow"] + P["wolfcamp"] + P["deep"]
    lon0 = min(p[1] for p in allp); lat0 = min(p[0] for p in allp)
    F = {k: residual_field(v, lon0, lat0) for k, v in P.items() if len(v) >= 50}
    print("\n  residual fields built:", {k: len(v) for k, v in F.items()})

    print("\n  [SPATIAL CONCORDANCE of residual structure across horizons]")
    cws, nws = concord(F.get("wolfcamp", {}), F.get("shallow", {}))
    cwd, nwd = concord(F.get("wolfcamp", {}), F.get("deep", {}))
    print("    Wolfcamp vs %-10s : r=%s (n=%d shared cells)" % (shallow, ("%.2f" % cws) if cws is not None else "n/a", nws))
    print("    Wolfcamp vs %-10s : r=%s (n=%d shared cells)" % (deep, ("%.2f" % cwd) if cwd is not None else "n/a", nwd))

    # per-closure corroboration: Wolfcamp residual highs that are also high above AND below
    wf = F["wolfcamp"]
    highs = sorted([k for k in wf if wf[k][2] >= 200 and True], key=lambda k: -wf[k][2])
    corrob = 0; rows = []
    for (i, j) in highs:
        s = F.get("shallow", {}).get((i, j)); d = F.get("deep", {}).get((i, j))
        sv = s[2] if s else None; dv = d[2] if d else None
        ok = (sv is not None and sv > 0) and (dv is not None and dv > 0)
        if s or d:
            rows.append((wf[(i, j)][0], wf[(i, j)][1], wf[(i, j)][2], sv, dv, ok))
            if ok:
                corrob += 1
    print("\n  per-closure stacking (Wolfcamp residual high, sign above & below):")
    print("    %-9s %-9s %8s %9s %9s %s" % ("lon", "lat", "wolf_ft", "shallow", "deep", "stacked?"))
    for lo, la, wv, sv, dv, ok in rows[:14]:
        print("    %-9.3f %-9.3f %8.0f %9s %9s %s"
              % (lo, la, wv, ("%.0f" % sv) if sv is not None else "  --", ("%.0f" % dv) if dv is not None else "  --",
                 "YES" if ok else "no"))
    print("\n  corroborated (high on Wolfcamp AND shallow AND deep): %d of %d evaluable"
          % (corrob, len(rows)))

    json.dump({"shallow": shallow, "deep": deep, "marker_counts": cnt.most_common(15),
               "concord_wolf_shallow": cws, "concord_wolf_deep": cwd,
               "closures_evaluable": len(rows), "closures_stacked": corrob},
              open(os.path.join(DATA, "multi_horizon_summary.json"), "w"), indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
        for n, (k, ti) in enumerate([("shallow", shallow), ("wolfcamp", "WOLFCAMP"), ("deep", deep)]):
            f = F.get(k, {})
            if not f:
                continue
            xs = [v[0] for v in f.values()]; ys = [v[1] for v in f.values()]; rs = [v[2] for v in f.values()]
            vmax = np.percentile(np.abs(rs), 95)
            sc = ax[n].scatter(xs, ys, c=rs, s=22, marker="s", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax[n].set_title("%s residual (ft)" % ti); plt.colorbar(sc, ax=ax[n]); ax[n].set_xlabel("lon")
        ax[0].set_ylabel("lat")
        png = os.path.join(DATA, "andrews_multihorizon.png")
        plt.tight_layout(); plt.savefig(png, dpi=110); print("\n  wrote map: %s" % png)
    except Exception as e:                                       # noqa: BLE001
        print("\n  (plot skipped: %r)" % e)


if __name__ == "__main__":
    run()
