#!/usr/bin/env python
"""Does TRAPPING STYLE control structure-pays? (PUBLIC data) -- the second dial.

Physical claim: structure pays only where there is structural RELIEF (closures = structural
trap); on a flat shelf (stratigraphic / no trap) structural position can't matter. Measures
local structural relief = spread of detrended KC-marker residual among nearby wells, then
tests whether structure->best12 strengthens with relief. Producing-zone diversity is a
second trap-style proxy (one dominant zone = structural; many zones = stratigraphic/patchy).

    python kgs_trapstyle.py
"""
from __future__ import annotations
import csv, re
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import spearmanr
from scipy.spatial import cKDTree

PROD = r"C:\Users\JT-DEV1\kgs_public\lease_prod.csv"
TOPS = r"C:\Users\JT-DEV1\kgs_public\tops\ks_tops.txt"
CKU = ["BARTON","ELLIS","RUSSELL","NESS","ROOKS","STAFFORD","BARBER","RICE","TREGO",
       "COMANCHE","KIOWA","PAWNEE","RUSH","ELLSWORTH","OSBORNE","LINCOLN"]
MARK = "KANSAS CITY"


def api10(s):
    d = re.sub(r"\D", "", s or "")
    return d[:10] if len(d) >= 10 else ""


def main():
    wells = {}
    for r in csv.DictReader(open(PROD, encoding="utf-8")):
        if r["max_wells"] != "1" or r["county"].upper() not in set(CKU):
            continue
        a = next((api10(t) for t in r["apis"].split(",") if api10(t)), "")
        if not a:
            continue
        try:
            wells[a] = dict(best12=float(r["best12_oil"] or 0), county=r["county"].upper(),
                            zone=r["zone"].strip())
        except ValueError:
            pass
    targets = set(wells)
    for r in csv.DictReader(open(TOPS, encoding="latin-1")):
        a = api10(r.get("API_NUM_NODASH") or r.get("API_NUMBER"))
        if a not in targets:
            continue
        form = (r.get("FORMATION", "") + " " + r.get("OLD_FORMATION", "")).upper()
        if MARK in form and "kc_sub" not in wells[a]:
            try:
                wells[a]["kc_sub"] = float(r["ELEVATION"]) - float(r["TOP"])
                wells[a]["lat"] = float(r["LATITUDE"]); wells[a]["lon"] = float(r["LONGITUDE"])
            except (ValueError, KeyError):
                pass
    W = [w for w in wells.values() if "kc_sub" in w and "lat" in w]
    print(f"CKU wells with KC structure + location: {len(W)}")

    # county-detrend -> structural residual (local relief)
    byc = defaultdict(list)
    for w in W:
        byc[w["county"]].append(w)
    for c, ws in byc.items():
        lon = np.array([w["lon"] for w in ws]); lat = np.array([w["lat"] for w in ws])
        sub = np.array([w["kc_sub"] for w in ws])
        A = np.c_[np.ones_like(lon), lon, lat]
        coef, *_ = np.linalg.lstsq(A, sub, rcond=None)
        for w, rr in zip(ws, sub - A @ coef):
            w["resid"] = rr

    # local structural relief: std of residual among neighbors (~5 mi)
    lat = np.array([w["lat"] for w in W]); lon = np.array([w["lon"] for w in W])
    coslat = np.cos(np.radians(np.mean(lat)))
    XY = np.c_[lat, lon * coslat]                      # ~equal-area degrees
    tree = cKDTree(XY)
    R = 0.07                                           # ~5 miles
    resid = np.array([w["resid"] for w in W])
    b12 = np.array([w["best12"] for w in W])
    relief = np.full(len(W), np.nan)
    for i in range(len(W)):
        idx = tree.query_ball_point(XY[i], R)
        if len(idx) >= 6:
            relief[i] = np.std(resid[idx])
    ok = np.isfinite(relief)
    print(f"  wells with >=6 neighbors (local relief computed): {ok.sum()}")

    print("\n[WELL-LEVEL] structure->best12 by LOCAL structural relief tercile")
    rr = relief[ok]; rs = resid[ok]; yy = b12[ok]
    q = np.quantile(rr, [1/3, 2/3])
    for name, m in [("LOW relief (flat shelf)", rr <= q[0]),
                    ("MID relief", (rr > q[0]) & (rr <= q[1])),
                    ("HIGH relief (closures)", rr > q[1])]:
        rho, p = spearmanr(rs[m], yy[m])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:24} n={m.sum():4}  relief~{np.median(rr[m]):5.0f}ft  "
              f"structure->best12 rho={rho:+.3f}  p={p:.2e} {sig}")

    # county-level: structure-pays vs relief amplitude and zone concentration
    print("\n[COUNTY-LEVEL] structure-pays rho vs trap-style proxies")
    rows = []
    for c, ws in byc.items():
        if len(ws) < 40:
            continue
        rsd = np.array([w["resid"] for w in ws]); y = np.array([w["best12"] for w in ws])
        srho, _ = spearmanr(rsd, y)
        relief_amp = float(np.std(rsd))                          # closure amplitude
        zc = Counter(w["zone"] for w in ws if w["zone"])
        tot = sum(zc.values())
        herf = sum((v/tot)**2 for v in zc.values()) if tot else np.nan   # 1=one zone, ~0=many
        rows.append((c, srho, relief_amp, herf, len(ws)))
    rows.sort(key=lambda x: -x[1])
    print(f"  {'county':10} {'struct_rho':>10} {'relief_ft':>9} {'zone_herf':>9} {'n':>4}")
    for c, sr, ra, hf, n in rows:
        print(f"  {c:10} {sr:+10.3f} {ra:9.0f} {hf:9.2f} {n:4}")
    sr = np.array([r[1] for r in rows]); ra = np.array([r[2] for r in rows])
    hf = np.array([r[3] for r in rows])
    c1, p1 = spearmanr(ra, sr); c2, p2 = spearmanr(hf, sr)
    print(f"\n  Spearman(structural relief, structure_rho) = {c1:+.3f}  p={p1:.3f}  (expect + : relief->structure pays)")
    print(f"  Spearman(zone concentration, structure_rho) = {c2:+.3f}  p={p2:.3f}  (expect + : one-zone->structural)")


if __name__ == "__main__":
    main()
