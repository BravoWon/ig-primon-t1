#!/usr/bin/env python
"""Recon: what does the KGS public record actually hold for the Valor prospect counties
(Finney/Gray = Old Paint+Algrim Morrow; Butler = Arbuckle)? Confirms data coverage, the producing
zones, oil productivity, and whether MORROW/ARBUCKLE tops exist before any analysis is built.

    python recon_valor.py
"""
from __future__ import annotations
import csv, re
import numpy as np
from collections import Counter, defaultdict

PROD = r"C:\Users\JT-DEV1\kgs_public\lease_prod.csv"
TOPS = r"C:\Users\JT-DEV1\kgs_public\tops\ks_tops.txt"
TARGET = {"FINNEY", "GRAY", "BUTLER"}


def a10(s):
    d = re.sub(r"\D", "", s or "")
    return d[:10] if len(d) >= 10 else ""


def plss_box(T, hemiR, Rrange):
    """Very rough Kansas (6th PM) township-center -> lat/lon, for bounding-box targeting only."""
    lat = 40.0 - (T - 0.5) * 6.0 / 69.0
    pm = 97.37
    dlon = (np.mean(Rrange) - 0.5) * 6.0 / (69.0 * np.cos(np.radians(lat)))
    lon = -(pm + dlon) if hemiR == "W" else -(pm - dlon)
    return lat, lon


def main():
    rows = defaultdict(list)
    zones = defaultdict(Counter)
    for r in csv.DictReader(open(PROD, encoding="utf-8")):
        c = r["county"].upper()
        if c not in TARGET:
            continue
        try:
            la, lo = float(r["lat"]), float(r["lon"])
            b12 = float(r["best12_oil"] or 0); cum = float(r["cum_oil"] or 0)
            mw = int(r["max_wells"] or 1)
        except (ValueError, KeyError):
            continue
        rows[c].append(dict(la=la, lo=lo, b12=b12, cum=cum, mw=mw, zone=(r.get("zone") or "").strip()))
        zones[c][(r.get("zone") or "(blank)").strip() or "(blank)"] += 1

    print("=== KGS lease_prod coverage (OIL only) for Valor prospect counties ===")
    for c in sorted(TARGET):
        rs = rows[c]
        if not rs:
            print(f"  {c}: NO leases found"); continue
        b12 = np.array([x["b12"] for x in rs]); cum = np.array([x["cum"] for x in rs])
        la = np.array([x["la"] for x in rs]); lo = np.array([x["lo"] for x in rs])
        nz = b12 > 0
        print(f"\n  {c}: {len(rs)} leases, {sum(x['mw'] for x in rs)} wells; "
              f"lat[{la.min():.2f},{la.max():.2f}] lon[{lo.min():.2f},{lo.max():.2f}]")
        print(f"     best12_oil>0: {nz.sum()} leases; median best12={np.median(b12[nz]) if nz.any() else 0:.0f} "
              f"cum_oil median={np.median(cum[cum>0]) if (cum>0).any() else 0:.0f} bbl")
        print(f"     top zones: " + ", ".join(f"{z}={n}" for z, n in zones[c].most_common(6)))

    # formation tops presence
    want = ["MORROW", "ARBUCKLE", "MISSISSIPPIAN", "LANSING", "TRENTON"]
    topcnt = defaultdict(Counter); ll = defaultdict(list)
    seen = defaultdict(set)
    for r in csv.DictReader(open(TOPS, encoding="latin-1")):
        # tops file has no county; filter by lat/lon box of the three counties (approx)
        try:
            la, lo = float(r["LATITUDE"]), float(r["LONGITUDE"])
        except (ValueError, KeyError):
            continue
        # Finney/Gray ~ 37.6-38.3N, 100.0-101.2W ; Butler ~ 37.5-38.0N, 96.6-97.2W
        region = None
        if 37.5 <= la <= 38.4 and -101.3 <= lo <= -100.0:
            region = "FINNEY/GRAY(box)"
        elif 37.4 <= la <= 38.05 and -97.25 <= lo <= -96.55:
            region = "BUTLER(box)"
        if not region:
            continue
        form = (r.get("FORMATION", "") + " " + r.get("OLD_FORMATION", "")).upper()
        ap = a10(r.get("API_NUM_NODASH") or r.get("API_NUMBER"))
        for w in want:
            if w in form:
                topcnt[region][w] += 1
                if ap:
                    seen[(region, w)].add(ap)

    print("\n=== formation TOPS present in prospect boxes (distinct wells) ===")
    for region in ["FINNEY/GRAY(box)", "BUTLER(box)"]:
        s = ", ".join(f"{w}={len(seen[(region,w)])}" for w in want if seen[(region, w)])
        print(f"  {region}: {s or 'none of the target formations found'}")

    print("\n=== rough PLSS boxes (sanity) ===")
    for name, T, hemi, Rr in [("Old Paint T23-24S R29-30W", 23.5, "W", [29, 30]),
                              ("Algrim T23-24S R29-30W", 23.5, "W", [29, 30]),
                              ("Butler Arbuckle T28S R5-6E", 28, "E", [5, 6])]:
        la, lo = plss_box(T, hemi, Rr)
        print(f"  {name:30} ~ ({la:.2f}, {lo:.2f})")


if __name__ == "__main__":
    main()
