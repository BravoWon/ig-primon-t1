#!/usr/bin/env python
"""Powered STRUCTURE -> production test, Kansas (PUBLIC data) -- the original question.

Subsea structural elevation of a regional marker = KB elevation - marker TOP depth
(from KGS bulk tops). Detrend against lat/long (regional dip) -> local structural relief
(high vs low). Test whether structural position predicts age-independent productivity
(best-12-month oil) on single-well leases in the Central Kansas Uplift carbonate.

    python kgs_structure.py 9 51   (county codes; default Barton+Ellis)
"""
from __future__ import annotations
import csv, os, sys
import numpy as np
from scipy.stats import spearmanr

TOPS = r"C:\Users\JT-DEV1\kgs_public\tops\ks_tops.txt"
MASTER = r"C:\Users\JT-DEV1\kgs_public\county_{}_master.csv"
MARKERS = ["LANSING", "KANSAS CITY", "MISSISSIPPIAN", "ARBUCKLE", "HEEBNER"]


def api10(s):
    import re
    d = re.sub(r"\D", "", s or "")
    return d[:10] if len(d) >= 10 else ""


def main():
    counties = sys.argv[1:] or ["9", "51"]

    # target wells: single-well leases joined to production
    wells = {}     # api10 -> dict(best12, cum, county)
    for cc in counties:
        path = MASTER.format(cc)
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r.get("max_wells") == "1" and r.get("cum_oil") not in ("", None):
                a = api10(r["api"])
                if a:
                    try:
                        wells[a] = dict(best12=float(r["best12_oil"] or 0),
                                        cum=float(r["cum_oil"]), county=cc)
                    except ValueError:
                        pass
    targets = set(wells)
    print(f"target single-well producers (counties {counties}): {len(targets)}")

    # stream bulk tops, keep our wells' marker tops + elevation + lat/long
    well_tops = {a: {} for a in targets}   # api10 -> {MARKER: subsea}, plus _lat/_lon/_elev
    n = 0
    with open(TOPS, encoding="latin-1", newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            n += 1
            a = api10(row.get("API_NUM_NODASH") or row.get("API_NUMBER"))
            if a not in targets:
                continue
            try:
                elev = float(row["ELEVATION"]); top = float(row["TOP"])
            except (ValueError, KeyError):
                continue
            subsea = elev - top
            form = (row.get("FORMATION", "") + " " + row.get("OLD_FORMATION", "")).upper()
            d = well_tops[a]
            d.setdefault("_lat", row.get("LATITUDE")); d.setdefault("_lon", row.get("LONGITUDE"))
            d.setdefault("_elev", elev)
            for m in MARKERS:
                if m in form and m not in d:
                    d[m] = subsea
    print(f"  scanned {n:,} tops rows")

    def detrend_corr(marker, outcome):
        pts = []
        for a in targets:
            d = well_tops[a]
            if marker in d and d.get("_lat") and d.get("_lon"):
                try:
                    pts.append((float(d["_lon"]), float(d["_lat"]), d[marker],
                                wells[a][outcome]))
                except (ValueError, TypeError):
                    pass
        if len(pts) < 30:
            return None
        lon, lat, sub, y = map(np.array, zip(*pts))
        # regional plane fit subsea ~ a + b*lon + c*lat ; residual = local structural relief
        A = np.c_[np.ones_like(lon), lon, lat]
        coef, *_ = np.linalg.lstsq(A, sub, rcond=None)
        resid = sub - A @ coef
        rho_raw, p_raw = spearmanr(sub, y)
        rho_res, p_res = spearmanr(resid, y)
        return len(pts), rho_raw, p_raw, rho_res, p_res, resid, y

    print("\n[POWERED structure -> production]  (expect + : structural highs produce more)")
    for outcome in ("best12", "cum"):
        print(f"\n  outcome = {outcome} oil:")
        for m in MARKERS:
            res = detrend_corr(m, outcome)
            if not res:
                continue
            nn, rr, pr, rrs, prs, resid, y = res
            sig = "***" if prs < 0.001 else "**" if prs < 0.01 else "*" if prs < 0.05 else ""
            print(f"    {m:14} n={nn:3}  raw rho={rr:+.3f}  | detrended rho={rrs:+.3f} "
                  f"p={prs:.2e} {sig}")
            if outcome == "best12" and m == "KANSAS CITY" and nn >= 30:
                order = np.argsort(resid)
                lo = y[order[:nn//3]]; hi = y[order[-nn//3:]]
                print(f"        structural HIGH tercile median best12={np.median(hi):,.0f} | "
                      f"LOW tercile={np.median(lo):,.0f}  lift={np.median(hi)/max(1,np.median(lo)):.2f}x")


if __name__ == "__main__":
    main()
