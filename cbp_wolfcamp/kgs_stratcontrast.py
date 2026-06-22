#!/usr/bin/env python
"""TRAP-STYLE CONTRAST: structural carbonate vs stratigraphic sandstone (PUBLIC data).

The clean version of the trap-style dial -- contrast trap STYLE, not degree. Runs the
identical structure->best12 test on:
  CARB = structural carbonate plays (Lansing / Kansas City / Arbuckle)   -> expect structure PAYS
  SAND = stratigraphic channel sandstone (Cherokee Group, SE Kansas)     -> expect structure NULL
If carbonate pays on structure and the sandstone channel play does not, trap style is the dial.

    python kgs_stratcontrast.py
"""
from __future__ import annotations
import csv, re
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

PROD = r"C:\Users\JT-DEV1\kgs_public\lease_prod.csv"
TOPS = r"C:\Users\JT-DEV1\kgs_public\tops\ks_tops.txt"
MARKERS = ["KANSAS CITY", "LANSING", "ARBUCKLE", "MISSISSIPPIAN", "CHEROKEE", "MARMATON"]


def api10(s):
    d = re.sub(r"\D", "", s or "")
    return d[:10] if len(d) >= 10 else ""


def main():
    grp = {}   # api -> (group, best12, county)
    for r in csv.DictReader(open(PROD, encoding="utf-8")):
        if r["max_wells"] != "1":
            continue
        z = r["zone"].strip().upper()
        if any(k in z for k in ("KANSAS CITY", "LANSING", "ARBUCKLE")) and "CHEROKEE" not in z:
            g = "CARB"
        elif "CHEROKEE" in z:
            g = "SAND"
        else:
            continue
        a = next((api10(t) for t in r["apis"].split(",") if api10(t)), "")
        if not a:
            continue
        try:
            grp[a] = dict(group=g, best12=float(r["best12_oil"] or 0), county=r["county"].upper())
        except ValueError:
            pass
    targets = set(grp)
    nC = sum(v["group"] == "CARB" for v in grp.values())
    nS = sum(v["group"] == "SAND" for v in grp.values())
    print(f"single-well oil leases: CARB(carbonate)={nC}  SAND(Cherokee channel)={nS}")

    tops = {a: {} for a in targets}
    for r in csv.DictReader(open(TOPS, encoding="latin-1")):
        a = api10(r.get("API_NUM_NODASH") or r.get("API_NUMBER"))
        if a not in targets:
            continue
        form = (r.get("FORMATION", "") + " " + r.get("OLD_FORMATION", "")).upper()
        d = tops[a]
        d.setdefault("_lat", r.get("LATITUDE")); d.setdefault("_lon", r.get("LONGITUDE"))
        for m in MARKERS:
            if m in form and m not in d:
                try:
                    d[m] = float(r["ELEVATION"]) - float(r["TOP"])
                except (ValueError, KeyError):
                    pass

    def test(group, marker):
        pts = defaultdict(list)
        for a, g in grp.items():
            if g["group"] != group:
                continue
            d = tops[a]
            if marker in d and d.get("_lat") and d.get("_lon"):
                try:
                    pts[g["county"]].append((float(d["_lon"]), float(d["_lat"]), d[marker], g["best12"]))
                except (ValueError, TypeError):
                    pass
        # county-detrend, pool residuals
        R, Y = [], []
        for c, lst in pts.items():
            if len(lst) < 15:
                continue
            lon, lat, sub, y = map(np.array, zip(*lst))
            A = np.c_[np.ones_like(lon), lon, lat]
            coef, *_ = np.linalg.lstsq(A, sub, rcond=None)
            R.extend(sub - A @ coef); Y.extend(y)
        if len(R) < 30:
            return None
        R, Y = np.array(R), np.array(Y)
        rho, p = spearmanr(R, Y)
        order = np.argsort(R); t = len(R)//3
        lift = np.median(Y[order[-t:]]) / max(1, np.median(Y[order[:t]]))
        return len(R), rho, p, lift

    print(f"\n  marker coverage (wells with that top):")
    for m in MARKERS:
        cC = sum(1 for a, g in grp.items() if g["group"] == "CARB" and m in tops[a])
        cS = sum(1 for a, g in grp.items() if g["group"] == "SAND" and m in tops[a])
        print(f"    {m:14} CARB={cC:4}  SAND={cS:4}")

    print("\n[STRUCTURE -> best-12 oil]  (detrended, county-pooled)")
    print(f"  {'group':6} {'marker':14} {'n':>4} {'rho':>7} {'p':>10} {'high/low lift':>13}")
    for group, markers in [("CARB", ["KANSAS CITY", "LANSING", "ARBUCKLE"]),
                           ("SAND", ["CHEROKEE", "MISSISSIPPIAN", "MARMATON"])]:
        for m in markers:
            res = test(group, m)
            if res:
                n, rho, p, lift = res
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {group:6} {m:14} {n:4} {rho:+.3f} {p:10.2e} {lift:11.2f}x  {sig}")
    print("\n  prediction: CARB structure PAYS (rho>0, lift>1.2); SAND structure NULL (rho~0, lift~1.0)")


if __name__ == "__main__":
    main()
