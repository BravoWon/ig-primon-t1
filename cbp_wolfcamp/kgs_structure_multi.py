#!/usr/bin/env python
"""Multi-county replication of STRUCTURE -> production (PUBLIC data, no scraping).

Uses only the two statewide bulk files already local: reduced lease production
(lease_prod.csv) and bulk tops (ks_tops.txt). For every Central-Kansas-Uplift county,
take single-well oil leases, attach structural elevation (KB elev - marker top, detrended
vs lat/long), and test structural relief -> best-12-month oil. Reports per-county
replication + a pooled (county-detrended) estimate.

    python kgs_structure_multi.py
"""
from __future__ import annotations
import csv, re
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

PROD = r"C:\Users\JT-DEV1\kgs_public\lease_prod.csv"
TOPS = r"C:\Users\JT-DEV1\kgs_public\tops\ks_tops.txt"
CKU = ["BARTON","ELLIS","RUSSELL","NESS","ROOKS","STAFFORD","BARBER","RICE","TREGO",
       "COMANCHE","KIOWA","PAWNEE","RUSH","ELLSWORTH","OSBORNE","LINCOLN"]
MARKERS = ["KANSAS CITY", "LANSING", "ARBUCKLE"]


def api10(s):
    d = re.sub(r"\D", "", s or "")
    return d[:10] if len(d) >= 10 else ""


def main():
    # 1) single-well oil leases in CKU counties -> api10 -> (best12, cum, county)
    wells = {}
    cset = set(CKU)
    for r in csv.DictReader(open(PROD, encoding="utf-8")):
        if r["max_wells"] != "1" or r["county"].upper() not in cset:
            continue
        a = ""
        for tok in r["apis"].split(","):
            a = api10(tok)
            if a:
                break
        if not a:
            continue
        try:
            wells[a] = dict(best12=float(r["best12_oil"] or 0), cum=float(r["cum_oil"]),
                            county=r["county"].upper())
        except ValueError:
            pass
    targets = set(wells)
    print(f"single-well oil leases across {len(CKU)} CKU counties: {len(targets)}")

    # 2) stream bulk tops -> per-well marker subsea + lat/lon
    wt = {a: {} for a in targets}
    n = 0
    with open(TOPS, encoding="latin-1", newline="") as fh:
        for row in csv.DictReader(fh):
            n += 1
            a = api10(row.get("API_NUM_NODASH") or row.get("API_NUMBER"))
            if a not in targets:
                continue
            try:
                subsea = float(row["ELEVATION"]) - float(row["TOP"])
            except (ValueError, KeyError):
                continue
            form = (row.get("FORMATION", "") + " " + row.get("OLD_FORMATION", "")).upper()
            d = wt[a]
            d.setdefault("_lat", row.get("LATITUDE")); d.setdefault("_lon", row.get("LONGITUDE"))
            for m in MARKERS:
                if m in form and m not in d:
                    d[m] = subsea
    print(f"  scanned {n:,} tops rows\n")

    def points(marker, county=None):
        out = []
        for a in targets:
            if county and wells[a]["county"] != county:
                continue
            d = wt[a]
            if marker in d and d.get("_lat") and d.get("_lon"):
                try:
                    out.append((float(d["_lon"]), float(d["_lat"]), d[marker], wells[a]["best12"]))
                except (ValueError, TypeError):
                    pass
        return out

    def detrend_resid(pts):
        lon, lat, sub, y = map(np.array, zip(*pts))
        A = np.c_[np.ones_like(lon), lon, lat]
        coef, *_ = np.linalg.lstsq(A, sub, rcond=None)
        return sub - A @ coef, y

    # 3) per-county replication (use the best-covered marker = KANSAS CITY, fallback LANSING)
    print("[PER-COUNTY replication: detrended structural relief -> best-12 oil]")
    print(f"  {'county':10} {'marker':12} {'n':>4} {'rho':>7} {'p':>9}  sig")
    pooled = defaultdict(list)   # marker -> list of (resid, y) pooled across counties
    summary = []
    for c in CKU:
        best = None
        for m in MARKERS:
            pts = points(m, c)
            if len(pts) >= 40:
                best = (m, pts); break
        if not best:
            continue
        m, pts = best
        resid, y = detrend_resid(pts)
        rho, p = spearmanr(resid, y)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {c:10} {m:12} {len(pts):4} {rho:+.3f} {p:9.1e}  {sig}")
        summary.append((c, rho, p, len(pts)))
        # pool: county-detrended residuals for the common marker (KANSAS CITY)
        for mm in MARKERS:
            ptm = points(mm, c)
            if len(ptm) >= 40:
                rsd, yy = detrend_resid(ptm)
                pooled[mm].extend(zip(rsd, yy))

    # 4) pooled estimate (county-fixed-effect detrended)
    print("\n[POOLED across CKU counties, county-detrended]")
    for m in MARKERS:
        if len(pooled[m]) >= 100:
            rs = np.array([x[0] for x in pooled[m]]); ys = np.array([x[1] for x in pooled[m]])
            rho, p = spearmanr(rs, ys)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            order = np.argsort(rs); t = len(rs) // 3
            lo = np.median(ys[order[:t]]); hi = np.median(ys[order[-t:]])
            print(f"  {m:12} n={len(pooled[m]):4}  rho={rho:+.3f}  p={p:.2e} {sig:3}  "
                  f"high/low tercile best12: {hi:,.0f}/{lo:,.0f} = {hi/max(1,lo):.2f}x")

    pos = sum(1 for _, r, p, _ in summary if r > 0)
    sigpos = sum(1 for _, r, p, _ in summary if r > 0 and p < 0.05)
    print(f"\n  counties tested: {len(summary)}  | positive rho: {pos}  | "
          f"positive & p<0.05: {sigpos}")


if __name__ == "__main__":
    main()
