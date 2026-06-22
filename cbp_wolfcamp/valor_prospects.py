#!/usr/bin/env python
"""Valor Energy prospect analysis (PUBLIC KGS data). Applies the value-geometry METHOD with the
RIGHT value-dimension per play, and locates each acquired footprint within its play:

  BUTLER ARBUCKLE (carbonate, structural)  -> structure (Arbuckle subsea) -> productivity, the edge
                                              our CKU work verified; + footprint maturity.
  OLD PAINT / ALGRIM (Morrow sandstone)    -> Morrow presence + interval thickness (sand-fairway
                                              proxy), NOT structure; + footprint development density.

Honest scope: KGS lease_prod is OIL-only; SW-Kansas Morrow is gas-prone, so Morrow oil metrics are a
floor, not the full value. Ohio Trenton (Harpers Station) has no KGS data -> handled in the memo only.

    python valor_prospects.py
"""
from __future__ import annotations
import csv, re
import numpy as np
from collections import Counter
from scipy.stats import spearmanr

PROD = r"C:\Users\JT-DEV1\kgs_public\lease_prod.csv"
TOPS = r"C:\Users\JT-DEV1\kgs_public\tops\ks_tops.txt"

# acquired footprints (approx lat/lon boxes from KS 6th-PM PLSS; county-filtered too)
FOOT = {
    "OldPaint+Algrim (Morrow)": dict(cty={"FINNEY", "GRAY"}, lat=(37.86, 38.12), lon=(-100.80, -100.42)),
    "Butler Arbuckle":          dict(cty={"BUTLER"},         lat=(37.52, 37.70), lon=(-96.98, -96.66)),
}


def a10(s):
    d = re.sub(r"\D", "", s or "")
    return d[:10] if len(d) >= 10 else ""


def detrend(v, lon, lat):
    A = np.c_[np.ones(len(v)), lon, lat]
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ coef


def load_leases(counties):
    wells = {}
    for r in csv.DictReader(open(PROD, encoding="utf-8")):
        if r["county"].upper() not in counties:
            continue
        a = next((a10(t) for t in r["apis"].split(",") if a10(t)), "")
        if not a:
            continue
        try:
            b12 = float(r["best12_oil"] or 0); cum = float(r["cum_oil"] or 0)
            la, lo = float(r["lat"]), float(r["lon"]); mw = int(r["max_wells"] or 1)
        except (ValueError, KeyError):
            continue
        wells[a] = dict(b12=b12, cum=cum, la=la, lo=lo, mw=mw, cty=r["county"].upper(),
                        zone=(r.get("zone") or "").strip())
    return wells


def attach_tops(wells, marks):
    T = set(wells)
    for r in csv.DictReader(open(TOPS, encoding="latin-1")):
        a = a10(r.get("API_NUM_NODASH") or r.get("API_NUMBER"))
        if a not in T:
            continue
        form = (r.get("FORMATION", "") + " " + r.get("OLD_FORMATION", "")).upper()
        try:
            top = float(r["TOP"])
        except (ValueError, KeyError):
            continue
        w = wells[a]
        try:
            w.setdefault("elev", float(r["ELEVATION"]))
        except (ValueError, KeyError):
            pass
        base = None
        try:
            base = float(r["BASE"])
        except (ValueError, KeyError, TypeError):
            pass
        for m in marks:
            if m in form:
                w.setdefault(m + "_top", top)
                if base and base > top:
                    w.setdefault(m + "_thk", base - top)


def in_box(w, box):
    return box["lat"][0] <= w["la"] <= box["lat"][1] and box["lon"][0] <= w["lo"] <= box["lon"][1]


def main():
    # ======================= BUTLER ARBUCKLE =======================
    print("=" * 72, "\nBUTLER ARBUCKLE  (carbonate / structural -- our verified play type)\n", "=" * 72)
    bw = load_leases({"BUTLER"})
    attach_tops(bw, ["ARBUCKLE", "MISSISSIPPIAN"])
    # structure edge on the Arbuckle carbonate set
    for mk in ["ARBUCKLE", "MISSISSIPPIAN"]:
        S = [w for w in bw.values() if (mk + "_top") in w and "elev" in w and w["b12"] > 0]
        if len(S) < 30:
            print(f"  {mk}: only {len(S)} tops+prod -- skip"); continue
        sub = np.array([w["elev"] - w[mk + "_top"] for w in S])        # subsea structure
        b12 = np.log(np.array([w["b12"] for w in S]))
        lo = np.array([w["lo"] for w in S]); la = np.array([w["la"] for w in S])
        rho, p = spearmanr(detrend(sub, lo, la), b12)
        print(f"  {mk:13} structure -> log best12:  n={len(S):4}  rho={rho:+.3f}  p={p:.1e}")
    # footprint maturity + structural position
    box = FOOT["Butler Arbuckle"]
    fp = [w for w in bw.values() if in_box(w, box)]
    cnt = sum(w["mw"] for w in fp)
    allb12 = np.array([w["b12"] for w in bw.values() if w["b12"] > 0])
    print(f"\n  FOOTPRINT (T28S R5-6E box): {len(fp)} leases / {cnt} wells; "
          f"median best12={np.median([w['b12'] for w in fp if w['b12']>0]) if any(w['b12']>0 for w in fp) else 0:.0f} bbl")
    arb_fp = [w for w in fp if "ARBUCKLE_top" in w and "elev" in w]
    if arb_fp:
        allsub = np.array([w["elev"] - w["ARBUCKLE_top"] for w in bw.values()
                           if "ARBUCKLE_top" in w and "elev" in w])
        fpsub = np.array([w["elev"] - w["ARBUCKLE_top"] for w in arb_fp])
        pct = 100 * (allsub < np.median(fpsub)).mean()
        print(f"  footprint Arbuckle structure: median subsea={np.median(fpsub):.0f} "
              f"= {pct:.0f}th pct of county (higher pct = structurally higher = better)")

    # ======================= MORROW (Old Paint / Algrim) =======================
    print("\n" + "=" * 72, "\nOLD PAINT / ALGRIM  (Morrow SANDSTONE -- value = sand fairway, NOT structure)\n", "=" * 72)
    mw = load_leases({"FINNEY", "GRAY"})
    attach_tops(mw, ["MORROW", "MISSISSIPPIAN"])
    box = FOOT["OldPaint+Algrim (Morrow)"]
    have_mor = [w for w in mw.values() if "MORROW_top" in w]
    have_thk = [w for w in have_mor if "MORROW_thk" in w]
    print(f"  Morrow tops in county set: {len(have_mor)} wells; with interval thickness (BASE): {len(have_thk)}")
    if have_thk:
        thk = np.array([w["MORROW_thk"] for w in have_thk])
        print(f"  Morrow interval thickness ft (sand-fairway proxy): "
              f"p10={np.percentile(thk,10):.0f} p50={np.median(thk):.0f} p90={np.percentile(thk,90):.0f}")
        prod = [w for w in have_thk if w["b12"] > 0]
        if len(prod) >= 20:
            th = np.array([w["MORROW_thk"] for w in prod]); bb = np.log(np.array([w["b12"] for w in prod]))
            rho, p = spearmanr(th, bb)
            print(f"  Morrow thickness -> log best12 (OIL leases only, n={len(prod)}): rho={rho:+.3f} p={p:.1e}")
    # productivity by county within footprint vs outside
    fp = [w for w in mw.values() if in_box(w, box)]
    fp_oil = [w for w in fp if w["b12"] > 0]
    print(f"\n  FOOTPRINT (T23-24S R29-30W box): {len(fp)} oil leases / {sum(w['mw'] for w in fp)} wells")
    if fp_oil:
        print(f"  footprint oil productivity: median best12={np.median([w['b12'] for w in fp_oil]):.0f} "
              f"median cum={np.median([w['cum'] for w in fp_oil]):.0f} bbl")
    for c in ["FINNEY", "GRAY"]:
        cw = [w for w in mw.values() if w["cty"] == c]
        dens = sum(w["mw"] for w in cw)
        print(f"  {c}: {len(cw)} oil leases / {dens} wells  (development density -> "
              f"{'OPEN' if dens < 200 else 'developed'})")
    print("\n  NOTE: oil-only record. SW-Kansas Morrow is gas-prone -> these oil numbers are a FLOOR.")


if __name__ == "__main__":
    main()
