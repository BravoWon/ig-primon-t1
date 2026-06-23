#!/usr/bin/env python
"""Put the Lansing-KC ISOPACH lead through the wringer (PUBLIC data).

The provisional lead: thinner Lansing-Kansas City interval -> more oil (rho~-0.17). Having
just caught a single-marker artifact, this gets adversarial scrutiny before belief:
  1 CLEAN     drop degenerate (<=0 / absurd) thickness picks -> does it survive?
  2 SPECIFIC  test MANY interval thicknesses -> is only Lansing-KC predictive, or does any
              thickness/depth correlate (=> generic depth/structure confound, not reservoir)?
  3 INDEPENDENT  thickness vs structure (Lansing subsea); confirm structure null on same set
  4 REPLICATE per-CKU-county thinner-pays
  5 SPATIAL   is thickness spatially smooth (real isopach) or pick-noise?

    python kgs_isopach_wringer.py
"""
from __future__ import annotations
import csv, re
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.spatial import cKDTree

PROD = r"C:\Users\JT-DEV1\kgs_public\lease_prod.csv"
TOPS = r"C:\Users\JT-DEV1\kgs_public\tops\ks_tops.txt"
CKU = {"BARTON","ELLIS","RUSSELL","NESS","ROOKS","STAFFORD","BARBER","RICE","TREGO",
       "COMANCHE","KIOWA","PAWNEE","RUSH","ELLSWORTH","OSBORNE","LINCOLN"}
# stratigraphic order shallow->deep
MARK = ["LANSING", "KANSAS CITY", "MARMATON", "CHEROKEE", "MISSISSIPPIAN", "ARBUCKLE"]


def a10(s):
    d = re.sub(r"\D", "", s or "")
    return d[:10] if len(d) >= 10 else ""


def detrend(v, lon, lat, cty):
    out = np.zeros_like(v, float)
    for c in set(cty):
        m = np.array([x == c for x in cty])
        if m.sum() < 15:
            out[m] = v[m] - np.mean(v[m]); continue
        A = np.c_[np.ones(m.sum()), lon[m], lat[m]]
        coef, *_ = np.linalg.lstsq(A, v[m], rcond=None)
        out[m] = v[m] - A @ coef
    return out


def main():
    wells = {}
    for r in csv.DictReader(open(PROD, encoding="utf-8")):
        if r["max_wells"] != "1" or r["county"].upper() not in CKU:
            continue
        a = next((a10(t) for t in r["apis"].split(",") if a10(t)), "")
        if a:
            try:
                wells[a] = dict(b12=float(r["best12_oil"] or 0), cum=float(r["cum_oil"]),
                                cty=r["county"].upper())
            except ValueError:
                pass
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
        w = wells[a]; w.setdefault("lat", float(r["LATITUDE"])); w.setdefault("lon", float(r["LONGITUDE"]))
        try:
            w.setdefault("elev", float(r["ELEVATION"]))
        except (ValueError, KeyError):
            pass
        for m in MARK:
            if m in form:
                w.setdefault(m, top)

    def arr(ws, k):
        return np.array([w[k] for w in ws])

    def corr(ws, x, y, lbl, sub=""):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 30:
            print(f"  {lbl:42} n={m.sum():4}  (too few)"); return None
        rho, p = spearmanr(x[m], y[m])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {lbl:42} n={m.sum():4}  rho={rho:+.3f}  p={p:.1e} {sig}  {sub}")
        return rho

    # base set: both Lansing & KC
    W = [w for w in wells.values() if "LANSING" in w and "KANSAS CITY" in w and "lat" in w]
    lon = arr(W, "lon"); lat = arr(W, "lat"); cty = [w["cty"] for w in W]
    b12 = arr(W, "b12")
    thick = arr(W, "KANSAS CITY") - arr(W, "LANSING")
    print(f"Lansing-KC isopach wringer: n={len(W)} CKU wells with both tops")
    print(f"  thickness ft: p1={np.percentile(thick,1):.0f} p10={np.percentile(thick,10):.0f} "
          f"p50={np.median(thick):.0f} p90={np.percentile(thick,90):.0f} p99={np.percentile(thick,99):.0f}")
    degen = (thick <= 5) | (thick > 700)
    print(f"  degenerate picks (<=5 or >700 ft): {degen.sum()} ({100*degen.mean():.0f}%)")

    print("\n[1] CLEAN -- does thinner-pays survive dropping degenerate picks?")
    corr(W, detrend(thick, lon, lat, cty), b12, "RAW detrended thickness -> best12")
    keep = ~degen
    corr([W[i] for i in np.where(keep)[0]],
         detrend(thick[keep], lon[keep], lat[keep], [cty[i] for i in np.where(keep)[0]]),
         b12[keep], "CLEANED detrended thickness -> best12")

    print("\n[2] SPECIFIC -- test EVERY available interval; is only Lansing-KC predictive?")
    print("    (if deeper/thicker intervals all correlate -> generic depth confound, NOT reservoir)")
    pairs = [("LANSING","KANSAS CITY"),("KANSAS CITY","MARMATON"),("KANSAS CITY","MISSISSIPPIAN"),
             ("LANSING","MISSISSIPPIAN"),("MISSISSIPPIAN","ARBUCKLE"),("KANSAS CITY","ARBUCKLE")]
    for a, b in pairs:
        ws = [w for w in wells.values() if a in w and b in w and "lat" in w]
        if len(ws) < 30:
            continue
        th = arr(ws, b) - arr(ws, a)
        good = (th > 5) & (th < 3000)
        ws2 = [ws[i] for i in np.where(good)[0]]
        if len(ws2) < 30:
            continue
        lo = arr(ws2, "lon"); la = arr(ws2, "lat"); ct = [w["cty"] for w in ws2]
        corr(ws2, detrend((arr(ws2, b) - arr(ws2, a)), lo, la, ct), arr(ws2, "b12"),
             f"{a[:4]}-{b[:4]} isopach -> best12")

    print("\n[3] INDEPENDENT of structure?")
    wS = [w for w in W if "elev" in w]
    if len(wS) > 30:
        loS = arr(wS, "lon"); laS = arr(wS, "lat"); ctS = [w["cty"] for w in wS]
        struc = arr(wS, "elev") - arr(wS, "LANSING")          # true Lansing structure
        thk = arr(wS, "KANSAS CITY") - arr(wS, "LANSING")
        dstr = detrend(struc, loS, laS, ctS); dthk = detrend(thk, loS, laS, ctS)
        print(f"  corr(detrended thickness, detrended structure) = {np.corrcoef(dthk, dstr)[0,1]:+.3f}  "
              f"(near 0 => independent)")
        corr(wS, dstr, arr(wS, "b12"), "true structure -> best12 (should be NULL)")
        corr(wS, dthk, arr(wS, "b12"), "thickness     -> best12")

    print("\n[4] REPLICATE per CKU county (cleaned, detrended thickness -> best12):")
    byc = defaultdict(list)
    for i, w in enumerate(W):
        if keep[i]:
            byc[w["cty"]].append(i)
    npos = nsig = nt = 0
    for c, idx in sorted(byc.items()):
        if len(idx) < 40:
            continue
        idx = np.array(idx)
        dt = detrend(thick[idx], lon[idx], lat[idx], [cty[i] for i in idx])
        rho, p = spearmanr(dt, b12[idx])
        nt += 1; npos += rho < 0; nsig += (rho < 0 and p < 0.05)
        sig = "*" if p < 0.05 else ""
        print(f"    {c:10} n={len(idx):4}  rho={rho:+.3f} {sig}")
    print(f"  counties: {nt}  thinner-pays(rho<0): {npos}  significant: {nsig}")

    print("\n[5] SPATIAL coherence (is thickness a smooth isopach or pick-noise?)")
    XY = np.c_[lat, lon * np.cos(np.radians(np.mean(lat)))]
    tree = cKDTree(XY); R = 0.07
    nbr = np.full(len(W), np.nan)
    for i in range(len(W)):
        j = [k for k in tree.query_ball_point(XY[i], R) if k != i]
        if len(j) >= 5:
            nbr[i] = np.median(thick[j])
    m = np.isfinite(nbr)
    sc = spearmanr(thick[m], nbr[m])[0]
    print(f"  corr(well thickness, neighbor-median thickness) = {sc:+.3f}  "
          f"(high => smooth real isopach; ~0 => noise)")


if __name__ == "__main__":
    main()
