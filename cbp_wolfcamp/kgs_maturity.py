#!/usr/bin/env python
"""Does STRUCTURE-PAYS decay with field MATURITY? (PUBLIC data) -- unifying mechanism test.

Hypothesis (from the Permian depletion-inversion): structural highs drain first, so the
structure->production signal should WEAKEN (or invert) as a field/well matures. Tests two ways:
  A) county-level: structure-pays rho vs county maturity (legacy-fraction / median depletion)
  B) well-level vintage cohorts: structure->best12 rho for legacy vs post-2000 wells, by era.

Maturity proxies available from 2000-2026 monthly production: first_ym (legacy = left-
truncated at 200001), n_months in window, cumulative oil (depletion).

    python kgs_maturity.py
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
            wells[a] = dict(best12=float(r["best12_oil"] or 0), cum=float(r["cum_oil"]),
                            county=r["county"].upper(), first_ym=int(r["first_ym"]),
                            n_months=int(r["n_months"]))
        except ValueError:
            pass
    targets = set(wells)

    # structure (KC subsea + lat/lon)
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
    W = [w for w in wells.values() if "kc_sub" in w]
    print(f"CKU single-well leases with KC structure: {len(W)}")

    # county-detrended structural residual
    byc = defaultdict(list)
    for w in W:
        byc[w["county"]].append(w)
    for c, ws in byc.items():
        lon = np.array([w["lon"] for w in ws]); lat = np.array([w["lat"] for w in ws])
        sub = np.array([w["kc_sub"] for w in ws])
        A = np.c_[np.ones_like(lon), lon, lat]
        coef, *_ = np.linalg.lstsq(A, sub, rcond=None)
        resid = sub - A @ coef
        for w, rr in zip(ws, resid):
            w["resid"] = rr

    # vintage flag: legacy (left-truncated at window start) vs datable post-2000
    yr = np.array([w["first_ym"] // 100 for w in W])
    legacy = np.array([w["first_ym"] <= 200012 for w in W])
    print(f"  legacy (pre-2000, truncated): {legacy.sum()}   post-2000 datable: {(~legacy).sum()}")
    print("  post-2000 first-production by era:")
    for lo, hi in [(2001,2005),(2006,2010),(2011,2015),(2016,2026)]:
        m = (~legacy) & (yr >= lo) & (yr <= hi)
        print(f"    {lo}-{hi}: {m.sum()}")

    resid = np.array([w["resid"] for w in W]); b12 = np.array([w["best12"] for w in W])
    cum = np.array([w["cum"] for w in W]); nmo = np.array([w["n_months"] for w in W])

    def rho(mask, lbl):
        if mask.sum() < 30:
            print(f"    {lbl:26} n={mask.sum():4}  (too few)"); return
        r, p = spearmanr(resid[mask], b12[mask])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {lbl:26} n={mask.sum():4}  structure->best12 rho={r:+.3f}  p={p:.2e} {sig}")

    print("\n[B] WELL-LEVEL vintage cohorts (does structure-pays weaken in mature/legacy wells?)")
    rho(legacy, "legacy (pre-2000)")
    rho(~legacy, "post-2000 (datable)")
    for lo, hi in [(2001,2008),(2009,2015),(2016,2026)]:
        rho((~legacy) & (yr >= lo) & (yr <= hi), f"new {lo}-{hi}")

    # depletion proxy within legacy: cum tercile (more cum = more drained)
    print("\n[B2] within LEGACY wells, by cumulative depletion (2000-26 cum oil):")
    leg = np.where(legacy)[0]
    cl = cum[leg]; order = leg[np.argsort(cl)]
    t = len(order)//3
    for name, idx in [("lowest-cum third", order[:t]), ("highest-cum third", order[-t:])]:
        m = np.zeros(len(W), bool); m[idx] = True
        rho(m, name)

    print("\n[A] COUNTY-LEVEL: structure-pays strength vs county maturity")
    rows = []
    for c, ws in byc.items():
        if len(ws) < 40:
            continue
        rs = np.array([w["resid"] for w in ws]); yy = np.array([w["best12"] for w in ws])
        sr, _ = spearmanr(rs, yy)
        legfrac = np.mean([w["first_ym"] <= 200012 for w in ws])
        medcum = np.median([w["cum"] for w in ws])
        rows.append((c, sr, legfrac, medcum, len(ws)))
    rows.sort(key=lambda x: -x[1])
    print(f"  {'county':10} {'struct_rho':>10} {'legacy_frac':>11} {'med_cum':>9} {'n':>4}")
    for c, sr, lf, mc, n in rows:
        print(f"  {c:10} {sr:+10.3f} {lf:11.2f} {mc:9,.0f} {n:4}")
    srs = np.array([r[1] for r in rows]); lfs = np.array([r[2] for r in rows])
    mcs = np.array([r[3] for r in rows])
    r1, p1 = spearmanr(lfs, srs); r2, p2 = spearmanr(mcs, srs)
    print(f"\n  Spearman(legacy_fraction, structure_rho) = {r1:+.3f}  p={p1:.3f}   (expect - : mature->weaker)")
    print(f"  Spearman(median_cum,      structure_rho) = {r2:+.3f}  p={p2:.3f}   (expect - : depleted->weaker)")


if __name__ == "__main__":
    main()
