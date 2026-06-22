#!/usr/bin/env python
"""Powered petrophysics -> production test, Kansas (PUBLIC data).

For each single-well-lease well with a log, compute classic reservoir-quality features
from the LAS (GR clean, porosity, resistivity -> net pay and porosity-feet phi*h) and
correlate against cumulative oil. phi*h is the textbook predictor of carbonate well
productivity -- this asks, at powered n, whether log rock quality actually predicts oil.

    python kgs_analyze.py [county_code]
"""
from __future__ import annotations
import csv, glob, os, sys, warnings, logging
import numpy as np
warnings.filterwarnings("ignore"); logging.getLogger("lasio").setLevel(logging.CRITICAL)
import lasio
from scipy.stats import spearmanr

CC = sys.argv[1] if len(sys.argv) > 1 else "9"
MASTER = rf"C:\Users\JT-DEV1\kgs_public\county_{CC}_master.csv"
LASDIR = rf"C:\Users\JT-DEV1\kgs_public\las\{CC}"


def curve(las, *names):
    for n in names:
        try:
            a = np.asarray(las[n], dtype=float); a[a <= -999] = np.nan
            if np.isfinite(a).sum() > 30:
                return a
        except Exception:
            pass
    return None


def petro(path):
    try:
        las = lasio.read(path, ignore_header_errors=True)
    except Exception:
        return None
    z = np.asarray(las.index, dtype=float)
    if z.size < 50:
        return None
    step = abs(np.nanmedian(np.diff(z)))
    if not (0 < step < 5):
        step = 0.5
    gr = curve(las, "GR", "GRGC", "SGR", "CGR")
    phi = curve(las, "DPOR", "CNPOR", "SPOR", "NPHI", "PHIN")
    rt = curve(las, "RILD", "RLL3", "RT", "RILM", "RXORT")
    if gr is None or phi is None:
        return dict(has=0)
    # porosity units: % -> fraction
    pf = phi.copy()
    if np.nanmedian(pf[np.isfinite(pf)]) > 1.5:
        pf = pf / 100.0
    pf = np.clip(pf, 0, 0.5)
    clean = np.isfinite(gr) & (gr < 60)
    res = clean & np.isfinite(pf) & (pf > 0.06)
    if rt is not None:
        res = res & np.isfinite(rt) & (rt > 8)        # hydrocarbon (high Rt)
    net_pay_ft = float(np.sum(res) * step)
    phi_h = float(np.nansum(np.where(res, pf, 0.0)) * step)     # porosity-feet
    clean_ft = float(np.sum(clean) * step)
    mean_phi = float(np.nanmean(pf[res])) if res.any() else np.nan
    return dict(has=1, net_pay_ft=round(net_pay_ft, 1), phi_h=round(phi_h, 2),
                clean_ft=round(clean_ft, 1), mean_phi=round(mean_phi, 3),
                gr_p50=round(float(np.nanmedian(gr)), 1))


def main():
    rows = list(csv.DictReader(open(MASTER, encoding="utf-8")))
    # single-well lease, joined to production, log present
    sub = []
    for r in rows:
        if r["las"] and r.get("cum_oil") not in ("", None) and r.get("max_wells") == "1":
            fp = os.path.join(LASDIR, r["kid"] + ".las")
            if not os.path.exists(fp):
                continue
            f = petro(fp)
            if f and f.get("has"):
                try:
                    r["cum"] = float(r["cum_oil"]); r["b12"] = float(r["best12_oil"] or 0)
                except ValueError:
                    continue
                r.update(f); sub.append(r)
    print(f"county {CC}: powered single-well + valid-log sample n = {len(sub)}")
    if len(sub) < 20:
        print("too few"); return

    cum = np.array([r["cum"] for r in sub])
    b12 = np.array([r["b12"] for r in sub])
    print(f"cum_oil  bbl: p10={np.percentile(cum,10):,.0f} p50={np.median(cum):,.0f} p90={np.percentile(cum,90):,.0f}")
    print(f"best12   bbl: p10={np.percentile(b12,10):,.0f} p50={np.median(b12):,.0f} p90={np.percentile(b12,90):,.0f}")

    def C(xk, yarr, lbl, expect):
        x = np.array([r[xk] for r in sub], float)
        m = np.isfinite(x) & np.isfinite(yarr)
        rho, p = spearmanr(x[m], yarr[m])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {lbl:30} n={m.sum():3}  rho={rho:+.3f}  p={p:.2e} {sig:3}  (expect {expect})")
        return rho

    feats = [("phi_h","porosity-feet (phi*h)","+"), ("net_pay_ft","net pay ft","+"),
             ("clean_ft","clean carbonate ft","+"), ("mean_phi","mean porosity","+"),
             ("gr_p50","GR p50","-")]
    print("\n[A] vs CUMULATIVE oil (age-biased):")
    for k, l, e in feats: C(k, cum, l, e)
    print("\n[B] vs BEST-12-MONTH oil (age-independent productivity -- the fair test):")
    for k, l, e in feats: C(k, b12, l, e)

    # age confound demonstration
    age = np.array([2026 - int(r["first_ym"][:4]) if r.get("first_ym") else np.nan for r in sub], float) \
        if "first_ym" in sub[0] else None
    s = sorted(sub, key=lambda r: r["phi_h"]); n = len(s); lo = s[:n//3]; hi = s[-n//3:]
    print(f"\n  phi*h top tercile (n={len(hi)}) median best12 = {np.median([r['b12'] for r in hi]):,.0f} bbl")
    print(f"  phi*h bot tercile (n={len(lo)}) median best12 = {np.median([r['b12'] for r in lo]):,.0f} bbl")
    lift = np.median([r['b12'] for r in hi]) / max(1, np.median([r['b12'] for r in lo]))
    print(f"  lift (top/bottom phi*h tercile, best12) = {lift:.2f}x")


if __name__ == "__main__":
    main()
