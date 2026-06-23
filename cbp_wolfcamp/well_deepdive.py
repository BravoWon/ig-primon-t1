#!/usr/bin/env python
"""Single-well channel deep-dive / validator (LOCAL ONLY -- proprietary MWD logs).

Goal: before trusting any MWD channel in a cross-well table, ground-truth each one
against physical expectation and cross-check it against the others. We never dump
per-foot proprietary curve values -- only aggregate stats, correlations, and PASS/
FLAG verdicts.

Validations performed
---------------------
  * coverage          : % non-null over the drilling interval
  * physical range     : each channel inside a plausible envelope?
  * MSE (Teale)        : mechanical specific energy per foot (bit-area sensitive in
                         ABSOLUTE level only; all correlations below are area-invariant)
  * slide vs rotate    : SLBD flag should partition RPM (slide => surface RPM ~ 0)
  * torque vs weight   : TQ should rise with WOB (bit aggressiveness)  -> Spearman > 0
  * MSE vs gamma       : clean (low-GR) rock should drill harder than shale -> rho < 0
  * MD vs TVD          : monotone, TVD <= MD, build section sanity

    python well_deepdive.py <DM_depth.las> [bit_diameter_in]
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import lasio
from scipy.stats import spearmanr

NULL = -999.25
# Plausible physical envelopes for surface/near-bit MWD channels (loose -- flag gross only)
ENVELOPE = {
    "ROP":  (0.0, 600.0,   "ft/hr"),
    "SWOB": (-2.0, 90.0,   "klbf"),   # small negative = slips/float, tolerated
    "WOBX": (-2.0, 90.0,   "klbf"),
    "TQA":  (0.0, 60.0,    "klb.ft"),
    "TQX":  (0.0, 60.0,    "klb.ft"),
    "RPM":  (0.0, 300.0,   "rpm"),
    "SPP":  (0.0, 7000.0,  "psi"),
    "SPPA": (0.0, 7000.0,  "psi"),
    "GRC":  (0.0, 350.0,   "API"),
    "INC":  (0.0, 120.0,   "deg"),
    "MTTVD":(0.0, 30000.0, "ft"),
    "HKLD": (0.0, 600.0,   "klbf"),
}


def col(las, name):
    try:
        a = np.asarray(las[name], dtype=float)
        a[a == NULL] = np.nan
        return a
    except Exception:
        return None


def first(las, *names):
    for n in names:
        a = col(las, n)
        if a is not None and np.isfinite(a).any():
            return a, n
    return None, None


def pct(x):
    return 100.0 * np.isfinite(x).mean()


def pick_torque(las, mask):
    """Return (torque_in_klbft, source_mnemonic, scale_applied).

    The SAME mnemonic (TQA) carries klb.ft on some wells and raw ft-lbf on others,
    regardless of the header unit string. Auto-detect by magnitude: repeatedly /1000
    until the median lands in a sane klb.ft envelope. Skip dead/zero channels.
    """
    best = None
    for nm in ("TQA", "TQX", "TQ"):
        a = col(las, nm)
        if a is None or not np.isfinite(a).any():
            continue
        ref = a[mask] if mask.any() else a
        med = np.nanmedian(np.abs(ref))
        if not np.isfinite(med):
            continue
        scale, m = 1.0, med
        while m > 100.0:                       # ft-lbf / lb-ft -> klb.ft
            m /= 1000.0
            scale /= 1000.0
        if 0.1 < m < 60.0:                      # in a sane klb.ft band
            return a * scale, nm, scale
        best = best or (a, nm, 1.0)
    return best if best else (col(las, "TQA"), "TQA", 1.0)


def teale_mse(wob_klbf, rpm, tq_klbft, rop_fthr, d_in):
    """Teale mechanical specific energy (psi). Bit area enters as 1/A on BOTH terms,
    so it scales the absolute level but cancels in every correlation/ratio below."""
    A = np.pi / 4.0 * d_in**2                       # in^2
    wob = wob_klbf * 1000.0                          # lbf
    tq = tq_klbft * 1000.0                           # ft-lbf
    with np.errstate(divide="ignore", invalid="ignore"):
        mse = wob / A + (120.0 * np.pi * rpm * tq) / (A * rop_fthr)
    return mse


def verdict(ok):
    return "PASS" if ok else "FLAG"


def main():
    path = sys.argv[1]
    d_in = float(sys.argv[2]) if len(sys.argv) > 2 else 8.75
    las = lasio.read(path, ignore_header_errors=True)

    well = str(las.well["WELL"].value) if "WELL" in las.well else "?"
    comp = str(las.well["COMP"].value) if "COMP" in las.well else "?"
    fld  = str(las.well["FLD"].value) if "FLD" in las.well else "?"
    md   = las.index
    print(f"\n=== DEEP DIVE: {well} | {comp} | {fld} ===")
    print(f"  file index: MD {md[0]:.0f}..{md[-1]:.0f} ft, {len(md)} samples, "
          f"step ~{np.nanmedian(np.diff(md)):.2f} ft, bit assumed {d_in}in")

    # ---- 1. coverage + physical range per channel -------------------------------
    print("\n  [1] per-channel coverage + physical range")
    print(f"    {'chan':6} {'cov%':>6} {'min':>9} {'p50':>9} {'max':>9}  {'unit':6} verdict")
    chans = {}
    for name, (lo, hi, unit) in ENVELOPE.items():
        a = col(las, name)
        if a is None:
            continue
        chans[name] = a
        fin = a[np.isfinite(a)]
        if fin.size == 0:
            print(f"    {name:6} {pct(a):6.1f}   (no valid samples)")
            continue
        inrange = np.mean((fin >= lo) & (fin <= hi))
        ok = inrange > 0.98
        print(f"    {name:6} {pct(a):6.1f} {fin.min():9.1f} {np.median(fin):9.1f} "
              f"{fin.max():9.1f}  {unit:6} {verdict(ok)}"
              + ("" if ok else f"  ({100*(1-inrange):.1f}% out of [{lo},{hi}])"))

    # pick canonical channels
    wob, wob_n = first(las, "SWOB", "WOBX")
    rpm = chans.get("RPM")
    rop = chans.get("ROP")
    spp, spp_n = first(las, "SPP", "SPPA")
    grc = chans.get("GRC")
    tvd = chans.get("MTTVD")
    slbd = col(las, "SLBD")

    # drilling mask: actually making hole, rotating, on bottom
    drilling = (np.isfinite(rop) & (rop > 5) & np.isfinite(wob) & (wob > 1)
                & np.isfinite(rpm) & (rpm > 10))
    # torque needs the drilling mask to auto-detect units -> pick it AFTER the mask
    tq, tq_n, tq_scale = pick_torque(las, drilling)
    sc = "x1" if tq_scale == 1.0 else f"x{tq_scale:g} (was ft-lbf)"
    print(f"\n  drilling samples (ROP>5, WOB>1, RPM>10): {drilling.sum()} "
          f"({100*drilling.mean():.0f}% of file)  [WOB={wob_n} TQ={tq_n} {sc} SPP={spp_n}]")

    # ---- 2. MSE (Teale) ---------------------------------------------------------
    print("\n  [2] MSE (Teale, rotating-only)")
    mse = np.full_like(rop, np.nan, dtype=float) if tq is None else teale_mse(wob, rpm, tq, rop, d_in)
    mse_d = mse[drilling]
    mse_d = mse_d[np.isfinite(mse_d) & (mse_d > 0)]
    if mse_d.size:
        print(f"    n={mse_d.size}  MSE ksi: p10={np.percentile(mse_d,10)/1e3:.1f} "
              f"p50={np.median(mse_d)/1e3:.1f} p90={np.percentile(mse_d,90)/1e3:.1f}")
        print(f"    (absolute level scales as 1/bit-area; correlations below are area-invariant)")

    # ---- 3. slide vs rotate (SLBD partitions RPM?) ------------------------------
    print("\n  [3] slide vs rotate  (SLBD flag should partition surface RPM)")
    if slbd is not None and np.isfinite(slbd).any() and rpm is not None:
        uvals = np.unique(slbd[np.isfinite(slbd)])
        on = np.isfinite(slbd) & np.isfinite(rpm)
        if set(np.unique(slbd[on]).tolist()) <= {0.0, 1.0}:
            sl = on & (slbd == 1)
            ro = on & (slbd == 0)
            rpm_sl = np.median(rpm[sl]) if sl.sum() else np.nan
            rpm_ro = np.median(rpm[ro]) if ro.sum() else np.nan
            if sl.sum() == 0:
                # no sliding -> vertical/rotary-only run; the partition test is N/A, not a fail
                print(f"    SLBD all rotate (slide n=0): vertical/rotary-only run -> N/A "
                      f"(needs a steered/lateral run to exercise this channel)")
            else:
                ok = (rpm_sl < rpm_ro) if np.isfinite(rpm_sl) and np.isfinite(rpm_ro) else False
                print(f"    SLBD in {{0,1}}: slide n={sl.sum()} med RPM={rpm_sl:.0f} | "
                      f"rotate n={ro.sum()} med RPM={rpm_ro:.0f}  -> {verdict(ok)}")
            # ROP penalty while sliding?
            if rop is not None:
                rop_sl = np.nanmedian(rop[sl & (rop > 0)]) if sl.sum() else np.nan
                rop_ro = np.nanmedian(rop[ro & (rop > 0)]) if ro.sum() else np.nan
                print(f"    ROP ft/hr: slide={rop_sl:.0f} rotate={rop_ro:.0f} "
                      f"(slide usually slower)")
        else:
            print(f"    SLBD values not binary {{0,1}}: {uvals[:6]} -- treating as continuous")
    else:
        print("    SLBD or RPM unavailable")

    # ---- 4. cross-consistency correlations --------------------------------------
    print("\n  [4] cross-channel physical consistency (Spearman over drilling samples)")

    def rho(a, b, m):
        mm = m & np.isfinite(a) & np.isfinite(b)
        if mm.sum() < 30:
            return np.nan, 0
        r, _ = spearmanr(a[mm], b[mm])
        return r, mm.sum()

    r_tw, n_tw = rho(tq, wob, drilling)
    print(f"    TQ vs WOB   rho={r_tw:+.3f} (n={n_tw})  expect >0 (bit bites harder)"
          f"  -> {verdict(np.isfinite(r_tw) and r_tw > 0)}")

    if grc is not None:
        r_mg, n_mg = rho(mse, grc, drilling & np.isfinite(mse) & (mse > 0))
        clean = drilling & np.isfinite(grc) & (grc < np.nanpercentile(grc[drilling], 33))
        shale = drilling & np.isfinite(grc) & (grc > np.nanpercentile(grc[drilling], 67))
        mse_clean = np.nanmedian(mse[clean & (mse > 0)]) / 1e3
        mse_shale = np.nanmedian(mse[shale & (mse > 0)]) / 1e3
        print(f"    MSE vs GR   rho={r_mg:+.3f} (n={n_mg})  expect <0 (clean rock harder)"
              f"  -> {verdict(np.isfinite(r_mg) and r_mg < 0)}")
        print(f"        clean(low-GR) MSE={mse_clean:.1f} ksi  vs  shale(high-GR)={mse_shale:.1f} ksi")

    r_sr, n_sr = rho(spp, rop, drilling)
    print(f"    SPP vs ROP  rho={r_sr:+.3f} (n={n_sr})  (weak/positive; pressure tracks hydraulics)")

    # ---- 5. trajectory sanity ---------------------------------------------------
    print("\n  [5] trajectory (MD vs TVD)")
    if tvd is not None and np.isfinite(tvd).any():
        m = np.isfinite(tvd)
        dtvd = np.diff(tvd[m])
        mono = np.mean(dtvd >= -0.01)
        tvd_le_md = np.mean(tvd[m] <= md[m] + 0.5)
        print(f"    TVD monotone-increasing frac={mono:.3f}  TVD<=MD frac={tvd_le_md:.3f}"
              f"  -> {verdict(mono > 0.98 and tvd_le_md > 0.98)}")
        print(f"    final MD={md[-1]:.0f} TVD={tvd[m][-1]:.0f} (vertical section -- "
              f"MD~TVD expected for a top-hole run)")

    print("\n  done.")


if __name__ == "__main__":
    main()
