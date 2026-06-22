#!/usr/bin/env python
"""Cross-well MWD feature table (LOCAL ONLY -- proprietary logs).

One row per drilling DEPTH-log run, with the five validated cleaning rules baked in
(derived + proven in well_deepdive.py -- imported here so there's a single source of
truth, no re-derivation):

  1. per-file torque unit auto-normalization      (pick_torque)
  2. WOB on-bottom masking                          (drilling mask, not raw SWOB)
  3. ROP spikes absorbed by rank/median statistics  (Spearman + medians, spike-robust)
  4. MSE-vs-GR computed PER RUN and tagged with depth regime  (never pooled)
  5. MSE absolute at a fixed reference bit; the rho columns are bit-invariant comparables

Output CSV is written to a LOCAL, NON-REPO directory so nothing proprietary can be
committed/exfiltrated. Aggregates print to stdout (for the data owner's own screen).

    python cross_well_table.py [root_dir] [out_csv]
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging
logging.getLogger("lasio").setLevel(logging.CRITICAL)   # silence per-curve parse chatter
import numpy as np
import lasio
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from las_inspect import walk_las
from well_deepdive import col, first, pick_torque, teale_mse

REF_BIT_IN = 8.75          # fixed reference; absolute MSE scales as 1/area, flagged below
DEF_ROOTS = [r"E:\JONESMWDSTK", r"E:\Data", r"E:\MWD-USB"]          # all local moat drives
DEF_OUT   = r"C:\Users\JT-DEV1\mwd_local_out\cross_well_table.csv"  # OFF the git repo

COLS = ["well", "company", "field", "indexed_by", "run_tag", "n_drill",
        "md_lo", "md_hi", "tvd_lo", "tvd_hi", "lat_frac", "section", "depth_regime",
        "tq_src", "tq_scale", "bit_in",
        "mse_p50_ksi", "gr_p50", "mse_clean_ksi", "mse_shale_ksi",
        "mse_gr_rho", "tq_wob_rho", "slide_frac", "rop_slide", "rop_rotate", "flags"]


def _rho(a, b, m):
    mm = m & np.isfinite(a) & np.isfinite(b)
    if mm.sum() < 30:
        return np.nan, 0
    r, _ = spearmanr(a[mm], b[mm])
    return r, int(mm.sum())


def hdr(las, key, default="?"):
    try:
        return str(las.well[key].value).strip()
    except Exception:
        return default


def is_depth_log(md):
    if md.size < 50 or not np.isfinite(md[0]):
        return False
    d = np.nanmedian(np.diff(md))
    return (0 <= md[0] < 40000) and (0.02 < abs(d) < 5)


def usable_depth_axis(a):
    """True if `a` can serve as a depth axis (a real DEPT curve on a time-indexed file)."""
    if a is None:
        return False
    fin = a[np.isfinite(a)]
    return (fin.size >= 50 and 0 <= np.min(fin) < 40000
            and np.max(fin) < 40000 and (np.max(fin) - np.min(fin)) > 100)


def process(path: Path):
    flags = []
    las = lasio.read(str(path), ignore_header_errors=True)
    idx = np.asarray(las.index, dtype=float)
    if is_depth_log(idx):
        md, indexed_by = idx, "depth"
    else:
        dept = col(las, "DEPT")                        # time-indexed file: fall back to DEPT curve
        if usable_depth_axis(dept):
            md, indexed_by = dept, "time"
            flags.append("time_indexed")               # stats are per-time-sample, not per-foot
        else:
            return None                                # truly no usable depth axis

    wob, _ = first(las, "SWOB", "WOBX")
    rpm = col(las, "RPM")
    rop = col(las, "ROP")
    grc, _ = first(las, "GRC", "GRM1", "GR", "GR_MWD")
    tvd, _ = first(las, "MTTVD", "TVD", "TVDE")
    slbd = col(las, "SLBD")
    if any(x is None for x in (wob, rpm, rop)):
        return None

    drilling = (np.isfinite(rop) & (rop > 5) & np.isfinite(wob) & (wob > 1)
                & np.isfinite(rpm) & (rpm > 10))
    if drilling.sum() < 200:
        return None                                   # not enough real drilling

    tq, tq_src, tq_scale = pick_torque(las, drilling)
    if tq_scale != 1.0:
        flags.append("tq_was_ftlbf")

    # MSE at the reference bit (absolute is bit-scaled; rho columns are bit-invariant)
    mse = teale_mse(wob, rpm, tq, rop, REF_BIT_IN)
    mok = drilling & np.isfinite(mse) & (mse > 0)
    mse_p50 = np.median(mse[mok]) / 1e3 if mok.any() else np.nan

    # gamma lithology split
    mse_clean = mse_shale = gr_p50 = np.nan
    mse_gr_rho = np.nan
    if grc is not None and np.isfinite(grc[drilling]).sum() > 50:
        gr_p50 = np.nanmedian(grc[drilling])
        lo = np.nanpercentile(grc[drilling], 33)
        hi = np.nanpercentile(grc[drilling], 67)
        clean = mok & (grc < lo)
        shale = mok & (grc > hi)
        if clean.any(): mse_clean = np.nanmedian(mse[clean]) / 1e3
        if shale.any(): mse_shale = np.nanmedian(mse[shale]) / 1e3
        mse_gr_rho, _ = _rho(mse, grc, mok)

    tq_wob_rho, _ = _rho(tq, wob, drilling)

    # slide / rotate
    slide_frac = rop_slide = rop_rotate = np.nan
    if slbd is not None and np.isfinite(slbd).any():
        sfin = slbd[np.isfinite(slbd)]
        if set(np.unique(sfin).tolist()) <= {0.0, 1.0}:
            sl = np.isfinite(slbd) & (slbd == 1)
            ro = np.isfinite(slbd) & (slbd == 0)
            slide_frac = float(sl.mean())
            if (sl & (rop > 0)).any(): rop_slide = float(np.nanmedian(rop[sl & (rop > 0)]))
            if (ro & (rop > 0)).any(): rop_rotate = float(np.nanmedian(rop[ro & (rop > 0)]))

    # trajectory -> section class from cos(inc) = dTVD/dMD (INC channel is too sparse)
    lat_frac = np.nan
    section = "?"
    tvd_lo = tvd_hi = np.nan
    if tvd is not None and np.isfinite(tvd).sum() > 10:
        tvf = tvd[np.isfinite(tvd)]
        tvd_lo, tvd_hi = float(tvf.min()), float(tvf.max())
        dmd = np.diff(md); dtvd = np.diff(tvd)
        good = np.isfinite(dtvd) & (dmd > 1e-6)
        if good.sum() > 30:
            cos_inc = np.clip(dtvd[good] / dmd[good], -1, 1)
            inc = np.degrees(np.arccos(cos_inc))
            lat_frac = float(np.mean(inc > 60))
            build_frac = float(np.mean(inc > 15))
            section = "LAT" if lat_frac > 0.30 else ("BUILD" if build_frac > 0.10 else "VERT")

    md_hi = float(md[np.isfinite(md)].max())
    depth_regime = "deep" if md_hi >= 5000 else "shallow"
    if mse_p50 > 300 or (np.isfinite(mse_p50) and mse_p50 < 1):
        flags.append("mse_suspect")

    run_tag = path.parent.parent.name[:40] if path.parent.name.upper() in ("LAS", "DM LAS") \
        else path.parent.name[:40]
    return {
        "well": hdr(las, "WELL"), "company": hdr(las, "COMP"), "field": hdr(las, "FLD"),
        "indexed_by": indexed_by,
        "run_tag": run_tag, "n_drill": int(drilling.sum()),
        "md_lo": round(float(md[np.isfinite(md)].min()), 0), "md_hi": round(md_hi, 0),
        "tvd_lo": round(tvd_lo, 0) if np.isfinite(tvd_lo) else "",
        "tvd_hi": round(tvd_hi, 0) if np.isfinite(tvd_hi) else "",
        "lat_frac": round(lat_frac, 3) if np.isfinite(lat_frac) else "",
        "section": section, "depth_regime": depth_regime,
        "tq_src": tq_src, "tq_scale": tq_scale, "bit_in": REF_BIT_IN,
        "mse_p50_ksi": round(mse_p50, 1) if np.isfinite(mse_p50) else "",
        "gr_p50": round(gr_p50, 1) if np.isfinite(gr_p50) else "",
        "mse_clean_ksi": round(mse_clean, 1) if np.isfinite(mse_clean) else "",
        "mse_shale_ksi": round(mse_shale, 1) if np.isfinite(mse_shale) else "",
        "mse_gr_rho": round(mse_gr_rho, 3) if np.isfinite(mse_gr_rho) else "",
        "tq_wob_rho": round(tq_wob_rho, 3) if np.isfinite(tq_wob_rho) else "",
        "slide_frac": round(slide_frac, 3) if np.isfinite(slide_frac) else "",
        "rop_slide": round(rop_slide, 0) if np.isfinite(rop_slide) else "",
        "rop_rotate": round(rop_rotate, 0) if np.isfinite(rop_rotate) else "",
        "flags": ";".join(flags),
    }


def wkey(name):
    """Normalize a WELL header to collapse case/spacing/punctuation variants of one well."""
    return "".join(ch for ch in name.upper() if ch.isalnum())


def dedup_runs(rows):
    """Collapse content-duplicate runs (same well + MD span, backed up to many folders).
    Keep the RICHEST copy -- most resolved fields, and prefer depth- over time-indexing."""
    def sig(r):
        return (wkey(r["well"]), round(float(r["md_lo"]) / 10.0), round(float(r["md_hi"]) / 10.0),
                round(int(r["n_drill"]) / 50.0))
    def richness(r):
        return (sum(1 for k in ("section", "gr_p50", "mse_gr_rho", "tvd_hi", "slide_frac")
                    if r[k] not in ("", "?"))
                + (1 if r["indexed_by"] == "depth" else 0))
    best = {}
    for r in rows:
        s = sig(r)
        if s not in best or richness(r) > richness(best[s]):
            best[s] = r
    return list(best.values())


def main():
    # args: any number of root dirs + an optional *.csv output path
    args = sys.argv[1:]
    out = Path(DEF_OUT)
    roots = []
    for a in args:
        (out := Path(a)) if a.lower().endswith(".csv") else roots.append(Path(a))
    if not roots:
        roots = [Path(r) for r in DEF_ROOTS]
    out.parent.mkdir(parents=True, exist_ok=True)

    files, seen = [], set()
    for root in roots:
        if not root.exists():
            print(f"  (root not found, skipping: {root})")
            continue
        for p in walk_las(root):
            k = str(p).lower()
            if k not in seen:
                seen.add(k)
                files.append(p)
    print(f"scanning {len(files)} .las across {len(roots)} root(s): "
          f"{', '.join(r.name for r in roots)}  (output -> {out}, LOCAL ONLY)")
    rows, skip, err = [], 0, 0
    for i, f in enumerate(files):
        try:
            r = process(f)
            if r is None:
                skip += 1
            else:
                rows.append(r)
        except Exception:
            err += 1
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(files)}  kept={len(rows)} skip={skip} err={err}", flush=True)

    n_raw = len(rows)
    rows = dedup_runs(rows)
    print(f"\n  de-duplicated backup copies: {n_raw} parsed rows -> {len(rows)} unique runs")

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    # ---- aggregate summary (stdout) --------------------------------------------
    print(f"\n=== CROSS-WELL TABLE: {len(rows)} runs kept "
          f"({skip} skipped, {err} errored) ===")
    wells = {wkey(r["well"]) for r in rows}
    n_time = sum(r["indexed_by"] == "time" for r in rows)
    print(f"  distinct wells (normalized): {len(wells)}   "
          f"({n_time} runs recovered via DEPT curve on time-indexed logs)")
    from collections import Counter
    fld = Counter(r["field"] for r in rows)
    print(f"  fields (runs): " + ", ".join(f"{k}={v}" for k, v in fld.most_common(8)))

    def col_arr(key, where=lambda r: True):
        return np.array([r[key] for r in rows if where(r) and r[key] != ""], dtype=float)

    n_ftlbf = sum("tq_was_ftlbf" in r["flags"] for r in rows)
    print(f"  rule-1 impact: {n_ftlbf}/{len(rows)} runs logged torque as raw ft-lbf "
          f"(MSE would be 1000x wrong without normalization)")

    for sec in ("VERT", "BUILD", "LAT", "?"):
        sub = [r for r in rows if r["section"] == sec]
        if sub:
            print(f"  section {sec:5}: {len(sub):3} runs")

    print("\n  MSE-vs-GR rho by depth regime (tests the deep-dive depth-dependence at scale):")
    for reg in ("shallow", "deep"):
        a = col_arr("mse_gr_rho", lambda r: r["depth_regime"] == reg)
        if a.size:
            print(f"    {reg:8}: n={a.size:3}  median rho={np.median(a):+.3f}  "
                  f"[frac negative={np.mean(a<0):.2f}]")

    print("\n  TQ-vs-WOB rho by section (bit-aggressiveness shallow vs drag-dominated deep):")
    for sec in ("VERT", "BUILD", "LAT"):
        a = col_arr("tq_wob_rho", lambda r: r["section"] == sec)
        if a.size:
            print(f"    {sec:5}: n={a.size:3}  median rho={np.median(a):+.3f}")

    print("\n  slide vs rotate ROP (steered runs only, slide_frac>0.02):")
    sl = col_arr("rop_slide", lambda r: r["slide_frac"] != "" and float(r["slide_frac"]) > 0.02)
    ro = col_arr("rop_rotate", lambda r: r["slide_frac"] != "" and float(r["slide_frac"]) > 0.02)
    if sl.size and ro.size:
        print(f"    n={sl.size} steered runs: median ROP slide={np.median(sl):.0f} "
              f"rotate={np.median(ro):.0f} ft/hr")

    print(f"\n  full table written to {out}")


if __name__ == "__main__":
    main()
