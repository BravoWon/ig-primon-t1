#!/usr/bin/env python
"""Quantify the ONE survivor of the rock->productivity null campaign (PUBLIC data).

Wringer survivor: total preserved Penn+Miss section thickness (Kansas City top -> a deep
marker) -> more oil (rho~+0.22 ***, consistent across Kans-Marm/Miss/Arbu, per-county
detrended). Hypothesis: this is a basin-accommodation / structural-position signal readable
from FREE public structure maps -- and it is the only thing that pays.

Same adversarial bar we used to KILL the isopach lead, now turned on the survivor:
  H  HEADLINE   cleaned per-county-detrended thickness -> log best12
  A  PARTIAL    control structure (we know it's null alone) AND absolute depth
                -> is it section THICKNESS, or just being deep / structurally low?
  B  FE REGR    log(best12) ~ z(thick)+z(struct)+z(depth) + county FE + vintage FE
                -> standardized beta on thickness, t-stat, incremental R^2
  C  EFFECT     within-county thickness quartile -> median best12 (monotone? how big?)
  D  MEDIATION  does the producing ZONE explain it away (thick -> better zone -> oil)?
  E  HOLDOUT    leave-one-county-out OOS rank-IC + random 70/30 -> does it GENERALIZE?

    python kgs_basin_position.py
"""
from __future__ import annotations
import csv, re
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import spearmanr, rankdata

PROD = r"C:\Users\JT-DEV1\kgs_public\lease_prod.csv"
TOPS = r"C:\Users\JT-DEV1\kgs_public\tops\ks_tops.txt"
CKU = {"BARTON","ELLIS","RUSSELL","NESS","ROOKS","STAFFORD","BARBER","RICE","TREGO",
       "COMANCHE","KIOWA","PAWNEE","RUSH","ELLSWORTH","OSBORNE","LINCOLN"}
DEEP = "MISSISSIPPIAN"   # primary deep marker (cleanest large-n survivor)


def a10(s):
    d = re.sub(r"\D", "", s or "")
    return d[:10] if len(d) >= 10 else ""


def zscore(v):
    s = np.std(v)
    return (v - np.mean(v)) / s if s > 0 else np.zeros_like(v)


def detrend_county(v, lon, lat, cty):
    """Remove per-county planar (lon,lat) trend -> relative-within-county residual."""
    out = np.zeros_like(v, float)
    for c in set(cty):
        m = np.array([x == c for x in cty])
        if m.sum() < 15:
            out[m] = v[m] - np.mean(v[m]); continue
        A = np.c_[np.ones(m.sum()), lon[m], lat[m]]
        coef, *_ = np.linalg.lstsq(A, v[m], rcond=None)
        out[m] = v[m] - A @ coef
    return out


def partial_spearman(x, y, controls):
    """Spearman of x vs y after linearly removing controls from the ranks of both."""
    rx, ry = rankdata(x), rankdata(y)
    C = np.c_[np.ones(len(x)), *[rankdata(c) for c in controls]]
    bx, *_ = np.linalg.lstsq(C, rx, rcond=None)
    by, *_ = np.linalg.lstsq(C, ry, rcond=None)
    ex, ey = rx - C @ bx, ry - C @ by
    return spearmanr(ex, ey)


def main():
    wells = {}
    for r in csv.DictReader(open(PROD, encoding="utf-8")):
        if r["max_wells"] != "1" or r["county"].upper() not in CKU:
            continue
        a = next((a10(t) for t in r["apis"].split(",") if a10(t)), "")
        if not a:
            continue
        try:
            b12 = float(r["best12_oil"] or 0)
        except ValueError:
            continue
        if b12 <= 0:
            continue
        ym = re.sub(r"\D", "", r.get("first_ym") or "")
        wells[a] = dict(b12=b12, cty=r["county"].upper(), zone=(r.get("zone") or "").strip(),
                        vint=(ym[:4] if len(ym) >= 4 else "?"))
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
        w.setdefault("lat", float(r["LATITUDE"])); w.setdefault("lon", float(r["LONGITUDE"]))
        try:
            w.setdefault("elev", float(r["ELEVATION"]))
        except (ValueError, KeyError):
            pass
        if "KANSAS CITY" in form:
            w.setdefault("KC", top)
        if DEEP in form:
            w.setdefault("DEEP", top)

    # base set: KC + deep marker + position
    W = [w for w in wells.values()
         if "KC" in w and "DEEP" in w and "lat" in w and "elev" in w]
    thick = np.array([w["DEEP"] - w["KC"] for w in W])
    good = (thick > 5) & (thick < 3000)
    W = [W[i] for i in np.where(good)[0]]
    thick = thick[good]
    b12 = np.array([w["b12"] for w in W])
    lon = np.array([w["lon"] for w in W]); lat = np.array([w["lat"] for w in W])
    cty = [w["cty"] for w in W]
    depth = np.array([w["KC"] for w in W])                 # absolute KC depth
    struct = np.array([w["elev"] - w["KC"] for w in W])    # KC subsea (structural position)
    logb = np.log(b12)

    print(f"[BASIN POSITION] KC->{DEEP} section thickness -> best12, CKU single-well leases")
    print(f"  n={len(W)}  thickness ft: p10={np.percentile(thick,10):.0f} "
          f"p50={np.median(thick):.0f} p90={np.percentile(thick,90):.0f}  "
          f"(median != 0 => real isopach, not the combined-marker trap)")

    # ---- H HEADLINE -------------------------------------------------------
    dthk = detrend_county(thick, lon, lat, cty)
    dlogb = detrend_county(logb, lon, lat, cty)
    rho, p = spearmanr(dthk, dlogb)
    print(f"\n[H] HEADLINE  per-county-detrended thickness -> log best12:  "
          f"rho={rho:+.3f}  p={p:.1e}")

    # ---- A PARTIAL --------------------------------------------------------
    print("\n[A] PARTIAL (is it thickness, or just depth / structural position?)")
    print(f"  structure -> log best12 (alone): "
          f"rho={spearmanr(detrend_county(struct,lon,lat,cty), dlogb)[0]:+.3f}")
    print(f"  depth     -> log best12 (alone): "
          f"rho={spearmanr(detrend_county(depth,lon,lat,cty), dlogb)[0]:+.3f}")
    rs, ps = partial_spearman(dthk, dlogb, [detrend_county(struct, lon, lat, cty)])
    print(f"  thickness -> best12 | structure:        rho={rs:+.3f}  p={ps:.1e}")
    rd, pd = partial_spearman(dthk, dlogb,
                              [detrend_county(struct, lon, lat, cty),
                               detrend_county(depth, lon, lat, cty)])
    print(f"  thickness -> best12 | structure+depth:  rho={rd:+.3f}  p={pd:.1e}")

    # ---- B FE REGRESSION --------------------------------------------------
    def dummies(keys):
        levels = sorted(set(keys))[1:]   # drop one for identifiability
        return np.array([[1.0 if k == lv else 0.0 for lv in levels] for k in keys])
    ztk, zst, zdp = zscore(thick), zscore(struct), zscore(depth)
    Xc = np.c_[np.ones(len(W)), ztk, zst, zdp, dummies(cty), dummies([w["vint"] for w in W])]
    beta, *_ = np.linalg.lstsq(Xc, logb, rcond=None)
    resid = logb - Xc @ beta
    dof = len(W) - Xc.shape[1]
    sigma2 = resid @ resid / dof
    XtX_inv = np.linalg.pinv(Xc.T @ Xc)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    r2_full = 1 - np.var(resid) / np.var(logb)
    # drop thickness col -> incremental R^2
    Xn = np.delete(Xc, 1, axis=1)
    bn, *_ = np.linalg.lstsq(Xn, logb, rcond=None)
    r2_noth = 1 - np.var(logb - Xn @ bn) / np.var(logb)
    print("\n[B] FE REGRESSION  log(best12) ~ z(thick)+z(struct)+z(depth) + county + vintage")
    print(f"  beta z(thickness) = {beta[1]:+.3f} log-units/sd  (t={beta[1]/se[1]:+.1f})  "
          f"=> {100*(np.exp(beta[1])-1):+.0f}% best12 per +1sd section")
    print(f"  beta z(struct)    = {beta[2]:+.3f} (t={beta[2]/se[2]:+.1f})   "
          f"beta z(depth) = {beta[3]:+.3f} (t={beta[3]/se[3]:+.1f})")
    print(f"  model R^2={r2_full:.3f}; without thickness R^2={r2_noth:.3f}  "
          f"-> thickness adds {r2_full-r2_noth:+.4f}")

    # ---- C EFFECT SIZE ----------------------------------------------------
    q = np.array([np.searchsorted(np.percentile(dthk, [25, 50, 75]), v) for v in dthk])
    print("\n[C] EFFECT SIZE  within-county-detrended thickness quartile -> median best12 (bbl):")
    for k in range(4):
        m = q == k
        print(f"    Q{k+1}  n={m.sum():4}  median best12 = {np.median(b12[m]):7.0f}")
    print(f"  Q4/Q1 median ratio = {np.median(b12[q==3])/max(1,np.median(b12[q==0])):.2f}x")

    # ---- D ZONE MEDIATION -------------------------------------------------
    print("\n[D] MEDIATION  does producing ZONE explain it away?")
    zc = Counter(w["zone"] for w in W).most_common(4)
    print("  top zones: " + ", ".join(f"{z or '(blank)'}={n}" for z, n in zc))
    zlev = sorted({w["zone"] for w in W})[1:]
    Zd = np.array([[1.0 if w["zone"] == lv else 0.0 for lv in zlev] for w in W])
    rz, pz = partial_spearman(dthk, dlogb,
                              [detrend_county(struct, lon, lat, cty)]
                              + [Zd[:, j] for j in range(Zd.shape[1])])
    print(f"  thickness -> best12 | structure + ZONE dummies:  rho={rz:+.3f}  p={pz:.1e}")

    # ---- E HOLDOUT --------------------------------------------------------
    print("\n[E] HOLDOUT  does the slope GENERALIZE out of sample?")
    ic = []
    counties = sorted(set(cty))
    for c in counties:
        te = np.array([x == c for x in cty]); tr = ~te
        if te.sum() < 30 or tr.sum() < 100:
            continue
        # fit relative thickness->relative logb on TRAIN counties, score on held-out county
        a_tr = np.c_[np.ones(tr.sum()), detrend_county(thick[tr], lon[tr], lat[tr],
                     [cty[i] for i in np.where(tr)[0]])]
        coef, *_ = np.linalg.lstsq(a_tr, detrend_county(logb[tr], lon[tr], lat[tr],
                     [cty[i] for i in np.where(tr)[0]]), rcond=None)
        pred = coef[0] + coef[1] * detrend_county(thick[te], lon[te], lat[te],
                     [cty[i] for i in np.where(te)[0]])
        act = detrend_county(logb[te], lon[te], lat[te],
                     [cty[i] for i in np.where(te)[0]])
        r = spearmanr(pred, act)[0]
        ic.append((c, te.sum(), r))
    for c, n, r in ic:
        print(f"    held-out {c:10} n={n:4}  OOS rank-IC = {r:+.3f}")
    if ic:
        arr = np.array([r for _, _, r in ic])
        print(f"  leave-one-county-out: mean OOS IC = {arr.mean():+.3f}  "
              f"({(arr>0).sum()}/{len(arr)} counties positive)")

    # random 70/30 on pooled detrended signal
    idx = np.argsort(rankdata(dthk + 1e-9 * dlogb))  # deterministic, no RNG
    te = (np.arange(len(W)) % 10) < 3                 # deterministic 30% holdout
    a_tr = np.c_[np.ones((~te).sum()), dthk[~te]]
    coef, *_ = np.linalg.lstsq(a_tr, dlogb[~te], rcond=None)
    pr = coef[0] + coef[1] * dthk[te]
    print(f"  random 30% holdout: OOS rank-IC = {spearmanr(pr, dlogb[te])[0]:+.3f}  "
          f"(n_test={te.sum()})")


if __name__ == "__main__":
    main()
