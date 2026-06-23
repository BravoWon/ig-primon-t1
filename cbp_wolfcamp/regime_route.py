#!/usr/bin/env python
"""ROUTE prime (from 'The Algorithmic Battlefield' / GTA 90-01-007): hierarchical classifier ->
per-class sub-policy. Applied to the unsupervised 2-regime split that PERSIST found in the KGS
value-geometry run, to (a) characterize what separates the regimes and (b) test whether
regime-routed value fields beat the pooled field through the same VERIFY (LOCO OOS rank-IC) gate.

    python regime_route.py
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.stats import spearmanr
import value_geometry as vg


def within_ic(pred, y, blocks, mask, min_n=25):
    out = []
    for c in sorted(set(b for b, m in zip(blocks, mask) if m)):
        sel = np.array([blocks[i] == c and mask[i] and np.isfinite(pred[i])
                        for i in range(len(y))])
        if sel.sum() >= min_n:
            r = spearmanr(pred[sel], y[sel])[0]
            if np.isfinite(r):
                out.append(r)
    return np.array(out)


def main():
    W = vg.load()
    Xfull, names = vg.featurize(W)
    y = np.log(np.array([w["b12"] for w in W]))
    blocks = [w["cty"] for w in W]

    # rebuild the same complete-core manifold and recover the unsupervised regimes (unwarped H0)
    core = ["subsea_KANS", "depth_KANS", "thk_KANS_MISS", "lat", "lon"]
    ci = [names.index(n) for n in core]
    keep = np.all(np.isfinite(Xfull[:, ci]), axis=1)
    idx = np.where(keep)[0]
    Xc = Xfull[np.ix_(idx, ci)]; Xc = (Xc - Xc.mean(0)) / Xc.std(0)
    n = len(idx)
    E, W0 = vg.knn_graph(Xc, k=10)
    lab = vg.cut_clusters(n, E, W0, k=3)
    sizes = Counter(lab)
    big = [c for c, _ in sizes.most_common(2)]            # two largest = the regimes
    big.sort(key=lambda c: np.median(Xfull[idx[lab == c], names.index("depth_KANS")]))  # shallower KC = shelf = regime 0 (label-stable)
    print(f"[ROUTE on the PERSIST 2-regime split]  complete-core n={n}; "
          f"cluster sizes={dict(sizes)} -> regimes {big}")

    # map regime label back to the full well index (-1 = not in a major regime / no core features)
    regime = np.full(len(W), -1)
    for k_local, gi in enumerate(idx):
        regime[gi] = 0 if lab[k_local] == big[0] else (1 if lab[k_local] == big[1] else -1)

    iKsub = names.index("subsea_KANS"); iKdep = names.index("depth_KANS")
    iMiss = names.index("subsea_MISS"); iThk = names.index("thk_KANS_MISS")
    print("\n[CHARACTERIZE] what separates the two regimes:")
    for r in (0, 1):
        m = regime == r
        b12 = np.exp(y[m])
        cty = Counter(blocks[i] for i in range(len(W)) if m[i]).most_common(3)
        miss_pct = 100 * np.isfinite(Xfull[m, iMiss]).mean()
        print(f"  regime {r}: n={m.sum():4}  KC depth med={np.nanmedian(Xfull[m,iKdep]):5.0f}ft  "
              f"KC subsea med={np.nanmedian(Xfull[m,iKsub]):5.0f}  thk(KC-MISS) med="
              f"{np.nanmedian(Xfull[m,iThk]):4.0f}  best12 med={np.median(b12):5.0f}bbl  "
              f"Miss%={miss_pct:3.0f}  top counties={[c for c,_ in cty]}")

    # ---- VERIFY gate: pooled vs regime-routed value field, LOCO OOS within-county rank-IC ----
    inreg = regime >= 0
    print(f"\n[VERIFY] on the {inreg.sum()} regime-assigned wells:")

    # (1) where does the signal live? per-regime IC of the single best coordinate (KC-MISS thickness)
    thk = Xfull[:, iThk]
    for r in (0, 1):
        ics = within_ic(thk, y, blocks, regime == r)
        print(f"  per-regime baseline (thickness KC->MISS) IC | regime {r}: "
              f"{ics.mean():+.3f}  ({(ics>0).sum()}/{len(ics)} counties +)" if len(ics)
              else f"  regime {r}: too few qualifying counties")

    # (2) pooled value field
    v_pool = vg.fit_value_field(Xfull[inreg], y[inreg], [blocks[i] for i in np.where(inreg)[0]])
    vp = np.full(len(W), np.nan); vp[inreg] = v_pool
    ic_pool = within_ic(vp, y, blocks, inreg)

    # (3) routed: fit a separate value field inside each regime (per-regime LOCO), then combine
    vr = np.full(len(W), np.nan)
    for r in (0, 1):
        m = regime == r
        gi = np.where(m)[0]
        vv = vg.fit_value_field(Xfull[m], y[m], [blocks[i] for i in gi])
        vr[gi] = vv
    ic_route = within_ic(vr, y, blocks, inreg)

    # (4) routed-soft: regime label added as a feature to the pooled model
    Xr = np.c_[Xfull, regime.astype(float)]
    v_soft = vg.fit_value_field(Xr[inreg], y[inreg], [blocks[i] for i in np.where(inreg)[0]])
    vs = np.full(len(W), np.nan); vs[inreg] = v_soft
    ic_soft = within_ic(vs, y, blocks, inreg)

    print(f"  pooled value field            IC={ic_pool.mean():+.3f}  ({(ic_pool>0).sum()}/{len(ic_pool)} +)")
    print(f"  ROUTED (per-regime fields)    IC={ic_route.mean():+.3f}  ({(ic_route>0).sum()}/{len(ic_route)} +)")
    print(f"  routed-soft (regime feature)  IC={ic_soft.mean():+.3f}  ({(ic_soft>0).sum()}/{len(ic_soft)} +)")
    lift = max(ic_route.mean(), ic_soft.mean()) - ic_pool.mean()
    verdict = ("SHIP routing" if lift > 0.02 else "HOLD (routing adds no OOS lift)")
    print(f"  --> routing lift = {lift:+.3f}   VERDICT: {verdict}")


if __name__ == "__main__":
    main()
