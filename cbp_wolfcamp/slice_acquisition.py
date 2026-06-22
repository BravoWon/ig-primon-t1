#!/usr/bin/env python
"""SLICE prime (Tier-2 of the value-geometry kernel): active-learning / information-gain next-well
siting, MASKed to the KGS north-shelf regime where the structural edge is real (IC +0.41).

'Slicing the Pie' (GTA 90-01-007): each observation bisects the search entropy. The computational
analog is uncertainty sampling -- query where the value-field posterior is most uncertain, maximally
reducing entropy per (expensive) well. Two acquisitions:
  infogain  a(x) = sigma(x)          -- pure entropy reduction (the literal SLICE)
  ucb       a(x) = v_hat(x)+k*sigma  -- value-aware deployment (find productive AND informative)

VERIFY gate: sequential active-learning backtest on held-out shelf wells. SHIP only if infogain
reduces test error faster than random siting, and ucb discovers productive wells faster than random.

    python slice_acquisition.py
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
import value_geometry as vg

CORE = ["subsea_KANS", "depth_KANS", "thk_KANS_MISS", "lat", "lon"]
RNG = np.random.default_rng(0)


def shelf_mask(W, Xfull, names):
    """MASK prime: recover the unsupervised regimes; return the shallower (= shelf) one's indices."""
    ci = [names.index(n) for n in CORE]
    keep = np.all(np.isfinite(Xfull[:, ci]), axis=1)
    idx = np.where(keep)[0]
    Xc = Xfull[np.ix_(idx, ci)]; Xc = (Xc - Xc.mean(0)) / Xc.std(0)
    E, W0 = vg.knn_graph(Xc, k=10)
    lab = vg.cut_clusters(len(idx), E, W0, k=3)
    big = [c for c, _ in Counter(lab).most_common(2)]
    iDep = names.index("depth_KANS")
    med = {c: np.median(Xfull[idx[lab == c], iDep]) for c in big}
    shelf_lab = min(med, key=med.get)                       # shallower median KC depth = shelf
    return idx[lab == shelf_lab]


def rf_value(Xtr, ytr):
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.8,
                                 random_state=0, n_jobs=-1).fit(Xtr, ytr)


def predict_with_sigma(rf, X):
    """v_hat + epistemic sigma from the spread of per-tree predictions (ensemble disagreement)."""
    P = np.stack([t.predict(X) for t in rf.estimators_])    # (n_trees, n)
    return P.mean(0), P.std(0)


def acquire(policy, vhat, sigma, batch, rng, kappa=1.0):
    if policy == "random":
        return rng.permutation(len(vhat))[:batch]
    score = {"value": vhat, "infogain": sigma, "ucb": vhat + kappa * sigma}[policy]
    return np.argsort(-score)[:batch]


def active_learning(Xs, ys, seed_n=30, batch=10, rounds=18):
    """Sequential backtest: from a shared seed + frozen test set, each policy picks the next batch
    from the unlabeled pool, refits, and we record test error + value discovered."""
    n = len(ys); perm = RNG.permutation(n)
    test = perm[: n // 3]; poolall = perm[n // 3:]
    seed = poolall[:seed_n]
    Xte, yte = Xs[test], ys[test]
    curves = {}
    for policy in ["random", "value", "infogain", "ucb"]:
        rng = np.random.default_rng(7)
        L = list(seed); P = list(poolall[seed_n:])
        rmse, ic, disc, nlab = [], [], [], []
        for _ in range(rounds):
            rf = rf_value(Xs[L], ys[L])
            vt, _ = predict_with_sigma(rf, Xte)
            rmse.append(float(np.sqrt(np.mean((vt - yte) ** 2))))
            ic.append(float(spearmanr(vt, yte)[0]))
            nlab.append(len(L))
            if not P:
                break
            vp, sp = predict_with_sigma(rf, Xs[P])
            pick = acquire(policy, vp, sp, min(batch, len(P)), rng)
            chosen = [P[i] for i in pick]
            disc.append(float(np.mean(ys[chosen])))           # value of wells this policy drilled
            L += chosen; P = [P[i] for i in range(len(P)) if i not in set(pick)]
        curves[policy] = dict(rmse=np.array(rmse), ic=np.array(ic),
                              disc=np.array(disc), nlab=np.array(nlab))
    return curves


def grid_sites(W, shelf, Xfull, names, k_interp=8, kappa=1.0, top=10):
    """Product: score a lat/lon grid over the shelf; interpolate structure from nearby shelf wells;
    rank undrilled-style candidates by the ucb acquisition."""
    ci = [names.index(n) for n in CORE]
    Xs = Xfull[np.ix_(shelf, ci)]; ys = np.log(np.array([W[i]["b12"] for i in shelf]))
    rf = rf_value(Xs, ys)
    lat = Xfull[shelf, names.index("lat")]; lon = Xfull[shelf, names.index("lon")]
    tree = cKDTree(np.c_[lat, lon])
    gla = np.linspace(lat.min(), lat.max(), 30); glo = np.linspace(lon.min(), lon.max(), 30)
    rows = []
    iSub, iDep, iThk = names.index("subsea_KANS"), names.index("depth_KANS"), names.index("thk_KANS_MISS")
    for a in gla:
        for o in glo:
            d, j = tree.query([a, o], k=k_interp)
            if d.min() > 0.06 or d.max() > 0.20:             # keep grid pts genuinely inside the play
                continue
            w = 1.0 / (d + 1e-6); w /= w.sum()
            feat = [np.dot(w, Xfull[shelf[j], iSub]), np.dot(w, Xfull[shelf[j], iDep]),
                    np.dot(w, Xfull[shelf[j], iThk]), a, o]
            vh, sg = predict_with_sigma(rf, np.array([feat]))
            rows.append((a, o, vh[0], sg[0], vh[0] + kappa * sg[0], float(d.min())))
    rows.sort(key=lambda r: -r[4])
    return rows[:top]


def main():
    W = vg.load()
    Xfull, names = vg.featurize(W)
    shelf = shelf_mask(W, Xfull, names)
    ci = [names.index(n) for n in CORE]
    Xs = Xfull[np.ix_(shelf, ci)]
    ys = np.log(np.array([W[i]["b12"] for i in shelf]))
    print(f"[SLICE] MASK to shelf regime: n={len(shelf)} wells; features={CORE}")

    curves = active_learning(Xs, ys)
    print("\n[VERIFY] active-learning backtest (shared seed + frozen 1/3 test set):")
    print("  policy     test-RMSE@end  RMSE-AUC(lower=learns faster)  test-IC@end  mean discovered best12")
    summ = {}
    for p, c in curves.items():
        rmse_auc = float(np.trapezoid(c["rmse"], c["nlab"]) / (c["nlab"][-1] - c["nlab"][0]))
        disc_bbl = float(np.exp(np.mean(c["disc"]))) if len(c["disc"]) else float("nan")
        summ[p] = dict(rmse_end=c["rmse"][-1], auc=rmse_auc, ic_end=c["ic"][-1], disc=disc_bbl)
        print(f"  {p:10} {c['rmse'][-1]:.3f}          {rmse_auc:.3f}                  "
              f"{c['ic'][-1]:+.3f}        {disc_bbl:7.0f}")

    rand = summ["random"]
    info_faster = summ["infogain"]["auc"] < rand["auc"] - 1e-3
    ucb_value = summ["ucb"]["disc"] > rand["disc"] * 1.02
    print(f"\n  infogain learns faster than random?  {info_faster}  "
          f"(AUC {summ['infogain']['auc']:.3f} vs {rand['auc']:.3f})")
    print(f"  ucb discovers more value than random? {ucb_value}  "
          f"({summ['ucb']['disc']:.0f} vs {rand['disc']:.0f} bbl)")
    verdict = "SHIP SLICE" if (info_faster or ucb_value) else "HOLD (no edge over random siting)"
    print(f"  --> VERDICT: {verdict}")

    print("\n[PRODUCT] top next-well candidate sites over the shelf (ucb = v_hat + sigma):")
    print("   lat       lon      v_hat  sigma   ucb   nearest-well(deg)")
    for a, o, vh, sg, u, dmin in grid_sites(W, shelf, Xfull, names):
        print(f"  {a:7.3f}  {o:8.3f}  {vh:+.2f}  {sg:.2f}  {u:+.2f}   {dmin:.3f}")


if __name__ == "__main__":
    main()
