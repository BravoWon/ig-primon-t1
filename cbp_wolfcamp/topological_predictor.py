#!/usr/bin/env python
"""Test the hypothesis: a strong public-data predictor EXISTS if you (a) dimension by every
available label and (b) use a discrete-topological INVERSE construction -- build the solution space
from the known knowns (producing wells) and work backward -- instead of forward regression.

Method (faithful to the claim):
  MAXIMAL LABELS   every label in lease_prod + ks_tops: all marker subsea + pairwise isopachs,
                   decline shape, vintage, zone, well dynamics, structural curvature, AND the
                   spatial-neighbor productivity of the known producers.
  TOPOLOGICAL      manifold (kNN graph) over the full label space; persistent-H0 regime structure;
                   the predictor for a new well is CONSTRUCTED from its nearest KNOWN producers on
                   the manifold (manifold-kNN = the discrete 1-skeleton / 'work backward from knowns').
  VERDICT          the crux is IN-SAMPLE vs OUT-OF-SAMPLE. A predictor that is strong in-sample but
                   collapses under leave-one-county-out / spatial-block holdout was spatial leakage,
                   not signal. Compared against: single coordinate, pure spatial-neighbor, forward GBM.

    python topological_predictor.py
"""
from __future__ import annotations
import csv, re
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor

PROD = r"C:\Users\JT-DEV1\kgs_public\lease_prod.csv"
TOPS = r"C:\Users\JT-DEV1\kgs_public\tops\ks_tops.txt"
CKU = {"BARTON","ELLIS","RUSSELL","NESS","ROOKS","STAFFORD","BARBER","RICE","TREGO",
       "COMANCHE","KIOWA","PAWNEE","RUSH","ELLSWORTH","OSBORNE","LINCOLN"}
MARK = ["LANSING","KANSAS CITY","MARMATON","CHEROKEE","MISSISSIPPIAN","ARBUCKLE"]
PAIRS = [("LANSING","KANSAS CITY"),("KANSAS CITY","MARMATON"),("KANSAS CITY","MISSISSIPPIAN"),
         ("MISSISSIPPIAN","ARBUCKLE"),("KANSAS CITY","ARBUCKLE"),("LANSING","MISSISSIPPIAN")]


def a10(s):
    d = re.sub(r"\D", "", s or "")
    return d[:10] if len(d) >= 10 else ""


def load():
    W = {}
    for r in csv.DictReader(open(PROD, encoding="utf-8")):
        if r["max_wells"] != "1" or r["county"].upper() not in CKU:
            continue
        a = next((a10(t) for t in r["apis"].split(",") if a10(t)), "")
        if not a:
            continue
        try:
            b12 = float(r["best12_oil"] or 0)
            if b12 <= 0:
                continue
            f12 = float(r["first12_oil"] or 0); pk = float(r["peak_oil"] or 0)
            cum = float(r["cum_oil"] or 0); nm = float(r["n_months"] or 0)
            la, lo = float(r["lat"]), float(r["lon"])
        except (ValueError, KeyError):
            continue
        ym = re.sub(r"\D", "", r.get("first_ym") or "")
        W[a] = dict(b12=b12, f12=f12, pk=pk, cum=cum, nm=nm, la=la, lo=lo,
                    cty=r["county"].upper(), zone=(r.get("zone") or "").strip(),
                    vint=float(ym[:4]) if len(ym) >= 4 else np.nan)
    T = set(W)
    for r in csv.DictReader(open(TOPS, encoding="latin-1")):
        a = a10(r.get("API_NUM_NODASH") or r.get("API_NUMBER"))
        if a not in T:
            continue
        form = (r.get("FORMATION", "") + " " + r.get("OLD_FORMATION", "")).upper()
        try:
            top = float(r["TOP"])
        except (ValueError, KeyError):
            continue
        w = W[a]
        try:
            w.setdefault("elev", float(r["ELEVATION"]))
        except (ValueError, KeyError):
            pass
        for m in MARK:
            if m in form:
                w.setdefault(m, top)
    return [w for w in W.values() if "elev" in w]


def featurize(W):
    # PRE-DRILL features ONLY: geology mappable from offset control + location + vintage.
    # NO production outcomes (first12/peak/cum/n_months or any ratio of them) -- those are
    # downstream of the target (best12) and leak it. This is the legitimate siting-decision space.
    names, X = [], []
    names += ["elev", "depth_KC", "lat", "lon", "vint", "n_tops"]
    for m in MARK:
        names.append("subsea_" + m[:4])
    for a, b in PAIRS:
        names.append("thk_%s_%s" % (a[:4], b[:4]))
    for w in W:
        kc = w.get("KANSAS CITY", np.nan)
        row = [w["elev"], kc, w["la"], w["lo"], w["vint"], sum(1 for m in MARK if m in w)]
        for m in MARK:
            row.append(w["elev"] - w[m] if m in w else np.nan)
        for a, b in PAIRS:
            row.append(w[b] - w[a] if (a in w and b in w) else np.nan)
        X.append(row)
    return np.array(X, float), names


def within_county_ic(pred, y, cty, mask):
    out = []
    for c in sorted(set(cty)):
        sel = np.array([cty[i] == c and mask[i] and np.isfinite(pred[i]) for i in range(len(y))])
        if sel.sum() >= 25:
            r = spearmanr(pred[sel], y[sel])[0]
            if np.isfinite(r):
                out.append(r)
    return np.array(out)


def main():
    W = load()
    X, names = featurize(W)
    y = np.log(np.array([w["b12"] for w in W]))
    cty = [w["cty"] for w in W]
    lat = X[:, names.index("lat")]; lon = X[:, names.index("lon")]
    n = len(W)
    print(f"[topological predictor]  n={n} CKU single-well leases, {X.shape[1]} PRE-DRILL labels "
          f"(geology/structure/isopachs + location + vintage; NO production outcomes)")

    # standardized, median-imputed copy for the manifold / kNN
    Xs = X.copy()
    for j in range(Xs.shape[1]):
        col = Xs[:, j]; med = np.nanmedian(col)
        col[~np.isfinite(col)] = med
        s = np.std(col); Xs[:, j] = (col - med) / s if s > 0 else 0.0
    GEO = [names.index(k) for k in ["lat", "lon"]]
    NONGEO = [j for j in range(Xs.shape[1]) if j not in GEO]

    # persistent-H0 structure of the full-label manifold (how many robust regimes?)
    ftree = cKDTree(Xs)
    d, idx = ftree.query(Xs, k=8)
    edges = sorted({(min(i, int(j)), max(i, int(j))): dd
                    for i in range(n) for j, dd in zip(idx[i, 1:], d[i, 1:])}.items(),
                   key=lambda kv: kv[1])
    parent = list(range(n)); sz = [1] * n
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    deaths = []
    for (a, b), w_ in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            deaths.append(w_); parent[rb] = ra; sz[ra] += sz[rb]
    print(f"  manifold H0: largest merge-scale gaps {np.round(np.sort(deaths)[-4:][::-1],2)} "
          f"-> a few robust regimes (multi-modal solution space)")

    def predict(scheme, kind):
        """scheme: 'insample'|'loco'|'spatial'. kind: 'coord'|'spatial'|'gbm'|'manifold'."""
        pred = np.full(n, np.nan)
        # define training-pool membership per test well
        if scheme == "loco":
            folds = {c: np.array([cty[i] == c for i in range(n)]) for c in set(cty)}
            groups = list(folds.values())
        elif scheme == "spatial":
            cell = np.array([f"{int(lat[i]/0.4)}_{int(lon[i]/0.4)}" for i in range(n)])
            groups = [cell == c for c in set(cell.tolist())]
        else:  # insample = leave-one-out (no block); pool = all others
            groups = None
        coordj = names.index("subsea_KANS")

        def fit_pool(te):
            tr = ~te
            if kind == "coord":
                pred[te] = Xs[te, coordj]                      # fixed coordinate, no fit
            elif kind == "spatial":
                pool = np.where(tr)[0]
                t = cKDTree(np.c_[lat[pool], lon[pool] * np.cos(np.radians(lat.mean()))])
                q = np.c_[lat[te], lon[te] * np.cos(np.radians(lat.mean()))]
                _, jj = t.query(q, k=min(8, len(pool)))
                pred[np.where(te)[0]] = np.median(y[pool][jj], axis=1)
            elif kind == "gbm":
                m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                                  l2_regularization=1.0, min_samples_leaf=25,
                                                  random_state=0).fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
            elif kind == "manifold":   # work backward from known producers in FULL label space
                pool = np.where(tr)[0]
                t = cKDTree(Xs[np.ix_(pool, NONGEO)])
                _, jj = t.query(Xs[np.ix_(np.where(te)[0], NONGEO)], k=min(12, len(pool)))
                pred[np.where(te)[0]] = np.median(y[pool][jj], axis=1)

        if groups is None:                                     # insample LOO via 5 random shards
            order = np.argsort([f"{cty[i]}{i}" for i in range(n)])
            for s in range(5):
                te = np.zeros(n, bool); te[order[s::5]] = True
                fit_pool(te)
        else:
            for te in groups:
                if te.sum() and (~te).sum() > 50:
                    fit_pool(te)
        return pred

    print("\n  predictor                         in-sample   LOCO(county)   spatial-block")
    allmask = np.ones(n, bool)
    rows = [("single coordinate (subsea KC)", "coord"),
            ("spatial-neighbor (known knowns)", "spatial"),
            ("forward GBM (all labels)", "gbm"),
            ("TOPOLOGICAL manifold-kNN (inverse)", "manifold")]
    res = {}
    for label, kind in rows:
        ics = {}
        for scheme in ["insample", "loco", "spatial"]:
            p = predict(scheme, kind)
            ics[scheme] = within_county_ic(p, y, cty, allmask).mean()
        res[label] = ics
        print(f"  {label:34} {ics['insample']:+.3f}      {ics['loco']:+.3f}        {ics['spatial']:+.3f}")

    base = res["single coordinate (subsea KC)"]["loco"]
    topo = res["TOPOLOGICAL manifold-kNN (inverse)"]
    print(f"\n  VERDICT (out-of-sample is what counts):")
    print(f"    single-coordinate LOCO baseline      = {base:+.3f}")
    print(f"    topological manifold-kNN LOCO        = {topo['loco']:+.3f}  "
          f"(in-sample {topo['insample']:+.3f} -> gap {topo['insample']-topo['loco']:+.3f})")
    print(f"    spatial-block (fairer to proximity)  = {topo['spatial']:+.3f}")
    win = topo['loco'] > base + 0.03 or topo['spatial'] > base + 0.05
    leak = topo['insample'] - max(topo['loco'], topo['spatial']) > 0.15
    print(f"    -> full-label topological beats one coordinate OOS? {win}")
    print(f"    -> large in-sample->OOS collapse (spatial leakage)? {leak}")


if __name__ == "__main__":
    main()
