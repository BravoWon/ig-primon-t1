#!/usr/bin/env python
"""value_geometry -- a runnable instance of the IGVF value-geometry kernel.

Implements the semantic primes of `IGVF_VALUE_ALGORITHM.md` as composable functions and runs
the full pipeline on the KGS public oil data, generalizing the single-coordinate basin-position
run into the complete kernel:

  SELECT  -> features X (full per-well geometric signature), value y=log(best12), blocks=county
  VALUE   -> v_hat = E[y|X] via NaN-native gradient boosting, leave-one-county-out OUT-OF-FOLD
  SPACE   -> kNN graph on standardized core geometric features
  WARP    -> g_v = e^{-beta*v_hat} . g0   (value contracts the metric)
  PERSIST -> H0 persistent homology by union-find (exact); robust clusters = real regimes
  BOUNDARY-> ||grad v_hat|| on the graph; nodal/choke set; is it spatially coherent?
  FLOW    -> rank acreage by v_hat
  VERIFY  -> within-county LOCO rank-IC of v_hat vs y, compared to the single-coordinate baseline.
             THE GATE: ship only if the full field beats the baseline AND generalizes (IC>0).

    python value_geometry.py
"""
from __future__ import annotations
import csv, re
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, rankdata
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor

PROD = r"C:\Users\JT-DEV1\kgs_public\lease_prod.csv"
TOPS = r"C:\Users\JT-DEV1\kgs_public\tops\ks_tops.txt"
CKU = {"BARTON","ELLIS","RUSSELL","NESS","ROOKS","STAFFORD","BARBER","RICE","TREGO",
       "COMANCHE","KIOWA","PAWNEE","RUSH","ELLSWORTH","OSBORNE","LINCOLN"}
MARK = ["LANSING","KANSAS CITY","MARMATON","CHEROKEE","MISSISSIPPIAN","ARBUCKLE"]  # shallow->deep
PAIRS = [("LANSING","KANSAS CITY"),("KANSAS CITY","MARMATON"),("MARMATON","CHEROKEE"),
         ("CHEROKEE","MISSISSIPPIAN"),("MISSISSIPPIAN","ARBUCKLE"),("KANSAS CITY","MISSISSIPPIAN"),
         ("KANSAS CITY","ARBUCKLE"),("LANSING","MISSISSIPPIAN")]


def a10(s):
    d = re.sub(r"\D", "", s or "")
    return d[:10] if len(d) >= 10 else ""


# ---------------------------------------------------------------- PRIMES (general) ----
def fit_value_field(Xfull, y, blocks):
    """VALUE prime: v_hat=E[y|X] via NaN-native gradient boosting, leave-one-block-out OUT-OF-FOLD.
    Returns genuinely out-of-sample predictions for every row (no row sees its own block in train)."""
    v = np.full(len(y), np.nan)
    for c in sorted(set(blocks)):
        te = np.array([b == c for b in blocks]); tr = ~te
        if te.sum() < 1 or tr.sum() < 60:
            continue
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400,
                                          l2_regularization=1.0, min_samples_leaf=25,
                                          random_state=0)
        m.fit(Xfull[tr], y[tr])
        v[te] = m.predict(Xfull[te])
    return v


def knn_graph(Xc, k=10):
    """SPACE prime: kNN graph on standardized features -> undirected edges + base (euclidean) weights."""
    tree = cKDTree(Xc)
    d, idx = tree.query(Xc, k=k + 1)
    edges, w = {}, []
    for i in range(len(Xc)):
        for j, dist in zip(idx[i, 1:], d[i, 1:]):
            a, b = (i, int(j)) if i < j else (int(j), i)
            if a != b:
                edges[(a, b)] = dist
    E = np.array(list(edges.keys())); W = np.array(list(edges.values()))
    return E, W


def warp_weights(E, W, v, beta):
    """WARP prime: g_v = e^{-beta*v}.g0  -> contract distances in high-value regions."""
    vn = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    return W * np.exp(-beta * 0.5 * (vn[E[:, 0]] + vn[E[:, 1]]))


def h0_persistence(n, E, W):
    """PERSIST prime (H0): exact single-linkage union-find barcode. Returns merge ('death') scales
    sorted descending -- large gaps = robust clusters. parent[] gives the final cluster labels
    when cut at the (k-1) largest deaths."""
    order = np.argsort(W); parent = list(range(n)); size = [1] * n
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    deaths = []
    for e in order:
        a, b = find(int(E[e, 0])), find(int(E[e, 1]))
        if a != b:
            deaths.append(W[e])
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a; size[a] += size[b]
    return np.array(sorted(deaths, reverse=True)), parent, find


def cut_clusters(n, E, W, k):
    """Cut the single-linkage tree into k clusters (drop the k-1 largest merge edges)."""
    order = np.argsort(W); parent = list(range(n)); size = [1] * n
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    merges = []  # (weight, edge_idx)
    for e in order:
        a, b = find(int(E[e, 0])), find(int(E[e, 1]))
        if a != b:
            merges.append(W[e])
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a; size[a] += size[b]
    thresh = sorted(merges, reverse=True)[k - 1] if len(merges) >= k else -np.inf
    parent = list(range(n)); size = [1] * n
    for e in order:
        if W[e] >= thresh:
            continue
        a, b = find(int(E[e, 0])), find(int(E[e, 1]))
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a; size[a] += size[b]
    lab = np.array([find(i) for i in range(n)])
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def value_gradient(n, E, v):
    """BOUNDARY prime: ||grad v|| per node = mean |v_i - v_j| over graph neighbors."""
    acc = np.zeros(n); cnt = np.zeros(n)
    for e in range(len(E)):
        a, b = int(E[e, 0]), int(E[e, 1]); d = abs(v[a] - v[b])
        acc[a] += d; acc[b] += d; cnt[a] += 1; cnt[b] += 1
    return acc / np.maximum(cnt, 1)


def within_block_ic(pred, y, blocks, min_n=30):
    """VERIFY prime: within-block rank-IC of prediction vs truth (mean over blocks)."""
    out = []
    for c in sorted(set(blocks)):
        m = np.array([b == c for b in blocks]) & np.isfinite(pred)
        if m.sum() >= min_n:
            r = spearmanr(pred[m], y[m])[0]
            if np.isfinite(r):
                out.append((c, int(m.sum()), r))
    return out


def morans_i(lat, lon, x, R=0.07):
    """Spatial coherence of a node attribute (is the nodal set clustered or scattered?)."""
    XY = np.c_[lat, lon * np.cos(np.radians(np.mean(lat)))]
    tree = cKDTree(XY); z = (x - x.mean()) / (x.std() + 1e-9)
    num = den = 0.0; W = 0.0
    for i in range(len(x)):
        for j in tree.query_ball_point(XY[i], R):
            if j != i:
                num += z[i] * z[j]; W += 1
    den = (z * z).sum()
    return (len(x) / W) * (num / den) if W > 0 and den > 0 else np.nan


# ---------------------------------------------------------------- KGS demo ------------
def load():
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
        if b12 > 0:
            wells[a] = dict(b12=b12, cty=r["county"].upper())
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
        for m in MARK:
            if m in form:
                w.setdefault(m, top)
    return [w for w in wells.values() if "lat" in w and "elev" in w and "KANSAS CITY" in w]


def featurize(W):
    """Build the full ragged per-well geometric signature (NaN where a marker is absent)."""
    names, rows = [], []
    names += ["lat", "lon", "elev"]
    for m in MARK:
        names += [f"depth_{m[:4]}", f"subsea_{m[:4]}"]
    for a, b in PAIRS:
        names += [f"thk_{a[:4]}_{b[:4]}"]
    for w in W:
        row = [w["lat"], w["lon"], w["elev"]]
        for m in MARK:
            t = w.get(m, np.nan)
            row += [t, (w["elev"] - t) if np.isfinite(t) else np.nan]
        for a, b in PAIRS:
            ta, tb = w.get(a, np.nan), w.get(b, np.nan)
            row += [(tb - ta) if (np.isfinite(ta) and np.isfinite(tb)) else np.nan]
        rows.append(row)
    return np.array(rows), names


def main():
    W = load()
    y = np.log(np.array([w["b12"] for w in W]))
    blocks = [w["cty"] for w in W]
    lat = np.array([w["lat"] for w in W]); lon = np.array([w["lon"] for w in W])
    Xfull, names = featurize(W)
    print(f"[value_geometry on KGS]  n={len(W)} CKU single-well leases, {Xfull.shape[1]} geometric features")
    print(f"  feature coverage (non-NaN %): " +
          ", ".join(f"{n}={100*np.isfinite(Xfull[:,i]).mean():.0f}" for i, n in enumerate(names)
                    if n.startswith(("subsea_K", "thk_KANS_MISS", "thk_KANS_ARBU"))))

    # ---- VALUE: out-of-fold leave-one-county-out value field ----
    v_oos = fit_value_field(Xfull, y, blocks)
    full_ic = within_block_ic(v_oos, y, blocks)

    # ---- baselines: single geometric coordinates (no training) ----
    iK = names.index("subsea_KANS"); iT = names.index("thk_KANS_MISS"); iD = names.index("depth_KANS")
    base = {"structure(subsea KC)": Xfull[:, iK], "thickness(KC->MISS)": Xfull[:, iT],
            "depth(KC)": Xfull[:, iD]}
    print("\n[VERIFY gate]  within-county leave-one-out rank-IC (mean over counties, n>=30):")
    for nm, coord in base.items():
        ics = within_block_ic(coord, y, blocks)
        arr = np.array([r for *_, r in ics])
        print(f"  baseline {nm:22} IC={arr.mean():+.3f}  ({(arr>0).sum()}/{len(arr)} counties +)")
    arrF = np.array([r for *_, r in full_ic])
    print(f"  FULL value field v_hat      IC={arrF.mean():+.3f}  ({(arrF>0).sum()}/{len(arrF)} counties +)")
    best_base = max(np.array([r for *_, r in within_block_ic(c, y, blocks)]).mean() for c in base.values())
    lift = arrF.mean() - best_base
    verdict = "SHIP" if (arrF.mean() > 0 and lift > 0.02) else \
              "HOLD (no lift over baseline)" if arrF.mean() > 0 else "NO-SHIP (does not generalize)"
    print(f"  --> lift over best baseline = {lift:+.3f}   VERDICT: {verdict}")

    # ---- SPACE/WARP/PERSIST/BOUNDARY on the complete-core manifold ----
    core_names = ["subsea_KANS", "depth_KANS", "thk_KANS_MISS", "lat", "lon"]
    ci = [names.index(n) for n in core_names]
    keep = np.all(np.isfinite(Xfull[:, ci]), axis=1)
    Xc = Xfull[np.ix_(np.where(keep)[0], ci)]
    Xc = (Xc - Xc.mean(0)) / Xc.std(0)
    yk = y[keep]; latk = lat[keep]; lonk = lon[keep]
    vk = v_oos[keep]; vk = np.where(np.isfinite(vk), vk, np.nanmean(vk))
    n = len(Xc)
    E, W0 = knn_graph(Xc, k=10)
    Wv = warp_weights(E, W0, vk, beta=1.5)

    print(f"\n[PERSIST]  H0 single-linkage on n={n} (complete-core manifold)")
    for tag, Wx in [("unwarped g0", W0), ("value-warped g_v", Wv)]:
        deaths, _, _ = h0_persistence(n, E, Wx)
        lab = cut_clusters(n, E, Wx, k=3)
        means = [yk[lab == c].mean() for c in np.unique(lab) if (lab == c).sum() > 5]
        spread = (max(means) - min(means)) if len(means) > 1 else 0.0
        sizes = sorted([(lab == c).sum() for c in np.unique(lab)], reverse=True)[:3]
        print(f"  {tag:18} top merge-scales={np.round(deaths[:4],2)}  "
              f"3-cut sizes={sizes}  value-spread(log bbl)={spread:.2f}")

    print("\n[BOUNDARY]  nodal/choke set: top-decile ||grad v_hat|| on the graph")
    grad = value_gradient(n, E, vk)
    nodal = grad >= np.quantile(grad, 0.90)
    mi = morans_i(latk, lonk, nodal.astype(float))
    print(f"  nodal wells: {nodal.sum()}  Moran's I (spatial clustering)={mi:+.3f}  "
          f"({'spatially coherent boundary' if mi > 0.1 else 'scattered ~ noise'})")
    print(f"  mean log-best12: nodal={yk[nodal].mean():.2f} vs non-nodal={yk[~nodal].mean():.2f}")

    print("\n[FLOW]  top-5 high-graded wells by v_hat (county, predicted, actual best12 bbl):")
    rank = np.argsort(-vk)
    for i in rank[:5]:
        wi = [w for j, w in enumerate(W) if keep[j]][i]
        print(f"    {wi['cty']:9} v_hat={vk[i]:+.2f}  actual={int(np.exp(yk[i])):>6}")


if __name__ == "__main__":
    main()
