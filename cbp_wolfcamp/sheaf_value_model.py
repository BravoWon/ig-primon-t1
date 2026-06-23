#!/usr/bin/env python
"""Cellular-sheaf value model (drop the county frame; think geometric modeler).

The well set is a CELL COMPLEX: wells = 0-cells; edges (1-cells) connect geologically-similar wells
(kNN in pre-drill feature space). On it we put the SHEAF OF LOCAL AFFINE VALUE-MODELS:
  - vertex stalk F(v) = R^{K+1}: coefficients theta_v=(b_v, w_v) of a local map  value ~= b_v + w_v.g
    where g = low-dim geology coordinate (K PCs of the pre-drill geology, NO location, NO outcomes).
  - RESTRICTION MAP on edge (u,v): evaluate the local model at the shared geology g_uv -> a scalar.
    Sheaf consistency (delta x = 0) demands the two neighboring local models AGREE there:
        b_u + w_u.g_uv  ==  b_v + w_v.g_uv.
  - sheaf Laplacian L = delta^T delta; sheaf Dirichlet energy x^T L x = sum_edges (model_u - model_v)^2.
VALUE is recovered as the HARMONIC EXTENSION / global section: minimize the sheaf energy + a data term
that pins the KNOWN producers' values. A held-out well's local model is fixed by its neighbors through
the restriction maps -> 'work backward from the known knowns'. The non-trivial restriction maps let the
geology->value GRADIENT vary across the basin (anisotropy/heterogeneity) -- the thing flat regression
and value-only smoothing (the trivial sheaf = plain graph diffusion / manifold-kNN) cannot capture.

Honest gate: buffered SPATIAL holdout (4x4 geographic blocks, 0.15deg buffer), global rank-IC,
identical CV for every predictor. Win = sheaf beats manifold-kNN (trivial sheaf) AND single coordinate.

    python sheaf_value_model.py
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
from scipy.stats import spearmanr
import topological_predictor as tp

K = 4          # geology PCs carried in each stalk's gradient
KNN = 10       # cell-complex degree
LAM = 1.0      # data-term weight
MU = 1e-2      # ridge on stalk coefficients
BUF = 0.15     # spatial buffer (deg) between train and test
GRID = 4       # NxN spatial holdout blocks
RHO = 0.5      # anchor weight: pull each stalk's intercept toward the manifold-kNN prior


def standardize_impute(X):
    Xs = X.astype(float).copy()
    for j in range(Xs.shape[1]):
        col = Xs[:, j]; med = np.nanmedian(col)
        col[~np.isfinite(col)] = med
        s = np.std(col); Xs[:, j] = (col - med) / s if s > 0 else 0.0
    return Xs


def solve_sheaf(phi, edges, known_idx, y, n, prior, rho):
    """Anchored harmonic extension: argmin sum_edges ||theta_u.phi_e - theta_v.phi_e||^2
       + LAM sum_known (theta_v.phi_v - y_v)^2 + rho sum_v (b_v - prior_v)^2
       + MU_b sum b_v^2 + MU_w sum||w_v||^2.
    The anchor (prior = manifold-kNN field) keeps every vertex bounded; the restriction-map
    GRADIENT (w_v) only REFINES it. Returns per-vertex value f_v(g_v)."""
    d = phi.shape[1]                                  # stalk dim K+1 (dim 0 = intercept b_v)
    MU_b, MU_w = 1e-3, 1.0                            # ridge: light on intercept, HARD on gradient
    rows, cols, vals = [], [], []

    def add_block(a, b, M):
        ra = a * d; rb = b * d
        for i in range(d):
            for j in range(d):
                if M[i, j] != 0.0:
                    rows.append(ra + i); cols.append(rb + j); vals.append(M[i, j])

    for (u, v) in edges:                              # restriction-map consistency at shared geology
        pe = 0.5 * (phi[u] + phi[v]); P = np.outer(pe, pe)
        add_block(u, u, P); add_block(v, v, P)
        add_block(u, v, -P); add_block(v, u, -P)
    bvec = np.zeros(n * d)
    for t in known_idx:                               # data term pins known producers
        P = LAM * np.outer(phi[t], phi[t]); add_block(t, t, P)
        bvec[t * d:(t + 1) * d] += LAM * y[t] * phi[t]
    # per-vertex anchor of the intercept to the kNN prior + split ridge
    diag = np.zeros(n * d)
    diag[0::d] = rho + MU_b                            # intercept rows
    for k in range(1, d):
        diag[k::d] = MU_w                             # gradient rows
    A = coo_matrix((vals, (rows, cols)), shape=(n * d, n * d)).tocsr()
    A = A + coo_matrix((diag, (np.arange(n * d), np.arange(n * d))), shape=(n * d, n * d)).tocsr()
    bvec[0::d] += rho * prior
    theta, _ = cg(A, bvec, rtol=1e-6, maxiter=3000)
    return np.einsum("nd,nd->n", theta.reshape(n, d), phi)


def main():
    W = tp.load()
    X, names = tp.featurize(W)                        # PRE-DRILL labels only (provenance-clean)
    y = np.log(np.array([w["b12"] for w in W]))
    n = len(W)
    lat = X[:, names.index("lat")]; lon = X[:, names.index("lon")]
    Xs = standardize_impute(X)
    print(f"[sheaf value model]  n={n} wells, {X.shape[1]} pre-drill labels; "
          f"stalk dim {K+1}, kNN cell complex degree {KNN}")

    # geology coordinate g = K PCs of NON-location pre-drill features
    gidx = [j for j, nm in enumerate(names) if nm not in ("lat", "lon")]
    Gc = Xs[:, gidx] - Xs[:, gidx].mean(0)
    _, _, Vt = np.linalg.svd(Gc, full_matrices=False)
    g = Gc @ Vt[:K].T; g = g / (g.std(0) + 1e-9)
    phi = np.c_[np.ones(n), g]                        # [1, g1..gK]  (restriction-map basis)

    # GEOLOGY-ONLY manifold: features = geology (what predicts); geography (lat/lon) is used ONLY
    # to define the honest spatial holdout below, never as a manifold feature. So "manifold-kNN" is
    # genuinely geology-kNN, and any sheaf gain reflects geological structure, not spatial proximity.
    Xg = Xs[:, gidx]
    tree = cKDTree(Xg); _, idx = tree.query(Xg, k=KNN + 1)
    edges = sorted({(min(i, int(j)), max(i, int(j)))
                    for i in range(n) for j in idx[i, 1:] if i != int(j)})

    # buffered spatial holdout blocks (NOT counties)
    la0, la1 = lat.min(), lat.max(); lo0, lo1 = lon.min(), lon.max()
    bi = np.minimum(((lat - la0) / (la1 - la0 + 1e-9) * GRID).astype(int), GRID - 1)
    bj = np.minimum(((lon - lo0) / (lo1 - lo0 + 1e-9) * GRID).astype(int), GRID - 1)
    block = bi * GRID + bj
    geo = np.c_[lat, lon * np.cos(np.radians(lat.mean()))]
    coordj = names.index("subsea_KANS")

    p_sheaf = np.full(n, np.nan); p_knn = np.full(n, np.nan); p_coord = Xs[:, coordj]
    for b in sorted(set(block.tolist())):
        te = block == b
        if te.sum() < 15:
            continue
        tt = cKDTree(geo[te])                          # buffer: drop train within BUF of any test well
        dmin, _ = tt.query(geo, k=1)
        tr = (~te) & (dmin > BUF)
        if tr.sum() < 200:
            continue
        known = np.where(tr)[0]
        # geology-only manifold-kNN field (trivial sheaf) -> baseline AND sheaf anchor/prior
        kt = cKDTree(Xg[known]); _, jall = kt.query(Xg, k=min(KNN, len(known)))
        prior = np.median(y[known][jall], axis=1)
        p_knn[np.where(te)[0]] = prior[te]
        # cellular sheaf REFINES the kNN prior through the restriction maps
        p_sheaf[np.where(te)[0]] = solve_sheaf(phi, edges, known, y, n, prior, RHO)[te]

    m = np.isfinite(p_sheaf) & np.isfinite(p_knn)
    print(f"\n  buffered spatial-block holdout: scored on {m.sum()} held-out wells "
          f"({GRID}x{GRID} blocks, {BUF}deg buffer), GLOBAL rank-IC:")
    ic_coord = spearmanr(p_coord[m], y[m])[0]
    ic_knn = spearmanr(p_knn[m], y[m])[0]
    ic_sheaf = spearmanr(p_sheaf[m], y[m])[0]
    print(f"    single coordinate (subsea KC)      IC = {ic_coord:+.3f}")
    print(f"    manifold-kNN  (TRIVIAL sheaf)       IC = {ic_knn:+.3f}")
    print(f"    CELLULAR SHEAF (restriction maps)   IC = {ic_sheaf:+.3f}")
    print(f"\n  VERDICT:")
    print(f"    sheaf beats manifold-kNN?   {ic_sheaf > ic_knn + 0.01}  "
          f"(delta {ic_sheaf - ic_knn:+.3f})")
    print(f"    sheaf beats single coord?   {ic_sheaf > ic_coord + 0.02}  "
          f"(delta {ic_sheaf - ic_coord:+.3f})")
    # decile lift: do top-decile sheaf-ranked held-out wells out-produce the rest?
    top = p_sheaf[m] >= np.quantile(p_sheaf[m], 0.9)
    print(f"    top-decile sheaf pick median best12 = {np.exp(np.median(y[m][top])):.0f} bbl "
          f"vs rest {np.exp(np.median(y[m][~top])):.0f} bbl "
          f"({np.exp(np.median(y[m][top]) - np.median(y[m][~top])):.2f}x)")


if __name__ == "__main__":
    main()
