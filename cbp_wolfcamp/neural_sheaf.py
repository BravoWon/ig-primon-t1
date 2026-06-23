#!/usr/bin/env python
"""Neural Sheaf Diffusion (the real ATFT/JTopo construction) vs the hand-designed linear sheaf.

Learned restriction maps: for each edge the relative transport R_ij = expm(skew(MLP(x_i,x_j))) in
SO(d) (a shared MLP -> regularized, not free per-edge params). Sheaf-Laplacian diffusion layers
align neighbor stalks through R before aggregating; a readout maps the stalk to value. Trained on the
known producers, predicts held-out wells. Same gate as everything else: geology-only manifold,
buffered spatial holdout, provenance-clean pre-drill labels. Compared on the SAME folds to the linear
sheaf (+0.314 @16-fold) and manifold-kNN (+0.266).

Question: do LEARNED restriction maps climb past the fixed-map linear sheaf, or is the geology-signal
ceiling already reached?

    python neural_sheaf.py
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import topological_predictor as tp
import sheaf_value_model as sv

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = 4            # stalk dimension
LAYERS = 3
HID = 32
EPOCHS = 300
GRID = 2         # 2x2 = 4 buffered spatial folds (each fold trains one net)


def skew(p, d):
    """p: (E, d(d-1)/2) -> (E,d,d) skew-symmetric."""
    E = p.shape[0]
    A = torch.zeros(E, d, d, device=p.device)
    iu = torch.triu_indices(d, d, 1)
    A[:, iu[0], iu[1]] = p
    return A - A.transpose(1, 2)


class NeuralSheaf(nn.Module):
    def __init__(self, nfeat, d=D, hid=HID, layers=LAYERS):
        super().__init__()
        self.d = d
        self.enc = nn.Linear(nfeat, d)
        self.rmlp = nn.Sequential(nn.Linear(2 * nfeat, hid), nn.SiLU(),
                                  nn.Linear(hid, d * (d - 1) // 2))
        self.W = nn.ParameterList([nn.Parameter(torch.eye(d) + 0.01 * torch.randn(d, d))
                                   for _ in range(layers)])
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))

    def forward(self, X, ei, ej, deg):
        R = torch.matrix_exp(skew(self.rmlp(torch.cat([X[ei], X[ej]], 1)), self.d))  # (E,d,d) SO(d)
        H = self.enc(X)
        for W in self.W:
            Hi, Hj = H[ei], H[ej]
            mj = Hj - torch.bmm(R, Hi.unsqueeze(-1)).squeeze(-1)        # into j: h_j - R h_i
            mi = Hi - torch.bmm(R.transpose(1, 2), Hj.unsqueeze(-1)).squeeze(-1)  # into i: h_i - R^T h_j
            m = torch.zeros_like(H).index_add_(0, ej, mj).index_add_(0, ei, mi)
            H = H - self.alpha * m / deg.clamp(min=1).unsqueeze(1)      # sheaf-Laplacian diffusion
            H = H + torch.tanh(H @ W)                                   # channel mix + residual
        return self.head(H).squeeze(-1)


def train_predict(Xg, ei, ej, deg, y, known, n, prior):
    """Anchored like the linear sheaf: the net REFINES the kNN prior (residual target + prior as an
    input feature), so it can't collapse in the held-out holes -- a fair test of learned-vs-fixed maps."""
    prn = (prior - prior.mean()) / (prior.std() + 1e-6)
    X = torch.cat([Xg, prn.unsqueeze(1)], 1)
    resid = y - prior
    rk = resid[known]; mu, sd = rk.mean(), rk.std() + 1e-6
    rt = (resid - mu) / sd
    net = NeuralSheaf(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = nn.MSELoss()
    net.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = lossf(net(X, ei, ej, deg)[known], rt[known])
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        corr = net(X, ei, ej, deg) * sd + mu                 # learned residual
    return (prior + corr).cpu().numpy()                      # anchor + refinement


def main():
    W = tp.load()
    Xn, names = tp.featurize(W)
    y = np.log(np.array([w["b12"] for w in W]))
    n = len(W)
    lat = Xn[:, names.index("lat")]; lon = Xn[:, names.index("lon")]
    Xs = sv.standardize_impute(Xn)
    gidx = [j for j, nm in enumerate(names) if nm not in ("lat", "lon")]
    Xg = Xs[:, gidx]                                   # geology-only manifold features
    print(f"[neural sheaf]  n={n}, {Xg.shape[1]} geology features, stalk d={D}, "
          f"{LAYERS} layers, device={DEV}")

    # geology-only kNN cell complex
    _, idx = cKDTree(Xg).query(Xg, k=sv.KNN + 1)
    eset = sorted({(min(i, int(j)), max(i, int(j))) for i in range(n) for j in idx[i, 1:] if i != int(j)})
    E = np.array(eset)
    deg = np.zeros(n)
    for a, b in E:
        deg[a] += 1; deg[b] += 1
    Xt = torch.tensor(Xg, dtype=torch.float32, device=DEV)
    ei = torch.tensor(E[:, 0], device=DEV); ej = torch.tensor(E[:, 1], device=DEV)
    degt = torch.tensor(deg, dtype=torch.float32, device=DEV)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)

    # linear-sheaf pieces (same geology PCs) for the matched baseline
    Gc = Xg - Xg.mean(0); _, _, Vt = np.linalg.svd(Gc, full_matrices=False)
    g = Gc @ Vt[:sv.K].T; g = g / (g.std(0) + 1e-9); phi = np.c_[np.ones(n), g]

    # buffered spatial folds (2x2), buffer 0.15deg
    la0, la1, lo0, lo1 = lat.min(), lat.max(), lon.min(), lon.max()
    bi = np.minimum(((lat - la0) / (la1 - la0 + 1e-9) * GRID).astype(int), GRID - 1)
    bj = np.minimum(((lon - lo0) / (lo1 - lo0 + 1e-9) * GRID).astype(int), GRID - 1)
    block = bi * GRID + bj
    geo = np.c_[lat, lon * np.cos(np.radians(lat.mean()))]

    p_neu = np.full(n, np.nan); p_lin = np.full(n, np.nan)
    p_knn = np.full(n, np.nan); p_coord = Xs[:, names.index("subsea_KANS")]
    for b in sorted(set(block.tolist())):
        te = block == b
        if te.sum() < 30:
            continue
        dmin, _ = cKDTree(geo[te]).query(geo, k=1)
        tr = (~te) & (dmin > sv.BUF)
        known = np.where(tr)[0]
        kt = cKDTree(Xg[known]); _, jall = kt.query(Xg, k=min(sv.KNN, len(known)))
        prior = np.median(y[known][jall], axis=1)
        p_knn[np.where(te)[0]] = prior[te]
        p_lin[np.where(te)[0]] = sv.solve_sheaf(phi, eset, known, y, n, prior, sv.RHO)[te]
        kt_known = torch.tensor(known, device=DEV)
        priort = torch.tensor(prior, dtype=torch.float32, device=DEV)
        pred = train_predict(Xt, ei, ej, degt, yt, kt_known, n, priort)
        p_neu[np.where(te)[0]] = pred[te]
        print(f"   fold block {b}: trained on {len(known)} known, scored {int(te.sum())} held-out")

    m = np.isfinite(p_neu) & np.isfinite(p_lin)
    print(f"\n  buffered spatial {GRID}x{GRID} holdout, scored {m.sum()} held-out, GLOBAL rank-IC:")
    for label, p in [("single coordinate", p_coord), ("manifold-kNN", p_knn),
                     ("LINEAR sheaf (fixed maps)", p_lin), ("NEURAL sheaf (learned SO(d))", p_neu)]:
        print(f"    {label:32} IC = {spearmanr(p[m], y[m])[0]:+.3f}")
    ic_lin = spearmanr(p_lin[m], y[m])[0]; ic_neu = spearmanr(p_neu[m], y[m])[0]
    print(f"\n  VERDICT: neural vs linear sheaf {ic_neu - ic_lin:+.3f}  "
          f"-> {'LEARNED maps climb higher' if ic_neu > ic_lin + 0.01 else 'no gain -- geology-signal ceiling reached (linear sheaf already captures it)'}")


if __name__ == "__main__":
    main()
