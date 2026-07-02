#!/usr/bin/env python
"""NODE 6 -- the two-estimator lambda agreement. The spec's open seam, executed.

Stochastic route: OUR SGLD local learning coefficient (the ladder's instrument, determinism-locked,
stability-gated cells). Discrete route: the EXACT Aoyagi-Watanabe (2005) closed-form RLCT for reduced-
rank regression y = BAx (input M, hidden H, output N, true rank r) -- the polytope/resolution result.

Six configs span all three AW regimes + both parities + a gauge anchor. Per config:
  theory     : lambda_AW(M,H,N,r) (exact, case-based)
  lambda-hat @ DEGENERATE point : A=[V;0], B=[U,0] zero-padded rank-r factorization -- the most singular
               stratum, where the local RLCT equals the global AW lambda. THE PRE-REGISTERED TEST.
  lambda-hat @ TRAINED point    : Adam-trained to the zero-loss variety (exploratory: does SGD land on
               the degenerate stratum, or a generic smoother one with larger local lambda?)

PRE-REGISTERED (spec v0.1 decision rule): agreement iff |lambda-hat - lambda_AW| / lambda_AW <= 0.10 at
the degenerate point, per config. All/most agree -> the discrete handle on the learning coefficient is
validated against the stochastic estimator (the spec's 'genuinely valuable' branch). Divergence -> its
location (which regime/config) is the finding. Either branch pays.

    python slt/node6_two_estimator.py
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import math, random
import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_DATA = 2000
EPS_GRID = [1e-6, 3e-6, 1e-5, 3e-5]
GAMMA_GRID = [30.0, 100.0]
CHAINS, DRAWS, BURNIN, DRIFT_TOL = 4, 200, 80, 0.15
CONFIGS = [  # (M, H, N, r)  spanning the AW regimes
    (3, 1, 3, 1),   # gauge anchor (H=r): lambda=(dim-gauge)/2=2.5, regime 1 even
    (3, 2, 3, 1),   # regime 1, odd parity     -> 3.5
    (4, 3, 4, 1),   # regime 1, even parity    -> 6.0
    (4, 3, 4, 2),   # regime 1, odd parity     -> 7.0
    (3, 8, 3, 1),   # regime 4 (M+N < H+r): overparam hidden -> MN/2 = 4.5
    (2, 2, 8, 2),   # regime 2 (M+H < N+r)     -> (HM-Hr+Nr)/2 = 8.0
]


def aw_lambda(M, H, N, r):
    """Aoyagi-Watanabe 2005: exact RLCT of reduced-rank regression."""
    if M + H < N + r:
        return (H * M - H * r + N * r) / 2
    if N + H < M + r:
        return (H * N - H * r + M * r) / 2
    if M + N < H + r:
        return M * N / 2
    s = M + H + N + r
    lam = (2 * (H + r) * (M + N) - (M - N) ** 2 - (H + r) ** 2)
    return (lam + (s % 2)) / 8


def set_det(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class RRR(nn.Module):
    def __init__(self, M, H, N):
        super().__init__()
        self.A = nn.Parameter(torch.randn(H, M) * 0.3)
        self.B = nn.Parameter(torch.randn(N, H) * 0.3)

    def forward(self, x):
        return x @ self.A.T @ self.B.T


def nll(model, x, y):
    return 0.5 * ((model(x) - y) ** 2).sum(-1).mean()          # unit-Gaussian NLL per sample (mean over n)


def sgld_llc(model, w_star, x, y):
    """Stability-gated SGLD lambda-hat: sweep (eps,gamma), keep stationary non-divergent cells."""
    n = len(x); beta = 1.0 / math.log(n); keys = list(w_star)
    with torch.no_grad():
        model.load_state_dict(w_star); L_star = nll(model, x, y).item()
    vals = []
    for gamma in GAMMA_GRID:
        for eps in EPS_GRID:
            lams, drifts, div = [], [], 0
            for c in range(CHAINS):
                model.load_state_dict(w_star); torch.manual_seed(7000 + c)
                tr = []
                for t in range(BURNIN + DRAWS):
                    L = nll(model, x, y); model.zero_grad(); L.backward()
                    with torch.no_grad():
                        for kk, prm in zip(keys, model.parameters()):
                            prm.add_(-0.5 * eps * (n * beta * prm.grad + gamma * (prm - w_star[kk]))
                                     + math.sqrt(eps) * torch.randn_like(prm))
                    if t >= BURNIN:
                        with torch.no_grad():
                            tr.append(nll(model, x, y).item())
                tr = np.array(tr)
                if not np.all(np.isfinite(tr)) or tr.mean() > 5 * L_star + 10:
                    div += 1; continue
                lams.append(n * beta * (tr.mean() - L_star))
                h = len(tr) // 2
                drifts.append(abs(tr[h:].mean() - tr[:h].mean()) / (tr.mean() + 1e-12))
            if lams and div == 0 and np.mean(drifts) < DRIFT_TOL:
                vals.append(np.mean(lams))
    model.load_state_dict(w_star)
    if not vals:
        return float("nan"), 0
    return float(np.median(vals)), len(vals)


def run_config(M, H, N, r, seed=0):
    set_det(seed)
    U = torch.randn(N, r, device=DEV) / math.sqrt(r)
    V = torch.randn(r, M, device=DEV) / math.sqrt(M)
    x = torch.randn(N_DATA, M, device=DEV)
    y = x @ V.T @ U.T                                          # exact rank-r truth, noiseless
    # --- degenerate point: zero-padded factorization (most singular stratum; theory's home) ---
    model = RRR(M, H, N).to(DEV)
    with torch.no_grad():
        model.A.zero_(); model.B.zero_()
        model.A[:r] = V; model.B[:, :r] = U
    w_deg = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        L_deg = nll(model, x, y).item()
    lam_deg, cells_d = sgld_llc(model, w_deg, x, y)
    # --- trained point: Adam from random init to the zero-loss variety ---
    set_det(seed)
    model2 = RRR(M, H, N).to(DEV)
    opt = torch.optim.Adam(model2.parameters(), lr=5e-3)
    for _ in range(15000):
        loss = nll(model2, x, y); opt.zero_grad(); loss.backward(); opt.step()
    w_tr = {k: v.detach().clone() for k, v in model2.state_dict().items()}
    with torch.no_grad():
        L_tr = nll(model2, x, y).item()
    lam_tr, cells_t = sgld_llc(model2, w_tr, x, y)
    return lam_deg, cells_d, L_deg, lam_tr, cells_t, L_tr


def main():
    print(f"[node6] two-estimator agreement: SGLD lambda-hat vs exact Aoyagi-Watanabe RLCT  (n={N_DATA})")
    print(f"  tolerance (pre-registered): |lam-hat - lam_AW|/lam_AW <= 0.10 at the DEGENERATE point\n")
    print(f"  {'(M,H,N,r)':>12} {'lam_AW':>7} | {'lam@degen':>10} {'cells':>6} {'err%':>7} {'verdict':>9} | {'lam@trained':>11} {'err%':>7}")
    agree, results = 0, []
    for (M, H, N, r) in CONFIGS:
        lam_th = aw_lambda(M, H, N, r)
        ld, cd, Ld, lt, ct, Lt = run_config(M, H, N, r)
        ed = abs(ld - lam_th) / lam_th * 100 if math.isfinite(ld) else float("nan")
        et = abs(lt - lam_th) / lam_th * 100 if math.isfinite(lt) else float("nan")
        ok = math.isfinite(ld) and ed <= 10
        agree += ok
        results.append((M, H, N, r, lam_th, ld, ed, lt, et))
        print(f"  {str((M,H,N,r)):>12} {lam_th:>7.2f} | {ld:>10.2f} {cd:>6} {ed:>6.1f}% {'AGREE' if ok else 'diverge':>9} "
              f"| {lt:>11.2f} {et:>6.1f}%   (L*_deg={Ld:.1e}, L*_tr={Lt:.1e})")
    print(f"\n  PRE-REGISTERED VERDICT: {agree}/{len(CONFIGS)} configs agree at the degenerate point (tol 10%).")
    if agree == len(CONFIGS):
        print(f"  ==> DISCRETE HANDLE VALIDATED: the exact polytope/resolution lambda and the stochastic SGLD")
        print(f"      lambda-hat AGREE across all three AW regimes. The spec's open seam closes in the positive:")
        print(f"      a cheap, exact, discrete route to the learning coefficient, cross-validated on our instrument.")
    elif agree >= len(CONFIGS) - 2:
        print(f"  ==> PARTIAL: agreement in most regimes; the divergent configs LOCATE where the stochastic")
        print(f"      estimator (or the stratum assumption) breaks. The boundary is the finding.")
    else:
        print(f"  ==> DIVERGENT: the two routes disagree broadly -- our SGLD instrument does not recover exact")
        print(f"      RLCTs at this n / these scales. An honest instrument-limit result.")
    # exploratory: did training find the degenerate stratum?
    diffs = [(f"{(M,H,N,r)}", lt - ld) for (M, H, N, r, _, ld, _, lt, _) in results
             if math.isfinite(lt) and math.isfinite(ld)]
    print(f"\n  exploratory (trained minus degenerate lambda-hat): " +
          ", ".join(f"{k}:{v:+.1f}" for k, v in diffs))
    print(f"  -> positive gaps = Adam lands on LESS singular strata than the theory point (expected for generic");
    print(f"     init); ~zero gaps = training is attracted to the degenerate stratum itself.")


if __name__ == "__main__":
    main()
