#!/usr/bin/env python
"""NODE 6 v2 -- two-estimator agreement with a properly calibrated recipe.

v1 produced NO measurement (0 valid cells): the transformer-scale eps grid never equilibrated on 6-60
parameter models (relaxation ~1/(eps*gamma) >> chain length). Diagnostic on ONE config -- (3,2,3,1),
hereby declared IN-SAMPLE -- revealed the standard SGLD-LLC systematics: under-equilibration biases
lambda-hat LOW at small eps; discretization heat biases it HIGH at large eps; localization gamma biases
it LOW (gamma->0 is the definition of local).

RECIPE (fixed by criteria, not by closeness to theory; identical for every config):
  gamma=10 (smallest non-divergent), BURNIN=1500 / DRAWS=1500 (kills the equilibration bias),
  eps in {1e-4, 2e-4, 3e-4} (the stationary band), validity = finite + drift < 0.10 + no divergence,
  lambda-hat = LINEAR EXTRAPOLATION eps->0 over valid cells (the principled discretization-bias fix).

PRE-REGISTERED: agreement iff |lam-hat - lam_AW|/lam_AW <= 0.10 at the constructed degenerate point.
Config (3,2,3,1) is IN-SAMPLE (used to calibrate) -- reported but NOT counted. The other FIVE configs
are out-of-sample; the verdict counts those alone.

    python slt/node6_v2_agreement.py
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import math, random
import numpy as np
import torch
from node6_two_estimator import aw_lambda, RRR, nll, set_det, CONFIGS, N_DATA, DEV

GAMMA, EPS_BAND, BURNIN, DRAWS, CHAINS, DRIFT_TOL = 10.0, [1e-4, 2e-4, 3e-4], 1500, 1500, 3, 0.10
INSAMPLE = (3, 2, 3, 1)


def lam_at_eps(model, w_star, x, y, eps):
    n = len(x); beta = 1.0 / math.log(n); keys = list(w_star)
    lams, drifts = [], []
    for c in range(CHAINS):
        model.load_state_dict(w_star); torch.manual_seed(7000 + c)
        tr = []
        for t in range(BURNIN + DRAWS):
            L = nll(model, x, y); model.zero_grad(); L.backward()
            with torch.no_grad():
                for kk, prm in zip(keys, model.parameters()):
                    prm.add_(-0.5 * eps * (n * beta * prm.grad + GAMMA * (prm - w_star[kk]))
                             + math.sqrt(eps) * torch.randn_like(prm))
            if t >= BURNIN:
                with torch.no_grad():
                    tr.append(nll(model, x, y).item())
        tr = np.array(tr)
        if not np.all(np.isfinite(tr)) or tr.mean() > 10:
            return float("nan"), float("nan")
        lams.append(n * beta * tr.mean())                       # L* ~ 0 at the variety
        h = len(tr) // 2
        drifts.append(abs(tr[h:].mean() - tr[:h].mean()) / (tr.mean() + 1e-12))
    model.load_state_dict(w_star)
    return float(np.mean(lams)), float(np.mean(drifts))


def lam_extrapolated(model, w_star, x, y):
    pts = []
    for eps in EPS_BAND:
        lam, drift = lam_at_eps(model, w_star, x, y, eps)
        ok = math.isfinite(lam) and drift < DRIFT_TOL
        pts.append((eps, lam, drift, ok))
    valid = [(e, l) for e, l, d, ok in pts if ok]
    if len(valid) < 2:
        return float("nan"), pts
    es = np.array([e for e, _ in valid]); ls = np.array([l for _, l in valid])
    slope, intercept = np.polyfit(es, ls, 1)                   # linear eps->0 extrapolation
    return float(intercept), pts


def degenerate_point(M, H, N, r, seed=0):
    set_det(seed)
    U = torch.randn(N, r, device=DEV) / math.sqrt(r)
    V = torch.randn(r, M, device=DEV) / math.sqrt(M)
    x = torch.randn(N_DATA, M, device=DEV); y = x @ V.T @ U.T
    model = RRR(M, H, N).to(DEV)
    with torch.no_grad():
        model.A.zero_(); model.B.zero_(); model.A[:r] = V; model.B[:, :r] = U
    w = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return model, w, x, y


def main():
    print(f"[node6-v2] recipe: gamma={GAMMA} burnin={BURNIN} draws={DRAWS} eps-band={EPS_BAND} -> eps->0 extrapolation")
    print(f"  in-sample calibration config: {INSAMPLE} (reported, NOT counted)\n")
    print(f"  {'(M,H,N,r)':>12} {'lam_AW':>7} {'lam-hat(0)':>11} {'err%':>7} {'cells':>6} {'scope':>12} {'verdict':>9}")
    agree = total = 0
    for (M, H, N, r) in CONFIGS:
        model, w, x, y = degenerate_point(M, H, N, r)
        lam_th = aw_lambda(M, H, N, r)
        lam, pts = lam_extrapolated(model, w, x, y)
        ncell = sum(1 for *_, ok in pts if ok)
        err = abs(lam - lam_th) / lam_th * 100 if math.isfinite(lam) else float("nan")
        insample = (M, H, N, r) == INSAMPLE
        ok = math.isfinite(lam) and err <= 10
        if not insample:
            total += 1; agree += ok
        detail = "  ".join(f"e{e:.0e}:{l:.2f}(d{d:.2f}{'' if k else '!'})" for e, l, d, k in pts)
        print(f"  {str((M,H,N,r)):>12} {lam_th:>7.2f} {lam:>11.2f} {err:>6.1f}% {ncell:>6} "
              f"{'IN-SAMPLE' if insample else 'out-of-sample':>12} {'AGREE' if ok else 'diverge':>9}")
        print(f"      cells: {detail}")
    print(f"\n  PRE-REGISTERED VERDICT (out-of-sample only): {agree}/{total} agree (tol 10%).")
    if agree == total:
        print(f"  ==> DISCRETE HANDLE VALIDATED out-of-sample: exact AW lambda == SGLD lambda-hat across the")
        print(f"      remaining regimes. The spec's open seam closes positive.")
    elif agree >= total - 1:
        print(f"  ==> NEAR-TOTAL: one divergent config locates the boundary; report it as the finding.")
    else:
        print(f"  ==> PARTIAL/DIVERGENT: the boundary structure across configs is the finding; report which")
        print(f"      regimes the stochastic estimator recovers and which it cannot at this n/recipe.")


if __name__ == "__main__":
    main()
