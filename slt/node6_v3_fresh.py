#!/usr/bin/env python
"""NODE 6 v3 -- the fresh-config confirmatory test.

v2's formal 0/5 was GATE MECHANICS, not physics: raw cells tracked the exact AW lambda to 1-18% across
all regimes, but a transformer-calibrated drift bar (0.10) rejected cells at 0.11-0.16, and the eps->0
extrapolation pointed the wrong way (the residual bias was UNDER-EQUILIBRATION -- lambda-hat still
rising toward theory with eps -- not discretization heat). All six v2 configs are now contaminated
(seen), so they can only VALIDATE the recipe in-sample. The confirmatory test moves to FIVE FRESH
configs, never measured, hand-verified against the formula, spanning ALL FOUR AW regimes.

RECIPE (fixed in advance, identical everywhere): gamma=10, BURNIN=6000 (kill equilibration bias),
DRAWS=3000, eps in {1e-4, 2e-4, 3e-4}, validity = finite + drift < 0.10, lambda-hat = MEAN of valid
cells (no directional extrapolation).

PRE-REGISTERED: fresh-config agreement iff |lam-hat - lam_AW|/lam_AW <= 0.10; verdict counts the FIVE
FRESH configs only; >= 4/5 -> discrete handle validated; else the gap structure is the finding.

    python slt/node6_v3_fresh.py
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import math
import numpy as np
import torch
from node6_two_estimator import aw_lambda, RRR, nll, set_det, N_DATA, DEV

GAMMA, EPS_BAND, BURNIN, DRAWS, CHAINS, DRIFT_TOL = 10.0, [1e-4, 2e-4, 3e-4], 6000, 3000, 3, 0.10
SEEN = [(3, 1, 3, 1), (3, 2, 3, 1), (4, 3, 4, 1), (4, 3, 4, 2), (3, 8, 3, 1), (2, 2, 8, 2)]
FRESH = [(5, 4, 5, 2), (4, 2, 6, 1), (6, 4, 4, 3), (2, 6, 2, 1), (5, 3, 3, 2)]
HAND = {(5, 4, 5, 2): 10.5, (4, 2, 6, 1): 6.0, (6, 4, 4, 3): 11.0, (2, 6, 2, 1): 2.0, (5, 3, 3, 2): 6.5}


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
        lams.append(n * beta * tr.mean())
        h = len(tr) // 2
        drifts.append(abs(tr[h:].mean() - tr[:h].mean()) / (tr.mean() + 1e-12))
    model.load_state_dict(w_star)
    return float(np.mean(lams)), float(np.mean(drifts))


def measure(M, H, N, r, seed=0):
    set_det(seed)
    U = torch.randn(N, r, device=DEV) / math.sqrt(r)
    V = torch.randn(r, M, device=DEV) / math.sqrt(M)
    x = torch.randn(N_DATA, M, device=DEV); y = x @ V.T @ U.T
    model = RRR(M, H, N).to(DEV)
    with torch.no_grad():
        model.A.zero_(); model.B.zero_(); model.A[:r] = V; model.B[:, :r] = U
    w = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cells = []
    for eps in EPS_BAND:
        lam, drift = lam_at_eps(model, w, x, y, eps)
        cells.append((eps, lam, drift, math.isfinite(lam) and drift < DRIFT_TOL))
    valid = [l for _, l, _, ok in cells if ok]
    return (float(np.mean(valid)) if valid else float("nan")), len(valid), cells


def main():
    for cfg, lam_hand in HAND.items():                          # formula/hand cross-check before anything
        assert abs(aw_lambda(*cfg) - lam_hand) < 1e-9, (cfg, aw_lambda(*cfg), lam_hand)
    print(f"[node6-v3] recipe: gamma={GAMMA} burnin={BURNIN} draws={DRAWS} eps={EPS_BAND}; mean of valid cells")
    print(f"  fresh-config AW values hand-verified. Verdict counts the FRESH five only.\n")
    print(f"  {'(M,H,N,r)':>12} {'lam_AW':>7} {'lam-hat':>9} {'err%':>7} {'valid':>6} {'scope':>15} {'verdict':>9}")
    stats = {"seen": [0, 0], "fresh": [0, 0]}
    for scope, configs in (("seen", SEEN), ("fresh", FRESH)):
        for (M, H, N, r) in configs:
            lam_th = aw_lambda(M, H, N, r)
            lam, nvalid, cells = measure(M, H, N, r)
            err = abs(lam - lam_th) / lam_th * 100 if math.isfinite(lam) else float("nan")
            ok = math.isfinite(lam) and err <= 10
            stats[scope][0] += ok; stats[scope][1] += 1
            tag = "in-sample(val)" if scope == "seen" else "FRESH"
            print(f"  {str((M,H,N,r)):>12} {lam_th:>7.2f} {lam:>9.2f} {err:>6.1f}% {nvalid:>6} {tag:>15} "
                  f"{'AGREE' if ok else 'diverge':>9}")
            print(f"      cells: " + "  ".join(f"e{e:.0e}:{l:.2f}(d{d:.2f}{'' if k else '!'})" for e, l, d, k in cells))
    sv, st = stats["seen"]; fv, ft = stats["fresh"]
    print(f"\n  in-sample recipe validation: {sv}/{st} of the seen configs agree.")
    print(f"  PRE-REGISTERED VERDICT (fresh only): {fv}/{ft} agree (tol 10%; bar: >=4/5).")
    if fv >= 4:
        print(f"  ==> DISCRETE HANDLE VALIDATED on virgin configs across all four AW regimes: the exact")
        print(f"      polytope/resolution lambda and the SGLD lambda-hat agree. The spec's open seam closes")
        print(f"      POSITIVE -- a cheap exact combinatorial route to the learning coefficient, cross-validated.")
    else:
        print(f"  ==> NOT VALIDATED at the bar. The per-config gap structure (which regimes, which lambda")
        print(f"      magnitudes) is the finding; report as the located boundary of the stochastic estimator.")


if __name__ == "__main__":
    main()
