#!/usr/bin/env python
"""NODE 1 v2 -- pass the calibration gate the v1 sweep FAILED.

v1 finding: with gamma=100 fixed, lambda-hat climbs monotonically with SGLD step size eps (5.7 -> 807,
no plateau) => the chain ESCAPES the local basin as eps grows instead of sampling a FIXED localized
posterior. The estimate was a sampler artifact. Per spec, node 2 is blocked until this is fixed.

The fix:
  1. JOINT (eps, gamma) sweep -- localization gamma trades off against step size; a valid LLC needs a
     region where the chain mixes LOCALLY (E[L] > L*) yet stays BOUNDED (doesn't drift to another basin).
  2. STATIONARITY diagnostic -- the post-burnin SGLD loss trace must mean-revert, not drift. drift =
     |mean(2nd half) - mean(1st half)| / mean(trace). A valid cell has small drift AND no divergence.
  3. CHECKPOINT the grokked weights so calibration iterates cheaply (no 10-min retrain each time).

PASS = a contiguous (eps,gamma) region that is non-divergent, stationary (drift < D), and where
lambda-hat agrees within tol. That lambda-hat is the calibrated readout. Else: still failing, report it.

    python slt/node1_v2_calibrate.py
"""
import os, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from node1_grok_llc import Transformer, make_data, loss_acc, train, DEV

CKPT = "slt/grok_ckpt.pt"
EPS_GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
GAMMA_GRID = [30.0, 100.0, 300.0, 1000.0, 3000.0]
CHAINS, DRAWS, BURNIN, DRIFT_TOL, LAM_TOL = 4, 200, 100, 0.15, 0.15
NAVY = "#15293f"


def train_or_load():
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=DEV)
        m = Transformer().to(DEV); m.load_state_dict(ck["w"])
        print(f"[v2] loaded grokked checkpoint (grok@{ck['grok_step']}, test_acc={ck['test_acc']:.3f})")
        xtr, ytr, _, _ = make_data()
        return m, ck["w"], (xtr, ytr)
    print("[v2] no checkpoint -- training to grok once (~10 min on GPU), then caching...")
    model, snaps, hist, (xtr, ytr), grok_step = train()
    w = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"w": w, "grok_step": grok_step, "test_acc": hist["te_acc"][-1]}, CKPT)
    print(f"[v2] cached -> {CKPT}")
    return model, w, (xtr, ytr)


def sgld_trace(model, w_star, xtr, ytr, eps, gamma):
    n = len(xtr); beta = 1.0 / math.log(n); keys = list(w_star)
    with torch.no_grad():
        L_star = F.cross_entropy(model(xtr), ytr).item()
    lams, drifts, diverged = [], [], 0
    for c in range(CHAINS):
        model.load_state_dict(w_star); torch.manual_seed(1000 + c)
        trace = []
        for t in range(BURNIN + DRAWS):
            L = F.cross_entropy(model(xtr), ytr); model.zero_grad(); L.backward()
            with torch.no_grad():
                for k, prm in zip(keys, model.parameters()):
                    grad = n * beta * prm.grad + gamma * (prm - w_star[k])
                    prm.add_(-0.5 * eps * grad + math.sqrt(eps) * torch.randn_like(prm))
            if t >= BURNIN:
                with torch.no_grad():
                    trace.append(F.cross_entropy(model(xtr), ytr).item())
        tr = np.array(trace)
        if not np.all(np.isfinite(tr)) or tr.mean() > 5 * L_star + 10:
            diverged += 1; continue
        lams.append(n * beta * (tr.mean() - L_star))
        h = len(tr) // 2
        drifts.append(abs(tr[h:].mean() - tr[:h].mean()) / (tr.mean() + 1e-9))
    model.load_state_dict(w_star)
    if not lams:
        return float("nan"), float("nan"), float("nan"), diverged
    return float(np.mean(lams)), float(np.std(lams)), float(np.mean(drifts)), diverged


def main():
    model, w_star, (xtr, ytr) = train_or_load()
    print(f"\n[v2] JOINT (eps,gamma) calibration sweep -- valid cell = non-divergent AND drift<{DRIFT_TOL}")
    print(f"{'gamma \\ eps':>12}" + "".join(f"{e:>11.0e}" for e in EPS_GRID))
    LAM = np.full((len(GAMMA_GRID), len(EPS_GRID)), np.nan)
    valid = np.zeros_like(LAM, dtype=bool)
    for gi, g in enumerate(GAMMA_GRID):
        cells = []
        for ei, e in enumerate(EPS_GRID):
            lam, sd, drift, div = sgld_trace(model, w_star, xtr, ytr, e, g)
            LAM[gi, ei] = lam
            ok = (div == 0) and math.isfinite(drift) and drift < DRIFT_TOL
            valid[gi, ei] = ok and math.isfinite(lam)
            tag = "*" if ok else ("d" if div else ".")        # * valid, d diverged, . drifting
            cells.append(f"{lam:9.1f}{tag}" if math.isfinite(lam) else "    nan ")
        print(f"{g:>12.0f}" + "".join(f"{c:>11}" for c in cells))
    print("  legend: * = valid (stationary, non-divergent)   d = diverged   . = drifting (escaped basin)")

    # find the largest stable plateau among valid cells: neighbors agreeing within LAM_TOL
    plateau = []
    for gi in range(len(GAMMA_GRID)):
        for ei in range(len(EPS_GRID)):
            if not valid[gi, ei]:
                continue
            neigh = [(gi + dg, ei + de) for dg, de in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
            agree = [LAM[gi, ei]]
            for ng, ne in neigh:
                if 0 <= ng < len(GAMMA_GRID) and 0 <= ne < len(EPS_GRID) and valid[ng, ne]:
                    if abs(LAM[ng, ne] - LAM[gi, ei]) / LAM[gi, ei] < LAM_TOL:
                        agree.append(LAM[ng, ne])
            if len(agree) >= 2:
                plateau.append((np.mean(agree), len(agree), GAMMA_GRID[gi], EPS_GRID[ei]))
    print()
    if plateau:
        plateau.sort(key=lambda x: -x[1])
        lam_cal, sz, g_c, e_c = plateau[0]
        print(f"[v2] PASS: stable plateau of {sz} agreeing valid cells around (gamma={g_c:.0f}, eps={e_c:.0e})")
        print(f"     CALIBRATED lambda-hat(grokked) = {lam_cal:.2f}   -> NODE 1 PASSES; node 2 unblocked.")
        print(f"     next: re-measure lambda-hat across the grok transition at this (eps,gamma), with error bars.")
    else:
        nvalid = int(valid.sum())
        print(f"[v2] STILL FAILING: {nvalid} valid cells but no agreeing plateau. The local posterior has no")
        print(f"     clean scale here -- options: finer (eps,gamma) grid near valid cells, longer chains,")
        print(f"     or a tempered/annealed SGLD. Node 2 stays blocked. (An honest second 'no'.)")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    disp = np.log10(np.where(np.isfinite(LAM) & (LAM > 0), LAM, np.nan))
    im = ax.imshow(disp, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(EPS_GRID))); ax.set_xticklabels([f"{e:.0e}" for e in EPS_GRID])
    ax.set_yticks(range(len(GAMMA_GRID))); ax.set_yticklabels([f"{g:.0f}" for g in GAMMA_GRID])
    ax.set_xlabel("SGLD step size eps"); ax.set_ylabel("localization gamma")
    for gi in range(len(GAMMA_GRID)):
        for ei in range(len(EPS_GRID)):
            if math.isfinite(LAM[gi, ei]):
                mark = "*" if valid[gi, ei] else ""
                ax.text(ei, gi, f"{LAM[gi,ei]:.0f}{mark}", ha="center", va="center",
                        color="w" if disp[gi, ei] < np.nanmean(disp) else "k", fontsize=7.5)
    ax.set_title("Node 1 v2: lambda-hat over (eps,gamma); * = valid (stationary). Plateau = calibrated.",
                 color=NAVY, fontsize=9.5)
    fig.colorbar(im, label="log10 lambda-hat"); fig.tight_layout()
    fig.savefig("slt/node1_v2_calibration.png", dpi=160); plt.close(fig)
    print("  wrote slt/node1_v2_calibration.png")


if __name__ == "__main__":
    main()
