#!/usr/bin/env python
"""Nodes 3-4 PHASE 3a: BLIND calibration of (alpha, beta).

Find topology strengths that (a) PRESERVE grok and (b) actually SATISFY the constraints (Lcons and Lcyc
driven genuinely low) -- else we'd report a null-by-omission (a constraint we think we imposed but didn't).
Blind to lambda-hat. Locks the (alpha, beta) for the phase-aligned main run (3b).

    python slt/node3_v3a_calibrate.py
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import math, random
import numpy as np
import torch
from node1_grok_llc import make_data, DEV
from node3_topo import TopoTF, task_loss, topo_losses, test_acc

STEPS, SEED = 22000, 0
GRID = [(0.05, 0.005), (0.1, 0.01), (0.3, 0.03)]     # (alpha, beta) candidates
CONS_OK, CYC_OK = 0.15, 1.0                          # "constraint satisfied" thresholds


def set_det(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def train_probe(mode, alpha, beta):
    set_det(SEED)
    xtr, ytr, xte, yte = make_data()
    model = TopoTF(mode).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    grok = None
    for step in range(STEPS):
        model.train(); out, ch = model(xtr)
        loss = task_loss(out, ytr)
        if mode in ("consistency", "cocycle"):
            lc, lcy = topo_losses(model, ch); loss = loss + alpha * lc + (beta * lcy if mode == "cocycle" else 0)
        opt.zero_grad(); loss.backward(); opt.step()
        if grok is None and step % 500 == 0:
            model.eval();
            if test_acc(model, xte, yte) > 0.9:
                grok = step
    model.eval()
    with torch.no_grad():
        _, ch = model(xtr); lcons, lcyc = topo_losses(model, ch)
        te = test_acc(model, xte, yte)
    return grok, te, float(lcons), float(lcyc)


def main():
    print("[node3-3a] blind (alpha,beta) calibration: preserve grok AND satisfy the topology constraints")
    print(f"  baseline anchor:")
    g, te, lc, lcy = train_probe("baseline", 0, 0)
    print(f"    baseline  grok@{g}  te {te:.2f}  (Lcons {lc:.3f} Lcyc {lcy:.2f} = untrained topology, reference)")
    print(f"\n  cocycle arm sweep (the most-constrained; if it groks + satisfies, consistency will too):")
    best = None
    for a, b in GRID:
        g, te, lc, lcy = train_probe("cocycle", a, b)
        grokd = g is not None and te > 0.95
        satisfied = lc < CONS_OK and lcy < CYC_OK
        tag = "GROK+SATISFIED" if (grokd and satisfied) else ("grok, weak-constraint" if grokd else "BROKE GROK")
        print(f"    a={a:<5} b={b:<6} grok@{str(g):>6} te {te:.2f}  Lcons {lc:.3f}  Lcyc {lcy:.2f}  -> {tag}")
        if grokd and satisfied and best is None:
            best = (a, b, g)
    print()
    if best:
        # confirm the consistency arm also groks at this alpha
        gc, tec, lcc, _ = train_probe("consistency", best[0], best[1])
        print(f"  consistency arm @ a={best[0]}: grok@{gc} te {tec:.2f} Lcons {lcc:.3f}")
        print(f"  -> LOCK alpha={best[0]}, beta={best[1]} (grok preserved, constraints satisfied). Run 3b.")
    else:
        print(f"  -> NO (alpha,beta) both grokked and satisfied the constraints. Options: widen grid, or the")
        print(f"     topology is intrinsically incompatible with grok here (a finding). Do NOT run 3b blind.")


if __name__ == "__main__":
    main()
