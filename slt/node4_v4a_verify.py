#!/usr/bin/env python
"""NODE 4 FINAL -- PHASE 4a: verify the robustified recipe (clip + softened topology), blind, 3 seeds.

Fixes from the 3b wreckage: (1) global grad clip 1.0 -- mandatory for the multiplicative operator
constraint (cubic product -> volatile early gradients -> the te=0.00 NaN deaths); applied to BOTH arms
(matched treatment; clipping is itself regularization, so it must not differ across arms). (2) alpha
0.05->0.02, beta 0.005->0.002 -- room for the task loss early. (3) consistency arm DROPPED (lesson
extracted: cosine-only leaks norm-collapse; the operator cocycle is what prevents it).

Verify, blind to lambda-hat, on seeds {0,1,2}: cocycle groks, no NaN, Lcyc -> 0 (constraint satisfied),
and baseline+clip still groks. Pass -> launch the N=10 gauntlet.

    python slt/node4_v4a_verify.py
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import math, random
import numpy as np
import torch
from node1_grok_llc import make_data, DEV
from node3_topo import TopoTF, task_loss, topo_losses, test_acc

ALPHA, BETA, CLIP, STEPS = 0.02, 0.002, 1.0, 26000


def set_det(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def probe(mode, seed):
    set_det(seed)
    xtr, ytr, xte, yte = make_data()
    model = TopoTF(mode).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    grok, nan_step = None, None
    for step in range(STEPS):
        model.train(); out, ch = model(xtr); loss = task_loss(out, ytr)
        if mode == "cocycle":
            lc, lcy = topo_losses(model, ch); loss = loss + ALPHA * lc + BETA * lcy
        if not torch.isfinite(loss):
            nan_step = step; break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)        # BOTH arms (matched treatment)
        opt.step()
        if grok is None and step % 500 == 0:
            model.eval()
            if test_acc(model, xte, yte) > 0.9:
                grok = step
    model.eval()
    with torch.no_grad():
        out, ch = model(xtr)
        te = test_acc(model, xte, yte)
        lcons, lcyc = topo_losses(model, ch)
    return grok, te, float(lcons), float(lcyc), nan_step


def main():
    print(f"[node4-4a] robustified verify: clip={CLIP} alpha={ALPHA} beta={BETA}; blind; seeds 0-2")
    ok = True
    for seed in (0, 1, 2):
        g, te, lc, lcy, nan = probe("cocycle", seed)
        stat = "NaN@" + str(nan) if nan is not None else ("grok+satisfied" if (g is not None and te > 0.95 and lcy < 1.0) else "PROBLEM")
        ok &= nan is None and g is not None and te > 0.95 and lcy < 1.0
        print(f"  cocycle  s{seed}: grok@{str(g):>6} te {te:.2f}  Lcons {lc:.3f}  Lcyc {lcy:.2f}  -> {stat}")
    g, te, _, _, nan = probe("baseline", 0)
    print(f"  baseline s0: grok@{str(g):>6} te {te:.2f}  (clip must not break the baseline)")
    ok &= nan is None and g is not None and te > 0.95
    print(f"\n  -> {'RECIPE VERIFIED: no NaNs, grok preserved, constraint satisfied. LAUNCH THE N=10 GAUNTLET.' if ok else 'RECIPE STILL BROKEN -- do not launch; adjust clip/alpha further.'}")


if __name__ == "__main__":
    main()
