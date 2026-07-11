#!/usr/bin/env python
"""NODE 1 -> NODE 2 GATE (Path B determinism protocol).

Path B (relative-shift) is only licensed if the relative lambda-hat shift is STABLE. Two tests:

  TEST 1 -- DETERMINISM (the v2/v3 culprit was GPU non-determinism): lock CUDA + framework determinism,
            train the grok TWICE with the same seed, compare weight fingerprints. Identical => GPU noise
            eliminated, "3 identical runs" are now reproducible by construction.
  TEST 2 -- RELATIVE-SHIFT STABILITY (the substantive gate): nodes 2-5 compare DIFFERENT models
            (baseline vs treatment), so the relevant noise is model-to-model. Train 3 seeds; measure the
            relative drop  D = (lambda_pre - lambda_post)/lambda_pre  across the grok transition each time.
            GATE CLEARS iff: sign(D) identical every run AND std(D)/mean(D) < 0.05.  Else -> roll to Path A.

The SGLD itself uses a FIXED seed across all measurements, so any variation in D is the MODEL's, not the
sampler's -- isolating exactly what nodes 2-5 will face.

    python slt/node1_v4_gate.py
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"            # required for deterministic cuBLAS (set first)
import math, random
import numpy as np
import torch
import torch.nn.functional as F
from node1_grok_llc import Transformer, make_data, loss_acc, DEV

STEPS, PRE, POST = 26000, 14000, 25999
EPS_CAL, GAMMA_CAL, CHAINS, DRAWS, BURNIN = 3e-6, 100.0, 4, 150, 80


def set_determinism(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fingerprint(state):
    """A cheap, determinism-sensitive weight fingerprint."""
    s1 = sum(float(v.double().sum()) for v in state.values())
    s2 = sum(float((v.double() ** 2).sum()) for v in state.values())
    return (round(s1, 6), round(s2, 6))


def train_grok(seed):
    set_determinism(seed)
    xtr, ytr, xte, yte = make_data()
    model = Transformer().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    snaps, grok = {}, None
    for step in range(STEPS):
        model.train(); l, _ = loss_acc(model, xtr, ytr)
        opt.zero_grad(); l.backward(); opt.step()
        if step in (PRE, POST):
            snaps[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if grok is None and step % 500 == 0:
            with torch.no_grad():
                _, ate = loss_acc(model, xte, yte)
            if ate > 0.9:
                grok = step
    with torch.no_grad():
        _, final_te = loss_acc(model, xte, yte)
    return model, snaps, grok, float(final_te), (xtr, ytr)


def sgld_llc(model, w_star, xtr, ytr):
    n = len(xtr); beta = 1.0 / math.log(n); keys = list(w_star)
    with torch.no_grad():
        L_star = F.cross_entropy(model(xtr), ytr).item()
    lams = []
    for c in range(CHAINS):
        model.load_state_dict(w_star); torch.manual_seed(7000 + c)        # FIXED sampler seed
        tr = []
        for t in range(BURNIN + DRAWS):
            L = F.cross_entropy(model(xtr), ytr); model.zero_grad(); L.backward()
            with torch.no_grad():
                for k, prm in zip(keys, model.parameters()):
                    g = n * beta * prm.grad + GAMMA_CAL * (prm - w_star[k])
                    prm.add_(-0.5 * EPS_CAL * g + math.sqrt(EPS_CAL) * torch.randn_like(prm))
            if t >= BURNIN:
                with torch.no_grad():
                    tr.append(F.cross_entropy(model(xtr), ytr).item())
        lams.append(n * beta * (np.mean(tr) - L_star))
    model.load_state_dict(w_star)
    return float(np.mean(lams))


def run(seed):
    model, snaps, grok, final_te, (xtr, ytr) = train_grok(seed)
    fp = fingerprint(snaps[POST])
    lam_pre = sgld_llc(model, snaps[PRE], xtr, ytr)
    lam_post = sgld_llc(model, snaps[POST], xtr, ytr)
    D = (lam_pre - lam_post) / lam_pre
    return dict(seed=seed, grok=grok, te=final_te, fp=fp, lam_pre=lam_pre, lam_post=lam_post, D=D)


def main():
    print(f"[gate] determinism locked (CUBLAS_WORKSPACE_CONFIG=:4096:8, use_deterministic_algorithms)")
    print(f"\n=== TEST 1: DETERMINISM (two identical seed-0 runs must match) ===")
    a, b = run(0), run(0)
    match = a["fp"] == b["fp"]
    print(f"  run A: grok@{a['grok']} te={a['te']:.3f} fp={a['fp']}  lam_pre={a['lam_pre']:.1f} lam_post={a['lam_post']:.1f}")
    print(f"  run B: grok@{b['grok']} te={b['te']:.3f} fp={b['fp']}  lam_pre={b['lam_pre']:.1f} lam_post={b['lam_post']:.1f}")
    print(f"  -> DETERMINISM: {'PASS (bit-identical grokked weights -- GPU noise eliminated)' if match else 'FAIL (weights differ despite locked seeds -- a non-deterministic op remains)'}")

    print(f"\n=== TEST 2: RELATIVE-SHIFT STABILITY across seeds {{0,1,2}} (the substantive gate) ===")
    runs = [a] + [run(s) for s in (1, 2)]
    print(f"  {'seed':>5}{'grok':>7}{'te':>7}{'lam_pre':>9}{'lam_post':>10}{'D=drop':>9}")
    for r in runs:
        print(f"  {r['seed']:>5}{r['grok']:>7}{r['te']:>7.3f}{r['lam_pre']:>9.1f}{r['lam_post']:>10.1f}{r['D']*100:>8.1f}%")
    Ds = [r["D"] for r in runs]
    same_sign = all(d > 0 for d in Ds) or all(d < 0 for d in Ds)
    cv = float(np.std(Ds) / abs(np.mean(Ds))) if np.mean(Ds) != 0 else float("inf")
    print(f"\n  mean drop {np.mean(Ds)*100:+.1f}%   std/mean (CV) {cv*100:.1f}%   sign-consistent: {same_sign}")
    clears = match and same_sign and cv < 0.05
    if clears:
        print(f"  -> GATE CLEARS: determinism locked, relative drop sign-consistent, CV {cv*100:.1f}% < 5%.")
        print(f"     Path B licensed -> proceed to NODE 2 (per-layer bottleneck), measuring lambda-hat SHIFTS.")
    elif match and same_sign:
        print(f"  -> GATE MARGINAL: determinism + sign hold, but CV {cv*100:.1f}% exceeds 5%. Relative shift is")
        print(f"     directionally stable but noisy across models -> node 2-5 need MANY seeds, or Path A. Decide.")
    else:
        print(f"  -> GATE FAILS: {'non-determinism remains' if not match else 'relative shift not sign-stable'}.")
        print(f"     Roll back to PATH A: import devinterp, establish absolute calibration before nodes 2-5.")


if __name__ == "__main__":
    main()
