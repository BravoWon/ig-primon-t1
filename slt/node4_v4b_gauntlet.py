#!/usr/bin/env python
"""NODE 4 FINAL -- PHASE 4b: the N=10 GAUNTLET. Baseline vs Cocycle, phase-aligned, lock-verified.

PRE-REGISTERED (fixed before any data):
  primary  : one-sided Mann-Whitney U, H1: lambda-hat(cocycle) < lambda-hat(baseline), p < 0.05.
             (nonparametric -- baseline lambda spread 18-58 in 3b says normality is fantasy)
  GOLDEN verdict requires ALL FOUR:
    (1) U-test passes (p < 0.05)
    (2) repr-norm(cocycle) within -25% of baseline mean (NOT the Node-2 shrink floor)
    (3) locks held: mean ||R-I|| > 1.0 and mean chunk-cos < 0.9
    (4) >= 8/10 seeds grok in BOTH arms (the recipe is robust, not cherry-picked survivors)
  anything less -> reported as exactly what it is (leak / fragility / null).
Recipe locked from 4a: clip=1.0 both arms, alpha=0.02, beta=0.002, Delta=5000, lambda on shared task loss.

    python slt/node4_v4b_gauntlet.py
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import math, random
import numpy as np
import torch
import torch.nn.functional as F
from node1_grok_llc import make_data, DEV
from node3_topo import TopoTF, task_loss, topo_losses, test_acc, CHUNK

ALPHA, BETA, CLIP, DELTA, STEPS_MAX = 0.02, 0.002, 1.0, 5000, 34000
SEEDS = list(range(10))
EPS_CAL, GAMMA_CAL, CHAINS, DRAWS, BURNIN = 3e-6, 100.0, 4, 120, 80


def set_det(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def sgld_llc(model, w_star, xtr, ytr):
    n = len(xtr); beta = 1.0 / math.log(n); keys = list(w_star); model.eval()
    with torch.no_grad():
        L_star = task_loss(model(xtr)[0], ytr).item()
    lams = []
    for c in range(CHAINS):
        model.load_state_dict(w_star); torch.manual_seed(7000 + c); tr = []
        for t in range(BURNIN + DRAWS):
            L = task_loss(model(xtr)[0], ytr); model.zero_grad(); L.backward()
            with torch.no_grad():
                for kk, prm in zip(keys, model.parameters()):
                    gr = prm.grad if prm.grad is not None else torch.zeros_like(prm)
                    prm.add_(-0.5 * EPS_CAL * (n * beta * gr + GAMMA_CAL * (prm - w_star[kk]))
                             + math.sqrt(EPS_CAL) * torch.randn_like(prm))
            if t >= BURNIN:
                with torch.no_grad():
                    tr.append(task_loss(model(xtr)[0], ytr).item())
        lams.append(n * beta * (np.mean(tr) - L_star))
    model.load_state_dict(w_star)
    return float(np.mean(lams))


def run(mode, seed):
    set_det(seed)
    xtr, ytr, xte, yte = make_data()
    model = TopoTF(mode).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    grok, target, snap = None, None, None
    for step in range(STEPS_MAX):
        model.train(); out, ch = model(xtr); loss = task_loss(out, ytr)
        if mode == "cocycle":
            lc, lcy = topo_losses(model, ch); loss = loss + ALPHA * lc + BETA * lcy
        if not torch.isfinite(loss):
            break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if grok is None and step % 250 == 0:
            model.eval()
            if test_acc(model, xte, yte) > 0.9:
                grok = step; target = min(step + DELTA, STEPS_MAX - 1)
        if target is not None and step == target:
            snap = {k: v.detach().clone() for k, v in model.state_dict().items()}; break
    if snap is None:
        return dict(mode=mode, seed=seed, grok=None, lam=float("nan"), rnorm=float("nan"),
                    rmi=float("nan"), ccos=float("nan"), lcyc=float("nan"))
    model.load_state_dict(snap); model.eval()
    with torch.no_grad():
        out, (Za, Zb, Zc) = model(xtr)
        rnorm = float(torch.cat([Za, Zb, Zc], -1).norm(dim=-1).mean())
        _, lcyc = topo_losses(model, (Za, Zb, Zc))
        I = torch.eye(CHUNK, device=DEV)
        rmi = float(np.mean([(M.weight - I).norm().item() for M in (model.Rab, model.Rbc, model.Rca)]))
        ccos = float(np.mean([F.cosine_similarity(a, b, -1).mean().item()
                              for a, b in ((Za, Zb), (Zb, Zc), (Zc, Za))]))
    lam = sgld_llc(model, snap, xtr, ytr)
    return dict(mode=mode, seed=seed, grok=grok, lam=lam, rnorm=rnorm, rmi=rmi, ccos=ccos, lcyc=float(lcyc))


def mannwhitney_one_sided(a, b):
    """H1: a < b. Exact-ish normal approximation; returns U and one-sided p."""
    a, b = np.asarray(a), np.asarray(b); n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b]); ranks = np.empty(len(allv))
    order = np.argsort(allv); sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    R1 = ranks[:n1].sum(); U = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2; sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U - mu) / sd
    p = 0.5 * (1 + math.erf(z / math.sqrt(2)))          # P(U <= observed) -- small U => a below b
    return U, p


def main():
    print(f"[node4-GAUNTLET] baseline vs cocycle, N={len(SEEDS)} seeds; clip={CLIP} a={ALPHA} b={BETA}")
    rows = []
    for mode in ("baseline", "cocycle"):
        for seed in SEEDS:
            r = run(mode, seed); rows.append(r)
            print(f"  {mode:9} s{seed}: grok@{str(r['grok']):>6}  lam {r['lam']:7.2f}  rnorm {r['rnorm']:7.1f}  "
                  f"||R-I|| {r['rmi']:5.2f}  ccos {r['ccos']:+.2f}  Lcyc {r['lcyc']:.2f}")

    def arm(mode):
        return [r for r in rows if r["mode"] == mode and r["grok"] is not None and math.isfinite(r["lam"])]
    B, C = arm("baseline"), arm("cocycle")
    lamB = [r["lam"] for r in B]; lamC = [r["lam"] for r in C]
    print(f"\n  grokked: baseline {len(B)}/{len(SEEDS)}  cocycle {len(C)}/{len(SEEDS)}")
    print(f"  lambda-hat  baseline: median {np.median(lamB):6.1f}  IQR [{np.percentile(lamB,25):.1f},{np.percentile(lamB,75):.1f}]  full [{min(lamB):.1f},{max(lamB):.1f}]")
    print(f"  lambda-hat  cocycle : median {np.median(lamC):6.1f}  IQR [{np.percentile(lamC,25):.1f},{np.percentile(lamC,75):.1f}]  full [{min(lamC):.1f},{max(lamC):.1f}]")
    rnB = np.mean([r["rnorm"] for r in B]); rnC = np.mean([r["rnorm"] for r in C])
    rmi = np.mean([r["rmi"] for r in C]); ccos = np.mean([r["ccos"] for r in C])
    U, p = mannwhitney_one_sided(lamC, lamB)
    drn = (rnC - rnB) / rnB * 100
    print(f"  repr-norm: baseline {rnB:.1f}  cocycle {rnC:.1f}  ({drn:+.1f}%)")
    print(f"  locks (cocycle): ||R-I|| {rmi:.2f}  chunk-cos {ccos:+.2f}")
    print(f"\n  PRE-REGISTERED VERDICT:")
    c1 = p < 0.05; c2 = drn > -25; c3 = rmi > 1.0 and ccos < 0.9
    c4 = len(B) >= 8 and len(C) >= 8
    print(f"    (1) Mann-Whitney one-sided (cocycle<baseline): U={U:.0f}, p={p:.4f}  -> {'PASS' if c1 else 'FAIL'}")
    print(f"    (2) repr-norm preserved ({drn:+.1f}% vs -25% floor)                 -> {'PASS' if c2 else 'FAIL'}")
    print(f"    (3) locks held (||R-I||={rmi:.1f}>1, ccos={ccos:.2f}<0.9)           -> {'PASS' if c3 else 'FAIL'}")
    print(f"    (4) robustness (>=8/10 grok both arms: {len(B)}/{len(SEEDS)}, {len(C)}/{len(SEEDS)})   -> {'PASS' if c4 else 'FAIL'}")
    if c1 and c2 and c3 and c4:
        print(f"\n  ==> GOLDEN: the operator cocycle SHIFTS the lambda-hat distribution left with representation")
        print(f"      norm preserved and verified non-trivial topology, N=10. Structural algebraic coherence")
        print(f"      flattens the basin independent of volume compression. The load-bearing claim REPLICATES.")
    else:
        failed = [n for n, c in zip("1234", (c1, c2, c3, c4)) if not c]
        print(f"\n  ==> NOT GOLDEN (criteria {','.join(failed)} failed). Report as-is: "
              f"{'a structurally-verified NULL' if not c1 else 'a leak/fragility finding'} at N=10.")


if __name__ == "__main__":
    main()
