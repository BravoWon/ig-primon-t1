#!/usr/bin/env python
"""Coherence Optimization Standard -- NODE 1: reproduce grokking + instrument a CALIBRATED lambda-hat.

The spec's rule: "Baseline grok. Reproduce grokking, instrument lambda-hat. Fail -> fix setup first."
Nothing downstream (bottleneck, coherence, the decisive node 4) is meaningful until:
  (a) grokking reproduces (clean memorize -> generalize transition on modular addition), and
  (b) the local learning coefficient lambda-hat is MEASURABLE and STABLE (an uncalibrated estimator
      makes nodes 2-6 unfalsifiable -- the session's spine: an unstable instrument is no instrument).

Testbed: 1-layer transformer on (a + b) mod p, Nanda "Progress Measures" config.
Estimator: SGLD local learning coefficient (Lau-Murfet / devinterp math), implemented transparently
  so every hyperparameter is visible for calibration. lambda-hat(w*) = n*beta*(E_sgld[L_n] - L_n(w*)),
  beta = 1/log n; SGLD samples the localized tempered posterior U(w)=n*beta*L_n(w)+(gamma/2)||w-w*||^2.

NODE-1 DELIVERABLE: a grok curve + an epsilon-sweep showing lambda-hat converges to a stable value
  (CV across the stable band < tol). Pass -> the ladder is real and we build node 2. Fail -> fix here.

    python slt/node1_grok_llc.py
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P = 113                                  # prime modulus
D_MODEL, N_HEADS, D_HEAD, D_MLP = 128, 4, 32, 512
TRAIN_FRAC, WD, LR, STEPS, SEED = 0.30, 1.0, 1e-3, 30000, 0
EVAL_EVERY, SNAP_AT = 200, [400, 1500, 5000, 12000, 29999]
NAVY, GREEN, RED, BLUE = "#15293f", "#1e7d34", "#c0392b", "#2c6fbb"


def make_data():
    a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij")
    a, b = a.flatten(), b.flatten()
    eq = torch.full_like(a, P)                                   # '=' token id = P
    x = torch.stack([a, b, eq], 1)                              # [N, 3]
    y = (a + b) % P                                             # target token
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(x), generator=g)
    ntr = int(TRAIN_FRAC * len(x))
    tr, te = perm[:ntr], perm[ntr:]
    return x[tr].to(DEV), y[tr].to(DEV), x[te].to(DEV), y[te].to(DEV)


class Transformer(nn.Module):
    """1-layer, attention + ReLU MLP, no biases/LN -- the Nanda grokking config."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(P + 1, D_MODEL)
        self.pos = nn.Parameter(torch.randn(3, D_MODEL) / math.sqrt(D_MODEL))
        self.Wq, self.Wk, self.Wv = (nn.Linear(D_MODEL, N_HEADS * D_HEAD, bias=False) for _ in range(3))
        self.Wo = nn.Linear(N_HEADS * D_HEAD, D_MODEL, bias=False)
        self.Win, self.Wout = nn.Linear(D_MODEL, D_MLP, bias=False), nn.Linear(D_MLP, D_MODEL, bias=False)
        self.unembed = nn.Linear(D_MODEL, P, bias=False)

    def forward(self, x):
        B = x.shape[0]
        h = self.embed(x) + self.pos[None]                      # [B,3,D]
        q, k, v = (W(h).view(B, 3, N_HEADS, D_HEAD).transpose(1, 2) for W in (self.Wq, self.Wk, self.Wv))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D_HEAD)
        mask = torch.triu(torch.ones(3, 3, device=x.device), 1).bool()
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        z = (att @ v).transpose(1, 2).reshape(B, 3, -1)
        h = h + self.Wo(z)
        h = h + self.Wout(F.relu(self.Win(h)))
        return self.unembed(h[:, -1])                           # predict at '=' position


def loss_acc(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits, y), (logits.argmax(-1) == y).float().mean().item()


def train():
    torch.manual_seed(SEED)
    xtr, ytr, xte, yte = make_data()
    n = len(xtr)
    model = Transformer().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.98))
    hist = {"step": [], "tr_acc": [], "te_acc": [], "tr_loss": [], "te_loss": []}
    snaps = {}
    print(f"[node1] dev={DEV} p={P} n_train={n} steps={STEPS} wd={WD}  (training to grok...)")
    for step in range(STEPS):
        model.train()
        l, _ = loss_acc(model, xtr, ytr)
        opt.zero_grad(); l.backward(); opt.step()
        if step in SNAP_AT:
            snaps[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            model.eval()
            with torch.no_grad():
                ltr, atr = loss_acc(model, xtr, ytr); lte, ate = loss_acc(model, xte, yte)
            for kk, vv in zip(hist, [step, atr, ate, ltr.item(), lte.item()]):
                hist[kk].append(vv)
            if step % (EVAL_EVERY * 10) == 0:
                print(f"  step {step:6d}  train_acc {atr:.3f}  test_acc {ate:.3f}  test_loss {lte:.4f}")
    grok_step = next((s for s, a in zip(hist["step"], hist["te_acc"]) if a > 0.9), None)
    print(f"  -> final train_acc {hist['tr_acc'][-1]:.3f}  test_acc {hist['te_acc'][-1]:.3f}"
          f"  | grok (test>0.9) at step {grok_step}")
    return model, snaps, hist, (xtr, ytr), grok_step


def sgld_llc(model, w_star, xtr, ytr, eps, gamma=100.0, chains=4, draws=200, burnin=50):
    """SGLD local learning coefficient. Returns lambda-hat mean/std over chains + diagnostics."""
    n = len(xtr); beta = 1.0 / math.log(n)
    keys = [k for k in w_star]
    with torch.no_grad():
        L_star = F.cross_entropy(model(xtr), ytr).item()
    lams, diverged = [], 0
    for c in range(chains):
        model.load_state_dict(w_star)
        torch.manual_seed(1000 + c)
        acc_L, m = 0.0, 0
        for t in range(burnin + draws):
            L = F.cross_entropy(model(xtr), ytr)
            model.zero_grad(); L.backward()
            with torch.no_grad():
                for k, prm in zip(keys, model.parameters()):
                    grad = n * beta * prm.grad + gamma * (prm - w_star[k])
                    noise = math.sqrt(eps) * torch.randn_like(prm)
                    prm.add_(-0.5 * eps * grad + noise)
            if t >= burnin:
                with torch.no_grad():
                    acc_L += F.cross_entropy(model(xtr), ytr).item(); m += 1
        Lbar = acc_L / max(m, 1)
        if not math.isfinite(Lbar) or Lbar > 5 * L_star + 10:
            diverged += 1; continue
        lams.append(n * beta * (Lbar - L_star))
    model.load_state_dict(w_star)                               # restore
    if not lams:
        return float("nan"), float("nan"), diverged, L_star
    return float(np.mean(lams)), float(np.std(lams)), diverged, L_star


def calibrate(model, snaps, data, hist, grok_step):
    """NODE-1 gate: sweep SGLD step size; lambda-hat must converge to a stable band."""
    xtr, ytr = data
    # use the final grokked weights
    w_final = {k: v.detach().clone() for k, v in model.state_dict().items()}
    print(f"\n[calibration] epsilon-sweep of lambda-hat on the GROKKED solution (looking for a stable band)")
    eps_grid = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
    sweep = []
    for eps in eps_grid:
        lam, sd, div, Lstar = sgld_llc(model, w_final, xtr, ytr, eps)
        sweep.append((eps, lam, sd, div))
        print(f"  eps={eps:.0e}  lambda-hat={lam:7.2f} +- {sd:5.2f}  (chains diverged: {div}/4,  L*={Lstar:.4f})")
    # stable band = consecutive eps whose lambda-hat agree within tol
    vals = [(e, l) for e, l, _, d in sweep if math.isfinite(l) and d == 0]
    stable, tol = None, 0.15
    for i in range(len(vals) - 1):
        e0, l0 = vals[i]; e1, l1 = vals[i + 1]
        if l0 > 0 and abs(l1 - l0) / l0 < tol:
            stable = (e0, e1, (l0 + l1) / 2); break
    if stable:
        print(f"  -> STABLE band: eps in [{stable[0]:.0e},{stable[1]:.0e}], lambda-hat ~ {stable[2]:.2f}  "
              f"(CV<{tol}) -> NODE 1 PASSES: estimator calibrated.")
    else:
        print(f"  -> NO stable band found -> NODE 1 FAILS the calibration gate; widen grid / tune gamma,burnin "
              f"before building node 2.")

    # lambda-hat across training snapshots (memorization vs generalization), at the stable eps if found
    eps_use = stable[1] if stable else 3e-5
    print(f"\n[trajectory] lambda-hat at snapshots (eps={eps_use:.0e}) -- does it move across the grok transition?")
    traj = []
    for s in sorted(snaps):
        lam, sd, div, _ = sgld_llc(model, snaps[s], xtr, ytr, eps_use)
        te = next((a for st, a in zip(hist["step"], hist["te_acc"]) if st >= s), float("nan"))
        traj.append((s, lam, sd, te))
        print(f"  step {s:6d}  test_acc~{te:.2f}  lambda-hat={lam:7.2f} +- {sd:5.2f}")
    model.load_state_dict(w_final)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(hist["step"], hist["tr_acc"], color=GREEN, label="train")
    ax[0].plot(hist["step"], hist["te_acc"], color=BLUE, label="test")
    if grok_step:
        ax[0].axvline(grok_step, color=RED, ls=":", lw=1, label=f"grok @ {grok_step}")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("accuracy"); ax[0].set_title("grokking (modular add mod p)", color=NAVY)
    ax[0].legend(frameon=False, fontsize=8)
    es = [e for e, l, _, d in sweep if math.isfinite(l)]
    ls = [l for e, l, _, d in sweep if math.isfinite(l)]
    ax[1].semilogx(es, ls, "o-", color=NAVY)
    ax[1].set_xlabel("SGLD step size eps"); ax[1].set_ylabel("lambda-hat")
    ax[1].set_title("LLC calibration: stable band = trustworthy estimator", color=NAVY)
    fig.tight_layout(); fig.savefig("slt/node1_grok_llc.png", dpi=160); plt.close(fig)
    print("  wrote slt/node1_grok_llc.png")


if __name__ == "__main__":
    model, snaps, hist, data, grok_step = train()
    if hist["tr_acc"][-1] < 0.99 or (grok_step is None):
        print("\n[node1] grokking did NOT reproduce (train<0.99 or no test>0.9). Fix setup before LLC.")
    else:
        calibrate(model, snaps, data, hist, grok_step)
