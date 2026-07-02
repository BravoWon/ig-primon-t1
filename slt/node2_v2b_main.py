#!/usr/bin/env python
"""NODE 2 v2 -- PHASE 2: phase-aligned (T_grok + Delta) lambda-hat, 4-arm isolation.

Fixes the time-in-basin confound of a fixed endpoint: VIB groks ~8k steps earlier, so a fixed-30k readout
would give it ~2x the post-grok SGD diffusion to settle flatter -- confounding intrinsic geometry with
"got there earlier." Fix: measure lambda-hat at T_grok + Delta (Delta=5000) for EVERY arm, so all get the
same post-grok maturation budget. Dynamic asymmetry -> T_grok; geometric asymmetry -> lambda-hat.

Arms (determinism locked; hyperparameters LOCKED from the blind Phase-1 calibration):
  baseline | vib (alpha=1e-3) | noise (sigma=0.05, no KL) | shrink (alpha=3e-5, deterministic ||mu||^2)
Covariates at T_grok+Delta (kill trivial counter-arguments): train-loss, weight-L2, representation-norm.
lambda-hat is BLIND (hyperparameters frozen before any lambda-hat is computed).

    python slt/node2_v2b_main.py
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from node1_grok_llc import make_data, DEV, P, D_MODEL, N_HEADS, D_HEAD, D_MLP

DELTA, STEPS_MAX, SEEDS = 5000, 34000, [0, 1, 2]
ARMS = {"baseline": ("baseline", 0.0), "vib": ("vib", 1e-3), "noise": ("noise", 0.05), "shrink": ("shrink", 3e-5)}
EPS_CAL, GAMMA_CAL, CHAINS, DRAWS, BURNIN = 3e-6, 100.0, 4, 120, 80


def set_det(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class TF(nn.Module):
    def __init__(self, mode="baseline", sigma=0.05):
        super().__init__(); self.mode = mode; self.sigma = sigma
        self.embed = nn.Embedding(P + 1, D_MODEL)
        self.pos = nn.Parameter(torch.randn(3, D_MODEL) / math.sqrt(D_MODEL))
        self.Wq, self.Wk, self.Wv = (nn.Linear(D_MODEL, N_HEADS * D_HEAD, bias=False) for _ in range(3))
        self.Wo = nn.Linear(N_HEADS * D_HEAD, D_MODEL, bias=False)
        self.Win, self.Wout = nn.Linear(D_MODEL, D_MLP, bias=False), nn.Linear(D_MLP, D_MODEL, bias=False)
        if mode in ("vib", "noise", "shrink"):
            self.mu = nn.Linear(D_MODEL, D_MODEL, bias=False)
            self.logvar = nn.Linear(D_MODEL, D_MODEL, bias=False)         # param-matched across bottleneck arms
        self.unembed = nn.Linear(D_MODEL, P, bias=False)

    def forward(self, x):
        B = x.shape[0]
        h = self.embed(x) + self.pos[None]
        q, k, v = (W(h).view(B, 3, N_HEADS, D_HEAD).transpose(1, 2) for W in (self.Wq, self.Wk, self.Wv))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D_HEAD)
        m = torch.triu(torch.ones(3, 3, device=x.device), 1).bool()
        att = att.masked_fill(m, float("-inf")).softmax(-1)
        z = (att @ v).transpose(1, 2).reshape(B, 3, -1)
        h = h + self.Wo(z); h = h + self.Wout(F.relu(self.Win(h)))
        r = h[:, -1]; pen = torch.zeros((), device=x.device)
        if self.mode == "vib":
            mu, lv = self.mu(r), self.logvar(r)
            r = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
            pen = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
        elif self.mode == "noise":
            mu = self.mu(r); r = mu + self.sigma * torch.randn_like(mu) if self.training else mu
        elif self.mode == "shrink":
            mu = self.mu(r); r = mu; pen = mu.pow(2).mean()
        return self.unembed(r), pen, r


def sgld_llc(model, w_star, xtr, ytr):
    n = len(xtr); beta = 1.0 / math.log(n); keys = list(w_star); model.eval()
    with torch.no_grad():
        L_star = F.cross_entropy(model(xtr)[0], ytr).item()
    lams = []
    for c in range(CHAINS):
        model.load_state_dict(w_star); torch.manual_seed(7000 + c)
        tr = []
        for t in range(BURNIN + DRAWS):
            L = F.cross_entropy(model(xtr)[0], ytr); model.zero_grad(); L.backward()
            with torch.no_grad():
                for kk, prm in zip(keys, model.parameters()):
                    gr = prm.grad if prm.grad is not None else torch.zeros_like(prm)
                    prm.add_(-0.5 * EPS_CAL * (n * beta * gr + GAMMA_CAL * (prm - w_star[kk]))
                             + math.sqrt(EPS_CAL) * torch.randn_like(prm))
            if t >= BURNIN:
                with torch.no_grad():
                    tr.append(F.cross_entropy(model(xtr)[0], ytr).item())
        lams.append(n * beta * (np.mean(tr) - L_star))
    model.load_state_dict(w_star)
    return float(np.mean(lams))


def run(mode, hp, seed):
    set_det(seed)
    xtr, ytr, xte, yte = make_data()
    model = TF(mode, sigma=hp if mode == "noise" else 0.05).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    alpha = hp if mode in ("vib", "shrink") else 0.0
    grok, target, snap = None, None, None
    for step in range(STEPS_MAX):
        model.train(); logits, pen, _ = model(xtr)
        (F.cross_entropy(logits, ytr) + alpha * pen).backward(); opt.step(); opt.zero_grad()
        if grok is None and step % 250 == 0:
            model.eval()
            with torch.no_grad():
                ate = (model(xte)[0].argmax(-1) == yte).float().mean().item()
            if ate > 0.9:
                grok = step; target = min(step + DELTA, STEPS_MAX - 1)
        if target is not None and step == target:
            snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
            break
    if snap is None:
        return dict(mode=mode, seed=seed, grok=None, te=0.0, lam=float("nan"),
                    trloss=float("nan"), wL2=float("nan"), rnorm=float("nan"))
    model.load_state_dict(snap); model.eval()
    with torch.no_grad():
        logits, _, r = model(xtr); trloss = F.cross_entropy(logits, ytr).item()
        te = (model(xte)[0].argmax(-1) == yte).float().mean().item()
        wL2 = float(math.sqrt(sum(float(p.pow(2).sum()) for p in model.parameters())))
        rnorm = float(r.norm(dim=-1).mean())
    lam = sgld_llc(model, snap, xtr, ytr)                                # BLIND readout at T_grok+Delta
    return dict(mode=mode, seed=seed, grok=grok, te=te, lam=lam, trloss=trloss, wL2=wL2, rnorm=rnorm)


def main():
    print(f"[node2-v2b] phase-aligned T_grok+Delta (Delta={DELTA}); locked hyperparams; blind lambda-hat")
    rows = []
    for name, (mode, hp) in ARMS.items():
        for seed in SEEDS:
            r = run(mode, hp, seed); rows.append((name, r))
            print(f"  {name:9} seed {seed}  grok@{str(r['grok']):>6}  te {r['te']:.2f}  "
                  f"lambda-hat {r['lam']:7.2f}  trloss {r['trloss']:.1e}  wL2 {r['wL2']:.1f}  rnorm {r['rnorm']:.2f}")

    def agg(name, key):
        vals = [r[key] for n, r in rows if n == name and r[key] is not None and (not isinstance(r[key], float) or math.isfinite(r[key]))]
        return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))

    print(f"\n  {'arm':9}{'T_grok':>9}{'lambda-hat':>14}{'wL2':>9}{'rnorm':>9}{'grokked/3':>11}")
    summ = {}
    for name in ARMS:
        gm = agg(name, "grok"); lm = agg(name, "lam"); wm = agg(name, "wL2"); rm = agg(name, "rnorm")
        ng = sum(1 for n, r in rows if n == name and r["grok"] is not None)
        summ[name] = dict(grok=gm[0], lam=lm[0], lam_sd=lm[1], wL2=wm[0], rnorm=rm[0], ng=ng)
        print(f"  {name:9}{gm[0]:>9.0f}{lm[0]:>9.1f}+-{lm[1]:<3.1f}{wm[0]:>9.1f}{rm[0]:>9.2f}{f'{ng}/3':>11}")

    b = summ["baseline"]
    print(f"\n  DECISIVE (phase-aligned lambda-hat vs baseline, matched post-grok maturation):")
    for name in ("vib", "noise", "shrink"):
        s = summ[name]
        if s["ng"] < 2 or not math.isfinite(s["lam"]):
            print(f"    {name:8}: insufficient grokked seeds ({s['ng']}/3) -- excluded"); continue
        dlam = (s["lam"] - b["lam"]) / b["lam"] * 100
        dw = (s["wL2"] - b["wL2"]) / b["wL2"] * 100
        dr = (s["rnorm"] - b["rnorm"]) / b["rnorm"] * 100
        print(f"    {name:8}: lambda-hat {dlam:+6.1f}%  | covariates weight-L2 {dw:+.1f}%, repr-norm {dr:+.1f}%")
    vib, shr, noi = summ["vib"], summ["shrink"], summ["noise"]
    if math.isfinite(vib["lam"]) and math.isfinite(shr["lam"]):
        beats_shrink = vib["lam"] < shr["lam"] * 0.85
        beats_noise = (not math.isfinite(noi["lam"])) or vib["lam"] < noi["lam"] * 0.85
        print(f"\n  -> vib lambda {vib['lam']:.1f} vs shrink {shr['lam']:.1f} vs noise {noi['lam']:.1f} (baseline {b['lam']:.1f})")
        if beats_shrink and beats_noise:
            print(f"     VIB's lambda-hat is lower than both controls at MATCHED maturation. Check covariates:")
            print(f"     if vib's weight-L2/repr-norm are NOT proportionally smaller, the geometry is IB-specific -> node 3.")
        else:
            print(f"     VIB does NOT clearly beat the controls at matched maturation -> the fixed-endpoint drop was")
            print(f"     largely time-in-basin; the IB lever on lambda-hat is not established. (Dynamic win via T_grok stands.)")


if __name__ == "__main__":
    main()
