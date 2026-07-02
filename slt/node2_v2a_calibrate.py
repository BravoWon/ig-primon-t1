#!/usr/bin/env python
"""NODE 2 v2 -- PHASE 1: BLIND grok-time calibration of the controls (no lambda-hat measured).

Goal: tune alpha (shrink) and sigma (noise) so each control groks in the SAME temporal band as vib
(~15k), so the node-2-v2 lambda-hat comparison is at MATCHED generalization pressure -- not confounded
by different learning dynamics. Tuned BLIND to lambda-hat (grok-time only) to avoid selecting a control
that says what we want. Pre-registered band = [12000, 19000] steps.

Escape hatch: if a control can't be made to grok IN-BAND (groks like baseline ~20k, or falls off the
cliff to no-grok), that asymmetry is itself the finding -- we do NOT force a mismatched control.

    python slt/node2_v2a_calibrate.py
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from node1_grok_llc import make_data, DEV, P, D_MODEL, N_HEADS, D_HEAD, D_MLP

STEPS, SEED = 30000, 0
BAND = (12000, 19000)
SHRINK_A = [3e-5, 1e-4, 2e-4, 3e-4]
NOISE_S = [0.05, 0.10, 0.15, 0.20]
VIB_A = 1e-3


def set_det(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class TF(nn.Module):
    def __init__(self, mode="baseline", sigma=0.1):
        super().__init__(); self.mode = mode; self.sigma = sigma
        self.embed = nn.Embedding(P + 1, D_MODEL)
        self.pos = nn.Parameter(torch.randn(3, D_MODEL) / math.sqrt(D_MODEL))
        self.Wq, self.Wk, self.Wv = (nn.Linear(D_MODEL, N_HEADS * D_HEAD, bias=False) for _ in range(3))
        self.Wo = nn.Linear(N_HEADS * D_HEAD, D_MODEL, bias=False)
        self.Win, self.Wout = nn.Linear(D_MODEL, D_MLP, bias=False), nn.Linear(D_MLP, D_MODEL, bias=False)
        if mode in ("vib", "noise", "shrink"):
            self.mu = nn.Linear(D_MODEL, D_MODEL, bias=False)
            self.logvar = nn.Linear(D_MODEL, D_MODEL, bias=False)   # present in all bottleneck arms (param-match)
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
            mu = self.mu(r)
            r = mu + self.sigma * torch.randn_like(mu) if self.training else mu   # FIXED sigma, no KL
        elif self.mode == "shrink":
            mu = self.mu(r); r = mu; pen = mu.pow(2).mean()
        return self.unembed(r), pen


def grok_time(mode, hp, seed=SEED):
    set_det(seed)
    xtr, ytr, xte, yte = make_data()
    model = TF(mode, sigma=hp if mode == "noise" else 0.1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    alpha = hp if mode in ("vib", "shrink") else 0.0
    grok = None
    for step in range(STEPS):
        model.train(); logits, pen = model(xtr)
        (F.cross_entropy(logits, ytr) + alpha * pen).backward(); opt.step(); opt.zero_grad()
        if grok is None and step % 250 == 0:
            model.eval()
            with torch.no_grad():
                ate = (model(xte)[0].argmax(-1) == yte).float().mean().item()
            if ate > 0.9:
                grok = step
    model.eval()
    with torch.no_grad():
        te = (model(xte)[0].argmax(-1) == yte).float().mean().item()
    return grok, te


def pick(cands, target):
    inband = [(hp, g) for hp, g, te in cands if g is not None and BAND[0] <= g <= BAND[1]]
    if inband:
        return min(inband, key=lambda x: abs(x[1] - target))
    return None


def main():
    print(f"[node2-v2a] BLIND grok-time calibration (seed {SEED}); pre-registered band {BAND}")
    base_g, base_te = grok_time("baseline", None)
    vib_g, vib_te = grok_time("vib", VIB_A)
    print(f"  anchors:  baseline grok@{base_g} (te {base_te:.2f})   vib(a={VIB_A}) grok@{vib_g} (te {vib_te:.2f})")
    target = vib_g if vib_g else 15000

    print(f"\n  shrink sweep (alpha):")
    shrink = []
    for a in SHRINK_A:
        g, te = grok_time("shrink", a); shrink.append((a, g, te))
        print(f"    alpha={a:.0e}  grok@{str(g):>6}  final_te {te:.2f}  {'[in band]' if g and BAND[0]<=g<=BAND[1] else ''}")
    print(f"\n  noise sweep (sigma):")
    noise = []
    for s in NOISE_S:
        g, te = grok_time("noise", s); noise.append((s, g, te))
        print(f"    sigma={s:.2f}  grok@{str(g):>6}  final_te {te:.2f}  {'[in band]' if g and BAND[0]<=g<=BAND[1] else ''}")

    ps, pn = pick(shrink, target), pick(noise, target)
    print(f"\n  target grok-time ~ {target} (vib); band {BAND}")
    print(f"  -> shrink: {f'alpha={ps[0]:.0e} groks@{ps[1]} (LOCKED)' if ps else 'NO in-band alpha -- shrink cannot be grok-matched (asymmetry = finding)'}")
    print(f"  -> noise : {f'sigma={pn[0]:.2f} groks@{pn[1]} (LOCKED)' if pn else 'NO in-band sigma -- noise cannot be grok-matched (asymmetry = finding)'}")
    print(f"\n  next: node2_v2b_main.py with these locked hyperparameters, 3 seeds, measure lambda-hat in-band only.")


if __name__ == "__main__":
    main()
