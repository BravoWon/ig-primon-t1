#!/usr/bin/env python
"""NODE 2: per-layer information bottleneck -- does it move lambda-hat BEYOND generic compression?

Spec node 2: "+ bottleneck only. Does per-layer IB change grok-time / lambda-hat / OOD?" Plus the
architectural fix (note #4): an IB term can lower lambda-hat simply by REGULARIZING. So the decisive
test is not "does VIB move lambda-hat" but "does VIB move it BEYOND a matched generic-compression control."

Three arms (determinism locked; relative-shift instrument from the node-1 gate, CV 3.5%):
  baseline : vanilla grok.
  vib      : stochastic bottleneck Z=mu+sigma*eps on the residual stream + VIB KL penalty alpha*KL
             (KL(N(mu,sigma)||N(0,I)) upper-bounds I(X;Z); task loss preserves I(Z;Y)). The real IB.
  shrink   : SAME stochastic-Z architecture, KL replaced by naive alpha2*||mu||^2 -- generic compression,
             not the information-theoretic objective. The control.

Readout: post-grok lambda-hat SHIFT vs baseline, per seed -> mean +- CV. Decisive: vib-shift vs shrink-shift.
Also grok-step and final test acc (does IB delay/prevent grok?).

    python slt/node2_bottleneck.py
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from node1_grok_llc import make_data, DEV, P, D_MODEL, N_HEADS, D_HEAD, D_MLP

STEPS, PRE, POST, SEEDS = 32000, 14000, 31999, [0, 1, 2]
ALPHA_VIB, ALPHA_SHRINK = 1e-3, 1e-3            # bottleneck strengths (approx matched; refine in v2)
EPS_CAL, GAMMA_CAL, CHAINS, DRAWS, BURNIN = 3e-6, 100.0, 4, 120, 80


def set_det(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class TF(nn.Module):
    """node-1 transformer + an optional bottleneck on the read-off representation."""
    def __init__(self, mode="baseline"):
        super().__init__(); self.mode = mode
        self.embed = nn.Embedding(P + 1, D_MODEL)
        self.pos = nn.Parameter(torch.randn(3, D_MODEL) / math.sqrt(D_MODEL))
        self.Wq, self.Wk, self.Wv = (nn.Linear(D_MODEL, N_HEADS * D_HEAD, bias=False) for _ in range(3))
        self.Wo = nn.Linear(N_HEADS * D_HEAD, D_MODEL, bias=False)
        self.Win, self.Wout = nn.Linear(D_MODEL, D_MLP, bias=False), nn.Linear(D_MLP, D_MODEL, bias=False)
        if mode in ("vib", "shrink"):
            self.mu = nn.Linear(D_MODEL, D_MODEL, bias=False)
            self.logvar = nn.Linear(D_MODEL, D_MODEL, bias=False)   # present in both -> param-matched control
        self.unembed = nn.Linear(D_MODEL, P, bias=False)

    def forward(self, x):
        B = x.shape[0]
        h = self.embed(x) + self.pos[None]
        q, k, v = (W(h).view(B, 3, N_HEADS, D_HEAD).transpose(1, 2) for W in (self.Wq, self.Wk, self.Wv))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D_HEAD)
        mask = torch.triu(torch.ones(3, 3, device=x.device), 1).bool()
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        z = (att @ v).transpose(1, 2).reshape(B, 3, -1)
        h = h + self.Wo(z)
        h = h + self.Wout(F.relu(self.Win(h)))
        r = h[:, -1]                                            # read-off at '='
        pen = torch.zeros((), device=x.device)
        if self.mode == "vib":
            mu, logvar = self.mu(r), self.logvar(r)
            r = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
            pen = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()      # KL upper-bounds I(X;Z)
        elif self.mode == "shrink":
            mu = self.mu(r); r = mu
            pen = mu.pow(2).mean()                                            # naive shrinkage (control)
        return self.unembed(r), pen


def loss_acc(model, x, y, alpha=0.0):
    logits, pen = model(x)
    return F.cross_entropy(logits, y) + alpha * pen, (logits.argmax(-1) == y).float().mean().item()


def train(mode, seed):
    set_det(seed)
    xtr, ytr, xte, yte = make_data()
    model = TF(mode).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    alpha = {"baseline": 0.0, "vib": ALPHA_VIB, "shrink": ALPHA_SHRINK}[mode]
    snaps, grok = {}, None
    for step in range(STEPS):
        model.train(); l, _ = loss_acc(model, xtr, ytr, alpha); opt.zero_grad(); l.backward(); opt.step()
        if step in (PRE, POST):
            snaps[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if grok is None and step % 500 == 0:
            model.eval()
            with torch.no_grad():
                _, ate = loss_acc(model, xte, yte)
            if ate > 0.9:
                grok = step
    model.eval()
    with torch.no_grad():
        _, te = loss_acc(model, xte, yte)
    return model, snaps, grok, float(te), (xtr, ytr)


def sgld_llc(model, w_star, xtr, ytr):
    n = len(xtr); beta = 1.0 / math.log(n); keys = list(w_star)
    model.eval()
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
                    gr = prm.grad if prm.grad is not None else torch.zeros_like(prm)  # task-flat params: localize only
                    g = n * beta * gr + GAMMA_CAL * (prm - w_star[kk])
                    prm.add_(-0.5 * EPS_CAL * g + math.sqrt(EPS_CAL) * torch.randn_like(prm))
            if t >= BURNIN:
                with torch.no_grad():
                    tr.append(F.cross_entropy(model(xtr)[0], ytr).item())
        lams.append(n * beta * (np.mean(tr) - L_star))
    model.load_state_dict(w_star)
    return float(np.mean(lams))


def main():
    print(f"[node2] determinism locked; per-layer bottleneck vs matched shrinkage control; relative lambda-hat shift")
    res = {m: [] for m in ("baseline", "vib", "shrink")}
    grok_te = {m: [] for m in res}
    for seed in SEEDS:
        for mode in ("baseline", "vib", "shrink"):
            model, snaps, grok, te, (xtr, ytr) = train(mode, seed)
            lam = sgld_llc(model, snaps[POST], xtr, ytr)
            res[mode].append(lam); grok_te[mode].append((grok, te))
            print(f"  seed {seed}  {mode:8}  grok@{str(grok):>6}  test_acc {te:.3f}  lambda-hat(post) {lam:.1f}")
    print(f"\n  {'arm':10}{'mean lambda':>12}{'vs baseline (shift)':>22}{'grok/te':>16}")
    base = np.array(res["baseline"])
    for m in ("baseline", "vib", "shrink"):
        arm = np.array(res[m])
        shifts = (arm - base) / base                                # per-seed relative shift vs baseline
        groks = [g for g, t in grok_te[m]]; tes = [t for g, t in grok_te[m]]
        if m == "baseline":
            print(f"  {m:10}{arm.mean():>12.1f}{'(reference)':>22}{f'{np.mean([g or 0 for g in groks]):.0f}/{np.mean(tes):.2f}':>16}")
        else:
            cv = np.std(shifts) / abs(np.mean(shifts)) if np.mean(shifts) != 0 else float("inf")
            print(f"  {m:10}{arm.mean():>12.1f}{f'{np.mean(shifts)*100:+.1f}% (CV {cv*100:.0f}%)':>22}"
                  f"{f'{np.mean([g or 0 for g in groks]):.0f}/{np.mean(tes):.2f}':>16}")
    vib_shift = np.mean((np.array(res["vib"]) - base) / base)
    shr_shift = np.mean((np.array(res["shrink"]) - base) / base)
    print(f"\n  DECISIVE: vib shift {vib_shift*100:+.1f}%  vs  shrink(control) shift {shr_shift*100:+.1f}%")
    diff = vib_shift - shr_shift
    if abs(diff) < 0.10:
        print(f"  -> VIB ~= shrink ({diff*100:+.1f}%): the IB effect on lambda-hat is GENERIC COMPRESSION, not")
        print(f"     IB-specific. Node 2 result: bottleneck is regularization in a costume (occupied territory).")
    else:
        print(f"  -> VIB differs from shrink by {diff*100:+.1f}%: the information-theoretic objective moves")
        print(f"     lambda-hat BEYOND generic compression. Node 2 finding: per-layer IB is a real lever. ->node 3.")


if __name__ == "__main__":
    main()
