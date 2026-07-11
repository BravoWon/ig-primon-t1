#!/usr/bin/env python
"""NODE 1 v3 -- the money measurement: lambda-hat trajectory across the grok, on the CALIBRATED instrument.

v2 found a stationary, gamma-robust plateau at eps=3e-6 (lambda-hat(grokked) ~ 15.5). Now we (1) confirm
that plateau with a FINER (eps,gamma) sweep, then (2) measure lambda-hat at DENSE snapshots spanning the
grok transition -- with error bars -- at the calibrated setting. Only now can we honestly test the parked
hypothesis: does lambda-hat DROP at the transition (SLT's prediction), or was the v1 ~10x drop a sampler
artifact? The instrument is calibrated; the number is allowed to say no.

    python slt/node1_v3_trajectory.py
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from node1_grok_llc import Transformer, make_data, loss_acc, DEV

WD, LR, STEPS, SEED = 1.0, 1e-3, 30000, 0
SNAP = [3000, 8000, 14000, 17000, 19000, 20000, 21000, 22000, 25000, 29999]
EPS_CAL, GAMMA_CAL = 3e-6, 100.0
EPS_FINE = [1e-6, 2e-6, 3e-6, 5e-6, 1e-5]
GAMMA_FINE = [30.0, 60.0, 100.0, 200.0, 300.0]
NAVY, GREEN, RED, BLUE = "#15293f", "#1e7d34", "#c0392b", "#2c6fbb"


def sgld(model, w_star, xtr, ytr, eps, gamma, chains, draws=300, burnin=100):
    n = len(xtr); beta = 1.0 / math.log(n); keys = list(w_star)
    with torch.no_grad():
        L_star = F.cross_entropy(model(xtr), ytr).item()
    lams, drifts, div = [], [], 0
    for c in range(chains):
        model.load_state_dict(w_star); torch.manual_seed(2000 + c)
        tr = []
        for t in range(burnin + draws):
            L = F.cross_entropy(model(xtr), ytr); model.zero_grad(); L.backward()
            with torch.no_grad():
                for k, prm in zip(keys, model.parameters()):
                    g = n * beta * prm.grad + gamma * (prm - w_star[k])
                    prm.add_(-0.5 * eps * g + math.sqrt(eps) * torch.randn_like(prm))
            if t >= burnin:
                with torch.no_grad():
                    tr.append(F.cross_entropy(model(xtr), ytr).item())
        tr = np.array(tr)
        if not np.all(np.isfinite(tr)) or tr.mean() > 5 * L_star + 10:
            div += 1; continue
        lams.append(n * beta * (tr.mean() - L_star))
        h = len(tr) // 2; drifts.append(abs(tr[h:].mean() - tr[:h].mean()) / (tr.mean() + 1e-9))
    model.load_state_dict(w_star)
    return (np.mean(lams) if lams else float("nan"),
            np.std(lams) if lams else float("nan"),
            np.mean(drifts) if drifts else float("nan"), div)


def main():
    torch.manual_seed(SEED)
    xtr, ytr, xte, yte = make_data()
    model = Transformer().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.98))
    snaps, hist = {}, {"step": [], "te": []}
    print(f"[v3] training (capturing {len(SNAP)} snapshots across the transition)...")
    for step in range(STEPS):
        model.train(); l, _ = loss_acc(model, xtr, ytr)
        opt.zero_grad(); l.backward(); opt.step()
        if step in SNAP:
            snaps[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step % 1000 == 0 or step == STEPS - 1:
            model.eval()
            with torch.no_grad():
                _, ate = loss_acc(model, xte, yte)
            hist["step"].append(step); hist["te"].append(ate)
    grok = next((s for s, a in zip(hist["step"], hist["te"]) if a > 0.9), None)
    print(f"  grok (test>0.9) at step {grok}; final test_acc {hist['te'][-1]:.3f}")
    w_grok = snaps[SNAP[-1]]

    # (1) finer confirmation sweep on the grokked solution
    print(f"\n[v3] finer plateau-confirmation sweep on grokked weights (valid = stationary, drift<0.15):")
    print(f"{'gamma\\eps':>10}" + "".join(f"{e:>10.0e}" for e in EPS_FINE))
    valid_lams = []
    for g in GAMMA_FINE:
        row = []
        for e in EPS_FINE:
            lam, sd, drift, dv = sgld(model, w_grok, xtr, ytr, e, g, chains=4, draws=200)
            ok = dv == 0 and math.isfinite(drift) and drift < 0.15
            if ok:
                valid_lams.append(lam)
            row.append(f"{lam:8.1f}{'*' if ok else '.'}" if math.isfinite(lam) else "    nan ")
        print(f"{g:>10.0f}" + "".join(f"{c:>10}" for c in row))
    if valid_lams:
        print(f"  -> valid-cell lambda-hat: median {np.median(valid_lams):.2f}  "
              f"IQR [{np.percentile(valid_lams,25):.1f}, {np.percentile(valid_lams,75):.1f}]  "
              f"(n={len(valid_lams)} stationary cells)")

    # (2) calibrated trajectory across the transition
    print(f"\n[v3] CALIBRATED lambda-hat trajectory (eps={EPS_CAL:.0e}, gamma={GAMMA_CAL:.0f}, 6 chains):")
    print(f"  {'step':>7}{'test_acc':>10}{'lambda-hat':>14}{'drift':>8}")
    traj = []
    for s in SNAP:
        lam, sd, drift, dv = sgld(model, snaps[s], xtr, ytr, EPS_CAL, GAMMA_CAL, chains=6, draws=300)
        te = next((a for st, a in zip(hist["step"], hist["te"]) if st >= s), hist["te"][-1])
        traj.append((s, te, lam, sd))
        print(f"  {s:>7}{te:>10.3f}{lam:>9.2f} +-{sd:>4.2f}{drift:>8.3f}")

    pre = [l for s, te, l, sd in traj if te < 0.5 and math.isfinite(l)]
    post = [l for s, te, l, sd in traj if te > 0.95 and math.isfinite(l)]
    if pre and post:
        drop = (np.mean(pre) - np.mean(post)) / np.mean(pre) * 100
        print(f"\n  pre-grok lambda-hat ~ {np.mean(pre):.1f}   post-grok ~ {np.mean(post):.1f}   "
              f"=> {drop:+.0f}% change across the transition")
        print(f"  -> {'lambda-hat DROPS at grok (SLT-consistent) -- now on a CALIBRATED instrument' if drop > 15 else 'no clean drop on the calibrated instrument -- v1 10x was largely sampler artifact' if abs(drop)<15 else 'lambda-hat RISES (against SLT prediction)'}.")

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    ax1.plot(hist["step"], hist["te"], color=BLUE, lw=1.5, label="test acc")
    if grok:
        ax1.axvline(grok, color="#bbb", ls=":", lw=1)
    ax1.set_xlabel("step"); ax1.set_ylabel("test accuracy", color=BLUE); ax1.set_ylim(-0.02, 1.05)
    ax2 = ax1.twinx()
    ss = [s for s, te, l, sd in traj if math.isfinite(l)]
    ls = [l for s, te, l, sd in traj if math.isfinite(l)]
    es = [sd for s, te, l, sd in traj if math.isfinite(l)]
    ax2.errorbar(ss, ls, yerr=es, fmt="o-", color=RED, capsize=3, label="lambda-hat (calibrated)")
    ax2.set_ylabel("lambda-hat (calibrated, eps=3e-6)", color=RED)
    ax1.set_title("Node 1 v3: calibrated lambda-hat across the grok transition", color=NAVY, fontsize=10.5)
    fig.tight_layout(); fig.savefig("slt/node1_v3_trajectory.png", dpi=160); plt.close(fig)
    print("  wrote slt/node1_v3_trajectory.png")


if __name__ == "__main__":
    main()
