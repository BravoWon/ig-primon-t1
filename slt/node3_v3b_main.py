#!/usr/bin/env python
"""Nodes 3-4 PHASE 3b: phase-aligned lambda-hat for baseline / consistency / cocycle.

Locked from 3a: alpha=0.05, beta=0.005 (grok preserved, Lcons=Lcyc=0 satisfied).
Phase-aligned T_grok+Delta readout (Delta=5000); lambda-hat on the shared 3-head TASK loss (blind).

Covariates at T_grok+Delta:
  weight-L2, repr-norm            -- Node-2 confound (is a low lambda-hat just volume compression?)
  L_cons, L_cyc                   -- was the topology actually imposed?
  ||R_ab - I||_F, mean cos(Za,Zb) -- LOCK-1 VERIFICATION: are the maps non-trivial & the chunks distinct?
                                     (if R~=I or chunks~=identical, the cocycle is VACUOUS -- any lambda
                                      effect is an artifact, regardless of the number)

GOLDEN readout: cocycle lambda-hat << baseline WHILE repr-norm stays near baseline (NOT collapsed like the
Node-2 shrink floor of ~7.5) AND the locks verify non-trivial -> structural coherence flattens the basin
WITHOUT volume compression. Anything else (repr collapse, or trivial maps) -> not lightning.

    python slt/node3_v3b_main.py
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import math, random
import numpy as np
import torch
import torch.nn.functional as F
from node1_grok_llc import make_data, DEV
from node3_topo import TopoTF, task_loss, topo_losses, test_acc, CHUNK

ALPHA, BETA, DELTA, STEPS_MAX, SEEDS = 0.05, 0.005, 5000, 34000, [0, 1, 2]
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
        if mode in ("consistency", "cocycle"):
            lc, lcy = topo_losses(model, ch); loss = loss + ALPHA * lc + (BETA * lcy if mode == "cocycle" else 0)
        opt.zero_grad(); loss.backward(); opt.step()
        if grok is None and step % 250 == 0:
            model.eval()
            if test_acc(model, xte, yte) > 0.9:
                grok = step; target = min(step + DELTA, STEPS_MAX - 1)
        if target is not None and step == target:
            snap = {k: v.detach().clone() for k, v in model.state_dict().items()}; break
    if snap is None:
        return dict(mode=mode, seed=seed, grok=None, te=0.0, lam=float("nan"), wL2=float("nan"),
                    rnorm=float("nan"), lcons=float("nan"), lcyc=float("nan"), rmi=float("nan"), ccos=float("nan"))
    model.load_state_dict(snap); model.eval()
    with torch.no_grad():
        out, (Za, Zb, Zc) = model(xtr); trloss = task_loss(out, ytr).item(); te = test_acc(model, xte, yte)
        wL2 = math.sqrt(sum(float(p.pow(2).sum()) for p in model.parameters()))
        rnorm = float(torch.cat([Za, Zb, Zc], -1).norm(dim=-1).mean())
        lcons, lcyc = topo_losses(model, (Za, Zb, Zc))
        I = torch.eye(CHUNK, device=DEV)
        rmi = float(np.mean([ (M.weight - I).norm().item() for M in (model.Rab, model.Rbc, model.Rca) ]))
        ccos = float(np.mean([F.cosine_similarity(a, b, -1).mean().item()
                              for a, b in ((Za, Zb), (Zb, Zc), (Zc, Za))]))
    lam = sgld_llc(model, snap, xtr, ytr)
    return dict(mode=mode, seed=seed, grok=grok, te=te, lam=lam, wL2=wL2, rnorm=rnorm,
                lcons=float(lcons), lcyc=float(lcyc), rmi=rmi, ccos=ccos)


def main():
    print(f"[node3-3b] phase-aligned T_grok+Delta; alpha={ALPHA} beta={BETA}; lock-verified cocycle")
    rows = []
    for mode in ("baseline", "consistency", "cocycle"):
        for seed in SEEDS:
            r = run(mode, seed); rows.append(r)
            print(f"  {mode:12} s{seed} grok@{str(r['grok']):>6} te{r['te']:.2f} lam{r['lam']:7.2f} "
                  f"rnorm{r['rnorm']:7.1f} wL2{r['wL2']:6.1f} Lcons{r['lcons']:.3f} Lcyc{r['lcyc']:.2f} "
                  f"||R-I||{r['rmi']:.2f} chunkcos{r['ccos']:+.2f}")

    def m(mode, k):
        v = [r[k] for r in rows if r["mode"] == mode and r[k] is not None and (not isinstance(r[k], float) or math.isfinite(r[k]))]
        return np.mean(v) if v else float("nan")

    print(f"\n  {'arm':13}{'T_grok':>8}{'lambda':>9}{'rnorm':>9}{'||R-I||':>9}{'chunkcos':>10}{'grokd':>7}")
    S = {}
    for mode in ("baseline", "consistency", "cocycle"):
        ng = sum(1 for r in rows if r["mode"] == mode and r["grok"] is not None)
        S[mode] = {k: m(mode, k) for k in ("grok", "lam", "rnorm", "wL2", "rmi", "ccos")}
        s = S[mode]
        print(f"  {mode:13}{s['grok']:>8.0f}{s['lam']:>9.1f}{s['rnorm']:>9.1f}{s['rmi']:>9.2f}{s['ccos']:>+10.2f}{f'{ng}/3':>7}")

    b, co = S["baseline"], S["cocycle"]
    print(f"\n  LOCK VERIFICATION (cocycle): ||R-I||={co['rmi']:.2f}  chunk-cos={co['ccos']:+.2f}")
    locks_ok = co["rmi"] > 1.0 and co["ccos"] < 0.9
    if not locks_ok:
        print(f"  -> LOCKS FAILED (maps ~identity or chunks ~identical): the cocycle is VACUOUS. lambda-hat")
        print(f"     comparison is meaningless -- topology was trivially satisfied. Redesign Lock 1.")
        return
    dlam = (co["lam"] - b["lam"]) / b["lam"] * 100
    drn = (co["rnorm"] - b["rnorm"]) / b["rnorm"] * 100
    print(f"  locks HELD (maps non-trivial, chunks distinct). Now the golden readout:")
    print(f"  cocycle lambda {co['lam']:.1f} vs baseline {b['lam']:.1f} ({dlam:+.1f}%)  |  "
          f"repr-norm {co['rnorm']:.1f} vs {b['rnorm']:.1f} ({drn:+.1f}%)")
    if dlam < -20 and drn > -25:
        print(f"  -> LIGHTNING: cocycle FLATTENS the basin (lambda down {dlam:+.0f}%) with repr-norm PRESERVED")
        print(f"     ({drn:+.0f}%, NOT collapsed like the Node-2 shrink floor). Structural coherence lowers the")
        print(f"     learning coefficient INDEPENDENT of volume compression. The load-bearing claim survives. ->paper")
    elif dlam < -20 and drn <= -25:
        print(f"  -> cocycle lowers lambda BUT via repr-norm collapse ({drn:+.0f}%) -- same trivial compression")
        print(f"     as Node-2 shrink, in topological clothing. Not lightning.")
    else:
        print(f"  -> cocycle does NOT lower lambda-hat beyond baseline ({dlam:+.0f}%). Topological coherence is")
        print(f"     not a geometric lever on the learning coefficient here. Clean null; the cocycle idea closes.")


if __name__ == "__main__":
    main()
