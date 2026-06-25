#!/usr/bin/env python
"""structure-vs-bits v4 — strong-baseline gate + the SHEAF-SIGNAL test (does the sheaf earn its name?).

Strong baseline: group-wise 4-bit (per-row, 128-col groups) -- real quant, not weak per-column RTN.
At EQUAL avg ~4.0 b/param (20% input-channels @ 8-bit, 80% @ 3-bit), allocate the salient channels by:
  magnitude   : ||W[:,j]||              (control)
  act-energy  : C_jj = E[x_j^2]         (AWQ diagonal -- what v2/v3 used)
  sheaf-spec  : sum_{top-K eigs of C=XtX} lambda_i * V[j,i]^2   (participation in the dominant
                CORRELATED modes -- the off-diagonal transport structure the diagonal cannot see)
Measure full-model held-out perplexity. Two verdicts:
  (Q1) does any allocation beat group-wise uniform-4 on a STRONG base?  (real lever vs patched-RTN)
  (Q2) does sheaf-spec beat act-energy?  (the sheaf signal vs AWQ-in-a-cape)

    python compress_v4_sheafsignal.py [hf_model_id]
"""
import sys
import glob
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GROUP, FRAC, HI, LO, KEIG = 128, 0.20, 8, 3, 16


def load_corpus():
    files = sorted(glob.glob("*.md")) + sorted(glob.glob("cbp_wolfcamp/*.md")) + sorted(glob.glob("*.py"))
    blobs = [open(f, encoding="utf-8", errors="ignore").read() for f in files]
    blobs = [b for b in blobs if len(b) > 200]
    full = "\n\n".join(blobs); n = len(full)
    return full[:int(n * 0.45)], full[int(n * 0.50):]


def gw_quant(W, bits_col, group=GROUP):
    Wq = torch.empty_like(W)
    for b in bits_col.unique().tolist():
        mask = bits_col == b; qmax = 2 ** (int(b) - 1) - 1
        sub = W[:, mask]; subq = sub.clone()
        for j0 in range(0, sub.shape[1], group):
            blk = sub[:, j0:j0 + group]; s = blk.abs().amax(1, keepdim=True) / qmax + 1e-12
            subq[:, j0:j0 + group] = torch.clamp(torch.round(blk / s), -qmax, qmax) * s
        Wq[:, mask] = subq
    return Wq


def topk_idx(score, frac=FRAC):
    return torch.argsort(score, descending=True)[:round(frac * score.numel())]


def main():
    print(f"[sheaf-signal v4]  model={MODEL}  dev={DEV}  group={GROUP} salient={int(FRAC*100)}% hi/lo={HI}/{LO}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    calib, test = load_corpus()
    targets = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and min(m.weight.shape) >= 256
               and any(t in n for t in ("mlp", "proj"))}

    # --- calibration: accumulate channel Hessian C = X^T X per layer (on CPU) ---
    C = {}
    def mk(nm):
        def hook(mod, inp):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            g = (x.t() @ x).cpu()
            C[nm] = C.get(nm, torch.zeros_like(g)) + g
        return hook
    handles = [m.register_forward_pre_hook(mk(n)) for n, m in targets.items()]
    ids_c = tok(calib, return_tensors="pt").input_ids[0]
    with torch.no_grad():
        for i in range(0, min(len(ids_c), 512 * 8), 512):
            model(ids_c[i:i + 512].unsqueeze(0).to(DEV))
    for h in handles:
        h.remove()

    # --- per-layer saliency -> salient column indices for each signal ---
    idx = {"magnitude": {}, "act-energy": {}, "sheaf-spec": {}}
    for n, m in targets.items():
        W = m.weight.detach().float()
        Cl = C[n].to(DEV)
        ev, V = torch.linalg.eigh(Cl)                       # ascending
        K = min(KEIG, ev.numel())
        sheaf_sal = ((V[:, -K:] ** 2) * ev[-K:]).sum(1)     # top-K correlated-mode participation
        idx["magnitude"][n] = topk_idx(W.norm(dim=0).to(DEV))
        idx["act-energy"][n] = topk_idx(torch.diagonal(Cl))
        idx["sheaf-spec"][n] = topk_idx(sheaf_sal)
        del Cl
    C.clear()
    overlap = np.mean([len(np.intersect1d(idx["act-energy"][n].cpu(), idx["sheaf-spec"][n].cpu()))
                       / len(idx["act-energy"][n]) for n in targets])
    print(f"  saliency overlap act-energy vs sheaf-spec: {overlap*100:.0f}% "
          f"(<100% => sheaf picks DIFFERENT channels than the diagonal)")

    orig = {n: m.weight.detach().clone().cpu() for n, m in targets.items()}

    def apply(scheme):
        for n, m in targets.items():
            W = orig[n].to(DEV); ncol = W.shape[1]
            if scheme == "fp16":
                Wq = W
            elif scheme == "gw-uniform4":
                Wq = gw_quant(W, torch.full((ncol,), 4, dtype=torch.long, device=DEV))
            else:
                bits = torch.full((ncol,), LO, dtype=torch.long, device=DEV)
                bits[idx[scheme][n]] = HI
                Wq = gw_quant(W, bits)
            m.weight.data = Wq.to(m.weight.dtype)

    ids = tok(test, return_tensors="pt").input_ids[0]
    def ppl():
        nll = ntok = 0; L = 512
        with torch.no_grad():
            for i in range(0, min(len(ids), L * 32), L):
                ch = ids[i:i + L].unsqueeze(0).to(DEV)
                if ch.shape[1] < 2:
                    break
                out = model(ch, labels=ch)
                nll += out.loss.item() * (ch.shape[1] - 1); ntok += ch.shape[1] - 1
        return float(np.exp(nll / ntok))

    res = {}
    for s in ("fp16", "gw-uniform4", "magnitude", "act-energy", "sheaf-spec"):
        apply(s); res[s] = ppl(); print(f"  {s:13} ppl = {res[s]:.3f}")
    print("\nVERDICTS (equal ~4.0 b/param, strong group-wise base):")
    best_alloc = min(("magnitude", "act-energy", "sheaf-spec"), key=lambda s: res[s])
    print(f"  Q1 allocation vs strong uniform-4: best={best_alloc} {res[best_alloc]:.3f} vs uniform {res['gw-uniform4']:.3f}"
          f"  -> {'ALLOCATION STILL HELPS' if res[best_alloc] < res['gw-uniform4'] else 'allocation does NOT help on a strong base'}"
          f" ({(res['gw-uniform4']-res[best_alloc])/res['gw-uniform4']*100:+.1f}%)")
    print(f"  Q2 sheaf-spec vs act-energy: {res['sheaf-spec']:.3f} vs {res['act-energy']:.3f}"
          f"  -> {'SHEAF SIGNAL WINS' if res['sheaf-spec'] < res['act-energy'] else 'sheaf is AWQ-in-a-cape (no gain over diagonal)'}"
          f" ({(res['act-energy']-res['sheaf-spec'])/res['act-energy']*100:+.1f}%)")


if __name__ == "__main__":
    main()
