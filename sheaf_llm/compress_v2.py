#!/usr/bin/env python
"""structure-vs-bits v2 — the ALLOCATOR test (sheaf as bit-scheduler, not compressor).

v0/v1 closed the "factor the weights" door. v1's data pointed at allocation. So: at EQUAL average
bits (~4 b/param), does spending more bits on SALIENT input-channels and fewer on the rest beat
uniform 4-bit -- and does USE-AWARE (activation-energy) saliency beat cheap MAGNITUDE saliency
(the doc's TopologicalQuantizer claim)?

Three arms, all at avg 4.0 b/param (20% of columns @ 8-bit, 80% @ 3-bit; uniform = all @ 4-bit):
  uniform4        : every input channel 4-bit
  mixed-magnitude : top-20% columns by ||W[:,j]|| -> 8-bit, rest 3-bit
  mixed-useaware  : top-20% columns by E[x_j^2] (activation energy) -> 8-bit, rest 3-bit
Scored under the use-aware (activation-weighted) metric AND unweighted.

    python compress_v2.py [hf_model_id]
"""
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CALIB = [
    "The sheaf Laplacian measures inconsistency after restriction maps align neighboring stalks.",
    "Quantization error is white only if the weights carry no exploitable low-rank structure.",
    "A mixture-of-experts model keeps a small working set hot while streaming the cold experts.",
    "Geometry is the coordinate system over the stack, not a single compressor that beats four-bit.",
    "Use-aware error weights each input channel by how much it actually fires during inference.",
    "The honest gate kills the cheap version of the idea before any GPU hours are spent.",
]


def rtn_percol(W, bits_col):
    m, n = W.shape; Wq = W.clone()
    for b in bits_col.unique().tolist():
        mask = bits_col == b; qmax = 2 ** (int(b) - 1) - 1
        cols = W[:, mask]; s = cols.abs().amax(0, keepdim=True) / qmax + 1e-12
        Wq[:, mask] = torch.clamp(torch.round(cols / s), -qmax, qmax) * s
    return Wq, bits_col.float().mean().item() + 16.0 / m       # +per-column fp16 scale


def relerr(W, Wh, S=None):
    if S is None:
        return (torch.norm(W - Wh) / torch.norm(W)).item()
    return (torch.norm((W - Wh) * S) / torch.norm(W * S)).item()


def alloc(n, score, k, hi=8, lo=3):
    bits = torch.full((n,), lo, dtype=torch.long, device=score.device)
    bits[torch.argsort(score, descending=True)[:k]] = hi
    return bits


def main():
    print(f"[allocator test v2]  model={MODEL}  dev={DEV}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and min(m.weight.shape) >= 512
               and any(t in n for t in ("mlp", "proj"))]
    targets = [targets[i] for i in np.linspace(0, len(targets) - 1, 6).astype(int)]

    moments = {}
    def mk(nm):
        def hook(mod, inp):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            moments[nm] = moments.get(nm, torch.zeros(x.shape[-1], device=x.device)) + (x * x).sum(0)
        return hook
    handles = [m.register_forward_pre_hook(mk(n)) for n, m in targets]
    with torch.no_grad():
        for s in CALIB:
            model(tok(s, return_tensors="pt").input_ids.to(DEV))
    for h in handles:
        h.remove()

    R = {a: {"w": [], "u": [], "b": []} for a in ("uniform4", "mixed-magnitude", "mixed-useaware")}
    for n, mod in targets:
        W = mod.weight.detach().float().to(DEV); m, ncol = W.shape
        d = moments[n].to(DEV); S = d.clamp_min(1e-8).sqrt().reshape(1, ncol)
        k = round(0.2 * ncol)
        arms = {
            "uniform4": torch.full((ncol,), 4, dtype=torch.long, device=DEV),
            "mixed-magnitude": alloc(ncol, W.norm(dim=0), k),
            "mixed-useaware": alloc(ncol, d, k),
        }
        for a, bits in arms.items():
            Wq, bavg = rtn_percol(W, bits)
            R[a]["w"].append(relerr(W, Wq, S)); R[a]["u"].append(relerr(W, Wq)); R[a]["b"].append(bavg)

    me = lambda a, k: float(np.mean(R[a][k]))
    print(f"\n{'arm':17}{'avg bits':>9}{'weighted-err':>14}{'unweighted-err':>16}  (lower=better)")
    for a in R:
        print(f"{a:17}{me(a,'b'):>9.2f}{me(a,'w'):>14.4f}{me(a,'u'):>16.4f}")
    u, mag, ua = me('uniform4', 'w'), me('mixed-magnitude', 'w'), me('mixed-useaware', 'w')
    print("\nVERDICTS (use-aware metric, all at ~4 b/param):")
    print(f"  allocation vs uniform-4 : use-aware {ua:.4f} vs uniform {u:.4f}  -> "
          f"{'ALLOCATION WINS' if ua < u else 'uniform wins'} ({(u-ua)/u*100:+.1f}%)")
    print(f"  use-aware vs magnitude  : {ua:.4f} vs {mag:.4f}  -> "
          f"{'USE-AWARE saliency wins' if ua < mag else 'magnitude saliency wins'} ({(mag-ua)/mag*100:+.1f}%)")


if __name__ == "__main__":
    main()
