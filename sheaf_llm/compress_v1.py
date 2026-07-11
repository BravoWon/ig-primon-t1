#!/usr/bin/env python
"""structure-vs-bits v1 — un-collapse the two axes v0 closed unfairly.

  AXIS 5 (metric): v0 used unweighted Frobenius. Re-score with a USE-AWARE (activation-weighted)
                   metric, and use use-aware-optimal low-rank (SVD of W*sqrt(diag E[x^2])).
  AXIS 8 (composition): test 4-bit + low-rank(RESIDUAL) -- structure ON TOP OF bits -- vs just
                   spending the same marginal bit on uniform 5-bit.

Questions: (Q1) does the marginal bit buy more as low-rank residual or as more quant levels?
           (Q2) is the quant residual more low-rank than the raw weight?
           (Q3) under a use-aware metric, does low-rank close the gap to 4-bit?

    python compress_v1.py [hf_model_id]
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


def rtn(W, bits, group=128):
    m, n = W.shape; Wh = W.clone(); qmax = 2 ** (bits - 1) - 1; g = min(group, n)
    for j0 in range(0, n, g):
        blk = W[:, j0:j0 + g]; s = blk.abs().amax(1, keepdim=True) / qmax + 1e-12
        Wh[:, j0:j0 + g] = torch.clamp(torch.round(blk / s), -qmax, qmax) * s
    return Wh, bits + 16.0 / g


def relerr(W, Wh, S=None):
    if S is None:
        return (torch.norm(W - Wh) / torch.norm(W)).item()
    return (torch.norm((W - Wh) * S) / torch.norm(W * S)).item()    # S = (1,n) column weights


def lowrank(A, r):
    U, S, Vh = torch.linalg.svd(A, full_matrices=False); r = min(r, S.numel())
    return (U[:, :r] * S[:r]) @ Vh[:r], (S[:r] ** 2).sum() / (S ** 2).sum()


def main():
    print(f"[structure-vs-bits v1]  model={MODEL}  dev={DEV}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()

    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and min(m.weight.shape) >= 512
               and any(t in n for t in ("mlp", "proj"))]
    idx = np.linspace(0, len(targets) - 1, 5).astype(int)
    targets = [targets[i] for i in idx]
    names = {n for n, _ in targets}

    # --- use-aware calibration: E[x^2] per in-feature for each target linear ---
    moments = {}
    def mk(nm):
        def hook(mod, inp):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            s = (x * x).sum(0)
            moments[nm] = moments.get(nm, torch.zeros_like(s)) + s
        return hook
    handles = [m.register_forward_pre_hook(mk(n)) for n, m in targets]
    with torch.no_grad():
        for s in CALIB:
            ids = tok(s, return_tensors="pt").input_ids.to(DEV)
            model(ids)
    for h in handles:
        h.remove()

    A = {"e4": [], "ehyb": [], "bhyb": [], "e5": [], "enR": [], "enW": []}
    B = {"e4w": [], "elp_w": [], "elua_w": []}
    for n, mod in targets:
        W = mod.weight.detach().float().to(DEV); m, nn_ = W.shape
        Q4, b4 = rtn(W, 4); A["e4"].append(relerr(W, Q4))
        R = W - Q4
        r = max(1, round(0.9 * m * nn_ / (16 * (m + nn_))))            # ~+0.9 bit as low-rank residual
        Rr, enR = lowrank(R, r)
        A["ehyb"].append(relerr(W, Q4 + Rr)); A["bhyb"].append(b4 + 16.0 * r * (m + nn_) / (m * nn_))
        A["enR"].append(enR.item())
        _, enW = lowrank(W, r); A["enW"].append(enW.item())
        Q5, b5 = rtn(W, 5); A["e5"].append(relerr(W, Q5))

        # use-aware metric (axis 5)
        d = moments[n].to(DEV); S = d.clamp_min(1e-8).sqrt().reshape(1, nn_)
        r4 = max(1, round(4 * m * nn_ / (16 * (m + nn_))))
        lp, _ = lowrank(W, r4)                                          # plain low-rank
        lua, _ = lowrank(W * S, r4); lua = lua / S                      # use-aware-optimal low-rank
        B["e4w"].append(relerr(W, Q4, S))
        B["elp_w"].append(relerr(W, lp, S))
        B["elua_w"].append(relerr(W, lua, S))

    me = lambda k, d=A: float(np.mean(d[k]))
    print("\n--- AXIS 8: composition (structure ON TOP OF bits) ---")
    print(f"  4-bit            @ ~4.12 b/param : err {me('e4'):.4f}")
    print(f"  4-bit + LR resid @ ~{me('bhyb'):.2f} b/param : err {me('ehyb'):.4f}")
    print(f"  uniform 5-bit    @ ~5.12 b/param : err {me('e5'):.4f}")
    win = "LR-RESIDUAL wins the marginal bit" if me('ehyb') < me('e5') else "more quant levels win the marginal bit"
    print(f"  Q1 -> {win}  ({(me('e5')-me('ehyb'))/me('e5')*100:+.1f}% vs uniform-5b)")
    print(f"  Q2 residual low-rankness: top-r energy  residual {me('enR'):.3f}  vs  raw-W {me('enW'):.3f}"
          f"  -> {'residual MORE low-rank' if me('enR') > me('enW') else 'residual NOT more low-rank'}")

    print("\n--- AXIS 5: use-aware metric (activation-weighted) @ 4 b/param ---")
    print(f"  4-bit (weighted err)                : {me('e4w', B):.4f}")
    print(f"  low-rank plain (weighted err)       : {me('elp_w', B):.4f}")
    print(f"  low-rank USE-AWARE (weighted err)   : {me('elua_w', B):.4f}")
    print(f"  Q3 -> use-aware low-rank vs plain: {(me('elp_w',B)-me('elua_w',B))/me('elp_w',B)*100:+.1f}%; "
          f"gap to 4-bit still {me('elua_w',B)/me('e4w',B):.1f}x")


if __name__ == "__main__":
    main()
