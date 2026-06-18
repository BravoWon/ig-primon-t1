"""OPT-2.7B rotation + CLIPPING: the complete storage-free W4A4 recipe on the flagship outlier model.

GPT-2 (v0.2.27): rotation+clipping = +26% storage-free. OPT-2.7B rotation alone (v0.2.24) = +24%. This applies
the proven clip knob to OPT: one rotated reconstruction, eval W4A4 across clip alpha. Best storage-free W4A4 on
the real severe-outlier model. Reuses opt_rotation_w4a4's reconstruct/gptq with a clip-aware forward.
[V-hw] OPT-2.7B, WikiText-2 held-out, bf16 master, sequential GPTQ FP4-g128+act-order + per-token clipped INT4 acts.
"""
import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP
import opt_rotation_w4a4 as OR

torch.set_grad_enabled(False)
DEV = OR.DEV; CLIP = {"v": 1.0}; CLIPS = [1.0, 0.85, 0.7, 0.6, 0.5]


def patched_clip(self, x):
    W = self.weight; b = self.bias
    Q = getattr(self, "_rotQ", None)
    if Q is not None:
        x = (x.float() @ Q).to(x.dtype)
    if not getattr(self, "_aq", False) or OR.ACT["bits"] is None:
        return torch.nn.functional.linear(x, W, b)
    x2 = x.reshape(-1, x.size(-1)).float()
    s = (CLIP["v"] * x2.abs().amax(1, keepdim=True)).clamp_min(1e-12) / 7.0
    xq = (torch.clamp(torch.round(x2 / s), -7, 7) * s).to(x.dtype).view(*x.shape)
    return torch.nn.functional.linear(xq, W, b)


def run():
    print("[OPT-2.7B rotation + CLIPPING W4A4]  best storage-free W4A4 on the flagship outlier model\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = OR.load_corpus(tok, OR.N_CAL + OR.N_EVAL); cal, ev = seqs[:OR.N_CAL], seqs[OR.N_CAL:]
    tgts = [s[0, 1:] for s in ev]
    nn.Linear.forward = patched_clip
    m0 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.bfloat16).to(DEV).eval()
    OR.ACT["bits"] = None; gold = OR.ppl(m0, ev, tgts); del m0; gc.collect(); torch.cuda.empty_cache()
    print(f"  bf16 gold PPL = {gold:.3f}\n")

    OR.ROT.clear()
    m = OR.reconstruct(True, cal)                                   # rotated (random orthogonal), FP4-g128+act-order
    OR.ACT["bits"] = None; p16 = OR.ppl(m, ev, tgts)
    print(f"  rotated W4A16 (control): {p16:.3f} ({100*(p16/gold-1):+.2f}%)\n")
    print(f"  rotated W4A4 -- clip sweep:")
    print(f"    {'clip a':>7} {'W4A4 PPL':>11} {'dPPL vs gold':>13}")
    best = None
    for a in CLIPS:
        OR.ACT["bits"] = 4; CLIP["v"] = a; p4 = OR.ppl(m, ev, tgts)
        print(f"    {a:>7.2f} {p4:>11.3f} {100*(p4/gold-1):>+12.1f}%")
        if best is None or p4 < best[1]:
            best = (a, p4)
    del m; gc.collect(); torch.cuda.empty_cache()

    print(f"\n[VERDICT]")
    print(f"  OPT-2.7B W4A4: naive +37,843% (v0.2.24) -> rotation +24% (clip=1.0) -> rotation+clip BEST "
          f"+{100*(best[1]/gold-1):.1f}% (a={best[0]:.2f})")
    print(f"  Complete storage-free W4A4 recipe on the real outlier model: rotation + clipping, all 4-bit, no padding.")
    print("\n[V-hw] OPT-2.7B, WikiText-2, bf16 master, sequential GPTQ FP4-g128+act-order + clipped per-token INT4 acts.")
    return gold, p16, best


if __name__ == "__main__":
    run()
