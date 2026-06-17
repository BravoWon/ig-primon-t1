"""T1_precision_map v0.2 -- HARDWARE: the F-app / P3 certification test, in the regime where it can bite.

The CPU F-app fired null because the reference (un-quantized full model) is fp32 ~= float64 to 1e-6 -- a
near-perfect reference makes certification redundant. But the GPU bridge showed the REALISTIC on-device
reference (a bf16 forward) carries ~1.5e-2 error. So the certification thesis finally has a fair test:

  Two references for SCORING which Conv1D units to keep bf16 vs demote to FP8 (W8A8, the large-error regime):
    refC (certified) : float64 CPU forward.            <- expensive, "the oracle"
    refP (practical) : bf16 forward on the 5070.       <- cheap, what you'd actually calibrate against on-device
  Importance(i) = marginal KL from demoting unit i to FP8 W8A8, scored against each reference. Compare the
  top-k protected sets. F-app FIRES (null) iff identical -> even a 1.5e-2-inexact reference picks the same units
  -> certification adds nothing even here. F-app does NOT fire -> the certified reference protects different units
  -> P3 revives ON HARDWARE, in the realistic on-device calibration scenario. Honest: outcome unknown, run it.
[V-hw] RTX 5070 sm_120, native FP8 e4m3 (_scaled_mm). float64 CPU reference. 6 texts. No timing claim.
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

import precision_allocator as PA

torch.set_grad_enabled(False)
DEV = "cuda:0"
TEXTS = [
    "The history of numerical analysis begins with the study of round-off error in finite-precision arithmetic.",
    "She walked into the room and immediately noticed that something was different about the arrangement.",
    "Large language models compose many layers, each performing matrix multiplications and nonlinear maps.",
    "In the morning the fishermen returned with their catch and sold it at the market by the harbor.",
    "The proof proceeds by induction on the number of layers, bounding the error contributed at each step.",
    "Quantization reduces memory but can amplify errors at sharp attention logits and saturated activations.",
]


def q_fp8(x):
    amax = x.abs().amax().clamp_min(1e-12); s = amax / 448.0
    return (x / s).to(torch.float8_e4m3fn), s.to(x.dtype)


def patched(self, x):
    size_out = x.size()[:-1] + (self.nf,)
    x2d = x.reshape(-1, x.size(-1)); W = self.weight; b = self.bias
    if getattr(self, "_prec", "bf16") == "fp8":
        x8, sx = q_fp8(x2d.float()); w8, sw = q_fp8(W.float())
        out = torch._scaled_mm(x8.contiguous(), w8.t().contiguous().t(),
                               scale_a=sx.float().to(DEV), scale_b=sw.float().to(DEV),
                               out_dtype=torch.bfloat16) + b
    else:
        out = torch.addmm(b, x2d, W)
    return out.view(size_out)


def kl(ref, q):
    rl = ref.double(); ql = q.double()
    lpr = torch.log_softmax(rl, -1); lpq = torch.log_softmax(ql, -1)
    return (lpr.exp() * (lpr - lpq)).sum(-1).mean().item()


def units(model):
    out = []
    for blk in model.transformer.h:
        out += [blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj]
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.sqrt((ra @ ra) * (rb @ rb)) + 1e-30))


def run():
    print(f"[GPU F-app / P3 hardware test]  RTX 5070 sm_120, W8A8 regime, float64 vs bf16 reference\n")
    tok = GPT2TokenizerFast.from_pretrained(PA.GPT2_PATH)
    m64 = GPT2LMHeadModel.from_pretrained(PA.GPT2_PATH).double().eval()
    Conv1D.forward = patched
    mg = GPT2LMHeadModel.from_pretrained(PA.GPT2_PATH).to(torch.bfloat16).to(DEV).eval()
    U = units(mg); nU = len(U)
    for u in U:
        u._prec = "bf16"

    impC = np.zeros(nU); impP = np.zeros(nU); floor = 0.0
    for txt in TEXTS:
        ids = tok(txt, return_tensors="pt").input_ids[:, :32]
        for u in U:
            u._prec = "bf16"
        refC = m64(ids).logits[0].float()                      # float64 certified reference
        refP = mg(ids.to(DEV)).logits[0].float().cpu()         # bf16 GPU practical reference (all-high)
        floor += kl(refC, refP)
        for i in range(nU):
            U[i]._prec = "fp8"
            cfg = mg(ids.to(DEV)).logits[0].float().cpu()       # demote unit i to FP8 W8A8
            U[i]._prec = "bf16"
            impC[i] += kl(refC, cfg)                             # vs certified
            impP[i] += kl(refP, cfg)                             # vs practical bf16 ref
    impC /= len(TEXTS); impP /= len(TEXTS); floor /= len(TEXTS)
    impC = impC - floor                                         # certified marginal (subtract the bf16 floor)

    sp = spearman(impC, impP)
    print(f"  bf16 practical-reference error vs float64 certified (the floor) = {floor:.3e}")
    print(f"  importance rank corr (certified vs practical-bf16 reference): Spearman = {sp:+.4f}\n")
    print(f"  protected-set agreement (top-k units kept bf16 under each reference):")
    print(f"  {'k':>3} {'cert==practical?':>16} {'overlap':>9}")
    all_eq = True
    for k in (4, 8, 12, 16, 24):
        sc = set(np.argsort(-impC)[:k].tolist()); sp_ = set(np.argsort(-impP)[:k].tolist())
        eq = (sc == sp_); all_eq &= eq
        print(f"  {k:>3} {str(eq):>16} {len(sc & sp_)/k:>8.0%}")

    print("\n[VERDICT -- does certification matter when the on-device reference is bf16-inexact?]")
    if all_eq and sp > 0.98:
        print(f"  F-app FIRES even here: a {floor:.1e}-inexact bf16 reference picks the SAME units as float64.")
        print("  -> certification adds nothing even in the realistic on-device regime. The thesis is dead on metal too;")
        print("     the bf16 reference's error is 'generic' -- it does not correlate with which unit to protect.")
    elif not all_eq:
        nfp = [k for k in (4, 8, 12, 16, 24)
               if set(np.argsort(-impC)[:k].tolist()) != set(np.argsort(-impP)[:k].tolist())]
        print(f"  F-app does NOT fire at budgets {nfp}: the certified (float64) reference protects DIFFERENT units")
        print(f"  than the cheap bf16 reference (Spearman {sp:+.3f}). -> P3 REVIVES ON HARDWARE: when you calibrate")
        print("     allocation on-device against bf16, certification changes the decision. The thesis has teeth here.")
    else:
        print(f"  mixed (Spearman {sp:+.3f}, sets mostly agree): report as-is, do not force P3.")
    print("\n[V-hw] RTX 5070 sm_120, FP8 W8A8 (_scaled_mm). float64 CPU vs bf16 GPU reference. No timing claim.")
    return impC, impP, floor


if __name__ == "__main__":
    run()
