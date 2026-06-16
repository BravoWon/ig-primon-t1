"""HARDWARE: does the certification edge survive PROPER (per-channel) FP8 quantization?

The +6% certified edge appeared with NAIVE per-tensor activation FP8 (all-W8A8 = 2.8e-2, brutal). Real
deployment uses per-token activation + per-channel weight scaling (SmoothQuant-style), which tames outlier
features. Question: once you quantize properly, does the certification edge survive, or was it an artifact of
doing FP8 badly?

NB rowwise _scaled_mm is unsupported on sm_120 in this torch build, so per-channel is FAKE-QUANT (faithful FP8
rounding with per-token/per-channel scales, bf16 matmul / fp32 accumulate -- isolates SCALING GRANULARITY, not
the rowwise kernel). Control: per-tensor fake-quant must ~match the real _scaled_mm per-tensor number (2.8e-2).

  Part 1: all-W8A8 deployment KL, per-tensor vs per-channel (does proper scaling fix the activation floor?).
  Part 2: certification P2 (held-out): with per-channel FP8 configs, does the float64-referenced allocation
          still beat the bf16-referenced allocation? (does the edge survive proper quantization?)
[V-hw] RTX 5070 sm_120, FP8 e4m3 fake-quant per-channel. float64 reference. score on A, eval on disjoint B.
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

import precision_allocator as PA
import gpu_fapp_hardware as H

torch.set_grad_enabled(False)
DEV = "cuda:0"
FP8 = torch.float8_e4m3fn
TEXTS_A = H.TEXTS
TEXTS_B = [
    "A river carved the canyon over millions of years, exposing layers of ancient sediment.",
    "The committee postponed the vote until the auditors finished reviewing the quarterly accounts.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose inside the chloroplast.",
    "He tuned the old radio carefully, searching the static for a station broadcasting the match.",
    "Differential equations describe how quantities change continuously with respect to one another.",
    "The bakery on the corner sells warm sourdough every morning before the commuters arrive.",
]


def patched(self, x):
    size_out = x.size()[:-1] + (self.nf,)
    x2d = x.reshape(-1, x.size(-1)); W = self.weight; b = self.bias
    mode = getattr(self, "_prec", "bf16")
    if mode == "bf16":
        out = torch.addmm(b, x2d, W)
    elif mode == "fp8t":                                          # per-tensor fake-quant
        sx = x2d.float().abs().amax().clamp_min(1e-12) / 448.0
        sw = W.float().abs().amax().clamp_min(1e-12) / 448.0
        xq = ((x2d.float() / sx).to(FP8).float() * sx).to(x2d.dtype)
        wq = ((W.float() / sw).to(FP8).float() * sw).to(W.dtype)
        out = torch.addmm(b, xq, wq)
    else:                                                        # fp8pc: per-token act + per-channel weight
        sx = x2d.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        sw = W.float().abs().amax(0, keepdim=True).clamp_min(1e-12) / 448.0
        xq = ((x2d.float() / sx).to(FP8).float() * sx).to(x2d.dtype)
        wq = ((W.float() / sw).to(FP8).float() * sw).to(W.dtype)
        out = torch.addmm(b, xq, wq)
    return out.view(size_out)


def run():
    print("[per-channel hardware] does the certification edge survive proper (per-channel) FP8?\n")
    tok = GPT2TokenizerFast.from_pretrained(PA.GPT2_PATH)
    m64 = GPT2LMHeadModel.from_pretrained(PA.GPT2_PATH).double().eval()
    Conv1D.forward = patched
    mg = GPT2LMHeadModel.from_pretrained(PA.GPT2_PATH).to(torch.bfloat16).to(DEV).eval()
    U = H.units(mg); nU = len(U)

    def ids_of(t):
        return tok(t, return_tensors="pt").input_ids[:, :32]

    def all_to(mode):
        for u in U:
            u._prec = mode

    def mean_kl(texts, m64ref=True):
        ks = []
        for t in texts:
            ids = ids_of(t)
            ref = m64(ids).logits[0].float()
            ks.append(H.kl(ref, mg(ids.to(DEV)).logits[0].float().cpu()))
        return float(np.mean(ks))

    # ---- Part 1: deployment floor, per-tensor vs per-channel
    all_to("bf16");  kl_bf16 = mean_kl(TEXTS_B)
    all_to("fp8t");  kl_pt = mean_kl(TEXTS_B)
    all_to("fp8pc"); kl_pc = mean_kl(TEXTS_B)
    print("(1) all-W8A8 deployment KL vs float64 (held-out B):")
    print(f"    bf16 floor = {kl_bf16:.3e}   per-tensor FP8 = {kl_pt:.3e} (real _scaled_mm gave 2.82e-2)   "
          f"per-channel FP8 = {kl_pc:.3e}")
    print(f"    -> per-channel scaling cuts W8A8 error {kl_pt/kl_pc:.2f}x; gap above the bf16 floor = "
          f"{kl_pc - kl_bf16:.3e}\n")

    # ---- Part 2: certification P2 with per-channel configs, held-out
    impC = np.zeros(nU); impP = np.zeros(nU); floorA = 0.0
    for t in TEXTS_A:
        ids = ids_of(t); all_to("bf16")
        refC = m64(ids).logits[0].float(); refP = mg(ids.to(DEV)).logits[0].float().cpu()
        floorA += H.kl(refC, refP)
        for i in range(nU):
            U[i]._prec = "fp8pc"; cfg = mg(ids.to(DEV)).logits[0].float().cpu(); U[i]._prec = "bf16"
            impC[i] += H.kl(refC, cfg); impP[i] += H.kl(refP, cfg)
    impC = impC / len(TEXTS_A) - floorA / len(TEXTS_A); impP /= len(TEXTS_A)

    refsB = [(ids_of(t), m64(ids_of(t)).logits[0].float()) for t in TEXTS_B]

    def true_kl_B(keep):
        for j, u in enumerate(U):
            u._prec = "bf16" if j in keep else "fp8pc"
        return float(np.mean([H.kl(rc, mg(ids.to(DEV)).logits[0].float().cpu()) for ids, rc in refsB]))

    print("(2) certification P2 (per-channel configs, held-out B): certified vs practical-bf16 allocation")
    print(f"    {'k(bf16)':>7} {'certified KL':>13} {'practical KL':>13} {'cert<=prac?':>11}")
    wins = 0; gains = []
    for k in (4, 8, 12, 16, 24):
        cset = set(np.argsort(-impC)[:k].tolist()); pset = set(np.argsort(-impP)[:k].tolist())
        klC = true_kl_B(cset); klP = true_kl_B(pset)
        wins += klC <= klP + 1e-12; gains.append((klP - klC) / klP * 100)
        print(f"    {k:>7} {klC:>13.4e} {klP:>13.4e} {str(klC <= klP + 1e-12):>11}")

    g = float(np.mean(gains))
    print(f"\n[VERDICT] per-channel: certified <= practical at {wins}/5 budgets; mean certified gain = {g:+.1f}% KL.")
    print(f"  (naive per-tensor gave +6.1% on held-out.)")
    if wins >= 4 and g > 1:
        print("  -> the certification edge SURVIVES proper per-channel quantization: it is robust, not an artifact")
        print("     of crude per-tensor FP8. Certification helps even when you quantize well [V-hw].")
    elif g <= 1 or wins <= 2:
        print("  -> per-channel quantization ERASES the edge: certification mattered only because per-tensor FP8 was")
        print("     crude. With proper scaling the cheap bf16 reference is good enough. Honest deflation of the edge.")
    else:
        print("  -> the edge shrinks but persists; report as-is.")
    print("\n[V-hw] RTX 5070 sm_120, FP8 per-channel fake-quant (rowwise kernel unsupported here). float64 ref. A->B held-out.")
    return g


if __name__ == "__main__":
    run()
