"""THE PAYOFF: does the certification edge survive REAL SmoothQuant, on a real-outlier model (OPT-2.7B)?

OPT-2.7B passed the validity gate (median outlier 25x; SmoothQuant crushes the W8A8 floor 1.2x) -- the test
GPT-2 could never support. Now the original question, answerable at last:

  configs quantized with REAL SmoothQuant (W8A8 + per-input-channel migration). Allocation = keep top-k units
  at bf16 ('hi'), rest at SmoothQuant-W8A8. Score the allocation against two references:
    certified = fp32 anchor (CPU, spot-licensed) ;  practical = bf16 on-device forward.
  Held-out: score on A, measure true KL (vs fp32 anchor) on disjoint B.
  Q: does the fp32-certified allocation still beat the cheap-bf16 allocation once activations are properly
     mitigated by SmoothQuant, on a model with genuine outliers? (vanish / survive / dominate)

Memory: fp32 anchor on CPU (cache refs), bf16 deployment + FP8 on GPU, sequential load.
[V-hw] RTX 5070 sm_120 (bf16) + CPU fp32 anchor. OPT-2.7B. score A, eval disjoint B.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; ALPHA = 0.5
TEXTS_A = OP.TEXTS
TEXTS_B = [
    "A river carved the canyon over millions of years, exposing layers of ancient sediment.",
    "The committee postponed the vote until the auditors finished reviewing the quarterly accounts.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose inside the chloroplast.",
    "He tuned the old radio carefully, searching the static for a station broadcasting the match.",
    "Differential equations describe how quantities change continuously with respect to one another.",
    "The bakery on the corner sells warm sourdough every morning before the commuters arrive.",
]


def run():
    print("[OPT-2.7B certification] does the edge survive REAL SmoothQuant on a real-outlier model?\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    def ids_of(t):
        return tok(t, return_tensors="pt").input_ids[:, :32]
    torch.nn.Linear.forward = OP.patched_linear

    # ---- Phase A: fp32 anchor on CPU -- cache references for A and B ----
    m32 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).eval()
    refA = [m32(ids_of(t)).logits[0].float() for t in TEXTS_A]
    refB = [m32(ids_of(t)).logits[0].float() for t in TEXTS_B]
    del m32; gc.collect()

    # ---- Phase B: bf16 deployment on GPU ----
    mg = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.bfloat16).to(DEV).eval()
    U = OP.units(mg); nU = len(U)

    # calibrate SmoothQuant s (per-input-channel act max over A)
    store = {}; hs = []
    def mk(u):
        def hook(mod, inp):
            a = inp[0].reshape(-1, inp[0].size(-1)).float().abs().amax(0)
            store[id(u)] = a if id(u) not in store else torch.maximum(store[id(u)], a)
        return u.register_forward_pre_hook(hook)
    for u in U:
        u._prec = "hi"; hs.append(mk(u))
    for t in TEXTS_A:
        mg(ids_of(t).to(DEV))
    for h in hs:
        h.remove()
    for u in U:
        amax = store[id(u)].clamp_min(1e-12)
        wmax = u.weight.float().abs().amax(0).clamp_min(1e-12)
        u._s = ((amax ** ALPHA) / (wmax ** (1 - ALPHA))).clamp(1e-3, 1e3).to(u.weight.dtype)

    # score impC (vs fp32 anchor) and impP (vs bf16 reference), SmoothQuant configs
    impC = np.zeros(nU); impP = np.zeros(nU); floorA = 0.0
    for ti, t in enumerate(TEXTS_A):
        ids = ids_of(t).to(DEV)
        for u in U:
            u._prec = "hi"
        refP = mg(ids).logits[0].float().cpu()
        floorA += OP.kl(refA[ti], refP)
        for i in range(nU):
            U[i]._prec = "smooth"; cfg = mg(ids).logits[0].float().cpu(); U[i]._prec = "hi"
            impC[i] += OP.kl(refA[ti], cfg); impP[i] += OP.kl(refP, cfg)
    impC = impC / len(TEXTS_A) - floorA / len(TEXTS_A); impP /= len(TEXTS_A)

    def true_kl_B(keep):
        for j, u in enumerate(U):
            u._prec = "hi" if j in keep else "smooth"
        return float(np.mean([OP.kl(refB[i], mg(ids_of(t).to(DEV)).logits[0].float().cpu())
                              for i, t in enumerate(TEXTS_B)]))

    # context: bf16 floor and all-SmoothQuant deployment on B
    for u in U:
        u._prec = "hi"
    kl_floor = float(np.mean([OP.kl(refB[i], mg(ids_of(t).to(DEV)).logits[0].float().cpu())
                              for i, t in enumerate(TEXTS_B)]))
    kl_allsq = true_kl_B(set())
    print(f"  bf16 floor (B) = {kl_floor:.3e}   all-SmoothQuant-W8A8 (B) = {kl_allsq:.3e}\n")

    print("  certified (fp32-ref) vs practical (bf16-ref) allocation, true KL vs fp32 anchor on held-out B:")
    print(f"  {'k(bf16)':>7} {'certified KL':>13} {'practical KL':>13} {'cert<=prac?':>11} {'overlap':>8}")
    wins = 0; gains = []
    for k in (8, 16, 24, 32, 48):
        cset = set(np.argsort(-impC)[:k].tolist()); pset = set(np.argsort(-impP)[:k].tolist())
        klC = true_kl_B(cset); klP = true_kl_B(pset)
        wins += klC <= klP + 1e-12; gains.append((klP - klC) / max(klP, 1e-30) * 100)
        print(f"  {k:>7} {klC:>13.4e} {klP:>13.4e} {str(klC <= klP + 1e-12):>11} {len(cset&pset)/k:>7.0%}")
    g = float(np.mean(gains))

    print(f"\n[VERDICT] OPT-2.7B + real SmoothQuant: certified <= practical {wins}/5; mean certified gain = {g:+.1f}% KL.")
    print("  (GPT-2-small per-tensor +6.1%; per-channel +4.0%; fp32-anchor floor sweep ~+2%.)")
    if wins >= 4 and g > 2:
        print("  -> the certification edge SURVIVES real SmoothQuant on a real-outlier model: even with proper")
        print("     activation mitigation, scoring against fp32 beats scoring against the cheap bf16 reference. [V-hw]")
    elif g <= 1 or wins <= 2:
        print("  -> SmoothQuant + scale ERASES the edge: with real mitigation the cheap reference is good enough.")
    else:
        print("  -> edge persists at reduced magnitude; report as measured, claim nothing larger.")
    print("\n[V-hw] RTX 5070 sm_120 (bf16) + CPU fp32 anchor. OPT-2.7B, real SmoothQuant configs. score A, eval B.")
    return wins, g


if __name__ == "__main__":
    run()
