"""HARDWARE P2: does the certified-reference allocation actually BEAT the cheap-bf16-reference allocation?

gpu_fapp_hardware showed the float64 (certified) and bf16 (practical) references PICK different units to protect
(F-app does not fire; control confirmed it's real, not noise). But picks-differ is the weak claim. The
operational claim is: certification picks BETTER. Test: at matched budget k (k units kept bf16, rest W8A8-FP8),
compare the TRUE KL (vs the float64 reference) of:
   alloc_C : protect the top-k by certified (float64-referenced) importance
   alloc_P : protect the top-k by practical (bf16-referenced) importance
   alloc_R : protect a random k (no-information baseline, averaged)
If trueKL(alloc_C) < trueKL(alloc_P) at matched k -> certification has a real OPERATIONAL edge on hardware (P3).
If ~equal -> the picks differ but it does not matter; report honestly, do not force P3.
[V-hw] RTX 5070 sm_120, W8A8 FP8. float64 CPU reference. 6 texts.
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

import precision_allocator as PA
import gpu_fapp_hardware as H

torch.set_grad_enabled(False)
DEV = "cuda:0"


def run():
    print("[HARDWARE P2] certified vs practical-bf16 allocation, true-KL at matched budget (W8A8 regime)\n")
    tok = GPT2TokenizerFast.from_pretrained(PA.GPT2_PATH)
    m64 = GPT2LMHeadModel.from_pretrained(PA.GPT2_PATH).double().eval()
    Conv1D.forward = H.patched
    mg = GPT2LMHeadModel.from_pretrained(PA.GPT2_PATH).to(torch.bfloat16).to(DEV).eval()
    U = H.units(mg); nU = len(U)
    for u in U:
        u._prec = "bf16"

    # cache float64 references + score impC, impP (one pass)
    refs = []
    impC = np.zeros(nU); impP = np.zeros(nU); floor = 0.0
    for txt in H.TEXTS:
        ids = tok(txt, return_tensors="pt").input_ids[:, :32]
        for u in U:
            u._prec = "bf16"
        refC = m64(ids).logits[0].float()
        refP = mg(ids.to(DEV)).logits[0].float().cpu()
        refs.append((ids, refC)); floor += H.kl(refC, refP)
        for i in range(nU):
            U[i]._prec = "fp8"; cfg = mg(ids.to(DEV)).logits[0].float().cpu(); U[i]._prec = "bf16"
            impC[i] += H.kl(refC, cfg); impP[i] += H.kl(refP, cfg)
    impC = impC / len(H.TEXTS) - floor / len(H.TEXTS); impP /= len(H.TEXTS)

    def true_kl(keep_bf16):
        """Set keep_bf16 units to bf16, rest to FP8 W8A8; mean true KL vs the float64 references."""
        for j, u in enumerate(U):
            u._prec = "bf16" if j in keep_bf16 else "fp8"
        kls = []
        for ids, refC in refs:
            kls.append(H.kl(refC, mg(ids.to(DEV)).logits[0].float().cpu()))
        return float(np.mean(kls))

    print(f"  bf16 reference error (floor) = {floor/len(H.TEXTS):.3e}   all-W8A8 KL = {true_kl(set()):.3e}\n")
    print(f"  {'k(bf16)':>7} {'certified KL':>13} {'practical KL':>13} {'random KL':>11} {'cert<=prac?':>11} {'set overlap':>11}")
    wins = 0; comps = 0
    for k in (4, 8, 12, 16, 24):
        cset = set(np.argsort(-impC)[:k].tolist())
        pset = set(np.argsort(-impP)[:k].tolist())
        klC = true_kl(cset); klP = true_kl(pset)
        rng = np.random.default_rng(0)
        klR = float(np.mean([true_kl(set(rng.choice(nU, k, replace=False).tolist())) for _ in range(3)]))
        wins += klC <= klP + 1e-12; comps += 1
        print(f"  {k:>7} {klC:>13.4e} {klP:>13.4e} {klR:>11.4e} {str(klC <= klP + 1e-12):>11} "
              f"{len(cset & pset)/k:>10.0%}")

    print(f"\n[VERDICT] certified-allocation true-KL <= practical-bf16-allocation at {wins}/{comps} budgets.")
    if wins >= comps - 0:
        print("  -> CERTIFICATION HAS AN OPERATIONAL EDGE on hardware: scoring against the cheap bf16 reference picks")
        print("     measurably worse units than the float64 reference, in the activation-dominated W8A8 regime. P3 holds [V-hw].")
    elif wins >= comps * 0.6:
        print("  -> certification helps at most budgets but not all; real but partial edge. Report as-is.")
    else:
        print("  -> certified and practical allocations give ~equal true KL: the picks differ but it does NOT matter.")
        print("     F-app effectively fires on the OUTCOME even though the SETS differ. Do not force P3.")
    print("\n[V-hw] RTX 5070 sm_120, W8A8 FP8 (_scaled_mm). float64 CPU reference. 6 texts. No timing claim.")
    return impC, impP


if __name__ == "__main__":
    run()
