"""HARDWARE P2, held-out: kill the 'home-field advantage' confound on the certification edge.

gpu_fapp_p2 scored importance against float64 AND measured true KL against float64 on the SAME texts -- so the
certified allocation could be overfitting to the scoring set. Clean test: derive the allocation on corpus A,
measure true (float64) KL on DISJOINT corpus B. If the certified (float64-referenced) allocation still beats the
practical (bf16-referenced) one on held-out B, the operational edge GENERALIZES and is real -- not leakage.
[V-hw] RTX 5070 sm_120, W8A8 FP8. float64 CPU reference.
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

import precision_allocator as PA
import gpu_fapp_hardware as H

torch.set_grad_enabled(False)
DEV = "cuda:0"
TEXTS_A = H.TEXTS                                            # score the allocation here
TEXTS_B = [                                                  # evaluate true fidelity here (disjoint)
    "A river carved the canyon over millions of years, exposing layers of ancient sediment.",
    "The committee postponed the vote until the auditors finished reviewing the quarterly accounts.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose inside the chloroplast.",
    "He tuned the old radio carefully, searching the static for a station broadcasting the match.",
    "Differential equations describe how quantities change continuously with respect to one another.",
    "The bakery on the corner sells warm sourdough every morning before the commuters arrive.",
]


def run():
    print("[HARDWARE P2 held-out] allocation scored on A, true-KL measured on disjoint B (kills leakage)\n")
    tok = GPT2TokenizerFast.from_pretrained(PA.GPT2_PATH)
    m64 = GPT2LMHeadModel.from_pretrained(PA.GPT2_PATH).double().eval()
    Conv1D.forward = H.patched
    mg = GPT2LMHeadModel.from_pretrained(PA.GPT2_PATH).to(torch.bfloat16).to(DEV).eval()
    U = H.units(mg); nU = len(U)
    for u in U:
        u._prec = "bf16"

    def ids_of(txt):
        return tok(txt, return_tensors="pt").input_ids[:, :32]

    # score impC, impP on A
    impC = np.zeros(nU); impP = np.zeros(nU); floorA = 0.0
    for txt in TEXTS_A:
        ids = ids_of(txt)
        for u in U:
            u._prec = "bf16"
        refC = m64(ids).logits[0].float(); refP = mg(ids.to(DEV)).logits[0].float().cpu()
        floorA += H.kl(refC, refP)
        for i in range(nU):
            U[i]._prec = "fp8"; cfg = mg(ids.to(DEV)).logits[0].float().cpu(); U[i]._prec = "bf16"
            impC[i] += H.kl(refC, cfg); impP[i] += H.kl(refP, cfg)
    impC = impC / len(TEXTS_A) - floorA / len(TEXTS_A); impP /= len(TEXTS_A)

    # held-out float64 references on B
    refsB = [(ids_of(t), m64(ids_of(t)).logits[0].float()) for t in TEXTS_B]

    def true_kl_B(keep_bf16):
        for j, u in enumerate(U):
            u._prec = "bf16" if j in keep_bf16 else "fp8"
        return float(np.mean([H.kl(refC, mg(ids.to(DEV)).logits[0].float().cpu()) for ids, refC in refsB]))

    print(f"  {'k(bf16)':>7} {'certified KL_B':>15} {'practical KL_B':>15} {'random KL_B':>12} {'cert<=prac?':>11}")
    wins = 0; comps = 0; gains = []
    for k in (4, 8, 12, 16, 24):
        cset = set(np.argsort(-impC)[:k].tolist()); pset = set(np.argsort(-impP)[:k].tolist())
        klC = true_kl_B(cset); klP = true_kl_B(pset)
        rng = np.random.default_rng(1)
        klR = float(np.mean([true_kl_B(set(rng.choice(nU, k, replace=False).tolist())) for _ in range(3)]))
        wins += klC <= klP + 1e-12; comps += 1; gains.append((klP - klC) / klP * 100)
        print(f"  {k:>7} {klC:>15.4e} {klP:>15.4e} {klR:>12.4e} {str(klC <= klP + 1e-12):>11}")

    print(f"\n[VERDICT -- held-out] certified <= practical on disjoint B at {wins}/{comps} budgets; "
          f"mean certified gain = {np.mean(gains):+.1f}% KL.")
    if wins >= comps - 0 and np.mean(gains) > 1:
        print("  -> the certification edge GENERALIZES to held-out text: it is a real (if modest) operational")
        print("     advantage, not leakage. P3 holds ON HARDWARE [V-hw], in the on-device-bf16-reference regime.")
    elif wins >= comps * 0.6:
        print("  -> certified wins on most held-out budgets: real but partial edge. Report as-is.")
    else:
        print("  -> on held-out text the edge vanishes: the same-text result was leakage. Do NOT claim P3.")
    print("\n[V-hw] RTX 5070 sm_120, W8A8 FP8. float64 reference. score on A, eval on disjoint B. No timing claim.")
    return wins, comps, float(np.mean(gains))


if __name__ == "__main__":
    run()
