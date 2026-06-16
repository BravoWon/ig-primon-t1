"""T1_precision_map v0.2 -- Stage 3 (sim): the LITERAL F-app, finally tested.

The registered F-app (pre-reg s.6) is a REFERENCE SWAP, not a scoring-method swap:
  "the certified-reference allocation produces the identical op-set at matched budget as the
   FP32-reference allocation -> certification adds nothing operational."
P3 (s.5): "the certified-vs-FP32 allocation differs at FP4 and agrees at FP8."

The earlier bake-off (mine AND the zip's) compared exact-vs-LINEAR *scoring*, both against a float64
reference -- orthogonal to this. Here the SCORING METHOD is held fixed (exact leave-one-low importance);
only the REFERENCE precision is swapped:
  certified pipeline : importance(i) scored against a float64 reference + float64 config forwards.
  FP32 pipeline      : importance(i) scored against an FP32 reference + FP32 config forwards (LAMP-style).
Top-B picks compared. F-app FIRES iff the two op-sets are identical at matched budget.

Honest prior (to be measured, not asserted): in weight-only fake-quant sim the FP32 full-model logits are
~1e-6 from float64, dwarfed by the quant-induced KLs (1e-3..4.5), so the importance ranking is likely
UNCHANGED -> F-app fires everywhere -> the certified-reference allocation edge is a HARDWARE (accumulation)
claim this sim cannot establish. If instead the picks differ, that is sim-evidence FOR the thesis. Either
way it is a [V]-grade honest result. [E-sim] CPU, 48 weight tensors, float64 vs float32 references.
"""
import numpy as np
import torch
from transformers import GPT2TokenizerFast

import precision_allocator as PA

torch.set_grad_enabled(False)

TEXTS = [
    "The history of numerical analysis begins with the study of round-off error in finite-precision arithmetic.",
    "She walked into the room and immediately noticed that something was different about the arrangement.",
    "Large language models compose many layers, each performing matrix multiplications and nonlinear maps.",
    "In the morning the fishermen returned with their catch and sold it at the market by the harbor.",
    "The proof proceeds by induction on the number of layers, bounding the error contributed at each step.",
    "Quantization reduces memory but can amplify errors at sharp attention logits and saturated activations.",
]
B_GRID = [4, 8, 12, 16, 24]


def overlap(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a), 1)


def topB(scores, B):
    return set(int(j) for j in np.argsort(-scores)[:B])


def importance(model, tensors, originals, base, ref_logits, ids, mask):
    """Exact leave-one-low importance scored against `ref_logits` (this pipeline's reference).
    s(i) = KL(ref || [group i demoted to base, rest bf16]) - KL(ref || [all bf16])."""
    nT = len(tensors)
    PA.apply_assignment(tensors, originals, ["bf16"] * nT)
    kl_base = PA.kl_flip(ref_logits, PA.logits_of(model, ids, mask), mask)[0]
    s = np.zeros(nT)
    for i in range(nT):
        tiers = ["bf16"] * nT; tiers[i] = base
        PA.apply_assignment(tensors, originals, tiers)
        s[i] = PA.kl_flip(ref_logits, PA.logits_of(model, ids, mask), mask)[0] - kl_base
    PA.restore(tensors, originals)
    return s


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.sqrt((ra @ ra) * (rb @ rb)) + 1e-30))


def run():
    tok = GPT2TokenizerFast.from_pretrained(PA.GPT2_PATH); tok.pad_token = tok.eos_token
    enc = tok(TEXTS, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    ids, mask = enc.input_ids, enc.attention_mask

    m64 = PA.load_model(torch.float64)
    m32 = PA.load_model(torch.float32)
    t64 = PA.allocatable(m64); t32 = PA.allocatable(m32)
    originals = PA.snapshot(t64)                                   # canonical fp64 originals (shared)
    nT = len(t64)

    ref64 = PA.logits_of(m64, ids, mask)                          # certified (float64) reference
    ref32 = PA.logits_of(m32, ids, mask)                          # FP32 reference (LAMP-style)
    ref_gap = float((ref32.to(torch.float64) - ref64).norm() / ref64.norm())

    print("[F-app reference swap]  literal test: exact importance, float64-ref vs FP32-ref, picks compared")
    print(f"  full-model reference logit gap ||ref_fp32 - ref_fp64|| / ||ref_fp64|| = {ref_gap:.2e}")
    print(f"  (the question: is this gap big enough to change which of {nT} weight tensors get promoted?)\n")

    verdicts = {}
    for base in ("fp8", "fp4"):
        s_cert = importance(m64, t64, originals, base, ref64, ids, mask)   # float64-referenced
        s_fp32 = importance(m32, t32, originals, base, ref32, ids, mask)   # FP32-referenced
        sp = spearman(s_cert, s_fp32)
        print(f"---- base = {base.upper()} ----")
        print(f"  importance rank corr (float64-ref vs FP32-ref): Spearman = {sp:+.4f}")
        print(f"  {'B':>3} {'cert==fp32 set?':>16} {'overlap':>9}")
        all_equal = True
        for B in B_GRID:
            cs, fs = topB(s_cert, B), topB(s_fp32, B)
            eq = (cs == fs); all_equal &= eq
            print(f"  {B:>3} {str(eq):>16} {overlap(cs, fs):>8.0%}")
        verdicts[base] = dict(sp=sp, fapp_fires=all_equal)
        print(f"  F-app at {base.upper()}: op-sets identical at every budget = {all_equal} "
              f"-> {'FIRES (certification adds no allocation edge here)' if all_equal else 'does NOT fire (the references pick differently)'}\n")

    print("=" * 88 + "\n[VERDICT -- literal F-app / P3]")
    f8, f4 = verdicts["fp8"], verdicts["fp4"]
    print(f"  F-app fires (identical picks) at FP8={f8['fapp_fires']}  FP4={f4['fapp_fires']}")
    if f8["fapp_fires"] and f4["fapp_fires"]:
        print("  => F-app FIRES at BOTH. In this weight-only fake-quant sim the FP32 reference is ~indistinguishable")
        print("     from the certified float64 reference (gap above << quant signal), so it picks the SAME ops.")
        print("     P3 NOT supported in sim. The certified-reference operational edge is a HARDWARE claim")
        print("     (FP32 accumulation error, real tensor cores) that THIS sim cannot establish. Honest, pre-registered")
        print("     branch (pre-reg s.7): retreat to the certified depth-law science; the allocator edge is unproven in sim.")
    elif f8["fapp_fires"] and not f4["fapp_fires"]:
        print("  => F-app fires at FP8 but NOT at FP4: the FP32 reference's error is large enough at FP4 to change")
        print("     the allocation -> P3 SUPPORTED even in sim (the registered prediction, now from the right test).")
    else:
        print("  => pattern does not match P3 (differs at FP8, or only at FP8). Report as-is; do not force P3.")
    print("\n[E-sim] CPU, 48 weight tensors, weight-only fake-quant, exact leave-one-low importance.")
    print("        Swaps ONLY the reference precision (float64 vs fp32). No hardware accumulation modeled.")
    return verdicts


if __name__ == "__main__":
    run()
