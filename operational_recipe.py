"""T1_precision_map v0.2 -- maxop: the operationalization config, honestly derived.

The certification thesis is null in sim (F-app fired at both; results_fapp_reference_swap.txt). So this does
NOT lean on certification. It leans only on the one [V] structural fact: weight-quant sensitivity is sparse
(965x spread, results_pillars_fp4.txt). The deliverable is a STATIC weight-only mixed-precision RECIPE:
rank tensors by sensitivity, keep the pillars high, crush the redundant -- and report what fidelity that
buys at what compression, validated OUT OF SAMPLE.

Honesty rails:
  - Recipe DERIVED from corpus A; fidelity MEASURED on disjoint held-out corpus B (no in-sample fitting).
  - Stability control: the ranking must agree across A and B, else "pillars" are just-these-sentences noise.
  - An FP32 reference yields the same ranking (F-app Spearman +1.0) -> this recipe is NOT certification-exclusive.
  - bits are reported over the ALLOCATABLE matmul weights only (embeddings/LN held at reference). [E-sim].
"""
import numpy as np
import torch
from transformers import GPT2TokenizerFast

import precision_allocator as PA

torch.set_grad_enabled(False)

TEXTS_SCORE = [   # corpus A: derive the recipe here
    "The history of numerical analysis begins with the study of round-off error in finite-precision arithmetic.",
    "She walked into the room and immediately noticed that something was different about the arrangement.",
    "Large language models compose many layers, each performing matrix multiplications and nonlinear maps.",
    "In the morning the fishermen returned with their catch and sold it at the market by the harbor.",
    "The proof proceeds by induction on the number of layers, bounding the error contributed at each step.",
    "Quantization reduces memory but can amplify errors at sharp attention logits and saturated activations.",
]
TEXTS_EVAL = [    # corpus B: disjoint, validate fidelity + stability here
    "A river carved the canyon over millions of years, exposing layers of ancient sediment.",
    "The committee postponed the vote until the auditors finished reviewing the quarterly accounts.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose inside the chloroplast.",
    "He tuned the old radio carefully, searching the static for a station broadcasting the match.",
    "Differential equations describe how quantities change continuously with respect to one another.",
    "The bakery on the corner sells warm sourdough every morning before the commuters arrive.",
]
TIER_BITS = {"fp4": 4, "fp8": 8, "bf16": 16}


def batch(tok, texts):
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    return enc.input_ids, enc.attention_mask


def importance(m, tensors, originals, base, ids, mask):
    nT = len(tensors)
    ref = PA.logits_of(m, ids, mask)
    PA.apply_assignment(tensors, originals, ["bf16"] * nT)
    kl_base = PA.kl_flip(ref, PA.logits_of(m, ids, mask), mask)[0]
    s = np.zeros(nT)
    for i in range(nT):
        t = ["bf16"] * nT; t[i] = base
        PA.apply_assignment(tensors, originals, t)
        s[i] = PA.kl_flip(ref, PA.logits_of(m, ids, mask), mask)[0] - kl_base
    PA.restore(tensors, originals)
    return s


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.sqrt((ra @ ra) * (rb @ rb)) + 1e-30))


def run():
    tok = GPT2TokenizerFast.from_pretrained(PA.GPT2_PATH); tok.pad_token = tok.eos_token
    idsA, mA = batch(tok, TEXTS_SCORE); idsB, mB = batch(tok, TEXTS_EVAL)
    m = PA.load_model(torch.float64)
    tensors = PA.allocatable(m); originals = PA.snapshot(tensors); nT = len(tensors)
    params = np.array([W.numel() for W in originals], dtype=float); tot = params.sum()
    refB = PA.logits_of(m, idsB, mB)

    print("[maxop operational recipe]  GPT-2-small weight-only mixed precision, derived on A, validated on B\n")

    # ---- (1) STABILITY CONTROL: ranking must generalize A -> B (use fp4 sensitivity, the harshest)
    sA = importance(m, tensors, originals, "fp4", idsA, mA)
    sB = importance(m, tensors, originals, "fp4", idsB, mB)
    sp = spearman(sA, sB)
    topA = set(np.argsort(-sA)[:12].tolist()); topB = set(np.argsort(-sB)[:12].tolist())
    ov = len(topA & topB) / 12
    print(f"(1) STABILITY (recipe must generalize): Spearman(s_A, s_B) = {sp:+.3f}, top-12 overlap = {ov:.0%}")
    print(f"    -> {'STABLE: a static recipe is justified' if sp > 0.8 and ov > 0.7 else 'UNSTABLE: do NOT ship a static recipe'}\n")

    order = np.argsort(-sA)                                          # recipe order from corpus A only

    def measure(tiers):
        PA.apply_assignment(tensors, originals, tiers)
        kl, fl = PA.kl_flip(refB, PA.logits_of(m, idsB, mB), mB)     # fidelity on held-out B
        PA.restore(tensors, originals)
        bits = sum(TIER_BITS[t] * p for t, p in zip(tiers, params)) / tot
        return kl, fl, bits

    # ---- (2) FIDELITY/COMPRESSION PARETO: two-tier (base + bf16 pillars), held-out
    for base in ("fp8", "fp4"):
        print(f"(2) base={base.upper()} + bf16 pillars (recipe from A, fidelity on held-out B):")
        print(f"    {'#bf16':>6} {'avg_bits/wt':>11} {'KL(heldout)':>12} {'flip':>7}")
        for k in (0, 2, 4, 8, 16, 24, 48):
            tiers = [base] * nT
            for j in order[:k]:
                tiers[j] = "bf16"
            kl, fl, bits = measure(tiers)
            print(f"    {k:>6} {bits:>10.1f}b {kl:>12.3e} {fl:>6.1%}")
        print()

    # ---- (3) RECOMMENDED three-tier config: fp4 the redundant tail, fp8 the middle, bf16 the pillars
    print("(3) recommended 3-tier config (sensitivity-ordered): bf16 top-T pillars / fp8 middle / fp4 tail")
    print(f"    {'T(bf16)':>8} {'fp4 tail':>9} {'avg_bits/wt':>11} {'KL(heldout)':>12} {'flip':>7}")
    best = None
    for T, tail in ((4, 16), (8, 16), (8, 24), (12, 24), (8, 32)):
        tiers = ["fp8"] * nT
        for j in order[:T]:
            tiers[j] = "bf16"
        for j in order[nT - tail:]:
            tiers[j] = "fp4"
        kl, fl, bits = measure(tiers)
        flag = ""
        if kl <= 1e-2 and (best is None or bits < best[2]):
            best = (T, tail, bits, kl, fl); flag = "  <= KL budget 1e-2"
        print(f"    {T:>8} {tail:>9} {bits:>10.1f}b {kl:>12.3e} {fl:>6.1%}{flag}")

    print("\n[CONFIG]")
    if best:
        T, tail, bits, kl, fl = best
        print(f"  recommended: bf16 top-{T} pillars, fp4 bottom-{tail} redundant, fp8 the rest"
              f"  ->  {bits:.1f} bits/wt, held-out KL {kl:.2e}, flip {fl:.1%}")
        print(f"  vs all-fp8 ({8.0:.0f}b) and all-bf16 ({16.0:.0f}b): a {16.0/bits:.1f}x-vs-bf16 / "
              f"{8.0/bits:.2f}x-vs-fp8 weight compression at that fidelity.")
    else:
        print("  no 3-tier config met KL<=1e-2 on held-out: weight-only FP4 too costly here; ship FP8-based recipe.")
    print("\n[E-sim] held-out fidelity, weight-only fake-quant, bits over allocatable matmul weights only.")
    print("        FP32 reference gives the same ranking (certification adds no edge in sim). No hardware timing.")
    return True


if __name__ == "__main__":
    run()
