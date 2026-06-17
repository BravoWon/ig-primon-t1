"""Driftwave grounding: the ACTUAL per-tensor FP4 sensitivity ranking on trained GPT-2-small.

Answers the concrete question "which weight tensors are the load-bearing 'pillars' vs the FP4-safe
'redundant' ones" with bytes, not narrative. Importance(i) = exact leave-one-low KL marginal:
  s(i) = KL(float64_ref || [tensor i -> FP4, rest bf16]) - KL(float64_ref || [all bf16]).
High s(i) = a pillar (keep high precision); low s(i) = FP4-safe.

Honest caveats baked in (the driftwave corrections):
  - The literal F-app test (results_fapp_reference_swap.txt) showed an FP32 reference gives the IDENTICAL
    ranking in sim (Spearman +1.0): so this list is NOT a certified-reference exclusive; FP32 picks the same.
  - all-FP4 weight-only is catastrophic (KL ~3.7-4.5, ~90% flip): "survived FP4" is the wrong frame.
    This is which tensors the allocator REFUSES to send to FP4, not a triumphant catalog of survivors.
[E-sim] CPU, float64 reference, 48 weight tensors, weight-only fake-quant.
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


def run():
    tok = GPT2TokenizerFast.from_pretrained(PA.GPT2_PATH); tok.pad_token = tok.eos_token
    enc = tok(TEXTS, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    ids, mask = enc.input_ids, enc.attention_mask
    m = PA.load_model(torch.float64)
    tensors = PA.allocatable(m); originals = PA.snapshot(tensors); nT = len(tensors)
    ref = PA.logits_of(m, ids, mask)

    PA.apply_assignment(tensors, originals, ["bf16"] * nT)
    kl_base = PA.kl_flip(ref, PA.logits_of(m, ids, mask), mask)[0]
    s = np.zeros(nT)
    for i in range(nT):
        tiers = ["bf16"] * nT; tiers[i] = "fp4"
        PA.apply_assignment(tensors, originals, tiers)
        s[i] = PA.kl_flip(ref, PA.logits_of(m, ids, mask), mask)[0] - kl_base
    PA.restore(tensors, originals)

    names = [n for n, _ in tensors]
    order = np.argsort(-s)
    print(f"[FP4 sensitivity ranking]  trained GPT-2-small, 48 weight tensors, exact leave-one-FP4 KL marginal\n")
    print(f"  PILLARS (highest sensitivity -> the allocator keeps these OUT of FP4):")
    for r in order[:10]:
        print(f"    {names[r]:>16}   leave-one-FP4 KL marginal = {s[r]:.3e}")
    print(f"\n  FP4-SAFE (lowest sensitivity -> the redundant interpolation, cheapest to crush):")
    for r in order[-10:][::-1]:
        print(f"    {names[r]:>16}   leave-one-FP4 KL marginal = {s[r]:.3e}")
    # structural read: which layer depths / which sublayer types dominate the pillars
    top12 = [names[r] for r in order[:12]]
    n_attn = sum("attn" in n for n in top12); n_mlp = sum("mlp" in n for n in top12)
    depths = sorted(int(n.split(".")[0][1:]) for n in top12)
    print(f"\n  structure of the top-12 pillars: attn={n_attn}/12  mlp={n_mlp}/12   layer depths={depths}")
    print(f"  ratio max/min sensitivity = {s[order[0]]/ (s[order[-1]]+1e-30):.1f}x  "
          f"(spread of importance across tensors)")
    print("\n[caveat] FP32 reference gives the SAME ranking (F-app Spearman +1.0); all-FP4 is catastrophic")
    print("         (KL ~3.7-4.5). This is which tensors are REFUSED to FP4, not 'survivors'. [E-sim]")
    return s, names


if __name__ == "__main__":
    run()
