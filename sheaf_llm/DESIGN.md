# The Small-Footprint LLM the Evidence Actually Supports
### (the de-mythologized "Prime Crystal" — every brick earned through a gate)

Ten falsifiable gates on a 6 GB GPU. We threw out the mysticism and kept what survived a curve that
moved the right way. Here is the design the evidence points to, and the honest gap to a real model.

## What we REFUTED (so we don't rebuild it)
- **Sheaf/structure as a compressor of a pretrained model** — factoring weights (v0/v1), allocating
  bits (v2/v3 on a strong base, v4), the sheaf-specific λ₁ signal (v4 = AWQ-in-a-cape). Plain dense +
  strong group-wise 4-bit beat all of it. Pretrained weights are full-rank; there is no free structure.
- **Native low-rank structure** (v5) and **MoE at toy scale** (v6 noise / v7 anti-scaling). Monarch (v8): a tie.
- The grandiose layer entirely: constants-from-topology, RH, "deterministic → no hallucination."

## What we VALIDATED (the bricks, with numbers, toy-scale char-LM)
1. **One-object-branched backbone (v9).** ONE shared base block + cheap per-layer *restriction-map*
   branches (FiLM / affine-diagonal) = dense quality at **~25% of the MLP params** (affine −0.47%).
   This is ALBERT-style cross-layer sharing made geometric: layers are *views of one object*. The
   saving scales with depth: shared ≈ base + N·(tiny branch) ⇒ ~N× fewer block-matrix params for deep N.
2. **Native low-bit weights (v8 / v10 / v11).** Ternary {−1,0,+1} — and the **prime {0,1,2} =
   Void/Identity/Prime** encoding (affine-equivalent, same 1.58 b/w) — is **near-lossless** at ~10×
   fewer bits, *because it is trained in natively* (impossible post-hoc). The middle level is free
   compute (~28% sparsity) on top of the bits. **2-bit knob (v11):** the asymmetric **{−1,0,1,3}** set
   beats ternary by ~2.5% and beats symmetric 2-bit — it keeps 0=sparsity (~28%) AND adds a heavy-tail
   outlier level (the '3' fires only ~4%, exactly the weights ternary clips). So the low-bit brick is a
   DIAL: 1.58 b/w ternary (max compression) ↔ 2 b/w {−1,0,1,3} (outlier-aware, more quality headroom).
   [methodology: pin a FIXED corpus before quoting final numbers — the training glob currently ingests
   newly-written scripts, so absolute vals drift across runs; within-run comparisons are valid.]
3. **The sheaf's real jobs:** ROUTER (which expert/branch fires) and VERIFIER (the +0.314 OOS value
   model) — never a compressor.

## THE DESIGN (the "new class")
A deep transformer where:
- the block is **ONE shared base** (attention + MLP) reused across all layers,
- each layer adds a **tiny restriction-map branch** (FiLM/affine; LoRA for more capacity),
- all weights are **native prime/ternary {0,1,2}** with Void-sparsity,
- at scale, experts are **MoE-routed** (hot set resident, cold streamed = the honest 70B-in-6 GB),
- a **sheaf consistency/value head** verifies outputs (the one thing that actually worked).

## The arithmetic reaches the envelope
Two independent levers that COMPOUND:
- cross-layer sharing: ~N× fewer unique block-matrices (deep model, ALBERT-grounded),
- native ternary: 16/1.58 ≈ **10×** fewer bits per param.
Combined, an aggressive deep design is ~tens-to-~100× fewer bits than fp16 dense. A 70B-class model
(140 GB fp16) → **single-digit GB** for the resident backbone, with MoE streaming the rest. **The 6 GB
envelope is reachable by this path** — not by re-representing weights (refuted), but by *sharing +
native low-bit*, which owe nothing to the cape.

## The HONEST gap (what's unproven)
Every brick is validated **piecewise at toy scale** (1.8 M-param char-LM). The bet is that they
**compound** (one-object × ternary × MoE) and that quality **holds at billions of params** — which a
6 GB GPU cannot show. That requires real compute (multi-GPU / cloud). Next, in order:
1. Stack the validated bricks (shared-base + ternary) in one toy model — does near-lossless survive *compounding*?
2. Scale depth (more layers) — does one-object sharing hold as N grows (ALBERT says yes to ~24)?
3. Real-scale run on proper GPUs — the only thing that converts this validated DESIGN into a model.

## What "accomplished the goal" means, honestly
We did not build a 70B-in-6 GB model (a 1660 Ti can't). We **killed the mystical compressor with eight
negatives, found the real design with two earned positives, and showed its arithmetic reaches the
envelope** — every component falsifiable, reproduced, and free of the grandiose layer. That *is* the
deliverable: not the Prime Crystal's cosmology, but the buildable small-footprint architecture hiding
underneath it, with the evidence to back each piece and an honest plan to scale.
