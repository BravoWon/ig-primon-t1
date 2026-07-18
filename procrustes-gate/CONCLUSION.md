# Procrustes-Import Gate — Conclusion (CLOSED 2026-07-18): the map is a CHANNEL, not a SOURCE

**Question.** DeltaSheaf proved the *isolated* geometry of an LLM ensemble is task-inert (arbitrary-basis
artifact). This gate asked the complementary question: is the exploitable geometry the **translation layer
between** distinct models — the orthogonal alignment map — rather than any single system's internal structure?
Rotation-only (Procrustes) because compression starves gradients (the Smith/Möbius result); rotation preserves.

## What is banked

**1. v1 — the mechanism is REAL (solid).** Pair Qwen2.5-3B (S, strong) → OLMo-2-1B (G, weak), matched hidden
dim 2048. Orthogonal Procrustes fit on both-correct anchors, transplant `R·h_S` decoded through G's own head:
**recovery 98.5%** on items G fails but S knows, vs every control at floor (random-R 6.2%, identity 8.7%,
wrong-item 29.3%, self 5.5%); holdout aligned cosine +0.315 vs random 0.000. Orthogonal rotation genuinely
transports a representation across independently-trained models — semantic communication where the Smith-chart
compression destroyed it. **Honest caveat:** final-layer `h_S` ≈ the answer, so v1 is the *routing-trivial*
regime — it proves the channel carries, not that it manufactures.

**2. Heterogeneity NO-GO on the blind spots (solid).** For *capability* transfer you need a specialist that
holds what the generalist lacks. Qwen2.5-Math-7B (specialist) on the 322 DeltaSheaf blind spots = **27.0%**
(chance 25%; Qwen-7B generalist 25.8%; gap +1.2%). The math specialist does not hold them either → the
pre-registered heterogeneity precondition FAILS, transplant not run. **Finding:** the blind spots are
**model-universal epistemic voids** (ensemble + scale + specialist all fail) — which is why CODA-2's
*external* fact-injection was the only thing that filled them. You cannot rotate into existence what neither
system possesses.

**3. GPA "terrain between" — the aligned consensus is an INERT AVERAGE (solid).** Generalized Procrustes
co-registration of the dim-matched trio (Qwen-3B / SmolLM2 / OLMo-2, native hiddens) into one consensus frame U.
- *Alignment is real and matters:* aligned consensus 62.7% vs raw-average 53.7% (co-registration recovers
  clean signal from frame-mismatch); universal read is partial and **asymmetric** — the strong model
  transplants everywhere (76–92%), the weak models don't read back into the strong (8–21%).
- *But the terrain holds no emergent signal:* aligned consensus **62.7% = best single map 62.6%**. Aligning
  three systems manufactured **nothing beyond the strongest source**. Not more than the sum of its maps.

**4. Uncollapsed separability — no geometric seam beyond difficulty (solid).** Directly testing the
"twisted-cohomology / spectral-gap" claim: do 0-of-5 blind (delusion) and 5-of-5 clean (truth) items form
distinct classes in the raw 5,120-D pairwise-residual space, above a label-free difficulty baseline?
Difficulty alone AUC **0.988**; geometry (5120→64) 0.798; geometry+difficulty 0.975 (< difficulty); geometry
increment over difficulty **−0.013, CI [−0.034,−0.002]**. The "distinct cohomology classes" are the
difficulty extremes (hardest vs easiest), **not** a geometric obstruction. The scalar collapse destroyed no
hidden seam — there was none beyond difficulty.

## The resolution
**The map between systems is the real geometry and the necessary channel — but it is a channel, not a source.**
It transports whatever the source holds (v1: 98.5% because S knew), bounded by the source; it does not
manufacture task signal (GPA consensus = best map; blind spots stay empty because no source holds them). The
program's two survivors — **grounding** (training-time) and **content-import** (CODA-2, 81.7%) — are both
*complementary content flowing between systems*; Procrustes is the confirmed **mechanism** of that flow and
never its **substance**. The lever is the content; the map is how it travels.

## Why the Feynman blueprint confirms the null instead of breaking it
Canonical geometry (ε-factorization, graph-Laplacian, twisted cohomology) *reveals* structure that exists —
Feynman integrals **are** periods of algebraic varieties, with genuine motivic content a canonical basis
exposes. Inter-model disagreement carries **one bit of real content — difficulty — and nothing more.** So
every basis, collapsed or uncollapsed — scalar residual, area, volume, cycle norms, restriction maps, inverse
maps, the whitened 320-D cochain complex, the rank-7 invariant spectrum, the GPA consensus, the raw 5,120-D
residual — returns difficulty and only difficulty. Match the basis to content that exists and it trivializes;
there is none here beyond the difficulty scalar. **The geometric program is triple-sealed: collapsed
(DeltaSheaf v0.2–v0.10), co-registered (GPA), and uncollapsed-residual (separability). Same floor every time.**

*(Reproduce: `build_v1.py` · `score_math.py` (`RESULTS_hetero.md`) · `build_gpa.py` (`RESULTS_gpa.md`) ·
`../deltasheaf-v02/separability.py` (`RESULTS_separability.md`). Pre-registration: `PREREG.md`.)*
