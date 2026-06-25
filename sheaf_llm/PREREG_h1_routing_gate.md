# Pre-registration — Čech H¹ global-consistency obstruction as a routing/verification signal

**Status: `[GATE]` (derive-before-numerics). NOT a validated design component.** This is the one
"coherence-transport" dimension with a falsifiable in-program core (the other three: #4 quarantined; #1
reduces to entropy-regularized routing; #2 to natural-gradient/Fisher optimization — both standard,
de-mythologized, `[E]`). H¹ enters the design **only if** it passes the decisive hypothesis below.

## Why this one is worth a gate
The sheaf's single banked win in this whole program was as a **verifier** (`sheaf_value_model.py`,
+0.314 OOS), not a compressor. Čech H¹ measures *exactly* whether local agreements glue into a global
section: **H¹ = 0 ⟺ a consistent global assignment exists; H¹ ≠ 0 ⟺ contextual** (locally/pairwise
consistent yet jointly irreconcilable). That is a precise, computable number — a candidate verifier of
*joint* consistency. The theorem is rigorous. **What is untested is whether it is operationally useful.**

## The construction is the ballgame (state it before numerics)
- **Cover:** a set of overlapping local contexts `U_i` over an input (overlapping windows / experts /
  sense-assignment sources). The choice of cover + sections *is* the result; a bad cover makes H¹ noise.
- **Local sections:** each context's assignment/distribution over a shared discrete latent on overlaps
  (sense / referent / POS of shared tokens).
- **Cohomology:** Čech complex over the nerve of the cover; compute H⁰ (does a global section exist) and
  H¹ (the obstruction), via the Abramsky–Brandenburger possibilistic construction (compatibility
  relations → cocycle/coboundary rank).

## Hypotheses (pre-registered)
- **H1:** inputs with H¹ ≠ 0 have higher model error / lower verifier confidence / higher human-rated
  ambiguity than H¹ = 0 inputs.
- **H2 — DECISIVE (the subword-gate shape):** H¹ predicts failure **better than the pairwise-disagreement
  baseline** (count/severity of first-order inconsistencies). The whole question is whether the
  *higher-order* gluing signal adds over pairwise — exactly as "does grounding add over subword."

## Controls
- **Pairwise-inconsistency baseline** (cheap, first-order) — the bar H2 must clear.
- **Shuffled-cover control** — H¹ from a meaningless cover must NOT predict (else we're fitting noise).
- **Known-contextual anchors** — Abramsky-style tables with proven H¹ ≠ 0 (PR box, Hardy) verify the
  computation itself is correct before any LLM claim.

## Falsifier
H¹ does **not** beat the pairwise baseline (no margin) ⇒ the obstruction is **mathematically real but
operationally inert**, joins entropy-reg/natural-gradient as "real, known, not the lever," and does
**NOT** enter the design as a veto. This is a live outcome and an acceptable result.

## Instantiation (runnable on this harness, CPU-fine)
1. **Sanity:** synthetic contextual vs non-contextual systems — H¹ computation reproduces known anchors.
2. **Discourse:** garden-path / globally-ambiguous sentences vs matched unambiguous controls — does
   H¹ ≠ 0 flag the genuinely non-reconcilable ones? (the cited application — *to be tested, not assumed.*)
3. **Model-in-the-loop:** overlapping context windows over real text → local next-token/sense
   distributions → glue test; correlate H¹ (and the pairwise baseline) with model error.

## Metrics
AUROC of H¹ vs pairwise baseline at predicting failure/ambiguity; anchor-detection on known-contextual
systems; the **margin** of H¹ over pairwise (the only thing that licenses "blocks the route").

## Scope / honest limits
Computability of H¹ over large covers is unaddressed; the cover/section modeling choice is load-bearing
and adversarial; "mathematically blocks the route" is licensed **only** if H2 passes with margin — until
then H¹ is a `[C]` conjecture with a derivation, not a design veto.
