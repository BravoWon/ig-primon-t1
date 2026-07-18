# Procrustes gate — heterogeneity go/no-go on the DeltaSheaf blind spots (2026-07-18)

Pair: S = Qwen2.5-Math-7B-Instruct (specialist), G = Qwen2.5-7B-Instruct (generalist, cached). Matched
hidden dim 3584. Test bed proposed: the 322 blind-spot items (where the whole ensemble + Qwen-7B failed).

**RESULT: NO-GO. Math-7B = 27.0% on 300 blind spots (chance 25%; Qwen-7B 25.8%) — gap +1.2%.**

The pre-registered heterogeneity precondition (S ≫ G on domain X) FAILS: the math specialist does not hold
the blind spots either. Per the gate's hard-gate rule, the transplant is NOT run on this set — you cannot
transplant knowledge the specialist doesn't have (the same wall the homogeneous ensemble hit).

**Finding (closes a loop):** the DeltaSheaf blind spots are **model-universal holes** — defeated by the
5-model ensemble, a 7B generalist, scale (Qwen-7B), AND a domain specialist. Nobody's knowledge, general or
specialized. This is exactly why CODA-2's *external* fact-injection (81.7%) was the only thing that filled
them: the content isn't in any model's weights; it has to be imported.

**Consequences for the Procrustes gate:**
- v1 mechanism stays CONFIRMED (pairwise latent transplant works, 98.5%, controls-clean; routing-trivial
  regime noted) — `RESULTS_v1.md`.
- The blind spots are the WRONG test bed for *capability* transfer (they test knowledge no model holds).
- Two live directions: (a) source a genuine specialist-advantage set (hard competition-math MC) where
  Math-7B truly leads Qwen-7B; (b) the multi-way "terrain between" test — Generalized Procrustes
  co-registration of the dim-matched trio (Qwen-3B/SmolLM2/OLMo-2, dim 2048) into a shared frame, then
  test universal cross-read + whether the *aligned* consensus carries signal the *raw* stalks (v0.4–v0.10,
  all un-co-registered) did not. (b) is the one door our banked nulls do not close.
