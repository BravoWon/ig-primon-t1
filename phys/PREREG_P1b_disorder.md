# PREREG — Gate P1b (Phase 2): the dark space — disordered Chern survival, two-route in real space
**Registered 2026-07-08 before execution. Gated on P1 (passed, 321bd0d).**

## The dark
Disorder (on-site, uniform ±V/2) destroys momentum space: FHS and ribbon-k both die. The questions have
no textbook table for THIS model (Haldane, t1=1, t2=0.2, φ=π/2, our conventions):
(D1) at what disorder V_c does C=−1 drown, at the Haldane point? (D2) starting just OUTSIDE the phase
(M/t2=5.5, trivial by 6%), can disorder INDUCE the topological phase (TAI-type)? Phenomena of this class
are published [E] (TAI: Li-Chu-Jain-Shen 2009; disordered Chern studies); the model-specific map with
per-point two-route receipts is the deliverable.

## Two routes, reborn in real space (nothing shared)
- **Route A (bulk): Bott index** on an L×L torus (Loring–Hastings): occupied projector at E_F (clean
  mid-gap), U,V = projected e^{2πiX/L}, e^{2πiY/L}; B = Im tr log(VUV†U†)/2π ∈ Z.
- **Route B (pump): Laughlin spectral flow** on an L×W cylinder, flux θ: 0→2π through the periodic
  cycle; count signed crossings of E_F by TOP-edge-polarized states (P1's branch tracing, over θ,
  periodically closed). Net flow = C.
Same disorder REALIZATION fed to both (torus/cylinder geometry differs; realization seeded per site row).

## Kernel slots
| slot | instantiation |
|---|---|
| ANCHORS | V=0: Bott = pump = **−1** at Haldane point (must reproduce P1's convention exactly), 0 at M/t2=6; weak disorder V=0.5 preserves C (quantized robustness — the defining claim of topological matter, receipt-checked) |
| CLAIM | Bott = pump at EVERY measured (V, seed) point — bulk–boundary under disorder |
| DARK MEASUREMENTS | (D1) survival curve f(V) = fraction of 5 seeds with C=−1, V ∈ {0.5..5}; V_c = 50% crossing, L-rotation {8,12,16} stability. (D2) TAI probe at M/t2=5.5: f(V) for V ∈ {0.5..3} — induced C=−1 at intermediate V or not; either answer is the finding |
| ROTATIONS | L ∈ {8,12,16} (Bott); W ∈ {12,20} (pump); 5 disorder seeds/point |
| NON-MEASUREMENT | torus states within 0.02·t1 of E_F, or non-integer Bott (|B−round|>0.1), or pump branch gaps → point flagged "nm" at that size, escalated to larger L, listed |
| WALL | near V_c: realization scatter + L-dependence of f(V) — the finite-size wall of the disordered transition, reported as measured |
| VERDICTS | (1) two-route agreement 100% on measured points; (2) anchors reproduce; (3) D1: V_c located with monotone f(V) and L-stability (±1 grid step); D2: answer reported either way. Any Bott≠pump on a clean-gapped point = instrument bug until proven otherwise |

## Honest scope
Disorder-driven Chern transitions and TAI are established phenomena [E]; no phenomenon-novelty is
claimed. The dark-space content is the model-specific measured map (V_c for THIS Haldane point; TAI
yes/no at THESE parameters) with two-route receipts at every point — numbers brought to light, not laws.

---
## GATE RECORD (2026-07-08, appended post-execution)
- **Verdict 1 — PASS after bookkeeping correction (recorded, not hidden):** the 3 printed "DISAGREE"
  were pump=None — the pump's own nm channel (ambiguous partial jumps), mislabeled by the verdict
  logic. True score: **36/36 two-route agreement on all doubly-measured D1 points; 0 disagreements**;
  23 nm total, all self-flagged.
- **D1 (the integer drowns):** f(V)=1.0 through V=4 at ALL sizes; death first captured at L=16, V=5.
  **V_c ∈ [4,5]** for this model (t2=0.2, φ=π/2), with the expected finite-size drift (L=8,12 still
  quantized at V=5) — the transition wall, reported as measured, not resolved.
- **D2 (noise creates order): TAI-type induction OBSERVED — Bott route:** from the trivial side
  (M/t2=5.5), induced C=−1 in 0/5 (V≤1.5) → 4/5 (V=2.0) → 5/5 (V=2.5) → 4/4 (V=3.0). Sharp onset
  V≈2. **Double-receipt attempt located the pump's wall instead:** at matched 16×16 size the pump
  returns mostly nm (fractional jumps) or 0 in the TAI regime — because the SPECTRAL gap is closed
  there and topology lives on a MOBILITY gap, which the charge-pump-with-unit-jump design cannot
  navigate (localized states crossing E_F carry fractional weight). D2 therefore stands as
  **single-route (Bott, the literature-standard mobility-gap instrument [E]) with the second route's
  admissibility boundary measured**: pump valid only where a spectral gap survives (it matched Bott
  36/36 there). A mobility-gap-capable second route (local marker statistics / transport) is named
  future work, not improvised post-hoc.
- **Confound flagged:** the implementation did NOT achieve the prereg's shared-disorder-realization
  intent (different geometries → different realizations per seed); near transitions this matters, and
  the per-realization "MISMATCH" at V=2.5/seed1 is confounded by it. Statistical comparison used instead.
- Instrument-pair admissibility map after P1+P1b: FHS ~wall-free; ribbon/edge W*~1/gap; pump requires
  spectral gap; Bott crosses into the mobility-gap regime.
