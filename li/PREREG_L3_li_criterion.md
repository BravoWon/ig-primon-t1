# PREREG — Gate L3: two-route Li coefficients on the critical line (kernel-instantiated)
**Registered 2026-07-06, before execution. Continues IG-PRIMON-T1 Phase 0 (v0.1, hash 0A8B…7918) and
instantiates `GATE_KERNEL.md` on the Li-criterion substrate. Amendments by versioned diff only.**

## Phase-0 closure recorded first
The registration's dangling cross-check is hereby closed: σ₃ derived zero-free from the locked c₁
(σ₃ = c₁/2 − λ(3) + 1 = −1.111582314521059227626682e−4) **matches Lehmer (1988) Table 5 row 3
(−.000111158231452105922762668239) to all 25 computed digits.** Cumulant route ≡ zero route, verified
across a 38-year-old published table. Phase 0 of v0.1: COMPLETE.

## Substrate
Li's criterion (Li 1997; Bombieri–Lagarias 1999): **RH ⟺ λ_n ≥ 0 for all n ≥ 1**, where
λ_n = Σ_ρ [1 − (1 − 1/ρ)^n] (nontrivial zeros, conjugate-paired). Two *independent* computational routes:
- **Route A (zeros):** direct sum over the first N nontrivial zeros + explicit analytic tail budget from
  dN(T) ≈ (2π)⁻¹ log(T/2π) dT. Touches the zeros.
- **Route B (cumulants, zero-free):** λ_n = Σ_{j=1}^n (−1)^{j+1} C(n,j) σ_j, with σ₁ = 1 + γ/2 − ½log 4π
  (closed form) and σ_j (j≥2) from the Taylor coefficients g_k of log(ε ζ(1+ε)) via the June lemma chain
  (full-zero-set power sums minus the closed trivial part λ_odd(k)−1; functional-equation flip). Computed
  by high-dps Cauchy contour integrals of log(ε ζ(1+ε)) — **never touches a zero.**

## Kernel slots
| slot | instantiation |
|---|---|
| CLAIM | (i) Routes A and B agree within the pre-stated tail budget for n = 1..40; (ii) λ_n > 0 throughout (consistent with RH; NOT a novelty claim — Keiper computed to n=7000); (iii) the residue λ_n − trend stays bounded relative to trend, where trend = (n/2)(log n − log 2π + γ − 1) (the RH fixed-point envelope) |
| CHAOS REFERENCE / fixed point | the smooth RH trend; **meaning = residue** (bounded oscillation ⟺ zeros on line) |
| ANCHORS | (a) λ₁ = 1 + γ/2 − ½log 4π ≈ 0.0230957 closed form, both routes; (b) the σ₃/Lehmer 25-digit match above; (c) **violation-sensitivity anchor**: inject a synthetic off-line zero quadruple {ρ*, 1−ρ*, conj} at ρ* = 0.95 + 2i into Route A — λ_n must go clearly negative / oscillate with growth ~|1−1/(1−ρ*)|^n ≈ 1.107^n by n ≤ 40. If the instrument cannot see a planted violation, no null about RH-consistency means anything |
| ROTATIONS | N-zeros ∈ {200, 500, 2000} (Route A stability under truncation); dps ∈ {40, 60} and contour radius ∈ {1.5, 2.0} (Route B stability); n-range split halves |
| VERDICTS | pre-registered: **AGREE** iff |A−B|/tail-budget ≤ 1.5 for all n ≤ 40 AND anchors (a),(c) pass. **INSTRUMENT-LIMIT** iff anchor fails or budget exceeded with structure (locate n, N, dps). Any λ_n < 0 (real, surviving both routes and rotations) would be… reported very carefully indeed, with the assumption of instrument error until proven otherwise (law #1) |
| NON-MEASUREMENT | binomial-sum cancellation in Route B grows ~2^n; if dps head-room is exhausted the verdict is "not measured at this n", never "disagreement" |
| RESIDUE | λ_n − trend, plotted with both routes overlaid; under RH it oscillates boundedly — the critical line's own "meaning off the fixed point" |
| LEDGER | this file + `gate_L3_li.py` receipts; failures banked; no novelty claims (occupied territory per the June lit-gate lineage: Keiper, Bombieri–Lagarias, Coffey, Voros) |

## Honest scope
This gate validates a *two-estimator instrument* on the Li substrate and demonstrates violation
sensitivity. It does **not** test RH in any region not already covered by published zero verification;
n ≤ 40 probes zeros up to height ~O(n) only. The value is (1) the closed Phase-0 chain now extended to
the λ_n level, (2) the kernel's seventh domain instantiation, (3) the platform for any later push to
large n (where genuinely un-probed territory begins, and where Route B's cancellation wall is the
known dragon — that wall's exact location is itself a deliverable).
