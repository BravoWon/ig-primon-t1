# PREREG — Gate C1: Choptuik critical collapse — the gravitational phase transition, two-route
**Registered 2026-07-09 before execution. First gate of the cosmos arc (PR #13).**

## Substrate
Massless scalar field, spherical symmetry, polar-areal gauge (Choptuik 1993):
Φ=∂_r φ, Π=(a/α)∂_t φ; constraints ∂_r a/a = (1−a²)/(2r)+2πr(Π²+Φ²), ∂_r α/α = ∂_r a/a+(a²−1)/r;
evolution ∂_tΦ=∂_r(αΠ/a), ∂_tΠ=r^{−2}∂_r(r²αΦ/a); m(r)=(r/2)(1−a^{−2}). Gaussian pulse, amplitude p.
Supercritical → horizon (2m/r→1); subcritical → dispersal. **Corrections adopted:** classical GR only,
no quantum-gravity claims; γ measured from scaling, not solver death.

## Two routes + external anchor
- **Route A (supercritical):** M_BH(p) ∝ (p−p*)^γ → γ_A from the log-log fit.
- **Route B (subcritical):** max Ricci |R|=8π|Φ²−Π²|/a² ∝ (p*−p)^{−2γ} (Garfinkle–Duncan) → γ_B.
  Different branch, different observable — internal two-route.
- **External anchor:** Gundlach's perturbation-eigenvalue value **γ = 0.374** (independent method
  entirely — the published constant plays the role AW theory played in node 6).

## Verdicts (fixed)
1. γ_A from a ≥2-decade linear window, R²>0.99, with the window rule pre-fixed (largest window with
   locally stable slope).
2. |γ_A − γ_B| ≤ 0.03 (branch consistency).
3. Both within **0.374 ± 0.02** (generous: first-build unigrid code, no AMR).
4. **The wall:** minimal grid N*(ln(p−p*)) and the bisection's f64 floor (|p−p*|/p* ≳ 1e−13) — the
   silicon boundary of a spacetime singularity search, mapped.
Controls/NM: convergence check (N vs 2N pre-collapse agreement); dispersal-limit mass→0 anchor;
runs with ambiguous horizon detection → nm, listed. Echoing period Δ≈3.44: stretch goal, expected nm
without AMR (declared).

## Honest scope
Choptuik 1993 / Gundlach reviews: fully occupied. Deliverable = the receipt genre on a gravitational
critical phenomenon + the resolution-wall law of unigrid collapse — and the kernel's 10th domain.

---
## GATE RECORD (2026-07-09, appended post-execution)
- **Verdicts 1–3: FAIL. Verdict 4 (the wall): DELIVERED — the failure IS the measurement, three ways:**
  (i) mass floor M ≈ 3.7e−3 ≈ dr/4 below ε~1e−5 (the smallest hole a cell can hold — 20 digits of
  bisected p* meaningless past it); (ii) curvature ceiling R_max ≈ 4.66e3 ≈ O(dr⁻²); (iii) the
  convergence control: M(N) vs M(N/2) differ 153% at ε=1e−3 — no converged scaling window exists at
  N=4096 unigrid (local slopes wander 0.23–0.52; the auto-fit's γ_A=0.247 / γ_B=0.172 are
  wall-contaminated numbers, NOT measurements of γ).
- **Instrument finding of record: unigrid at dr=0.0146 reaches ~1.5 decades from criticality and no
  further — re-deriving, with receipts, why AMR was the enabling instrument of Choptuik 1993** (the
  critical echo cascade shrinks like e^{−nΔ}; a fixed grid cannot follow it).
- Bugs on the tombstone: Picard constraint saturation (fixed pre-launch: exact linear solve);
  lapse exp-overflow in deep-collapse gauge normalization (cosmetic, noted for v2: normalize in log).
- **Escalations, costed:** (a) uniform N≈50–75k (mass floor →2e−4, ~2 more decades): ~5–9 h CPU,
  feasible-heavy; (b) Garfinkle double-null scheme (naturally focusing — the right v2); (c) true AMR
  (the full instrument, a project). p* = 0.03353442212120 (this grid, this pulse) stands as the
  bisection artifact; γ remains NOT MEASURED here, honestly.

---
## C1b RECORD (2026-07-09, escalation (b) attempted — WIP, honestly parked)
Double-null scheme drafted from scratch (derivations anchor-checked analytically: (r·r,u),v = −a²/4
flat-exact; m≡0, σ,u≡0 flat). Instrument arc: (i) axis-sliver bug — grid-aligned cumsums missed the
moving-axis segment ∫_u^{v_ia}, making r,u oscillate as 1/r → σ blow-up even in near-flat space;
FIXED (offset-aware trapezoid integration; flat + weak-field now stable through full march).
(ii) Strong-field axis instability REMAINS (σ overflow near collapse → NaN before clean horizon
detection). Diagnosis of record: naive (r, σ, φ) characteristic marching is axis-unstable at strong
fields — which is precisely WHY Garfinkle 1995 evolves his regularized h-variables; we re-derived the
necessity of his variable choice empirically. **v2 next rung: implement Garfinkle's actual h-variable
scheme (from the paper, not memory), or the N≈50–75k uniform-grid route (costed ~5–9 h).** γ remains
NOT MEASURED; no number is reported from the unstable code.
