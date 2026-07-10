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

---
## C1c GATE RECORD (2026-07-09, Garfinkle's actual h-variables from gr-qc/9412008 — **γ MEASURED**)
**Instrument:** `gate_C1c_garfinkle.py`, faithful to the paper: evolve h with Φ=h̄; g=exp(4π∫q dr),
q=(h−h̄)²/r; ḣ=(g−ḡ)(h−h̄)/2r, ṙ=−ḡ/2; axis Taylor forms on first 3 pts (eqs. 10–14); grid rides
ingoing rays (natural focusing); midpoint-insertion regrid; MOTS at ḡ/g<0.02, M=r/2. N=800 rays.
The C1b strong-field instability is GONE — h-variables carry p=0.05–0.1 straight to clean MOTS.
p*(N=800) = 0.01280497040512 (bracket 2.1e−14, the f64 floor again); p*(N=1600) = 0.012794010326.

**VERDICTS (registered rules, corrected fitter — see fitter disclosure below):**
1. **PASS** — γ_A = **0.3681**, R² = 0.9997, over **4.00 decades** (ε ∈ [1e−5, 1e−1]).
   Grid-converged: independent re-bisection + refit at N=1600 gives γ_A = 0.3696 (|Δγ| = 0.0015);
   with ε defined per-grid the ε=1e−3 masses agree to 2%. (N=400 is below the scheme's stability
   edge — blows up unphysically — so the in-run N/2 control recorded instability, not error.)
2. **FAIL** — |γ_A − γ_B| = 0.066 > 0.03. γ_B (registered window rule) = 0.3024 from the only
   qualifying window (1.0 decade, ε ∈ [3.2e−3, 3.2e−2], R² = 0.999).
3. **FAIL** (as the registered conjunction) — γ_A within anchor (|0.3681 − 0.374| = 0.006 ✓);
   γ_B off by 0.072 ✗. **The Gundlach anchor is HIT on the mass branch — first γ measurement of
   the arc after C1 (wall) and C1b (instability).**
4. Walls mapped again, now for the focusing instrument: mass floor M ≈ 1.38e−3 below ε ≈ 3e−9
   (2.7× below C1's floor with 800 rays vs 4096 cells — the focusing dividend); curvature ceiling
   max h₁² ≈ 1.06e5 below ε ≈ 1e−9; stability edge between N=400 (blow-up) and N=800 (convergent).

**Fitter disclosure (Law #6 applied):** the in-run `fitwin` had bugs provable by inspection —
median over ALL local slopes (plateau-poisoned), non-contiguous keep-span, signed decade width. Its
advisory numbers (γ_A=0.2610, γ_B=0.3044, "−1.0 decades") are disowned. `gate_C1c_refit.py`
implements the FULL registered conjunction (largest contiguous window with locally-stable slopes
AND window R²>0.99); the plateau self-excludes (R²=0.29). Raw arrays untouched.

**Route-B diagnosis of record:** the DSS fine structure (Gundlach; Hod–Piran) modulates the
curvature branch ~12× stronger than the mass branch (fitted ln-amplitude 0.26 vs 0.021). Local
slopes oscillate 0.15↔0.64 with ~2-decade spacing, so no long window has "locally stable slope" —
the registered rule and the wiggle are structurally incompatible on this branch. Wiggle-aware
4-parameter fit gives γ_B ≈ 0.33 (advisory, NOT registered). Verdict 2/3 FAILs stand as registered.

**Echo stretch (declared "expected nm"): upgraded to DETECTION, not measurement.** Four channels:
(i) h₁(u) zero-crossing interval ratios at p*(1−1e−12): 13 crossings, Δ ∈ [2.9, 5.6] (noisy — ~3
echoes fit above the 1e−12 floor); (ii) Route-A dense wiggle (quarter-decade scan): Δ = 3.85;
(iii) Route-B dense wiggle: Δ = 2.65; (iv) slope-peak spacing by eye: 2.0 decades → Δ ≈ 3.39.
All bracket/are consistent with Choptuik Δ = 3.44; none is a receipt-standard measurement.
**Δ: DETECTED (periodic term improves both routes' fits ~2×), NOT MEASURED.**

Artifacts: `gate_C1c_results.json` (+refit/echo/wiggle blocks), `gate_C1c_dense.json`,
`gate_C1c_conv1600.json`, `gate_C1c_refit.py`, `gate_C1c_echo.py`, `gate_C1c_wiggle.py`.
Bugs on the tombstone: dispersal stall (regrid halves du geometrically; fixed with the physical
early-exit m_out < 1e−3·m_init); g-overflow at strong field (clipped exponent, cosmetic).
**Costed next rungs:** Δ to receipt standard needs either deeper ε (higher-precision p*, mp-float
bisection) or the field-profile echo overlay (Garfinkle Figs. 2–6); Route-B γ needs a
wiggle-native registered rule (pre-register the 4-param fit as primary on a fresh gate).
