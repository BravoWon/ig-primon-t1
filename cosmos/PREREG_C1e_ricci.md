# PREREG — Gate C1e: the true Ricci scalar — closing the Route-B loop
**Registered 2026-07-09 before execution. Cosmos arc, PR #13. Successor to C1d per its costed rung.
Instrument: `gate_C1c_garfinkle.py` scheme + `gate_C1e_ricci.py` observable; banked
p* = 0.01280497040512, p*(1600) = 0.012794010326; banked γ_A = 0.3681 (C1c verdict 1).**

## What the papers actually say (the C1e correction to C1d's diagnosis)
Re-read of gr-qc/9412008 (eqs 1, 4, 7, 8, 9) + fetch of Garfinkle–Duncan gr-qc/9802061: GD's
observable is "the maximum curvature at the position of the CENTRAL OBSERVER" — the axis. Deriving
the axis limit of R = 8π∇Φ·∇Φ in the paper's variables gives R_axis = −2π h₁²: **C1d's "proxy" WAS
the GD observable up to a constant.** C1d's finding therefore re-attributes: not proxy-vs-Ricci,
but AXIS-FIT RESOLUTION — the 4-point Taylor fit smears the deepest echo's h₁, clipping the max
harder as ε→0. C1d's V5 pass hid a directional drift (+0.0147 toward anchor per N-doubling;
tolerance 0.02 too loose to catch it). This prereg registers the resolution-robust observable.

## Observable (primary, fixed)
**max over run and slice of |R|**, with R computed at every ray from the paper's equations
(no axis fit involved off-axis):
  R = −16π Φ̇ (h−h̄)/(g r);  Φ′ = (h−h̄)r′/r (eq 4);
  Φ̇ = [ḡh̄ − h₀ − ∫₀^r (g−ḡ)h̄ dr̃/r̃]/(2r)  (d/du of eq 4 at fixed v via eqs 8, 9 + moving-axis
  boundary term h₀/2); axis Taylor overrides Φ̇→h₁/4, R→−2πh₁².
The GD-literal axis channel (max 2πh₁²) is tracked as a DECLARED-BIASED diagnostic, not a verdict.
Route A is not rerun (mass observable unchanged); V2 uses banked γ_A = 0.3681.

## Disclosure of prior looks
Instrument probe (2 runs, disclosed): ε=1e−3 → slice/axis ratio 1.61; ε=1e−6 → ratio 5.81 (the
under-resolution signature motivating this gate). Crude two-point slopes from those probes:
γ_slice ≈ 0.38, γ_axis ≈ 0.29. These two points are part of the registered dataset and carry no
independent verdict weight; all C1d peeks remain disclosed there.

## Data + pipeline (identical to C1d where not stated)
Eighth-decade ε = 10^(−k/8), k = 16..60 (ε ∈ [3.16e−8, 1e−2]), all 45 evolutions fresh (new
observable). Pipeline = C1d's calibrated 2-harmonic fit, P-scan [2.0, 9.0]×400, 200 permutation
surrogates, seed 20260709, half-split phase coherence. C1d's synthetic calibration covers this
pipeline unchanged (same code imported).

## In-run anchors (must pass BEFORE launch)
(a) Weak-field linearity: max|R| ∝ p² exactly as p→0 — spread of max|R|/p² over p ∈ {1e−6, 1e−5,
1e−4} below 1%. (b) Axis consistency: |R| at the first non-Taylor ray vs the 2πh₁² channel within
5% (catches Φ̇-integral bugs). **Amendment pre-launch (disclosed):** anchor (b) as first drafted
used the initial slice, where the pulse sits at r=2 and the near-axis field is ~2e−7 of peak — it
compared fit-noise to roundoff (14.5% "failure" at ~1e−14 magnitudes; anchor (a) passed at 4e−5
spread simultaneously). Amended substrate: synthetic axis-active slice h = c·r·e^(−r²/2)
(h₁ = c exactly, full Φ̇/I3 path exercised with real signal); require slice-R at ray 3 within 5%
of 2πc² AND the axis-fit channel within 1%. Amendment made before any gate data was collected.

## Verdicts (numbered to match C1d; fixed)
1. **Detection** (unchanged rule): surrogate p < 0.05 AND 2.2 < P < 8.5 AND |Δφ| ≤ π/3.
   Fails ⇒ nm, no downstream claims.
2. **γ two-route:** |γ_B − 0.374| ≤ 0.02 AND |γ_A(C1c banked) − γ_B| ≤ 0.03.
3. **Δ anchor:** Δ_B = 2γ_B·P_B within 3.4453 ± 0.25.
4. Not carried (C1d role was the Route-A wiggle channel, known nm; the axis channel here is a
   declared-biased diagnostic and cannot be a verdict).
5. **Convergence control, TIGHTENED per the C1d drift lesson:** N=1600, banked p*(1600),
   quarter-decade k′ = 8..30, same pipeline (no surrogates): **|γ_B(1600) − γ_B(800)| ≤ 0.01**
   AND |Δ_B(1600) − Δ_B(800)| ≤ 0.3; the SIGNED drift is reported either way.

## Non-measurement discipline
As C1d: P at scan boundary → nm; wrong-phase runs dropped and listed; if V1 fails no γ_B/Δ_B
claims. If V5 fails on the tightened drift clause, γ_B is reported as "not yet converged" — an
instrument bound, not a measurement.

## Honest scope
GD 1998 measured exactly this scaling with this class of code; occupied. Deliverable = the
receipt-grade closure (or honest non-closure) of the C1c→C1d Route-B thread: whether the
curvature branch, measured with the resolution-robust invariant, lands on the Gundlach anchor
with the echo period the C1d rule already detected.

---
## GATE RECORD (2026-07-09, appended post-execution)
Anchors passed pre-launch (p²-linearity spread 3.9e−5; amended axis-active anchor 0.12%; the
initial-slice anchor draft compared noise and was amended pre-launch, disclosed above).
45/45 evolutions, zero wrong-phase drops.

**VERDICTS: 1 PASS · 2 FAIL · 3 PASS · 5 FAIL.**
1. **Detection PASS:** p = 0.005, |Δφ| = 0.06, P = 4.281 interior, amplitude 0.479 (sharper than
   C1d's 0.322 — the unsmeared observable sees more wiggle, as predicted).
3. **Δ_B = 3.568 vs anchor 3.4453 ± 0.25: PASS (off 0.122) — the echoing period, REGISTERED-
   MEASURED on the curvature branch.** C1's original prereg declared Δ a stretch goal "expected
   nm without AMR"; it is now a two-verdict pass (V1+V3) without AMR. The period is the robust
   invariant: P = 4.28–4.30 across all three Route-B observables (C1d axis-h₁², C1e axis-|R|,
   C1e slice-|R|) and dΔ = 0.151 under N-doubling (within its registered 0.3).
2. **FAIL — and the failure completes the two-sided kill:** γ_B(slice-max) = 0.4167 (off +0.043),
   vs γ_B(axis) = 0.3140 (off −0.060; reproduces C1d to 4 decimals — cross-gate instrument
   receipt). The two max-selection biases BRACKET the anchor from opposite sides: axis clips the
   deep echoes (under-resolution), slice-max harvests 1/r-amplified noise near the ray-drop floor.
5. **FAIL on the tightened drift clause — the control did its job:** dγ(800→1600) = +0.0323 > 0.01
   (drifting AWAY from anchor with resolution: finer grids harvest deeper spikes; smoking gun in
   the control table: N=1600, ε=1e−7 → R_slice = 2.2e8, two decades above both neighbors).
   Per prereg: **γ_B is NOT YET CONVERGED — an instrument bound, not a measurement.** The Δ half
   of the control passed; the Δ claim (V3) stands with this noted.

**ADDENDUM (2026-07-10, before C1f registration):** V3's PASS is ENTANGLED with V5's failure —
Δ_B = 2γ_B·P used the non-converged γ_B = 0.4167 as a factor. The robust C1d/C1e content is the
PERIOD P = Δ/(2γ) = 4.28–4.30 (stable across observables and N), which pins only the RATIO Δ/γ =
8.56–8.60, not Δ itself (with the converged mass-branch γ_A = 0.3681 the same P gives Δ ≈ 3.15,
outside the band). V3 is DOWNGRADED from "Δ measured" to "Δ/γ measured": the wiggle route cannot
break the γ–Δ degeneracy by construction. Gate C1f (echo timing + echo counting, both
selection-free) is registered to measure Δ and γ independently and supersedes this claim.

**Finding of record:** extremal (max-over-run) Route-B observables are exponent-corrupting in
BOTH directions on this instrument class — under-resolved maxima clip (γ low), fully-resolved
maxima noise-harvest (γ high, non-convergent; the P4.3 f64-extrema lesson recurring at the PDE
layer). The Route-B γ remains honestly unmeasured here; its two biased values bracket the anchor.
**Costed next rungs (not run):** (i) selection-robust observable — R at FIXED echo index, or the
envelope growth-rate of R_axis(τ) within single runs (no cross-ε max selection at all);
(ii) registered noise floor for the slice max (quantile-max or r-floor tied to local ray
spacing) — cheap but needs its own convergence proof; (iii) N≥3200 ladder to test whether the
slice-max drift saturates. Arc status: γ measured on the mass branch (C1c, 0.3681 ± converged),
Δ measured on the curvature branch (C1e, 3.568 ∈ 3.4453 ± 0.25), Route-B γ = the named open wall.
