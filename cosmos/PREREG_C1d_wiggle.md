# PREREG — Gate C1d: the wiggle-native Route B — DSS fine structure as the PRIMARY model
**Registered 2026-07-09 before execution. Cosmos arc, PR #13. Successor to C1c per its costed rung:
"Route-B γ needs a wiggle-native registered rule (pre-register the 4-param fit as primary on a fresh
gate)." Instrument unchanged: `gate_C1c_garfinkle.py` (Garfinkle h-variables, N=800, banked
p* = 0.01280497040512; p*(N=1600) = 0.012794010326, banked bracket 4.3e−11).**

## Disclosure of prior peeks (chain of custody)
C1c's post-hoc quarter-decade wiggle fits were SEEN before this registration: Δ_A = 3.85,
Δ_B = 2.65, wiggle-γ_B ≈ 0.33 (full-range, 1-harmonic). Those numbers are therefore NOT evidence
for this gate. The out-of-sample content of C1d: (i) NEW eighth-decade fill-in points (odd-k, never
run); (ii) a 2-harmonic model (the sawtooth question, never fit); (iii) a pre-registered fit RANGE
excluding the corrections-dominated first decade (never applied — the peeked γ_B ≈ 0.33 included
it); (iv) a surrogate-null detection gate + half-split phase coherence (never computed);
(v) an N=1600 convergence control on (γ_B, Δ_B) (never run).

## Claim under test
Gundlach / Hod–Piran DSS fine structure: critical scaling laws carry a periodic modulation,
  ln y = c + s·ln ε + Σ_{m=1,2} [a_m sin(2πm·lnε/P) + b_m cos(2πm·lnε/P)],
with the SAME period P = Δ/(2γ) in ln ε on both branches. Route B (subcritical max h₁², s = −2γ_B)
is PRIMARY — the branch where the stable-slope rule failed structurally in C1c. Route A
(supercritical M, s = +γ_A) is the second channel, conditional (its modulation amplitude ~0.02 may
sit at grid noise; see V4). Anchors: γ = 0.374, Δ = 3.4453 (Gundlach, perturbation theory —
independent method).

## Data (fixed before execution)
Eighth-decade grid ε = 10^(−k/8). Banked C1c dense points (even k) reused verbatim; NEW odd-k
points collected now, same instrument, same banked p*. Fit ranges (fixed):
- **Route B: ε ∈ [3.16e−8, 1e−2]** (k = 16..60; ~45 pts; excludes the corrections-dominated first
  decade at the top and the banked curvature ceiling ε ≲ 1e−8 at the bottom; ~2.75 anchor periods,
  ~16 samples/period).
- **Route A: ε ∈ [1e−5, 1e−1]** (k = 8..40; ~33 pts = C1c's verdict-1 window; its local slopes are
  clean from ε = 1e−1, so no top-decade exclusion).
Period scan P ∈ [2.0, 9.0] ln-ε, 400 points, linear-least-squares at each P (6 params); same scan
for real data and surrogates. RNG seed 20260709. No other ranges, scans, or models will be tried.

## Instrument calibration (must pass BEFORE launch — else fix instrument, not thresholds)
Synthetic smoke on the same ε grids: (a) injected power law + sawtooth (γ = 0.374, Δ = 3.4453,
amplitude 0.26, Gaussian ln-noise 0.05) → pipeline must recover γ within ±0.01, Δ within ±0.15,
detection PASS; (b) null synthetic (no wiggle, same noise) → detection must FAIL (p ≥ 0.05).

## Verdicts (fixed)
1. **Detection (Route B):** permutation-surrogate gate — periodic SSE reduction (SSE_wig/SSE_lin)
   below the 5th percentile of 200 residual-permutation surrogates (p < 0.05), AND best P interior
   (2.2 < P < 8.5), AND half-split phase coherence: fundamental phase difference between the two
   ln-ε halves at fixed global P satisfies |Δφ| ≤ π/3. Fails ⇒ wiggle nm, NO γ_B/Δ_B claims,
   gate FAILS honestly.
2. **γ two-route:** |γ_B − 0.374| ≤ 0.02 AND |γ_A − γ_B| ≤ 0.03 (γ_A from the same pipeline).
   This is the bet: the peeked 0.33 came from the corrections-dominated full range; if the
   registered range still gives γ_B outside the band, the proxy (max h₁² vs Ricci) or the range is
   the finding — recorded as FAIL either way.
3. **Δ anchor (primary channel):** Δ_B = 2γ_B·P_B within **3.4453 ± 0.25**.
4. **Two-channel Δ (conditional):** if Route A passes its own detection gate (same rule as V1),
   |Δ_A − Δ_B| ≤ 0.4. If Route A detection fails, this verdict is declared nm (Route A advisory) —
   NOT a gate failure.
5. **Convergence control:** N=1600, banked p*(1600), quarter-decade k' = 8..30 over the same
   Route-B range, same pipeline (no surrogates — control only): |γ_B(1600) − γ_B(800)| ≤ 0.02 AND
   |Δ_B(1600) − Δ_B(800)| ≤ 0.3.

## Non-measurement discipline
P at scan boundary → nm (P3's ν=0.80 lesson). Any evolve() returning the wrong phase (bh in
subcritical scan / disp in supercritical) → point dropped and listed. Ceiling/corrections points
outside registered ranges are reported as diagnostics only. If V1 fails, Δ and γ_B are NOT
reported as measurements anywhere downstream.

## Honest scope
Gundlach 1997 / Hod–Piran 1997 predicted and measured the fine structure with AMR-grade codes;
fully occupied territory. Deliverable = the receipt genre on the fine structure from a 800-ray
focusing code + the first wiggle-native registered RULE in this program (the C1c lesson turned
into instrument), win or lose.

---
## GATE RECORD (2026-07-09, appended post-execution)
Calibration passed pre-launch (injected sawtooth: γ to 4 decimals, Δ within 0.05, detection fires;
both nulls silent). 45 Route-B + 33 Route-A points, zero wrong-phase drops.

**VERDICTS: 1 PASS · 2 FAIL · 3 FAIL · 4 nm · 5 PASS.**
1. **Detection PASS, decisively:** surrogate p = 0.005 (floor of 200), half-split phase coherence
   |Δφ| = 0.04 rad, P = 4.298 interior, SSE ×0.286, fitted ln-amplitude 0.322. The DSS fine
   structure on the curvature branch is REAL, periodic, and phase-coherent — C1c's diagnosis
   confirmed under a registered rule.
2. **FAIL — the registered bet lost, and the loss is the measurement:** γ_B = 0.3140 (off anchor
   0.060; |γ_A−γ_B| = 0.051) even with the corrections decade excluded and the wiggle modeled.
3. **FAIL:** Δ_B = 2γ_B·P = 2.699 (off 0.746). Note the decomposition: the PERIOD P = 4.30 is only
   ~7% from the anchor-predicted Δ/(2γ) = 4.61; Δ_B fails chiefly by inheriting γ_B's low bias
   (advisory, non-registered: 2·0.374·P = 3.215, inside ±0.25; 2·γ_A·P = 3.14).
4. **nm as pre-declared:** Route A amplitude 0.019 ≈ grid noise; p = 0.015 < 0.05 but phases
   decohere (|Δφ| = 1.26 > π/3) → detection fails → advisory only (Δ_A = 3.59, γ_A = 0.3653).
5. **PASS:** N=1600 re-run: γ_B(1600) = 0.3287 (dγ = 0.0147), Δ_B(1600) = 2.699 (dΔ = 0.000 —
   the period-mass product converged to 3 decimals while its factors each moved).

**Finding of record:** max-over-run h₁² scales with a grid-CONVERGED effective exponent
−2(0.314±0.015), NOT −2γ = −0.748, over ε ∈ [3.2e−8, 1e−2]. The C1c/C1d proxy ("axis curvature
∝ h₁²") is not Garfinkle–Duncan's max Ricci at the accumulation point: max-over-run selects
whichever discrete echo wins at each ε — an ε-dependent selection the registered R_max observable
does not have — and carries a −0.12 exponent bias. The wiggle-native RULE itself is vindicated
(V1+V5: detected, phase-coherent, converged); the OBSERVABLE is the remaining instrument error.
**Costed next rung (C1e if ignited):** evolve the true axis Ricci scalar from the h-variables
(Garfinkle's paper gives the curvature forms) or measure the envelope at fixed echo index;
re-register verdicts 2–3 unchanged against the new observable.
