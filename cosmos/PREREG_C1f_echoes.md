# PREREG — Gate C1f: kill the max() — echo timing + echo counting, both selection-free
**Registered 2026-07-10 before execution. Cosmos arc, PR #13. Successor to C1e per its costed
rung (fixed-echo route). Instrument: `gate_C1c_garfinkle.py` scheme; banked p* = 0.01280497040512,
p*(1600) = 0.012794010326, γ_A = 0.3681 (C1c, converged).**

## The degeneracy this gate breaks (and the C1e addendum it enforces)
The wiggle route (C1d/C1e) measures ONLY the period P = Δ/(2γ) — the ratio Δ/γ = 8.56–8.60 —
never Δ or γ separately (C1e's V3 "Δ measured" is downgraded by addendum: it used the
non-converged γ_B as a factor). Max-over-run observables corrupt γ two-sidedly (C1d/C1e finding).
C1f uses two observables with NO max() and NO cross-ε amplitude selection:
- **Route 1 (echo timing → Δ directly):** in a single deep-subcritical run, the zero crossings of
  h₁(u) accumulate geometrically toward the critical accumulation point: u_i = u* − C·e^(−iΔ/2)
  (two crossings per DSS period). Fitting (u*, C, Δ) per run measures Δ with no γ anywhere.
  Crossings are timing events: amplitude noise cannot move them.
- **Route 2 (echo counting → 2γ/Δ):** the solution follows the critical solution for
  τ* = γ·ln(1/ε) + const before departing (departure scale ε^γ; γ = 1/λ of the one unstable
  mode), so the total crossing COUNT is N(ε) = (2γ/Δ)·ln(1/ε) + const. Integers cannot be
  noise-harvested. Then **γ_B ≡ (Δ_timing/2)·slope_N** — exponent from timing + counting alone.
Cross-relation: 1/slope_N = Δ/(2γ) = P must reproduce the banked wiggle period (consistency).

## Data (fixed)
- Route 1: h₁(u) traces at ε ∈ {1e−10, 3.16e−11, 1e−11, 3.16e−12, 1e−12, 3.16e−13} (6 runs,
  N=800, banked p*).
- Route 2: crossing counts on the C1d eighth-decade grid, ε = 10^(−k/8), k = 16..60 (45 runs,
  N=800, traces collected, count = number of sign changes of h₁ over the full run).
- Control: N=1600 (banked p*(1600)): Route 2 on quarter-decade k′ = 8..30; Route 1 at
  ε ∈ {1e−11, 1e−12}.

## Registered extraction rules
Route 1 per run: crossings = sign changes of trace h₁; drop the first crossing (initial-pulse
transient); accumulating prefix = maximal subsequent run of strictly shrinking intervals; require
≥5 crossings in the prefix, else that run is nm for timing. Fit: scan u* over
(u_last, u_last + 2·(last shrinking interval)] on 2000 points; at each u*, linear fit
ln(u* − u_i) vs i → slope = −Δ/2; pick u* minimizing SSE. Primary Δ_timing = MEDIAN over the ≥4
qualifying runs (median, not mean — pre-fixed). Robustness channel (reported, non-verdict):
even/odd crossing split (each spaced Δ in τ) — two slopes per run.
Route 2: N(ε) integer counts; single least-squares line N vs ln(1/ε) over the full registered
grid; slope_N and its R². If R² < 0.95 → Route 2 nm.

## Instrument calibration (must pass BEFORE launch)
(a) Synthetic crossing sequences u_i = u* − C·e^(−iΔ/2) with Δ = 3.4453, alternating-phase
perturbation (±20% interval modulation) and 1% timing jitter, 6–10 crossings: recover Δ within
±0.15. (b) Synthetic counts N = round(a·ln(1/ε) + b) on the registered grid with a = 1/4.606:
recover slope within ±5%. (c) One pilot run at ε = 1e−11 (disclosed): trace extraction must
yield ≥5 accumulating crossings, else the trace machinery (not thresholds) is fixed first.

## PILOT-DRIVEN AMENDMENTS (2026-07-10, before gate execution, all disclosed)
The registered pilot fired three instrument findings; the following amendments were made BEFORE
any gate data was collected:
(A1) **Counting doubles — slope_N = 4γ/Δ, not 2γ/Δ.** The subcritical field echoes back OUT
     during dispersal: every inbound crossing has an outbound partner. Pilot receipt: deep local
     count slope (13−9)/ln(10⁴) = 0.4343 vs 4·0.374/3.4453 = 0.4343 — to four decimals.
     Hence γ_B = (Δ_timing/4)·slope_N and the consistency relation is 2/slope_N = P.
(A2) **Timing needs 4× u-sampling.** At banked CFL=0.4 the trace SKIPS crossings near the
     accumulation (pilot interval ratios ≈ e^(−Δ) instead of e^(−Δ/2)); two sign flips inside one
     step annihilate. Route 1 instrument amended to CFL = 0.1 (N=800), with p* RE-BISECTED to the
     f64 floor under exactly that configuration (deep-ε runs are p*-locked to the discrete
     dynamics: a CFL=0.1 probe against the CFL=0.4 p* returned bh at ε=1e−11 — inside the shifted
     p*'s error, meaningless). Route 2 stays on the banked instrument (counting is skip-robust in
     aggregate; residual skips appear as fit scatter, gated by R²).
(A3) **Prefix rule alternation-tolerant:** accumulating prefix = maximal run with d_i < d_{i−2}
     (DSS phases alternate within a period; strict d_i < d_{i−1} truncates at the first upturn).
(A4) **Verdict-4 control re-axed:** the timing failure mode is SAMPLING, not grid — timing
     control = CFL 0.05 with its own re-bisection (2 runs); the grid control (N=1600, banked
     p*(1600), CFL=0.4) applies to slope_N only.
(A5) **Prefix terminates at the turnaround** = the global minimum crossing interval (the
     closest-approach signature); the alternation-tolerant rule alone runs through the turnaround
     into out-phase crossings (pilot receipt: prefix=8 swallowed 3 outbound crossings → Δ=2.76).
(A6) **The final pre-turnaround interval is excluded** — departure squeeze: the run leaves the
     critical solution mid-period (pilot receipt: last inbound ratio 0.010 ≈ e^(−4.6), physically
     impossible for a Δ=3.44 ladder).
(A7) **Two event families, joint fit.** At the f64 floor τ* ≈ 10–12 yields only ~5 inbound
     crossings (2 transient + 1 squeezed → 2–3 clean intervals): information starvation is the
     measured wall. Amendment: between consecutive zero crossings lies exactly one echo extremum
     of h₁; its TIME is a within-run locator (no cross-ε amplitude comparison — the max() ban is
     untouched). Event set = crossings ∪ inter-crossing extremum times, two interleaved geometric
     ladders sharing Δ: joint fit (u*, Δ, C_family1, C_family2) — u* scanned, shared slope closed
     form, per-family intercepts. Both families individually obey A5/A6 (drop first event,
     exclude final pre-turnaround event). Calibration (a) extended to 2-family synthetics.

(A8) **Calibration criteria corrected (synthetic-side only; instrument untouched):** the counting
     synthetic double-counted noise (iid uniform ON TOP of integer rounding — the real N(ε) is a
     deterministic staircase; correct model = pure quantization + occasional ±1 skip errors, 15%
     of points; criterion = 20-draw mean recovery within 5%). The timing recovery criterion is
     aligned to the VERDICT band (±0.25; ±0.15 was tighter than ±10% phase alternation permits on
     qualifying ladders) and the qualifying rule is fixed at ≥4+3 events per run (3+2 has one
     degree of freedom). Real-pilot disclosure: Δ = 3.580 at ε = 1e−10/1e−11/1e−12 (identical —
     deep runs share the trajectory before departure, so Route-1 runs are NOT independent
     samples; the IQR criterion of verdict 1 is therefore trivially small and is reported as
     such, not as evidence of precision).

## Verdicts (fixed, as amended pre-execution)
1. **Δ direct:** Δ_timing within **3.4453 ± 0.25**, from ≥4 qualifying runs; run-to-run spread
   (IQR) reported; IQR > 1.0 → nm regardless of median.
2. **γ_B selection-free:** γ_B = (Δ_timing/4)·slope_N within **0.374 ± 0.02** AND
   |γ_B − γ_A(0.3681)| ≤ 0.03.
3. **Cross-route consistency:** |2/slope_N − 4.28| ≤ 0.5 (the integer route must reproduce the
   banked amplitude-route period P; C1d/C1e values 4.28–4.30 were seen and are banked — this is
   a consistency check against prior receipts, disclosed as such, not fresh evidence).
4. **Convergence control:** |Δ_timing(CFL=0.05) − Δ_timing(CFL=0.1 median)| ≤ 0.15 AND
   |slope_N(N=1600) − slope_N(800)| ≤ 0.05·slope_N(800). Signed drifts reported.

## Non-measurement discipline
Any run with < 5 prefix crossings: listed, excluded from Route 1 (expected at the shallow end).
If < 4 runs qualify → verdict 1 nm. bh-outcome runs at deep subcritical ε (bisection-floor
contamination): dropped and listed. If verdict 1 is nm, verdict 2 is nm (γ_B needs Δ_timing).
No amplitude of h₁ enters any verdict quantity anywhere.

## Honest scope
Choptuik 1993 measured Δ by echo inspection; Gundlach 1997 computed it spectrally (3.4453);
Hod–Piran and GD98 did the fine structure. Occupied. Deliverable = the selection-free
timing+counting receipt genre (Δ and γ from event statistics with no extremal operator), the
degeneracy-breaking closure of the Route-B thread, and — if verdicts pass — the honest version
of the claim C1e had to retract: Δ measured without AMR.

---
## GATE RECORD (2026-07-10, appended post-execution)
Calibration passed (A8 criteria); both re-bisections hit the f64 floor (p* CFL-shifts ~1.4e−6
relative — 10⁵× the probed ε, vindicating A2's re-bisection mandate).

**VERDICTS: 1 PASS-BUT-VOIDED · 2 FAIL · 3 FAIL (by 0.003) · 4 FAIL (both axes).**
1. Δ_timing = 3.580 (all 6 runs identical — one trajectory, as pre-disclosed), off 0.135, inside
   the band — **but v4's sampling control voids the interpretation:** Δ(CFL=0.05) = 2.716, drift
   −0.864. NOT sampling-converged ⇒ **Δ remains unmeasured.** Diagnosis: at finer sampling the
   ladder extends (8+7 events) and the A5 turnaround detector (global interval argmin) swallows
   outbound events — 2.716 ≈ the pilot's known-contaminated 2.76. The rule was frozen mid-gate;
   the verdict stands.
2. FAIL — γ_B = 0.4732 from a corrupted slope × a non-converged Δ. 3. FAIL — 2/slope_N = 3.783
   vs banked P 4.28, |d| = 0.497 against tolerance 0.500.
4. FAIL, both axes, correctly: dD = −0.864; dslope = −38%.

**Finding of record — the counting cliff:** N(ε) is not a staircase: N = 3 (one decade), one
step to 4, then FLAT for three decades (the physics demands a step per decade), then an eruption
to 9–13 below ε ≈ 2.4e−7 with ±2 scatter. Mechanism: du = CFL·(5th-pct ray spacing) shrinks as
deeper runs focus the grid, so resolution onset — not echo count — sets N. The N=1600 control
moved the cliff (slope −38%). **Event statistics did not escape the wall: the wall moved from
the observable (C1d clip / C1e harvest) into the SAMPLER.** R² = 0.54 < 0.95 additionally put
Route 2 nm by the registered gate.

**Arc-level instrument theorem (measured in triplicate):** at fixed-VMAX N=800, the echo ladder
is unresolvable regardless of observable class — amplitude (C1d), invariant (C1e), or event
statistics (C1f). The arc has now re-derived the third piece of the 1995 toolchain from its
failure modes: C1 → AMR; C1b → h-variables; **C1f → Garfinkle's Section III outermost-gridpoint
tuning** ("the light ray that just barely hits the singularity", iterated) — the ray-allocation
trick that concentrates the whole grid on the echo region. **Costed C1g (not run):** implement
Section III per-run tuning for the few deep timing runs (pilot run → locate the marginal ray →
re-truncate → iterate 2–3×; p* re-bisected per configuration; A5 turnaround rule re-registered
inbound-only) — a half-day compute project; the one named instrument between this program and Δ.
Route-B γ and Δ both remain honestly unmeasured. p*(CFL=0.1) = 0.01280498781335,
p*(CFL=0.05) = 0.01280500095554 (brackets 2.11e−14).
