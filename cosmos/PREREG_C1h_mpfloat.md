# PREREG — Gate C1h: the mp-float bisection — buying τ-depth with digits
**Registered 2026-07-11 before execution. Cosmos arc, PR #13. The escalation named in C1c and
proven necessary by C1g: at the f64 floor, τ* = γ·ln(1/ε) ≈ 10 caps the ladder at 6–7 crossings
and no ray allocation adds τ-depth — only ε-digits do. C1h adds the digits.**

## Instrument
- **Arithmetic:** vectorized double-double (DD) — each number a (hi, lo) f64 pair, error-free
  transforms (TwoSum / Dekker split / TwoProd), ~31 significant digits, implemented over numpy
  (`cosmos/dd_kernel.py`). Prefix sums via Hillis–Steele doubling (log₂N vectorized DD adds —
  a sequential Kahan loop would be 5×10⁹ Python iterations; this is the enabling trick). DD exp
  via ln2 range-reduction + r/2¹⁰ scaling + degree-9 Taylor + 10 squarings. Division via Newton
  refinement. Detection-side quantities (MOTS ratio, du quantile, crossing signs, freeze/budget
  logic) read the hi parts in f64 — thresholds are coarse; only the state-update chain is DD.
- **Noise-floor closure (why 31 digits suffice):** f64's measured threshold fog ≈ 1e−13 implies
  roundoff amplification ≈ 1e3 on 1e−16; DD's 1e−32 per-step roundoff ⇒ fog ≈ 1e−28, well below
  the ε target.
- **Configuration:** C1g's sealed Section III endpoint, carried whole — v_out = 4.134212319612,
  N = 800, CFL = 0.1, and every C1g termination guard (freeze via 20k-quiet + 1.05×THRESH, 2M
  step budget with closest-approach classification, MOTS r-floor 5e−4, overflow→bh, A5c
  geometric band-gate, ε-backoff on all deep-run paths INCLUDING the sampling control — closing
  C1g's disclosed asymmetry).
- **Bisection:** seeded at p*(C1g) = 0.012793009036827 ± 1e−11 relative, endpoint labels
  VERIFIED at DD before descent (widen ×10 up to ±1e−8 if the DD threshold moved; nm beyond).
  3-worker trisection to **1e−22 relative** (registered target; ~33 levels; stretch to 1e−25 if
  wall-clock permits, reported either way).

## Calibration (must pass BEFORE launch)
(a) DD kernel vs mpmath dps=40: add/sub/mul/div/exp/prefix-sum on 10⁴ random values spanning
1e−30..1e+30 — max relative error < 1e−28 (exp: < 1e−27).
(b) Instrument continuity: evolve_dd at ε = 1e−9 (both arithmetics valid there) must reproduce
the f64 instrument's outcome and its band-gated crossing TIMES to |δu| < 1e−6 (the DD trajectory
is not bit-identical — roundoff differs — but at ε ≫ both fogs the physical ladder must match).
(c) DD threshold sanity: the seeded bracket's endpoints must label (disp, bh) at DD.

## Amendment C1h-2 (2026-07-11, pre-descent-data — bracket-inversion forensics)
First launch: the seeded ±1e−11 bracket came back INVERTED (lo=bh, hi=disp) — impossible for a
monotone boundary, so ≥1 mislabel. Forensics (termination-cause instrumentation, banked in the
diag log): the hi probe dispersed honestly (drain, closest approach mots 0.41); the lo probe
burned the full 2M-step budget stuck at u = 5.0214 with **mots_min = −inf** — ray-crossing drove
ḡ negative (grid pathology, not a horizon) and the budget classifier took −inf as
closest-approach ⇒ bh; the freeze never engaged because sub-resolution h₁ flicker (flips every
1–2 steps) kept resetting the quiet counter — the A5b flicker signature attacking the GUARD
layer. Two fixes, both termination-side: (i) mots_min tracks POSITIVE mots only (negative =
ray-crossing; the MOTS trigger likewise requires mots > 0); (ii) a sign flip resets quiet/u_lc
only if ≥ 5 steps from the previous flip (sub-resolution flicker must not veto the freeze).
**Measured en route (banked):** the DD threshold sits (1e−10, 1e−9) relative ABOVE the f64 seed
— the f64 p*'s absolute position error, directly measured; bracket verification therefore starts
at ±1e−9 (narrower windows are known-inverted). Descent restarted from scratch under the fixed
classifier (levels 1–4 of the first descent discarded unexamined).

## Amendment C1h-3 (2026-07-11, pre-verdict — the quiet-collapse robbery)
The first full descent sealed a bracket at 2e−23 whose ε = 1e−14 verdict run returned bh — nine
orders above the bracket floor, impossible unless the WHOLE bracket sat above the true threshold.
Mechanism: post-departure, a collapse-bound run's unstable mode grows MONOTONICALLY — no h₁
crossings, so 20k quiet steps accumulate while mots is still above the freeze guard; the freeze
drains the grid before the horizon resolves and the probe is mislabeled disp. Each robbery pulls
lo upward into true-bh territory; eighteen self-consistent levels then descend inside the wrong
region ("the freeze's premise — a collapse in progress is never quiet — is FALSE during
post-departure growth"). Fix: the freeze additionally requires mots to be NON-DECREASING (5k-step
trend sampler; a downtrend vetoes the freeze); the 2M budget still bounds any slow hover and
classifies by positive-only mots_min. The displaced first-descent p*
(0.012793009044178402 − 5.78e−19, bracket 2e−23) is BANKED AS DIAGNOSTIC: the fixed re-descent's
p* against it measures the robbery-zone width directly. Full re-descent from the ±1e−9 seed.

## Verdict runs and verdicts (fixed)
Deep runs at **ε = 1e−14, 1e−18, 1e−22** — the new axis is DEPTH: three ladders of increasing
τ* (≈ 12, 15.5, 19). Ladder fit = C1f's `fit_timing2`, events = A5c band-gated (unchanged).
1. **Ladder depth:** ≥ 16 band-gated events at ε = 1e−22 (f64 ceiling was 13–15; the digits must
   buy rungs or the premise fails).
2. **Δ anchor:** median Δ over the three depths within **3.4453 ± 0.25**.
3. **Consistency, two axes:** (i) depth-consistency: max pairwise |ΔΔ| across the three depths
   ≤ 0.20 (C1g's config-scatter was ±0.34 on 6-rung ladders; deeper ladders must tighten or the
   scatter law is depth-independent — reportable either way); (ii) sampling: CFL = 0.05 at the
   same v_out (own DD trisection, ε-backoff enabled), |Δ(0.05) − Δ_median| ≤ 0.20.
4. **Closure (carried from C1g, tolerances frozen):** implied γ_B = Δ_median/(2·4.29) within
   0.374 ± 0.03 AND |γ_B − γ_A(0.3681)| ≤ 0.03.

## Non-measurement discipline
Verdict-3 failure voids verdict 2 (as the arc's controls always have). Bracket-verification nm
⇒ gate nm (no DD threshold hunt beyond ±1e−8 — that would be a new registration). Step-budget
exhaustions logged with their closest-approach classification. If wall-clock exceeds 48 h the
run is checkpointed at the deepest completed level and the gate reports at that depth (bracket
level reached is itself a deliverable: the DD fog floor, measured).

## Honest scope
Gundlach's Δ = 3.4453 is spectral and exact to four digits; Choptuik/Garfinkle measured echoes
with AMR and dedicated tuning. Occupied. Deliverable = Δ from event timing at 31 digits on an
800-ray Section III grid — or the honest depth-scaling law of the ladder-fit scatter, and the
DD fog floor as the next-instrument constant. Win or lose, the arc's number-type wall gets its
first quantitative probe.
