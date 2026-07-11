# PREREG — Gate P4.1: the parity-split fractal wall, out-of-sample
**Registered 2026-07-08 before execution.** P4's exploratory observation (odd-q min-gaps geometric
~e^{−1.07q}; even-q wide) is hereby frozen into predictions and tested on denominators never measured.

## Frozen from P4 (in-sample, q ≤ 12)
Odd law: min-gap(q) = A·e^{−c·q} fit on q ∈ {5,7,9,11} → c₀ = 1.07. Even: {1.65, 0.77, 0.137, 0.066,
0.065} at q = 4..12.

## Design
- **Pass 1 (the wall, eigenvalues only):** all reduced p/q, q = 3..21, N=96 magnetic-BZ grid;
  min indirect gap per q (same definition as P4), measured down to 1e−9 (f64 eigh floor ~1e−12).
  **Anchor:** q ≤ 12 values must reproduce P4's table within 5% (cross-run, finer grid).
- **Pass 2 (receipts extension):** full two-route FHS-vs-Diophantine on every gap > 1e−4 for
  q = 13..16 (receipt admissibility declared: FHS not attempted below 1e−4 — nm, listed).

## Pre-registered verdicts
1. **Out-of-sample odd law:** extrapolating the frozen q≤11 fit to q ∈ {13,15,17,19,21}: per-point
   |log(gap_pred/gap_meas)| < 0.5 (factor ≤1.65), AND combined odd fit q=5..21 gives c ∈ [0.92, 1.22]
   with R² > 0.99. Else: the measured law replaces the frozen one, per standing rule.
2. **Parity persistence:** even-q min-gap > 10× the odd-law prediction at the same q, for all new even q.
3. **Location hypothesis:** the odd-q minimal gap sits adjacent to the spectrum center, r ∈ {(q∓1)/2},
   at every odd q (in-sample AND new).
4. **Receipts:** 100% FHS = Diophantine on all measured gaps q = 13..16.
## Honest scope
Butterfly bandwidth/gap asymptotics are studied territory (Thouless-lineage; WKB gap laws) — [E], no
law-novelty claimed; the deliverable is the out-of-sample receipt for OUR measured law + ~500 new
two-route gap receipts.

---
## GATE RECORD (2026-07-08, appended post-execution)
- **Verdict 1 FAIL → REPLACED (the standing rule's best case):** log-errors grow linearly (0.18→0.80)
  — the frozen c=1.075 was small-q-biased. Combined odd fit q=5..21: **c = 1.128, R² = 0.9997**; the
  measured tail is an exact decade per Δq=2 (ratios 9.90/10.00/10.07/10.12), asymptotic slope drifting
  toward ln(10)/2 ≈ 1.151 [post-hoc curiosity, flagged, not claimed].
- **Verdict 2 PASS:** even-q min-gaps sit 1.2e3–4.7e4× above the odd law. Parity split confirmed
  out-of-sample.
- **Verdict 3 FAIL → REPLACED by a sharper regularity:** the global odd-q minimal gap NEVER sits at the
  center; it sits at the **outermost gap (r ∈ {1, q−1}) of flux 2/q or its mirror (q−2)/q — uniformly,
  every odd q = 5..21** (r=1 vs q−1 alternation = E→−E symmetric partners, tie-broken numerically).
- **Verdict 4 PASS: 432/432 receipts** (q=13..16, gaps>1e−4, 22 nm listed). Program total: **744/744**
  FHS = Diophantine.
- **Anchor finding:** odd-q cross-run 0.0–1.8% ✓; even-q deviated up to 81% — **P4's coarse-grid wall
  table under-resolved even-q band extrema; P4.1's N=96 values supersede** (P4's receipt verdicts
  unaffected: receipt gaps ≫ threshold). The anchor caught the parent gate's flaw — as designed.
