# PREREG — Gate P4.3: the loaded gun — flux 3/q must decay at exactly 4C/(3π)
**Registered 2026-07-09 before execution.**

## The inherited prediction (Front B / Gu–Xu §5, via P4.2's mechanism)
At flux P/q the lowest Landau cluster splits per the inverted-flux secular polynomial F_{1/P}. For
P=3, F_{1/3}(x) = x³−6x has OPEN gaps at first order ⇒ the outermost gap (r=1) opens at the
ONE-instanton scale e^{−A/φ}, A = 8C, φ = 6π/q ⇒ **c₃ = 4C/(3π) = 0.3887479** — exactly one third of
the p=2 exponent (which needed two instantons because F_{1/2} is Dirac-gapless). This gate is the
out-of-sample test of the MECHANISM, not just the constant: a miss bites P4.2's parent claim.

## Design
- Measure the **r=1 outermost gap** (E₂min − E₁max, same indirect definition as P4.1/P4.2) of flux
  3/q for odd q coprime to 3: q ∈ {13,17,19,23,25,29,31,35,37,41,43,47,49,53,55,59,61,65,67,71,73}.
  Slow decay ⇒ f64 suffices throughout (g(73) ~ 1e−10 > f64 floor); mp50 spot-check at q=49 (<0.1%).
- Local slopes between consecutive q; window-rotated 1/q extrapolation + Aitken (P4.2 machinery).

## Pre-registered verdicts
1. **The constant:** extrapolated c₃∞ ∈ 4C/(3π)·(1 ± 0.02) = [0.3810, 0.3965].
2. **The ratio (the mechanism's cleanest form):** c₂∞/c₃∞ = 3 ± 0.06, using P4.2's confirmed
   c₂∞ = 4C/π.
3. **Fixed-exponent fit:** ln g + (4C/3π)q = a + α·ln q with rms < 5% over the full range.
4. **Cross-validation:** f64 vs mp50 at q=49 within 0.1%.
FAIL branches: c₃∞ outside band ⇒ the one/two-instanton assembly is wrong or incomplete for this gap
family — reported as the finding, P4.2's mechanism section flagged. All slopes/fits reported regardless.

---
## GATE RECORD (2026-07-09, appended post-execution)
- **Raw run: verdicts 1–2 FAILED — via an instrument hook, documented.** The slope sequence descends
  monotonically for 17 points (0.4045→0.3956, mids 21–66) then hooks UP at the last two — and the
  window extrapolator swallowed the hook (c₃∞=0.4055 > its own tail: impossible for a
  monotone-from-above sequence). Diagnosis: f64 extrema-refinement harvests noise-maxima at gaps
  ≲1e−12 (bias 0.2–2.3%).
- **Tail verification (N=192 + mp40, q=67/71/73):** corrected gaps 9.901e−12 / 2.038e−12 / 9.266e−13
  — the hook vanishes; descent restored (corrected tail slopes 0.3961, 0.3959, 0.3949, 0.3951, 0.3942).
- **CORRECTED VERDICTS — ALL PASS:**
  (1) c₃∞ = 0.38843 ± 0.00180 vs 4C/(3π) = 0.38875 — **|d| = 0.00032, dead center**;
  (2) exponent ratio c₂/c₃ = **3.0025** (target 3 ± 0.06);
  (3) fixed-exponent fit over 11 decades: α = −0.365, rms 1.84%;
  (4) f64-vs-mp40 at q=49: 0.0000%.
- **THE MECHANISM HOLDS OUT-OF-SAMPLE.** One instanton at p=3 (F_{1/3} open at first order), two at
  p=2 (F_{1/2} Dirac-gapless) — the exact factor of three, measured on a flux family the theory was
  never fitted to. P4.2's parent claim survives its child's bite; P4.1's parity wall now has its
  mechanism confirmed from two independent flux families.
- Instrument note for the admissibility chart: f64 extrema-refinement is biased at gaps ≲1e−11
  (noise-max harvesting) — mp verification required below that line for slope work.
