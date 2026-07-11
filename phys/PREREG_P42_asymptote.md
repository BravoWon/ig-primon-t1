# PREREG — Gate P4.2: the odd-q gap asymptote — is c∞ = ln(10)/2? (candidate discrimination)
**Registered 2026-07-08 before execution.**

## The question
P4.1 measured min-gap(odd q) at the outermost gap of flux 2/q with an apparently exact decade per
Δq=2. But the LOCAL slopes c(q) = [ln g(q)−ln g(q+2)]/2 already drift: 1.146, 1.1515, 1.1546, 1.157
(q=13..21) — rising THROUGH ln(10)/2. The "decade law" may be a passing tangent.

## Candidate set (frozen now)
c∞ ∈ { ln(10)/2 = 1.151293 · 2/√3 = 1.154701 · π/e = 1.155727 · 7/6 = 1.166667 ·
"1.128 (P4.1 combined, no closed form)" · NONE-OF-THESE }.

## Two fronts
- **Front A (measurement):** g(q) for odd q = 13..33 at flux 2/q only (P4.1's location law), same
  indirect-gap definition. f64 coarse (N=96) + local extrema refinement; **mpmath (dps 50) eigenvalues
  for q ≥ 25** (g < 1e−11; f64 floor ~1e−13). Anchors: (i) q=13..21 must reproduce P4.1 within 10%
  (refinement can only shrink); (ii) f64-vs-mpmath cross-validation at q=25 (<0.1%).
  Extrapolation: fit c(q) = c∞ + b/q (+d/q² check) on the deepest 6 slopes; σ from fit + fit-window
  rotation. **Decision rule: a candidate survives iff |c∞ − cand| ≤ 2σ.** One survivor → identified
  [E, gated on Front B]. ln(10)/2 excluded → the P4.1 curiosity is dead (reported with relish).
- **Front B (literature):** the lowest-Landau-cluster splitting of Harper–Hofstadter at flux p/q is
  classical semiclassics territory (Wilkinson; Helffer–Sjöstrand). Adversarial lit-gate dispatched in
  parallel: does a closed form for this exponent exist in print? Any Front-A "identification" is
  demoted to confirmation if the constant is published (expected).

## Honest scope
Whatever survives, the claim ceiling is: a measured asymptotic constant for a specific gap family,
cross-validated f64/mpmath, matched (or not) to a published semiclassical form. No novelty wording
until Front B returns.

---
## GATE RECORD (2026-07-08, both fronts closed)
- **Front A instrument arc:** v1's mpmath route cast eigenvalues to float on return — q≥31 contaminated
  (the q=33 "gap" was 10× DBL_EPSILON, caught on sight; independently flagged by Front B). Fixed (mpf
  end-to-end), extended to q=37: thirteen clean gaps, 2.5e−5 → 2.1e−17, P4.1 anchors all ok, slopes
  monotone 1.14628 → 1.16352.
- **Front B (lit-gate): PUBLISHED-IN-ESSENCE — the constant is 4C/π = 1.1662436, C = Catalan.**
  Duan–Gu–Hatsuda–Sulejmanpasic (JHEP 01 (2019) 079, eq. 3.22/A.9): one-instanton action A = 8C —
  verbatim our arccosh integral. Gu–Xu (arXiv:2406.18098 §5): at flux P/Q with P=2 the secular
  polynomial F_{1/2} = x²−4 is Dirac-gapless at one instanton ⇒ the outermost doublet gap opens at the
  TWO-instanton scale e^{−2A/φ} ⇒ c = 16C/(4π) = 4C/π. **This also explains P4.1's parity wall
  mechanistically.** Higher-instanton P≥2 splitting coefficients are explicitly open in the 2024
  literature — the numerical law for THIS gap family appears measured here first; the exponent is not
  a new constant.
- **Verdicts:** ln(10)/2 EXCLUDED (~7σ; the P4.1 curiosity was a tangent). 7/6 excluded as closed form
  by the exact action (Δ = 4.2e−4 from 4C/π — indistinguishable in our data; the published action
  arbitrates). **Identification → demoted to CONFIRMATION per prereg:** Aitken c∞ = 1.16766 descending
  toward 4C/π; fixed-exponent fit over 12 decades: ln g = 4.158 + 0.170·ln q − (4C/π)q, rms 0.9%.
- **Out-of-sample prediction inherited from Front B (P4.3 if pursued):** flux 3/q outermost gap decays
  with c₃ = 4C/(3π) ≈ 0.38875 (F_{1/3} has open first-order gaps) — 3× slower than p=2.
- Note for the ledger: **C = β(2)** — the wall exponent of the butterfly is a Dirichlet L-function
  special value. The program's arithmetic thread reaches one level deeper than P4 knew.
