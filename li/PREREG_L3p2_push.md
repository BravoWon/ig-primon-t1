# PREREG amendment — L3 Phase 2: push n, map the wall, chart violation sensitivity
**Registered 2026-07-06 before execution. Versioned amendment to `PREREG_L3_li_criterion.md`.**

## Occupied-territory declaration (binding on all wording)
Single-route λ_n computations exist far beyond this phase's range: Keiper (1992) n≤7000; Maślanka
(~2004) n~3300; **Johansson (Arb, rigorous ball arithmetic) n ≤ 100,000**. No λ_n VALUE in this phase
is novel. The candidate-novel deliverables, flagged [C] pending a lit-gate: (i) the explicit two-route
cross-verification receipt (zeros vs zero-free cumulants, matched budgets) at n ~ 50–400; (ii) the
**wall law** for the cumulant-binomial route as a measured equation; (iii) the **violation-sensitivity
curve** (smallest n at which a planted off-line zero of strength β becomes visible in λ_n).

## Design
- **Route B (zero-free), re-engineered for depth:** Taylor coefficients a_k of the ENTIRE function
  Z(1+ε) = εζ(1+ε) by trapezoid-DFT on |ε| = R (exponentially convergent; no branch issues), then the
  exact power-series log recurrence k·g_k = k·a_k − Σ i·g_i·a_{k−i} → σ_j → binomial λ_n. Working
  dps 90, R = 3.0, M = 4096, n ≤ 400. Anchors carried over: λ₁ closed form; σ₂, σ₃ vs Lehmer;
  **λ_40 vs the Phase-1 run's 30.47737542 (cross-run, different contour machinery).**
- **Route A (zeros):** 100k Odlyzko ordinates (T≈74,920, 9 dp) + **semi-analytic tail** (integral of the
  pair-term against the Riemann–von Mangoldt density, NOT a bound), with an explicit error budget:
  S(T)-fluctuation term ≈ 3·2(1−cos(nθ(T))) + zero-accuracy propagation + f64 floor.
- **Wall map (its own falsifiable law):** measured digits-lost(n) via dps-pair agreement must follow
  **digits_lost(n) ≈ n·[log₁₀(3/R) + log₁₀(1 + 1/|ρ₁|)]** (contour term + binomial-cancellation term,
  |ρ₁| = 14.1436). Verdict: linear fit slope within 15% of prediction, R² > 0.99 — else the wall model
  is wrong and the measured law replaces it.
- **Sensitivity curve:** planted quadruples at β ∈ {0.55, 0.6, 0.7, 0.8, 0.9}, γ* = 14.13; record first
  n with λ_n < 0; check against growth-rate theory n* ~ −log(λ_trend)/log|1−1/(1−ρ*)|.

## Pre-registered verdicts
1. **AGREE** iff |A + tail − B| ≤ 3× Route-A budget for all tabulated n ≤ 400.
2. **WALL LAW** holds per above, giving n_wall(dps) — the frontier equation for any future deep push.
3. **SENSITIVITY** monotone in β with theory-consistent slope ⇒ the instrument's detection frontier is
   charted (what strength of RH violation this method would see, at what n).
Non-measurement discipline: any n where Route B's dps-pair agreement dies is reported "not measured".

---
## GATE RECORD (2026-07-06, appended post-execution; amends nothing silently)
- **Verdict 1 — PASS.** Two-route agreement n=50..400: worst ratio 0.30 of budget; at n=400,
  λ = 748.31558… agrees across routes to |diff| 1.7e−5 (~2e−8 relative). All λ_n > 0.
- **Verdict 2 — measured law replaces prediction (its own pre-registered branch).** Clean (90,130)
  dps-pair: **digits_lost ≈ 0.2947·n (R²=0.9999)** ⇒ n_wall(D) ≈ (D+3)/0.295. Predicted 0.0297 was
  wrong (recurrence conditioning ≈ log₁₀2/n — post-hoc hypothesis). First pair's 0.2366 (R²=0.94) was
  contaminated by full-precision-death flattening. Pricing: n=2000 two-route needs dps≈590 (Arb/C project).
- **Verdict 3 — completed after a design error taught the law.** Plant at γ*=14.13 was invisible to
  n=400 (growth 1.002) — the miss *measured* the height-dependence. At γ*=2: monotone frontier
  β=0.60→n*=231, 0.70→90, 0.80→51, 0.90→26; β=0.55 beyond window (needs ~560). Height scaling at
  β=0.9: n*(γ=2)=26, n*(γ=4)=227, γ=8 & 14.13 undetected (predicted ~416, ~1300) — superquadratic
  (~γ^2.6 on two points; shape, not exponent, is the claim).
- Reproducibility: zeros file via `curl -sL http://www.dtc.umn.edu/~odlyzko/zeta_tables/zeros1`
  (gitignored; 100k ordinates, 9 dp). Artifacts: push_wall.py, push_wall_results.json, push_wall.png.
