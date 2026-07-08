# PREREG — L3 Phase 3: the deep push to n=1000 (priced by the measured wall law)
**Registered 2026-07-08 before execution. Amendment to PREREG_L3p2_push.md.**

The Phase-2 wall law n_wall(D) = (D+3)/0.295 prices n=1000 at dps≈292. Recipe: DPS_MAIN=360,
DPS_CHECK=310 (both above the wall at n=1000, with 65/15 effective digits at the deepest row),
M=2048 contour nodes (aliasing negligible: Z entire of order 1), R=3.0. Route A: 100k Odlyzko zeros +
semi-analytic tail as in Phase 2.

## Pre-registered predictions (the law is now on trial, not just in use)
1. **Wall-slope reproduction:** the (310,360) dps-pair digits-lost slope = **0.2947 ± 15%** with
   R² > 0.99. The measured Phase-2 law must predict a pair it never saw.
2. **Two-route agreement** at n ∈ {400,500,600,700,800,900,1000} within 3× the Route-A budget —
   with n=400 doubling as a cross-run anchor (Phase-2: λ₄₀₀ = 748.31558284, |diff| 1.7e−5).
3. **Anchors:** λ₁ exact; σ₂/σ₃ = Lehmer; λ₄₀ = 30.4773754237807 (cross-run).
Occupied-territory declaration unchanged (Johansson n≤100k single-route rigorous). The deliverable is
the deepest two-route matched-budget receipt + the wall law's out-of-sample confirmation.
Non-measurement discipline: rows where the dps-pair has <6 agreeing digits are "not measured".
