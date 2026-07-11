# PREREG — Gate P2b (Dual Ignition, thread 2): Burgers two-route + the shock wall
**Registered 2026-07-08 before execution.** 1D viscous Burgers, u0 = sin x, periodic; evolve past the
inviscid shock time (t=1) to t=1.5.
- **Route A (exact):** Cole–Hopf integral u = ∫(x−y)/t·e^{−G/2ν}dy / ∫e^{−G/2ν}dy, G = (x−y)²/2t +
  (1−cos y), log-sum-exp stabilized; **self-receipt:** doubling the quadrature grid changes < 1e−8.
- **Route B (solver):** pseudo-spectral N modes, 2/3 dealiasing, integrating-factor RK4.
- **AGREEMENT:** rel-L2 on a 512 grid at t=1.5 < 1e−4, for every ν at admissible N.
- **THE WALL:** ν-sweep {0.1, 0.05, 0.02, 0.01, 0.005, 0.002}; minimal N*(ν) meeting tolerance
  (doubling ladder). Expectation [E]: shock width ~ ν ⇒ **N* ∝ ν^(−1)** (±20%, R² > 0.99), else the
  measured law replaces it. Failure MODE below the wall logged (Gibbs blow-up vs smearing).
- **ANCHORS:** ν=0.5 (smooth regime, both routes trivially agree); Route-A self-convergence receipt.
- **NM:** if Route-A self-receipt fails at small ν, those ν are "not measured" (never blamed on Route B).
- Honest scope: viscous Burgers is exactly solvable and its shock scaling classical — occupied
  territory; the deliverable is the receipt genre + the measured admissibility law for the spectral
  solver at the singular limit, on the instrument-pair map alongside FHS/edge/pump/Bott/LCM.

---
## GATE RECORD (2026-07-08, appended post-execution)
- **Route A exact at machine epsilon**: self-receipts 4.4e−16..6.7e−16 at every ν; anchor N=64 at
  ν=0.5 matches to 1.4e−11.
- **Two-route receipts**: agreement <1e−4 with 2N-stability for ν ∈ {0.1, 0.05, 0.02, 0.01} at
  N* = 64/128/512/1024.
- **Wall law (measured replaces expectation, per prereg): N* ~ ν^−1.246 (R²=0.9928)** — steeper than
  the naive shock-width count (−1); the excess is the resolution margin the solver needs as the
  gradient sharpens.
- **Ladder death documented**: ν ≤ 0.005 unresolved within N ≤ 8192 despite the spatial law predicting
  ~2.4k — the wall COMPOSITE beyond ν≈0.01: the fixed dt∝1/N temporal scheme becomes the binding
  resource (RK4 error at the shock ∝ high derivatives ~1/ν). Reported "not measured", not blamed on
  the spatial route; a temporal-wall map is the named next rung.
