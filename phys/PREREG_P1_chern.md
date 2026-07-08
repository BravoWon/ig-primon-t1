# PREREG — Gate P1: two-route Chern number on the Haldane model (kernel instantiation #8)
**Registered 2026-07-08 before execution.**

## Substrate
Haldane honeycomb model (t1 NN hopping, t2·e^{iφ} NNN, sublattice mass M) — the archetypal Chern
insulator. Exact phase diagram known: C = ±1 for |M| < 3√3·t2·|sin φ|, else 0 (the anchor AND the wall).

## Two routes (must agree with NO tuning between them)
- **Route A (bulk):** Chern number by Fukui–Hatsugai–Suzuki lattice-gauge Berry-curvature integration
  over the Brillouin zone (plaquette link variables; integer-quantized by construction on any
  admissible grid).
- **Route B (edge):** zigzag ribbon spectrum H(k_x); count net chiral edge modes crossing the bulk
  mid-gap on one edge (sign = velocity × edge polarization). Bulk–boundary correspondence is the claim
  under test: C_bulk = C_edge, computed by unrelated machinery (2×2 Bloch algebra vs 2W×2W ribbon
  diagonalization).

## Kernel slots
| slot | instantiation |
|---|---|
| CLAIM | C_bulk = C_edge at every gapped sweep point (φ ∈ {±π/6..±5π/6}, M/t2 ∈ 0..6) |
| ANCHORS | C(M=0, φ=π/2) = ±1 (Haldane point; sign fixed by convention, recorded once); C→−C under φ→−φ; C=0 deep in trivial phase; measured phase boundary = 3√3·t2·sin φ |
| CHAOS REF / fixed point | here inverted: the **quantized integer is the invariant message**; local Berry curvature is the continuously-varying "surface" — the one gate where the fixed point is NOT anonymous, it's the physics |
| ROTATIONS | FHS grid N ∈ {12,24,48}; ribbon width W ∈ {20,40}; mid-gap reference from bulk bands |
| WALL | approach the transition (gap→0): minimal admissible FHS grid N*(gap) — fit power law N* ∝ gap^{−p}; expectation [E] p ≈ 1 (Dirac-mass Berry-curvature concentration), tolerance ±30%, else measured law replaces it |
| NON-MEASUREMENT | points with bulk gap < 0.05·t1: edge counting ambiguous → "not measured", excluded from verdict 1 (listed, not hidden) |
| VERDICTS | (1) two-route agreement 100% on measured points; (2) all anchors reproduce; (3) wall power law with R² > 0.95, exponent reported. Any single disagreement on a well-gapped point = bulk–boundary violation in OUR instrument = instrument bug until proven otherwise (Law #1) |

## Honest scope
The Haldane phase diagram is textbook (occupied territory — Haldane 1988, TKNN, Hatsugai); the gate
validates the *method genre* (two-route topological receipts + admissibility wall law) on exactly-known
ground before it is pointed at any lattice where the answer is NOT known. That pointing is P1-Phase-2,
gated on this passing.

---
## GATE RECORD (2026-07-08, appended post-execution)
- **Verdict 2 PASS**: all anchors (C=−1 at Haldane point, sign convention recorded; flips with φ;
  trivial 0; edge=bulk) and the exact boundary 3√3·t2·sinφ reproduced across the 49-point sweep.
- **Verdict 1 as pre-registered: FAIL — located, and it is not physics.** The two flagged points
  (φ=±π/2, M/t2=5.0, gap≈0.078) have bulk(12,24,48)=edge(40,60) in full agreement; **only W=20 fails
  (returns 0)** — the edge modes hybridize when ribbon width < localization length ξ≈v/gap≈19 cells.
  The two-route claim proper holds **49/49**; the rotation clause tripped because the gap>0.05
  non-measurement threshold admitted two points inside the W=20 admissibility wall. Threshold was set
  wrong, not the correspondence.
- **Verdict 3 PASS with the measured law replacing the expectation**: FHS is essentially WALL-FREE —
  N*=6 constant down to gap 0.006 (p≈0.00, R²=1.00; the known coarse-grid integer exactness of the
  lattice-gauge method — my [E] p≈−1 was Berry-concentration reasoning that does not bind FHS).
  **The true wall lives in Route B: W* ~ ξ ~ 1/gap** (measured at the flagged points: W=20 fails,
  W=40 works at gap 0.078). Two routes, two walls: one flat, one 1/gap — the admissibility map of
  the instrument pair.
- Debug ledger: two instrument bugs caught by anchors before the run (sorted-eigh bands cannot cross →
  physical-branch tracing; the Haldane-point crossing sits exactly on the periodic BZ seam → close the
  curve over the wrap). Both would have read as "bulk-boundary violation" if trusted naively.
