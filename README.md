# IG-PRIMON-T1 — Information Geometry of Arithmetic Gases

A disciplined, computationally-assisted research program on the **Fisher / Ruppeiner information geometry**
of prime (primon / Riemann-gas) and related disordered systems. The canonical state of the program lives in
the **consolidated results ledger** (`T1_consolidated_results_ledger_v0_4.md`); the frozen predictions are in
the **pre-registration** (`T1_preregistration_v0_6.md`).

## Discipline

Every claim carries an honest status tag — **[V]** verified by a reproducible receipt, **[E]** defensible
extrapolation, **[C]** conjecture, **[GATE]** derive-before-numerics — and no result is registered without a
runnable artifact. Amendments are versioned diffs (no silent edits).

## Three papers

| paper | subject | status |
|---|---|---|
| **Paper 1** — `paper1_arithmetic_dictionary_draft_v0_1.tex` | A real-axis spectral dictionary reading the arithmetic of number fields (unit rank, signature, `hR/w`, discriminant) off one Fisher expansion at the Hagedorn point — no continuation, no zero locations. | draft |
| **Paper 2** — `paper2_curvature_dichotomy_FULL_v0_1.tex` (+ PDF) | A Ruppeiner curvature dichotomy for generalized prime gases: temperature-driven ⇒ `R→0` (flat), fugacity-driven ⇒ `|R|→∞`. | compiled |
| **Paper 3** — `paper3_criticality_diagnostic_draft_v0_1.tex` | A curvature diagnostic separating continuous-RSB criticality from kinematic volume-divergence, across three venues (ridge / SK spin glass / spherical perceptron). | draft |

## Computational receipts (CPU-only Python: numpy / scipy / mpmath)

- `module_e_radius_finding.py`, `audit_independent.py` — Result A (arithmetic dictionary) + audit.
- `module_L_ridge_curvature.py` — Result C kinematic side (ridge double descent), 40-dps.
- `module_L_SK_converse.py` — Result C genuine side (SK at the dAT line).
- `module_L_perceptron_{replica,replicon,curvature,finiteT}.py` — the perceptron storage/jamming archetype
  (regime split; continuous replicon; `(α,ε)` indefinite first pass; `(β,ε)` positive-definite curvature `[V]`).

Each receipt is anchored to an exact reference value (e.g. Gardner `α_c(0)=2`, Ising `s(0)=log2`, the
engine `R=−1` pin) and reproduces every numerical claim it backs.

## Reproducibility

Python 3 with `numpy`, `scipy`, `mpmath` (40-dps where cancellation requires it). Run any receipt directly,
e.g. `python module_L_perceptron_finiteT.py`. No GPU/accelerator is used; all certification is CPU-side.
