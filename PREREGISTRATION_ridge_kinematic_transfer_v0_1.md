# PRE-REGISTRATION — Kinematic Ridge Transfer Test (v0.1, frozen 2026-06-15)

**Protocol of record.** This document is the frozen prediction. Phases execute in order; §6 is
locked before any net is touched (the firewall-teeth rule: the number is predicted before it is
measured). Receipt: `module_L_ridge_transfer.py`. No silent edits; amendments are versioned diffs.

## 0. Purpose
Test whether the **kinematic** side of Result C — a *bounded* Ruppeiner curvature on a metric whose
*volume* diverges (the double-descent peak as a geometric **fake** transition) — transfers from the
exactly-solvable Gaussian ridge (`module_L_ridge_curvature.py`, [V]) to a **feature-learning** net.
This produces knowledge: the falsifier can fire.

## 1. Object & estimator (reused, unchanged)
The ridge teacher–student statistical manifold: student weights `w` with `P(w) ∝ exp(−βE − βλ·½‖w‖²)`,
sufficient statistics `(E, ½‖w‖²)`, natural parameters `(θ¹,θ²)=(−β,−βλ)`. `log Z` is the Gaussian
integral diagonal in the eigenbasis `{μ_i}` of the design Gram matrix with teacher projections `{d_i}`.
Ruppeiner `R = −N/(2 det g²)`, engine pinned to `R=−1` on the normal family. Estimator functions
`setup / psi_derivs / R_curv / R_hp` are imported from `module_L_ridge_curvature.py` **unchanged**.

## 2. Rungs
- **Rung 1 (control, Phase 1):** Gaussian teacher–student (the design matrix *is* random Gaussian `X`).
  The estimator must reproduce the analytic kinematic signature. Mandatory; a failed control voids the run.
- **Rung 3 (test, Phase 2):** a narrow nonlinear feature-learning MLP; the estimator runs on the
  **learned-feature** Gram matrix `ΦᵀΦ/H` (Φ = trained hidden activations), sweeping `α = P/H`.
- **Rung 2b (optional, Phase 2b):** tiny GPT-2-class transformer — only after Phase 2 is clean.

## 3. Dual-precision certification (mandatory)
At every `α`, `R` and `det g` are computed in **float64** (`psi_derivs`+`R_curv`) and **40-dps**
(`R_hp`). An `α` is **certified** iff `|R_f64 − R_40dps| / |R_40dps| ≤ τ_prec`; **precision-contaminated**
`α` are excluded from the verdict (never interpreted). The exclusion rule has teeth: §6 includes a
**noiseless (σ=0) contamination control** demonstrating that float64 *does* break at the exact rank-1
collapse, so the guard is not decorative.

## 4. Execution order
Phase 0 (freeze §6, this commit) → Phase 1 (Rung 1 control; STOP if fail) → Phase 1b (precision-teeth
control) → Phase 2 (Rung 3 test, scored vs §6 + falsifiers) → Phase 2b (optional).

## 5. Falsifiers & verdict
For the **test** rung, on certified `α`:
- **PASS (kinematic):** the signature S1–S3 below holds — `|R|` bounded, `det g` volume-divergent, `R<0`.
- **F-genuine:** `|R| → ∞` (exceeds `B` *and grows* as the grid refines near `α=1`) on a
  **positive-definite** metric (`det g>0` throughout). The reportable surprise — **logged, not buried.**
- **F-spurious:** the divergence sits on `det g→0` / indefinite (`det g≤0`) — a coordinate artifact;
  **quarantined**, not a finding.

## 6. FROZEN PREDICTION  🔒 (locked 2026-06-15, pre-data)
**Config.** `β = 1`, ridge `λ = 1e-5`, teacher noise `σ = 0.5`, feature dim `N = 800` (Rung 1),
spectrum = Marchenko–Pastur (Gaussian design). Estimator at 40-dps; engine pinned `R=−1`.

**`α_grid`** (refined near 1): `0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05, 1.1, 1.3, 1.6, 2.0`.

**Tolerances.** `τ_prec = 1e-6` (certification exclusion). `τ_curve = 2×` (Rung-1 `max|R|` must land
within a factor 2 of the frozen reference and `≤ B`).

**R-bound** `B = 1e-2`. The frozen analytic reference (Gaussian, seed 3) is *bounded*: `max|R| ≈ 1.17e-3`
at `α=2`, rising monotonically from `~4e-12` at `α=0.5`; `det g` spans **8.8 orders** (`4.0e14 → 6.5e5`)
across the sweep; `R<0` throughout. This is the volume-divergent-but-flat-to-bounded kinematic signature.

**Signature (the verdict predicate).**
- **S1 (bounded):** `|R| ≤ B` on every certified `α`, and `max|R|` does **not** grow as the near-`α=1`
  grid is refined.
- **S2 (volume divergence present):** `det g_max / det g_min ≥ 10³` across the sweep (the metric volume
  genuinely diverges — the kinematic mechanism), with `det g > 0` throughout.
- **S3 (Gaussian/negative class):** `R < 0` on every certified `α` (the double-descent class, distinct
  from the Hagedorn `R→0⁺` positive side).

**Precision-teeth control (frozen expectation):** with `σ = 0` (noiseless), the float64/40-dps
disagreement near `α=1` **exceeds `τ_prec`** (the exact rank-1 collapse), so those `α` are excluded —
proving the certification guard bites.

## 7. Guards (frozen)
- No CurvAttention. No genuine-side numerics — §6.7 (the SGLD/replica object) stays a derivation-first
  gate, untouched by this run.
- Dual-precision certification mandatory; uncertified `α` never enter the verdict.
- Rung 1 before Rung 3; a failed control voids the run.
- Report findings, not a rubber-stamp; a fired **F-genuine** is a result, not a failure.
- Receipts byte-for-byte untouched; this run only adds new files. Branch/commit on the user's word.

## 8. Receipts
`module_L_ridge_transfer.py` — Phase 1 (Rung 1 + precision-teeth control) and Phase 2 (Rung 3 MLP).

## 10. Freeze date & changelog
**Frozen 2026-06-15** (pre-data). v0.1 initial registration. Amendments require a versioned diff.
