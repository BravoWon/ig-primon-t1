# IG-PRIMON-T1 — Information Geometry of Arithmetic Gases

A disciplined, computationally-assisted research program on the **Fisher / Ruppeiner information geometry**
of prime (primon / Riemann-gas) and related disordered systems. The canonical state of the program lives in
the **consolidated results ledger** (`T1_consolidated_results_ledger_v0_4.md`); the frozen predictions are in
the **pre-registration** (`T1_preregistration_v0_6.md`). The **operationalization session** — installable
package, hardware-execution track, Result-C kinematic transfer `[V]`, and the genuine-side singularity-gate
analysis — is registered in `T1_resultC_amendment_v0_5.md`.

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

Python 3.11+ with `numpy`, `scipy`, `mpmath` (40-dps where cancellation requires it). Run any receipt
directly, e.g. `python module_L_perceptron_finiteT.py`. All **certification** is CPU-side.

## Operational layer (`ig_primon`, v0.4)

The experimental receipts are packaged as installable software **without editing the `[V]` artifacts**
(the program's no-silent-edit discipline). The `ig_primon` package wraps them with a machine-checkable
anchor suite, a hardware scan, and the doctrine's Precision–Certification Firewall.

```bash
pip install -e ".[dev]"          # certification stack (numpy/scipy/mpmath) + pytest
pip install -e ".[dev,gpu]"      # also CuPy; then: pip install "cupy-cuda12x[ctk]" for the CUDA headers

igprimon verify                  # run every anchor (the operational acceptance gate); exit 0 = all reproduced
igprimon verify --quick          # skip the slow high-precision anchors
igprimon verify --json           # machine-readable report
igprimon list                    # list anchors, groups, and receipts
igprimon run <receipt>           # run a receipt's full certification output (e.g. perceptron-finiteT)
igprimon hwscan                  # scan the device; print the Tier-C / Tier-E map
igprimon firewall                # Precision–Certification Firewall (CUDA Tier-E if available)
igprimon precision-matrix        # certify inference primitives (GEMM/softmax/norm/attention) × {device}×{precision}
igprimon precision-matrix --sweep  # reduction-width / context-length precision fragility
igprimon precision-bf16          # bf16 pass (torch): real-inference map + the LayerNorm range inversion
igprimon precision-fp8           # fp8 pass (torch): Blackwell fp8 GEMM + the range-vs-mantissa tradeoff
igprimon run depth-map           # T1 precision depth-N error map receipt (for precision-composition experiment)
```

**Anchors.** `igprimon verify` re-checks, programmatically, every exact reference value the receipts pin —
the engine `R=-1` pin, the flat-product `R=0` control, Gardner `α_c(0)=2`, Ising `s(0)=ln2`, Krauth–Mézard
`α_RS≈0.833`, the replicon→Gardner limit, the perceptron `χ·(α_AT−α)→3.22` and positive-definite
`|R|·(β_AT−β)²→11.8`, the SK `h=0` closed form and dAT `χ_SG·λ_AT→const`, the ridge validation and
double-descent dichotomy, the radius/unit-rank dictionary, and the registered `C` constant within its
`6e-31` budget. A drift (a dependency bump, a precision regression) now fails CI instead of going unnoticed.

**Precision–Certification Firewall** (`module_hw_firewall.py` / `ig_primon.firewall`). The hardware track's
H1 flagship, realized around its real thesis — *agreement is not verification*. A fast FP32 **Tier-E**
explorer (a CUDA GPU when present, else vectorised CPU FP32) proposes candidates; the slow exact **Tier-C**
authority (mpmath, dps≥50) adjudicates. The load-bearing case is a **near-miss** kernel wrong by only ~3–4×
the float32 epsilon: it passes an FP32-grade tolerance *and* would fool an FP32 "certify-by-agreement"
reference (whose own error is ~1 eps), yet Tier-C **rejects** it because its deviation exceeds what FP32
roundoff can explain. An honest kernel (within the FP32 noise floor) certifies; a gross kernel fails outright.
A GPU number is `[E-hw]` (exploratory); only a Tier-C reproduction within the noise floor licenses `[V]`.

**Precision-certification matrix** (`ig_primon.precision` / `igprimon precision-matrix`, plus
`igprimon precision-bf16` via torch). The firewall applied to the inference primitives (GEMM, softmax,
LayerNorm, attention) across `{RTX 5070, GTX 1660 Ti, CPU} × {fp64, fp32, tf32, fp16, bf16}`, each
certified against an fp64 reference. Findings: GEMM is the easy primitive (fp16 safe; tf32 the sweet
spot); **bf16 — the deployed format — is ~8× coarser than fp16** uniformly, so fp16-safe ≠ bf16-safe.
Attention is the most fragile primitive and the one whose fragility genuinely scales with **context**
(fp16 breaks by ~256). LayerNorm reduces over the **hidden dim, not context**, and its fp16 sum-of-squares
overflow is a *range artifact* that **bf16 avoids entirely**. Sharp-logit (peaked) softmax — not diffuse —
is the fp16 soft spot. **fp8** (`precision-fp8`) runs real Blackwell tensor-core GEMM (E4M3 at 134 TFLOP/s,
2× bf16) and shows the defining fp8 tension: E4M3 is finer but *saturates* past its ±448 range (the LLM
activation-outlier regime), E5M2 holds bf16-range but is mantissa-starved — so range, not mantissa, is what
binds. Scope: inference numerics only. Whether locally-safe ops compose to globally bounded output through N
layers is the named open frontier (`T1_precision_map_v0_1.md`), not built.
