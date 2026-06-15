# IG-PRIMON-T1 — Operationalization Amendment (v0.5, 2026-06-15)

**Status:** signed amendment registering the **operationalization session** into the canonical record;
folds into a future consolidated-results-ledger v0.5. HONEST_CLAIMS throughout: **[V]** verified by a
reproducible receipt, **[partial]** validated-but-not-certified, **[conditional]** result is an explicit
if-and-only-if, **[analysis]** pencil derivation, **[infra]** tooling. No silent edits.

**One-line summary.** The experimental program was ported to an operational footing — an installable,
anchor-verified package and a hardware-execution track on real CUDA silicon — and Result C was carried
*off* the exactly-solvable archetypes toward real learning systems: the **kinematic** side transferred and
is `[V]`; the **genuine** side reduced, on paper, to a decidable rate competition, with the honest finding
that the gate (not a sampler) was always the bottleneck.

---

## A. Hardware-execution track (operational layer)

- **`[infra]` Installable package + `igprimon` CLI.** `ig_primon/` wraps the `[V]` receipts **without
  editing them** (byte-for-byte unchanged). `igprimon verify` re-checks **14 anchors** against the
  receipts' pinned values (engine `R=−1`, Gardner `α_c(0)=2`, Ising `s(0)=ln2`, Krauth–Mézard `0.833`, the
  perceptron `χ·Δα→3.22` and `|R|·Δβ²→11.8`, SK dAT, ridge double descent, radius/unit-rank, the `C`
  constant `<6e-31`); 23 tests; CI. Two guard-less receipts are run as subprocess and asserted against
  their own output. Erratum **E6** logged (the 31-digit `C` was within budget, not beyond).
- **`[V-realized]` Precision–Certification Firewall** (`module_hw_firewall.py`, doctrine §1 H1 flagship).
  Realized on the **RTX 5070 (Blackwell sm_120)**: a fast FP32 Tier-E explorer proposes, CPU/mpmath dps≥50
  Tier-C certifies, and a **near-miss kernel (~3–4× float32-eps) is caught** — *agreement is not
  verification*, demonstrated not asserted. Hardware doctrine amended to **v0.3** (real device map:
  i9-9900K + RTX 5070 + GTX 1660 Ti; the Snapdragon `NO-FP64-ACCEL`/`NPU-FORMAT` walls obsoleted).
- **`[V]` Precision-certification map for inference primitives** (`igprimon precision-matrix / precision-bf16
  / precision-fp8`, `T1_precision_map_v0_1.md` v0.3). GEMM/softmax/LayerNorm/attention ×
  {fp64,fp32,tf32,fp16,bf16,fp8} × {RTX 5070, GTX 1660 Ti, CPU}, each certified vs an fp64 reference.
  Findings: GEMM is the easy primitive (fp16 safe; tf32 the sweet spot; fp8 E4M3 at 134 TFLOP/s);
  **reductions need fp32-accumulate**; the fp16 LayerNorm overflow is a **range artifact bf16 avoids**;
  **attention is the most fragile** (context-scaling); softmax fp16 error **rises with peakedness** (not
  diffuseness — a corrected intuition); bf16 is ~8× coarser than fp16; fp8's binding constraint is **range,
  not mantissa**. Scope wall: numerical precision of inference primitives only (not speculative decoding /
  LLM-judge).

## B. Result C — carried off the exactly-solvable archetypes

- **`[V]` Kinematic transfer** (`PREREGISTRATION_ridge_kinematic_transfer_v0_1.md` + `module_L_ridge_
  transfer.py`, Amendment A1). The double-descent peak as a **geometric fake transition** — bounded
  Ruppeiner `R` on a volume-divergent, positive-definite metric — **transfers from the exactly-solvable
  Gaussian ridge to a feature-learning MLP**. Frozen pre-reg; Gaussian control reproduced out-of-sample;
  the binding adjudicator is the **finite-size-scaling refinement** (`max|R|` near `α=1` shrinks
  `1.0e-2 (H=64) → 1.1e-3 (H=512)` toward the Gaussian scale, `det g>0`), which **refuted** a mechanical
  finite-`H` F-genuine false positive. The kinematic side of Result C now holds *in the wild*.
- **`[partial]` Genuine-side instrument control** (`module_L_perceptron_mcmc_control.py`). Metropolis-on-
  sphere on the exactly-solvable spherical perceptron (the Θ-energy has no gradient → **not SGLD**),
  certifying the cumulant→geometry→`R` machinery against the `[V]` analytic receipt. Energy sector matched
  ~13% (MCMC error); the **Edwards–Anderson overlap `q*` was reproduced** — which **corrected the pencil**:
  `⟨q⟩ = q* ≠ 0` (spin-glass order), so `g_βε ≠ 0` and the receipt's `pxy=0` is a near-critical
  approximation, not an exact symmetry. The β-power normalization is **unresolved** (acceptance/β-lever too
  small); **NOT a clean certification.** The binding difficulty is equilibrating the glassy posterior — the
  difficulty the singular case faces by design.
- **`[conditional]` Genuine-side singularity gate** (`T1_genuine_side_singularity_gate_v0_1.md`). On paper:
  the genuine metric-curvature diagnostic (`|R|→∞` on `det g>0`) is **closed for a singular (real-net)
  posterior iff** its criticality co-locates with the singular limit `β→∞` **and** the overlap
  susceptibility grows no faster than `β²` (toy DOS `ρ(E)~E^{λ−1} ⟹ g_ββ~λ/β²`, `λ` = the RLCT; threshold
  `p=2`). **Otherwise — finite-`β` transition — the genuine signature survives** (`det g→+∞`). This
  **refuted** an earlier unconditional "closed" claim. Parity correction folded in: `pxy, pxxy, pyyy` are
  near-critical approximations on a `q→−q` parity that fails in the glassy phase — a certification must
  measure all three ε-odd terms.

## C. Honest scope walls (what is NOT claimed)

- **No CurvAttention.** The twin-forward-pass / attention-gating route is a structural category error
  (a forward-pass moment is not `χ_SG`); gated out, twice. The genuine object lives in the **learning
  ensemble** (weight replicas from a tempered posterior), not a forward pass.
- **No real-net genuine certification.** The genuine side is a **rate conditional**, not open or closed; the
  next move is a model for `χ(β)` on the singular set (pencil), not a sampler. Where it closes, the live
  invariant is the **LLC/RLCT** — not a Ruppeiner curvature.
- **SGLD untested.** The control used Metropolis (the Θ-energy is non-differentiable); the SGLD-specific
  machinery is exercised only on smooth energies, which is the singular case.
- **Pending on user's word:** branch `operational-layer` is unpushed; the Paper-3 PDF (with the
  Ersoy–Wiesner positioning) is uncompiled (no local TeX toolchain); Papers 1–2 undrafted.

## D. Write-up positioning
Paper 3 gains (i) the **Ersoy–Wiesner** (arXiv:2505.06597) related-work paragraph distinguishing the
Ruppeiner fluctuation geometry from their deterministic error-surface curvature, and (ii) the honest
scope statement the singularity gate supplies: **the curvature diagnostic classifies the non-singular
archetypes; the singular real-net case is a decidable rate competition that, where it closes, hands off to
the LLC.** That is the program's true, falsifiable "novel for AI" claim — sharper than any inference-time
curvature gadget.

## E. Record note — V/E/C on the maker-catches (both halves of the loop)
Four times the loop corrected its own maker, but they are not one species. **Three were errors overturned
by the data** — the precision expectation (`σ=0` would contaminate float64; it did not), the F-genuine
false positive (refuted by the FSS refinement), and `g_βε=0 "by symmetry"` (refuted by `⟨q⟩=q*`). **The
fourth was an overclaim hedged by the verify-half, not the data** — the *unconditional* singularity closure
was not wrong so much as overstated, and the toy-`det g` calculation (verification, no sampler) downgraded
it to the rate conditional. So the honest tally is **three data-caught errors + one verification-hedged
overclaim** — generate *and* verify each doing their job, which is the more accurate (and better) record.
