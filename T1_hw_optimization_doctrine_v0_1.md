# IG-PRIMON-T1 — Hardware-Optimization Doctrine (v0.2, 2026-06-14)

**Program ID:** IG-PRIMON-T1 (hardware-execution track)
**Status:** [GATE] — this document is the gate. It supersedes the *Hardware doctrine* section of
`T1_preregistration_v0_6.md` (which scoped FP64 CUDA / consumer-Blackwell / H100–H200 — hardware this
program does not run on) and re-targets the entire compute plan to the **real device**. No
hardware-execution claim enters the ledger until this doctrine is locked and the per-module gates below
are met.

**v0.2 (this revision).** The §4 premise was put through its own `[GATE]` — the derivation that should
have preceded the spec (`T1_moduleL_hw_gate_v0_1.md`). **It failed:** the frozen-forward-pass overlap
carries no statistical-geometric object (no free energy, no Hessian metric, no susceptibility), so §4 is
re-tagged **[C]** and **removed as flagship**. The honest near-term flagship is the §1 firewall realized
as a **local-model-as-verifier** harness (real value, runs today, needs none of §4). Result C's genuine
objects relocate to the **learning posterior over weights** = the program's parked §6.7. See §4, §6, and
the Changelog. v0.1 (prior): initial re-target; PCF; three Walls; toolchain gates; per-module map with
§4 as flagship.

**Discipline carried throughout (unchanged from the trunk).** HONEST_CLAIMS: **[V]** verified,
**[E]** defensible extrapolation, **[C]** conjecture, **[GATE]** derive/spec before numerics,
**[BLOCKED]** prerequisite missing — stated, not hidden. One new tag scoped to this track:
**[E-hw]** — a number produced by an accelerator (NPU/GPU) at reduced precision, *exploratory by
construction*, never promotable to **[V]** except by Tier-C reproduction (§1). No-silent-edit on
registered items; versioned-diff amendments only (§7).

**Why this document exists.** The trunk (`T1_consolidated_results_ledger_v0_3.md`) is a *hypothesis /
experiment set*: every `[V]` is a CPU correctness receipt at FP64 or 40–90 decimal places
(`module_L_*.py`, `module_e_*.py`, `audit_independent.py`). The prior doctrine's GPU roles were scoped
but never executed ("zero cloud spend through Phase 3"), and they assumed an FP64-throughput device.
The actual machine is an **ARM + NPU** Copilot+ PC whose accelerators are *low-precision*. Naïvely
"porting the math to the accelerator" would forfeit exactly the precision the `[V]` tags rest on. This
doctrine resolves that by making the precision mismatch the organizing principle rather than a defect.

---

## 0. Walls (hardware-specific; they bound everything below)

- **NO-FP64-ACCEL (structural, load-bearing).** Neither the Hexagon NPU (INT8/INT16/FP16) nor the
  Adreno X1-45 GPU (FP32/FP16) provides FP64 — let alone the 40–90 dps the program certifies at. The
  accelerators **cannot certify anything**. They can only *propose* candidate numbers. This is not a
  performance footnote; it is a hard ceiling that determines the entire pipeline architecture (§1).
- **NPU-FORMAT (GGUF ≠ QNN).** The local model is a GGUF artifact (`Qwopus3.5-4B-coder-Q8_0.gguf`),
  which the llama.cpp runtime executes on **CPU (Oryon NEON / i8mm)** or the **Adreno GPU
  (OpenCL/Vulkan)** — **not** on the Hexagon NPU. The Hexagon path requires a model compiled to a QNN
  context binary via the Qualcomm AI Engine Direct (QNN) SDK, which is **not installed** (§2). Until
  that conversion exists, the operational substrate for Result C (§4) is CPU/Adreno forward passes, and
  any "runs on the NPU" statement is **[BLOCKED]**, not assumed.
- **INFERENCE ≠ LEARNING (scopes the Result-C operational claim).** A forward pass of a *trained,
  frozen* LLM is **not** a training-time phase transition. The exactly-solvable `[V]` backbone of
  Result C (linear-ridge double descent; SK at the dAT instability) is a statement about *learning*
  manifolds. The operational protocol in §4 measures the **representation-overlap geometry of a live,
  frozen model** and *applies the diagnostic taxonomy as a classifier*. It does **not** claim to observe
  a learning transition on the local model. This wall is what keeps the narrative from outrunning the
  math — the same discipline the trunk was built on.
- **CERTIFICATION-LOCALITY.** Every invariant that the ledger reads (`R`, `det g`, `g_εε`, radii,
  amplitudes) is certified **only** on Tier-C (§1). An accelerator may *locate* a feature (a peak, a
  sign change, a soft mode); only Tier-C may *measure* it for the record.

---

## 1. The Precision–Certification Firewall (PCF) — the core architecture

Two tiers, one firewall between them. This is the hardware specialization of the trunk's standing rule
*"GPU output is exploratory until certified; agreement is not verification."*

**Tier-C — Certify.** Oryon ARM64 CPU; `mpmath`/`sympy` at **dps ≥ 40** (up to 90 where the trunk
requires it). 8 cores, NEON SIMD. The **sole authority for the `[V]` tag.** Pure-Python bignum
(no `gmpy2`, §2) — correct but throughput-limited; this constrains *how much* we certify, never
*whether* a certified number is trustworthy.

**Tier-E — Explore.** The accelerators, emitting **[E-hw]** candidates only:
- **Adreno X1-45 GPU** (FP32/FP16) via **ONNX Runtime DirectML EP** — reachable **today** (§2).
- **Hexagon NPU** (INT8/INT16/FP16, ~45 TOPS) via the **ONNX Runtime QNN EP** — **[BLOCKED]** until
  the QNN toolchain is installed (gate `H-NPU`, §2).

**The firewall rule (binding).** No Tier-E number may carry `[V]`. A Tier-E result is promoted to the
ledger only after Tier-C **reproduces it within the module's stated acceptance budget** (§5). The
accelerator's role is to be *fast and wrong-in-the-last-digits*; the CPU's role is to be *slow and
right*. A workflow that lets an `[E-hw]` number reach a paper untouched by Tier-C is a methodology
violation, identical in kind to a `[V]` tag without a reproducible receipt.

**Pipeline falsifier (of the kernel, not the mathematics).** If Tier-C **cannot** reproduce a Tier-E
candidate to the stated budget, the conclusion is *the accelerator kernel is wrong* (precision loss,
layout bug, quantization artifact) — **halt the module and audit the kernel.** The underlying theorem
is untouched; what failed is the hardware port. This mirrors the trunk's "an audit reporting zero
findings is presumptively not an audit": a Tier-E run that *always* agrees with Tier-C to full precision
is presumptively not exercising the accelerator.

---

## 2. Device map & toolchain gates (honest current state — verified 2026-06-14)

**The machine.**

| component | spec | precision | program role |
|---|---|---|---|
| **CPU** | Qualcomm Oryon (Snapdragon X Plus X1P42100), ARM64, 8c/8t | FP64 + arbitrary (mpmath) | **Tier-C, sole `[V]`** |
| **NPU** | Hexagon (Snapdragon X Plus), ~45 TOPS | INT8/INT16/FP16 | Tier-E (forward passes); **[BLOCKED]** |
| **GPU** | Adreno X1-45 | FP32/FP16 | Tier-E (grids, scans, MC) — **live** |
| **OS** | Windows 11 Pro ARM64, build 26200 | — | — |
| **Python** | 3.14 ARM64 (`C:\Python314`) | — | — |

**Toolchain — present (Tier-C ready, Tier-E/Adreno ready).**

| package | version | tier | note |
|---|---|---|---|
| `mpmath` | 1.3.0 | Tier-C | high-precision authority |
| `numpy` / `scipy` / `sympy` | 2.4.2 / 1.17.1 / 1.14.0 | Tier-C | exact-saddle, Gauss–Hermite, symbolic |
| `onnxruntime-directml` | 1.24.4 | Tier-E | **Adreno GPU reachable now** (DirectML EP) |
| `torch` | 2.12.0 | Tier-E (CPU) | forward-pass probes; ARM64 CPU build |
| LM Studio + `Qwopus3.5-4B-coder-Q8_0.gguf` (+ `mmproj-F32.gguf`) | — | §4 substrate | runs CPU/Adreno via llama.cpp |

**Toolchain — missing (the gates).**

- **`[BLOCKED] H-NPU`** — Hexagon execution needs (i) the **QNN SDK** (Qualcomm AI Engine Direct),
  (ii) `onnxruntime-qnn` (the QNN EP), and (iii) a **QNN context binary** of the target model/op-graph
  (NPU-FORMAT wall). **Empirically verified blocked (2026-06-14):** `onnxruntime.get_available_providers()`
  returns `['DmlExecutionProvider', 'CPUExecutionProvider']` only — **no `QNNExecutionProvider`** — and no
  QNN HTP backend DLL exists anywhere on disk. Worse, the gate is not a one-line install on *this* stack:
  the only `onnxruntime-qnn` wheel resolvable for **cp314 / native-ARM64** is `…-cp314-cp314-win_amd64.whl`
  (x86-64 — wrong architecture to drive the Hexagon NPU, and would collide with the native-ARM64
  `onnxruntime-directml`). Clearing `H-NPU` is therefore a scoped sub-project: a native `win_arm64` QNN EP
  (likely on an older Python such as 3.12-arm64 where Qualcomm publishes wheels) or the QAIRT / AI-Hub
  toolchain, plus model conversion. **Until `H-NPU` clears, Tier-E runs on the Adreno GPU via DirectML**
  (`DmlExecutionProvider`, confirmed live); the NPU is a scoped upgrade, **not** a precondition for
  operational status.
- **`gmpy2` absent (perf note, not a gate).** `mpmath` falls back to pure-Python integers — correct,
  but multi-× slower on the bignum-heavy certifications (the 90-dps `C` constant, the 40-dps `R_hp`).
  Acceptance budgets are unaffected (they are correctness budgets). Optional remediation: build/install
  `gmpy2` for ARM64, or stage the heaviest certifications. Tracked, not blocking.

---

## 3. Per-module hardware mapping

The discriminating question per module: *what does the accelerator actually accelerate, and can Tier-C
certify the result it proposes?* Modules whose work is throughput-light FP64 stay on the CPU; modules
with genuine throughput pressure get a Tier-E explorer behind the firewall.

| module | Tier-E (accelerator) role | Tier-C certification | device fit | gate |
|---|---|---|---|---|
| **A** — Fisher metric of the primon gas | optional FP32 pole-subtracted β-grid sweep (Adreno), `I(β)−(β−1)⁻²` to avoid cancellation | mpmath dps≥50 series + direct, ≤1e-12 inside radius | throughput-light; **CPU alone suffices** | none beyond §1 firewall (low priority) |
| **B** — Knauf spin chain (Monte Carlo) | parallel-tempered Metropolis, FP16/FP32 on **Adreno** (DirectML/compute) | **exact finite-N recursion on CPU** (the H2.1 validation gate) | Adreno-native; **NPU ill-suited** (rejection/RNG/branching) | H2.1: MC-vs-exact within 3σ or **halt** |
| **C** — two-parameter curvature grid | FP32 ψ-derivative grid (Adreno), singular part pre-subtracted analytically | FP64/mpmath spot certification at ≥20 points, dps≥50 | Adreno FP32; the doctrine's "FP64 grid" → **CPU FP64** | derivation **locked** (prereg H3.2, v0.4) + ≤1e-12 |
| **D** — zeros as Fisher zeros (scan) | FP32 argument-principle sweep over rectangles (Adreno) | **CPU-Arb interval** certification of any anomaly | FP64-light scan; Adreno fits | LMFDB/Odlyzko match within FP64 budget |
| **L** — Result C (learning diagnostics) | ~~forward-pass overlap~~ **[C], GATE-FAILED** (`T1_moduleL_hw_gate_v0_1.md`): no Gibbs object on frozen activations. Relocated to the §6.7 learning posterior (teacher–student perceptron, CPU/numpy) | curvature/replicon at **40 dps on CPU** once a valid object exists (`module_L_ridge_curvature.py`, `module_L_SK_converse.py`) | CPU (Tier-C); LLM has **no science role** here | derive §6.7 object first |

**Flagship (v0.2): the firewall realized as a local-model-as-verifier harness** (§6 H1), *not* Module L —
the §4 forward-pass route was gated and failed (§4 below). A/C/D are re-targeted for completeness and are
the natural firewall shakedown vehicles; B is the only module with a hard accelerator dependency for its
*science* (its statistics need the throughput), and even there the certification is the exact recursion on
CPU. Module L's science returns only via the §6.7 learning posterior (a CPU derivation, not a hardware
pipeline) — see §4 and `T1_moduleL_hw_gate_v0_1.md`.

---

## 4. Result-C operationalization — GATED, FAILED → [C] (superseded by the §6.7 relocation)

**This section's v0.1 premise did not survive its own gate.** Per the program's `[GATE]` rule (derive
before you measure), the forward-pass-overlap proposal was put through `T1_moduleL_hw_gate_v0_1.md`
*before* any pipeline was built. **Verdict: a frozen model's two-forward-pass overlap carries no
statistical-geometric object.** The measured overlap is a *moment* (the Legendre dual of a natural
parameter); there is no `exp(ε·O)` tilt of a fixed-weight model, hence no free energy `ψ(·,ε)`, no Hessian
metric `g = ∂²ψ/∂θ²`, and therefore no susceptibility `g_εε = ∂²ψ/∂ε²` and no `|R|→∞` theorem. A
max-entropy surrogate *can* be fitted, but its curvature describes the surrogate's estimation geometry,
not the model's learning. Full derivation + the (R1)–(R4) checklist: `T1_moduleL_hw_gate_v0_1.md`.

**Re-tag.** The v0.1 §4 claim is **[C]** (conjecture/analogy), **not [E-hw]** — an `[E-hw]` tag asserts a
reduced-precision result of a *valid* computation, and §4 had no valid computation to be imprecise about.
§4 is **removed as the operational flagship**. The `INFERENCE ≠ LEARNING` wall (§0) barred the wrong
*interpretation* but not the *non-existence of the object*; this gate supplies the missing test, so the
wall now stands for the right reason.

**Relocation — where Result C's objects genuinely live.** The learning manifold is the **Gibbs posterior
over weights** `P_β(w) ∝ exp(−βL(w;D) − βλ·½‖w‖²)` — the ridge module generalized — with the divergent
side given by **two replicas coupled through `ε` conjugate to the weight/function overlap** `q = w¹·w²/N`,
whose susceptibility `β²·Var(q)` diverges at an RSB/dAT instability. That is the SK construction on a
*learning* posterior, and it is exactly the program's already-parked **§6.7** (Ising/Gardner perceptron
storage transition). The honest minimal realization is a small **teacher–student perceptron**
(numpy/torch, CPU/Tier-C, continuous with `module_L_ridge_curvature.py`) — **no LLM, no NPU, no
`trl`/`peft`** (the latter absent on this box anyway). The frozen local model has **no role in the
science** of Result C; its honest role is operational verification infrastructure (§6 H1).

**Historical note (no-silent-edit).** The struck v0.1 §4 proposed: two noisy Qwopus-4B passes over a
fixed prompt batch `𝒫` → per-layer representation overlap `ε(ℓ) = ⟨H¹_ℓ,H²_ℓ⟩/‖·‖‖·‖` → the four-tier
curvature taxonomy over a device-knob control (Q8↔Q4 / temperature / depth), tagged `[E-hw]`. It is
recorded here as the falsified prediction it became, per the program's changelog discipline — not deleted.

---

## 5. Acceptance budgets & falsifiers

Per-module, in the trunk's style (a falsifier that, if tripped, halts the module and escalates to audit).

- **Firewall (global).** Every `[E-hw]→[V]` promotion: Tier-C reproduces the Tier-E candidate within the
  module budget below. **Falsifier:** non-reproduction ⇒ kernel audit, no ledger entry.
- **Module A.** Adreno FP32 grid vs mpmath series, **≤ 1e-12 relative** inside the radius of convergence;
  CPU-Arb agreement ≥ 20 spot points at dps≥50. Falsifier: any point outside 1e-12 not explained by
  documented FP32 round-off.
- **Module B.** GPU MC vs exact finite-N Knauf recursion (CPU), **within 3σ** of pre-stated MC error
  across β∈[1.2,3]. Falsifier: any systematic > 3σ deviation ⇒ **module halts** (nothing downstream is
  interpretable — same rule as prereg H2.1).
- **Module C.** Derivation locked (prereg H3.2 / v0.4, `R ~ Δ₃(α+1)/(2A²)·ε^{2α}`); Adreno FP32 grid
  certified at ≥20 points to **≤ 1e-12** away from the transition line; the `(L+κ)⁻²` model of record
  (pure-power fits recorded as structurally misleading). Falsifier: locked-derivation vs grid disagreement
  outside CI ⇒ audit.
- **Module D.** Adreno scan reproduces `N(T)` and the first 10⁴ ordinates vs LMFDB/Odlyzko within the
  FP64 budget; any candidate anomaly → **CPU-Arb interval certification** before it is reported.
  Falsifier: any Arb-certified anomaly (reported, not interpreted).
- **Module L (relocated to §6.7 — v0.2).** The forward-pass route is gated out (§4); the live budget is
  the **teacher–student perceptron** learning posterior. (i) **Object first:** a Gibbs measure with `ε`
  as a *natural parameter* conjugate to the replica overlap `q` (not a measured moment) must be written
  before any curvature is computed — the gate that §4 failed. (ii) **Diagnostic well-posedness:**
  `det g > 0` on the certified manifold before any `|R|→∞` reading is interpreted (a `det g→0` divergence
  is the degenerate/disqualified case, not criticality). (iii) Verdicts certified at Tier-C (40 dps),
  continuous with `module_L_ridge_curvature.py` / `module_L_SK_converse.py`. Falsifier: any `|R|→∞` claim
  whose metric is not the Hessian of an explicit free energy ⇒ not a finding (the §4 failure mode).

Negative results are registered regardless (trunk rule).

---

## 6. Sequencing (Phases H0–H3)

- **H0 — Lock this doctrine.** This document is the `[GATE]`. No hardware-execution claim is registered
  until H0 closes (user review, §below).
- **H1 — Firewall shakedown = the flagship (v0.2).** Stand up the Tier-C ↔ Tier-E harness on the
  *easiest* module (A or C): prove an Adreno FP32 number reproduces at CPU mpmath within budget, and prove
  a *deliberately under-precision* kernel **fails** the firewall (so the falsifier has teeth). Deliverable:
  one `module_hw_firewall_*.py` receipt + a ledger line. This is **auditable local generate-and-verify
  infrastructure** — real near-term value, runs on Oryon today, depends on none of the gated §4. It is the
  honest operational deliverable, named for what it is (not "operationalizing Result C").
- **H2 — Result-C science via §6.7 (derivation, not a hardware pipeline).** Realize the relocated object
  (§4): a small teacher–student perceptron learning posterior with `ε` conjugate to the replica overlap,
  curvature certified at Tier-C, continuous with `module_L_ridge_curvature.py`. Pure CPU/numpy; the LLM and
  NPU play no part. This is the genuine Result-C extension the v0.1 §4 tried to leapfrog.
- **H3 — Modules B/D as throughput justifies.** Adreno MC (B) and Adreno scan (D) only if the science
  needs the throughput; both certify on CPU. Module A grid is optional (CPU suffices). **NPU sub-project:
  not scoped** — no validated workload to accelerate (the §4 workload was gated out).

**Hardware doctrine of record (supersedes prereg §"Hardware doctrine").** Tier-C (Oryon/mpmath) is the
certification authority for the entire program; Tier-E (Adreno now, Hexagon after `H-NPU`) is exploratory
behind the firewall; **zero cloud spend** — the H100/H200 path named in the prereg belongs to a different
workload and is not triggered here.

---

## 7. Changelog

**v0.1 (2026-06-14).** Initial registration of the hardware-execution track. Supersedes the *Hardware
doctrine* section of `T1_preregistration_v0_6.md` (FP64 CUDA / H100–H200 — wrong hardware). Establishes:
the real device map (Snapdragon X Plus X1P42100 — Oryon ARM64 CPU + Hexagon NPU + Adreno X1-45 GPU, Win11
ARM64, Python 3.14), verified 2026-06-14; the **Precision–Certification Firewall** (Tier-C CPU/mpmath sole
`[V]`; Tier-E accelerators `[E-hw]` only) and the new `[E-hw]` tag; the three hardware Walls (`NO-FP64-ACCEL`,
`NPU-FORMAT`, `INFERENCE ≠ LEARNING`); the toolchain gates (`[BLOCKED] H-NPU` for Hexagon; `gmpy2` perf
note); the per-module mapping (A/B/C/D + L-flagship); the Result-C operational protocol on the live
Qwopus-4B model; acceptance budgets/falsifiers; and the H0–H3 sequencing. No claim of the trunk ledger
(`T1_consolidated_results_ledger_v0_3.md`) is altered; this track adds an execution layer beneath the
existing `[V]` results, it does not touch them. Amendment rule: versioned diff, no silent edit.

**v0.2 (2026-06-14).** §4's premise gated and **falsified** before any build (`T1_moduleL_hw_gate_v0_1.md`):
a frozen model's forward-pass overlap carries no statistical-geometric object — `ε` is a moment, not a
natural parameter; no free energy, no Hessian metric, no susceptibility, no `|R|→∞` theorem (the steelman
max-ent surrogate measures estimator geometry, not learning). Changes: §4 re-tagged **[C]** and **removed
as flagship** (struck-but-recorded per no-silent-edit); the **flagship becomes the §1 firewall as a
local-model-as-verifier harness** (§3, §6 H1); Result C's science **relocated to the learning posterior =
parked §6.7** (teacher–student perceptron, CPU/numpy — no LLM/NPU/`trl`/`peft`), now §6 H2; §3 table L row,
§5 Module L falsifier, and §6 H1–H3 updated to match; **NPU sub-project de-scoped** (no validated workload).
Stack verified 2026-06-14: `trl`/`peft`/`accelerate`/`datasets` absent; `transformers 5.9.0`, `torch 2.12.0`,
numpy/scipy/mpmath present. No claim of the trunk ledger altered. The error this records is precisely the
inversion the program's `[GATE]` rule exists to prevent — gate caught it; logged, not smoothed.

— End of Hardware-Optimization Doctrine v0.2. Amendments require a versioned diff; silent edits void the registration.
