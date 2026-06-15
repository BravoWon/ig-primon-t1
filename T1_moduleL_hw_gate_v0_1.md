# IG-PRIMON-T1 — Module L Hardware-Operationalization GATE (v0.1, 2026-06-14)

**Status: [GATE].** Pure derivation, no hardware, no budget. This document answers the question the
`[GATE]` rule exists to force, *before* any measurement pipeline is specified or built:

> Does the §4 protocol of `T1_hw_optimization_doctrine_v0_1.md` — "two noisy forward passes of a frozen
> model → representation-overlap `ε` → the Ruppeiner curvature / replicon taxonomy" — have a
> well-defined statistical-geometric **object** (a free energy whose Hessian is the metric and whose
> `ε`-second-derivative is a susceptibility), the way Result C's exactly-solvable backbone does?

**Verdict (derived below): NO, as specified.** The construction fails at the level of *what is being
differentiated*. §4 is therefore re-tagged **[C]** (conjecture/analogy), **not [E-hw]**, and is removed
as the operational flagship. The objects Result C describes live on the **learning posterior over
weights**, which is the program's already-parked §6.7 (perceptron storage transition) — not on
frozen-inference activations. Detail follows.

---

## 1. The object that must exist (the SK / ridge template)

Result C's curvature is a *theorem* because both validated sides are exponential families with a genuine
thermodynamic geometry. The machinery requires four things; call them the **gate checklist**:

- **(R1) An ensemble.** A family of probability measures `p_θ(x) ∝ exp(θ·T(x))` over a state space `x`,
  indexed by **natural parameters** `θ`, with **sufficient statistics** `T(x)`.
- **(R2) A free energy.** `ψ(θ) = log ∫ exp(θ·T(x)) dx` — the log-partition / cumulant generating
  function.
- **(R3) A metric that is a Hessian.** `g_ij = ∂²ψ/∂θ_i∂θ_j = Cov_θ(T_i, T_j)`. Because it is a
  covariance of sufficient statistics, it is **positive semidefinite by construction**; its third
  derivatives (`∂³ψ`) are the consistent inputs the Ruppeiner scalar `R = −N/(2 det g²)` needs.
- **(R4) Criticality as a divergence of a covariance component.** A susceptibility `g_εε = ∂²ψ/∂ε² =
  Var_θ(O)` diverges when a conjugate field `ε` couples to an order-parameter statistic `O` whose
  fluctuations blow up (the replicon → 0). That divergence, **on a still-positive-definite metric**, is
  what `|R| → ∞` certifies.

**SK converse instance** (`module_L_SK_converse.py`): `θ ⊇ {ε}`, `O = Σ_i s¹_i s²_i` (inter-replica
overlap), `ψ₂(β,h,ε)` the two-replica RS free energy, `g_εε = ∂²ψ₂/∂ε² = β χ_SG`, diverging at dAT.
**Ridge instance** (`module_L_ridge_curvature.py`): `p(w) ∝ exp(−βE − βλ·½‖w‖²)`, `θ = (−β,−βλ)`,
`T = (E, ½‖w‖²)`, `g = Hess log Z`. Both satisfy (R1)–(R4) exactly. That is the bar §4 must clear.

---

## 2. The test: does the frozen-forward-pass overlap provide it?

Walk the checklist for the §4 construction (model weights **fixed**; "noise" = sampling temperature `T`,
input jitter, quantization; `ε(ℓ) = ⟨H¹_ℓ,H²_ℓ⟩/‖H¹_ℓ‖‖H²_ℓ‖` a measured overlap between two passes).

- **(R1) Ensemble?** *Partially.* The noise induces **some** distribution `Q_φ(H)` over activations,
  parameterized by knobs `φ` (temperature, jitter). So a measure exists.
- **(R2)/(R3) The fatal gap — `ε` is a moment, not a natural parameter.** A frozen model is **not**
  defined by a variational free energy one extremizes over the overlap. There is **no term
  `exp(ε·O)`** in the generating measure of fixed weights, hence **no `ψ(·,ε)`** and **no self-consistent
  saddle in `ε`**. The measured overlap is `ε ≈ E_Q[O]` — a **moment**, the Legendre **dual** of a
  natural parameter, not the parameter itself. The Ruppeiner/susceptibility story is told on the `θ`
  (natural-parameter) side: `g = ∂²ψ/∂θ²`. §4 places its coordinate on the **moment** side and then
  applies `θ`-side machinery. You cannot differentiate `ψ` with respect to a moment to recover a
  covariance; that is a category error in the Legendre geometry, independent of any numerics.
- **Consequence.** A "metric" built by finite-differencing measured overlaps is **not** the Hessian of
  any log-partition function. It carries **no PSD guarantee**, its "third derivatives" have **no
  consistent generating object**, and so `R = −N/(2 det g²)` is an arbitrary ratio of empirical finite
  differences — **not** the Ruppeiner curvature of a manifold. `|R| → ∞` would then correspond to **no
  replicon, no dAT, no theorem**: just a small denominator.
- **(R4) Criticality?** Vacuous — there is no susceptibility-as-second-derivative to diverge, only the
  empirical variance of a correlation, which is a property of the noise process, not of a phase boundary.

**Checklist result: (R1) weak, (R2) absent, (R3) absent, (R4) vacuous.** Fails.

---

## 3. The steelman, and why it still does not deliver Result C

The only honest way to manufacture a Gibbs object here is a **maximum-entropy surrogate**: fit an
exponential family `p_η(H¹,H²) ∝ exp(η·O + …)` to the measured activation statistics, with `η` a genuine
natural parameter whose expectation `E_η[O]` equals the measured overlap. This *does* satisfy (R1)–(R3)
— but observe what it costs:

1. **The coordinate flips.** The conjugate field is now `η` (the natural parameter), and the measured
   overlap is `E_η[O] = ∂ψ/∂η` — its **dual**. §4's "use the measured overlap `ε` as the coordinate" was
   already in the wrong frame; the valid object is parameterized by `η`, not by the overlap.
2. **The meaning collapses.** `g` and `R` are now properties of the **surrogate fit's estimation
   geometry** — the information geometry of *your maximum-entropy model of the activations* — **not** of
   the model's learning dynamics. `|R| → ∞` would flag a degeneracy of the *fit* (e.g. the moment-matching
   becoming ill-posed), which is a statement about your estimator, not a learning phase transition. It is
   not the dAT replicon and does not inherit Result C's interpretation.

So even the steelman confirms the gate's point: **derive the object first.** The surrogate is buildable
but answers a different question; the frozen-forward-pass overlap does not carry the learning-transition
geometry §4 claimed for it.

---

## 4. Verdict and re-tag

- **§4-as-specified is [C]**, a conjecture/analogy — not **[E-hw]**. An `[E-hw]` tag asserts a *reduced-
  precision result of a valid computation*; §4 has no valid computation to be imprecise about. The
  re-tag is the honest correction.
- **§4 does not headline.** It is removed as the operational flagship of the hardware track.
- The `INFERENCE ≠ LEARNING` wall (doctrine §0) was **necessary but not sufficient**: it correctly barred
  the *interpretation* (don't call a forward pass a learning transition) but did not test the *existence*
  of the computed quantity. This gate supplies the missing test. The wall stays; it is now load-bearing
  for the right reason.

---

## 5. Where Result C's objects genuinely live (the honest relocation)

The learning manifold is the **Gibbs posterior over weights**, not the activation overlap of frozen
inference:

`P_β(w) ∝ exp(−β L(w;D) − βλ·½‖w‖²)`  — natural params `(−β,−βλ)`, sufficient stats `(L, ½‖w‖²)`.

This is the *literal generalization* of the ridge module (which is its linear/Gaussian special case),
so (R1)–(R3) hold by the same argument that already earned the ridge result its **[V]**. The
genuine-criticality (divergent) side is the **replicated** posterior: two replicas `w¹,w²` from `P_β`
(or trained on shared data `D`), coupled by a field `ε` conjugate to the **weight/function overlap**
`q = w¹·w²/N`. Then

`g_εε = ∂²ψ₂/∂ε² = β² Var(q)`  — the overlap susceptibility, which diverges at an RSB / dAT-type
instability. **This is the SK construction transported onto a learning posterior** — `|R| → ∞` is a
theorem here, and the ensemble is over *weights/training*, so it honors `INFERENCE ≠ LEARNING` by
construction.

**This is not new territory to invent — it is the program's own parked §6.7** (ledger v0.3): "a literal
*learning* realization of the divergent branch … the Ising / Gardner perceptron storage transition."
§4 tried to *leapfrog* §6.7 (the hard, real object) with a frozen-inference shortcut that has no object.
The gate's instruction is: **go through §6.7, not around it.**

**Minimal rigorous realization (stack-checked):** a small trainable **teacher–student perceptron** —
numpy/torch only, certifiable on the Oryon CPU at Tier-C, continuous with `module_L_ridge_curvature.py`.
It needs **no 4B model and no `trl`/`peft`/`accelerate`/`datasets`** (absent on this box, verified
2026-06-14); `transformers 5.9.0` + `torch 2.12.0` are present but not required for the minimal object.
The frozen LLM has no role in the *science* of Result C; its honest role is operational infrastructure
(§6).

---

## 6. Consequence for the hardware doctrine

- **Demote §4** from flagship to a **[C]** appendix pointing at this gate. Amend
  `T1_hw_optimization_doctrine` to v0.2 accordingly (versioned diff, no silent edit).
- **The honest near-term build** is the Precision–Certification Firewall + a **local-model-as-verifier**
  harness: auditable generate-and-verify infrastructure (the Oryon-CPU mpmath authority checking
  candidate numbers), which has real near-term value, runs today, and needs **none** of §4. Label it as
  what it is — *auditable local verification infra* — not "operationalizing Result C."
- **NPU sub-project: do not scope it now.** Not because it is merely non-blocking (doctrine §2), but
  because there is **no validated workload to accelerate**. A Q8 4B on Oryon is already a usable verifier.
- **The science next step, if pursued,** is the §5 derivation realized as the teacher–student perceptron
  (the §6.7 branch) — a derivation-and-small-compute task on the CPU, not a hardware pipeline.

---

## 7. Changelog

**v0.1 (2026-06-14).** Gate executed on the doctrine's §4 premise. Verdict: the frozen-forward-pass
representation overlap carries **no statistical-geometric object** (no free energy, no Hessian metric,
no susceptibility) — `ε` is a moment, not a natural parameter (Legendre-frame error), and the steelman
max-ent surrogate measures estimator geometry, not learning. §4 re-tagged **[C]**, removed as flagship.
Result C's genuine objects relocated to the learning posterior over weights = the program's parked §6.7
(teacher–student / perceptron). Stack verified: `trl`/`peft`/`accelerate`/`datasets` absent;
`transformers 5.9.0`, `torch 2.12.0`, numpy/scipy/mpmath present; minimal object needs only the latter.
No claim of the trunk ledger altered. This gate triggers `T1_hw_optimization_doctrine` v0.1 → v0.2.

— End of Module L hardware gate v0.1. Amendments require a versioned diff; silent edits void the registration.
