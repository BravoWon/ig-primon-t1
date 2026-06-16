# Pre-Registration — Certified Precision-Allocation for Small Transformers

**T1_precision_map v0.2 — "from primitives to a shippable allocator"**

**Program ID:** IG-PRIMON-T1 (hardware-execution / operationalization track)
**Status:** `[PREREG]` — application-first frozen prediction. Successor to `T1_precision_map_v0_1.md` (the inference-primitive precision matrix, `[V]`), which named this exact open frontier: *whether locally-precision-safe ops compose to globally bounded output through N layers.* Frozen **before any depth-N scan or allocation run**, per the program's control-before-scan and derive-before-numerics rules.
**Date locked:** 2026-06-16 (v0.2, pre-data). The five prior-art citations were verified **at the primary source on 2026-06-16** (see §0.1); the §3.1 derivation gate is **done** (`precision_recursion_gate.py`, amendment v0.2.1) — it certified the recursion + softmax Jacobian and sharpened P1/F1 to the measured typical-case. **Status: FROZEN, Stage-1-admissible.**
**Discipline:** HONEST_CLAIMS — **[V]** verified by a reproducible receipt, **[E]** defensible extrapolation, **[C]** conjecture, **[GATE]** analytic derivation required before any numeric claim is registered. No silent edits; amendments are versioned diffs.
**Scope:** Inference-only forward pass; decoder-only transformers ≤ 355M params; single consumer GPU (RTX 5070, Blackwell sm_120: native FP8 *and* FP4 5th-gen tensor cores, 12 GB GDDR7) + CPU (mpmath, dps≥50) as the certification authority.
**Receipts (to be written; none may pre-empt a `[V]` tag):** `precision_recursion_gate.py`, `precision_depth_map.py`, `precision_allocator.py`, `allocator_bakeoff.py`.

---

## 0. Governing principle and reframe

**The dual maxim (design principle).** *There is no worth to theory without application, and no worth to application without the theory that certifies it isn't a mirage.* This program's own receipts prove the second clause — the +0.93, the 670× (rank-unfolding destroying spectral variance), the GOE/GUE symmetry-class mismatch, the jTopo misattribution were all *applications* (code, runs, numbers, headlines) of **negative** worth because the gating theory was skipped or weak. Accordingly, in this registration the **shippable instrument is the hypothesis (§5) and the certification is the gate (§3)** — not the reverse. The first version of this plan inverted that (a measured depth-curve as the deliverable, the instrument as a Stage-3 appendix); v0.2 corrects it.

### 0.1 Verified foundation (source-confirmed at primary source 2026-06-16)

Three load-bearing prior-art anchors + two adjacent papers, each checked at the primary arXiv source this session (not accepted from a synthesis subagent — *agreement is not verification* applies to citations too). The body-level claims below were the ones an earlier synthesis draft asserted as "confirmed" before they had actually been read; they are confirmed **verbatim at source** as recorded here, and the one number the draft got wrong (LAMP's operating points) is corrected (see changelog).

| anchor | source-verified (2026-06-16) | role here |
|---|---|---|
| **Budzinskiy, Fang, Zeng, Petersen — "Numerical Error Analysis of Large Language Models," arXiv 2503.10251** (13 Mar 2025; Vienna + Huawei) | ✅ verbatim. Theorem 4.30 bounds the L-block output round-off; *"First, it grows exponentially with L. This is not a surprise for a worst-case analysis, which ours is"*; *"the median error is orders of magnitude lower than the mean error; this suggests that large relative round-off errors are rather rare."* Validated on **random-weight** single-head transformers (*"d,n,D=20 and L=40 … 5000 random initializations"*). | the **worst-case ceiling**; its random-weight regime is the **C2 reproduction target** (§4). |
| **Baek — "Numerical Fragility in Transformers," arXiv 2510.21770** (17 Oct 2025) | ✅ verbatim (abstract). Attention forward-error factors into κ_score · κ_softmax · κ(V); a **residual-relaxation/attenuation** inequality (factor (1+ρ)/(1−ρ) under small-gain ‖J_f‖₂<1); a LayerNorm-ε indicator. Validated **Tiny-ViT/CIFAR-10 only** — explicitly *not* large LMs. | the **mechanistic backbone** for the κ_softmax attribution arm (§3, §5, F2). |
| **Budzinskiy +7 — "LAMP: Look-Ahead Mixed-Precision Inference of LLMs," arXiv 2601.21623** (29 Jan 2026; v2 7 May 2026) | ✅ verbatim. Local f(g(x)) composition; recomputes amplified ops higher; **FP32 reference** (*"Our reference model uses FP32 inference uniformly for all FP operations"*); GPT-2; **explicitly declines the global bound** (*"the global rounding error appears to be difficult to tame, the local rounding error at each composition is easier to control—the core idea of our work"*). **Operating points (corrected to source, v2):** *"12×, 83×, and 385× reductions in KL divergence at recomputation rates of only 0.3%, 1.6%, and 7.6%."* | the **method we upgrade** (certified reference, not FP32) and the **baseline to beat** (§8). |

**Two adjacent papers, source-verified 2026-06-16:**

- **Must-distinguish (the Ersoy–Wiesner of this arc): Or Zamir — "A Note on Non-Composability of Layerwise Approximate Verification for Neural Inference," arXiv 2602.15756** (17 Feb 2026). ✅ verified. Proves a worst-case *impossibility*: *"for any neural network, we can construct a functionally equivalent network for which adversarially chosen approximation-magnitude errors in individual layer computations suffice to steer the final output arbitrarily (within a prescribed bounded range)."* **Distinction (load-bearing, stated before H1 freezes):** that result is an *adversarial functionally-equivalent construction* (a pathological function-preserving twin — note the resonance with this program's permutation-equivalence instincts); **this** registration is a *typical-case* claim on **actual trained** GPT-2/Pythia weights. Their theorem bounds what *can* be built adversarially; H1 measures what *trained weights actually do*. The two do not collide — but the document records the boundary so no reviewer can collapse it to "already known non-composable."
- **Corroborating: Arbuzov, Bei, Dong, Kalaev, Shvets — "Beyond Exponential Decay: Rethinking Error Accumulation in Large Language Models," arXiv 2505.24187** (30 May 2025; v2 May 2026). ✅ verified. LLM errors *"are concentrated at sparse 'key tokens' (5–10% of total tokens) representing critical decision junctions,"* not uniformly. Independent support (in an autoregressive, not round-off, guise) for the **headline framing of §5**: the signal is *which tokens/heads fire*, not the depth-law magnitude.

### 0.2 The gap, stated as exactly what it is

Worst-case depth-N theory exists (Budzinskiy, random-weight), ViT-only layer theory exists (Baek), real-LLM **local**-error control exists (LAMP, FP32-referenced, global bound declined). **No published work has produced a *certified* (not FP32-agreement) characterization of *typical-case* error amplification through the full depth of a real decoder-only LLM, nor a precision allocator built on a certified rather than FP32 reference.** Median ≪ mean (Budzinskiy) means the worst-case is loose, which is *why* the typical-case + which-inputs-fire question is open and worth a falsifiable instrument.

### 0.3 What this deliberately does NOT use (anti-scope-creep)

Assessed redundant or cooling for this target this session and **excluded**: sparse autoencoders / dictionary learning; general activation steering (AxBench: cheap baselines win); general Fisher/Hessian training-dynamics geometry; TDA / persistent homology on internals; Ruppeiner-curvature diagnostics. This is a **numerical-analysis + certification** program. The only "geometry" admitted is condition numbers (Baek). Smuggling the Paper-3 curvature toolkit back in is a registration violation.

---

## 1. The measurable

- **Pre-LN block error recursion.** For a residual block `x_{l+1} = x_l + F_l(x_l)` (GPT-2: `F_l = MLP∘LN2∘(x'+·)` after `x' = x_l + Attn∘LN1`), the first-order round-off recursion is `ε_{l+1} = (I + J_{F_l})·ε_l + δ_l`, with `δ_l` the locally-injected round-off of block `l`'s low-precision ops and `J_{F_l}` the block Jacobian. Baek's small-gain claim is that `‖J_{F_l}‖₂ < 1` makes the residual structure **attenuate** depthwise accumulation (factor (1+ρ)/(1−ρ)); the worst-case (Budzinskiy) is that it **compounds** exponentially. **Which one governs *trained* weights on *typical* inputs is the empirical object.**
- **Per-block condition numbers (Baek, instantiated for GPT-2-small):** κ_score = ‖Q‖‖K‖/(‖S‖√d), κ_softmax = max row-wise softmax-Jacobian sensitivity (J = diag(ψ)−ψψᵀ), κ(V) = σ_max(V)/σ_min(V), LayerNorm κ < 3‖x‖∞/‖c(x)‖, RMSNorm κ ≤ 2.
- **Certified relative output error** `E_cert(L)`: relative error of the low-precision (bf16/fp8/**fp4**) forward pass vs an exact reference, sampled over a frozen set of (token, position, layer) triples.
- **The allocation oracle** `A_cert`: per-op precision assignment that keeps `E_cert` under a budget at minimum recompute cost, where the downstream-amplification factor driving each flag is computed against a **certified** reference (the instrument; §8).

**Reference scheme (computationally honest — the key design choice):** full-network mpmath at dps≥50 is **intractable** for a 124M/12-layer/d=768 model and is explicitly out of scope. Instead: **float64 is the working reference** for the depth curves; **mpmath dps≥50 is the *spot-certifier of float64's adequacy*** at the frozen sampled positions (float64-vs-mpmath within the float64 noise floor → float64 licensed as the reference *there* — the Firewall move applied correctly), **and** the exact oracle for the **local look-ahead amplification factors** that drive allocation flags (small, tractable in mpmath, exactly the f(g(x))-local quantity LAMP estimates against FP32). mpmath certifies the *reference's right to be trusted* and the *allocation oracle*, not a full exact forward.

---

## 2. The dominant confounds — stated first, on purpose

1. **FP32-agreement masquerading as certification (the program's whole thesis).** LAMP and essentially all prior allocation work adjudicate against an FP32 model that is *itself inexact* (confirmed: LAMP's reference is uniform FP32). The differentiator of this work collapses to nothing if the certified reference is never allowed to *disagree* with FP32 in a way that changes a decision. This is not a side-risk; it is the load-bearing claim, and it has its own falsifier (F-app, §6).
2. **Median ≪ mean (Budzinskiy).** The typical-case depth curve may be *undramatic* (near-linear, "nothing fires"). A flat curve is still a `[V]` result, but the **headline is *which* inputs/heads/tokens fire, not the curve's magnitude** (corroborated by arXiv 2505.24187's key-tokens). Any framing that leads with "the depth-law is X" is mis-set.
3. **Range vs. mantissa.** If `E_cert` is dominated by a *range* artifact (LayerNorm/softmax overflow — the v0.1 finding that bf16 avoids fp16's overflow), the "composition" framing is wrong and the problem reduces to already-known range results. This must be separated, not assumed away (F3).

---

## 3. `[GATE]` — derive-before-numerics (the only item left before the freeze is complete)

Before **any** GPU scan:

1. **Re-derive and symbolically certify** the per-block first-order recursion `ε_{l+1} = (I+J_{F_l})ε_l + δ_l` for the *specific* GPT-2-small architecture, instantiating Baek's κ-factors and Budzinskiy's componentwise condition numbers. Freeze the predicted depth-law form and the κ_softmax-concentration prediction as the objects to be tested. **[DONE 2026-06-16 — `precision_recursion_gate.py`: the first-order recursion is certified (remainder O(ε²)); the softmax Jacobian `diag(ψ)−ψψᵀ` is certified vs autodiff (err 2.8e-17, spectral ≤ max ψ); and the prediction is SHARPENED — `∏‖I+J_F‖` is exponential at any scale, so P1/F1 are reframed to the measured typical-case `E_cert(L)`. The remaining per-op κ-factor instrumentation (κ_score, κ(V), LN) is definitional and folded into the Stage-2 depth-map build where it is used. §3.1 satisfied for Stage-1 admissibility.]**
2. **Body-claims verified (done, at source 2026-06-16):** Budzinskiy exponential-in-L + median≪mean — confirmed verbatim. LAMP declines the global bound + uses an FP32 reference — confirmed verbatim. LAMP operating points corrected to source (0.3% / 1.6% / 7.6% @ 12× / 83× / 385×). *These come out of the gate; they are now part of the record (§0.1), not assumptions.*
3. **Non-composability distinction (done, §0.1):** Zamir, arXiv 2602.15756 cited and distinguished (adversarial functionally-equivalent vs typical-case-trained). H1 may freeze **once §3.1 is done.**
4. **Architecture-variation note (accurate, kept):** GPT-2 = *sequential* pre-LN (Attn then MLP); Pythia/GPT-NeoX = *parallel* pre-LN (Attn and MLP on the same LN input, summed). The two give different `J_{F_l}` structure → a clean cross-check on whether any attenuation law is architecture-robust, not an artifact of sequential residual.

---

## 4. Controls — control-before-scan

- **C1 (harness identity).** float64-vs-float64 forward → `E_cert = 0` to machine precision. Validates the pipeline.
- **C2 (reproduce the published worst-case — HARD GATE).** On **random-weight** transformers in Budzinskiy's regime (d=n=D=20, L=40, scaled toward GPT-2-small shape), `E_cert(L)` must reproduce **essentially exponential mean growth with median ≪ mean**. **If C2 does not reproduce the published worst-case, the harness is wrong and no trained-weight scan runs.** Non-negotiable.
- **C3 (κ_softmax shuffle-control).** Randomize which heads/tokens are labeled high-κ_softmax; the κ–error correlation must **vanish** under shuffle. Guards the F2 attribution against a spurious fit.
- **C4 (single-primitive isolation).** Re-run the v0.1 matrix entries (GEMM/softmax/LayerNorm/attention in isolation) to confirm depth-N composition is doing something **beyond** per-op error — i.e. `E_cert(L)` is not just `L ×` a single-op error.

**Phase gate:** trained-weight scanning (§5) is admissible **iff** C1 passes, **C2 reproduces the exponential worst-case**, and C4 confirms a composition effect exists.

---

## 5. Frozen hypothesis and prediction 🔒 (locks once §3.1 done; pre-data)

**H1 (application-first).** *A precision allocator built on a **certified** reference (`A_cert`) can hold a fixed output-fidelity budget (KL / flip-rate vs the certified reference) at **lower recompute cost** than uniform precision **and** than LAMP's FP32-referenced allocator at matched budget, on real Blackwell FP8/FP4 — and the recompute it spends concentrates on a **sparse, κ_softmax-identified** set of ops/tokens.* The depth-law measurement (§1) is the **instrument that makes `A_cert` trustworthy**, not the deliverable.

**Frozen choices.**
- **Models:** GPT-2-small (124M, 12L, d=768) primary; GPT-2-medium (355M) + Pythia-160M for the depth/width + sequential/parallel-residual cross-check; OpenAI weight-sparse models (arXiv 2511.13653) as a high-interpretability cross-check arm only.
- **Precisions:** bf16, FP8 (E4M3), **FP4** (Blackwell MX) — block-internal ops low, accumulation/reference per §1.
- **Reference:** float64 working reference, mpmath dps≥50 spot-certifier + local allocation-oracle (§1).
- **Inputs:** frozen held-out slice of OpenWebText/FineWeb (typical-case arm); a separately-labeled sharp-logit / attention-sink arm (adversarial), **never mixed into the typical-case estimate**.
- **Sampling (frozen, since mpmath is CPU-bound):** a pre-registered subsample of (token, position, layer) triples; sampling scheme is part of the freeze, not a post-hoc choice.
- **Metrics:** KL divergence + top-1 flip rate vs the certified reference; recompute rate; benchmarked against LAMP's reported operating points (**source values: ≈0.3% / 1.6% / 7.6% recompute, yielding 12× / 83× / 385× KL reduction**).

**Frozen predictions (the verdict predicates).**
- **P1 (typical-case depth-law — sharpened by the §3.1 gate):** the **measured** typical-case `E_cert(L)` on trained weights + typical inputs grows **sub-exponentially (near-linear)** via the median≪mean / benign-direction mechanism — *not* via a small-gain bound (the worst-case product `∏‖I+J_F‖` is exponential at **any** weight scale, certified in `precision_recursion_gate.py`) — *and* the deviations toward compounding concentrate on high-κ_softmax (sharp-logit, near-saturation) tokens/heads.
- **P2 (the allocator wins):** `A_cert` meets a fixed KL/flip budget at recompute cost **≤** uniform and **≤** LAMP-at-matched-budget.
- **P3 (where the certification earns its worth):** the certified-vs-FP32 allocation **differs at FP4** (no clean FP4 oracle → FP32 is not a trustworthy reference there) and **agrees at FP8** (FP32 adjudicates FP8 fine). **FP4-on-Blackwell is therefore the build target; FP8 is the redundancy control.**

---

## 6. Falsifiers — each can actually fire

- **F1 (kills P1 / H1's premise):** the **measured typical-case** `E_cert(L)` grows **exponentially in L on trained weights** (log-error-vs-L slope significantly positive at the frozen threshold) → the benign-direction / median≪mean attenuation does **not** save typical inputs on real LLMs. *(Fires on the measured typical-case curve, NOT on any spectral norm exceeding 1 — the §3.1 gate certified the worst-case compounds at every scale regardless.) A genuinely important negative result, logged not buried.*
- **F2 (kills the mechanistic attribution):** high-κ_softmax tokens do **not** carry disproportionate error — the C3 shuffle-control correlation is indistinguishable from the real assignment. The depth-law `[V]` survives; the κ_softmax story does not.
- **F3 (kills the "composition" framing):** `E_cert` is dominated by a **range** artifact (LN/softmax overflow), not mantissa composition → problem reduces to the known v0.1 range findings; **pivot** to characterizing the range-vs-mantissa boundary as the deliverable (still novel, still certified).
- **F-app (kills the certification's *operational* worth — the application's own falsifier):** the certified-reference allocation produces the **identical** op-set at matched budget as the FP32-reference allocation → certification adds nothing operational and only the science remains. **Predicted to fire at FP8 (expected, that's the control) and *not* fire at FP4 (that's the claim).** If F-app fires at FP4 too, the instrument has no edge over LAMP and the work collapses to the depth-law science.

---

## 7. Outcomes and what each licenses (pre-registered)

| result | licenses |
|---|---|
| C1 pass + C2 reproduces exponential + C4 composition effect | harness `[V]`; trained-weight scan admissible |
| `E_cert(L)` curve on trained GPT-2/Pythia, float64-referenced + mpmath-spot-certified | **`[V]`** — first certified typical-case depth-error map for a real decoder-only LLM |
| P1 holds (sub-exponential + κ_softmax concentration), F1/F2 don't fire | **`[E]`** — the typical-case attenuation law and its mechanistic driver |
| P2 + P3 hold (allocator beats uniform & LAMP at FP4; F-app fires only at FP8) | **`[C]→[E]`** — a certified FP4/FP8 precision allocator with a real, falsified edge over the FP32-referenced state of the art, running on owned hardware |
| F1 fires | **`[V]` negative result** — residual attenuation does not transfer to real LLMs (Baek's ViT theory bounded) |
| F-app fires at FP4 | instrument has no operational edge; retreat to the depth-law science only (still `[V]`) |

---

## 8. The build — application terminus

The shippable artifact is `precision_allocator.py`: **LAMP's mechanism with a certified reference.** It takes (model, precision palette {bf16,FP8,FP4}, fidelity budget) and emits a per-op precision assignment whose downstream-amplification flags are computed against the mpmath-certified local oracle (§1) rather than FP32. `allocator_bakeoff.py` benchmarks it against (a) uniform precision and (b) a faithful LAMP re-implementation at matched recompute budget (anchored to the source operating points 0.3% / 1.6% / 7.6%), on the RTX 5070's native FP8/FP4 tensor cores, on held-out text. It plugs into a real inference path (the program's deployed inference surfaces) as a numerical-fragility flag-and-recompute pass. **The worth is concentrated at FP4** (P3): the regime where no clean oracle exists, where FP32-agreement is least trustworthy, where the literature is thinnest, and where the hardware runs natively.

---

## 9. Scope walls (hard)

- Inference-only forward pass; **no training, no backprop error** (different problem — see §11).
- ≤ 355M params; **explicitly not claiming transfer to 70B** — the depth-law may change with scale; that is a future `[GATE]`, not an extrapolation taken here.
- Single-GPU, single-node; no distributed numerics.
- Typical-case estimate uses only the frozen natural-text slice; the adversarial sharp-logit arm is separately labeled and never pooled in.
- mpmath certification is **sampled** (frozen scheme); a full exact forward is out of scope and not claimed.
- FP4 tooling (TensorRT-LLM / llama.cpp FP4 paths) is immature → FP4 results are certified **especially** carefully; an FP4 number is `[E-hw]` until the mpmath oracle backs the allocation decision.
- Not a TDA / Ruppeiner / SAE / steering application (§0.3).

---

## 10. Execution order and thresholds

- **Stage 0 (now):** this freeze. `[GATE]` §3.1 derivation written and symbolically certified → `precision_recursion_gate.py`. *The only remaining pre-scan item.*
- **Stage 1 (mo. 1–2) — controls:** build the mpmath-certified harness; pass C1; **reproduce Budzinskiy's exponential on random weights (C2 HARD GATE)**; C4. *Do not scan trained weights until C1/C2/C4 pass.*
- **Stage 2 (mo. 2–4) — certified depth map on trained weights:** `precision_depth_map.py`. Test P1; run C3 + F1/F2/F3. *If F3 fires, pivot to the range-vs-mantissa boundary as the deliverable.*
- **Stage 3 (mo. 4–6) — the allocator ([C]→[E]):** only if P1 holds and F2/F3 don't fire. `precision_allocator.py` + `allocator_bakeoff.py`; test P2/P3/F-app against uniform and LAMP at matched budget. *Promotion to `[E]` requires a measured KL/flip improvement at fixed recompute on held-out data; F-app must fire only at FP8, not FP4.*

---

## 11. Parked / future gates (named, not chased)

- **`[GATE]` loss-spike bridge.** Budzinskiy's footnote: across many training forward passes, "individually rare" large round-off errors become "relatively frequent" and may drive **loss spikes**. A certified per-step numerical-fragility signal as a training-instability early-warning is the natural training-side extension — and the one place the program's Paper-3 transition-detection interest could legitimately re-enter. Derive before any training run; not in this registration's inference-only scope.
- **Scale `[GATE]`:** whether the typical-case depth-law holds past 355M. Requires hardware this program does not have; named, not attempted.

---

## 12. File index

**To be written (no `[V]` pre-empted):** `precision_recursion_gate.py` (§3 derivation + symbolic cert), `precision_depth_map.py` (§5 certified depth map + C1–C4, F1–F3), `precision_allocator.py` (§8 the instrument), `allocator_bakeoff.py` (§8 vs uniform + LAMP).
**Antecedent (`[V]`):** `T1_precision_map_v0_1.md`, `ig_primon/precision.py`, `ig_primon/torch_precision.py`, `ig_primon/firewall.py`, `module_hw_firewall.py` (the Precision–Certification Firewall this allocator's certified reference is built on).

---

## Changelog

**v0.1 → v0.2 (2026-06-16).** Created. Reframed application-first per the dual maxim (§0): the certified allocator is the hypothesis (§5/§8), the depth-law measurement is the instrument, not the deliverable (v0.1's inversion corrected). Folded in the source-verifications of the three prior-art anchors (Budzinskiy 2503.10251, Baek 2510.21770, LAMP 2601.21623) and the two adjacent papers (Zamir 2602.15756, Arbuzov et al. 2505.24187) — **all five confirmed at the primary arXiv source on 2026-06-16.** Registered the application falsifier F-app and the FP8-redundant / FP4-load-bearing split (P3). Excluded TDA/Ruppeiner/SAE/steering by scope (§0.3). Reference scheme fixed to float64-working + mpmath-spot-certifier + local-oracle (§1) for computational honesty.

**Correction (no silent edit):** an earlier synthesis draft asserted the three body-claims (Budzinskiy exponential-in-L + median≪mean; LAMP declines-global + FP32-reference) as "confirmed verbatim" **before** they were read; they were subsequently confirmed verbatim **at source on 2026-06-16** and now stand as record. The draft's LAMP operating points (≈0.9% / 3.4% / 15% / 34.3% recompute, 10–1000× KL) were **wrong**; the source (LAMP v2) reports **12× / 83× / 385× at 0.3% / 1.6% / 7.6%**, corrected in §0.1, §5, §8. The draft's truncated titles for 2602.15756 and 2505.24187 were completed with verified authors (Zamir; Arbuzov, Bei, Dong, Kalaev, Shvets).

**Amendment v0.2.1 (2026-06-16 — §3.1 gate done; versioned, no silent edit).** `precision_recursion_gate.py` written and run: (1) the first-order block recursion `ε_{l+1}=(I+J_F)ε_l+δ_l` certified (O(ε²) remainder); (2) the softmax Jacobian `diag(ψ)−ψψᵀ` certified vs autodiff (err 2.8e-17, spectral radius ≤ max ψ). It **caught and corrected an imprecision in P1/F1**: the worst-case spectral product `∏‖I+J_F‖` is exponential at *every* weight scale (random-weight `‖I+J_F‖>1` always), so attenuation is **not** a small-gain / `‖J_F‖<1` consequence — it is the *typical-case* median≪mean effect (demonstrated: typical random-direction amplification runs 16–730× below the worst-case product; near-flat 1.27× over 8 blocks at small scale). P1/F1 reframed to the **measured typical-case `E_cert(L)`**, not a spectral threshold (§1 intent, §5-P1, §6-F1). Had the spectral-norm framing frozen, F1 would have spuriously fired on every input — the gate caught a false-negative before any scan.

**Status:** with §3.1 satisfied, the registration is **FROZEN and Stage-1-admissible** (harness controls C1/C2/C4 may begin). The per-op κ-factor instrumentation (κ_score, κ(V), LN) is definitional and built into Stage 2 where it is used.

**Amendment v0.2.2 (2026-06-16 — Stages 1–3 executed in sim; versioned, no silent edit; receipts banked).** The full pipeline ran on trained GPT-2-small (124M), CPU, float64-referenced fake-quant. Outcomes against the frozen predicates:

- **Stage 1 controls + Stage 0 gate — PASS `[V]`.** `precision_recursion_gate.py` (first-order recursion + softmax Jacobian), `precision_depth_map.py` (C1 identity, C2 HARD GATE reproduces Budzinskiy exponential mean / median≪mean on random weights, C4 composition), and the Stage-3 quantizer gate (`precision_allocator.py`: FP8 E4M3 / FP4 E2M1 grids certified to spec, round-to-nearest, tier ordering). One control mis-specification was caught and corrected in the open (per-element relative half-ulp is the wrong invariant for absmax-scaled float quant → corrected to exact-nearest + absolute half-ulp).
- **P1 — HOLDS `[V/E-hw]`** (`results_precision_gpt2_scan.txt`). Measured typical-case `E_cert(L)` is **sub-exponential**: float32 log-median slope **+0.005/layer** (growth 0.69×), bf16 **+0.036/layer** (growth 0.96×), vs the Stage-1 random-weight worst-case **+0.285/layer**. Sub-exponential and roughly flat — *not* a strong contraction.
- **P2 — HOLDS `[E-sim]`** (`_bakeoff_stdout.txt`). The sensitivity-ordered (certified/exact) allocation beats uniform and a linear (LAMP-style) scorer at matched budget, both bases, with a documented greedy caveat (leave-one-out marginals do not compose; the certified pick loses at one small FP4 budget). Caveat noted: exact > linear is partly expected.
- **P3 / F-app — the registered test was initially MIS-BUILT, then run correctly; P3 NOT supported in sim.** The first bake-off (and an independent alt-environment run) compared exact-vs-**linear scoring** against a float64 reference — orthogonal to the registered predicate, which (§5/§6) is a **reference swap** (certified float64 vs FP32, scoring held fixed). The earlier "F-app fires only at FP8 → edge at FP4" reading was **withdrawn**. The literal F-app (`fapp_reference_swap.py`, `results_fapp_reference_swap.txt`) holds the exact importance fixed and swaps only the reference: the fp32-vs-float64 full-model logit gap is **2.4e-7**, importance rank corr **+1.0000**, op-sets **identical at every budget** → **F-app FIRES at both FP8 and FP4.** Per the §7 pre-registered branch ("F-app fires at FP4 → no operational edge; retreat to the depth-law science"), the certified-reference *operational* edge is a **hardware/scale (FP32-accumulation) claim this sim cannot establish.** The depth-law science (P1 + controls) is the standing `[V]` deliverable.
- **Structural finding `[V-sim]` + operational config** (`results_pillars_fp4.txt`, `results_operational_recipe.txt`). Weight-quant sensitivity is **sparse** (965× spread max/min) and **stable out-of-sample** (Spearman +0.972, 83% top-12 overlap across disjoint corpora); the load-bearing pillars are **early-layer MLP** weights (h0.mlp.c_proj dominates, ~10/12 of the top pillars are MLP), the FP4-safe tensors are **mid-network attention projections**. Derived-on-A / validated-on-B config: **FP8 weight-only is the viable regime** (~8 bits/wt, held-out KL 4.4e-3, 5.2% flip; +bf16 pillars → 13.1 bits, KL 2.9e-3, 1.0% flip); **weight-only FP4 is not viable at 124M** even with pillar protection (best 3-tier 7.9 bits → KL 0.195, 34% flip). This is a small-model result (pre-reg scope ≤355M; FP4 weight-only is known to work on larger, more-redundant models — not extrapolated here). The recipe is **not certification-exclusive** (an FP32 reference yields the same ranking).

Net: the **science arm is `[V]`** (certified typical-case depth law + sparse, stable sensitivity structure); the **application arm's certified edge is unproven in sim and correctly relocated to hardware** — the honest, pre-registered outcome, not a clean confirmation.

**Amendment v0.2.3 (2026-06-16 — first hardware contact: RTX 5070 sm_120 Blackwell; versioned, no silent edit; receipts banked).** The §7-relocated "hardware claim" was taken to the actual silicon. Native FP8 e4m3 GEMM runs via `torch._scaled_mm` (FP4 `float4_e2m1fn_x2` dtype present but the GEMM path / rowwise scaling are unsupported in this build — the FP4 frontier). Findings, each `[V-hw]`, controlled:

- **Sim-vs-silicon bridge (`results_gpu_fp8_bridge.txt`).** The weight-only sim was **faithful on its object** — the weight-FP8 marginal on metal (R1−R0 = 5.66e-3) ≈ the CPU sim's weight-only FP8 (5.17e-3, ~1.1×). **But that object is sub-dominant.** The bf16 **activation floor** (R0 = 1.50e-2) is ~**877×** the sim's bf16-weights-only (1.7e-5): the sim held activations at float64 and never saw the activation/residual-stream precision cost, which **dominates** real error. The whole weight-allocation arc (Stage 3) optimized a ~5e-3 term while a ~1.5e-2 term — untouchable by *weight* allocation — sat next to it. Full W8A8 = 2.82e-2.
- **F-app / P3 in the regime where it can bite (`results_gpu_fapp_hardware.txt` + `..._control.txt`).** The CPU F-app fired null because the reference was fp32 ≈ float64 to 2.4e-7. The **realistic on-device reference is a bf16 forward** carrying 1.5e-2 error. Scored against it, importance ranks **diverge** from the float64-certified ranking (Spearman +0.39, top-k overlap 50–71%) — i.e. **F-app does NOT fire** on hardware. **Controlled:** the 5070's bf16 forward is bit-deterministic (run-twice KL = 0.0) and the bf16-referenced ranking is 100% reproducible (self-Spearman +1.0) → the divergence is **real signal, not GPU noise.**
- **P3 operational edge, leakage-controlled (`..._p2.txt`, `..._p2_heldout.txt`).** The certified (float64-referenced) allocation achieves **lower true KL** than the cheap-bf16-referenced allocation at matched budget — 5/5 in-sample **and** 5/5 on **disjoint held-out** text (**+6.1%** mean KL; not leakage). **First positive evidence for the program's core application thesis.**
- **Robustness to proper quantization (`results_gpu_perchannel.txt`) — the honest deflation.** Per-token-act + per-channel-weight FP8 (fake-quant; rowwise kernel unsupported on sm_120) cut W8A8 error only **1.10×** — finer scales don't fix transformer **outlier *channels*** (that needs SmoothQuant-style migration). Under per-channel, the certification edge **erodes to +4.0% / 3-of-5 budgets** (2 budgets the cheap reference wins). So the edge is **real but small and regime-sensitive: largest when quantization is crudest, fading toward noise as it improves.**

**Net (P3 resolved, honestly):** the certification thesis is **neither the sim's null nor a robust large edge** — it is a **genuine, modest (~4–6%), fragile** operational effect, hardware-confirmed in the crude-quantization / inexact-on-device-reference regime and eroding with care. `[V-hw]`, scoped to 124M, weight-vs-bf16 unit allocation, no throughput/timing claim. **Open next gate (named, not chased here):** SmoothQuant-style activation migration to actually crush the 1.5e-2 activation floor, and the question it poses — does the certified edge vanish into the lowered baseline, or become the only differentiator left once activations are handled?

— End of `T1_precision_map` v0.2. Amendments require a versioned diff; silent edits void the registration.
