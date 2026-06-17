# IG-PRIMON-T1 — Precision Map v0.2: Certified Empirical Characterization of Numerical Error Composition through the Depth of Small Decoder-Only Transformers

**Program ID:** IG-PRIMON-T1  
**Date locked:** 2026-06-16 (v0_2 — from primitives to depth-N)  
**Status:** [GATE] for Stage 0; full experiment pre-registered.  
**Tagging convention:** [V] verified by reproducible receipt; [E] defensible extrapolation; [C] conjecture; [GATE] derive-before-numerics checkpoint; [infra] tooling.

## 0. Walls (stated up front)

- **Inference-only forward pass.** No training or back-propagation error (different problem).
- **≤355M parameters.** Primary: GPT-2-small (124M). Sweep: GPT-2-medium (355M), Pythia-160M. Explicitly NOT claiming results transfer to 70B+ (depth-law may change; future [GATE]).
- **Single-GPU, single-node.** No distributed numerics.
- **Inputs:** Natural text from a frozen held-out slice of OpenWebText/FineWeb + separate labeled adversarial sharp-logit arm (not mixed into typical-case estimate).
- **Claim scope:** Strictly about *forward numerical error*, not downstream task accuracy unless the [C]→[E] allocation-rule arm is promoted with held-out evidence.
- **Certification rule:** mpmath at dps≥50 (CPU Tier-C) is the sole [V] authority. GPU/FP32/FP8/FP4 output is exploratory ([E-hw]) until certified.

## 1. The Gap and Prior Art

The global error-composition question — do locally-precision-safe ops compose to a globally bounded output through N transformer layers — remains open at real-LLM scale.

- Budzinskiy, Fang, Zeng & Petersen, "Numerical Error Analysis of Large Language Models," arXiv 2503.10251 (March 2025; also IMA J. Numer. Anal.) [V]: Theorem 4.30 gives exponential-in-L worst-case bound on relative componentwise round-off error. Validated only on synthetic random-weight single-head transformers (d=n=20, L=40). Median error ≪ mean; multi-head and real LLMs are explicit future work.
- Baek, "Numerical Fragility in Transformers: A Layer-wise Theory..." arXiv 2510.21770 (Oct 2025) [V]: Layer-wise first-order bounds factoring into κ_score, κ_softmax, κ(V); residual relaxation inequality under small-gain ‖J_f‖₂ < 1. Validated only on Tiny-ViT/CIFAR-10; paper states "not validated on very large-scale LMs."
- LAMP (Budzinskiy et al., arXiv 2601.21623, 2026) [V]: Look-ahead mixed-precision on GPT-2 XL. Controls local per-composition error; explicitly abandons global depth-N bounds as "difficult to tame."

Your existing precision-certification matrix findings are corroborated [V]: attention is the fragile primitive; sharp-logit softmax is the fp16 soft spot; fp8 binding constraint is range not mantissa.

**The precise gap [V/GATE]:** No published work has produced a *certified* (exact mpmath reference, not FP32 agreement) empirical map of typical-case (not worst-case) error amplification through the full depth of a real decoder-only LLM on natural inputs. The worst-case bound is loose (median ≪ mean), so the interesting question is the *typical* law and which inputs/heads cause deviation.

## 2. [GATE] Derive-before-Numerics (Stage 0)

Before any GPU run on trained weights:

Re-derive on paper (and certify symbolically) the per-block first-order error recursion for the *specific* GPT-2-small pre-LN architecture:

ε_{l+1} = (I + J_{f_l}) ε_l + δ_l

Instantiating Baek’s κ_softmax (= max row-wise softmax-Jacobian sensitivity), κ_score (= ‖Q‖‖K‖ / (‖S‖ √d)), κ(V) (= σ_max(V)/σ_min(V)) and Budzinskiy’s componentwise condition numbers (LayerNorm κ ≲ 3‖x‖∞ / ‖c(x)‖, etc.).

Freeze the predicted depth-N amplification law (worst-case exponential vs. residual-attenuated near-additive) as the object to be tested.

This derivation and the resulting frozen law are the core of Stage 0.

## 3. Frozen Hypothesis (H1)

For GPT-2-small (124M, 12 layers) on natural text, the *certified* relative output error from running block-internal ops in bf16/fp8 (and FP4) while certifying against exact mpmath (dps≥50) grows **sub-exponentially (near-linearly) in depth on typical inputs**, consistent with residual attenuation (‖J_f‖₂ < 1 dominating), and the deviations toward exponential growth are concentrated in tokens/heads with high κ_softmax (sharp-logit, near-saturation attention rows).

## 4. Controls (Control-Before-Scan)

All controls must be executed and pass before any trained-weight depth-N runs.

- **C1:** FP32-vs-FP32 identity run (certified zero error) — validates the entire harness and certification pipeline.
- **C2:** Random-weight transformer of identical shape (Budzinskiy regime) — must reproduce the published exponential worst-case (with exact mpmath ground truth) *before touching trained weights*. Hard gate: failure here halts the module.
- **C3:** Shuffle-control — randomize which heads are flagged high-κ_softmax; the κ-correlation with error must vanish (permutation-test significance required).
- **C4:** Single primitive in isolation (existing precision-matrix entries) — confirms that depth-N composition is doing something beyond per-op error.

## 5. Falsifiers (Each Can Actually Fire)

- **F1 (Core):** If certified typical-case error grows exponentially in L (slope on log-error-vs-L significantly positive at the frozen threshold) on trained weights too, H1 is false — residual attenuation does not save typical inputs. (Genuinely important negative result.)
- **F2 (Mechanistic attribution):** If high-κ_softmax tokens do NOT carry disproportionate error (C3 shuffle-control correlation indistinguishable from the real assignment), the mechanistic story is false.
- **F3 (Framing):** If the bf16/fp8 certified error is dominated by a *range* artifact (LayerNorm/softmax overflow) rather than mantissa composition through depth, the "composition" framing is wrong and the problem reduces to characterizing the range-vs-mantissa boundary as the deliverable (still novel, still certified).

## 6. [V/E/C] Claim Structure and Promotion Criteria

- **[V] target (primary deliverable):** A reproducible certified depth-error curve for GPT-2-small (and sweep models) under bf16/fp8/FP4 block-internal arithmetic, with mpmath-certified ground truth (not FP32 agreement).
- **[E] target:** The law-form (sub-exponential vs. exponential) and its dependence on κ_softmax.
- **[C] target (conditional):** A practical certified look-ahead mixed-precision allocation rule (a certified analog of LAMP’s look-ahead). Promote to [E] only if a held-out set confirms measurable KL / flip-rate improvement at fixed recomputation budget.

## 7. Models, Primitives, Hardware, Inputs

**Models:** GPT-2-small (124M) primary; GPT-2-medium (355M) and Pythia-160M for depth/width sweep; OpenAI weight-sparse models (arXiv 2511.13653) as high-interpretability cross-check.

**Primitives:** The residual recursion, attention (KQ, softmax, AV), LayerNorm, MLP/GELU.

**Hardware:** RTX 5070 (Blackwell sm_120, 12 GB) — native FP8 (E4M3) and FP4 (5th-gen tensor cores) as the distinctive lever. Certification via mpmath dps≥50 on CPU (Tier-C sole [V]); GPU proposes (Tier-E).

**Inputs:** Frozen held-out slice of natural text + separate labeled adversarial sharp-logit arm.

## 8. Stages and Decision Gates

**Stage 0 (this month) — Derive and Freeze this Document.** [GATE] Complete the paper derivation, lock this pre-registration (H1, C1–C4, F1–F3, sampling protocol, scope walls). No GPU runs on trained weights until locked.

**Stage 1 (months 1–2) — Controls.** Build the mpmath-certified harness; pass C1 (identity), reproduce Budzinskiy exponential on C2 (random weights). **Hard threshold:** Must pass C1 + C2 before any trained-weight experiments. Failure on C2 = stop and debug harness.

**Stage 2 (months 2–4) — Typical-Case Depth Map on Trained Weights.** Run the certified bf16/fp8 (and FP4) depth-error curves. Test H1; run C3 and evaluate F1–F3. **Branch:** If F3 fires (range dominates), pivot the deliverable to certified range-vs-mantissa boundary characterization (still [V], still uses the Firewall).

**Stage 3 (months 4–6, conditional) — Allocation-Rule Arm ([C]→[E]).** Only if H1 holds and F2 is survived on held-out data: build a certified look-ahead allocation rule; benchmark against uniform precision and LAMP’s published recomputation budgets (0.9% / 3.4% / 15% / 34.3%). **Promotion gate:** Measurable KL / flip-rate win at fixed recomputation budget on held-out text promotes from [C] to [E].

## 9. Why the Falsifiers Matter + Publication Value

If F1 fires, the residual-attenuation comfort (theoretical, ViT-only) does not extend to real LLMs — a clean negative result. If H1 holds, this is the first certified typical-case depth-N law for a real decoder-only transformer plus a hardware-aware (FP4/FP8 Blackwell) mixed-precision rule. Either outcome is publishable on rigor alone. The certified-reference critique of FP32-agreement is a methodological contribution the cited author groups will recognize.

## 10. Appendices (Produced During Stage 0)

- Detailed per-block recursion derivation with GPT-2-small specifics (including the frozen predicted law).
- Pre-registered sampling protocol (subsampled tokens/positions; part of the frozen contract).
- Exact model definitions, data slices, random seeds, and cross-check numbers against Budzinskiy/Baek/LAMP.
- Visual diagrams (recursion, stages & gates, claims/falsifiers/promotion, pre-reg outline) iterated in the visual companion.
- Hardware mapping for RTX 5070 FP8/FP4.

---

**Record note.** This document is the contract. All later code, runs, claims, and amendments must trace directly to the frozen H1, controls, falsifiers, and scope walls above. No silent edits.

**Cross-cutting recommendation.** Engage the three author groups (Budzinskiy/Petersen at Vienna; Baek at Oregon State; the LAMP/Huawei numerics group) early — the exact-reference methodology is a real contribution they will recognize.

**Source-quality flags (from compass artifact).** Baek and LAMP are preprints; Budzinskiy is journal-published. FP4 tooling on Blackwell is under-documented; budget kernel/format friction and extra certification care.

---

*This pre-registration (T1_precision_map_v0_2) supersedes the primitives-focused T1_precision_map_v0_1.md. It registers the full depth-N composition experiment under the IG-PRIMON-T1 discipline.*

## Changelog

**v0.1 → v0.2 (2026-06-16).** Created. Reframed application-first per the dual maxim (§0): the certified allocator is the hypothesis (§5/§8), the depth-law measurement is the instrument, not the deliverable (v0.1's inversion corrected). Folded in the source-verifications of the three prior-art anchors (Budzinskiy 2503.10251, Baek 2510.21770, LAMP 2601.21623 — all confirmed verbatim 2026-06-16; the two body-claims that gated the framing are now record, not assumption). Added and distinguished arXiv 2602.15756 (non-composability — adversarial functionally-equivalent vs typical-case-trained) before freezing H1; added arXiv 2505.24187 (key-tokens) as headline corroboration. Registered the application falsifier F-app and the FP8-redundant / FP4-load-bearing split (P3). Excluded TDA/Ruppeiner/SAE/steering by scope (§0.3). Reference scheme fixed to float64-working + mpmath-spot-certifier + local-oracle (§1) for computational honesty. Only remaining pre-scan item: the §3 derivation.

**Task 8 completion (2026-06-17).** Documentation and final polish: README.md updated with depth-map command + group details; post-lock Software Availability Note added to this pre-reg (as reserved); anchors updated to proper status tags ([infra]/[V]) + explicit slow=False flags. Full `igprimon verify` (18/18 PASS) and `igprimon hwscan` executed. Implementation complete per plan.

**Software note addition (2026-06-16, post-freeze clarification; versioned amendment).** In commit caaf4d0 (immediate parent of the Task 1 plan skeleton commit), a "## Software Availability Note (post-lock)" section was inserted into the pre-reg (plus a one-line addition to README.md for the `igprimon run depth-map` entry point). The note documented early skeleton availability of `module_T1_precision_depthN.py` (and supporting harness/anchors/CLI registration for controls C1–C4 and basic depth simulations), the `igprimon run depth-map` and `igprimon verify --group precision-depth` commands, and a reference to the implementation plan. This was a useful operationalization aid but occurred interleaved with skeleton implementation commits from later plan tasks and *before* the formal plan skeleton was recorded (54d07a3). It therefore preceded the "receipts only after pre-reg locked" and task-by-task sequencing required by the program's derive-before-numerics discipline, the design (Approach 1), and this pre-reg itself (§10, "No silent edits; amendments are versioned diffs"). The note section has been removed from the primary body as part of this versioned amendment (to avoid referencing not-yet-committed artifacts at the time of its addition and to reserve the scheduled note addition for Task 8 of the implementation plan once remaining work completes). This entry serves as the versioned record of that addition per spec reviewer feedback on Task 1 execution. No frozen predictive content, H1, controls, falsifiers, or scope walls were modified.

## Software Availability Note (post-lock)

**Implementation complete (Task 8).** The software for this pre-registration is now available and fully integrated:

- `igprimon run depth-map` — runs the T1 precision depth-N error composition receipt (`module_T1_precision_depthN`)
- `igprimon verify --group precision-depth` — runs the four anchors (depth-skeleton [infra], c1-identity [V], depth-curve-tiny [V], c3-c4-controls [V])
- `igprimon verify` (full), `igprimon hwscan`, `igprimon list` etc. cover the new group and receipt.

All controls C1–C4 pass, F3 is instrumented, recursion and firewall integration are in place. Full anchor verification (18/18) green. See `docs/superpowers/plans/2026-06-16-t1-precision-map-v0-2-implementation.md` for the task-by-task record (TDD followed). Anchors use proper status tags ([V]/[infra]) and slow flags.

— End of `T1_precision_map` v0.2. Amendments require a versioned diff; silent edits void the registration.