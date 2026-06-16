# Design Document: Operationalizing T1_precision_map_v0_2 — Certified Depth-N Numerical Error Composition

**Date:** 2026-06-16  
**Source Idea:** Compass Artifact (full pre-registration-grade report provided by user, located at `\\?\C:\Users\Deving-1\Documents\dev\ig-primon-t1\docs\compass_artifact_wf-8d400e68-aa25-4216-af62-2ccfc3560021_text_markdown.md`)  
**Related Visual Companion Screens:** http://localhost:61470 (experiment-stages-gates.html, error-recursion-diagram.html, claims-falsifiers-promotion.html, t1-precision-map-v02-pre-reg-outline.html)

## 1. Background and Idea

The IG-PRIMON-T1 program has a named open frontier in inference numerics: whether locally precision-safe operations compose to a globally bounded output through N layers of a real decoder-only transformer. 

The compass artifact provides a rigorous, pre-registration-grade analysis of the field (mid-2026 state):
- Mechanistic interpretability for optimization has cooled on SAEs as a general tool (GDM update, AxBench).
- Quantization is mature engineering (AWQ, FP8 production default).
- Existing theory (Budzinskiy et al. arXiv 2503.10251) gives exponential worst-case bounds but only on synthetic random-weight models.
- Layer-wise theory (Baek arXiv 2510.21770) is validated only on Tiny-ViT.
- Real-LLM work (LAMP) controls local error and abandons the global bound.

The user's existing Precision-Certification Firewall (Tier-E explorer proposes, exact mpmath Tier-C certifies; "agreement is not verification") plus the program's generate-and-verify, derive-before-numerics, no-silent-edit discipline is uniquely positioned to close this gap with the first *certified* (not FP32-agreement) empirical map of typical-case error amplification on real small LLMs.

The artifact proposes a scoped, falsifiable experiment: **T1_precision_map_v0_2: from primitives to depth-N**.

It defines:
- A [GATE] derivation of the per-block error recursion before any numerics.
- Frozen H1 (sub-exponential growth on typical inputs due to residual attenuation).
- Explicit controls (C1–C4) with hard thresholds.
- Falsifiers (F1–F3) that can actually fire and change the plan.
- [V/E/C] claim structure with promotion gate for the conditional allocation rule.
- 4-stage execution plan with clear decision points.
- Strict scope walls (≤355M params, inference-only, RTX 5070 / Blackwell FP8+FP4, natural + sharp-logit inputs).

This design turns the report into executed, disciplined work inside the existing IG-PRIMON-T1 program and ig_primon operational layer.

## 2. Approaches Considered

**Approach 1 — Strict Fidelity / Pre-Registration Lock First (Recommended and Approved)**  
Produce and lock the frozen `T1_precision_map_v0_2.md` (Stage 0) first, using the visual companion to iterate the mathematical objects (recursion, stages, claims/falsifiers). Only after user approval of the locked pre-reg do we build the harness, receipts, anchors, and run the controls. This preserves the program's DNA and the artifact's own "derive-before-numerics" rule. Lowest risk of scope creep or honesty violations.

**Approach 2 — Dual-Track (Derivation + Minimal Harness Skeleton in Parallel)**  
Run Stage 0 derivation while simultaneously prototyping the error accumulator and certification integration in code. Higher speed but risk of "frozen" elements being influenced by early code.

**Approach 3 — Receipt-First Executable Core**  
Build runnable artifacts that can pass C1/C2 first, then write the pre-reg around the working code. Fastest empirical feedback but inverts the declared order of the program and the artifact.

**Decision:** Approach 1 approved by user. All subsequent design follows this.

## 3. Approved High-Level Execution Model (Design Section 1)

### Overall Goal
Execute the exact 4-stage plan in the compass artifact as a new experiment track *inside* IG-PRIMON-T1, producing a locked pre-registration document as the immediate deliverable, followed by clean extensions to the operational layer.

### Primary Short-Term Deliverable
Frozen `T1_precision_map_v0_2.md` (placed in project root alongside other T1_*.md files). This becomes the contract for all later work.

### Integration with Existing Artifacts
- New top-level receipt(s) appear only *after* the pre-reg is locked (following the "untouched receipts + wrapper" pattern).
- The existing `ig_primon.firewall`, `hardware`, `precision` (matrix + torch bf16/fp8), anchors, and harness become direct dependencies.
- New anchors will pin certified depth-error statistics, C2 reproduction of the known exponential, and falsifier outcomes.
- The prior `T1_precision_map_v0_1.md` is the direct predecessor; v0_2 is the depth-N composition extension.
- All program hygiene (honest tags, no silent edits, Tier-C as sole [V] authority) applies unchanged.

### Use of Visual Companion
The companion (http://localhost:61470) is the primary tool for iterating the mathematical and planning objects during Stage 0:
- experiment-stages-gates.html
- error-recursion-diagram.html
- claims-falsifiers-promotion.html
- t1-precision-map-v02-pre-reg-outline.html (and any additional screens requested during derivation)

These become figures/appendices in the frozen .md. User reviews in browser; we revise content files until they match the locked text.

### Success Criteria for the Execution Model
- Stays strictly inside the artifact's scope walls.
- Stage 0 produces a locked pre-reg before any material implementation or large runs on trained weights.
- The Firewall's exact-reference critique is carried through as a methodological contribution.
- Later stages extend the operational layer cleanly.

## 4. Detailed Structure for the Frozen Pre-Registration Document (Design Section 2)

The document to be produced and locked is `T1_precision_map_v0_2.md`.

Its structure (approved via the outline visual and Section 2) directly follows the compass artifact + IG-PRIMON-T1 conventions:

- Header & Conventions (Program ID, title, date locked, versioning, tagging)
- 0. Walls (hard, up front — verbatim from artifact)
- 1. The Gap & Prior Art (with honest [V] tags)
- 2. [GATE] Derive-before-numerics (commitment to paper derivation of the recursion for GPT-2-small pre-LN)
- 3. Frozen Hypothesis (H1 — verbatim)
- 4. Controls (C1–C4 with hard C1+C2 threshold before trained weights)
- 5. Falsifiers (F1–F3 with decision branches)
- 6. [V/E/C] Claim Structure & Promotion Criteria (primary [V] depth-error curves with exact ground truth; [E] law-form; [C] allocation rule with explicit promotion gate)
- 7. Models, Primitives, Hardware, Inputs (GPT-2-small primary + sweeps; RTX 5070 FP8/FP4; mpmath dps≥50 Tier-C)
- 8. Stages & Decision Gates (summary of the 4 stages)
- 9. Why the Falsifiers Matter + Publication Value
- 10. Appendices (detailed recursion derivation, pre-registered sampling protocol, model/data/seeds, the visual diagrams, cross-check numbers)

**Process to Produce It (Stage 0):**
- Iterative derivation (paper/symbolic) + visual iteration in the companion.
- The outline visual (t1-precision-map-v02-pre-reg-outline.html) serves as the working skeleton.
- User reviews text + visuals until consistent.
- Lock the .md (no further edits except via new amendment).

**Visuals Supporting This Section:**
All four screens listed above, plus any requested during derivation (e.g., sampling protocol, exact GPT-2-small block with error points).

## 5. Components, Architecture, and Data Flow

**Artifacts to Produce:**
1. The frozen `T1_precision_map_v0_2.md` (Stage 0 deliverable).
2. (Post-lock) New receipt module(s) implementing the depth-N error accumulator and certification harness.
3. New anchors in ig_primon for the experiment (C2 reproduction, certified depth curves, falsifier outcomes).
4. Optional thin CLI integration (e.g., `igprimon precision-depth` or similar) following existing patterns.
5. Supporting visual artifacts (iterated in .superpowers/brainstorm/ during Stage 0; key ones referenced in the pre-reg).

**Architecture:**
- Follows existing ig_primon split: immutable receipt logic at top level + operational wrapper in `ig_primon/`.
- Certification engine reuses `firewall.run_firewall` pattern and `hardware.scan`.
- Error accumulation implements the (frozen) recursion ε_{l+1} = (I + J_{f_l}) ε_l + δ_l.
- Tier-E (GPU FP32/FP8/FP4 explorer) proposes candidates for per-block errors; Tier-C (mpmath) certifies.

**Data Flow (High Level):**
Derivation (paper + visuals) → Locked pre-reg → Harness build (reusing existing precision/firewall) → Run controls (C1 identity, C2 random weights — must pass) → Run depth map on trained models with certification → Analyze falsifiers (F1/F2/F3) → (Conditional) Allocation rule arm.

**Error Handling:**
The certification *is* the error handling — any deviation beyond the frozen budget fails the anchor. Near-miss cases (the original Firewall's strength) are explicitly tested.

**Testing:**
- All new anchors run via `igprimon verify`.
- C1/C2 act as the gate tests.
- Full reproducibility via the program's existing CI/anchor discipline.

## 6. Scope and Non-Goals

**In Scope (exactly per artifact):**
- The 4 stages as described.
- Models ≤355M.
- Inference-only forward numerical error.
- RTX 5070 hardware with its native low-precision capabilities (FP8 + FP4 as distinctive).
- Natural + sharp-logit input arms.
- Exact mpmath certification as sole [V] authority.

**Explicitly Out of Scope (per walls):**
- Any claim for 70B+ models.
- Training or back-propagation error.
- Downstream task accuracy unless the [C] arm is promoted with held-out evidence.
- Distributed / multi-node numerics.
- Changes to existing [V] receipts (no-silent-edit).

## 7. Risks and Mitigations

- Falsifiers firing (especially F3 range dominance or F1 exponential on trained weights): Expected and valuable; the plan has explicit pivots.
- Certification cost (mpmath dps≥50 slow): Mitigated by pre-registered subsampling (part of the frozen protocol).
- Hardware limits (12 GB VRAM): Deliberately scoped to models that fit.
- Tooling immaturity for FP4 on Blackwell: Budget time; certify results especially carefully.
- Median ≪ mean in theory (Budzinskiy): The "nothing fires" outcome is still a valid [V] result; focus is on *which* inputs make amplification occur.

## 8. Milestones and Decision Gates

- End of Stage 0: Locked `T1_precision_map_v0_2.md` + reviewed visuals. No code for trained weights.
- End of Stage 1: C1 + C2 pass (harness validated, random-weight exponential reproduced).
- Stage 2: Depth map complete + falsifiers evaluated (F3 may cause pivot).
- Stage 3 (only if gates passed): Allocation rule with promotion evidence.

## 9. Post-Design Next Steps (per Brainstorming Protocol)

1. User reviews this design doc.
2. If approved, produce a first draft of `T1_precision_map_v0_2.md` (in worktree root) for review/lock, iterating visuals as needed.
3. Once pre-reg locked, transition via writing-plans skill to detailed implementation plan for the harness/receipts (Stages 1+).
4. All future work traces to the locked pre-reg.

## 10. Self-Review of This Design (Inline)

- No placeholders or TBDs.
- Consistent with compass artifact, IG-PRIMON-T1 discipline, and ig_primon patterns.
- Scope is tight (exactly the report's walls; no expansion).
- Ambiguities removed (e.g., "when code is written" is explicitly post-lock; sampling is pre-registered in the pre-reg).
- Approach 1 fidelity maintained throughout.
- Visual companion usage is concrete and already demonstrated with four live screens.

This design is ready for user review. Once approved, we can proceed to drafting the actual frozen pre-registration document.