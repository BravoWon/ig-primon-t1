# DeltaSheaf-v0.2 — Boundary Layer as Hole Mapper
Clean-build charter · fresh directory `deltasheaf-v02/` · pre-code freeze
**Document status: FROZEN (v0.2.1)** · Date of freeze: 16 July 2026
SHA-256 of this document: see `SPEC.hash` (computed over these exact bytes at write time)
Principle: Derive → Lock → Build → Measure. No post-hoc geometry changes.

> Provenance: merges the 2026-07-16 review (fixed-choice task · cycle-residual-only signature · full
> control suite with leak sentinel · powered verdict logic) with the reply-text-embedding stalk, cosine
> readout, dead-zone handling, and gold verification from the collaborator draft. Two banked negative
> priors are acknowledged (§0): the H¹ routing gate found higher-order gluing **inert over pairwise
> (+0.000)**, and the recursion/nonuniform gates found typed sheaf composition never separates from
> generic recurrence. This SPEC gives the one genuinely-untested claim (decode-to-recover) a fair,
> decisive shot; the thread prior favors FALSIFIED, which is a valid, valuable outcome.
>
> **Amendment v0.2.1 (2026-07-16, from the Phase-1 instrument smoke):** on a 0-of-N blind-spot set majority
> `A = 0%` by construction, so majority-relative thresholds are degenerate — **all verdict thresholds are
> CHANCE-relative** (chance = 1/n_options). And a feature-column permutation is absorbed by the linear
> decoder, so **`ctrl-shuffle` is a label-pairing (row) permutation** that breaks the signature↔item
> correspondence. No geometry-substrate change; verdict logic + one control corrected before the run.

---

## 0. Governing Principle (Locked)
The boundary layer maps the manifold holes. Inter-model cycle inconsistencies are the observable structure
of topological obstructions in the collective knowledge manifold. On items where the ensemble is jointly
wrong, the **cycle-residual Hole Signature** must carry decode-recoverable information about the correct
region — beyond first-order disagreement, beyond disagreement *volume*, and beyond the hard-item answer
prior — even when the correct region lies outside the span of every emitted answer. That is the operational
meaning of "the delta is the *map*, not the *volume*."

## 1. Geometry Substrate (Locked — inherited, unchanged)
- Dual-dimension stalks in ℝ⁵¹²: high-dim adapted path (preserves mismatched geometry) + low-dim working
  path, mixed by a learned router g ∈ [0,1].
- Edge objects: Δ_ij = R_ij(stalk_i) − stalk_j, R_ij a learned linear restriction map; complete digraph on
  the N=5 ensemble nodes.
- Sheaf energy = **cycle residual only**: `sheaf_energy = Σ_{i<j<k} ‖(δΔ)_ijk‖²`, `(δΔ)_ijk = Δ_ij + Δ_jk − Δ_ik`.
- **No `‖Δ_ij‖²` term anywhere.** Individual delta magnitudes remain free.

## 2. Stalk Definition (Locked)
Stalk for model i on an item = **nomic embedding of the model's full reply text** (short rationale + final
choice), Matryoshka-truncated to ℝ⁵¹², then passed through the locked dual-path + router. The frozen prompt
template elicits a brief rationale **and** the choice (never a bare letter) — so that even when models pick
the same wrong option, their reasoning differs and deltas do not collapse to zero. All embeddings (stalks,
options, decoder outputs) live in the **same** nomic ℝ⁵¹² space.

## 3. Primary Object: Hole Signature (Locked — cycle-residuals only)
`H = P · vec({(δΔ)_ijk : i<j<k})`, `P` a **fixed seeded Gaussian (JL) projection** ℝ^{C(5,3)·512}=ℝ^{5120} → ℝ^{d_sig}, **d_sig = 128** (seed frozen at freeze).
- **Edge deltas are NOT in H.** They are the input of control `B_edge` only, reduced by the *same* JL
  procedure to ℝ^{128} (capacity parity with C_cycle).
- Rationale for reduction: raw residual dim 5120 ≫ sample size; `P` is fit-free (no leakage).

## 4. Decoder Input / Output (Locked)
- Primary arms (`C_cycle`, `B_edge`) and controls receive **only** their designated input vector — never the
  question embedding, raw stalks, or surface answers.
- Decoder emits a vector in the shared nomic ℝ⁵¹² space. Prediction = the option whose nomic embedding has
  **highest cosine similarity** to the decoder output. This makes "point to gold *outside* the emitted span"
  a well-posed geometric target.
- **The strong claim, named for `C_cycle`:** from the geometry of five mutually-inconsistent wrong answers
  *alone*, produce a vector nearest to a gold option **no model emitted**.
- Decoder (primary): linear map ℝ^{d_sig} → ℝ⁵¹², L2 λ frozen, early-stopped on a clean-only val split.
  Secondary: 1-hidden-layer MLP (reported, not the primary verdict driver). Identical across all arms;
  **only the input differs.**

## 5. Experimental Arms & Controls (Locked) — identical decoder, identical train-only gold; input is the only difference
| arm | decoder input | isolates / kills |
|---|---|---|
| **A** | majority vote (no decoder) | baseline |
| **ctrl-blind** | fixed random/constant vector (item-independent) | answer-prior / label-shift leak on the adversarial set |
| **ctrl-mag** | `‖H‖` scalar only | "disagreement *volume*, not map" |
| **B_edge** | edge deltas {Δ_ij}, JL→d_sig | first-order / pairwise (the H¹-gate bar) |
| **C_cycle** | **Hole Signature H (cycle residuals only)** | **the sheaf / topology claim** |
| **ctrl-shuffle** | H with seeded **label-pairing (row) permutation** — breaks H↔item; a feature-column permutation is absorbed by a linear decoder | fitting noise / decoder capacity |

## 6. Data Construction (Locked — hybrid)
- **Task:** ≥4-choice MC; exact-match via cosine-nearest option in frozen nomic ℝ⁵¹² space.
- **Gate set (confirmatory):** ≥ **300** blind-spot items from a trusted-label benchmark (MMLU primary,
  ARC-Challenge fallback). A blind-spot item = **0-of-5 models correct AND ≥2 distinct options chosen**
  across the ensemble (items where all five pick the *same* wrong option are a documented geometric
  **dead zone**, excluded from the gate set). Held out from all training.
- **Anchor set (interpretability only, NOT in the verdict):** ~30 hand-curated items, each gold label
  **independently verified**; uncertain items flagged for **user sign-off** before `items.hash`. Used for
  qualitative "is it doing something sensible?" inspection, never for PASS/FAIL.
- **Clean-train:** ≥ 1000 benchmark items (maps/router/adapter/decoder). Files: `data/items.jsonl` +
  `data/items.hash`.

## 7. Hypotheses & Gates (Locked, concrete)
- **H3 — hardness gate (hard):** the gate set is 0-of-5-correct with ≥2-option diversity by construction;
  require n ≥ 300 such items, else **abort** and expand the screening pool. Report the achieved count.
- **H4 — leak sentinel (hard):** `ctrl-blind` accuracy on the gate set ≤ **CHANCE + ε** (CHANCE = 1/n_options;
  majority A is degenerate at 0% on a 0-of-N blind-spot set, so the bar is chance, not A). If `ctrl-blind`
  beats chance by more than ε, the adversarial set has an exploitable answer-prior → the comparison is
  **VOID** until the set is de-biased. *(Single most likely false-positive path; hard gate.)*
- **Pre-measurement diagnostic (report only):** mean `‖H‖` on gate vs clean items (non-emptiness ratio).
- **H1 — primary:** on the gate set, `C_cycle` exact-match exceeds **both** `B_edge` **and** `ctrl-mag` by
  ≥ **δ_min = +0.10 absolute**, holding in the mean over ≥5 seeds, with a paired test clearing (McNemar or
  paired bootstrap 95% CI excluding 0). Also require `C_cycle > ctrl-shuffle` and `C_cycle > A`.
- **H2 — secondary:** `C_cycle` shows ≥ +0.05 over `ctrl-blind` (structure beyond leak) on the gate set.

## 8. Metrics & Verdict Logic (Locked — not deferred)
Report EM accuracy per arm on **clean** and **gate** sets, mean ± sd over ≥5 decoder seeds, plus the anchor
set qualitatively.
- **PASS** iff **all**: H3 passes · H4 clean · `C_cycle − B_edge ≥ +0.10` (paired test clears) ·
  `C_cycle − ctrl-mag ≥ +0.10` (paired test clears) · `C_cycle > ctrl-shuffle` · `C_cycle > CHANCE + ε`
  (note: `> A` is degenerate — A = 0% on the blind-spot set by construction).
- **FALSIFIED** iff `C_cycle ≤ B_edge` or `≤ ctrl-mag` within noise (test does not clear) → the cycle
  structure is mathematically real but **operationally inert at decode** (H¹ redux, one level down). A live,
  thread-consistent, acceptable outcome; rescue path (low prior) = a non-nested / real-MoE cover.
- **VOID** iff H3 aborts (set not hard enough) or H4 fires (label-shift leak).
- Note: δ_min = +0.10 is a **decisive-effect** bar — a genuine but <10-pt effect reads FALSIFIED. Chosen
  deliberately: only a decisive map is worth building on, given the +0.000 H¹ prior.
- **Frozen constants:** `δ_min = +0.10` · `ε = +0.02` · `d_sig = 128` · seeds ≥ 5 · paired bootstrap 10 000
  resamples, seeded, 95% CI.

## 9. Evidence Handling (Locked)
Gold is **train-only** supervision for the decoder (Arm C_cycle and every control that carries evidence).
It is never a graph vertex; cycle energy and H use only the N=5 model nodes at train and test. At test, no
arm sees gold.

## 10. Train Protocol (Locked)
Restriction maps, router, adapters, decoder trained **only** on the clean set. Gate + anchor sets are
evaluation-only. All arms share the identical clean/gate split; controls differ from `C_cycle` **only** in
decoder input. `B` (no-evidence variant) trains with no gold term; the evidence arm uses train-only gold.

## 11. Seeds, Hashes, Freeze Procedure (Locked)
Frozen before the first model query: ensemble list (N=5) + per-model revision hashes · prompt template ·
JL seed (`P`) · shuffle seed · decoder-init & data-split seeds · bootstrap seed · `d_sig, δ_min, ε` ·
`data/items.hash` · `SPEC.hash = SHA-256(this file)`. No geometry-substrate changes after freeze.
Ensemble (revisions pinned at freeze): 5 diverse small instruct models across ≥3 families (e.g.
Qwen2.5 · Llama-3.2 · Phi-3.5 · Gemma-2 · Mistral), chosen for genuine disagreement.

## 12. Scope Wall (Locked)
- Does not claim the boundary layer reliably recovers truth.
- A PASS licenses only: *"on this task and ensemble, the cycle-residual structure carries decode-recoverable
  information about jointly-wrong items beyond first-order deltas, disagreement volume, and the hard-item
  answer prior."* Nothing broader.
- A null (FALSIFIED) under these controls is informative and is the thread's prior expectation.

## 13. Implementation Order
1. Fresh dir `deltasheaf-v02/`; write this SPEC.md; compute `SPEC.hash`.
2. Ensemble scoring: 5 frozen models → reply text (rationale + choice) → nomic ℝ⁵¹² stalks.
3. Locked substrate: dual-path stalks + router, restriction maps, cycle-residual energy.
4. Hole Signature H (cycle residuals → seeded JL → ℝ¹²⁸); `B_edge` input (same reduction).
5. Decoder (linear primary; MLP secondary) → nomic ℝ⁵¹² → cosine-nearest option.
6. The six arms, differing only in input.
7. Build gate set (≥300, 0-of-5, ≥2-option) + anchor set (~30, verified, user sign-off); run **H3** and the
   **H4 leak sentinel** and the pre-measurement diagnostic **before** inspecting any arm result.
8. Freeze ensemble/revisions, all seeds, hashes, `δ_min/ε/d_sig`, prompt template.
9. Run all arms, ≥5 decoder seeds each; compute the §8 verdict.
10. `RESULTS.md`: per-arm mean ± sd + paired tests + the explicit §8 verdict; anchor set qualitatively.

No changes to the geometry substrate after step 8.

*End of Frozen Specification (v0.2).*
