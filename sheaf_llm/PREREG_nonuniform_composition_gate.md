# Pre-registration — non-uniform (non-commutative) composition: the recursion gate's honest rescue

## Why this gate exists
The recursion gate (`recursion_gate.py`) fired the falsifier on typed sheaf composition: `grounded-sheaf`
(typed, weight-tied restriction maps `R_subj/R_verb/R_child` folded along the tree) ≡ `grounded-GRU`
(generic recurrence on the same grounded features) on everything, at comparable params. Its own stated
caveat named the **one** escape it could not close:

> "the target is a *uniform* recurrence (easy to roll out); a **non-uniform** compositional task is the
> only place typed composition could still separate — low prior given the thread, but the honest rescue."

That target was `val(clause) = (ss(subj) + 2·val(child) + verb) % S` — **abelian**: verbs enter by
commutative mod-add, the child by a scalar, so the whole fold is an order-insensitive mod-`S` accumulation
a single shared recurrence represents trivially. This gate builds the non-uniform target the caveat
demands and runs it against the abelian one **as an in-script control**, changing *nothing else*.

## The one dimension that changes (everything else held fixed)
Same three arms, same architectures (verbatim from `recursion_gate.py`), same vocabulary, same nested-SVO
trees, same train/test depth split, same per-arm shared data stream. The **only** change is the target's
composition operator:

- **abelian (CONTROL, commutative):** two variants (see AMENDMENT v2 — the clean one-property twin turned
  out to be un-learnable, forcing the split):
  - `abelian_sym` — the clean commutative twin `val ← (val + ss(subj) + verb) % S`, differing from the
    payoff in *exactly* the operator (add vs permute). Documents the parity cliff (below).
  - `abelian_wt` — the **learnable** commutative anchor `val ← (2·val + ss(subj) + verb) % S` (this is
    `recursion_gate.py`'s actual fold). Weighted positions keep it GD-learnable, so it yields a *valid*
    "architectures tie" null. This is what H2 is tested on.
- **non-abelian (PAYOFF, this gate):** `val ← P[verb][ (val + ss(subj)) % S ]`, leaf `val = ss(obj)`,
  where `P` is `NV` fixed random permutations of `{0..S-1}` (one per verb), drawn once from a fixed seed
  and identical across all arms and both tasks. Permutations do not commute, so the composed value is a
  function of the **ordered sequence of (verb-type, subject-shift)** edges along the actual tree — not of
  any commutative summary of them.

Trees: `(S V O) → (S V (S V O)) → …`; train depths {1,2,3}; test depths {1..6}; **extrapolation** = {4,5,6}
(deeper than trained) × {seen, unseen} words. `S=8` supersenses, `NV=8` verbs, `D=64`, `HID=128`,
`STEPS=9000`, `BS=128`, `LR=2e-3`, `SEED=0`. Chance = `1/S = 0.125`.

Every arm *learns* its maps; none is handed the permutations. The sheaf's only privilege is the correct
inductive bias — one tied linear map per **edge type**, folded in the true tree order. The GRUs see the
edge type as an input feature and must learn to condition on it. That is exactly the contrast the
recursion gate tested; only the target's commutativity is flipped.

## Hypotheses (pre-registered)
- **H1 — PAYOFF (decisive):** on the **non-abelian** target, `grounded-sheaf` extrapolates to unseen depth
  (4–6) materially better than `grounded-GRU` on **seen** words — the typed tied fold is finally the lever
  because order/type now matter. Margin threshold: sheaf − gru extrapolation acc **> 0.15**.
- **H2 — CONTROL (causal isolation):** on the **abelian** target, `grounded-sheaf ≈ grounded-GRU`
  (reproduce the recursion gate's null, |Δ| ≤ 0.15). Required so that any H1 separation is attributable to
  non-commutativity and not to the architecture per se.
- **H3 — grounding word-win persists:** both grounded arms beat `flat` on **unseen** words at all depths
  on both targets (the thread's one robust win; a sanity anchor, not the novelty here).

## Controls
- **Abelian control in the same script** — the causal isolator for H1 (H2 above).
- **Shared per-arm data stream** — every arm trains/tests on the identical generated stream (the PR#13
  discipline), so no arm is advantaged by luckier data.
- **Parameter count printed per arm** — if `grounded-GRU` has ≥ `grounded-sheaf` params, the capacity
  confound runs *against* an H1 win (the semantic-gate discipline). Reported, not hidden.
- **In-distribution fit reported (depths 1–3)** — if no arm fits the trained depths, the task is too hard
  at this scale and the extrapolation comparison is declared uninformative rather than spun.

## Falsifier
On the non-abelian target, `grounded-GRU` extrapolates to depth **as well as** `grounded-sheaf`
(sheaf − gru ≤ 0.15) → the typed tied fold adds nothing **even when the task is built to require ordered,
typed composition.** That is the final nail: specifically-sheaf structure never separates from generic
grounded recurrence, across uniform *and* non-uniform composition. A live, acceptable outcome; the thread's
prior favors it.

## Honest limits (stated before the numerics)
- One task family (permutation composition over WordNet supersenses); one scale (≤ 0.3 M params).
- If **all** arms decay to chance at extrapolation (task too hard to extrapolate for anyone), the gate is
  uninformative on H1 — that is reported as such, not as a sheaf win or loss.
- Confirms/refutes an inductive-bias edge at small scale; says nothing about billions of params.

## Metrics
Extrapolation accuracy (mean over depths 4–6) per arm, per task, split seen/unseen words; in-distribution
accuracy (depths 1–3); the sheaf−gru margin on the non-abelian seen-word extrapolation (the number that
licenses "typed composition is the lever"); param counts. Chance line on every plot.

## AMENDMENT (v2, 2026-07-14) — the control had to be split, and the sheaf had to converge
Run 1 (`nonuniform_gate.py`) exposed two instrument faults, both caught by the program's own discipline
(one by the round-trip verifier from the code alone, one by reading the run):
1. **The clean commutative twin is parity-hard.** `abelian_sym` = `(val + ss(subj) + verb) % S` unrolls to
   an *unweighted* symmetric modular sum `(Σ ss + Σ verb) mod S` — a generalized-parity task GD-trained
   RNNs cannot optimize. Both GRUs sat exactly at chance (loss = ln 8 = 2.080), so the control produced no
   valid "architectures tie" null. Fix: add `abelian_wt` = `(2·val + ss(subj) + verb) % S`
   (`recursion_gate.py`'s actual weighted fold), which is GD-learnable and yields the valid H2 null. Keep
   `abelian_sym` only to *document* the cliff.
2. **The sheaf was undertrained at the comparison point** (run-1 non-abelian loss 0.055 vs GRU 0.001).
   Fix: 16 000 steps + print final loss per arm so convergence is visible; compare extrapolation only once
   both grounded arms are converged.
Also fixed the two verdict-reporting residuals the verifier named (H3 now covers both grounded arms on all
tasks; H2 is guarded against asserting a null when arms failed to learn). `nonuniform_gate_v2.py`, 3 tasks
× 3 arms × 2 seeds. Round-trip re-verified: **drift 0.97 → SHIP**, both residuals closed, no new drift.

## RESULT (`nonuniform_gate_v2.py`, 2026-07-14): FALSIFIER FIRED — holds after convergence. Grounding, not sheaf structure, is the lever.
Converged (both grounded arms in-dist ≈ 1.00), 2-seed mean, chance 0.125:

| task | arm | loss | in-dist d1-3 | extrap d4-6 seen | extrap unseen |
|---|---|---|---|---|---|
| **abelian_wt** (valid H2 anchor) | flat / gru / sheaf | 0.000 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.12 / 1.00 / 1.00 |
| **abelian_sym** (parity twin) | flat / gru / sheaf | 2.08 / 2.08 / 0.69 | 0.12 / 0.12 / 0.62 | 0.13 / 0.13 / 0.62 | 0.12 / 0.12 / 0.63 |
| **nonabelian** (PAYOFF) | flat / gru / sheaf | 2.08 / 0.015 / 0.037 | 0.13 / 1.00 / 0.99 | 0.13 / **0.91** / **0.79** | 0.13 / 0.91 / 0.78 |

- **H2 (valid null) ✓** — on the learnable commutative fold all three arms tie at 1.00 (flat on seen only;
  both grounded arms on seen+unseen). Architectures do not inherently differ; the instrument is valid.
- **H1 (decisive) — FALSIFIER FIRED.** On the non-commutative task built to *require* ordered typed
  composition, the generic grounded-GRU extrapolates to depth **better** than the typed sheaf
  (0.91 vs 0.79, margin −0.13), *after* the sheaf is trained to convergence (in-dist 0.99, loss 0.037).
  The typed weight-tied restriction-map fold is **not** the lever — if anything it extrapolates modestly
  worse. (Honest refinement: run-1's larger −0.19 gap was partly undertraining and partly single-seed luck
  in the GRU; the converged 2-seed truth is that *both* grounded arms decay with depth, the GRU slower.)
- **H3 (grounding word-win) ✓** — on both learnable tasks both grounded arms crush flat on unseen words
  (1.00 vs 0.12; 0.91/0.78 vs 0.13). The dictionary generalizes zero-shot; the sheaf's *structure* adds nothing.
- **Parity cliff documented** — `abelian_sym`: GRUs at chance (loss 2.08), the structured sheaf fold partly
  cracks it (0.62). A minor aside that the sheaf's inductive bias differs from the GRU's — not a compositional win.

**Verdict:** the recursion gate's stated "honest rescue" (non-uniform composition) is **resolved-negative.**
Specifically-sheaf typed composition does not separate from generic grounded recurrence, on uniform *or*
non-uniform targets. **Grounding is the lever; sheaf geometry is not** — the thread's central finding, now
holding at the one gate that was its last chance to fail.

**Honest limits carried forward:** (b) H1's programmatic guard uses arm-max in-dist fit, not the sheaf's own
(here immaterial — sheaf in-dist 0.99 — but a stricter convergence contract would gate on the sheaf directly);
one task family; ≤ 0.3 M params; says nothing about billions of params or a fluent generative LM.

## Status
`[GATE]` → RUN, **FALSIFIED** (see RESULT). Typed sheaf composition is NOT a validated design component,
on non-uniform composition as on uniform. The buildable substrate remains **grounding (inherited dictionary)
+ generic recurrence**, not typed sheaf structure.
