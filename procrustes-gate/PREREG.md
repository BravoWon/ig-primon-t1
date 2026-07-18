# Procrustes-Import Gate — pre-registration (2026-07-18)

**Lineage.** Successor to DeltaSheaf (CLOSED: rotation-*invariants* of inter-model disagreement are
task-inert) and CODA-2 (import recovers the holes when the *content* is right — text channel, 81.7%).
This gate tests the **latent** channel with the rotation *operation* (not its invariants) as the payload.

## Claim
For a **heterogeneous** pair — a specialist S that holds latent structure a generalist G lacks — an
orthogonal **Procrustes** rotation aligning their hidden-state spaces lets us **transplant** S's
representation into G and recover items G fails but S knows: beyond random rotation, beyond identity
injection, beyond a wrong-item transplant, and (the *value* question) beyond simply handing G the fact as
text (CODA-2's **81.7%**).

## Why this is not already in the DeltaSheaf graveyard
1. DeltaSheaf refuted what rotation *preserves* (the rank-7 invariant object) as a decodable signal. This
   uses what rotation *does* — the specific alignment matrix R — as the transported object. Our nulls are
   silent on it.
2. v0.2 restriction maps `R_ij` were **ridge-LS — a strict superset of orthogonal Procrustes** — and
   recovered nothing **on the 0-of-5 blind spots** because the ensemble was **homogeneous**: nothing to
   transplant. This gate requires **heterogeneity**, and its test set is therefore **items G fails & S
   succeeds on domain X**, *not* the blind spots. You cannot rotate a feature into existence.

## Precondition — instrument-first HARD GATES (abort if either fails)
- **Heterogeneity:** pick (S, G, domain X) with `acc_S(X) − acc_G(X) ≥ 0.30`. Verify first; else no transplant
  is possible — pick another pair.
- **Alignment validity:** fit R on anchor items, then on a **disjoint anchor holdout** require R to
  materially improve cross-model representation agreement vs random-R (cross-model nearest-neighbour
  retrieval accuracy, or aligned-distance reduction). If R does not align even anchors, abort — the spaces
  aren't Procrustes-comparable at this layer.

## Setup
- **Models (cached-first):** primary = a genuine specialist (e.g. Qwen2.5-Coder / a math-tuned 7B) vs a
  general 7B; fallback = strong/weak pair from the 5-model ensemble on an MMLU subject with a large gap
  (choose S,G,X by the heterogeneity metric above).
- **Layer L:** sweep mid–late layers; pick the one with best **held-out anchor** alignment (chosen on
  anchors only, never on test recovery).
- **Reps:** hidden state at the answer-relevant position(s) (last token / option tokens),
  `output_hidden_states=True`.
- **Split:** anchors (fit R) disjoint from test items (G-fails-S-succeeds on X), seeded.

## Mechanism (primary arm)
1. **Procrustes:** `R = U Vᵀ` where `SVD(H_Gᵀ H_S) = U Σ Vᵀ` → the rotation-only map S→G on anchors.
2. **Transplant:** run S forward on a test item, grab `h_S^L`; inject `ĥ = R h_S^L` into G at layer L
   (residual-add or replace at the position), continue G's forward, read the answer.
3. Score recovery on held-out test items.

## Controls (all pre-registered — this is the whole point)
| control | isolates | expected if claim true |
|---|---|---|
| **text-import** (G + fact-in-context = 81.7%) | the *value* falsifier | transplant **>** 81.7% |
| **routing** (just use S's answer) | the triviality upper bound | transplant ≈/≤ routing (names the confound) |
| **random-rotation** (`R_rand h_S`) | is the *alignment* doing work? | transplant **>** random-R |
| **identity / no-align** (`R=I`, raw `h_S`) | are the spaces natively comparable? | identity fails; transplant wins |
| **self-transplant** (G→G, R≈I, inject G's own h) | injection surgery validity | near no-op (accuracy preserved) |
| **wrong-item transplant** (S's h for a *different* item) | injection artifact vs real transfer | does NOT recover / corrupts |

## Decision rules (FIXED)
- **MECHANISM CONFIRMED** iff recovery beats random-R **and** identity **and** wrong-item — each CI-clean,
  paired, held-out — **and** self-transplant is a near-no-op.
- **VALUE CONFIRMED** iff mechanism confirmed **and** recovery **> 81.7%** (paired).
- Mechanism yes, value no (≤81.7%): latent channel is **real but not worth it vs text** — bank as
  "expensive parlor trick" (Gemini's own bar).
- Recovery ≈ random-R: Procrustes alignment **inert** → bank null.

## Named confounds (up front, not buried)
- **Routing triviality:** if `h_S` encodes the answer and G reads it, recovery ≈ using S. Then "value" is
  *capability-fusion / efficiency*, not raw recovery. The random-R + wrong-item controls isolate whether the
  **alignment** (not merely S's decision) is the mechanism; the routing baseline bounds the triviality. The
  confound is fully retired only by the v4 subspace arm below (transplant a *competence direction*, not a
  per-item state).
- **Researcher DOF:** layer and Procrustes±scale are chosen on the **anchor** alignment metric, pre-committed
  before any test-recovery number is computed.
- **Injection artifacts:** overwriting a hidden state can destabilize G's forward pass; the self-transplant
  control detects it.

## Machinery (already built — put a gradient through it)
- Closed-form Procrustes needs only SVD. The **differentiable Givens/butterfly rotation** already sits idle
  in `deltasheaf-v02/build_v10.py` (21-plane) and `cocycle/components/projector.py` — that's the engine for
  the **learned-alignment** escalation (fit R by gradient on a *transfer* objective on the Stiefel manifold,
  vs the least-squares closed form).

## Escalation ladder
- **v1** — closed-form Procrustes (SVD on anchors), single chosen layer, injection + full control suite.
- **v2** — Procrustes+scale; layer sweep; multi-position injection.
- **v3** — **learned** alignment: gradient through the butterfly/Cayley Stiefel path on a transfer loss vs
  the closed-form baseline (the idle-machinery test).
- **v4** — **subspace / steering transplant**: extract a *domain-competence direction* from S over anchor
  X-items (not a per-item state), rotate into G, steer G on new X-items. This is the one that **cannot
  reduce to routing** — the genuinely non-trivial capability transplant.
