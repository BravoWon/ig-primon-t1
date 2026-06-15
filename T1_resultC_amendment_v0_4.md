# IG-PRIMON-T1 — Result C Amendment (v0.4 proposal, 2026-06-14)

**Status: REGISTRATION ACT — SIGNED & APPLIED 2026-06-14 (folded into `T1_consolidated_results_ledger_v0_4.md`).**
This is the versioned diff that carries the trunk v0.3 → v0.4 for **Result C only**. It records a *relabel*
and a *strengthening*; it alters **no existing `[V]`** (v0.4 was created by byte-copying v0.3 and editing
only the Result-C sections §3/§5/§6.7/§7 + header). In the program's amendment-doc pattern
(cf. `T1_HE2_amendment_v0_5.md`); retained as the historical diff.

Supporting receipts (all CPU, certified this session): `module_L_perceptron_replica.py`,
`module_L_perceptron_replicon.py`, `module_L_perceptron_curvature.py` (indefinite first pass),
`module_L_perceptron_finiteT.py` (positive-definite closure); gate doc
`T1_moduleL_perceptron_gate_v0_1.md` (v0.4).

---

## Act 1 — Relabel the genus (Result C §3 and Paper 3, §5)

**Falsified framing:** "Geometry of **learning** transitions." The kinematic side (ridge double descent) is
a learning model, but the genuine side is **not** about learning: SK is a spin glass, and the perceptron
result below is a *storage / constraint-satisfaction* transition. "Learning" overreached.

**Replacement (registered):** the genus is **the geometry of continuous-RSB criticality vs. kinematic
volume-divergence in disordered systems.** Learning (ridge), spin glass (SK), and **storage/jamming**
(perceptron) are *venues*, not the subject. The four-tier diagnostic and both `[V]` anchors are unchanged
in content; only the framing/title change.

- **Paper 3 retitle (registered, pending the manuscript):** from "Geometric diagnostics of learning
  transitions" → e.g. **"A curvature diagnostic for continuous-RSB criticality vs. kinematic
  volume-divergence in disordered systems"** (learning, spin-glass, and storage as the three venues).
  Title wording to be finalized at draft.

## Act 2 — Register the perceptron as the third archetype (Result C §3; §6.7 → resolved)

The Gardner **spherical-perceptron storage** problem is a clean third archetype that carries **both** sides
of the dichotomy in one model:

| venue | side | status |
|---|---|---|
| convex (κ≥0) storage / SAT | **kinematic** (`q→1` at capacity = volume → point) | [V] `α_c(0)=2` exact |
| non-convex (κ<0) jamming | **genuine** (continuous replicon, `χ=1/λ_repl→∞` at the AT line) | [V] (§6.7 closed) |
| Ising weights (storage / learning) | frozen-1RSB / first-order (method fails) | [V]/lit: `α_RS=0.833`, learn α≈1.245 |

**§6.7 resolution (supersedes the v0.3 "likely frozen-1RSB / first-order" caution).** Ising **is**
frozen-1RSB/first-order (method fails, as feared); but the **non-convex spherical perceptron** is a
continuous replicon and **carries the SK-converse method** — the divergent branch's realization the program
sought. Receipts: replicon `λ_repl=1−α∫Dt[G₁′]²` validated against Gardner (`λ_repl→0` at α=2, κ=0);
`α_AT(κ)<α_c(κ)` continuous (κ=−0.1…−0.5); `χ·(α_AT−α)→3.22` (κ=−0.5).

## Act 3 — Register the curvature result, with its honest status (Result C §3, new sub-entry)

Two claims, two tags — kept separate (this is the load-bearing correction):

- **Susceptibility-genuine — `[V]`** (the perceptron is on the genuine side *by the susceptibility
  criterion*): `g_εε ~ 1/λ_repl` is one real conjugate-field susceptibility, divergence coordinate-
  independent, Gardner-anchored, component-driven (freeze χ → kills it), not the CW degeneracy. (Act 2.)
- **Curvature-genuine — `[V]`** (`module_L_perceptron_finiteT.py`): the literal Ruppeiner scalar `|R|→∞`
  as `χ² ~ 1/(β_AT−β)²` (`|R|·(β_AT−β)²→11.8`) on the **finite-T `(β,ε)` manifold**, where β is a genuine
  natural field (`g_ββ=2φ''=2Var(energy)/N>0`, verified) tuning the replicon (`λ_repl(β)→0` at finite
  `β_AT=5.99`; κ=−0.5, α=4.2). Both coordinates are conjugate fields ⇒ the metric is a covariance ⇒
  **`det g>0` throughout and diverging** — a diverging **component** on a non-degenerate positive-definite
  metric, the literal Riemannian twin of SK's `R_ε`. The first pass tuned by the structural load α
  (`module_L_perceptron_curvature.py`) gave the same divergence on an *indefinite* metric (`det g<0`) and is
  recorded as the cautionary intermediate the refined diagnostic correctly flagged — not promotable on its
  own. Curvature-genuine `[E]→[V]`, capstone closed.

## Unchanged (explicit)

NO-RH, ZFD/Siegel wall, Results A & B, the audited Module A/C/D core, every constant, and every prior `[V]`
— untouched. This amendment is additive to Result C plus a relabel; it corrects no certified number.

---

## Proposed v0.3 → v0.4 changelog line (for the ledger on sign)

> v0.3 → v0.4 (2026-06-14): Result C relabel + §6.7 resolution. Genus retitled from "learning transitions"
> to "continuous-RSB criticality vs. kinematic volume-divergence in disordered systems" (learning/spin-
> glass/storage as venues; Paper 3 retitle registered). §6.7 closed: Ising perceptron = frozen-1RSB/first-
> order (method fails); **non-convex spherical perceptron = continuous replicon, carries the SK-converse
> method** at the AT/jamming line: susceptibility-genuine `χ=1/λ_repl→∞` **[V]**; curvature-genuine
> `|R|~χ²→∞` **[V]** on the positive-definite finite-T `(β,ε)` manifold (`det g>0`, `g_ββ>0`; the
> indefinite `(α,ε)` first pass retained as cautionary). Third archetype registered (carries both sides).
> Receipts: module_L_perceptron_{replica,replicon,curvature}.py; gate T1_moduleL_perceptron_gate_v0_1.md.
> No existing `[V]` altered. Versioned diff, no silent edit.

— End of Result C amendment v0.4 (proposal). Apply to the ledger only on the user's sign.
