# IG-PRIMON-T1 — Genuine-Side Singularity Gate (analysis, v0.2, 2026-06-15)

**Status:** `[GATE-ANALYSIS]` — pencil, no sampler. Decides *on paper* whether the genuine-side
metric-curvature diagnostic (Result C, `|R|→∞` on a positive-definite metric) survives the **singular**
(real-net) case, before any SGLD/MCMC effort. Companion to the partial control
`module_L_perceptron_mcmc_control.py`. No silent edits; amendments are versioned diffs.

**v0.2 (this revision).** §9 locates the `χ`-divergence and finds the §5 rate competition is a **special
case (branch 2) only** — generically the singular suppression and the genuine divergence are not
co-located, so the gate's real content is a **trichotomy on the net posterior's RSB structure**, not a
rate. v0.1 (§§1–8) stands as the `det g`-race setup it was.

**Verdict (honest, conditional — NOT a closure):** the genuine signature is **not** unconditionally
killed by singularity. It **survives** at any finite-temperature transition, and the singular structure
closes the gate **only** in the narrow regime where the criticality co-locates with the singular limit
(`β→∞`) *and* the susceptibility grows no faster than `β²`. The genuine-side status thus reduces to a
**decidable rate competition**, not a one-line closure. (This corrects the v0 first pass, which over-claimed
"closed under the generic-singularity assumption"; the toy below refutes the unconditional form.)

---

## 1. The condition that must break
Definition `def:diag` (Paper 3): a **genuine** transition is `|R|→∞` produced by a diverging metric
*component* on a **non-degenerate, positive-definite** metric (`det g > 0`). A `det g → 0` / indefinite
divergence is **F-spurious — disqualified**. So closing the genuine side means showing the singular
structure forces `det g → 0` (or `< 0`) at the transition.

## 2. SLT spine (why real nets are the singular case)
Overparametrized nets are **non-identifiable**: permutation, scaling, and dead-unit degeneracies are
positive-dimensional, not measure-zero. On those sets the Fisher information degenerates, `det I(w) → 0`,
and the Bayes free energy is governed by the **RLCT** `λ` (`F_n ≈ λ log n`; local version = the LLC), not
the dimension/Hessian. The field uses `λ` *because* metric-Hessian geometry fails at singularities.

## 3. `det g` is a competition, not a one-sided closure
`det g = g_ββ·g_εε − g_βε²`. The **genuine mechanism drives it the wrong way for a closure**: in the
spherical-perceptron `[V]` (`module_L_perceptron_finiteT`), `g_εε = χ → ∞` at `β_AT` while `g_ββ` stays
finite, so `det g → +∞` — divergence on a metric that stays positive-definite *and blows up positively*.
To close the gate, the singular structure has to override that and force `det g → 0`.

## 4. The singular suppression of `g_ββ` (toy density of states)
A singular minimum has DOS `ρ(E) ~ E^{λ−1}` (the RLCT exponent). Then
`Z(β) = ∫_0^∞ E^{λ−1} e^{−βE} dE = Γ(λ)·β^{−λ}`, so `⟨E⟩ = λ/β`, `Var(E) = λ/β²`, hence
`g_ββ ~ λ/β² → 0` as `β → ∞`. The same `λ` is the RLCT (`F_n ≈ λ log n`, `β ~ n`): **the singularity
suppresses the energy-fluctuation metric through the very exponent SLT lives on.** This is the engine of
any closure — but it is a `β→∞` (low-temperature) effect.

## 5. The race (the toy verdict)
`det g ~ (λ/β²)·χ − g_βε²`.

- **Finite-`β` transition** (`β_AT < ∞`; perceptron, SK at dAT): `g_ββ = O(1)`, `χ → ∞` ⟹ **`det g → +∞`.
  Genuine signature SURVIVES — gate OPEN.** The singular suppression never reaches a finite-`β` transition.
- **Singular-limit transition** (`β → ∞`, where a real-net posterior concentrates on the singular minima),
  `χ ~ β^p`: `det g ~ λ·β^{p−2} − g_βε²`, with `g_βε →` finite (`q* → 1`). Then
  - `p < 2` ⟹ `det g → 0` (or `< 0`, indefinite) ⟹ **F-spurious — gate CLOSED**;
  - `p > 2` ⟹ `det g → +∞` ⟹ genuine survives, **gate OPEN**;
  - `p = 2` ⟹ `det g →` finite ⟹ borderline (`R` finite, no clean divergence either way).

  **Threshold `p = 2`.** The gate closes iff the singular suppression (`λ/β²`) beats the susceptibility
  divergence in the `β→∞` limit.

## 6. Conclusion — the conditional, stated as exactly what it is
The genuine-side metric-curvature diagnostic is closed for a singular posterior **iff (a)** its criticality
co-locates with the singular limit `β→∞` **and (b)** `χ = O(β²)` there (`p ≤ 2`). **Otherwise — finite-`β`
transition, or `p > 2` — the genuine signature survives on the singular net** (and the sampler difficulty
becomes the binding wall again). This reduces the genuine side to two decidable questions:
1. Is the relevant transition at finite `β`, or co-located with `β→∞`?
2. If co-located, is `χ` sub-quadratic in `β` (`p ≤ 2`)?
Both are pencil-decidable *given a model for `χ(β)` on the singular set* — which is the next derivation, not
a sampler.

## 7. Corrections folded in (no-silent-edit)
- **v0 over-claim corrected.** The first pass called the gate "closed under the generic-singularity
  assumption." The §5 toy **refutes the unconditional form**: at finite-`β` transitions `det g → +∞` and the
  genuine signature survives. The corrected statement is the §6 rate conditional.
- **Parity (`g_βε`, and now `pxxy`, `pyyy`).** The control measured `⟨q⟩ = q* ≠ 0` (Edwards–Anderson order:
  thermal symmetry breaks even when disorder symmetry holds). The receipt's hard-coded `pxy = pxxy = pyyy = 0`
  rest on a `q → −q` parity that **fails in the glassy phase** — they are **near-critical approximations**,
  exact only where `g_εε → ∞` swamps them (which protects the receipt's `[V]` near `β_AT`, not a control at
  non-critical `β`). Any genuine certification must **measure all three ε-odd terms and inherit none of the
  zeros.**

## 8. Implication (the terminus)
No sampler was ever the bottleneck — the **gate analysis** is. The diagnostic provably `[V]`-classifies the
**non-singular** archetypes (perceptron, SK); the **singular real-net** case routes to the §5–§6 rate
competition, and where it closes, the live invariant is the **LLC/RLCT** — *not* a Ruppeiner curvature, a
different object the field already uses for the right reason. That is the honest "novel for AI" claim: the
curvature diagnostic classifies the non-singular archetypes, and the singular case is a decidable rate
competition that — where it closes — hands off to the LLC.

**Footnote (weaker, not load-bearing).** There is a prior question of whether a real net's posterior even
*has* the continuous-RSB/replicon transition the genuine signature detects. If it does not, `χ` never
diverges and the genuine side is vacuous for real nets regardless of `det g` — a cleaner closure, but one
that cannot be asserted without evidence (some work argues nets are glassy; unsettled). [v0.2: this
footnote is **promoted to load-bearing** — see §9.]

---

## 9. Pencil pass 2 (v0.2) — locating the χ-divergence: the rate competition is branch-2-only

The §5 race tacitly assumed the singular suppression (`g_ββ ~ λ/β²`, a **β→∞** effect) and the genuine
divergence (`χ → ∞`) compete at the *same* `β`. **They generically do not.** `χ = N·Var(q)` diverges
**only at a continuous-RSB (de Almeida–Thouless) transition**; everywhere else it is finite. So the gate is
decided not by a rate but by **where that transition sits relative to the β→∞ singular limit** — the
co-location question. Three branches:

1. **Finite-β RSB transition** (`β_AT < ∞`; the spherical-perceptron `[V]` case). There `g_ββ = O(1)` is
   finite while `χ → ∞`, so `det g → +∞` — the **genuine signature SURVIVES, gate OPEN.** The singular
   suppression is a β→∞ effect and **never reaches a finite-β transition**; the §5 race does not apply.
   (Cost: the sampler is then needed after all — the equilibration wall returns.)
2. **Zero-temperature (β→∞) RSB transition** — a genuine glass transition *co-located* with the singular
   limit, `χ ~ β^p`. **This is the only branch where the §5 race bites:** CLOSED (`det g → 0`/indefinite,
   F-spurious) iff `p ≤ 2`, OPEN iff `p > 2`.
3. **No RSB transition** (the minima form a connected manifold; mode connectivity, single state). Two
   replicas spread over the manifold, so `χ` saturates to a finite, β-independent constant (the manifold's
   overlap variance) — **never diverges.** The genuine signal is **VACUOUS**; the gate is closed for
   *absence of criticality*, not by suppression.

**Refined verdict.** The v0.1 rate conditional is the **branch-2 special case.** For a general singular
posterior the binding question is not a rate but a **LOCATION/STRUCTURE**: does it have a finite-β RSB
transition (branch 1 → open, signature survives), a zero-T glass transition (branch 2 → the `p` vs `2`
race), or no RSB at all (branch 3 → vacuous)? The spherical perceptron is **branch 1** — which is exactly
why its genuine result is `[V]` and a control was even conceivable. Whether a real net's posterior is
branch 1, 2, or 3 is the **spin-glass-of-nets** question: mode-connectivity evidence leans **branch 3**
(vacuous), some glassiness work leans 1/2 — **UNSETTLED**, and decidable only with evidence on net-posterior
*structure*, not with a curvature computation and not with a sampler for this program's object.

**Net.** The singular suppression is generically the **wrong tool** to close the gate — it lives at β→∞,
where (branches 1 and 3) there is no divergence to suppress. The gate's real content is **which branch** —
i.e. the RSB structure of the net posterior — a sharper, more falsifiable terminus than the rate
conditional alone, and one that routes the genuine side to an *empirical* question about loss landscapes
rather than to any sampler this program would build.
