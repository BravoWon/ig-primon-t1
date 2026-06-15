# IG-PRIMON-T1 — Perceptron Storage Archetype GATE (v0.4, 2026-06-14)

**Status: [GATE] CLOSED, genuine side [V] (susceptibility *and* curvature).** Analytic derivation + four
CPU-certified receipts (`module_L_perceptron_replica.py`, `module_L_perceptron_replicon.py`,
`module_L_perceptron_curvature.py`, `module_L_perceptron_finiteT.py`), no accelerator, no LLM. This is the
object-existence gate for the relocation chosen in `T1_moduleL_hw_gate_v0_1.md`: the genuine-criticality
side of the curvature dichotomy, shown *absent* on frozen LLM forward-pass overlaps, was relocated to a
**Gibbs posterior over weights** = the trunk's parked §6.7 (perceptron).

**v0.4 — curvature `[E]→[V]` closed (this revision).** `module_L_perceptron_finiteT.py` redoes the curvature
on the **finite-T `(β,ε)` manifold**, where β is a genuine natural field (`g_ββ=2φ''=2 Var(energy)/N > 0`,
verified) that *also* tunes the replicon (`λ_repl(β)→0` at a finite `β_AT=5.99`, at κ=−0.5, α=4.2). Both
coordinates are conjugate fields ⇒ the metric is a genuine covariance ⇒ **`det g>0` throughout and
diverging** (0.88→21.5), with **`|R|~χ²~1/(β_AT−β)²→∞`** (`|R|·(β_AT−β)²→11.8`). This is a diverging metric
**component** on a **non-degenerate, positive-definite** metric — the genuine side of the refined diagnostic
*in full*, the literal Riemannian twin of SK's `R_ε`. The v0.3 `[E]` (indefinite `(α,ε)` coords) is
**resolved to `[V]`**; the `(α,ε)` run is retained as the cautionary intermediate that the diagnostic
correctly flagged. Receipt-generating phase **complete**.

**v0.3 — the literal curvature (this revision).** Promotes the result from the susceptibility *proxy*
(`χ = 1/λ_repl`) to the **Ruppeiner scalar itself**: `module_L_perceptron_curvature.py` computes `R` on the
ε-augmented manifold and confirms **`|R| → ∞` scaling as `χ² ~ 1/(α_AT−α)²`** (`|R|·(α_AT−α)² → 0.155`,
engine == analytic leading form), driven entirely by the soft mode (`R = 0` when χ is frozen). This is the
literal-curvature twin of SK's `R_ε ~ χ_SG²` at dAT. **Honest caveat (§3b):** the load α is a *structural*
tuning coordinate, not a natural field, so the (α,ε) Hessian is **indefinite** (`det g<0`); `|det g|→∞`
rules out the spurious `det g→0` (Curie–Weiss) case, but the strictly **positive-definite** realization the
refined diagnostic prefers needs the **finite-T `(β,ε)` manifold** (β natural *and* tuning the replicon —
the SK gold standard). The claim is therefore registered on the program's stated **invariant** (`|R|`-
divergence, the convention Wall), with the positive-definite upgrade named as the next receipt.

**v0.2 — two corrections from review (2026-06-14).**
1. **Relabel of the genus (substantive).** `G_E` below is the log-volume of weights satisfying a *margin
   constraint on random patterns* — the **Gardner storage / constraint-satisfaction problem**, and the
   κ<0 instability is a **jamming / SAT-UNSAT** transition. This is *not* teacher–student generalization;
   "learning" was never the true subject (SK is not a learner either). The genus is **the geometry of
   continuous-RSB criticality vs. kinematic volume-divergence in disordered systems**; learning (ridge),
   spin glass (SK), and storage/jamming (here) are **venues**. The perceptron is a clean **third
   archetype** that carries *both* sides in one model (convex κ≥0 = kinematic; non-convex κ<0 = genuine).
2. **[E]→[V] closure (the headline now earns its tag).** The v0.1 "the method transfers to κ<0" was
   `[E]`/literature. It is now **[V]**: `module_L_perceptron_replicon.py` computes the spherical replicon
   eigenvalue (validated against Gardner — see §3) and exhibits `χ = 1/λ_repl` diverging *continuously* as
   `α→α_AT(κ)⁻` for κ<0 — the direct twin of SK's `g_εε ~ χ_SG ~ 1/λ_AT`.

The question §6.7 left open:

> Does the perceptron transition support the **SK-converse method** — augment with a field `ε` conjugate
> to the inter-replica overlap, stay replica-symmetric, *approach* (not cross) the de Almeida–Thouless
> instability so the overlap susceptibility `χ ∝ ∂²s/∂ε²` diverges continuously and `|R|→∞` is a
> theorem? Or is it **frozen-1RSB / first-order**, where the free energy jumps and there is no marginal
> line to ride?

**Verdict (derived + certified below): it splits by weight geometry.** §6.7's caution ("likely
frozen-1RSB / first-order") is **correct for Ising weights** and **superseded for the non-convex
spherical perceptron**, which *is* a clean continuous replicon and **[V] carries the method**. So §6.7
has a realization that works — a storage/jamming criticality, not a learning one.

---

## 1. The replicated free energy (the object §4 lacked, here present)

**Setup.** Weights `w ∈ ℝ^N` (spherical, `‖w‖²=N`) or `w ∈ {±1}^N` (Ising). `p = αN` random patterns
`ξ^μ`, labels `σ^μ`. Gardner volume of weight space storing all patterns at stability `κ`:

`V = ∫ dμ(w) ∏_{μ=1}^{p} Θ( σ^μ (w·ξ^μ)/√N − κ )`.

Quenched average by replicas, `(1/N)\overline{\ln V} = lim_{n→0} (\overline{V^n}−1)/(nN)`. Introduce the
inter-replica overlaps `q_{ab} = w^a·w^b/N` and their conjugates; the replica-symmetric (RS) ansatz
`q_{ab}=q (a≠b)` gives a free entropy that **factorizes into an entropic and an energetic term**:

`s(q) = G_S(q) + α G_E(q)`,  `G_E(q) = ∫ Dt · ln H(u)`,  `u = (κ − √q · t)/√(1−q)`,

with `Dt = e^{−t²/2}dt/√(2π)`, `H(x)=∫_x^∞ Dz = ½ erfc(x/√2)`. The **entropic** term carries the weight
geometry:

- **Spherical:** `G_S(q) = ½[ ln(1−q) + q/(1−q) ]` (log-volume of the sphere at fixed overlap).
- **Ising:** `G_S(q) = extr_{q̂}[ −½ q̂(1−q) + ∫ Dz · ln 2cosh(√q̂ · z) ]`, conjugate `q̂` from the
  Hubbard–Stratonovich of the discrete sum; saddle `q = ∫ Dz · tanh²(√q̂ z)`.

**This is a genuine exponential-family / thermodynamic object** — `G_S, G_E` are log-partition pieces, the
metric is their Hessian in the natural parameters, and the overlap `q` (and its conjugate field `ε`) enter
as **natural parameters**, not as measured moments. Requirements (R1)–(R3) of the §4 gate are therefore
**satisfied here**. This is the whole point of the relocation: the *Gibbs posterior over weights* (a
disordered-system storage problem) carries the object the *frozen forward pass* did not. The remaining
question is purely about the **nature of the transition**.

---

## 2. The regime test

The SK-converse method needs a **continuous replicon**: an AT line where the replicon eigenvalue
`λ_repl → 0` *continuously*, so the overlap susceptibility (response of `q` to its conjugate `ε`) diverges
continuously and can be approached from the RS side without crossing into RSB. The replicon (AT) condition,
derived from the second variation of `s[Q]` (entropic Hessian `−½/(1−q)²` vs. the energetic vertex) and
certified in §3, is **`λ_repl = 1 − α∫Dt[G₁′(u)]²`** with `G₁=H′/H` and `G₁′(u)=−uG₁−G₁²`; RS goes unstable
when the energetic kernel overwhelms the entropic stiffness. Two structural facts decide the outcome:

1. **Convexity (spherical, κ≥0).** The constraint set is convex in `w`; the RS solution is stable up to
   capacity — **no AT crossing**. The only transition is `q→1` at the Gardner capacity `α_c(κ)`: a
   **metric-volume** effect (the solution space shrinks to a point), not a replicon softening.
2. **Discreteness (Ising) / negative margin (spherical κ<0).** Ising weights cannot interpolate, so the
   solution space **fragments and freezes** (q jumps toward 1 in the dominant cluster) — frozen-1RSB,
   discontinuous. Negative margin (κ<0) makes the spherical problem **non-convex**, opening a *continuous*
   full-RSB AT line below capacity.

---

## 3. Certified results (`module_L_perceptron_replica.py`, run 2026-06-14)

**Anchors (pin the algebra) — both exact:**
- Spherical Gardner capacity `α_c(κ) = [∫_{−κ}^∞ Dt (t+κ)²]^{−1}`; **`α_c(0) = 2.0` exact** (mpmath, |err|=0).
- Ising RS free entropy **`s(0) = ln 2` exact** (|err| = 0).

**[V] Spherical, convex (κ ≥ 0).** RS saddle `q*(α)` at κ=0 rises **smoothly** to 1 with **no jump**:

| α | 0.5 | 1.0 | 1.5 | 1.8 | 1.95 | 1.99 |
|---|---|---|---|---|---|---|
| q* | 0.307 | 0.585 | 0.822 | 0.938 | 0.986 | 0.991 |

`q→1` continuously as `α→α_c=2`. Convex ⇒ RS-exact ⇒ **no replicon instability**. In Result C's language
this is the **kinematic** side: a volume divergence (solution space → point), the storage analog of double
descent — bounded/flat curvature, not genuine criticality.

**[V] Ising (κ = 0).** RS free entropy goes **negative**:

| α | 0.2 | 0.4 | 0.6 | 0.8 | 0.83 | 1.0 | 1.2 |
|---|---|---|---|---|---|---|---|
| s | +0.550 | +0.396 | +0.228 | +0.035 | +0.003 | −0.204 | −0.572 |

Brentq zero-crossing: **`α_RS = 0.83308`**, coinciding with the **Krauth–Mézard frozen-1RSB capacity
`α_c = 0.833`** [literature] to four figures. A *negative quenched entropy for discrete weights is
impossible* (it counts a finite set), so RS must break — and it breaks by **freezing**: the entropy hits
zero *at* the transition because the dominant cluster collapses to `q→1` **discontinuously**. There is **no
marginal replicon line to ride**. (Annealed bound `α ≤ 1` at κ=0, also reproduced, brackets it from above.)

**[V] Spherical, non-convex (κ < 0) — now certified (`module_L_perceptron_replicon.py`).** The replicon
eigenvalue `λ_repl = 1 − α∫Dt[G₁′(u)]²` (with `G₁=H′/H`, `G₁′=−uG₁−G₁²`, q the RS saddle) is **validated
against Gardner**: at κ=0 it → 0 *continuously* exactly at α=2=α_c (the AT line meets capacity —
`λ_repl = 0.60, 0.39, 0.16, 0.050, 0.0081` at α = 1.0, 1.5, 1.9, 1.99, 1.999). For κ<0 it crosses zero
**before capacity, continuously**:

| κ | α_AT | α_c | q at AT |
|---|---|---|---|
| −0.1 | 2.322 | 2.353 | 0.993 |
| −0.2 | 2.669 | 2.783 | 0.978 |
| −0.3 | 3.048 | 3.311 | 0.955 |
| −0.5 | 3.951 | 4.770 | 0.896 |

and the **SG susceptibility `χ = 1/λ_repl` diverges continuously** as `α→α_AT⁻` (κ=−0.5): `χ = 16.7, 32.8,
65.0, 161, 322, 644` at `α_AT−α = 0.20 … 0.005`, with `χ·(α_AT−α) → 3.22` (a clean `1/(α_AT−α)` power law,
no jump). This is the **direct twin of SK's `g_εε ~ χ_SG ~ 1/λ_AT`** (`module_L_SK_converse.py`): a genuine
continuous replicon on the RS side. The SK-converse "approach, don't cross" strategy transfers — **`[V]`,
not literature.**

**[E / literature] Ising teacher–student learning.** The learning (generalization) transition is
**first-order**: generalization error jumps discontinuously to zero at `α ≈ 1.245` (Györgyi 1990;
Sompolinsky–Tishby–Seung 1990) — the textbook first-order learning transition, again *not* a continuous
replicon.

---

## 3b. The curvature itself — report `R`, do not assume it (`module_L_perceptron_curvature.py`)

`R` is the one quantity in this program that cannot be pre-committed: **a diverging susceptibility does not
hand you `|R|→∞`** — the convex side of *this same model* (κ≥0, q→1 at capacity) is the standing proof, a
divergence with `R` bounded (kinematic, the storage twin of double descent). So the κ<0 run *decides* `R`.
Run against the trunk's own refined diagnostic (three questions; only "diverges + positive-definite
component blow-up" earns the genuine `[V]`).

Construction: on the RS side the overlap sector is a **Gaussian soft mode**, so
`ψ(α,ε) = 2 s(α) + χ(α)·ε²/2 + O(ε⁴)`, `χ=1/λ_repl` (§3). At ε=0 the engine `R=−N/(2 det g²)` takes
`g_εε=χ` (singular), `g_αα=2s''` (regular), `∂_α g_εε ~ χ²` (the one singular 3rd derivative); leading order
`R ≈ (λ')²χ²/(4 s'')`. Receipt (κ=−0.5, α_AT=3.951):

| α_AT−α | χ=1/λ_repl | det g | R (engine) | \|R\|·(α_AT−α)² | R (analytic) |
|---|---|---|---|---|---|
| 0.40 | 8.6 | −12.2 | −0.553 | 0.088 | −1.97 |
| 0.10 | 32.8 | −83.2 | −14.65 | 0.147 | −18.98 |
| 0.02 | 161.5 | −493.6 | −385.6 | 0.154 | −405.6 |
| 0.01 | 322.3 | −1009.7 | −1549.9 | 0.155 | −1589.4 |

**Q1 — diverge, or cancel like the convex side?** **Diverges.** `|R|·(α_AT−α)² → 0.155` (nonzero constant)
⇒ `|R| ~ 1/(α_AT−α)² ~ χ²`, engine matching the analytic leading form. The singular powers do **not** cancel
— decisively unlike the convex side (where they do). `[V]`.

**Q3 — component effect or volume effect?** **Component.** Freezing χ gives `R=0`, so the divergence is
driven *entirely* by the soft mode `g_εε=χ→∞`; the volume effect (`q→1`) is the *separate* convex-side
mechanism and sits at α_c (=4.77 here), not at α_AT (=3.95), so it is not even active at the AT line. `[V]`.

**Q2 — positive-definite component blow-up, or det g→0 / indefinite?** **Neither cleanly — this is the
honest gap.** `|det g|→∞` (not the `det g→0` collapse), so it is **not** the spurious Curie–Weiss
degeneracy. But `det g<0`: the (α,ε) Hessian is **indefinite**, because the load α is a *structural* tuning
coordinate, not a natural field (`g_αα=2s''<0`, the free entropy is concave). So `R` is registered on the
program's stated **invariant** (`|R|`-divergence; sign/signature convention-dependent — the convention
Wall), with the strictly **positive-definite** gold standard (SK's `det g=β²/λ>0`) **not** established here.
It requires the **finite-T `(β,ε)` manifold** (β natural, `g_ββ>0`, *and* tuning the replicon).

**Bookkeeping (two claims, not one).** The "genuine side" tangles two separate statements, and they get
different tags. **Susceptibility-genuine is `[V]`** (§3): `g_εε ~ 1/λ_repl` is one real conjugate-field
susceptibility (ε *is* a natural field), its divergence is coordinate-independent, Gardner-anchored, and
component-driven — the perceptron sits on the genuine side *by the susceptibility criterion*, full stop.
**Curvature-genuine is now `[V]`** — *via the positive-definite manifold, not the indefinite one*. `|R|~χ²`
holds on both, but `|R| ~ correlation volume` is **Riemannian**, so it needs a positive-definite metric.
The `(α,ε)` run (Q1/Q3 above) gives the divergence on an *indefinite* metric (`det g<0`, α structural) — a
cautionary intermediate, retained as such, not promotable on its own. The **finite-T `(β,ε)` run**
(`module_L_perceptron_finiteT.py`) supplies the positive-definite metric: β is a genuine natural field
(`g_ββ=2φ''>0`, verified) that tunes the replicon, so `det g>0` throughout and diverging, with
`|R|~χ²~1/(β_AT−β)²→∞` (`|R|·(β_AT−β)²→11.8`, κ=−0.5, α=4.2, β_AT=5.99). Diverging component on a
non-degenerate positive-definite metric — the refined diagnostic's genuine side in full. **`[E]→[V]`.**

---

## 4. Verdict — the dichotomy across disordered-system venues

The genus is **continuous-RSB criticality vs. kinematic volume-divergence in disordered systems**. The
perceptron *storage* problem is the third archetype (after SK spin-glass and ridge learning), and it
carries **both** sides:

| disordered system (venue) | transition | dichotomy class | SK-converse method |
|---|---|---|---|
| **spherical perceptron, convex (κ≥0)** — storage/SAT | `q→1` at capacity (volume → point) | **kinematic** (volume divergence) | n/a — no criticality (like ridge double descent) |
| **spherical perceptron, non-convex (κ<0)** — jamming | **continuous full-RSB**, replicon→0 on AT line | **genuine criticality** `[V]` | **TRANSFERS** — `χ=1/λ_repl→∞`, `\|R\|→∞` |
| **Ising perceptron (storage)** | **frozen-1RSB**, q jumps (`α_RS=0.833`) | discontinuous (freezing) | **FAILS** — no marginal line |
| **Ising perceptron (learning)** | **first-order** (gen-error jumps, α≈1.245) | discontinuous | **FAILS** — free energy jumps |

**The realization that works is the non-convex spherical perceptron — a storage/jamming criticality, not a
learning one.** It is a genuine disordered-system manifold (a Gibbs posterior over weights — not a frozen
forward pass, so INFERENCE ≠ LEARNING is respected); it has a continuous replicon AT line that is the exact
analog of the SK dAT instability; and the SK-converse machinery (`module_L_SK_converse.py`) carries over
verbatim — augment with `ε` conjugate to the overlap, approach the AT line, and both `χ = 1/λ_repl` (§3) and
the Ruppeiner scalar `|R| ~ χ²` (§3b) diverge continuously, the latter on a **positive-definite** metric
(finite-T `(β,ε)`, `det g>0`). This is the divergent branch's realization the program wanted — **reachable
without RSB, exactly as SK was, and now `[V]`-certified** (susceptibility *and* curvature), the literal
Riemannian twin of SK's `R_ε`.

**Honest scope.** This is the Gardner *storage / constraint-satisfaction* problem, not teacher–student
generalization — "learning" is a venue label that overreached. The correct statement is sharper: the
curvature dichotomy's genuine side is realized at a **continuous jamming/SAT-UNSAT transition**. §6.7's
instinct was right about Ising (frozen-1RSB / first-order — method fails) and is superseded for the
spherical case: continuous weights with negative margin dissolve the obstruction.

---

## 5. Status of each claim

- **[V] (`module_L_perceptron_replica.py`):** the object exists (replicated RS free energy, exponential-
  family metric); the spherical-convex transition is a continuous volume effect (kinematic); the Ising
  transition is a discontinuous freezing (frozen-1RSB, `α_RS=0.83308`). Anchored by `α_c(0)=2` and
  `s(0)=ln2`, both exact.
- **[V] (`module_L_perceptron_replicon.py`) — closed this revision:** the non-convex spherical
  continuous-replicon claim. The replicon eigenvalue `λ_repl = 1−α∫Dt[G₁′(u)]²` is validated against
  Gardner (`λ_repl→0` at α=2=α_c for κ=0), crosses zero continuously at `α_AT(κ)<α_c(κ)` for κ<0, and the
  susceptibility `χ=1/λ_repl` diverges as `χ·(α_AT−α)→3.22` (κ=−0.5) — the direct twin of SK's
  `g_εε ~ χ_SG ~ 1/λ_AT`. The earlier `[E]` is **resolved to `[V]`**; no residual extrapolation remains in
  the regime verdict.
- **Curvature-genuine `[V]` (`module_L_perceptron_finiteT.py`) — §3b:** on the positive-definite finite-T
  `(β,ε)` manifold, `det g>0` throughout and diverging, `|R|~χ²~1/(β_AT−β)²→∞` (`|R|·(β_AT−β)²→11.8`) — a
  diverging **component** on a non-degenerate positive-definite metric, the literal Riemannian twin of SK's
  `R_ε`. Self-checks: `λ_repl→1` at β=0, `→` storage value at β→∞, `g_ββ=2φ''>0`, finite `β_AT=5.99`. The
  indefinite `(α,ε)` first pass (`module_L_perceptron_curvature.py`) is retained as the cautionary
  intermediate the diagnostic correctly flagged.
- **[E / literature, not load-bearing]:** Ising teacher–student *learning* is first-order at α≈1.245
  (Györgyi 1990) — cited as a venue contrast, not relied upon by the certified result.

**Where this stops.** The capstone is now closed at `[V]` (susceptibility *and* curvature, positive-definite),
and the receipt-generating phase is **complete**: no further archetypes, no new manifolds, no remaining
`[E]→[V]` debts on Result C. The genuine side *is* the literal curvature statement, verified. The value from
here is consolidation only — the registration acts (§6) and the manuscripts — not a fifth receipt.

---

## 6. Consequence for the trunk

`T1_consolidated_results_ledger_v0_3.md` §6.7 should be amended (ledger v0.4, versioned diff) to record
both corrections: **(1) the relabel** — Result C / Paper 3's "geometry of *learning* transitions" overreached;
the genus is **continuous-RSB criticality vs. kinematic volume-divergence in disordered systems**, with
learning, spin glass, and storage/jamming as venues; and **(2) the closed result** — the perceptron *storage*
problem is the third archetype, carrying both sides (convex κ≥0 = kinematic; non-convex κ<0 = genuine
continuous replicon, `[V]`-certified here, `χ=1/λ_repl→∞` at the jamming/AT line). The Ising perceptron is
frozen-1RSB/first-order (method fails); the divergent branch's realization is the **non-convex spherical
perceptron**, a storage/jamming criticality. This is a *strengthening* of Result C, not a correction of any
existing `[V]`. (Not edited here — a trunk amendment, and a paper retitle, are registration acts for the
user to sign.)

---

## 7. Changelog

**v0.1 (2026-06-14).** Regime gate executed. Derived the replicated RS free energy (spherical & Ising),
confirmed it is a genuine exponential-family object (the §4 relocation is well-founded), classified the
transition by weight geometry. Certified (`module_L_perceptron_replica.py`): spherical `α_c(0)=2` exact;
spherical `q*→1` continuous (convex/kinematic); Ising `s(0)=ln2` exact and RS entropy negative with
zero-crossing `α_RS=0.83308 ≈` Krauth–Mézard 0.833 (frozen-1RSB). Non-convex spherical continuous-replicon
left as `[E]`/literature.

**v0.2 (2026-06-14).** Two corrections from review. **(1) Relabel:** the genus is *continuous-RSB
criticality vs. kinematic volume-divergence in disordered systems* (not "learning"); the perceptron is the
**storage/jamming** third archetype carrying both sides; the κ<0 instability is a jamming/SAT-UNSAT
transition, not generalization (`INFERENCE ≠ LEARNING` honored, and "learning" itself demoted to a venue
label). **(2) `[E]→[V]` closure:** added `module_L_perceptron_replicon.py` — spherical replicon eigenvalue
validated against Gardner (`λ_repl→0` at α=2, κ=0), `α_AT(κ)<α_c(κ)` continuous for κ=−0.1…−0.5,
`χ=1/λ_repl` diverging as `χ·(α_AT−α)→3.22` (κ=−0.5) — the perceptron twin of `module_L_SK_converse.py`.
The genuine-side claim is now `[V]`. Title and §§3–6 updated; no trunk `[V]` altered. Recommends ledger
v0.4 (relabel + third archetype) and a Paper 3 retitle — both deferred to the user as registration acts.

**v0.3 (2026-06-14).** The literal curvature, run report-don't-confirm (§3b, `module_L_perceptron_curvature.py`).
Against the trunk's three-question refined diagnostic: **Q1 diverges** (`|R|~χ²~1/(α_AT−α)²`,
`|R|·(α_AT−α)²→0.155`, engine == analytic — *not* the convex-side cancellation), **Q3 component** (freeze
χ → R=0; volume effect sits at α_c, not α_AT) — both `[V]`; **Q2 positive-definite NOT clean** — `|det g|→∞`
rules out the spurious `det g→0` collapse, but `det g<0` (indefinite) because the load α is structural, so
`R` is registered on the program's `|R|`-divergence **invariant**, with the strictly positive-definite
`(β,ε)` finite-T realization **named, not pulled**. Declared the receipt phase complete pending the
positive-definite signature. No trunk `[V]` altered.

**v0.4 (2026-06-14).** Curvature-genuine `[E]→[V]` **closed** (`module_L_perceptron_finiteT.py`). The v0.3
`[E]` rested on indefinite `(α,ε)` coordinates (α structural, `g_αα<0`); redone on the finite-T `(β,ε)`
manifold where β is a genuine natural field (`g_ββ=2φ''=2Var(energy)/N>0`, verified) that also tunes the
replicon (`λ_repl(β)→0` at finite `β_AT=5.99`; κ=−0.5, α=4.2 in the finite-T RSB window). Result: **`det g>0`
throughout and diverging** (0.88→21.5), **`|R|~χ²~1/(β_AT−β)²→∞`** (`|R|·(β_AT−β)²→11.8`) — a diverging
component on a non-degenerate positive-definite metric, the refined diagnostic's genuine side in full and the
literal Riemannian twin of SK's `R_ε`. Self-checks: `λ_repl→1` at β=0, `→` storage value at β→∞. The `(α,ε)`
run is retained as the cautionary intermediate the diagnostic correctly flagged. §§title, header, 3b, 4–7 and
Paper 3 §5 updated; `T1_resultC_amendment_v0_4.md` updated to curvature-`[V]`. **Receipt-generating phase
complete** — capstone closed; remaining value is consolidation (ledger v0.4, manuscripts). No trunk `[V]`
altered.

— End of perceptron storage archetype gate v0.4. Amendments require a versioned diff; silent edits void the registration.
