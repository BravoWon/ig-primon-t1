# IG-PRIMON-T1 — Consolidated Results Ledger (v0.4, 2026-06-14)

**Purpose.** The trunk. One place that states what the program has *established*, with receipts and
honest status tags, the scope walls that bound it, and the two write-ups it feeds. This is not the
pre-registration (the frozen-prediction document, `T1_preregistration_v0_6.md`); this is the
results synthesis — what is now known, verified, and defensible.

**Discipline carried throughout.** HONEST_CLAIMS: **[V]** verified, **[E]** defensible
extrapolation, **[C]** conjecture; **gate** (derive before numerics); no-silent-edit on registered
items. Sign conventions stated where they bite.

**v0.4 (this revision).** **Result C relabel + §6.7 closed (third archetype).** Two corrections, no existing
`[V]` altered (folds `T1_resultC_amendment_v0_4.md`). (1) **Relabel of the genus:** Result C / Paper 3's
"geometry of *learning* transitions" overreached — the genuine side is not learning (SK is a spin glass; the
new result is constraint-satisfaction). The genus is **the geometry of continuous-RSB criticality vs.
kinematic volume-divergence in disordered systems**; learning (ridge), spin glass (SK), and storage/jamming
(perceptron) are *venues*. Paper 3 retitled (§5). (2) **§6.7 closed — the perceptron storage problem is the
third archetype, carrying *both* sides:** convex (κ≥0) storage = kinematic (`q→1` at the Gardner capacity,
`α_c(0)=2` exact); Ising = frozen-1RSB/first-order (RS entropy negative, `α_RS=0.833` = Krauth–Mézard;
method fails, as feared); **non-convex spherical (κ<0) = continuous replicon, carries the SK-converse method
[V]** — replicon `λ_repl=1−α∫Dt[G₁′]²` (Gardner-anchored: `λ_repl→0` at α=2, κ=0), `α_AT(κ)<α_c(κ)`
continuous, susceptibility `χ=1/λ_repl→∞` (`χ·(α_AT−α)→3.22`), and the literal Ruppeiner scalar `|R|~χ²→∞`
on a **positive-definite** finite-T `(β,ε)` metric (`det g>0`, `g_ββ>0`; `|R|·(β_AT−β)²→11.8`) — the literal
twin of SK's `R_ε`. Receipts (§7): `module_L_perceptron_{replica,replicon,curvature,finiteT}.py`; gate
`T1_moduleL_perceptron_gate_v0_1.md`. The `(α,ε)` indefinite first pass is retained as the cautionary
intermediate the refined diagnostic correctly flagged. Results A and B unchanged.

**v0.3 (prior).** **Closes the §6.6 converse test.** Result C (§3) now demonstrates *both* sides of the kinematic-vs-interacting dichotomy on exactly-solvable models — the genuine-criticality side via the **Sherrington–Kirkpatrick spin glass at its de Almeida–Thouless instability** (stay replica-symmetric, approach the dAT line, no RSB ansatz). Adds the **four-tier diagnostic taxonomy** (linear-learning *kinematic* / Curie–Weiss *degenerate* / bare-SK *blind* / ε-augmented-SK *genuine*), the **dAT-line confirmation** (`g_εε ~ χ_SG ~ 1/λ_AT`, replicon-tracked across the `(T,h)` plane, by implicit differentiation), the **ML translation** (the geometric coordinate `ε` = representation-overlap between two noisy forward passes), and a **finite-difference methodological note** (3rd-derivative `R` breaks down near critical saddles; reliance on the `h=0` analytic anchor and the linear-response metric component). Registers receipt `module_L_SK_converse.py` (§7); §6.6 → **CLOSED**, with the first-order / Ising-perceptron realization spun out to §6.7. Results A and B unchanged.

**v0.2 (prior).** Adds **Result C / Module L** — geometric diagnostics of learning transitions (§3); records its write-up target (Paper 3, §5) and the non-Gaussian converse test as a parked branch (§6.6); registers the receipt `module_L_ridge_curvature.py` (§7). Results A and B are substantively unchanged; v0.1 §§3–6 renumber to §§4–7, and section cross-references are updated to match — both `(§3)→(§4)` pointers (open branches; file index) and the Paper 2 open-item line (whose general-rate blocker had already closed in v0.1).

**The object.** The canonical primon (Riemann-gas) and its generalizations: a grand-canonical gas
whose partition function is a zeta / Dirichlet series, studied via its **Fisher / information
geometry on the real axis near the Hagedorn point β = 1**. Two results are complete on this object; both live on
the real temperature axis, neither touches the complex zeros. A third result (Result C, §3) carries the curvature machinery of Result B *off* the primon gas onto the statistical manifold of learning — a taxonomic boundary between arithmetic-gas and learning manifolds.

---

## 0. Walls (stated up front, because they bound everything below)

- **NO-RH.** Nothing in this program bears on the *location* of the zeros of any zeta or
  L-function. Every result is geometry of the partition function on the real `(β, log z)` plane;
  the zeros enter only as data read off that geometry, never as a target constrained by it. This is
  not modesty — it is the structural ceiling, and it is load-bearing in both results below.
- **ZFD-conditionality (Siegel wall).** The arithmetic dictionary's *general* statement is
  conditional on a per-field hypothesis ZFD(K) — no nontrivial zero of ζ_K inside the relevant disk.
  This is **not** implied by GRH and would follow only from a no-Siegel-zero theorem plus a uniform
  low-lying-zero bound, neither available. The program does not move this wall; it states results
  unconditionally where it can and flags ZFD where it cannot.
- **Convention-dependence.** The Ruppeiner scalar curvature's *sign* is convention-dependent
  (sphere-positive here); only `|R|`-vanishing vs `|R|`-divergence is invariant. Claims are phrased
  on the invariant.
- **Framing vs theorem.** Result A (the dictionary) is **framing-class**: every arithmetic fact it
  uses is classical; the contribution is the information-geometric *reading*. Result B (the
  curvature dichotomy) is a **genuine theorem** about a class, built on the classical Ruppeiner
  machinery. The distinction is kept explicit and not blurred.

---

## 1. Result A — the real-axis arithmetic dictionary (Module E)

**Claim.** The Fisher expansion of the K-primon gas about the Hagedorn point,
`I_K(1+ε) = ∂²_β log ζ_K(1+ε)`, reads the **leading arithmetic of the number field K** off the
structure of its nearest singularities and its constant term — on the real axis, with no complex
continuation. Write `G(ε) = log[ε·ζ_K(1+ε)]` (analytic at ε=0), `b_k = [ε^k]G`.

The dictionary, four entries:

| invariant read | from | mechanism | status |
|---|---|---|---|
| **unit rank** `ρ = r₁+r₂−1` | radius of `{b_k}` / leading amplitude `\|b_k\|·k·R^k → m` | `ord_{s=0} ζ_K = ρ` (binding zero at distance 1) | [V] for tested fields; [E] general under ZFD |
| **signature** `(r₁, r₂)` | order of the *next* singularity (the `r₂/2^k` tail) | `ord_{s=−1} ζ_K = r₂` | [V] (seen in ℚ(∛2) residual) |
| **`hR/w`** (class no. × regulator / roots of unity) | sub-leading log-singularity amplitude `A = exp H(−1)` | `ζ_K^*(0) = −hR/w` (class number formula at 0) | [V], 13 digits |
| **discriminant** `\|d_K\|` | constant term ÷ sub-leading amplitude, `Res/A` | `Res_{s=1}/A = 2^{r₁}(2π)^{r₂}/√\|d_K\|` | [V], exact |

**The rank-reading law (Lemma E.1, v0.2).** Under ZFD(K), with binding zero at ε₀ of order m,
`|b_k|·k·R^k → m`, `R = |ε₀|` the radius. Specializations: `ρ ≥ 1 ⟹ R = 1, m = ρ` (reads the exact
unit rank); rank 0 → ℚ: R = 3, imaginary quadratic: R = 2. **Biconditional, honestly stated:**
`radius = 1 ⟹ ρ ≥ 1` (unconditional, modulo standard zero-free regions); `ρ ≥ 1 ⟹ radius ≤ 1`, with
**equality iff ZFD(K)**. Failure modes if ZFD breaks: a Siegel real zero β₀ binds at `1−β₀ < 1`
(ratio `→ 1/(1−β₀) ≠ 1`), or a low-lying complex zero inside the disk. Proof: Hecke's completed
Λ_K, Γ-factor pole orders give the trivial-zero structure; log-singularity expansion gives the
amplitude law. (`T1_lemma_E1_proof_v0_2.md`.)

**Verified instances** [V] — `|b_k|·k·R^k → m` to 6–13 s.f., order cross-checked by
`ζ_K(2h)/ζ_K(h) → 2^m`:

| field | signature | rank | m read | route |
|---|---|---|---|---|
| ℚ | (1,0) | 0 | R=3, m=1 | ζ |
| ℚ(√−3) | (0,1) | 0 | R=2, m=1 | ζ·L(χ_{−3}) |
| ℚ(√5), ℚ(√2) | (2,0) | 1 | R=1, m→1 | ζ·L(χ), Hurwitz |
| cyclic cubic, cond. 7 | (3,0) | 2 | R=1, m→2 | ζ·L(χ)·L(χ²), order-3 χ |
| **ℚ(∛2)** | **(1,1)** | **1** | **R=1, m→1** | **degree-2 S₃ Artin L (AFE)** |

The ℚ(∛2) case (this session) **closes the last open HE.2 obligation** — the (1,1) mixed cubic that
needs the non-abelian Artin L-function. Built from first principles, gate-then-numerics: disc K =
−108, resolvent ℚ(√−3), conductor 𝔣 = (6) with N(𝔣) = 36 (conductor-discriminant), seed
ψ(𝔭) = cubic residue symbol (verified 20/20 primes < 80 against root-counting of x³−2 mod p), root
number ε = +1 (derived without local Gauss sums: Dedekind ζ_K and ζ both have root number +1 by
Hecke, so W(ρ) = +1; cross-checked by induction W(ρ) = W(ψ)·λ(k/ℚ)). AFE validated by
theta-modularity (1e−31), Dirichlet match, and a wrong-ε control that shatters the match. Result:
`|b_k|·k → 1` (1.00012507 by k=13), residual exactly `r₂/2^k = 1/2¹³` — the next trivial zero
(`ord_{s=−1} = r₂ = 1`, distance 2) bleeding in. So the dictionary's signature-reading and
rank-reading are both confirmed through a non-abelian L-function.

**hR/w reading** [V], this session — sub-leading amplitude `A = exp(H(−1))`, `H = G − m log(1+ε)`:

```
ℚ(√5):  A = 0.24060591253  vs  hR/w = log(golden ratio)/2 = 0.24060591253   (13 digits)
ℚ(√2):  A = 0.44068679351  vs  hR/w = log(1+√2)/2        = 0.44068679351   (13 digits)
```

**discriminant reading** [V], this session — `|d_K| = (2^{r₁}(2π)^{r₂}·A/Res)²`:

```
ℚ(√5):  |d_K| read = 5.0   (true 5)
ℚ(√2):  |d_K| read = 8.0   (true 8)
```

**Contribution class: framing.** `ord_{s=0} ζ_K = ρ`, `ζ_K^*(0) = −hR/w`, and the residue formula
are all classical. The contribution is the assembled reading: the prime gas's real-axis
thermodynamics encodes the field's signature, `hR/w`, and discriminant *simultaneously*, off one
Fisher expansion near β = 1. **Natural endpoint:** further sub-leading terms read only higher
derivatives of the special values — diminishing arithmetic interest, same framing class.

---

## 2. Result B — the Ruppeiner curvature dichotomy (Module C⁺)

**Theorem.** For grand-canonical generalized prime gases `log Ξ = Σ_{k≥1}(z^k/k)·𝒫(kβ)`, the
Ruppeiner scalar curvature at the transition is decided by where the singularity lives:
**temperature-driven (Hagedorn) ⇒ R → 0** (asymptotically flat); **fugacity-driven (condensation)
⇒ |R| → ∞**. The discriminant is one algebraic condition: **R → 0 ⟺ the singular sector factorizes
as `e^y·φ(x)`** (no nontrivial fugacity–temperature coupling). Full statement, proof, and scope:
`T1_moduleC_universality_theorem_v0_1.md`.

**Proof spine.** (i) Lemma C.1 [V, certified]: `ψ_sing = e^y·φ(x) ⟹` row 3 of the curvature
numerator `N` equals row 1 identically ⟹ singular contribution to `N` vanishes at all orders.
(ii) k=1-only-singular [V under H1]: a temperature singularity is probed at `kβ ≈ k ≥ 2` for
`k ≥ 2` (analytic), so the only singular term is `z·𝒫(β) = e^y·φ(x)`. (iii) `N` is then sourced by
the finite analytic background while `det g` diverges; the ratio → 0.

**Four instances** [V], one engine (classical validates it):

| gas | type | curvature | rate / amplitude |
|---|---|---|---|
| classical ideal gas | product form | `R = 0` (10⁻³¹) | exact |
| primon gas (prime zeta, log sing.) | temp-driven | `R → 0⁺` | `~1/(L+κ)²`, Δ₃ ≈ 5.04518818501443, κ = 0.832503 |
| all-integer Bose gas (ζ−1, pole sing.) | temp-driven | `R → 0⁺` | `~ε²`, `R/ε² → Δ₃ = 5.693982` |
| ideal Bose gas (BEC) | fugacity-driven | `\|R\| → ∞` | `~τ^{−1/2}` |

Two flat instances with **different spectra and different singularity types** (primon: primes, log;
integer gas: all integers, pole), plus the divergent complement. Two of the four contain no primes
at all — this is what makes it statistical mechanics, not number theory.

**Proof status — now [V], general.** The cofactor reduction (Appendix A, `paper2_appendixA_…tex`)
derives the rate for an arbitrary leading singularity `𝒫(1+ε) ~ A·ε^{−α}`:
`R ~ [Δ₃(α+1)/(2A²)]·ε^{2α}` (pole order α > 0), and `R ~ Δ₃/(2(L+κ)²)` (logarithmic, the α→0
marginal case). Verified numerically α = 1, 2, 3 on a tunable-order family (R/ε^{2α} → Δ₃(α+1)/2)
plus the log (primon, amplitude 2.5226) and α=1 (integer gas, 5.694) cases. **Flatness is
unconditional within the class**: even if Δ₃ = 0 (non-physical), `R → 0` survives one power faster
(ε^{2α+1}); only the explicit *amplitude* needs Δ₃ ≠ 0, which is **automatic** when the spectral
coefficients are positive (`𝒫(k) = Σ_q q^{−k} > 0`). The prior [E] (general-rate claim) is
**resolved to [V]**; the dichotomy theorem stands without a residual extrapolation.

**Contribution class: genuine theorem.** A curvature dichotomy for a class, with the standard
Ruppeiner "|R| ~ correlation volume" recovered as the *complement* of the flat (Hagedorn) case.

---

## 3. Result C — the curvature dichotomy across disordered systems (Module L)

**Genus (relabel, v0.4).** Result C is **not** about learning per se. Its subject is **continuous-RSB
criticality vs. kinematic volume-divergence in disordered systems**; learning (ridge), spin glass (SK), and
storage/jamming (perceptron) are *venues*. The diagnostic is the object of study; the known phases of each
venue validate it. The four-tier taxonomy and both original anchors below are unchanged in content — only
the framing is corrected. (Genuine side now has **three** archetypes; see "Third archetype" below.)

**Claim.** The Ruppeiner machinery of Result B, carried *off* the primon gas onto a disordered-system
statistical manifold, separates a **kinematic / volume divergence** (e.g. double descent) from
**genuine interacting criticality**: the former leaves the scalar curvature bounded — or vanishing —
while the metric *volume* diverges; only the latter gives `|R| → ∞`. The double-descent peak is,
geometrically, a *fake* transition. **Both sides are now demonstrated on exactly-solvable models** —
the kinematic side on linear ridge regression (double descent, below), the genuine-criticality side
on the Sherrington–Kirkpatrick spin glass at its de Almeida–Thouless instability (the converse test,
§6.6, now closed; see "The converse" below). Receipts: `module_L_ridge_curvature.py`,
`module_L_SK_converse.py`.

**The manifold.** Gibbs measure over student weights of the linear (Gaussian) teacher–student /
ridge regression, `P(w) ∝ exp(−β E − βλ·½‖w‖²)`, `E = ½‖y − Xw/√N‖²`. Sufficient statistics
`(E, ½‖w‖²)`, natural parameters `(θ₁, θ₂) = (−β, −βλ)` — the faithful analog of the gas's
`(E, N) ↔ (−β, log z)`, with the ridge field λ in the **fugacity role**. `log Z` is an exact
Gaussian integral, diagonal in the eigenbasis of `M = XᵀX/N`; the metric and its third derivatives
are closed-form sums over the spectrum `{μᵢ}` and teacher projections `{dᵢ}`. The interpolation
threshold `α = P/N = 1` is the transition: the Marchenko–Pastur soft edge touches zero, the smallest
`μ` collapses, and as the ridge `λ → 0` the smallest `kᵢ = β(μᵢ+λ) → 0`.

**Verification** [V] — exact large-N limit; analytic Hessian validated against finite differences of
`log Z` (max rel. err **8.9e−6**); engine pinned `R = −1` on the normal family. High precision
**40 dps is required, not optional**: the metric collapses to rank-1, so float `det g` and the
curvature numerator lose ~11 digits to cancellation — at single precision the bounded `R` would
masquerade as a divergence. At β = 1, α = 1, ridge `λ → 0`:

| quantity | behavior as λ → 0 | reading |
|---|---|---|
| `det g` (metric volume) | **diverges `~ λ^{−3/2}`** | MP soft-edge fluctuation explosion |
| metric rank | **collapses to rank-1** (defect → 1.000) | one soft mode dominates |
| `R`, noiseless teacher | **`→ 0`, `~ λ`** (−2.0e−5 → −8.1e−10, λ = 1e−2…1e−6) | asymptotically flat |
| `R`, noisy teacher (σ = 0.5) | **bounded, `≈ −1×10⁻⁴`**, strictly negative; seed-robust | bounded floor, never diverges |

So the divergence is **entirely a metric-volume effect**; the scalar curvature is bounded (noisy) or
vanishes (noiseless). `R = −N/(2 det g²)` holds with `N` and `det g²` diverging at *matching* rates —
the **singular powers cancel exactly**, the same mechanism proved for the flat (Hagedorn) case in
Paper 2 (Lemma C.1), here operating on a **covariance matrix** rather than an arithmetic spectrum.
Mechanism, explicit: noiseless, the soft mode is a *free* log-det mode (signal weight
`d²_min ~ μ²_min → 0`) → flat, like the classical ideal-gas baseline; noise gives the soft mode a
*source* (`d²_min →` noise floor) → bounded-nonzero curvature. Neither diverges.

**Taxonomy — the four-tier diagnostic.** Linear learning's double descent falls in the **strictly
negative** (Gaussian / repulsive) curvature class — matching the normal family (`R = −1`) and the
exact small-N teacher–student perceptron (fraction `R < 0` = 1.00) — and is **distinct from the
Hagedorn point**, which approaches zero from the **positive** side (`R → 0⁺`): same dichotomy *side*
(non-divergent), opposite sign-class. With the converse now closed (below), the diagnostic's full
resolving power is a four-tier taxonomy. One convention throughout (sphere-positive):

| system / transition | metric `det g` | scalar `R` | verdict |
|---|---|---|---|
| ideal Bose gas (BEC), fugacity-driven | non-degenerate, diverging component | `\|R\| → ∞` | **genuine criticality** (positive-control anchor) |
| primon / integer gas (Hagedorn), temp-driven | regular | `→ 0⁺` | kinematic-flat (positive) |
| **linear learning** (double descent), λ→0 at α=1 | `→ ∞` (rank-1 collapse) | `→ 0⁻` / bounded `< 0` | **kinematic** — volume divergence, `R` flat-to-bounded |
| **Curie–Weiss**, dependent coords (`E ∝ M²`) | `→ 0` / indefinite | spurious `\|R\| → ∞` | **degenerate** — coordinate artifact, disqualified |
| **bare SK** `(β, h)`, across dAT | finite (indefinite at larger `h`) | bounded | **blind** — replicon transverse to `h`, misses the transition |
| **ε-augmented SK** `(β, ε)`, across dAT | `> 0`, diverging component | `\|R\| → ∞` | **genuine criticality** — `g_εε ~ χ_SG ~ 1/λ_AT` |

**The refined diagnostic.** `|R| → ∞` *alone* proves nothing — it must be a **diverging metric
*component* on a non-degenerate, positive-definite metric** (genuine: BEC, ε-SK), not a `det g → 0` /
indefinite degeneracy (spurious: Curie–Weiss). And the manifold must **carry the order parameter's
conjugate field as a coordinate**, or it is blind to the transition entirely (bare SK). Curie–Weiss
and bare-SK are the two cautionary mirrors of the double-descent result; BEC and ε-augmented SK are
the two genuine anchors.

**The converse — genuine interacting criticality (SK spin glass)** [V]. The Sherrington–Kirkpatrick
spin glass (`J = 1`, field `h`) populates the divergent side, closing §6.6. Strategy: stay
**replica-symmetric** and *approach* the de Almeida–Thouless (dAT) instability — never cross it, so no
RSB ansatz is needed. Augment the bare `(β, h)` manifold with a field `ε` conjugate to the
inter-replica overlap `O = Σ_i s¹_i s²_i` (two real replicas); the two-replica RS free energy reduces
to `2×` single-replica at `ε = 0` (verified, `|diff| = 0`). The `ε`-direction metric component is the
**spin-glass susceptibility**, `g_εε = β·∂p/∂ε = β χ_SG`, which diverges at dAT (replicon → 0).

- **`h = 0` — closed form.** `ψ₂(β,ε) = (β²/2)(1−p²) + log4 + log cosh(β²p + βε)`, `p = tanh(β²p+βε)`.
  With `λ ≡ 1−β²` (the replicon at `h=0`): `R_bare = 4/β²` (**bounded → blind**); `R_ε = 2/[β²λ²] → ∞`,
  i.e. `|R| ~ χ_SG²`, on `det g = β²/λ > 0`. Analytic, confirmed numerically.
- **`h ≠ 0` — along the dAT line.** No closed form; the coupled `(p,q)` saddle is solved on
  Gauss–Hermite, and `g_εε` is obtained by **implicit differentiation** (linear response — stable,
  unlike 3rd-derivative finite differences). It scales cleanly as `g_εε = β χ_SG ~ C(h)/λ_AT`, the
  constant stabilizing as `λ_AT → 0`: `h=0.3 → C ≈ 1.30`, `h=0.5 → C ≈ 1.07` (`χ_SG·λ_AT` flat to
  ~3 digits across `λ_AT = 0.3 … 0.02`). The divergence mechanism is exactly `det(I − M) → 0` — the
  two-replica saddle's stability matrix going singular, which **is** the replicon condition.
  `det g_ε > 0` and growing throughout; the bare `(β,h)` `R` stays bounded (`≈ 16` at `h=0.3`) —
  **blindness confirmed off-axis**. So `|R_ε| → ∞` **all along the dAT line**: the diagnostic tracks
  the replicon instability *wherever it sits* in the `(T,h)` plane, not the symmetric `h=0` point only.
- **ML translation.** The geometric coordinate `ε` is the **representation-overlap between two noisy
  forward passes** of a learner — the inter-replica overlap field made operational. The
  genuine-criticality direction the geometry needs is this overlap, not any uniform-field axis.
- **Methodological note** (HONEST_CLAIMS — friction recorded, not smoothed). The **finite-difference
  3rd-derivative `R` breaks down at `λ_AT ≤ 0.05`** (double precision on an `fsolve`'d saddle: `R`
  sign-flips, and the finite-diff `g_εε` itself diverges from the true value — `31` vs `97` at
  `λ_AT = 0.02`). The `|R| → ∞` claim therefore rests on the **`h=0` closed-form anchor** + the
  **stable linear-response metric component** (clean to `λ_AT = 0.02`) + the `R` growth where the
  3rd-diff is still reliable (`λ_AT ≳ 0.1`: `26 → 41 → 99`) — **not** on the edge `R` values, which are
  numerical artifacts. A deliberate warning for anyone computing Ruppeiner curvature numerically near a
  critical point. Receipt: `module_L_SK_converse.py`.

**Third archetype — the perceptron storage problem (v0.4, §6.7 closed)** [V]. The Gardner spherical
perceptron carries **both** sides of the dichotomy in one model, and supplies the literal-criticality
realization §6.7 sought. **Convex (κ≥0) storage = kinematic:** `q→1` at the Gardner capacity (`α_c(0)=2`
exact) — a volume effect, the storage twin of double descent. **Ising weights = frozen-1RSB/first-order:**
RS free entropy `s(0)=ln2` goes negative, zero-crossing `α_RS=0.83308 ≈` Krauth–Mézard 0.833; discontinuous
freezing, the SK-converse method fails (as §6.7 feared). **Non-convex (κ<0) = genuine continuous replicon:**
replicon `λ_repl=1−α∫Dt[G₁′(u)]²` (Gardner-anchored — vanishes at α=2 for κ=0), crosses zero continuously at
`α_AT(κ)<α_c(κ)`; the overlap susceptibility `χ=1/λ_repl→∞` (`χ·(α_AT−α)→3.22`, κ=−0.5) and the literal
Ruppeiner scalar `|R|~χ²~1/(β_AT−β)²→∞` on a **positive-definite** finite-T `(β,ε)` metric (`det g>0`,
`g_ββ=2 Var(E)/N>0`; `|R|·(β_AT−β)²→11.8`) — the literal twin of SK's `R_ε`, the genuine side of the refined
diagnostic in full. A first computation tuned by the structural load α gives the same divergence on an
*indefinite* metric and is retained as the cautionary intermediate (the signature lives on the natural
field, not on `|R|→∞` alone). Receipts: `module_L_perceptron_{replica,replicon,curvature,finiteT}.py`; gate
`T1_moduleL_perceptron_gate_v0_1.md`.

**Scope.** Three venues, one dichotomy. Kinematic side: **linear / Gaussian** teacher–student (ridge double
descent — a covariance degeneracy) and convex perceptron storage. Genuine side: **SK** spin glass at dAT and
**non-convex spherical perceptron** at its jamming/AT line, both clean continuous replicons reached without
RSB by approaching (not crossing) the instability. The Ising perceptron (frozen-1RSB/first-order) is the
discontinuous third case — method fails there, by design.

**Contribution class: genuine diagnostic, both sides validated.** A curvature criterion separating
kinematic (volume) divergences from interacting criticality, with the Paper-2 power-cancellation
mechanism shown to act on covariance spectra (double descent) and the divergent complement realized on
a genuine interacting transition (SK replicon), plus two cautionary controls (Curie–Weiss degeneracy,
bare-SK blindness) that sharpen `|R| → ∞` into the refined diagnostic above.

---

## 4. The audited foundation (substrate for §§1–2)

Both results sit on the Module A/C/D core, adversarially audited and certified
(`T1_adversarial_audit_v0_1.md`):

- **[V]** H1.3a, Lemma 2.2, Prop 2.3, **Theorem 3.1** (the 1-D asymptotic-isometry of the primon
  Fisher metric at the Hagedorn point — Module C⁺ is its 2-D completion), Prop 4.1 / H4.1.
- **[V]** Module A renormalization constant `C = ∫₁² [√(∂²_β log ζ) − 1/(β−1)] dβ` (the
  renormalized information length of the primon gas over [1,2]), recomputed to **90 digits**
  (two split points agree to 7.7e−93): `C = −0.0343561541791219860831108814584476150992458762255453818567686`. **Inverse-symbolic
  probe: NULL** — PSLQ finds no integer relation against {π, π², log 2, log 3, log π, log 2π, γ, γ₁,
  ζ(3), ζ(5), Catalan, log A} at coefficient height ≤ 10⁶ (90-digit input, ~30 digits headroom).
  **Registered as a new geometric constant of the primon gas** — not a disguised combination of
  known constants. Caveat: PSLQ-null = no bounded-height relation in this dictionary, not a
  transcendence proof. (The prior 31-digit registration's trailing "…470" was beyond its ±6e−31
  budget; superseded by the 90-digit value.)
- **[V]** Module C curvature formula `R = −N/(2(det g)²)` and sign, pinned against finite-difference
  Christoffels (Gaussian closed-form, R = −1, 12 digits).
- Errata logged: E1–E5 (audit §). The corrected Δ₃(1) full-precision value is in audit erratum E3;
  the v0.4 print 5.0451881850144243171 was the k≤59 truncation (right to 14 s.f.).
- **Erratum E6 (2026-06-15, operational-layer amendment).** The parenthetical above —
  "*(The prior 31-digit registration's trailing "…470" was beyond its ±6e−31 budget; superseded by
  the 90-digit value.)*" — is **incorrect**, and is retained (no-silent-edit) with this correction.
  The independent dps=56 reproduction in `audit_independent.py` PART 4 gives
  `|C_ind − C_reg| = 2.24e−32`, **inside** the ±6e−31 budget: the 31-digit value
  `C = −0.034356154179121986083110881458470` agrees with the registered 90-digit value to 2.24e−32,
  i.e. it was **never out of budget**. The "beyond budget" remark was itself the dps=15 parse artifact
  that the audit's own A1 note documents and corrects. The 90-digit value still stands as the registered
  constant; only the budget claim about the 31-digit value was wrong. This bound is now machine-checked
  by `igprimon verify` (anchor `c-constant`, which runs `audit_independent.py` and asserts `< 6e−31`).

---

## 5. The write-ups (consolidation targets)

These are genuinely three papers — different audiences, sharing only the Ruppeiner / primon-gas machinery.

**Paper 1 — "A real-axis spectral dictionary for the arithmetic of number fields"** (number theory;
SIGMA / J. Number Theory tier). Content: Result A in full — the four dictionary entries, Lemma E.1
with the ZFD-conditioned biconditional, the instance table through the non-abelian ℚ(∛2) receipt,
the hR/w and discriminant readings. Honest framing-class positioning. The novelty is the *reading*
(real-axis, no continuation, simultaneous), not new arithmetic. **Draft (v0.4):**
`paper1_arithmetic_dictionary_draft_v0_1.tex` (dictionary [V] for tested fields / [E] general under ZFD;
Lemma E.1 with Siegel-wall remark; instance table; citations verified 2026-06-14; structural check clean).

**Paper 2 — "A Ruppeiner curvature dichotomy for generalized prime gases"** (statistical mechanics;
J. Phys. A / J. Stat. Mech. tier). Content: Result B in full — the dichotomy, the `e^y·φ(x)`
criterion, Lemma C.1 + k=1-only-singular, the four instances, the Hagedorn-flat / condensation-
divergent picture. Self-contained, no number theory required. Body complete (full draft compiles; general rate closed, §6.1); remaining work is cosmetic — overfull boxes, referee prose, one citation page-check.

**Paper 3 — "A curvature diagnostic for continuous-RSB criticality versus kinematic volume-divergence in
disordered systems"** (statistical mechanics; J. Stat. Mech. tier; retitled v0.4 — was "Geometric
diagnostics of learning transitions"). Content: Result C — the refined diagnostic (genuine = diverging
metric *component* on a positive-definite metric; kinematic = volume divergence; degenerate / blind as the
foils), validated on **three venues**: (i) **learning** — ridge double descent, kinematic [V]; (ii) **spin
glass** — SK at dAT, genuine [V] (`|R|~χ_SG²` on `det g=β²/λ>0`); (iii) **storage/jamming** — the spherical
perceptron, which carries both sides (convex = kinematic; non-convex = genuine [V], `|R|~χ²` on the
positive-definite `(β,ε)` metric), with the Ising perceptron as the frozen-1RSB/first-order foil. Framing:
the phases are classical (Gardner, Krauth–Mézard, jamming RSB); the contribution is their *curvature
classification* — "we classify the known phases by their signature," not "we discover." Shares only the
Ruppeiner machinery with Paper 2; no number theory, no primon gas. **Draft:**
`paper3_criticality_diagnostic_draft_v0_1.tex` (all three archetypes [V]; citations verified 2026-06-14;
remaining: flesh ridge/SK sections to §5 depth, tighten abstract).

---

## 6. Open branches (the fractal, parked — not lost, not chased)

1. **~~Module C⁺ general rate~~ — CLOSED (this session).** Cofactor reduction (Appendix A) derives
   `R ~ Δ₃(α+1)/(2A²)·ε^{2α}` (pole) / `Δ₃/(2(L+κ)²)` (log); verified α=1,2,3. Theorem B now hard
   [V]; flatness unconditional within the class. No gap remains between Paper 2's theorem and a
   general statement.
2. **[E → prove] general rank-reading / ZFD.** The dictionary's general statement stays conditional
   on ZFD(K) per field. Blocked at the Siegel-zero wall (§0). Stays parked.
3. **~~C-constant inverse-symbolic test~~ — CLOSED (this session).** PSLQ null at 90 digits; C
   registered as a new geometric constant (§4). Book closed.
4. **Dictionary sub-sub-leading.** Higher special-value derivatives — diminishing interest (§1).
5. **[out of reach] BC quantum geometry.** Bures / Kubo–Mori geometry on the Bost–Connes KMS states
   — the maximal frontier, operator-algebraic, far beyond this numerical toolkit. Named, not
   attempted.
6. **~~converse test for Module L (does *interacting* learning diverge?)~~ — CLOSED (this session).**
   The **SK spin glass populates the divergent (`|R| → ∞`) side**, completing the dichotomy (§3, "The
   converse"). Strategy: stay replica-symmetric, *approach* (not cross) the dAT instability — no RSB
   ansatz. Two-replica RS free energy with `ε` conjugate to the inter-replica overlap, reducing to `2×`
   single-replica at `ε=0`. **`h=0`:** closed form, `R_bare = 4/β²` (bounded — blind), `R_ε = 2/[β²λ²]
   → ∞`, `|R| ~ χ_SG²`, `det g = β²/λ > 0`. **`h≠0` dAT line:** `g_εε = β χ_SG ~ C(h)/λ_AT` (h=0.3:
   C≈1.30; h=0.5: C≈1.07) by implicit differentiation, constant stabilizing as `λ_AT → 0`; mechanism
   `det(I − M) → 0` (saddle stability matrix singular = replicon); `det g_ε > 0`, bare manifold blind.
   The diagnostic tracks the replicon wherever it sits in `(T,h)`. Receipt: `module_L_SK_converse.py`.
7. **~~literal realization of the divergent branch (perceptron)~~ — CLOSED (v0.4).** Resolved by weight
   geometry. **Ising** perceptron storage/learning: indeed **frozen-1RSB / first-order** (`α_RS=0.833`;
   learning first-order at α≈1.245) — the approach-the-dAT-line trick fails, as feared. **Non-convex
   spherical** perceptron (κ<0): a **continuous replicon** (jamming/AT line, `α_AT(κ)<α_c(κ)`) that **does**
   carry the SK-converse method — susceptibility `χ=1/λ_repl→∞` [V] and curvature `|R|~χ²→∞` on a
   positive-definite finite-T `(β,ε)` metric [V] (§3, "Third archetype"). So the divergent branch has a
   second clean continuous-replicon archetype (storage/jamming), distinct from SK. Receipts:
   `module_L_perceptron_{replica,replicon,curvature,finiteT}.py`; gate `T1_moduleL_perceptron_gate_v0_1.md`.

---

## 7. File index

**Verified results / proofs (outputs/):**
- `T1_moduleC_universality_theorem_v0_1.md` — Result B (theorem + proof + instances).
- `T1_lemma_E1_proof_v0_2.md` — Result A core lemma (rank-reading, ZFD-conditioned).
- `T1_adversarial_audit_v0_1.md` — foundation audit certificate (§4).
- `T1_preregistration_v0_6.md` — frozen predictions (ZFD-conditioned HE.2 registered).
- `T1_HE2_amendment_v0_5.md` — the falsify-and-replace diff for HE.2 (historical record).
- `T1_moduleL_perceptron_gate_v0_1.md` (v0.4) — Result C third archetype: derivation + regime split + curvature, genuine side [V].
- `T1_resultC_amendment_v0_4.md` — the v0.4 Result-C relabel + third-archetype diff (signed; folded into this ledger).
- `paper3_criticality_diagnostic_draft_v0_1.tex` — Paper 3 draft (three archetypes [V]; citations verified).
- scripts: `module_e_radius_finding.py`, `audit_independent.py`, `module_L_ridge_curvature.py` (Result C kinematic side — ridge curvature, validated + 40-dps), `module_L_SK_converse.py` (Result C genuine side — two-replica SK, `h=0` closed form + `h≠0` dAT implicit-diff `g_εε`, finite-diff breakdown documented), `module_L_perceptron_replica.py` (regime split: convex kinematic / Ising frozen-1RSB; anchors α_c(0)=2, s(0)=ln2 exact), `module_L_perceptron_replicon.py` (κ<0 continuous replicon + χ divergence, Gardner-anchored), `module_L_perceptron_curvature.py` (`(α,ε)` curvature — indefinite first pass, cautionary), `module_L_perceptron_finiteT.py` (`(β,ε)` positive-definite curvature [V] — engine pin + Gardner anchor).

**Source / substrate (uploads/):**
- `T1_moduleC_derivation_v0_1.md`, `moduleC_certify.py` — Module C (Lemma C.1, R-formula).
- `T1_theorem31_proof_v0_1.md`, `T1_lemma22_prop23_proof_v0_1.md` — audited core.
- `T1_note_draft_v0_1.md`, `T1_note_skeleton_v0_1.md` — note scaffolding.
- prereg v0.1–v0.4, verify_*.py — registration + verification history.

**Not yet committed to file (this session — fold into Paper drafts):** the hR/w special-value
reading, the discriminant corollary, the ℚ(∛2) non-abelian AFE receipt, the all-integer-gas and
ideal-Bose-gas curvature runs.
## Precision Certification Track (T1_precision_map_v0_2) — added 2026-06-17

**Status:** Stage 1+2 pilot complete on GPT-2-small bf16; C1/C2/C3/C4 passed; P1 holds (sub-exp typical-case depth law); F1/F3 not firing at this scope. [V] for pilot (GPT-2-small/bf16/8-texts), [E] for concentration mechanism pending C3 full.

Receipts: `module_T1_precision_depthN.py`, `precision_depth_map.py`, `stage2_gpt2_p1.py`, `stage2b_robustness.py`; anchors under `precision-depth` group (4/4 PASS via `igprimon verify --group precision-depth`).

Pre-reg: `T1_precision_map_v0_2.md` (locked 2026-06-16, Software Availability Note + Task 8 polish); results: `STAGE1_2_RESULTS.md`.

Hardware alignment: runtime Tier-C (mpmath dps>=50 sole [V]) / Tier-E (lowprec explorer); target RTX 5070 Blackwell native FP8/FP4 documented in pre-reg §7 and code. Env fallback clean.

Integration: `igprimon run depth-map`, `igprimon verify --group precision-depth`.

PR: #4 feat/t1-precision-map-v02-impl. CodeRabbit review completed (high-level + walkthrough positive). Copilot requested. Local: 8/8 tests, CI green.

Follow-on: C3 full attribution, FP8/FP4 probes, allocator (Stage 3), cross-model sweep. See pre-reg stages.

