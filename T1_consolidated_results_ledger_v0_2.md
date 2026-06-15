# IG-PRIMON-T1 — Consolidated Results Ledger (v0.2, 2026-06-14)

**Purpose.** The trunk. One place that states what the program has *established*, with receipts and
honest status tags, the scope walls that bound it, and the two write-ups it feeds. This is not the
pre-registration (the frozen-prediction document, `T1_preregistration_v0_6.md`); this is the
results synthesis — what is now known, verified, and defensible.

**Discipline carried throughout.** HONEST_CLAIMS: **[V]** verified, **[E]** defensible
extrapolation, **[C]** conjecture; **gate** (derive before numerics); no-silent-edit on registered
items. Sign conventions stated where they bite.

**v0.2 (this revision).** Adds **Result C / Module L** — geometric diagnostics of learning transitions (§3); records its write-up target (Paper 3, §5) and the non-Gaussian converse test as a parked branch (§6.6); registers the receipt `module_L_ridge_curvature.py` (§7). Results A and B are substantively unchanged; v0.1 §§3–6 renumber to §§4–7, and section cross-references are updated to match — both `(§3)→(§4)` pointers (open branches; file index) and the Paper 2 open-item line (whose general-rate blocker had already closed in v0.1).

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

## 3. Result C — geometric diagnostics of learning transitions (Module L)

**Claim.** The Ruppeiner machinery of Result B, carried *off* the primon gas onto a **learning**
statistical manifold, separates a **kinematic / volume divergence** (e.g. double descent) from
**genuine interacting criticality**: the former leaves the scalar curvature bounded — or vanishing —
while the metric *volume* diverges; only the latter gives `|R| → ∞`. The double-descent peak is,
geometrically, a *fake* transition. Receipt: `module_L_ridge_curvature.py`.

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

**Taxonomy.** Learning geometry falls in the **strictly negative** (Gaussian / repulsive) curvature
class — matching the normal family (`R = −1`) and the exact small-N teacher–student perceptron
(fraction `R < 0` = 1.00) — and is therefore **distinct from the Hagedorn point**, which approaches
zero from the **positive** side (`R → 0⁺`). Same dichotomy *side* (non-divergent), opposite
sign-class: a rigid taxonomic boundary between arithmetic-gas and learning manifolds. The three
observed behaviors, one convention (sphere-positive):

| manifold / transition | `R` | class |
|---|---|---|
| ideal Bose gas (BEC), fugacity-driven | `+∞` | genuine interacting criticality |
| primon / integer gas (Hagedorn), temp-driven | `→ 0⁺` | kinematic-flat (positive) |
| linear learning (double descent), λ→0 at α=1 | `→ 0⁻` noiseless / bounded `< 0` noisy | kinematic-flat / bounded (negative) |

**Scope — the honest wall.** This is the **linear / Gaussian** teacher–student. Its double descent is
a *covariance degeneracy*, which is precisely *why* it is geometrically flat-to-bounded: there is no
genuine interacting critical point to source a curvature divergence. The result does **not** establish
that all learning transitions are non-divergent — the converse (a genuinely non-Gaussian transition)
is the open branch (§6.6) and is hard (it enters replica-symmetry-breaking).

**Contribution class: genuine diagnostic.** A curvature criterion separating kinematic (volume)
divergences from interacting criticality, validated on an exactly-solvable learning manifold, with the
Paper-2 power-cancellation mechanism shown to act on covariance spectra as well as arithmetic ones.

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

---

## 5. The write-ups (consolidation targets)

These are genuinely three papers — different audiences, sharing only the Ruppeiner / primon-gas machinery.

**Paper 1 — "A real-axis spectral dictionary for the arithmetic of number fields"** (number theory;
SIGMA / J. Number Theory tier). Content: Result A in full — the four dictionary entries, Lemma E.1
with the ZFD-conditioned biconditional, the instance table through the non-abelian ℚ(∛2) receipt,
the hR/w and discriminant readings. Honest framing-class positioning. The novelty is the *reading*
(real-axis, no continuation, simultaneous), not new arithmetic.

**Paper 2 — "A Ruppeiner curvature dichotomy for generalized prime gases"** (statistical mechanics;
J. Phys. A / J. Stat. Mech. tier). Content: Result B in full — the dichotomy, the `e^y·φ(x)`
criterion, Lemma C.1 + k=1-only-singular, the four instances, the Hagedorn-flat / condensation-
divergent picture. Self-contained, no number theory required. Body complete (full draft compiles; general rate closed, §6.1); remaining work is cosmetic — overfull boxes, referee prose, one citation page-check.

**Paper 3 — "Geometric diagnostics of learning transitions" (the double-descent peak is
geometrically trivial)** (statistical mechanics of learning; J. Stat. Mech. / NeurIPS-theory tier).
Content: Result C — the kinematic-vs-interacting curvature criterion, the exact ridge-regression
manifold, the rank-1 power-cancellation, the noiseless-flat / noisy-bounded rates, and the
negative-sign taxonomy against the Hagedorn point. Shares only the Ruppeiner machinery with Paper 2;
no number theory, no primon gas. Stands alone; the non-Gaussian converse (§6.6) would extend it but
is not required.

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
6. **[hard — assess first] converse test for Module L (does *interacting* learning diverge?).**
   Result C shows the *linear / Gaussian* learning transition is non-divergent (kinematic). Open
   question: does a *genuinely non-Gaussian* learning transition populate the divergent (`|R| → ∞`)
   side — the Ising-perceptron first-order storage transition, or a feature-learning / specialization
   transition? Hard: enters replica-symmetry-breaking, where the free energy loses analyticity and a
   clean Hessian determinant is **not** a drop-in. Parked until RSB tractability is assessed; the
   linear baseline (Result C) is locked first.

---

## 7. File index

**Verified results / proofs (outputs/):**
- `T1_moduleC_universality_theorem_v0_1.md` — Result B (theorem + proof + instances).
- `T1_lemma_E1_proof_v0_2.md` — Result A core lemma (rank-reading, ZFD-conditioned).
- `T1_adversarial_audit_v0_1.md` — foundation audit certificate (§4).
- `T1_preregistration_v0_6.md` — frozen predictions (ZFD-conditioned HE.2 registered).
- `T1_HE2_amendment_v0_5.md` — the falsify-and-replace diff for HE.2 (historical record).
- scripts: `module_e_radius_finding.py`, `audit_independent.py`, `module_L_ridge_curvature.py` (Result C receipt — ridge curvature, validated + 40-dps).

**Source / substrate (uploads/):**
- `T1_moduleC_derivation_v0_1.md`, `moduleC_certify.py` — Module C (Lemma C.1, R-formula).
- `T1_theorem31_proof_v0_1.md`, `T1_lemma22_prop23_proof_v0_1.md` — audited core.
- `T1_note_draft_v0_1.md`, `T1_note_skeleton_v0_1.md` — note scaffolding.
- prereg v0.1–v0.4, verify_*.py — registration + verification history.

**Not yet committed to file (this session — fold into Paper drafts):** the hR/w special-value
reading, the discriminant corollary, the ℚ(∛2) non-abelian AFE receipt, the all-integer-gas and
ideal-Bose-gas curvature runs.
