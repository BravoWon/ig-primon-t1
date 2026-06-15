# IG-PRIMON-T1 — Consolidated Results Ledger (v0.1, 2026-06-14)

**Purpose.** The trunk. One place that states what the program has *established*, with receipts and
honest status tags, the scope walls that bound it, and the two write-ups it feeds. This is not the
pre-registration (the frozen-prediction document, `T1_preregistration_v0_6.md`); this is the
results synthesis — what is now known, verified, and defensible.

**Discipline carried throughout.** HONEST_CLAIMS: **[V]** verified, **[E]** defensible
extrapolation, **[C]** conjecture; **gate** (derive before numerics); no-silent-edit on registered
items. Sign conventions stated where they bite.

**The object.** The canonical primon (Riemann-gas) and its generalizations: a grand-canonical gas
whose partition function is a zeta / Dirichlet series, studied via its **Fisher / information
geometry on the real axis near the Hagedorn point β = 1**. Two results are complete; both live on
the real temperature axis, neither touches the complex zeros.

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

## 3. The audited foundation (substrate for §§1–2)

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

## 4. The two write-ups (consolidation targets)

These are genuinely two papers — different audiences, sharing only the primon-gas framework.

**Paper 1 — "A real-axis spectral dictionary for the arithmetic of number fields"** (number theory;
SIGMA / J. Number Theory tier). Content: Result A in full — the four dictionary entries, Lemma E.1
with the ZFD-conditioned biconditional, the instance table through the non-abelian ℚ(∛2) receipt,
the hR/w and discriminant readings. Honest framing-class positioning. The novelty is the *reading*
(real-axis, no continuation, simultaneous), not new arithmetic.

**Paper 2 — "A Ruppeiner curvature dichotomy for generalized prime gases"** (statistical mechanics;
J. Phys. A / J. Stat. Mech. tier). Content: Result B in full — the dichotomy, the `e^y·φ(x)`
criterion, Lemma C.1 + k=1-only-singular, the four instances, the Hagedorn-flat / condensation-
divergent picture. Self-contained, no number theory required. One open item to close first (§5.1).

---

## 5. Open branches (the fractal, parked — not lost, not chased)

1. **~~Module C⁺ general rate~~ — CLOSED (this session).** Cofactor reduction (Appendix A) derives
   `R ~ Δ₃(α+1)/(2A²)·ε^{2α}` (pole) / `Δ₃/(2(L+κ)²)` (log); verified α=1,2,3. Theorem B now hard
   [V]; flatness unconditional within the class. No gap remains between Paper 2's theorem and a
   general statement.
2. **[E → prove] general rank-reading / ZFD.** The dictionary's general statement stays conditional
   on ZFD(K) per field. Blocked at the Siegel-zero wall (§0). Stays parked.
3. **~~C-constant inverse-symbolic test~~ — CLOSED (this session).** PSLQ null at 90 digits; C
   registered as a new geometric constant (§3). Book closed.
4. **Dictionary sub-sub-leading.** Higher special-value derivatives — diminishing interest (§1).
5. **[out of reach] BC quantum geometry.** Bures / Kubo–Mori geometry on the Bost–Connes KMS states
   — the maximal frontier, operator-algebraic, far beyond this numerical toolkit. Named, not
   attempted.

---

## 6. File index

**Verified results / proofs (outputs/):**
- `T1_moduleC_universality_theorem_v0_1.md` — Result B (theorem + proof + instances).
- `T1_lemma_E1_proof_v0_2.md` — Result A core lemma (rank-reading, ZFD-conditioned).
- `T1_adversarial_audit_v0_1.md` — foundation audit certificate (§3).
- `T1_preregistration_v0_6.md` — frozen predictions (ZFD-conditioned HE.2 registered).
- `T1_HE2_amendment_v0_5.md` — the falsify-and-replace diff for HE.2 (historical record).
- scripts: `module_e_radius_finding.py`, `audit_independent.py`.

**Source / substrate (uploads/):**
- `T1_moduleC_derivation_v0_1.md`, `moduleC_certify.py` — Module C (Lemma C.1, R-formula).
- `T1_theorem31_proof_v0_1.md`, `T1_lemma22_prop23_proof_v0_1.md` — audited core.
- `T1_note_draft_v0_1.md`, `T1_note_skeleton_v0_1.md` — note scaffolding.
- prereg v0.1–v0.4, verify_*.py — registration + verification history.

**Not yet committed to file (this session — fold into Paper drafts):** the hR/w special-value
reading, the discriminant corollary, the ℚ(∛2) non-abelian AFE receipt, the all-integer-gas and
ideal-Bose-gas curvature runs.
