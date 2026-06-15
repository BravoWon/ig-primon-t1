# Lemma E.1 — The K-primon Fisher radius and the Dirichlet unit rank

**IG-PRIMON-T1, Module E. Proof document v0.1 (2026-06-13).**
Closes (in part) the HE.2 [E → prove] obligation. Companion receipts: `module_e_radius_finding.py`, the cubic probe, and the amplitude-law verification (orders 1–2, radii 1/2/3).
**Discipline:** HONEST_CLAIMS — [V] verified, [E] defensible extrapolation, [C] conjecture.

---

## Setup

Let K be a number field of signature (r₁, r₂), degree n = r₁ + 2r₂, with **Dirichlet unit rank** ρ := r₁ + r₂ − 1 (Dirichlet unit theorem). Let ζ_K(s) be its Dedekind zeta function — a simple pole at s = 1, no other poles. The K-primon gas has Fisher information

  I_K(β) = ∂²_β log ζ_K(β),  β real, near the Hagedorn point β = 1.

Put s = 1 + ε and define the **regular log-generator**

  G(ε) := log[(s − 1) ζ_K(s)] = log[ε · ζ_K(1 + ε)].

Because (s − 1)ζ_K(s) is regular and nonzero at s = 1 (its value there is the residue Res_{s=1} ζ_K = (2^{r₁}(2π)^{r₂} h R)/(w √|d_K|) ≠ 0, by the class number formula), G is analytic at ε = 0. Since log ζ_K(1+ε) = −log ε + G(ε),

  I_K(1 + ε) = ε^{−2} + G''(ε),  c_k^K := [ε^k] (I_K(1+ε) − ε^{−2}) = (k+2)(k+1) · b_{k+2},

where b_k := [ε^k] G. The radius of {c_k^K} equals the radius of {b_k}, namely the distance from ε = 0 to the nearest singularity of G — i.e. the nearest **zero of (s − 1)ζ_K(s)**. The factor (s − 1) cancels the pole and contributes no zero, so the singularities of G are exactly the zeros of ζ_K.

---

## Statement

**(i) Entire of order 1.** (s − 1)ζ_K(s) is entire of order 1.

**(ii) Trivial-zero structure.** ζ_K has trivial zeros at s ≤ 0 with orders fixed by the signature:
  ord_{s=0} ζ_K = r₁ + r₂ − 1 = ρ,  ord_{s=−(2j−1)} ζ_K = r₂,  ord_{s=−2j} ζ_K = r₁ + r₂  (j ≥ 1),
and the leading term at s = 0 is ζ_K(s) = −(hR/w) s^ρ (1 + O(s)) (nonzero coefficient). Hence the nearest trivial zero to s = 1 lies at distance

  R_triv = 1 if ρ ≥ 1 (zero at s = 0);  = 2 if ρ = 0, r₂ ≥ 1 (zero at s = −1);  = 3 if ρ = 0, r₂ = 0 (zero at s = −2).

**(iii) Unconditional bound.** If ρ ≥ 1, then s = 0 is a zero of order ρ at distance 1, so **radius ≤ 1**.

**(iv) Radius identity.** radius = distance from s = 1 to the nearest zero of ζ_K = min(R_triv, R_nt), where R_nt is the distance to the nearest **nontrivial** zero. radius = R_triv iff ζ_K has no nontrivial zero in the open disk |s − 1| < R_triv.

**(v) Hypothesis ZFD(K).** *ζ_K has no nontrivial zero in the punctured disk 0 < |s − 1| < R_triv.* Under ZFD(K), with ε₀ the location and m the order of the binding (nearest) zero:

  G(ε) = m · log(1 − ε/ε₀) + H(ε),  H analytic on |ε| < d,  d = next-nearest zero distance > |ε₀|,

and therefore the **amplitude law**

  |b_k| · k · R^k → m  (R = radius = |ε₀|, m = order of the binding zero).

**(vi) Specializations.** Under ZFD(K):
- ρ ≥ 1 ⟹ radius = 1 and |b_k| · k → m = ρ — the expansion reads the **exact unit rank**;
- ρ = 0 ⟹ Q: radius 3, m = 1; imaginary quadratic: radius 2, m = 1.

Direction radius = 1 ⟹ ρ ≥ 1 is **unconditional** (ρ = 0 forces R_triv ≥ 2, so radius ≥ 2 absent an in-disk nontrivial zero — and an in-disk nontrivial zero only lowers the radius further below 2, never to 1 from the rank-0 side without first violating known zero-free regions near s = 1; for the rank-0 fields of interest ZFD is immediate). The converse ρ ≥ 1 ⟹ radius = 1 is the part requiring ZFD(K).

---

## Proof

**(i).** Hecke's completed zeta Λ_K(s) = |d_K|^{s/2} Γ_R(s)^{r₁} Γ_C(s)^{r₂} ζ_K(s), with Γ_R(s) = π^{−s/2}Γ(s/2), Γ_C(s) = 2(2π)^{−s}Γ(s), extends to a function that is holomorphic except for simple poles at s = 0 and s = 1 and satisfies Λ_K(s) = Λ_K(1 − s); it is of order 1 (Stirling on the Γ-factors, Phragmén–Lindelöf on ζ_K in vertical strips). Thus ζ_K = Λ_K / (|d_K|^{s/2} Γ_R^{r₁} Γ_C^{r₂}) is meromorphic of order 1 with a single simple pole at s = 1, and (s − 1)ζ_K is entire of order 1. ∎(i) [V] — standard (Hecke; Lang, *ANT*, Ch. XIII; Neukirch, Ch. VII).

**(ii).** Λ_K is holomorphic and nonzero on s ≤ 0 except for its simple pole at s = 0; the Γ-factors carry the only poles there. Γ(s/2) has simple poles at s = 0, −2, −4, …; Γ(s) at s = 0, −1, −2, …. Hence the Γ-factor Γ_R^{r₁} Γ_C^{r₂} has a pole of order:
- r₁ + r₂ at s = 0,  • r₂ at s = −(2j−1),  • r₁ + r₂ at s = −2j (j ≥ 1).

For Λ_K = (Γ-factors) · |d_K|^{s/2} · ζ_K to be holomorphic at each s < 0 (and to have only a simple pole at s = 0), ζ_K must vanish to exactly the Γ-factor pole order there, minus 1 at s = 0 (where Λ_K itself keeps a simple pole). This gives ord_{s=0} = r₁ + r₂ − 1 = ρ, ord_{s=−(2j−1)} = r₂, ord_{s=−2j} = r₁ + r₂. The leading coefficient at s = 0 is −hR/w ≠ 0 (analytic class number formula at s = 0; equivalently the residue at s = 1 transported by the functional equation). The distances R_triv follow by inspection. ∎(ii) [V] — classical.

**(iii).** By (ii), ρ ≥ 1 makes s = 0 a zero of (s − 1)ζ_K of order ρ ≥ 1 at distance 1 from s = 1; a power series cannot converge past its nearest singularity, so radius ≤ 1. ∎(iii) [V].

**(iv).** The singularities of G(ε) = log[(s − 1)ζ_K] are the zeros of (s − 1)ζ_K = the zeros of ζ_K (the (s−1) factor removes the pole and adds no zero). The radius of a power series equals the distance to its nearest singularity. Trivial zeros nearest s = 1 lie at R_triv; nontrivial zeros lie in the critical strip 0 < Re s < 1. Whichever is closer sets the radius; the trivial one wins iff no nontrivial zero lies strictly inside |s − 1| < R_triv. ∎(iv) [V].

**(v).** Under ZFD(K) the binding zero is the nearest trivial zero, at ε₀ (|ε₀| = R_triv = R) with order m. Write (s − 1)ζ_K(s) = (ε − ε₀)^m A(ε) with A analytic and A(ε₀) ≠ 0 in a neighborhood reaching the next zero at distance d > |ε₀| (this neighborhood is genuinely zero-free by ZFD plus (ii)). Then

  G(ε) = m log(ε − ε₀) + log A(ε) = m log(1 − ε/ε₀) + [m log(−ε₀) + log A(ε)],

where the bracket H(ε) is analytic on |ε| < d. Since log(1 − x) = −Σ_{k≥1} x^k/k,

  b_k = [ε^k]G = −m/(k ε₀^k) + [ε^k]H,  and  [ε^k]H = O(d^{−k}).

Therefore |b_k| = (m/k) |ε₀|^{−k} (1 + O((|ε₀|/d)^k)), giving |b_k| · k · R^k = m (1 + O((R/d)^k)) → m. The sign is b_k = −m/(k ε₀^k); for ε₀ = −1 this is m(−1)^{k+1}/k, matching the observed alternation. ∎(v) [V] given ZFD.

**(vi).** Substitute (ii): ρ ≥ 1 ⟹ ε₀ = −1, R = 1, m = ρ, so |b_k|·k → ρ. ρ = 0, r₂ = 0 (Q): ε₀ = −3, R = 3, m = 1. ρ = 0, r₂ ≥ 1 (imaginary quadratic): ε₀ = −2, R = 2, m = 1. ∎(vi).

---

## On the hypothesis ZFD(K)

ZFD(K) is a genuine, field-specific condition, **not** a consequence of GRH. A nontrivial zero ½ + it under GRH lies at distance √(¼ + t²) from s = 1, which is inside the unit disk whenever |t| < √3/2 ≈ 0.866. Two failure modes for ρ ≥ 1:

- **Exceptional (Siegel) real zero** β₀ ∈ (0,1): distance 1 − β₀ < 1, binds before s = 0 ⟹ radius = 1 − β₀ < 1, and |b_{k+1}/b_k| → 1/(1−β₀) > 1 rather than → 1.
- **Low-lying complex zero** with (1−σ)² + t² < 1.

For the verified fields — Q(√5), Q(√2) (ρ = 1), the cyclic cubic of conductor 7 (ρ = 2), and the rank-0 controls Q, Q(√−3) — ZFD holds, and the **numerics are the certificate**: |b_k|·k·R^k → m to 6–7 significant figures, which an in-disk zero at distance R' < R would have prevented (it would have pinned the limit to 1/R' ≠ 1/R). The lowest zeros of these small-discriminant fields sit well above the relevant ordinate threshold and none carries a Siegel zero.

## Status and registration impact

- **[V] unconditional:** (i)–(iv); the amplitude law (v) as an analytic statement; radius ≤ 1 for ρ ≥ 1; radius = 1 ⟹ ρ ≥ 1.
- **[V] for the five tested fields:** ZFD verified numerically ⟹ radius = R_triv and the exact rank/order reading.
- **[E → prove], conditional:** the general "ρ ≥ 1 ⟹ radius = 1 and |b_k|·k → ρ" holds **under ZFD(K)**, a checkable per-field hypothesis, expected but not unconditionally provable (it would follow from a no-Siegel-zero theorem plus a uniform low-lying-zero bound, neither available in general).

**Registration impact (HE.2, → v0.6):** the v0.5 phrasing "radius = 1 for every field of unit rank ≥ 1" is to be qualified by ZFD(K). Corrected biconditional: radius = 1 ⟹ ρ ≥ 1 (unconditional); ρ ≥ 1 ⟹ radius ≤ 1, with equality iff ZFD(K). The amplitude law (v) replaces the bare "|b_k|·k → rank" with the distance-aware |b_k|·k·R^k → (order of nearest zero), of which the rank reading is the R = 1 case. Contribution class unchanged: **framing** — the trivial-zero structure (ii) is classical; the contribution is its information-geometric reading on the real temperature axis.
