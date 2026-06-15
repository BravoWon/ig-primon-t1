# IG-PRIMON-T1 — Registration amendment v0.4 → v0.5 (DRAFT for lock)

**Per the standing rule: "Amendments require a versioned diff; silent edits void the registration." This is that diff. Companion receipts: `module_e_radius_finding.py` (quadratic radius) and the cubic-field probe (rank extraction).**

---

## Revised HE.2 (supersedes the v0.4 text)

**HE.2 [REVISED v0.5 — the K-primon Fisher radius reads the Dirichlet unit rank].**
The radius of convergence of the Fisher expansion I_K(1+ε) = ∂²_β log ζ_K about the Hagedorn point β=1 equals the distance from s=1 to the nearest zero of (s−1)ζ_K(s). For all but the smallest fields the binding zero is at **s=0**, where ζ_K vanishes to order **r₁+r₂−1** — the Dirichlet unit rank (classical; from the functional equation / Brauer). Hence:

- **unit rank ≥ 1 ⟹ radius = 1**, for every K with infinite unit group (every number field except Q and imaginary quadratic). The s=0 zero sits at distance 1 from the Hagedorn point and supersedes every deeper trivial zero.
- **unit rank 0 ⟹ radius ≥ 2**: Q → 3 (ζ trivial zero at s=−2); imaginary quadratic → 2 (odd-character L-trivial zero at s=−1).

Moreover the **order m = r₁+r₂−1 itself is recovered from the leading coefficient asymptotic**. With G(ε) = log[(s−1)ζ_K(1+ε)] and log-series coefficients b_k := [ε^k]G, the m-fold zero of ζ_K at s=0 makes I_K carry an m-fold pole at ε=−1, giving

  |b_k|·k → m   (equivalently c_k^K ~ m·(−1)^{k+1}(k+1), the pole amplitude).

So the real-axis thermodynamic geometry of the prime gas reads the **exact** Dirichlet unit rank off its radius (rank 0 vs ≥1, refined to 3/2 within rank 0 by character parity) and its leading coefficient amplitude (the order m).

**Status.** [V] for K = Q, Q(√−3) (rank 0), Q(√5), Q(√2) (rank 1, m→1), and the cyclic cubic field of conductor 7, signature (3,0) (rank 2, m→2): numerically verified, |b_k|·k → m to 7 significant figures with the predicted O(k·3⁻ᵏ) tail (next singularity at s=−2, distance 3). Order cross-checked by ζ_K(2h)/ζ_K(h) → 2ᵐ (2.004, 2.003, 4.006). [E → prove] for the general statement (rank ≥ 1 ⟺ radius 1; |b_k|·k → r₁+r₂−1), which follows from ord_{s=0} ζ_K = r₁+r₂−1 plus the Hadamard / log-singularity argument of Lemma 2.2; to be written as a lemma. **Not yet a receipt:** the mixed cubic, signature (1,1), rank 1 — covered by the general argument but unverified numerically, as it requires the degree-2 Artin L-function of the S₃ Galois closure, outside the Dirichlet/Hurwitz route used here. (Another order-1 case; non-discriminating. The rank-2 cyclic cubic was the decisive test.)

**Contribution class: framing.** The arithmetic fact (ord_{s=0} ζ_K = r₁+r₂−1) is classical. The contribution is its information-geometric reading — the unit rank surfacing as the convergence radius and leading coefficient amplitude of the K-primon Fisher expansion on the real temperature axis, with no complex continuation. Same claim-class as the H1.3b reading.

**NO-RH-CLAIM:** unaffected.

**Falsifier.** Any number field of unit rank ≥ 1 whose K-primon Fisher expansion has radius ≠ 1; or any field for which |b_k|·k fails to converge to r₁+r₂−1.

**Correction note.** The v0.4 HE.2 predicted radius 3 for real quadratic fields. Wrong: it tracked the even-character trivial zeros at s=−2, −4 (and correctly the doubling at s=−2 against ζ's zero there) but overlooked the s=0 zero — the unit-rank zero — which binds first, at distance 1. The analytic gate caught the error before the Module E coefficient tables were generated.

---

## Changelog entry

v0.4 → v0.5 (2026-06-13): **Module E item HE.2 revised — registered prediction falsified and replaced by a stronger statement.** The v0.4 claim (imaginary quadratic radius 2; real quadratic radius 3) was wrong in the real-quadratic case: the K-primon Fisher radius is set by the s=0 zero of ζ_K, of order r₁+r₂−1 (Dirichlet unit rank), at distance 1 — not by the s=−2 even-character trivial zeros. Revised HE.2: radius = 1 for every field of unit rank ≥ 1 (all K except Q and imaginary quadratic); radius ≥ 2 only at rank 0 (Q → 3, imaginary quadratic → 2); and the order of the s=0 zero — the exact unit rank — is read off the leading coefficient asymptotic, |b_k|·k → r₁+r₂−1. Verified [V]: Q, Q(√−3) (rank 0); Q(√5), Q(√2) (rank 1, m→1); cyclic cubic conductor 7 / signature (3,0) (rank 2, m→2); to 7 significant figures (cubic probe; order cross-checked ζ_K(2h)/ζ_K(h) → 2ᵐ). [E → prove]: the general rank ≥ 1 ⟹ radius 1 statement and the |b_k|·k → rank reading, pending a lemma from ord_{s=0} ζ_K = r₁+r₂−1 and the Lemma 2.2 argument. Not yet a receipt: the (1,1) mixed cubic (needs the degree-2 Artin L-function). Contribution class: framing (the arithmetic fact is classical; the IG reading is the contribution). Sections amended: header, HE.2, Changelog. No constants of Modules A/C/D altered; no other HE item altered; NO-RH-CLAIM unaffected. Companion receipts: module_e_radius_finding.py + cubic extension. Amendment per the standing rule — versioned diff, no silent edit.
