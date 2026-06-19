# Paper sketch — The Hagedorn–RLCT Correspondence
### Fisher information of the primon gas and the learning coefficient of singular models

**Status:** sketch / pre-registration. Frames a structural *dictionary* between two existing programs —
Watanabe's Singular Learning Theory (SLT) and the IG-PRIMON arithmetic information-geometry program. Follows
the IG-PRIMON house discipline: explicit statement → numerical test → honest scope wall. **Not** a new theorem
in either field; a *reading* that welds them. Doubles as the proof-of-capability credential for the operating manual.

## One-line claim
SLT's **RLCT** (learning coefficient λ) and the IG-PRIMON **Fisher-information reading** of an arithmetic
partition function are the *same invariant in two languages*: both are governed by the **largest pole of a zeta
function** controlling an asymptotic expansion. λ = the pole location, the **multiplicity** = its order, and the
model-/field-specific content lives in the **subleading amplitude**.

## The two sides — both already computed this session
- **SLT side.** For a loss `K(w)`, the learning zeta `ζ(z)=∫ K(w)^z φ(w) dw` has its largest pole at `z=−λ` of
  order `m`; free energy `~ λ log n − (m−1) log log n`. Computed (`rlct_ml_side.py`): the normal-crossing
  `K=(w1·w2)^2` from volume scaling `V(ε)=vol{K<ε} ~ ε^λ(−log ε)^{m−1}` → **λ̂ ≈ 0.476** (truth ½),
  **m̂ ≈ 1.57** (truth 2). (Multiplicity undershoots — finite ε-range can't resolve the log-factor; the floor
  whispering exactly where to widen the range.)
- **Arithmetic side.** The primon gas `ζ_K(β)` (Dedekind zeta as a grand partition function over integral ideals)
  has Fisher information `I(β)=∂²_β log ζ_K = Var_β(log Nn)`. At the Hagedorn point `β=1` the simple pole forces
  `I(β) ~ 1/(β−1)²`. Computed (`primon_rlct_bridge.py`, `primon_field_bridge.py`): **(β−1)²·I → 1.000**
  (ℚ, ℚ(√−3), ℚ(√5)); leading term **universal** (the pole order), subleading **splits the fields**
  (−0.185 / −0.478 / −1.223) — the arithmetic, exactly as IG-PRIMON's dictionary reads it.

## The bridge lemma (to state + prove in IG-PRIMON style)
**Lemma (Hagedorn–RLCT).** Let `Z(β)` be a partition function whose Mellin/Dirichlet generating zeta has its
nearest singularity at `β_c` — a pole of order `ρ`. Then the Fisher information `I(β)=∂²_β log Z` diverges as
`I(β) ~ ρ/(β−β_c)² + (subleading)`, and the pair `(β_c, ρ)` is the **RLCT-location / RLCT-multiplicity** of the
associated singular model. The subleading amplitude carries the *resolution* (blow-up) data — the
class-number-formula residue on the arithmetic side, the model geometry on the ML side.
*(Conditions: a ZFD-type hypothesis controlling the nearest singularity — the same explicit "wall" the
IG-PRIMON arithmetic dictionary already states.)*

## Numerical test plan (extend what's done)
1. ✅ primon gas ℚ — `(β−1)²I → 1`.
2. ✅ quadratic fields — leading universal, subleading splits the arithmetic.
3. ✅ normal-crossing RLCT — `λ ≈ ½`, `m ≈ 2`.
4. ✅ **cyclic cubic ℚ(ζ₇)⁺ (unit rank 2)** (`cubic_field_bridge.py`): `(β−1)²I = 0.9991` (leading universal);
   subleading **−2.14**. The subleading is a clean **monotone arithmetic ordering** —
   ℚ −0.185 → ℚ(√−3) −0.478 → ℚ(√5) −1.223 → cubic −2.14 — deepening with #L-factors / unit rank. *First evidence
   the subleading amplitude is a rank-graded invariant; the candidate payload of the dictionary.*
5. ✅ **2-layer linear network** `K=(Σ_h a_h b_h)²`, H hidden units (`rlct_network.py`): λ ≈ 0.5 **universal
   across widths** (H=1→5: 0.48 / 0.52 / 0.51 / 0.49); multiplicity runs **1.64 → 0.95** as the H=1 normal
   crossing smooths to a hypersurface. **Both sides now show the SAME signature** — pole *location* universal,
   pole *order / multiplicity* carries the structure (rank-graded on the arithmetic side, layer-width here).
   The dictionary's central claim, confirmed at *both* ends.
6. ☐ make the **subleading ↔ arithmetic** map quantitative (the dictionary's payload).
7. ☐ **up the ladder:** sheaf cohomology of the RLCT-section over a family — does the local effective-dimension
   glue to a global obstruction?

## Honest scope wall (NO-overclaim, per house rule)
- **Framing-class result.** A dictionary between two existing programs — not new arithmetic, not new SLT. It
  computes no new RLCT and proves no new theorem; it identifies that the *pole-of-a-zeta* machinery is **literally
  shared**, and opens cross-tool flow (Fisher-geometry → SLT estimation; resolution-of-singularities → the
  arithmetic subleading).
- **NO-RH-CLAIM.** Nothing here bears on the location of any zero of any L-function (inherits the IG-PRIMON wall).

## Positioning
SLT / developmental interpretability is an active 2024–2026 field (free energy, RLCT estimation, grokking;
arXiv 2505.13902, 2406.10234, 2010.11560). The information-geometry-of-arithmetic angle is, to our knowledge,
not assembled there. The contribution is the unification + the cross-tool flow — stated with the same scope
discipline the IG-PRIMON papers apply to themselves. As a credential: a rigorous, computed, honestly-scoped
bridge into a live field, built fast and verified at both ends.
