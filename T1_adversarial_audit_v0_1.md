# IG-PRIMON-T1 — Adversarial Audit Certificate v0.1

**Auditor:** Claude (Anthropic), 2026-06-12/13. **Stratum (per Section 0 external-input rule):** this document is [new claim]-grade input with execution receipts — every check below was executed in-session; outputs are reproduced verbatim or to stated digits. **Scope:** Lemma 2.2 / Proposition 2.3 proof document v0.1; Theorem 3.1 proof document v0.1; Module C derivation v0.1; note draft v0.1 (incl. Proposition 4.1); registrations v0.1–v0.4; all six companion scripts.

**Verdict: PASS, with findings.** Both proof documents are sound and survive line-by-line audit. The Module C derivation is sound; its certification record carries two wording errata and one precision-reporting erratum (E1–E3, none affecting the LOCKED result). All six scripts reproduced exactly. Three of four independent checks pass cleanly; the fourth (C corroboration) passes inside the registered budget with a noted caveat on its own sub-budget digits. Authorized flips: **H1.3a → [V]**, **Lemma 2.2 → [V]**, **Proposition 2.3 (as stated, full-zero-set form) → [V]**, **Theorem 3.1 → [V]**, **Proposition 4.1 / H4.1 → [V]**. Registration bumps v0.4 → v0.5 (changelog block in §7). Per the standing rule that an audit reporting zero findings is presumptively not an audit: findings are E1–E5 plus the auditor's own A1.

---

## 1. Method

Three strata, in order: (1) hand verification of every stated inequality, identity, interchange, and branch argument in the proof documents — no step accepted on authority; (2) exact reproduction of all six companion scripts in a fresh environment; (3) independent checks that do **not** reuse the program's computational routes (different identities, different summation targets, different quadrature subdivisions, a closed-form known-answer benchmark). Stratum 3 exists because reproduction alone only certifies that the code computes what the code computes.

## 2. Reproduction record

| Script | dps | Claimed | Reproduced | Status |
|---|---|---|---|---|
| verify_t1.py | 50 | c₀–c₆ bit-exact vs H1.1/H1.2; ε⁷ tail 4.11e−32 at ε=1e−4; variance residual 1.6e−9 | identical | ✓ |
| lemma22_table.py | 50 | E/bound 0.134…0.230, all inside; sign (−1)^{k+1} | identical | ✓ |
| verify_h13b_gate.py | 50/25 | Z(2)–Z(6) residuals ≤ 3.6e−51; 150-zero Z(2) diff 4.91e−3 vs ~4.9e−3 tail; Z(3) 2.07e−8 | identical | ✓ |
| verify_thm31.py | 30 | columns 0.0452272535 / 0.0467149231 / 0.0468693353; col 3 → −0.0172295440 | identical | ✓ |
| moduleC_certify.py | 40 | R_fd/R_det ratio 1.0 at both pins; flat sector −1.7e−40 / −7.6e−14; Δ₃, κ, panel 2.2762→2.5225268; slope +0.18 | identical | ✓ |
| verify_C_dps80.py | 85 | δ=1e−3: C = −0.0343561541791219860831108814584704275…, binding bound 5.4e−31; δ=1e−2 control deviation 2.27e−23 < 5.4e−22 | identical | ✓ |

The dps=85 rerun reproduces the v0.3 gate record digit-for-digit, including both error-model numbers cited in registration v0.3/v0.4 and draft Table A5.

## 3. Independent checks (none reuse program routes)

### 3.1 Gaussian pin of the Module C curvature formula — PASS, exact

The candidate formula R = −N/(2 (det g)²) with N the 3×3 second/third-partial determinant was evaluated on a closed-form benchmark with known answer: the normal family N(μ, σ²) in natural parameters (t₁, t₂) = (μ/σ², −1/(2σ²)), Hessian potential ψ = −t₁²/(4t₂) − ½log(−2t₂), whose Fisher geometry has constant scalar curvature −1 in the sphere-positive convention. Result: **R = −1.0 exactly at all four test points** ((0,−½), (0.3,−0.7), (−1.2,−0.25), (2.4,−3.1)); flat-product control ψ = −log x − log y returns R = 0 exactly. This pins formula **and** sign convention analytically — a known-answer test independent of the finite-difference Christoffel route in moduleC_certify.py — and retroactively hardens the entire Grok adjudication: the refutation's sign and magnitude axes now rest on a closed-form benchmark, not only on FD numerics.

### 3.2 Proposition 2.3 as stated — PASS, with tail-tracking

The proof document's claim c_k = −(k+1)[T(m) + NT(m)], m = k+2, tested in its **stated** form (not the Z(m) rearrangement that verify_h13b_gate.py tests): trivial part by the closed form (−1)^m[(1−2^{−m})ζ(m) − 1], nontrivial part by direct summation of 2 Re((ρ−1)^{−m}) over the first J = 200 zeros (mpmath zetazero), against density-based tail budgets.

| k | pred from zeros | locked c_k | \|diff\| | tail budget | inside |
|---|---|---|---|---|---|
| 0 | −0.19166792635 | −0.18754623284 | 4.12e−3 | 4.13e−3 | ✓ |
| 1 | +0.10337728677 | +0.10337726407 | 2.27e−8 | 9.41e−6 | ✓ |
| 2 | −0.04425495377 | −0.04425497648 | 2.27e−8 | 2.29e−8 | ✓ |
| 3 | +0.01809791155 | +0.01809791155 | 2.79e−13 | 5.67e−11 | ✓ |
| 4 | −0.00723397602 | −0.00723397602 | 1.39e−13 | 1.41e−13 | ✓ |

The k = 0 case sits at 99.8% of budget, so it was escalated to a tail-tracking test: at J = 200/300/400 the residual is 4.12e−3 / 3.20e−3 / 2.66e−3 against budgets 4.13e−3 / 3.21e−3 / 2.66e−3 — **ratio 0.998, 0.998, 0.999**. The residual does not merely sit inside the bound; it *is* the tail, to three digits, at every J. Same diagnostic standard the program applies to its own lemma22 table (error resolved by the next zeros in the queue), now satisfied by the proposition in its full-zero-set form.

### 3.3 Module C constants by independent route (mp.primezeta, k ≤ 400, dps = 40) — adjudicates E3

- Replication of the document's k ≤ 59 cutoff: 5.045188185014424317143705 — matches the printed Δ₃(1) = 5.0451881850144243171 through all 20 printed digits (diff 4.4e−20). **The document printed its truncated sum.**
- Tail-controlled value (k ≤ 400, tail < 1e−110): **Δ₃(1) = 5.04518818501443066970…** Truncation error of the k ≤ 59 sum: 6.35e−15.
- Consequence: the printed value is correct to 14 significant figures; printed digits 15–20 are truncation artifacts. True amplitude **Δ₃(1)/2 = 2.52259409250721533485…** (document printed …072121586, same defect). E3 confirmed.
- κ = 0.83250321174454033893 vs document's 0.83250321174454 — all 14 printed digits correct (diff 3.3e−16). A₀ = −0.31571845205389… consistent. The five-digit panel certification (2.5225268 vs amplitude) is untouched — the defect lives three orders of magnitude below anything the certification claims.

### 3.4 Registered constant C — PASS inside budget

Independent route: dps = 56, split δ = 1e−4 (vs the gate's 1e−3/1e−2), subdivision [1+δ, 1.2, 1.6, 2] (vs [1+δ, 1.5, 2]). Result: C_ind = −0.03435615417912198608311088145844761509927; **|C_ind − C_reg| = 2.24e−32**, a factor ~27 inside the registered ±6e−31. Caveat, against the auditor's own run: the direct integrand √I − 1/ε near β = 1+1e−4 loses ~9 digits to cancellation, which mpmath's quad error estimate (1e−60) does not capture; C_ind's digits below the registered budget are therefore noise, and where C_ind and the dps=85 gate value disagree (at the 1e−32 scale), the gate value is the more trustworthy. Three routes — the v0.3 gate record, today's bit-exact dps=85 rerun, and this independent run — now agree within 2.3e−32. The registered 31 digits and the ±6e−31 budget stand exactly as written.

## 4. Proof-audit findings

### 4.1 Lemma 2.2 / Proposition 2.3 document — SOUND

All four parts verified line by line. (ii): the Hadamard/genus-1 setup, the elementary estimate |Log(1−w)+w| ≤ |w|²/(2(1−|w|)), the constant-difference branch argument on the simply connected D₃, and the Fubini interchange under Σ|ε_j|⁻² < ∞ are each correct as written. (i): the radius-≤-3 argument via identity-theorem extension to e^G and F(−3) = 0 is airtight. (iii): term-by-term double differentiation justified by local uniform convergence. (iv): the trivial-tail integral comparison (worst case m = 2: 0.0918 ≤ 5/49 ✓), the factorization |ρ−1|^{−m} ≤ 14^{−(m−2)}t^{−2}, the majorant N(t) ≤ (t/2π)log t for t ≥ 14 (checked against Backlund-type bounds at t = 14; margin linear vs logarithmic error), and ∫₁₄^∞ t^{−2}log t dt = (log 14 + 1)/14 → Σ_ρ t^{−2} < 0.17 all verified by hand. The referee surface (R1–R7) is accurate; R5's claim that |t| ≥ 5 would suffice with constant 0.17·5^{−k} also checks.

Recommendation (non-blocking): state explicitly, with the citation, that (s−1)ζ(s) has order 1 — the fact is in Titchmarsh §2.12 via the functional equation and Stirling, and the document uses it, but a referee will want the sentence rather than the implication.

### 4.2 Theorem 3.1 document — SOUND

(i): compactness/principal-branch argument for analyticity of g on a neighborhood of [0,1] is correct; second-order vanishing of √(1+w) − 1 gives the removable singularity and g′(0) = c₀/2. (ii): the integral decomposition is exact. (iii): composition coefficients g₁ = c₀/2, g₂ = c₁/2, g₃ = c₂/2 − c₀²/8 re-derived independently; the C⁴-remainder argument is right and the displayed numerical values (+0.046886558210091, −0.017229544011064, +0.00663105) are correct. (iv): correct. Remark 3.2's chart computation ds = (1 + (c₀/2)e^{−2u} + O(e^{−3u}))du verified; the asymptotic-isometry statement is the genuinely invariant content and is correctly distinguished from the chart-dependent Var(E) divergence. The verify_thm31.py table confirms a **third** order (t₄) that was never separately registered — an unforced over-determination in the program's favor.

Blemish (cosmetic): the "+ ⋯" in the proof of (iv)'s distance display is notational slop; one substitution sentence fixes it.

### 4.3 Proposition 4.1 (note draft §4) — SOUND

(a) Poles of the finite bosonic product solve p^{−β} = 1 ⇒ β = 2πik/log p; fermionic zeros solve p^{−β} = −1 ⇒ β = iπ(2k+1)/log p; both loci on Re β = 0, the bosonic product zero-free where finite. (b) Distance from β₀ ≥ 1 to −2n is β₀ + 2n ≥ 3 with equality only at (1, −2); nontrivial zeros at distance ≥ t₁ > 14. (c) follows. The inoculation paragraph draws exactly the licensed conclusion and no more.

Recommendation (one sentence, referee-surface): distinguish the **finite-prime (mode) truncation** used here from **integer truncation** (partial sums Σ_{n≤N} n^{−β}), whose zeros — per the Turán partial-sum literature — are *not* confined to Re β = 0 and do enter the strip. The no-go is specific to the physically natural prime-grand-canonical regularization; the draft's scoping is correct but currently implicit, and this is the first escape hatch a referee will probe.

### 4.4 Module C derivation — SOUND, with errata E1–E3

Lemma C.1's row-cancellation flatness verified symbolically (ψ₁ = e^y φ(x) forces row 3 ≡ row 1 in N); the structural consequence — the singular sector cannot source curvature — is correctly drawn. The expansion's partials, the exact cancellation of singular terms in Row3 − Row1 (k² − k vanishes at k = 1, so the difference row is analytic), the cofactor leading orders C₁ ~ −z²L/ε², C₂ ~ −2z²L/ε³, C₃ ~ −z²/ε⁴, the dominance N = −Δ₃z²/ε⁴(1+O(εL)), the positive sign, and the LOCKED profile R = Δ₃/(2z²(L+κ)²) were each re-derived independently. The Gaussian pin (§3.1) closes the one gap in the original certification (formula pinned only against FD at two points). The H3.2/H3.3 demotions and the three-axis external refutation are correctly logged in v0.4.

## 5. Errata register

**E1 (Module C §3, one character).** det g relative correction stated (1+O(ε²L)); the −2zb/ε cross term in ψ_xxψ_yy − ψ_xy² makes it O(ε/L). Non-propagating — absorbed by the final O(1/L) envelope in R; the LOCKED line is unaffected.

**E2 (Module C §5, wording).** "converging at the derived O(1/L) rate" — the panel residuals along z = 1 (successive deltas 0.246, 0.036, 0.0047, 6.0e−4, 6.7e−5; ratios ≈ 6.8–9) track **O(εL)**, not O(1/L): κ-inclusion removes the genuine 1/L term by construction, which is exactly why the (L+κ)⁻² profile works. Reword to "residuals at the O(εL) level, as expected once κ absorbs the 1/L correction"; bare R·L² converging far slower stands as the evidence that κ is real.

**E3 (Module C §3/§5 and the v0.4 changelog, precision).** Δ₃(1) printed to 20 digits from the k ≤ 59 sum; digits 15–20 are truncation artifacts (tail 6.35e−15). Correct values: Δ₃(1) = 5.04518818501443066970…, amplitude 2.52259409250721533485…; or truncate the print to 14 sf. κ's 14 printed digits are all correct. The v0.4 changelog repeats the defective digits and should carry the correction in v0.5. Five-digit certification unaffected.

**E4 (registrations v0.3 and v0.4, hygiene).** Header line 1 ("Hypothesis Program v0.2") and the end-line ("End of pre-registration v0.2") were never bumped in either version; the changelog is authoritative but the version string contradicts it. Fix both strings in v0.5.

**E5 (note draft conformity, three items).** (a) Line 4 cites "Pre-Registration v0.3" — stale; cite v0.5 or "current". (b) Footer says "the registration bumps to v0.4" — stale; v0.5. (c) §5 Outlook describes Module C as "behind a derive-then-lock gate on its exponent" — that gate closed in v0.4 with R → 0⁺. Either update the outlook to report the result (which changes the note's claim surface and arguably strengthens it: a computed instance where Ruppeiner phenomenology fails for a limiting-temperature transition) or keep §5 silent on C; as written it misstates the program's own state of record. Author's call — flagged, not made. Positive conformity findings: the abstract correctly dropped the Darboux wording in favor of the exact-representation form; NO-RH-CLAIM sentences present in §2 framing context and §4; AI disclosure present; Appendix A tables match reproduced script outputs.

**A1 (auditor's own).** audit_independent.py parsed its locked reference constants at mpmath's default dps=15 (constants defined before precision was raised), so PART 4's printed "registered" line was a parse artifact and the printed |diff| = 1.85e−18 was wrong. Caught in-session; comparison redone with a dps=60 parse (result in §3.4); PART 2 conclusions unaffected (all diffs ≥ 1.39e−13 ≫ 1e−16 parse error); PART 3 unaffected (comparisons ran on internally computed high-dps values). Script shipped corrected.

## 6. Verdict and authorized flips

Adversarial audit PASS. Flips authorized for the v0.5 bump: H1.3a [E → prove] → **[V]**; Lemma 2.2 → **[V]**; Proposition 2.3, full-zero-set form as stated → **[V]** (the Z(m) rearrangement was already [V] via the v0.2 gate; the direct-sum check in §3.2 closes the as-stated form); Theorem 3.1 → **[V]**; Proposition 4.1 / H4.1 → **[V]**. Module C derivation stands as certified, with E1–E3 corrections to be folded into a v0.2 of that document.

## 7. Draft changelog block (v0.4 → v0.5, ready to lock)

> v0.4 → v0.5 (2026-06-13): Adversarial audit executed (external auditor: Claude/Anthropic; certificate T1_adversarial_audit_v0_1.md, archived with execution receipts). All six companion scripts reproduced exactly, including the dps=85 H1.4 gate. Independent checks: Module C curvature formula and sign convention pinned analytically against the closed-form Gaussian benchmark (R = −1 exact at four points; flat control exact); Proposition 2.3 verified in its as-stated full-zero-set form by direct summation over the first 200–400 nontrivial zeros with residual/tail-budget ratio 0.998–0.999 at k = 0 and all k ≤ 4 inside budget; Module C constants recomputed by an independent primezeta route to k ≤ 400; registered constant C corroborated by an independent dps=56 quadrature (different split, different subdivision) to 2.24e−32, inside the ±6e−31 budget. FLIPS: H1.3a → [V]; Lemma 2.2 → [V]; Proposition 2.3 (as stated) → [V]; Theorem 3.1 → [V]; Proposition 4.1/H4.1 → [V]. ERRATA logged from the audit: E1 — Module C det g correction is O(ε/L), not O(ε²L) (absorbed; LOCKED result unaffected); E2 — Module C §5 convergence wording corrected to O(εL) (κ absorbs the 1/L term by construction); E3 — Δ₃(1) corrected to 5.04518818501443066970… (prior print was the k≤59 truncated sum, wrong from the 15th significant figure; amplitude 2.52259409250721533485…; κ unaffected; five-digit certification unaffected; this changelog's v0.4 entry inherits the correction); E4 — version strings in the header and end-line, stale at "v0.2" since the v0.2→v0.3 bump, fixed. Note draft conformity items (stale registration citation, stale footer bump target, §5 Module C status) recorded for draft v0.2. Auditor's own erratum A1 (constant-parse precision in the audit script) caught and corrected in-session; conclusions unaffected. Recommendations carried as pre-submission obligations: explicit order-1 statement for (s−1)ζ(s) with Titchmarsh §2.12 citation in Lemma 2.2; one-sentence Turán/integer-truncation contrast in Proposition 4.1; notation fix in Theorem 3.1(iv). No constants of Modules A/D altered beyond the E3 digit correction in Module C's record.

## 8. Residual pre-submission obligations

Reference page-range and edition checks (already flagged in the draft footer). The three referee-surface recommendations of §4 (order-1 sentence; Turán contrast sentence; (iv) notation). E5(c) decision on §5 Outlook. Module C derivation v0.2 incorporating E1–E3. Optional, cheap, and worth it: fold the §3.2 tail-tracking table and the Gaussian pin into the note's Appendix A — both are one-table additions that pre-answer the two most likely referee probes of Proposition 2.3 and the Module C formula respectively.

— End of audit certificate v0.1. Companion: audit_independent.py (corrected per A1).
