# Phase 1 Protocol — Real-Net Instrument and its Control (v0.1, 2026-06-16)

**Status:** `[PROTOCOL]` — companion to `PREREGISTRATION_genuine_branch_placement_v0_1.md`. Specifies the
real-net instrument **and its validating control on paper, before any real-net chain runs** — the same
control-before-scan rule that gated Phase 0. Frozen before implementation.

---

## 0. What Phase 0 earned, and what it did NOT
Phase 0 (SK) validated the **detector**: FSS on `χ = N·Var(q)` detects a known divergence (positive control
fired), saturates where there is none (negative control), and the equilibration guard flagged the one hard
point instead of rubber-stamping it. **But SK tested neither risk that decides the real-net answer.** SK has a
*trivial* overlap (spin configurations) and equilibrates in a blink under Glauber. The two things that were
ever in doubt —

1. a meaningful, **permutation-invariant** overlap in **function space**, and
2. **real-net equilibration** (the killer confound) —

SK has neither. Phase 1 must build the control for *those*. The detector is not the instrument.

## 1. The function-space overlap (never weights)
**Probe set.** Fix `D`, `|D|` held-out inputs from the data distribution, frozen for the whole experiment.

**Readout.** For replica `a` and input `x`, `φ_a(x)` is a normalized scalar/vector readout of the net's
**output** (logits), centered across `D` and unit-normalized. Function-space ⇒ permutation/scaling invariant
*by construction* — two functionally identical nets with permuted hidden units give identical `φ`. **Weights
never enter.**

**Overlap.** `q_ab = (1/|D|) Σ_{x∈D} ⟨φ_a(x), φ_b(x)⟩`, normalized so `q(a,a) = 1`.

**Susceptibility.** `χ_F = N_eff · Var_{replica pairs}(q_ab)`. The FSS readout is the **trend of `χ_F` vs
width `H`** (saturate vs grow), which is robust to the absolute `N_eff`; `N_eff` is *calibrated against the
controls*, not assumed. (Honest open parameter, flagged.)

**Sanity checks — must pass before the overlap is used at all:**
- `q(a,a) = 1` exactly.
- **Permutation-invariance guard, made executable:** permute the hidden units of replica `a`; `q` must be
  unchanged to machine precision. This is the "weights forbidden" rule turned into a test that *fails loudly*
  if function-space purity is ever violated.
- **Null/limit check:** two independent untrained random-init nets give a small baseline `q ≈ q_null`; two
  samples from a sharply concentrated (low-`T`) posterior give `q → 1`. The overlap must move between these
  limits as `β` varies, or it is not measuring anything.

## 2. The equilibration diagnostic (inconclusive ≠ Branch 3)
The killer (prereg §2): a stuck sampler → low `Var(q)` → fake "`χ_F` saturates → vacuous," which confirms the
Branch-3 prior **for the wrong reason**.

- `K ≥ 4` independent chains from independent random inits, same posterior.
- **Cross-chain agreement:** an R̂-style statistic (between-chain vs within-chain variance) on `⟨q⟩` and
  `⟨q²⟩`; require `R̂ ≤ 1.1`.
- **Autocorrelation:** integrated autocorrelation time `τ_int` of `q` must satisfy `chain length ≥ 50·τ_int`.
- **HARD RULE.** If cross-chain `R̂` fails **or** `τ_int` is unresolvable → the point is **`INCONCLUSIVE`,
  logged explicitly as "cannot equilibrate," and is *not* Branch 3, *not* a χ-saturation.** A null is a
  result; a fake null is a mirage. This rule is non-negotiable and applies even when the result would flatter
  the prior.

## 3. The mode-connectivity cross-check — the LOAD-BEARING control
**Design refinement over prereg v0.1:** connectivity is promoted from a §4 afterthought to **the validating
control that gates the scan.** Rationale: the prior leans Branch 3 *and* a stuck chain fakes Branch 3, so an
**independent** tiebreaker is mandatory, and direct connectivity is it.

- Obtain two solutions in the same region (two posterior samples, or two independently trained minima at the
  operating `T`).
- **Align** replica `b` to `a` modulo permutation: Ainsworth weight/activation matching, or the Entezari
  permutation search (the verified `ainsworth2023` / `entezari2022` of Paper 3).
- **Barrier:** along `w(t) = (1−t)w_a + t·π(w_b)`, `t∈[0,1]`, measure
  `Δ = max_t L(w(t)) − max(L(w_a), L(w_b))`. Low `Δ` (post-alignment) = mode-connected; high `Δ` = not.
- **Agreement criterion (this is the control):**
  - connected (low `Δ`) **and** `χ_F` saturates (equilibrated) → **coherent Branch-3** signal (real, not stuck).
  - high `Δ` **and** `χ_F` grows (equilibrated) → **coherent Branch-1/2** signal.
  - **Disagreement** (connected but `χ_F` grows; or high `Δ` but `χ_F` saturates) → **the instrument is
    broken.** Report *no branch*; debug before any scan. This branch of the table is the mirage-catcher — the
    thing that would have killed the +0.93 before it shipped.

## 4. Only then — the FSS scan (prereg §4)
With §§1–3 passing on the controls, sweep width `H` on the real posterior; read `χ_F`'s `H`-trend
(saturate = Branch 3 / grow = Branch 1/2). Every point carries its equilibration verdict **and** its
connectivity cross-check. Single-`H` inadmissible (the F-genuine lesson); `INCONCLUSIVE` points are excluded
from the trend and logged, never silently dropped.

## 5. The order is the result-vs-mirage decision
**§1 → §2 → §3 are the CONTROL; §4 is the SCAN.** No scan result is admissible unless its region passed
1–3. Phase 0 controlled the *detector* (SK); this controls the *real-net instrument* (function-space overlap
+ equilibration + independent connectivity). Build and validate in this order, or the scan produces a number
with no right to be trusted.

## 6. Open design choices (to pin before build, honestly flagged)
- **Task/model.** Smallest genuinely singular + over-parametrized setting: teacher–student regression (clean
  overlap, known structure) **vs** a small classification task (closer to "real," messier overlap). Lean
  teacher–student first — the overlap and the AT analogy are cleanest there.
- **Readout `φ`.** Logit cosine vs prediction-margin sign vs centered-output inner product.
- **`N_eff` normalization** for `χ_F` (calibrated on the connectivity-controlled region).
- **Sampler.** SGLD vs full-batch Langevin vs HMC; the equilibration budget that makes §2 satisfiable at the
  smallest useful width.
These are pinned in a v0.2 of this protocol *before* a real-net chain runs, not during.

## 7. Adjudication plan (when back, fresh head) — the run-length settler
The mapping sweep (§4 grid) varies `H, n_data, β, seed` at a **single fixed chain length**. That can
*map* where the stuck-vs-glassy entanglement is severe (large `q_within − q_between`, high `fs_R̂`), but it
**cannot settle stuck-vs-glassy on its own**: broken ergodicity (genuine RSB) and a merely stuck sampler look
*identical at any finite time*. A fixed length maps the problem; it does not decide it.

The settler is **run-length dependence**. For the handful of grid points where the entanglement is severe,
re-run at **2–3× the chain length** and watch `(q_within − q_between)`:
- gap **shrinks** with run length → *slow-but-ergodic* (the sampler was just slow; not genuine RSB).
- gap **stays fixed** with run length → *stuck OR truly broken ergodicity* — the genuinely hard pair, which
  then needs the §3 connectivity cross-check + independent longer chains to push on, and `INCONCLUSIVE` where
  it still can't be separated.

Adjudication order: (a) read the grid → locate severe-entanglement points; (b) run-length sweep those points
(2–3×) → shrink vs fixed; (c) where fixed, bring the §3 connectivity 2×2 to bear; (d) only then a branch read,
region by region, `INCONCLUSIVE` logged wherever stuck-vs-glassy cannot be separated. Fresh head, not the tail
of a long day.
