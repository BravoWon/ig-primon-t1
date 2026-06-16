# Preregistration — Empirical Branch-Placement of Neural-Network Posteriors (v0.1, 2026-06-16)

**Status:** `[PREREG]` — design frozen **before any sampler is written**, per the program's
control-before-scan rule. Companion to `T1_genuine_side_singularity_gate_v0_1.md` (v0.3), which reduced the
genuine-side question to a **trichotomy** on the posterior's RSB structure and left the branch placement of
real nets `UNSETTLED` (leaning branch 3). This document preregisters the experiment that would move that
verdict from *leans* to *evidence* — and, equally, the conditions under which the experiment yields **nothing**.

This crosses a line the program deliberately held (gate §8: *"no sampler was ever the bottleneck — the gate
analysis is"*). It is justified **only** by the control scaffold of §3. Without the controls firing, no result
here means anything.

---

## 0. Question and prior
**Q.** Does a real, over-parametrized network's Bayesian posterior sit in **branch 1/2** (continuous-RSB; the
inter-replica overlap susceptibility `χ` diverges at a de Almeida–Thouless point — the genuine Ruppeiner
signature *survives*) or **branch 3** (no RSB / mode-connected; `χ` saturates — the signature is *vacuous*,
routing the measurement to the LLC)?

**Prior (gate v0.3, source-verified):** *leans branch 3, UNSETTLED.* Connectivity evidence is direct; the
over-parametrized glassy claim is analogy-grade or actively negative (Baity-Jesi). The lean is itself a thing
not to overclaim.

## 1. The measurable
**Order parameter — function-space overlap.** Two independent posterior samples ("replicas") `a, b` at inverse
temperature `β`; overlap defined on a **fixed probe set** `D` of inputs, **not in weight space**:
```
q_ab = (1/|D|) Σ_{x∈D}  s( f_a(x), f_b(x) )
```
with `s` a normalized similarity (cosine of logits, or agreement for classifiers). **Weight-space overlap is
forbidden** — permutation/scaling symmetry makes it report branch 3 for a trivial, wrong reason (this is the
entire reason the mode-connectivity literature needs permutation alignment). Function-space `q` is
permutation-invariant by construction.

**Susceptibility.** `χ = N_eff · Var_replicas(q)`. Branch 1/2: `χ` grows/diverges toward a transition.
Branch 3: `χ` saturates to a finite, `β`-independent constant. **The `N_eff` normalization is NOT assumed** —
it is *calibrated in Phase 0* against the perceptron control, where the correct `χ = N·Var(q)` is known.

## 2. The dominant confound — stated first, on purpose
**Non-equilibration mimics branch 3.** A sampler stuck in one basin reports low `Var(q)` → false "`χ`
saturates → vacuous." Because the prior already leans branch 3, *a lazy sampler confirms the prior for the
wrong reason.* Therefore:

> **No branch-3 conclusion is admissible unless (a) the positive control fired, (b) the negative control
> saturated, and (c) the equilibration diagnostic passed.** A saturating `χ` from an unconverged chain is
> `VACUOUS`, not branch 3, and must be reported as a null.

## 3. Control-before-scan (Phase 0 — mandatory gate)
The instrument — Langevin posterior sampling + function-space `χ` + finite-size scaling — is validated on two
**solvable systems with known answers** before any real net. These reuse the program's own archetypes.

- **POSITIVE control (must show `χ` growth).** Negative-margin spherical perceptron (`κ<0`) — the program's
  branch-1 `[V]` system, `χ = 1/λ_repl ~ 1/(α_AT−α)` (receipt `module_L_perceptron_replicon.py`). The *sampled*
  instrument must reproduce `χ` growth toward `α_AT`. **If it cannot detect the divergence it already knows is
  there, the instrument is blind and the program stops here.** This also fixes `N_eff`.
- **NEGATIVE control (must show saturation).** Convex perceptron (`κ≥0`) or a Gaussian ridge model — no
  replicon; `χ` must saturate. Guards against an instrument that "sees" divergence everywhere.

**Phase-0 gate:** proceed to Phase 1 **iff** positive fires **and** negative saturates **and** both
equilibrate (multi-chain agreement). Otherwise: instrument-not-ready null; do not touch a real net.

## 4. Phase 1 — the real net (only after the gate)
- **Model.** Smallest genuinely singular + over-parametrized net (2-layer MLP), small task (teacher–student or
  small classification) so the posterior is samplable.
- **Equilibration diagnostic.** `K` independent chains from different inits; `q`-statistics must agree across
  chains (an R̂-style cross-chain criterion) and the relaxation time must be `<<` run length. **Reported
  explicitly; a fail is a null, not a branch read.**
- **Finite-size scaling.** Sweep width `H` (the singular direction). Read whether `χ` at the candidate
  transition **grows with `H`** (branch 1/2) or **saturates** (branch 3). Single-`H` is inadmissible — the
  F-genuine false positive (refuted only by `H=64→512` FSS) is the standing precedent.
- **Independent cross-check.** Direct mode-connectivity (linear-after-permutation barrier, à la
  Entezari/Ainsworth) on the *same* nets. The bridge predicts connectivity ⟺ `χ` saturation. Agreement
  strengthens the read; **disagreement is itself a finding** (the [E] bridge would be failing).

## 5. Outcomes and what each licenses (preregistered)
| Result | Licenses |
|---|---|
| `χ` saturates + connectivity present + controls fired + equilibrated | **Evidence for branch 3 (vacuous).** Not "proof" — bounded to tested architectures/tasks; the LLC handoff of Paper 3 §5.4 is the live invariant. |
| `χ` grows with `H` + AT-like point + equilibrated | **Branch 1/2 — the genuine signature SURVIVES on real nets.** The bigger surprise, *against* the prior. Would upgrade Paper 3 §5.4 from "argue" toward a finite-`β`/zero-`T` claim. |
| Controls fail, or no equilibration, or χ-trend ambiguous | **VACUOUS.** Report the null; infer no branch. (A fired falsifier is a result.) |

## 6. Honest scope (the wall, stated plainly)
Mean-field RSB intuition on a *finite* real net is an extrapolation; FSS is the only bridge and it is finite.
The result classifies the **tested** nets, not "all DNNs." The branch-3 prior must not pre-bias the read —
which is exactly why §2 makes equilibration load-bearing and §3 makes the positive control a hard gate. We do
not move this wall by wanting to.
