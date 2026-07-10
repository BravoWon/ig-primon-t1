# PREREG — Gate C1g: Garfinkle Section III — the outermost gridpoint rides the marginal ray
**Registered 2026-07-10 before execution. Cosmos arc, PR #13. Successor to C1f per its costed
rung. The paper's own prescription (gr-qc/9412008 §III): "choosing the outermost gridpoint to be
the null geodesic that just barely hits the singularity... This whole process is then iterated
(a few times)." C1f proved in triplicate that WITHOUT this, fixed-VMAX N=800 cannot resolve the
echo ladder in any observable class. C1g implements it.**

## Protocol (fixed)
Configuration = (N=800, CFL=0.1, v_out); grid = linspace(v_out/N, v_out, N); initial pulse = the
SAME functional form as C1c (h = p·e^(−z²)(3r²−2r³z/σ), R0=2, σ=0.5) truncated at v_out — each
v_out is a NEW discrete family: **`p*` re-bisected to the f64 floor per configuration** (C1f
receipt: instrument changes shift p* by ≫ the probed ε). Rays outside the marginal ray cannot
causally influence the critical point; truncating them re-allocates the whole grid to the echo
region's domain of dependence.

**Iteration (instrument tuning, NOT verdict data):** start v_out⁰ = 3.0 ("slightly higher than
the estimate" — flat estimate of the marginal ray ≈ u*/2 ≈ 2.5). Per iteration: bisect p*; run
deep subcritical (ε = 1e−12) with trace collecting (u, h₁, r_outer); measure the turnaround u_t
(min crossing interval) and r_outer(u_t); refine v_out ← v_out − r_outer(u_t) +
max(0.1·r_outer(u_t), 0.02). Stop after 3 refinements or when r_outer(u_t) < 0.05. The FINAL
configuration feeds the verdicts; the penultimate configuration (already bisected) serves as the
free ray-allocation control. Bisection bracket auto-widened if truncation moves p* out of
(0.005, 0.1) (checked, doubling hi up to 0.4).

## Termination amendment (2026-07-10, pre-verdict, disclosed — the it0 nm)
The protocol's first attempt nm'd at it0: the C1c m_out early-exit (an anti-stall device tuned
to the VMAX=6 grid) fires DURING the echo phase on truncated grids — the bounce flux crosses the
near-in outer boundary early; measured: exit at u = 4.857 < u* with only 2 crossings banked, at
both ε = 1e−12 and 1e−6. It can also mislabel barely-supercritical runs (MOTS forms ≈ u* ≈ 5.0),
so it0's p* was discarded. Replacement termination (crossing-TIMING based, amplitude-free, the
gate's native currency): dispersal ⇔ no h₁ sign change for 1.5 u-units after the last one
(u > u_lc + 1.5); the m_out exit survives only for runs with no crossings at all (far
subcritical — no ladder to clip). bh detection (MOTS) untouched. All configurations re-bisected
under the amended termination.

**Termination amendment 2 (2026-07-10, pre-data, disclosed):** amendment 1's u_lc+1.5 dispersal
clock is correct but reintroduced the stall in slow motion — post-echo regrids keep halving
spacings, so the 1.5-unit tail march cost 10–15 min per disp-side run (3.8 CPU-hours of it0
bisection with zero output; measured). Fix: the regrid FREEZES once crossings have stopped
(u > u_lc + 0.2) and no MOTS is imminent (min ḡ/g > 5×threshold) — the grid then drains in O(N)
steps. Labels unchanged (bh fires in the spiral where crossings are ongoing and the freeze never
engages; disp reached faster). Bisection progress is now logged per iteration.

**Termination amendment 3 (2026-07-10, pre-data — the near-critical hover hole):** amendment 2's
freeze guard (min ḡ/g > 5×THRESH) blocked freezing exactly where bisection spends its expensive
probes: near-critical dispersals hover at mots 0.04–0.1 after closest approach — no bh, no
freeze, du halves geometrically (one probe burned 45 min; measured, killed). Guard loosened to
2×THRESH, plus a quiet-trigger: 20k consecutive steps with no h₁ crossing AND mots > 1.5×THRESH
freezes regardless (a collapse in progress is never quiet — the echoes are crossings; MOTS
detection continues every step on the frozen grid). Labels unchanged; the bisection reruns from
scratch under the single consistent instrument.

**Protocol amendments 4 (2026-07-10, pre-config-data — the it0 nm's, all measured):**
(A) *Iteration direction:* v_out⁰ = 3.0 sits BELOW the marginal ray — the flat estimate u*/2
ignores gravitational focusing (rays arrive early, so the marginal ray starts HIGHER): measured,
the v_out=3.0 grid drains at u = 4.859 while the cascade runs to u* ≈ 5.01, leaving 2–4
crossings. The protocol now approaches from above per the paper's own "slightly too large":
few-events ⇒ raise v_out ×1.06 and re-bisect (≤3 raises, else nm), then refine downward as
registered. (B) *A5b flicker rejection:* the drained-endgame grid flips the 4-point h₁ fit for
hundreds of noise "crossings" (spaced ~1e-9 in u) that hijack the turnaround argmin; events now
require grid extent r_outer > 1e-2 (instrument-capacity floor, amplitude-free) plus ≥5-step sign
runs. (C) *MOTS r-floor:* r[MOTS] > 5e-4 — a "horizon" at r ~ 1e-8 on a drained 8-ray grid is
fit noise (the family's banked mass floor is 1e-3-scale); one deep run false-triggered exactly
this way while a 4e-13-neighboring run did not. (D) *ε-backoff implemented* as registered
(deep-run bh at the floor ⇒ retry at 3.16× ε, attempts logged), and the deep-run status string
is now printed (the events=0 line was undiagnosable from the log alone — observability receipt).

**Termination amendment 5 (2026-07-10, pre-config-data — the freeze ate the ladder):** amendment
2's u-based freeze term (engage 0.2 u-units after the last crossing) fires INSIDE the cascade:
the ladder's own inter-crossing gaps reach 1.04 in u (measured: 3.62→4.65), so the regrid froze
between echoes and degraded axis resolution exactly where the next crossing needed it (it1
receipt: crossing 3 at u≈4.95 unresolved; also measured independently: v_out=3.18's outermost
ray dies at u=4.931 < u* — still below the marginal ray, so the raise was correct too). The
u-based term is RETRACTED; the step-based quiet trigger (20k crossing-free steps + MOTS guard)
is the sole freeze criterion — it adapts to du naturally (early gaps ≈ 3.5k steps: no fire;
post-cascade drain: fires). Freeze timing alters evolution dynamics ⇒ ALL configurations
re-bisect again under the final instrument. The relaunch RESUMES at v_out = 3.37: the
below-marginal refutations of 3.0 and 3.18 rest on measured outermost-ray ARRIVALS (4.859 and
4.931, both < u* ≈ 5.01) — ray trajectories are regrid-independent, so those receipts carry.

**Protocol amendment 6 (2026-07-10, pre-config-data — arrival-targeted ascent):** measured
arrivals 4.859 / 4.931 / 4.955 at v_out = 3.0 / 3.18 / 3.37, gains collapsing ~3× per rung: the
focusing PILE-UP — d(arrival)/dv → 0 approaching the marginal ray from below, so fixed-factor
raises crawl while the deep ladder (crossings at u = 5.005–5.0095) sits on the far side.
Amended ascent: JUMP the pile-up to a safe rung (v_out = 4.5, beyond any plausible v_marg by
≥ 0.8; escalate to 6.0 = the untruncated grid, a ladder-complete configuration by construction,
if 4.5 still starves), then descend from above via the REGISTERED refinement (v_out ← v_out −
r_ut + margin — arrival-informed by construction, since r_ut measures the idle grid at the
turnaround). A descent step that undershoots into few-events splits v_out against the last good
configuration (bisection on v). Ascent budget 4; arrival now logged per iteration.

**Amendment 7b (2026-07-10, pre-config-data — overflow-cascade classification):** a trisection
probe crashed the worker pool with LinAlgError (SVD non-convergence in the 4-point axis fit):
the g-clip bounds g but not ḣ = (g−ḡ)(h−h̄)/2r, so h can go non-finite one step before MOTS in
violent collapse. Non-finite field state now classifies bh in-loop (the overflow cascade is
collapse-side by receipt — it is the same warning family that accompanies every bh spiral), with
a worker-level exception fallback to bh. Banked p*'s unaffected: the sequential bisections never
reached this branch (no crash, no non-finite label), so their outcomes are bit-identical.

**Mid-gate disclosures (2026-07-10, from PR #13 review — CodeRabbit; before any verdict was
computed; NO tolerance or rule changed):**
(i) *Bracket guard:* the registered auto-widening only handled p* escaping ABOVE hi; a p* below
lo would have produced a silently wrong bisection, not a failure. Guard added to the code (lo
must disperse, halving down to 1e−4, else nm). The running it0 log visibly brackets on both
sides (disp 0.0109 / bh 0.0139), so no collected data is affected.
(ii) *Verdict-4 corner, disclosed as registered:* with the banked P̄ = 4.29 frozen, verdict 4's
two clauses jointly admit only Δ ∈ [3.20, 3.42] — it CANNOT pass if Δ lands exactly on the
3.4453 anchor (the γ_A clause misses by 0.003). This corner was unexamined at registration; it
encodes the banked P-tension (P̄ is ~7% below the anchor-predicted Δ/(2γ) = 4.61). Tolerances
remain FROZEN per mid-gate rule; verdict 4 is to be read accordingly: it tests closure among OUR
banked measurements, and a Δ at the exact anchor would honestly fail it.

## Event extraction (A5 re-registered INBOUND-ONLY — the C1f contamination channel)
Crossings (u > 1.0, interpolated) + inter-crossing extremum times (C1f A7), but the event set is
cut STRICTLY INBOUND: turnaround = global-minimum crossing interval; events = crossings/peaks
strictly before that interval's second endpoint; per family drop first (transient) and last
(squeeze) — C1f's A6. Joint 2-family ladder fit (C1f `fit_timing2`, unchanged — calibration
banked there under A8, recovery within ±0.25 on adversarial synthetics, 0.057 at 6+5 events).
Qualifying run: ≥ 4+3 events post-trim.

## Verdicts (fixed)
1. **Instrument receipt — the ladder lengthens:** the final configuration yields ≥ 10 inbound
   post-trim events (families summed) at ε = 1e−12. (C1f's inbound-only equivalent was ~7; the
   entire point of §III is more resolvable echoes.)
2. **Δ direct:** median Δ over ε ∈ {1e−11, 1e−12, 3.16e−13} at the final configuration within
   **3.4453 ± 0.25**. (Same-trajectory caveat as C1f: the three runs are one approach; the median
   is one measurement, stated as such.)
3. **The control C1f died on, doubled:** sampling |Δ(CFL=0.05, own re-bisection) − Δ(CFL=0.1)| ≤
   0.15 AND ray-allocation |Δ(final v_out) − Δ(penultimate v_out)| ≤ 0.15. Signed drifts
   reported. Fails ⇒ Δ NOT converged, no measurement claimed (verdict 2 voided as in C1f).
4. **Closure:** implied γ_B = Δ/(2P̄) with the banked ratio P̄ = 4.29 (C1d/C1e spread 4.28–4.30)
   within **0.374 ± 0.03** AND |implied γ_B − γ_A(0.3681)| ≤ 0.03. (Compound tolerance: ±0.15 in
   Δ propagates to ±0.017 in γ; band set at 0.03 pre-execution.)

## Non-measurement discipline
As the arc: nm on < 4+3 events; verdict 2 voided by verdict-3 failure; iterations that fail to
bracket p* or whose deep run goes bh (bisection-floor contamination) are listed and the
iteration retried with the previous v_out + 20% margin; if the protocol cannot reach
r_outer(u_t) < 0.4 in 3 refinements, the gate is nm with the wall mapped (v_out floor).
No counting route (the C1f cliff is an N=800 sampler property; honest omission).

## Honest scope
This IS Garfinkle's 1995 method, finally implemented whole; Δ = 3.4453 is Gundlach's. Occupied.
Deliverable = the completed re-derivation receipt (the arc's three failure-modes each re-derived
one piece of the 1995 toolchain; C1g assembles them and tests whether the ladder opens), and —
if verdicts pass — Δ measured without AMR at N=800, the claim C1e retracted and C1f voided.
