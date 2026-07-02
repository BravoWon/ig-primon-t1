# Coherence Optimization Standard — the ladder, executed. Final report.

*Spec v0.1 asked for every node able to fail, decision rules fixed in advance, and execution as the only
arbiter. Executed 2026-06-28 → 2026-07-01 on one RTX 5070. This is what actually returned, including the
failures. Verdict up front:*

> **The central thesis is FALSE at this testbed.** Enforcing verified, non-trivial sheaf-style algebraic
> coherence (cosine consistency + operator cocycle over a functional Čech cover) does **not** lower the
> learning coefficient λ̂ beyond what its incidental representation-shrinkage buys. N=10 pre-registered
> gauntlet: Mann–Whitney p=0.145 (fail), repr-norm −30.9% (fail, past the −25% floor), and — decisively —
> **norm-controlled, the arm effect vanishes and flips (+6.3 λ̂ units, exploratory)**. λ̂ correlates with
> repr-norm at +0.77…+0.84 *within both arms*. A structurally-verified null: the constraint was real,
> active, satisfied, non-trivial — and it was not the lever. The spec's own bar ("if the coherence penalty
> does not move λ̂, the thesis is false") was not met.

## Node-by-node record

| node | question | verdict |
|---|---|---|
| **1** | grokking reproducible? λ̂ measurable? | grok ✓ (p=113, ~20k, every run). λ̂ calibration **hard**: v1 ε-runaway (140×); v2 plateau **didn't replicate** (GPU non-determinism); v3 chains drift on the degenerate grokked landscape. **Fix: determinism lock (CUBLAS workspace + deterministic algorithms) → bit-identical runs**; relative-shift instrument validated (drop at grok: −86.9% mean, CV 3.5% across seeds). **λ̂ collapses ~5–7× at the grok transition** — determinism-locked, seed-robust, SLT-consistent (relative, not absolute). |
| **2** | does a per-layer bottleneck (VIB) move λ̂ specifically? | **NULL.** Phase-aligned (T_grok+Δ) + 4-arm isolation: VIB λ̂ 1.7 = deterministic shrink 1.7; mechanism localized by covariate to **representation-norm collapse (−95%)**, which any repr penalty produces; fixed-σ noise (no norm pressure) *raises* λ̂. VIB = shrinkage in an information-theoretic mask. (First 3b run's "−88% IB win" was a time-in-basin + invalid-control artifact — caught.) |
| **3–4** | does topological coherence (consistency + cocycle) flatten the basin? | **NULL (structurally verified).** Design: functional cover at '=' (3 chunks of the final residual — token-window covers provably break against the causal mask), independent heads (anti-trivial Lock 1), cosine consistency (anti-norm-collapse Lock 2), operator-Frobenius cocycle (anti-Z→0 Lock 3), lock-verification diagnostics (‖R−I‖, chunk-cos), blind (α,β) calibration, grad-clip **both arms** (matched treatment), phase-aligned λ̂, N=10, pre-registered 4-criterion verdict. Result above. |

## What survived (the positives, at true size)
1. **λ̂ drops sharply at grokking** (node 1) — a clean, determinism-locked, seed-robust *relative* result on modular addition; consistent with SLT's degeneracy prediction. Absolute calibration remains open (chains drift on degenerate landscapes; the polytope cross-check — spec node 6 — was correctly split off: tractable only on structured proxies, cf. the 2-layer-linear RLCT receipt).
2. **The operator cocycle is a rigid-body constraint on representation scale** (mechanistic sub-finding): cosine-only consistency **leaks** (SGD collapses ‖Z‖ ~−78% and inflates the unpenalized heads); adding ‖R_ca R_bc R_ab − I‖²_F anchors the maps' singular values ≈1 and blocks the collapse (−31% vs −78%). Replicated across phases.
3. **Lock-1 engineering works**: independent random heads force the same algorithmic state into genuinely distinct bases (chunk-cos ≈ 0) with actively-translating restriction maps (‖R−I‖ ≈ 9), while grokking 10/10. Forcing non-vacuous bundle structure is *buildable*; it just doesn't pay in λ̂.
4. **VIB accelerates grokking** on some seeds — but not uniquely (single-seed "unique acceleration" retracted at N=3).
5. **The "lightning" (λ̂=1.37, norms held, locks held, single seed)** was an artifact of the *unstable optimizer regime* (α=0.05, unclipped, multiplicative-gradient blowups killing sibling seeds); under the stable recipe the same seed reads 43.4. Buried with a receipt.

## The instrument lessons (the durable assets)
- **SGLD-λ̂ on grokked (highly degenerate) landscapes is treacherous**: step-size runaway, non-stationary chains, GPU-noise-sensitive plateaus, and **λ̂ ∝ repr-norm (+0.77…+0.84)** — a scale sensitivity that can masquerade as geometry. Any λ̂ claim without a norm covariate is suspect.
- **The confound stack that made the final null believable**: determinism lock → phase-aligned T_grok+Δ (time-in-basin) → blind hyperparameter calibration → param-matching → matched grad-clip → scale-invariant consistency + operator cocycle → lock-verification diagnostics → pre-registered nonparametric verdict → norm-controlled exploratory annex. Six auto-verdicts were overridden by hand along the way ("LIGHTNING → paper" among them).
- **Artifacts caught in this ladder alone**: v1 ε-runaway; v2 phantom plateau; v3 drifting chains under a favorable trajectory; node-2's invalid control (un-grokked seed) and time-in-basin mirage; the n=1 lightning. Every one would have been a false positive under single-run, single-seed, auto-verdict practice.

## Node 6 — the two-estimator λ agreement (the spec's open seam): EXECUTED, POSITIVE

Stochastic route = the ladder's SGLD λ̂; discrete route = the **exact Aoyagi–Watanabe (2005) closed-form
RLCT** for reduced-rank regression `y = BAx`, measured at the constructed most-singular point (zero-padded
rank-r factorization). The instrument story, told straight:
- **v1: non-measurement.** The transformer-scale ε grid never equilibrated on 6–60-param models (relaxation
  ≫ chain length); the stability gate correctly emitted *nothing* (0 valid cells everywhere).
- **Diagnostic (one config, thereby burned):** mapped the three SGLD-LLC systematics — under-equilibration
  (λ̂ low), discretization heat (λ̂ high), γ-confinement (λ̂ low; γ→0 is the definition of local).
- **v2: formal 0/5 — gate mechanics, not physics.** Raw cells tracked theory to 1–18% in every regime; a
  transformer-calibrated drift bar and a wrong-direction ε→0 extrapolation produced the NaNs. Peeking burned
  all six configs for confirmatory use.
- **v3: the fresh-config confirmatory test.** Recipe frozen (γ=10, burn-in 6000, mean of stationary cells);
  seen six demoted to in-sample; verdict counted on **five virgin configs spanning all four AW regimes**
  (λ = 2→11, incl. regime 3, never touched), bar pre-registered at ≥4/5 within 10%.

**RESULT: 4/5 fresh configs AGREE (7.4–9.7% err, stationary chains) — the bar is met.** The fifth
(2,6,2,1) failed only the drift gate; its raw cells read within 3–8% of λ=2. Across all 11 configs,
gate-blind: **λ̂ = 0.889·λ_AW + 0.16, r = 0.992** — near-perfect linear tracking of the exact discrete
RLCT up to one multiplicative calibration constant.

**Caveats, surfaced:** (i) a uniform ~5–13% LOW bias (residual confinement/equilibration) — the validation
holds at 10% tolerance and would fail at 5%; in-sample went 2/6 because the tolerance line cuts through the
bias distribution, not because seen/fresh differ. (ii) (2,2,8,2) is a genuine outlier (18% low, never
stationary). (iii) n=2000, one seed per config, the degenerate point only (the trained-point comparison —
does SGD find the singular stratum — remains open). Partial occupied territory: SGLD-LLC vs DLN theory has
precedent (Furman–Lau); the specific contribution here is the *pre-registered fresh-config protocol* and
the regime-spanning tracking line on our from-scratch, stability-gated estimator.

## Status of the spec
**All six nodes executed.** Nodes 1–5 (node 5's random-structure control subsumed by the noise arm +
norm-channel analysis — the "gain" was never the geometry). Node 6: **positive** — the discrete
combinatorial route to the learning coefficient validated out-of-sample against the stochastic estimator
(the one branch the spec marked "genuinely valuable, genuinely novel"). The bottleneck term: **dropped**
(occupied territory). The coherence terms: **closed as a λ̂ lever at this testbed**; the operator-cocycle
rigidity result stands as the exportable engineering finding. The honest boundary from the spec held
throughout: no metaphysical premise was needed; the central thesis died by its own pre-registered bar, and
the seam it said might pay — paid.

## Reproduce
`slt/node1_grok_llc.py` → `node1_v2_calibrate.py` → `node1_v3_trajectory.py` → `node1_v4_gate.py` →
`node2_bottleneck.py` → `node2_v2a_calibrate.py` → `node2_v2b_main.py` → `node3_topo.py` +
`node3_v3a_calibrate.py` → `node3_v3b_main.py` → `node4_v4a_verify.py` → `node4_v4b_gauntlet.py`.
Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Each script prints its verdict; figures in `slt/*.png`.
