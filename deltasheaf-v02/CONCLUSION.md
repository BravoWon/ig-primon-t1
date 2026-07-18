# DeltaSheaf — Conclusion (CLOSED 2026-07-18: geometry null on every axis; blind spots are importable FACT-GAPS — genuine fact-content recovers them (81.7%) though the real retrieval pipeline imports nothing; ONE geometric positive: ensemble error-detection → selective prediction)

**Thesis tested:** "the delta is the map" / "the boundary layer maps the manifold holes" — that the geometry
of inter-model disagreement carries recoverable, exploitable information. Tested along seven independent axes
(recovery, scale, routing, within-node confidence, detection, inverse geometry, chance-whitened full complex)
plus the rank measurement that closes the space. Refined law under test: *import from a complementary system
recovers the hole.*

**One-line verdict:** the between-model geometry is a genuinely rich rank-7 object that carries no task
signal beyond a difficulty scalar, in any encoding, at any target; the one real, verified positive is that
**plain ensemble confidence detects the strongest model's errors better than the model itself** (+0.026 AUC
cross-model increment, ablation-proven), which converts into a working selective-prediction tool
(+3.9 points at 50% coverage). Detector, not generator; confidence, not geometry.

## What is banked (trustworthy)

**1. v0.2 — sheaf geometry over an LLM ensemble: FALSIFIED (solid).** 5 families (26M–7B), 322 blind-spot
items (0-of-5), ridge-LS restriction maps, cosine readout, 5 seeds. All arms at chance — **map** (`B_edge`),
**displacement** (`C_cycle`), **area** (`C_area`), **volume** (`ctrl_mag`); under MLP, `C_cycle` loses to
its own shuffle. Instrument validated on synthetic (detects a planted map); round-trip: pipeline recovers on
clean (0.37) but not blind-spot (0.21). A genuine null: intra-ensemble disagreement geometry carries no
recoverable pointer to the missed answer.

**2. Strong-model premise: scale-invariant blind spots (solid).** Qwen2.5-7B (~70% full MMLU) on the same
322 = **25.8% = chance**. A bigger *same-kind* model shares the hole; scaling the LLM imports nothing.

**3. v0.4 — nodes of importance (per-item routing): NULL, shuffle-controlled (solid).** Reframed to a
*decidable* target: on the 3848 items where ≥1 model is right, can a map from the boundary-layer projections
+ their deltas pick **which model to trust**? Held-out split, 5 seeds. Result: every router — boundary-delta
map (76.5%), raw stalks (77.2%), both (76.3%), **and the label-shuffle control (76.7%)** — collapses to
"always trust the strongest model" (best-single 76.8%, majority 71.7%). The shuffle control sitting *at* the
map's accuracy is the decisive signature: the input structure carries **no per-item routing signal**. The
map captured 0% of the 23-point headroom (best-single → oracle 100%). *(Reproduce: `build_v4.py`,
`RESULTS_v4.md`.)*

**4. v0.5 — within-model confidence (the orthogonal construct): a REAL signal that FAILS the merit bar
(solid, verifier-hardened).** Re-abstracted off the between-model axis onto a *within*-node relation the
delta structurally lacks: each model's own answer-confidence, extracted by teacher-forcing its stored
reasoned reply (one forward pass, no generation; features = 4-way letter dist, answer logit-margin,
reasoning perplexity).
- **Instrument: the signal is real.** Within each model, confidence separates its own correct from wrong
  answers — `logit_margin` pooled **AUC 0.744** (per-model up to 0.77), vs the between-model geometry's
  AUC ≈ 0.5. These small models *do* know, somewhat, when they're right. This is the first live signal in
  the whole DeltaSheaf arc.
- **But routing fails the merit bar.** "Trust the most-confident model" = 77.0% vs always-best 76.1% on all
  3848 items — **McNemar p = 0.21, not significant**. Learned router +1.66% across 20 splits, **95% CI
  [−0.26%, +2.81%] (includes 0)**. Knowing a model's *self*-reliability doesn't tell you *which minority
  model* to trust when the strongest is wrong.
- **Geometry stays dead.** confidence+geometry **union (74.5%) < confidence alone (78.4%)** — the
  between-model delta adds nothing and dilutes. *(Reproduce: `confidence.py`, `build_v5.py`, `v5_stats.py`;
  `RESULTS_v5.md`, `RESULTS_v5_stats.md`.)*

**5. v0.6 — ensemble error-DETECTION: the arc's one real positive (ablation-verified).** Reframed from
"who is right" (dead) to "is the strongest model wrong" (decidable): a learned combiner over all 5 models'
confidence features predicts phi's errors at AUC 0.805 vs phi's own best 0.777–0.778. The decisive ablation:
phi's own multi-signal recalibration adds **+0.001** (nothing); the **cross-model increment is +0.026,
95% CI [+0.010,+0.040], 100% of 15 splits**; and the other four models ALONE (no phi features) hit AUC 0.691
— the information about when phi errs demonstrably lives in the other nodes. Confidence-weighted fusion is
*suggestive but unbanked* (77.8% vs 76.1%, McNemar p=0.010 raw but Bonferroni×5 ≈ 0.05).
*(Reproduce: `build_v6.py`, `v6_stats.py`, `v6_ablation.py`.)*

**6. v0.7–v0.9 — the geometry is null at DETECTION too, in every encoding (solid).** Pointed the whole v0.2
apparatus at the new decidable target with a label-free difficulty control: raw restriction-map residuals
AUC 0.515 (chance), cycle+area 0.557, and adding geometry to difficulty *subtracts* (−0.008). Inverse maps /
round-trip non-invertibility / asymmetry (v0.8): 0.572 alone, −0.019 when added (0% of splits positive).
The fairest instrument (v0.9) — whiten the FULL cochain complex (10 edges + 10 triangles, 320-D) against the
permutation-null "geometry of chance" — lifts geometry-only to its arc-best **0.656** and proves the complex
genuinely deviates from chance (0.21× isotropic ≈ 5× more coherent than random registration)… and it is
STILL fully subsumed by the difficulty scalar (0.787; Δ −0.011). The structure is real; it is task-inert.
*(Reproduce: `build_v7.py`, `build_v8.py`, `build_v9.py`.)*

**7. v0.10 — the rank theorem (closes the space).** The complete rotation-invariant relational geometry of a
5-model configuration is 15 invariants (10 cosines + 5 norms); the ensembles occupy **~7 orthogonal modes**
at 95% variance (per-item shape ~3–4-D; λ₅≡0). No single mode is informative (best folded AUC 0.547); the
task signal is diffuse across ~5 modes and caps at **AUC 0.606 — losing to one confidence scalar (0.767)**.
Since every probe (maps, cocycles, inverses, whitened complex) is a re-encoding of this same ≤15-number
object, there is no stronger geometric probe left to run: the space is spanned and measured empty.
**Closing theorem: between-model = rank-7 and task-inert; within-model = rank-1 and the only live signal.**
*(Reproduce: `build_v10.py`.)*

**8. SELECTIVE — the positive productized (operational win).** Abstention by the ensemble error-detector:
AURC 0.167 vs 0.186 for phi's own confidence (**Δ +0.018, CI [+0.012,+0.025]**, 15 splits); accuracy at 50%
coverage **90.1% vs 86.2% (+3.9)**, +2.7 at 30%, gains fading to 0 at full coverage exactly as a real
calibration signal should. Operating thresholds per target coverage saved for deployment.
*(Artifacts: `selective.py`, `selective_thresholds.json`, `selective_risk_coverage.png`.)*

**9. v0.3 — retrieval-import: REFUTED on a VALIDATED instrument (solid, 2026-07-18).** History: the 3B
reader degraded on context (−7.3% on EASY — the `/driftwave` round-trip that voided the first null); prompt
fix failed. Resolution ran the checkpointed 7B protocol (`openbook7b.py`), control-first: n=150 abort fired
(Δ−4.7%) but McNemar p=0.19 → pre-registered extension to n=400. **Control PASSED at n=400: 88.0%→86.0%
(Δ−2.0%, p=0.256) — the 7B reader is VALID (the 3B degradation does not persist at scale; the n=150 abort
was noise).** Gate (322 blind spots, chance 25%): **closed 25.8% → open 25.8%, 29 flips each way, McNemar
p=1.000 — a perfectly symmetric null.** Passage-present items sit AT chance (23.0%). **Retrieval-at-inference
does not recover scale-invariant reasoning holes.** Named caveat: passage relevance never well-measured
(gold-substring ~2/322) — oracle-relevant import untested; what is refuted is the real pipeline (Wikipedia
top-hit → context). *(Record: `RESULTS_v3_7b.md`.)*

**10. CODA — the import CEILING (oracle-relevance, 2026-07-18): the holes are REACHABLE.** Pre-registered
ceiling test (`oracle7b.py`): note stating the gold answer appended to context, instrument-check-first.
v1 instrument FAILED informatively (un-instructed notes adopted for only +3.0% on easy items; autopsy: the
7B argues AGAINST "verified" notes when its own derivation disagrees — catch #8). v2 (instructed trust):
instrument 94.0% ✅ → **gate 25.8% → 89.8% with gold notes** — near easy-level (94.0%). Copy-control: wrong
notes adopted 90.4% (accuracy 1.6%) — under instructed trust the reader is a ~90% note-follower uniformly,
so the ceiling measures *mechanical adoption*, exactly what a ceiling should. **The blind spots are
fact-gaps at the interface, not adoption-impossible reasoning-holes.** *(Record: `RESULTS_ceiling.md`.)*
Also banked en route: **CASCADE null** — defer-to-majority on detector-flagged items gains +0.2% (ns);
on the items where phi fails, majority fails too. The ensemble's knowledge is *decline*, not *redirect*
(`RESULTS_cascade.md`).

**11. CODA-2 — FACT-FORM import: the last rung, CLOSED — genuine import, not copying (2026-07-18).** The
89.8% ceiling used answer-STATING notes (copy-rate 90.4% = mechanical). CODA-2 tests the sharper rung: a
3B-generated **encyclopedia FACT sentence** (leak-filtered: no answer/letters), read by the 7B under a
neutral prompt. Instrument corrected (v1 easy≥93% bar was **saturation-invalid** — fact notes can't lift
already-known easy items — catch #9); the valid, unsaturated integration check is the DOWNWARD direction:
**wrong facts on easy items crater 86.0% → 35.0% (gap +51.0%)** ⇒ the reader strongly integrates fact-form
context. Gate: **closed 25.8% → FACT-GOLD 81.7%** (only 8 pts under the answer-form ceiling), and the
copy-control certifies integration-not-deference: **fact-WRONG craters accuracy to 6.8%** (below chance)
while adopting the stated wrong letter only 31.1% (vs answer-form's mechanical 90.4%) — a **75-point swing
driven by fact *correctness***, not letter-copying. **The blind spots are genuinely importable fact-gaps;
real fact content (not just a stated answer) recovers them; the 89.8% was never mere copying.**
*(Record: `RESULTS_factform.md`; `factform7b.py`.)*

**The import law, final form (CLOSED):** "import from a complementary system recovers the hole" —
**SURVIVES as training-time grounding** (terminus, seed-robust, works with *imperfect* signals) and at
inference is now fully resolved: the **context channel is OPEN and carries genuine fact-import** (fact-form
81.7%, integration-certified), while the **real Wikipedia pipeline imported nothing** (perfect null;
passage-present AT chance). So the binding constraint is unambiguously **retrieval CONTENT quality** — not
the model, not the channel, not copying. Better retrieval is a live lever. The next channel (untested here,
pre-registered): **latent-space import via orthogonal Procrustes** between *heterogeneous* models
(`../procrustes-gate/PREREG.md`), with fact-form's 81.7% as the bar to beat. The earlier retracted synthesis stays retracted; the
honest final form: *holes are importable in principle; nothing deployed imported them in practice.*

## Instrument discipline (SEVEN near-misses caught before believing a false result)
1. Degenerate substrate (R_ij=identity → zero residuals) → used ridge-LS maps.
2. Empty retrieval (throttled to 1%) → caught by non-empty diagnostic; fixed to 73%.
3. Degenerate cosine reader → moved to open-book.
4. Degrading open-book reader (−7.3% on EASY items) → voided the v0.3 null (would have banked "import can't
   fill reasoning-holes").
5. **v0.5 fragile positive: "confidence beats best-single 78.4 ± 0.2%."** The ±0.2% was *seed-stability*,
   not sampling error (binomial SE ≈ ±1.3% on 962 items; the +1.6% was ~15 items). An adversarial
   `driftwave-verifier` (fresh context) flagged it PLAUSIBLE-not-confirmed; the McNemar + multi-split recurse
   converted it to an honest null (p=0.21). The sharpest catch — a positive we *wanted* to be true.
6. **v0.6 attribution over-read:** "ensemble predicts phi's errors" nearly banked with phi's OWN features in
   the detector — could have been mere self-recalibration. The verifier demanded the ablation; it showed
   recalibration = +0.001 and the increment genuinely cross-model (+0.026). (Same round, the v0.7 difficulty
   control deflated the *structural* reading: most of the raw detector signal is the difficulty common-mode;
   the honest survivor is the +0.02–0.03 cross-model increment, not "the delta is a structural detector.")
7. **7B control underpowered:** the pre-registered abort fired at Δ−4.7% (n=150) but McNemar p=0.19 — banking
   "7B also degrades" on that alone would itself have been noise-reading. Extended to n=400 with the rule
   fixed before relaunch.

## Standing position
"**The delta is the map**" is dead in every operational form tested — generation (v0.2), scale (7B),
routing (v0.4), confidence-routing (v0.5), detection (v0.7), inverse geometry (v0.8), chance-whitened full
complex (v0.9) — and v0.10 proves the space is *spanned*: the between-model geometry is rank-7 and task-inert,
losing to one within-model confidence scalar in total. The geometric exploration is **FROZEN** (the harness
is kept as an evaluation instrument), and **the thread is CLOSED**: every question it opened has a banked
answer. Durable positives: (a) the **cross-model error-detection increment** (+0.026 AUC, ablation-verified)
and its **selective-prediction tool** (+3.9 pts @ 50% coverage); (b) the program survivor **grounding** —
now sharpened by v0.3's split verdict: *import pays baked into representations at training time, not appended
to context at inference.* Named residuals (open, not blocking closure): fusion (borderline, needs a
pre-registered single-weighting replication); detector generalization off-MMLU / strong leader;
oracle-relevant retrieval (the pipeline was refuted; perfect-relevance import was never reached).

*(Reproduce: `build.py`[+mlp] · ensemble `score/embed` · retrieval `retrieve.py`/`build_v3.py`/`openbook.py`/
`openbook7b.py` · round-trip `easy_check.py` · routing `build_v4.py` · confidence `confidence.py`/`build_v5.py`/
`v5_stats.py` · detection `build_v6.py`/`v6_stats.py`/`v6_ablation.py` · geometry-at-detection `build_v7-v10.py` ·
tool `selective.py`. Results: `RESULTS*.md`.)*
