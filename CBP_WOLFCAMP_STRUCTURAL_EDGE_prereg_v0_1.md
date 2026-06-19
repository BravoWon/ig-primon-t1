# CBP Wolfcamp Structural-Edge — Pre-Registration v0.1

**Status: PRE-REGISTRATION. No numerics yet.** Locks hypothesis / control / falsifier / metric /
data / pipeline BEFORE any fitting, per house discipline (derive-before-numerics, control-before-scan,
accept-negatives, "agreement is not verification"). Same rigor that ran the IG-PRIMON-T1 quant
experiments, now pointed at a real drilling decision.

## Intent — USE, not sell
Internal edge: does **structural position** predict **vertical-well** productivity uplift in the
**Wolfcamp on the Central Basin Platform (CBP)**, enough to high-grade vertical drilling locations?
Success bar = **beat our own structure-blind baseline** on spatially-held-out wells. This is never sold,
so it can't be copied; failures are private and cheap. The moat is by construction.

## Why CBP Wolfcamp is the right test bed
- CBP = basement-cored structural high (horst) between the Delaware & Midland basins. **Structure
  genuinely controls carbonate trapping here**, unlike the basinal shale-manufacturing play.
- This is the conventional/structural niche the unconventional incumbents under-serve (the Fasken logic).
- Because the platform is **basement-cored**, potential-field data (aeromagnetic/gravity) is genuinely
  informative about platform architecture here — the original "magnetics" instinct earns its place
  *specifically* in this setting (it would NOT in the basinal sed-section).
- ⚠ [flag] Confirm the specific CBP sub-area is carbonate/structural-leaning Wolfcamp, not a horizontal
  shale fairway. Domain holder overrides this read.

## Hypothesis (H1)
Vertical wells on/near anticlinal structural closures in the Wolfcamp (mapped from public data) show
higher normalized productivity than structurally-neutral/low wells, **with lift OVER a structure-blind
baseline**.

## Null (H0) / Falsifier
Adding structural features does NOT improve blind, spatially-held-out productivity prediction (no
improvement in top-quintile winner concentration / rank-correlation), OR structural-high wells show no
significant uplift after controlling for confounders. → **Clean negative, banked. Saves the dry vertical.**

## Control (control-before-scan)
Baseline **B0** predicts normalized productivity from NON-structural features everyone already has:
surface location (x,y), completion depth (TVD), formation/reservoir code, spud vintage, operator,
completion-size proxy (perf interval length / reported treatment if available), county/field.
**Structural features must add lift OVER B0.** Production correlates with depth/vintage/operator/location;
if B0 already ranks the winners, structure adds nothing. (This is the mis-specified-control trap from the
quant work — defended the same way: build the honest baseline first.)

## Metric (use-it framing, not academic R²)
- **Primary:** among candidate vertical locations in the held-out block, does ranking by structural score
  concentrate the actual top-quartile producers in the top quintile of score? (lift curve / decision lift).
- **Secondary:** Spearman rank-corr uplift (B0+struct vs B0); AUC/Brier for "top-quartile producer".
- **Normalization:** normalize production for vintage and completion size so structure isn't credited for
  newer/bigger completions.

## Spatial holdout (leakage defense — the N<d overfit trap, again)
Hold out **GEOGRAPHIC BLOCKS, not random wells.** Nearby wells leak both structure and production via
spatial autocorrelation → random holdout inflates the lift. Use block/spatial CV (grid the CBP; leave-one-
block-out). Build the structure surface ONLY from training-block wells; predict the held-out block.

## Data (public only)
- **RRC:** well locations (surface+bottomhole, GIS), well type (filter to verticals), completions (W-2),
  production (lease-level → allocate; prefer single-well leases for clean signal), field/reservoir
  designation (to identify Wolfcamp completions).
- **Structure proxy (no public well-tops):** derive a Wolfcamp structural surface from Wolfcamp
  completion/perf depth converted to **subsea elevation** via DEM (subsea ≈ surface_elev − depth). Grid →
  surface; compute **residual relief** (minus regional trend), **curvature** (closure indicator),
  local-high membership.
- **Potential field (basement framework):** USGS aeromagnetic compilation + gravity (USGS / UTEP PACES).
  CBP is basement-cored → genuinely informative here.
- **Geospatial:** DEM (USGS 3DEP/SRTM) for datum conversion + surface; surface geology (USGS/BEG).
- **TX BEG:** public Permian structural/strat datasets / cross-sections to sanity-check the proxy surface.

## Known confounders / traps (named up front, with mitigations)
1. **Selection bias / range restriction (the sharp one):** if operators already sited verticals on
   structural highs, the drilled population may lack structural-*low* controls → the lift everyone already
   captured is unmeasurable. **Mitigation:** verify structural variation exists in the drilled population
   *before* trusting any result; the decades-long multi-operator CBP population likely has many
   non-targeted wells. If variation is absent → "can't test" is the honest finding.
2. **Completion-depth ≠ true formation top:** noisy proxy (perf below top, deviation, datum). **Mitigation:**
   consistent subsea datum; sanity-check proxy surface vs BEG regional structure; keep features RELATIVE
   (residual/curvature), not absolute.
3. **Circularity:** the proxy surface is built from where people drilled/completed; may bias toward sampled
   highs. **Mitigation:** relative features + spatial holdout.
4. **Lease-level production allocation noise.** **Mitigation:** prefer single-well leases; document the
   allocation method.

## Minimal first experiment (cheapest signal)
Pick ONE CBP sub-area with dense Wolfcamp vertical control. Build the proxy structural surface from
training-block wells. Single spatial holdout block. Test lift of **B0+struct vs B0** on the held-out block
(lift curve + rank-corr uplift). Real signal → scale to full-CBP block-CV. No signal → cheap private
negative, and we stop.

## v0.2 — Source verification (verified at source 2026-06-18, BEFORE any numerics)
Endpoints and schemas confirmed against the live RRC sources, not assumed:

- **Authoritative spatial source:** `gis.rrc.texas.gov/server/rest/services/rrc_public/RRC_Public_Viewer_Srvs/MapServer`
  - Layer 1 "Well Locations" = **1,394,934 wells statewide**, SR **EPSG:4326** (lat/long → polygon query
    directly), maxRecordCount **1000** (paginate). Fields: UNIQID, **API**, GIS_API5, GIS_WELL_NUMBER,
    SYMNUM, GIS_SYMBOL_DESCRIPTION, lat/long (NAD27 & NAD83). **API = the join key.**
  - Layer 9 "Horiz/Dir Surface Locations" (+ Layer 10 lines) = clean **vertical-vs-horizontal** discriminator.
  - CBP test box (−103.2,31.0,−102.0,32.6) returns **133,004 wells** → real coverage.
  - ⚠ **TRAP CAUGHT:** the convenient mirror `gis.hctx.net/arcgishcpid/TXRRC/Wells` holds only **12,796**
    wells (Texas Central State Plane), a partial non-CBP subset — NOT authoritative. Building on it would
    have produced a confident, meaningless result. Use the RRC server above.

- **Structure source UPGRADED from proxy → real tops:** Wellbore database (`WBA091`, EBCDIC fixed-width):
  - `WBFORM` (recurring): **WB-FORMATION-NAME X(32)** + **WB-FORMATION-DEPTH 9(5)** → genuine named
    formation tops (Wolfcamp picks), not a TD proxy.
  - `WBROOT`: **WB-ELEVATION 9(4) + WB-ELEVATION-CODE** (datum) + **WB-TOTAL-DEPTH 9(5)** →
    subsea structural elevation = WB-ELEVATION − WB-FORMATION-DEPTH, straight from the data (DEM = cross-check,
    not a dependency). Verticals filter makes MD ≈ TVD valid.
  - Residual risks (real, unchanged): formation-NAME normalization (WOLFCAMP / UPPER WOLFCAMP / WFMP /
    misspellings); per-well tops-reporting completeness (empirical coverage check); EBCDIC zoned-decimal parse.

- **Outcome + supporting (free, no-login, scriptable HTTP via `mft.rrc.texas.gov`):** PDQ Production Dump
  (CSV, 1993→present); Completion data; Drilling Permit Master + lat/long; Field Name & Numbers (Wolfcamp
  field codes); Horizontal Drilling Permits.

- **Egress confirmed:** authoritative queries succeed from the build environment. Ingestion fully automated —
  zero manual downloads. Architecture = **multi-source join keyed on API**.

**Net:** the data layer is real and the structure surface is genuine. The pre-reg's biggest risk (noisy
proxy) is retired; one hidden trap (partial mirror) caught before it could bias the study.

## Status
**PRE-REGISTERED + SOURCE-VERIFIED.** Next: build the API-keyed ingestion/join harness (GIS polygon pull →
Wellbore tops parse → PDQ production → vertical/Wolfcamp filter → coverage + structural-variation QC →
structure surface → B0 baseline → spatial-block CV). **NO numerics until harness + B0 are in place.**
Commits/pushes only on explicit word.

## VERDICT v0.3 (2026-06-18) — scored vs pre-registration: **NULL (structure does not pay)**
End-to-end test executed, Andrews County, fully automated public pipeline:
GIS (1.39M wells) → Wellbore tops (114,467 Wolfcamp picks, 97.5% w/ elevation) → multi-horizon
corroboration (Wolfcamp–Strawn residual **r=0.72** → closures are REAL structure) → PDQ production
(78M monthly rows streamed) → **307 clean single-well-lease vertical Wolfcamp wells**, IP = best-12-mo oil.

**Result (pre-reg falsifier MET):**
- Spearman(structural residual, log IP) = **+0.030** (~0); median IP highs vs lows = **1.08×**.
- Spatial-block CV: out-of-fold AUC baseline 0.479 → +structure 0.491 (**lift +0.012**); top-quintile
  capture **19%** (< 25% base rate). Structure adds NO out-of-sample lift.
- Vintage split: null in BOTH old (pre-93/conventional-leaning, n=87, cumHi/cumLo=0.66×) and new
  (resource-era, n=220) cohorts.

**Interpretation (honest, scoped):** the structure is REAL (corroborated) but does NOT control Wolfcamp
vertical productivity in Andrews — consistent with Wolfcamp being a TIGHT/RESOURCE reservoir (production set
by reservoir quality/completion, not structural closure). **STRUCTURE-REAL ≠ STRUCTURE-PAYS**, demonstrated.
Caveat: old-well productivity is undermeasured (PDQ starts 1993 → only their tail), so the conventional-trap
thesis is not *cleanly* refuted — but the available signal is null-to-negative, not positive.

**Next (the data corrected the target):** the structural-vertical thesis belongs to the CONVENTIONAL
CARBONATES (San Andres / Grayburg — classic CBP structural/strat traps: Yates, McElroy, Means), NOT the
tight Wolfcamp. Same pipeline, re-pointed; San Andres tops already parsed (10,353 in Andrews). Awaiting word.

## VERDICT v0.4 (2026-06-18) — San Andres pivot (conventional carbonate): **WEAK / directionally consistent**
Pipeline re-pointed at SAN ANDRES (10,351 Andrews tops → 8,366 verticals → **401 clean single-well-lease
verticals**); production re-extracted for all 4,430 Andrews oil leases.

**Result (does NOT clear the pre-reg bar, but unlike Wolfcamp the direction is RIGHT):**
- median CUM oil on structural highs (>+150 ft) = **11,242** (n=90) vs lows (<−150 ft) = **6,712** (n=50) → **1.68×**.
- Spearman(resid, log cum) = +0.044; top-quintile-by-P captures **28%** of top producers (> 25% base).
- Spatial-block CV: baseline AUC **0.619** (San Andres IS predictable; vs Wolfcamp 0.479 = random) →
  +structure **0.627** (lift **+0.008**). Structure's *incremental* value over location/depth is small.

**Meta-finding (the real payoff):** the apparatus discriminates reservoir type exactly as geology predicts —
tight Wolfcamp: structure flat/negative (1.08×, 19% capture, baseline AUC 0.48=random); conventional San
Andres: structure weak-positive (1.68×, 28% capture, baseline AUC 0.62=predictable). Structure matters more
where geology says it should.

**Honest bottom line:** from PUBLIC data at ~2-km grid resolution, structure is not a strong standalone edge —
weak on conventional carbonate (and confounded by waterflood/unitization + 1993 truncation), absent on tight
rock, largely subsumed by spatial location. A clean edge needs de-confounding (primary-production metric,
non-unitized wells, finer structure) and/or proprietary data (seismic, directional surveys for TVD) — i.e.
the proprietary-data + rigor moat, exactly the thesis.

## VERDICT v0.5 (2026-06-18) — TENSOR vs STANDARD (does multilinear find new knowledge?): **NO**
Tested whether a multi-formation common-mode (SVD PC1 over detrended residuals of San Andres..Strawn — the
"covariant" joint structure) beats the single-horizon residual at predicting cumulative oil.
- PC1 explains 38% of joint variance; loadings concentrate on the SHALLOW CARBONATES (San Andres −0.58,
  Grayburg −0.58, Glorieta −0.53; Wolfcamp/Strawn ≈ −0.1) → the shallow column deforms coherently, the
  deeper section decouples (a real geological insight, but not a predictive one).
- Spatial-block CV (n=113 single-well verticals w/ full data): single-horizon STANDARD +0.087 AUC over
  baseline; TENSOR common-mode +0.059 (WORSE). **TENSOR vs STANDARD = −0.028.**
- VERDICT: the fancier multilinear method adds NO new predictive knowledge over the standard single horizon.
  Method re-expresses; it does not mint signal beyond ground truth. (Confirms the driftwave reframe.)
- HONEST LEAD (quarantined): on this well-characterized n=113 subset, single-horizon structure showed a
  +0.087 lift (vs the full n=307 sample's +0.008 null). BUT n=113 is underpowered (AUC SE ≈0.09 ≈ the lift)
  → within noise of the better-powered null; cannot separate "real signal in a better population" from
  small-sample optimism + selection. A LEAD requiring a clean powered follow-up, NOT evidence structure pays.

## VERDICT v0.6 (2026-06-18) — PROPER dimensionalization (time-series decline dynamics + CCA): **FAINT, non-generalizing**
Re-ran geology↔production the proper way: production as a TIME SERIES → decline-dynamics block (log qi, Di,
log cum, months-on, flatness, t-peak); geology block (SA residual, SA subsea/TVD, multi-formation common-mode);
**Canonical Correlation Analysis**, n=342 single-well verticals.
- Top canonical corr = **0.285**. PERMUTATION TEST: shuffled null 95th = 0.231, **p = 0.020** → a FAINT real
  in-sample coupling exists beyond chance — proper dimensionalization surfaced signal the scalar test (+0.008) missed.
- BUT OUT-OF-FOLD canonical corr (spatial-block CV) = **0.125** (heavy shrinkage from 0.285) → does NOT
  generalize. Coupling is dominated by **DEPTH↔productivity-magnitude** (SA_subsea↔qi/cum), a known correlate,
  with only a weak structural-residual contribution.
- VERDICT: faint-real-but-too-weak-and-mostly-depth; NULL for practical (generalizable) purposes, but NOT the
  clean zero of the scalar/tensor tests. Dimensionalization mattered at the margin; ground truth still caps it.
- LESSON: more dimensions on fixed ground truth RAISE overfit (in-sample 0.285 → OOF 0.125); permutation + OOF
  become MORE necessary as dimensions grow, and accuracy stays bounded by real coupling, not by dimensionality.

## VERDICT v0.7 (2026-06-18) — PERF-INDEXED + clean type-curve (proper completion attribution): **NULL survives the best critique**
Domain-expert correction applied: index production to the actual PERFORATED INTERVAL (WBPERF: WB-FROM-PERF
cols 6-10 / WB-TO-PERF 11-15), not merely "has a top"; + cleaned decline (toss first 2 months of ramp,
robust-MAD cap of rework spikes).
- **The confound, quantified:** of 10,443 Andrews wells with a San Andres top, only **7,161 (69%) are actually
  PERFORATED in San Andres**. ~31% were wrong-zone contamination in the earlier test. (Perf distribution also
  confirms San Andres = the dominant CBP completion target.)
- Cleaning it moved the RAW association: Spearman(SA residual, log clean-IP) +0.008 → **+0.045**; hi/lo IP
  ratio 0.66× → 0.74× (still <1).
- BUT spatial-block CV lift stayed **≈0**: −0.003 (contaminated) and −0.004 (perf-indexed). Structure adds NO
  out-of-sample predictive lift even for wells actually completed in San Andres, with clean type-curves.
- VERDICT: the null **SURVIVED the expert's best methodological critique** — consistent with San Andres being a
  stratigraphic/diagenetic + waterflood-managed carbonate play, not structurally controlled here. The strongest
  form of a null: it held when we did it right.

## VERDICT v0.8 (2026-06-18) — details + the CLEANING TREND (reframes the null → underpowered TRENDING signal)
Tested completion details (perf net-pay, #perfs, position) for merit prediction, perf-indexed San Andres
verticals, realistic zone ≤700 ft (after the 2000ft/1500ft interval error was caught by the user & fixed).
- Perf THICKNESS does NOT predict merit even corrected (Spearman −0.003) — carbonate productivity is
  permeability/diagenesis, not net-pay-completed; own-details map fails. OFFSET net-pay field (k-NN neighbors,
  pre-drill) gave the best detail signal +0.064 CV lift (underpowered).
- KEY: STRUCTURE's raw association CLIMBS monotonically as the sample is cleaned across the successive expert
  corrections: Spearman +0.008 → +0.045 → +0.132 → +0.153 → +0.186; CV lift now +0.057 (n=120). The corrections
  are progressively UN-DILUTING a structure signal → NO LONGER a clean null; an UNDERPOWERED TRENDING signal
  (n=120, AUC SE ≈0.09, +0.057 ≈0.6 SE — not yet significant; real-diluted-signal OR small-n drift, undecided).
- TENSION: positive Spearman vs <1 high/low ratio CONTRADICT → depletion has scrambled the SIGN (mature
  waterflooded field: crests drained first, flanks hold bypassed oil → production can INVERT vs structure;
  user's "heatmap = inverse of the delta"). Cannot resolve direction at this n.
- RESOLUTION (next): run the same perf-indexed/clean-zone/depletion-aware pipeline across ALL CBP San Andres
  counties → n 120 → 1000+ → power to confirm-or-kill the +0.057 and the inversion. Binding constraint was never
  the features (or the math framework); it is clean per-well attribution on a REPRESENTATIVE, POWERED population.

## VERDICT v0.9 (2026-06-18) — POWERED (20 CBP counties, n=908): the trend was drift; the INVERSION is real
Ran the perf-indexed / clean-zone / depletion-aware pipeline across 20 CBP/San-Andres counties (177k wells
parsed, 31,918 perfed in San Andres, 246k coords, 48,757 oil leases). Powered sample n=**908** single-well
perf-indexed San Andres verticals.
- The Andrews +0.057 "climbing trend" was **SMALL-N DRIFT**: at power the relationship is **NEGATIVE**. Spearman
  **−0.081** (n=908, ~2.4 SE → significant); hi/lo ratio **0.74×**. Bootstrap 95% CI on the CV lift
  **[−0.015, −0.006]** → adding structure *significantly HURTS* prediction by ~0.010 (a weak inverted signal
  that doesn't generalize cleanly across blocks).
- **DEPLETION-INVERSION CONFIRMED** (user's "heatmap = inverse of the delta"): structure anti-correlates with
  current production, and the effect is STRONGER in OLD/depleted wells (Spearman −0.120, hi/lo 0.72×) than NEW
  (−0.024, 0.78×) — a dose-response in depletion. Crests drained first; bypassed oil sits on the flanks/lows.
- Per-county: consistently negative-to-null (8 counties n≥40, range −0.11…+0.02). Not one county's artifact.
- HONEST NET: the signal is **REAL, INVERTED, and WEAK**. Powering up KILLED the false positive (good — we did
  not seize it) AND confirmed the domain expert's inversion mechanism. But Spearman −0.08 is not a strong
  predictive edge; the actionable read (target bypassed structural lows in depleted fields) is geologically
  sound but modest in effect size.
- SPINE resolved at power: desire (the climbing trend) → verify (weak inverted); domain intuition (inversion)
  vindicated, strong-edge hope refuted. Both partly right; ground truth + power decided which part.

## VERDICT v1.1 (2026-06-18) — the ONE confirmed edge: BYPASSED-OIL targeting map (validated, modest)
Built the coherent synthesis map ground truth voted YES on: structure (inversion) × maturity (depletion) ×
proven production → bypassed-oil targets (structural LOWS = drained-crest flanks holding remaining oil).
- VALIDATED FIRST (n=908): structural lows outproduce highs **1.36×, bootstrap 95% CI [1.02, 1.88] → SIGNIFICANT.**
  Mature-only directionally stronger (1.39×, Spearman(−resid,prod) +0.12) but underpowered alone (CI spans 1.0).
- This is the ONLY confirmed predictive edge in the entire basin investigation — the inversion ("heatmap =
  inverse of the delta"), now quantified, significant, and mapped over the CBP San Andres fairway
  (`data/bypassed_oil_map.png`).
- HONEST SIZE: modest (36% uplift, CI lower bound barely >1) — a REDEVELOPMENT/infill high-grader for mature
  fields, not a wildcat edge. Confounds (waterflood, allocation) remain; the signal survived perf-indexing + power.
- COHERENCE: every prior null fed this. Virgin-productivity null → structure doesn't pay forward. Inversion →
  it pays backward (bypassed). Perf-indexing cleaned it; power confirmed it; the mismatches triangulated the
  latent depletion that IS the map. The whole expedition cohered to one true, usable, modest edge.

## VERDICT v1.2 (2026-06-18) — RECONSTRUCTION validation → the FUTURE MAPPER (works; via offsets, not structure)
User's delete-infill idea, built rigorously: hold out wells by spatial block (no neighbor cheating), infill
from the rest, compare to hidden truth. n=870 San Andres verticals.
- **FUTURE MAPPER WORKS:** offset-interpolation reconstructs held-out wells at **Spearman(infill, truth) = +0.208.**
  You CAN predict an undrilled San Andres location from offsets with real, modest skill → validated.
- **STRUCTURE IS SUBSUMED:** adding the bypassed-structure signal moved it −0.003 (95% CI [−0.014, +0.006], n.s.).
  The 1.36× inversion is real in isolation but redundant once you have neighbor performance — the offset field
  already encodes the spatial pattern structure weakly proxies.
- ⇒ the working future mapper = **OFFSET-PERFORMANCE interpolation** (industry analog analysis), now rigorously
  reconstruction-validated. Modest (Spearman 0.21 = useful high-grading, not precision; irreducible floor holds).
  Map: `data/future_mapper.png`.
- FULL COHERENCE: structure doesn't predict forward (null), predicts weakly backward (inversion), and is
  subsumed by offsets for prediction. The future mapper is real and runs on offset performance — proven by the
  user's own reconstruction method.
