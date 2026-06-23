# Tosco Branch 4 RR — Production-Success Report
### Dynamics + Cross-Domain Verification & Validation

*Well: TOSCO BRANCH 4 (re-entry), NDIC File #41700, API 33-011-01562, NENE 19-129N-103W, Bowman
County, ND. Operator Marlo Operating (working interest Valor Energy Partners); directional/MWD Charger
Services; rig Stoneham 16. Status DRILLING (NDIC 06/19/2026); confidential to 12/19/2026. Quantitative
backbone: `tosco_cos_model.py` (Monte-Carlo CoS, results in `tosco_cos_model_results.txt`). Scope:
**probability of a successful producing well — not economics.***

---

## 0. Executive summary
**Triangulated probability of a successful producer: central ≈ 0.64–0.72; 80% band ≈ 0.40–0.85.**
Two independent estimators bracket it: a reference-class **outside view = 0.77** and a correlated
stage-gate **inside view = 0.50**. Their **0.27 divergence is the headline V&V result** — and it is
*diagnostic*: it localizes essentially all of the residual uncertainty to **short-radius re-entry
mechanical execution** (the #1 tornado factor, swing 0.088), not to reservoir presence. The reservoir
is effectively de-risked (re-entry of a logged wellbore **+** an adjacent producing discovery); the
bet is *drilling and landing the lateral*, which is largely **observable in advance** from one
document — the original wellbore's casing/cement-bond log.

| Estimator | P10 | **P50** | P90 | mean |
|---|---|---|---|---|
| Reference-class base rate (Red River dev/extension) | 0.62 | **0.73** | 0.83 | 0.72 |
| **Outside view** (base + this-well adjustments) | 0.61 | **0.77** | 0.88 | 0.76 |
| **Inside view** (correlated stage-gate product) | 0.30 | **0.50** | 0.67 | 0.49 |
| **Combined (triangulated)** | 0.37 | **0.64** | 0.85 | 0.62 |

---

## 1. The well — facts and the decisive reclassification
**Confirmed in NDIC public data** (not inferred): File #41700, permit type **RE-ENTRY**, surface NENE
Sec 19-129N-103W, bottom-hole NWNE Sec 19 — a short lateral kept inside the NE/4 drilling unit
(consistent with the high-DLS short-radius build in the cost docs). NDIC files it as **WILDCAT**, but:

- **Sister well Tosco Branch 5 (File #41701) is the *discovery well* of a newly-found pool and is
  PRODUCING** (~65 bopd, Apr 2026). NDIC spacing **Case #32729** seeks to develop that pool and
  "define the field limits."
- Therefore Branch 4 RR is a **re-entry step-out ~1 mile into a Marlo-discovered, currently-producing
  pool** — the "WILDCAT" tag is *administrative* (no field named yet), **not** exploration risk.

This single reclassification moves the reference base rate from ~0.20 (true Red River wildcat) to the
**0.70–0.86 development/extension tier** (NDGS SW-ND Red River: Camel Hump 6/7 = 0.86; step-outs
14/20 = 0.70). The whole analysis rests on this; it is **verified** against NDIC primary records (§6).

---

## 2. Cross-domain V&V framework (method)
Each domain produces an **independent** point estimate of CoS from its own evidence. If they
**converge**, the estimate is *validated by triangulation*; where they **diverge**, the disagreement
itself is the signal (it names the controlling uncertainty). This is the same discipline used
throughout this program: trust nothing on one view; cross-check across independent lenses.

| Domain | Independent CoS read | Drives which gate |
|---|---|---|
| **Geology** | ~0.88 — reservoir present & charged | G1 reservoir |
| **Drilling / mechanical eng.** | ~0.70 — short-radius re-entry through old casing | G2 casing, G3 curve/lateral |
| **Reservoir eng.** | ~0.80 — landing in oil, deliverability, no depletion | G4 placement, G5 rate |
| **Statistics / base rate** | ~0.74 — analog Red River dev/extension success | overall anchor |

The four domain reads (0.70–0.88) **converge** on a high-but-not-certain well; the *lowest* read
(drilling/mechanical, 0.70) is the binding constraint — and it is exactly what the inside-view
stage-gate model amplifies (§4).

---

## 3. The four domains, granular

### 3a. Geology — CoS ≈ 0.88 (reservoir presence & charge)
- **Re-entry of a logged wellbore:** the original Branch 4 vertical already penetrated and logged the
  section. Reservoir *presence and quality are known before the bit turns*. Empirical anchor: the
  DOE/Luff Bowman-County program drilled **0 of 4 dry holes** from absence of reservoir — every
  failure was fluids/mechanics, never "no rock."
- **Adjacent producing discovery:** Branch 5 proves the pool makes oil ~1 mile away.
- **Trend:** T129N R103–104W sits in the SW-ND **Red River "B" fairway** (Cedar Hills trend) — a
  continuous ~8–10 ft dolomite (anhydrite seal above, tight limestone below) where in-fairway
  reservoir-presence risk is low.
- **Residual geologic risk:** the target is **confidential**; if it is **Madison/Mission Canyon**
  rather than Red River "B," geologic risk rises (trap- and porosity-cycle-controlled, *not*
  continuous) → −5 to −10 CoS points. This is the second-largest tornado factor (swing 0.053).

### 3b. Drilling / mechanical engineering — CoS ≈ 0.70 (the binding constraint)
- A short-radius re-entry must (i) confirm old casing/cement integrity, (ii) set a whipstock and mill
  a window (or cut casing + cement plug), (iii) build 25–30°/100 ft on slim tooling, (iv) drill the
  lateral. The DOE/Luff program is a catalogue of how this fails **even when the geology is known**:
  parted BHAs, six tool failures on one curve, fishing, 30–70% cost overruns.
- **Favorable for this well:** the 25–30°/100 ft build is *gentler* than the ~72°/100 ft (80-ft-radius)
  attempts that gave Luff its worst trouble; modern slim-hole MWD/motors (Charger scope) are far
  better than the 1997 vintage in that study.
- **The key unknown:** integrity of the *original* Branch 4 casing and cement across the salt section.
  This is the dominant residual risk (tornado #1, swing 0.088) and is **directly readable** from the
  old wellbore's cement-bond log and mechanical history — the #1 diligence item.

### 3c. Reservoir engineering — CoS ≈ 0.80 (landing in oil + deliverability)
- The canonical re-entry failure is a *geologic bullseye that is an economic miss*: landing the thin
  lateral **below the oil-water contact** (Luff "Greni": perfect 2,959-ft lateral, 67 ft low → 341
  bwpd / 10 bopd), or into a **depleted** zone (low BHP → lost circulation, weak lift).
- **Favorable here:** this is a **new pool** (Branch 5 just discovered) → likely **virgin pressure**,
  which sidesteps the depletion failure mode that killed the Luff wells (tornado factor, swing 0.042).
- **Residual:** thin-pay geosteering and OWC position — readable from the original logs + Branch 5
  fluid contacts.

### 3d. Statistics / base rate — CoS ≈ 0.74 (the outside-view anchor)
- Reference class = SW-ND Red River **development/extension** wells (NDGS): pooled **20/27 ≈ 0.74**
  (Camel Hump 6/7; step-out 14/20). True under-explored Red River wildcats are only ~0.20 — which is
  *not* this well's class.
- This is the most reliable single number for forecasting (an *outside view* grounded in realized
  frequencies), and it anchors the model's prior.

---

## 4. The quantitative model (granular)
Two independent estimators, then triangulate. Inputs are **structured expert-judgment priors**
(transparent, in logit units) except the reference class, which is the empirical anchor. The value is
the *structure* (decomposition + correlation + triangulation + sensitivity + dynamics), not false
precision.

**Outside view.** Start from the reference-class Beta(20+1, 7+1) (P50 0.73), apply this-well logit
adjustments (each with uncertainty): re-entry-logged **+0.30**, Branch-5-producing **+0.40**, Red
River fairway **+0.15**, virgin-pressure new pool **+0.20**, gentle build **+0.10**; minus
short-radius mechanical **−0.55**, confidential/Madison **−0.20**, pool-edge **−0.18**. Net shift
**+0.22 logit** → **outside CoS P50 = 0.77**.

**Inside view.** Five sequential gates (G1 reservoir 0.90, G2 casing 0.85, G3 curve 0.88, G4 placement
0.85, G5 rate 0.90) with **two shared latent factors** — execution quality `qe` and geology quality
`qg` — inducing realistic correlation (a competent operator + sound wellbore tends to pass *all*
mechanical gates together). Product → **inside CoS P50 = 0.50** (simulated mean 0.49 — just *below*
the naive 0.515 product-of-base-probs: per-gate uncertainty lowers each gate's expected pass-rate
below its base [Jensen's inequality on the concave logistic], and positive correlation only partially
offsets it; chaining five ~0.85 gates is intrinsically demanding).

**Why they diverge (the V&V payload).** The outside view's 0.74 base rate *already embeds* execution
risk (those analog wells had to be drilled and completed to count as successes), so adjusting it
slightly gives 0.77. The inside view *re-derives* execution risk explicitly and **compounds** it
across five gates → 0.50. The truth is between, and the gap **is** the message: **the answer is
governed by how correlated the execution gates are** — i.e., by wellbore quality and
operator/vendor competence. Resolve the casing-integrity unknown and the two views converge upward.

---

## 5. Dynamics — risk retirement through the well lifecycle
The CoS is not static; it **updates as each stage is observed**. Treating the build as an ordered
(State → Activity → State) chain (the same ontology used across this program), the conditional
probability of an ultimate producer climbs as gates pass:

| Stage observed to PASS | gate P | **P(success | reached here)** |
|---|---|---|
| — start (permitted re-entry) — | 0.90 | **0.51** |
| G2 old casing/cement integrity confirmed | 0.85 | **0.57** |
| G3 window cut + short-radius curve built | 0.88 | **0.67** |
| G4 lateral landed in oil (above OWC) | 0.85 | **0.77** |
| G5 commercial rate on test | 0.90 | **0.90** |

**Reading:** most of the risk lives in the **early drilling stages** (casing → curve → lateral
placement). Once the lateral is landed in oil (post-G4), the conditional CoS is already ~0.77 and the
only thing left is deliverability. **The casing-bond log (pre-G2) is therefore the highest-information
event** — it retires the single largest risk before a dollar of drilling is spent on the curve.

**Production dynamics (if successful).** Red River "B" carbonate: **IP tens-to-low-hundreds bopd**
(Branch 5 ≈ 65 bopd is the direct analog; field discovery well 1-Peterson hit ~497 bopd), **shallow
decline 5–15%/yr** (flat vs Bakken's hyperbolic), **low initial water if landed updip** (rising with
depletion), **EUR ~40–150 Mbbl** typical Bowman (core Red River 150–500 Mbbl), **20–40+ yr life** with
waterflood/EOR. *High probability of a producer of modest size — not a big well.*

---

## 6. Verification — "did we build the estimate right?" (internal)
- **Primary-source check:** every load-bearing fact (file #, API, re-entry permit type, Branch-5
  discovery, Case #32729, DRILLING status) is taken from **NDIC primary records** (daily activity
  report 06/19/2026; docket 04/22/2026), not secondary aggregators. ✔
- **No leakage / forward-only:** the CoS model is a forward Monte Carlo over priors; there is no
  fitted predictor and thus no target/spatial leakage (the failure mode that produced the +0.90
  mirage in the Kansas work). ✔
- **Internal arithmetic:** combined P50 0.64 lies between the two component P50s (0.50, 0.77) as it
  must; reference base (0.73) < outside (0.77) consistent with a net-positive +0.22 shift; tornado
  swings sum-of-effects consistent with the net shift. ✔
- **Self-diagnostic:** the **inside/outside divergence is reported, not hidden** — and is used to
  locate the controlling uncertainty (mechanical execution) rather than averaged away silently. This
  is the verification working *as designed*.

## 7. Validation — "is the estimate right?" (external)
- **Reference-class anchor (NDGS):** 0.70–0.86 dev/extension success — the outside view (0.77) sits
  squarely inside it. ✔
- **Mechanism check (DOE/Luff Bowman re-entries):** 0/4 dry from no-reservoir but only ~1/4 clearly
  economic — *validates both* the high reservoir-presence CoS (0.88) **and** the heavy mechanical/
  placement discount (the inside view's 0.50). Both independent views are corroborated by the same
  field dataset — strong cross-validation. ✔
- **Direct analog (Branch 5 producing):** confirms the pool flows oil — validates G1/G5 optimism. ✔
- **Convergence test:** four domain reads (0.70–0.88) converge on "likely producer, modest size";
  the one low read (drilling, 0.70) is correctly flagged as binding. ✔
- **Residual divergence (0.27 outside-vs-inside):** *not* resolved by data we have — it is gated on
  the confidential target and the un-logged old-casing integrity. Stated as the open item, not
  papered over. ⚠

## 8. Caveats, data gaps, and the diligence that collapses the uncertainty
1. **Target formation confidential to 12/19/2026.** Assessment assumes Red River "B" (best case). If
   **Madison**, shave 5–10 CoS points. → *Confirm zone.*
2. **Old-casing/cement integrity is un-logged for us.** This single unknown drives the #1 tornado
   factor and most of the inside/outside divergence. → *Pull the original Branch 4 cement-bond log and
   mechanical history — the highest-information, lowest-cost diligence item.*
3. **Branch 5 decline curve** (paywalled scout ticket) would calibrate magnitude and G5. → *Buy the
   scout ticket or subscribe (NDIC).*
4. **Magnitude ≠ probability.** Even a clean success is a modest conventional producer — IP ~tens-to-low-hundreds bopd (direct offset Branch 5 ≈ 65 bopd), not Bakken-scale. (County avg ~18 bopd is the stripper-weighted all-well figure, not this well's rate.)

## 9. Bottom line
A re-entry step-out into a **Marlo-discovered, currently-producing** Bowman County Red River pool is a
**development-class bet — central CoS ≈ 0.64–0.72, 80% band ≈ 0.40–0.85.** The reservoir is
effectively proven; **the well will live or die on short-radius re-entry execution and landing the
thin lateral in oil** — both of which are *forecastable in advance* from the original wellbore's logs
and Branch 5's performance. The cross-domain V&V converges on "likely producer, modest size," and
**pinpoints the one document (the old casing/cement-bond log) that would convert most of the residual
uncertainty into knowledge before committing the curve.**
