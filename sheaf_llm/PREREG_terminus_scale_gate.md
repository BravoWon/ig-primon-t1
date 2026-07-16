# Pre-registration — the TERMINUS gate: does the grounding PPL edge survive SCALE? (HF Jobs)

## Why this gate exists (the one open terminus of the whole thread)
Every structural claim for the sheaf has died at a gate; the single surviving lane is **grounding — an
inherited dimensional dictionary (WordNet supersense + hypernym-depth) concatenated onto token embeddings.**
Brick 3′ hardened (`subword_gate.py`) pinned its generative value against a *real* subword baseline:

> grounding is a **small but robust** generative positive — **~2% overall PPL, ~2.3% at embedding-starved /
> rare positions** — at **3 M params / ~1 M tokens**, every seed negative.

The thread's own stated terminus: **does that thin edge survive scale?** Two mechanisms predict it should
*shrink*: at billions of params with far more data, (1) rare-word token embeddings are *better learned*, so
the "help where the token embedding is starved" mechanism weakens; and (2) routing *every* token through a
grounding pathway may *constrain* a large fluent model. This gate measures the **edge-vs-scale curve** and
reads the trend. It needs real GPUs → **HF Jobs**.

## The claim under test
`grounded = concat(token_emb, supersense_emb, depth_emb) → d_model` (a thin grounding top-up on a real
BPE/subword backbone — the *corrected* architecture from `subword_gate.py`, not the `UNK` strawman) yields a
positive held-out PPL improvement over `flat` (BPE + token embeddings only), and that improvement **does not
vanish** as (params, tokens) grow along a scale ladder.

## Hypotheses (pre-registered)
- **H_scale — FALSIFIER direction (thread prior: moderate):** Δ_rare(scale) and Δ_overall(scale) both trend
  to **≤ 0 within seed noise** by the top of the ladder → grounding is a small-model / small-data crutch that
  better-learned rare embeddings erase. Grounding is **not** a scale-robust lever; the grounded-routing
  program has no live substrate at scale.
- **H_persist — CONFIRM direction:** Δ_rare(scale) stays **bounded away from 0** (constant or slow decay,
  and **> the grounded-random control**) at the top of the ladder → grounding is a genuine scale-robust
  inductive prior. The terminus is a real positive; grounded-sheaf routing at billions of params is worth
  building.
- **H_fluency — SECONDARY (the grounded-vs-fluent tension):** does routing *every* token through grounding
  hurt **overall** PPL at scale even while helping rare positions? At 3 M there was no tension (grounding
  helped overall). Track whether Δ_overall goes negative (a tension emerges) as size grows.

**The result is the CURVE Δ(scale), not one number** — the thread's signature. A single big model would be
un-baselined; the ladder *is* the finding.

## Design
### Arms (identical backbone; three per scale point)
1. **`flat`** — GPT-style causal Transformer, GPT-2 BPE, token embeddings only. The real subword baseline.
2. **`grounded`** — same, plus WordNet `(supersense, depth-bucket)` embeddings concatenated at
   WordNet-noun positions, projected to `d_model`. `= subword_gate.py`'s corrected `char+grounded` shape,
   generative.
3. **`grounded-random` (CRITICAL control)** — byte-identical to `grounded` but each lemma is mapped to a
   **fixed random** `(supersense, depth)` (shuffled once). Isolates *semantic* grounding from the mere extra
   capacity/params of the grounding pathway. (The adversarial_gate discipline: a matched-capacity placebo.)

**Matched params:** the grounding pathway is a thin concat+projection; total grounded params are held **≤
`flat`** (shrink token-emb rank if needed) so the capacity confound runs *against* grounding. Params printed.

### Scale ladder (Chinchilla-ish, ~20 tokens/param) — bridges up from the existing 3 M point
| pt | params | tokens | GPU (HF Jobs flavor) | ~wall/run | arms×seeds |
|---|---|---|---|---|---|
| **S0** | ~10 M | ~200 M | `l4x1` / `a10g-small` | ~20–40 min | 3 arms × **2 seeds** = 6 |
| **S1** | ~50 M | ~1 B | `a10g-large` | ~1–2 h | 3 arms × 1 = 3 |
| **S2** | ~200 M | ~4 B | `a100-large` | ~3–6 h | 3 arms × 1 = 3 |
| **S3 (optional)** | ~800 M | ~16 B | `h100` (or `a10g-largex4`) | ~12–24 h | 3 arms × 1 = 3 |

Reading S0→S2 is the sweet spot: three points across ~1.5 orders of magnitude already show whether Δ decays
toward 0. S3 is only run if S0→S2 is ambiguous (the honest tie-breaker), not by default.

### Data
Real running English at scale: **FineWeb-Edu `sample-10BT`** (clean, deduped, license-clear), streamed from
the Hub; identical GPT-2 BPE and identical token stream across arms at each scale point. Fixed held-out val
shard (never trained).

### Metrics (held-out)
- **Overall val PPL**, `flat` vs `grounded` → **Δ_overall(scale)**.
- **Frequency-bucketed PPL at noun positions**, especially the **rare-noun** bucket (grounding's mechanism
  lives here) → **Δ_rare(scale)** — the primary curve.
- **Embedding-starved stress test** (kept from `generative_gate.py`): a held-out noun set forced to `UNK`
  on *input* for both arms; `grounded` still receives its `(supersense, depth)` → **Δ_starved(scale)**.
- All three arms on all three curves; `grounded` vs **`grounded-random`** margin on every curve.

### Falsifier / Confirm (restated crisply)
- **Falsifier:** by the top of the ladder, `grounded − flat ≤ 0` (overall AND rare) within seed noise, i.e.
  Δ trending to zero — *and* `grounded ≈ grounded-random` (any residual gain is capacity, not meaning).
- **Confirm:** at the top, `grounded − flat > 0` on the rare bucket **and** `grounded > grounded-random`
  with margin — a real, semantic, scale-surviving edge.

## The load-bearing engineering risk (named before building)
Attaching **word-level** WordNet features onto a **subword** BPE stream. A WordNet noun may split into
several BPE tokens; lemma→synset needs word boundaries and POS. **Mitigation:** a one-time preprocessing job
tags nouns (spaCy/NLTK), maps each recognized noun-lemma → `(supersense, depth-bucket)`, aligns to BPE tokens
(feature on every subword of the noun; zero-vector elsewhere / for WordNet-OOV), and emits a **grounding-id
stream parallel to the token-id stream** as a Hub dataset. This alignment is the #1 place a bug silently
nulls or leaks the signal → **a round-trip alignment check is a gating prerequisite** (decode a sample: do
grounding ids land on the right words?). Training consumes the precomputed streams, so the GPU jobs stay clean.

## HF Jobs execution plan
**Prereq:** HF **Pro/Team/Enterprise** plan (Jobs are paid). Auth: `rOGUEgRINGO` ✓ (plan tier: confirm).
Every job pushes to Hub (ephemeral env) with `secrets={"HF_TOKEN": "$HF_TOKEN"}`; Trackio for live curves.

- **Job 0 — preprocess (CPU, `cpu-performance`/`cpu-xl`, ~$1–3):** tokenize FineWeb-Edu shard + build the
  grounding-id stream + WordNet feature tables + the shuffled table for `grounded-random`; run the alignment
  round-trip check; push `rOGUEgRINGO/terminus-grounded-fineweb` (dataset). **Blocks all GPU jobs.**
- **Jobs 1..N — train (GPU), one per (size × arm × seed):** custom PEP-723 UV script (torch + datasets +
  tokenizers + trackio), nanoGPT-style from-scratch loop with the grounded embedding layer; stream the
  preprocessed dataset; eval on the held-out shard with the three bucketed metrics; push model + a
  `metrics.json` to `rOGUEgRINGO/terminus-<size>-<arm>-s<seed>`; log to Trackio project `terminus-scale`.
- **Aggregate:** a tiny CPU job collects every `metrics.json` into `rOGUEgRINGO/terminus-results` and plots
  Δ(scale) — the deliverable curve. (Or done locally.)

### Cost (rough, FLOPs-based ≈ 6·N·D; refine before launch)
| ladder depth | GPU spend (approx) |
|---|---|
| S0 only (6 runs, `l4x1`) | **~$5** — a cheap first read; already extends the 3 M point to 10 M |
| S0→S1 | ~$15–25 |
| **S0→S2 (recommended full read)** | **~$70–110** |
| + S3 (tie-breaker) | +$300–600 |
Plus preprocessing ~$1–3. Numbers are order-of-magnitude; a per-size estimate is refined before each launch.

## Honest limits
- One grounding source (WordNet EN nouns) and one corpus (FineWeb-Edu); English-only.
- "Survives scale" is read from S0→S2 (~10–200 M); it *extrapolates* toward billions, it does not reach them.
  S3 narrows but does not close that gap; true billion-scale is a separate, much larger commitment.
- Grounding is attached to **nouns** only (WordNet's strongest tier); verbs/adjs are future scope.
- PPL is the proxy; downstream task lift is not measured here.

## RESULT (2026-07-15, run LOCALLY on RTX 5070 — not HF Jobs): grounding SURVIVES scale and is SEMANTIC.
Run as a **data-scale** ladder (the sharpest test of the rare-word mechanism) on a single GPU instead of the
cloud ladder: one ~40 M-param GPT (GPT-2 BPE, d=320/6L), real running English (**FineWeb-Edu**, 124 M-token
corpus with a validated WordNet↔BPE grounding stream), token budgets **{8M, 30M, 90M}**, arms flat /
grounded / grounded-random, single seed. Payoff bucket = learnable-but-rare nouns (train freq 3–50). Code:
`terminus/{preprocess,train,aggregate}.py`.

| tokens | overall edge vs flat | rare-noun edge vs flat | **rare-noun edge vs RANDOM placebo** |
|---|---|---|---|
| 8M | +6.1% | +12% | **+2.9%** |
| 30M | +9.4% | +22% | **+7.7%** |
| 90M | +8.1% | +31% | **+11.9%** |

- **SURVIVES:** overall PPL edge holds ~8% across an 11× data increase — does **not** decay (falsifier
  "small-data crutch" refuted).
- **SEMANTIC, not capacity:** grounded beats grounded-random at every scale and the margin **grows**
  (2.9→11.9%) — real supersenses beat shuffled ones, cleanly (same words per budget). A random per-token
  feature buys ~3–4% capacity; inherited *meaning* adds a widening edge on top.
- **First structural positive of the whole program.** Sheaf composition, Möbius geometry, and SHA "safety
  certification" all died at their gates; grounding is the one inherited-structure idea that survives and
  strengthens.

**Honest limits:** 8M→90M at 40M params, single seed, undertrained regime; the overall edge ticked down
slightly 30M→90M (9.4→8.1), so billions-scale survival is *extrapolated, not shown*. The rare-noun trend
conflates data-scale with a shifting rare-bucket composition (n drops 144k→24k→2.7k) — hence the clean
load-bearing signals are the **overall** edge (stable) and the **within-budget grounded-vs-random** margin
(growing). Multi-seed + true param-scale (and the real billions terminus) remain open.

## RESULT 2 (2026-07-15) — 3-SEED HARDENING: the placebo margin is SEED-ROBUST (9/9)
Re-ran the full ladder at **seeds 1 and 2** (18 runs, ~3 h on the RTX 5070; `terminus/run_hardening.ps1`), so
every (budget × arm) now has 3 seeds. Aggregated with per-seed grounded-vs-random pairing (`aggregate.py`):

| tokens | overall edge | rare edge | grounded-vs-RANDOM (rare), per-seed | mean |
|---|---|---|---|---|
| 8M  | +7.0% | +10.8% | +2.9 / +3.8 / +2.5 | **+3.1%** |
| 30M | +9.0% | +20.6% | +7.7 / +6.2 / +7.5 | **+7.1%** |
| 90M | +8.9% | +31.9% | +11.9 / +7.1 / +12.2 | **+10.4%** |

- **SEED-ROBUST:** every grounded-vs-random rare margin is positive across **all 3 seeds × all 3 budgets
  (9/9)** — seed 0 (RESULT 1) was not lucky. The overall edge holds ~7–9% across 11× data (no decay), and the
  semantic (vs-placebo) margin grows +3.1→+10.4%, replicating RESULT 1's single-seed trend at 3 seeds.
- **Round-trip-verified:** 4 of the 9 per-seed margins were recomputed by hand from the raw `nll_rare` values
  and matched the aggregator to 0.1%. (The aggregator crashed on a `seed`-key bug — the older single-seed s0
  files predate the in-JSON `seed` field — fixed by stamping the seed from the filename; pairing re-checked
  after the fix.)
- Multi-seed is now **closed positive**; true param-scale is addressed by **RESULT 3** below; the billions
  terminus remains open.

## RESULT 3 (2026-07-16) — PARAM-scale ladder, 3-SEED: grounding survives PARAMETER scaling too (SEED-ROBUST 9/9)
The complement to RESULT 2: hold **data fixed (30M tokens)** and vary **model size** — the axis RESULT 1/2 held
fixed at 40M, and the one where "better-learned rare embeddings erase the edge" actually lives. Run locally
(RTX 5070; `terminus/run_pscale_harden.ps1`), 3 param sizes × 3 arms × 3 seeds = 27 runs. (The 148M point I
first tried maxed the 12 GB card and thrashed to ~4.5 s/step; capped at ~90M with headroom — kept fp32, so the
mid point stays an exact anchor.) Aggregated per-seed (`aggregate_pscale.py`):

| params (flat) | overall edge | grounded-vs-RANDOM (rare), per-seed | mean |
|---|---|---|---|
| ~14M | +6.9% | +1.6 / +1.2 / +1.0 | **+1.3%** |
| ~40M | +9.0% | +7.7 / +6.2 / +7.5 | **+7.1%** |
| ~90M | +9.9% | +14.6 / +12.2 / +13.6 | **+13.5%** |

- **SEED-ROBUST:** every grounded-vs-random rare margin is positive across **all 3 sizes × 3 seeds (9/9)**; the
  falsifier ("bigger models learn rare embeddings better → edge decays") did **not** fire — the margin holds and
  grows (+1.3 → +13.5%); overall edge holds ~7–10%.
- **Exact instrument weld:** the ~40M point (`ps040`, d=320/6L) reproduces RESULT 2's 30M point **bit-for-bit at
  every seed** (grounded PPL 222.13 / 224.62 / 225.76; vs-random margins +7.7/+6.2/+7.5 *identical*). The two
  ladders cross and agree exactly. Round-trip-verified: all 9 per-seed margins recomputed by hand from raw NLLs.
- **Honest confound:** at **fixed tokens**, a bigger model is more **undertrained** (30M ≪ Chinchilla for 90M),
  which *favors* grounding — so the clean claim is **"survives / does not decay"**, not "grows because of params."
  A Chinchilla-scaled ladder (tokens ∝ params) and true billions remain the sharper, still-open tests.

## Status
`[GATE]` → **RUN (local), POSITIVE + SEED-ROBUST on BOTH scaling axes (3-seed; data-scale 2026-07-15,
param-scale 2026-07-16).** Grounding beats its matched-capacity random-supersense placebo at **every seed on
both ladders — 9/9 across data (8→90M tokens) AND 9/9 across params (14→90M)** — with no decay. The buildable
substrate — inherited dimensional dictionary + subword backbone — is confirmed as a *real, semantic* lever on
real language at this scale. Open: a Chinchilla-scaled (tokens ∝ params) ladder and true billions — the
cloud/HF-Jobs commitment.
