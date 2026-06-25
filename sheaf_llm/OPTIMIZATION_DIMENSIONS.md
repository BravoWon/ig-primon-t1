# Optimization Dimensions — "represent big-model capability in a tiny envelope"

Driftwave Target Semantic Shape. **v0 (`compress_v0.py`) tested exactly ONE cell** of this grid:
{object=W-absolute, structure=linear low-rank/block-PCA, bits=uniform-4 baseline, training=post-hoc,
metric=**unweighted Frobenius**, regime=all-resident, granularity=per-matrix, composition=single}.
Its verdict ("4-bit beats factorization ~5×") is **only valid for that cell.** The space is ~8-dimensional
and the leverage lives in the axes v0 collapsed.

| # | Axis | v0 collapsed to | Open range | Honest potential |
|---|------|-----------------|-----------|------------------|
| 1 | **Object factored** | raw W (absolute) | quant-residual `R=W−Q(W)`; cross-layer stack; activations/KV-cache; delta-from-shared | residual + cross-layer + KV are live; **W-absolute is the one dead cell v0 proved** |
| 2 | **Structure family** | global SVD + block-PCA | tensor-train/MPO; Monarch/butterfly; **codebook VQ/PQ**; Kronecker; learned-sheaf-with-graph | codebook (AQLM/QuIP#/QTIP → ~2 b/w near-lossless) and MPO are SOTA-real; learned-sheaf untested |
| 3 | **Bit allocation** | uniform 4-bit (baseline) | salient-mixed (doc); GPTQ/AWQ error-feedback; 2-bit codebook; ternary | sub-2-bit near-lossless exists today (AQLM/QTIP) |
| 4 | **Training** | post-hoc PTQ | calibration; distillation; **native structured training** | BitNet b1.58 = existence proof native unlocks bit-regimes PTQ cannot |
| 5 | **Metric / objective** | **unweighted Frobenius** | Hessian / use-aware (GPTQ); downstream task loss; KL-to-teacher | v0 scored error in directions the model doesn't care about — the *wrong objective* (your own precision-arc finding) |
| 6 | **Memory regime** | all-resident, compressed | streamed/offload; **MoE sparse-active working set**; small-resident + oracle cascade; recompute | MoE-active is the *legitimate* "70B working set in 6 GB" |
| 7 | **Granularity** | per-matrix | per-layer-group; whole-model-joint; per-head; per-channel | cross-layer joint exploits residual-stream redundancy v0 threw away |
| 8 | **Composition** | single method | **product of axes** | the real frontier is the *stack*, not any one trick |

## Leverage ranking (potential × cheapness)
1. **Use-aware metric + right baselines** — re-score with activation/Hessian weighting against GPTQ/AWQ & a codebook. Cheap; can re-rank the whole table. *(v0's metric was the unfair part.)*
2. **Composition: 4-bit + low-rank residual / cross-layer shared atoms** — structure *on top of* bits, not *instead of*. Cheap.
3. **2-bit codebook (AQLM-style)** — the real sub-4-bit lever.
4. **MoE active-parameter working set** — the honest 70B-in-6 GB interpretation (stream cold experts).
5. **Native sheaf training** — the only door for the *strong* "restriction maps ARE the parameters" thesis.

## The sheaf's actual job (reframe)
v0 tested the sheaf as a **compressor** that competes with 4-bit — it lost. Its more plausible role is as the
**coordinate system / allocator over the composed stack**: λ₁-saliency decides *which* weights get bits,
restriction maps are the *transport/consistency between* the quantized + sparse + streamed pieces, and the
Laplacian is the *budget scheduler*. That role is untested and is what the doc's "TopologicalQuantizer" and
"KV-Cache Governor" actually describe.

## v1 results (Qwen2.5-0.5B) — axes 5 & 8 un-collapsed
- **Axis 5 (metric):** use-aware-optimal low-rank is **+34%** better than naive low-rank (0.63→0.41 weighted)
  — the metric *was* unfair — **but 4-bit still wins 3.2×** even at the fairest. Narrowed, not flipped.
- **Axis 8 (composition):** 4-bit + low-rank residual barely helps (0.133→0.121) and **loses to uniform 5-bit
  (0.062)**; the quant residual is *more* high-rank than W (0.165 vs 0.279). No structure hiding in the residual.
- **Verdict:** per-weight bits dominate linear structure of pretrained weights, robustly. The two things that
  helped (use-aware weighting; "more levels") both point to the **sheaf-as-ALLOCATOR** role, not factorization.

## v2 results — the ALLOCATOR test (first earned brick, on a proxy)
At equal avg ~4.0 b/param, use-aware metric: **use-aware allocation 0.224 < uniform-4 0.269 (+16.8%)**;
**use-aware saliency < magnitude saliency 0.384 (+41.8%)**; magnitude allocation is *worse than uniform*.
→ allocate bits to channels that FIRE, not channels that are BIG (reproduces [[precision-arc-endpoint]]).
**CAVEAT:** win is on the weighted PROXY only; unweighted err is worse (by design). NEXT GATE = real
perplexity at equal bits. The sheaf-specific question (does λ₁/topological saliency beat activation
saliency?) is still open — v2 used activation energy E[x²], i.e. AWQ-style, not yet the sheaf signal.

## v3 results — DOWNSTREAM gate (perplexity, held-out)
Equal ~4.0 b/param: fp16 **14.5** · uniform-4 RTN **169** · use-aware allocation **89** → allocation
**halves ppl vs uniform (+47.6%)**, downstream. Proxy → downstream both passed; allocator brick HOLDS.
**CAVEAT:** the uniform baseline is naive per-column RTN (weak; real GPTQ/AWQ 4-bit ≈ 16–20). So v3
proves the allocation PRINCIPLE, not superiority over a strong quantizer. NEXT GATE: does use-aware
allocation still help *on top of* group-wise/GPTQ 4-bit? (else it just patched RTN.) Then: λ₁ vs
activation saliency (the sheaf-specific signal), then the big doors (2-bit codebook / MoE / native).

## v4 results — strong baseline + SHEAF SIGNAL (post-hoc quadrant CLOSED)
Strong group-wise 4-bit ppl **18.1** (≈ fp16 14.2). Mixed allocation does **NOT** beat it: best
act-energy 20.9 (−15.6%), magnitude 75, sheaf-spec 21.6. **→ the v2/v3 allocation brick was PATCHING
WEAK RTN; on a strong quantizer plain uniform 4-bit wins.** Sheaf-spectral λ₁-saliency (77% channel
overlap with the diagonal) is **−3.4% vs act-energy = AWQ-in-a-cape, no gain.**
**VERDICT: the entire post-hoc quadrant is closed** — factorization, allocation, AND sheaf-signal all
lose to strong 4-bit. The ONLY surviving door is **native training** (structure built in, not excavated).

## Updated leverage ranking (post-v4)
- ❌ DEAD (tested): factor pretrained weights · allocate bits on a strong base · λ₁/sheaf saliency over AWQ.
- ✅ ONLY DOORS LEFT (all require *building/training*, not re-representation):
  1. **Native structured training** — at equal params, does a structured/restriction-map MLP beat dense? (v5)
  2. **MoE active-parameter working set** — the honest 70B-in-6 GB (different mechanism, not compression).
  3. **2-bit codebook (AQLM/QTIP)** — a quantization advance, orthogonal to the sheaf.

## v5 results — NATIVE low-rank (trained from scratch, equal params)
1.890M params each: **dense val 2.828 beats lowrank-wide 2.893 (−2.3%)**. Low-rank "restriction maps
not weights" loses post-hoc AND native. **Six straight negatives** for sheaf/structure-as-compression
(factor ✗ · allocate ✗ · λ₁-signal ✗ · low-rank-native ✗). Honest prior update: the *compression*
thesis is looking dead. Remaining native families with a real shot: **MoE** (equal active, more total
= honest 70B-in-6 GB) and **Monarch** (equal total). Reframe: the sheaf's one real win in this whole
program was the VALUE/VERIFICATION model (`sheaf_value_model.py`, +0.314 OOS), not compression — its
home looks like ROUTING (MoE) + VERIFICATION, not bytes.

## v6 results — MoE gate (FIRST positive, marginal; eval-bug caught & fixed)
First run had a mismatched-batch eval bug (val rose monotonically) → DISCARDED. Fixed: at equal ACTIVE
compute (~1.18M), MoE (E=4, 5.44M total) beats dense (1.89M) **2.977 vs 3.000 (+0.76%)** — MARGINAL,
single-seed, within noise (moe curve bounced), but the FIRST positive in 7 gates and on the RIGHT
mechanism: store-many/activate-few = honest 70B-in-6 GB (working set fits, stream cold experts).
**Reframe (now load-bearing): sheaf = ROUTER + VERIFIER, NOT compressor** (matches the +0.314 OOS
`sheaf_value_model.py`). Next (v7): scale E ∈ {1,2,4,8} — monotone val↓ in E ⇒ mechanism compounds at
scale (path validated); flat ⇒ noise.

## v7 results — MoE scaling REFUTES v6 (gain was noise)
Controlled E-sweep (same init, equal active compute): val gets WORSE with E — **E=1(dense) 3.010 <
E=2 3.014 < E=4 3.044 < E=8 3.046**. Dense is BEST; v6's +0.76% was noise. MoE does NOT pay at this
toy scale (1.9M params, char-LM, thin data-per-expert) — EXPECTED: MoE is a scale phenomenon. This is
NOT a refutation of MoE-at-scale (real: Mixtral/DeepSeek) — but I cannot demonstrate it on a 6 GB GPU
and will not claim a win I can't reproduce.

## v8 — last structure cells (butterfly + ternary)
Butterfly (Monarch-family, equal params): val 3.109 = dense 3.109 **EXACT TIE** — structured is an
equivalent reparam, not more-per-param. **Ternary (BitNet ~1.58 b/w MLP): +0.74% val for ~10× fewer
MLP bits = NEAR-LOSSLESS.** → the real compression lever is NATIVE LOW-BIT, not sheaf structure
(confirms [[precision-arc-endpoint]]). Ternary 70B → ~14 GB (not 6), reproducible, owes nothing to the cape.

## v9 — ONE-OBJECT-BRANCHED (the cell v0-v8 missed; user's reframe — and it's different)
NOT per-matrix factorization. ONE shared base MLP across ALL layers + a cheap per-layer BRANCH
(restriction map: FiLM / affine-diagonal / rank-8 LoRA) = "the weight geometry is one object, branched."
Early (60-iter) signal: shared-* MATCH dense at **~25-30% of MLP params** (lora edges ahead). This is
the cross-layer shared-generator cell every prior gate assumed away (they all treated layers as
INDEPENDENT matrices). Full run pending. If it holds at convergence, the thesis is REAL here — and
STACKS with ternary (×~10 bits) toward genuine compression. [bug found+fixed: GPT passed 'shared-film',
Branch checked 'film' → branches were empty; smoke caught it.]

## CONSOLIDATED VERDICT (updated through v9)
DEAD: factoring/allocating/λ₁-signal over INDEPENDENT per-layer weights (v0-v7) — dense+4bit wins.
TIE: Monarch/butterfly (equiv reparam). REAL LEVERS (the roads that don't treat layers as independent
full-rank matrices): (1) **native low-bit / ternary** — near-lossless ~10× (v8); (2) **one-object-branched
cross-layer sharing** — promising at ~28% params (v9, under test). The compression dream lives — NOT as
sheaf-factoring-a-matrix, but as native-low-bit × shared-generator. The user was right twice: the gate
is for killing bad ideas cheaply, but I was over-collapsing the dimensionalization before testing the
cells that matter (cross-layer one-object + native low-bit).

## (superseded) leverage ranking (post-v1)
1. **Use-aware bit ALLOCATION** (salient → more bits; the doc's TopologicalQuantizer done right). Cheap; the data points here.
2. **2-bit codebook (AQLM/QTIP-style)** — quant levels are where bytes pay; this is the real sub-4-bit frontier.
3. **MoE active-parameter working set** — honest 70B-in-6 GB (stream cold experts).
4. **Native sheaf/ternary training** (BitNet existence proof) — only door for the strong "maps ARE the params" thesis.
- DEAD (now tested twice): linear factorization of pretrained weights (low-rank / block / residual), even use-aware.

## Quarantine (unchanged)
constants-from-topology · RH-resolution · "deterministic → prevents all hallucination" · Waypoint Grub.
Dimensionalizing the *real* optimization manifold does not resurrect the mythic shell.
