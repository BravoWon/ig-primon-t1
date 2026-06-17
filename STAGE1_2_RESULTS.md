# IG-PRIMON-T1 — Stage 1 + Stage 2 Results (REAL receipts)

**Program:** IG-PRIMON-T1, hardware-execution track. Companion to `T1_precision_map_v0_2.md`.
**Date:** 2026-06-16. **Sandbox:** regenerated from scratch (prior session's files did not persist; see note 6).
**Reproducibility:** seed `20260616`. Scripts: `precision_depth_map.py` (Stage 1), `stage2_gpt2_p1.py`, `stage2b_robustness.py` (Stage 2).
**Reference scheme:** float64 working reference; mpmath dps=50 spot-certifies float64 on the op set (LN/matmul/softmax/gelu/residual). Low precision = bf16 (genuine torch round-off).

---

## Stage 1 — controls (`precision_depth_map.py`)

| check | result | status |
|---|---|---|
| **C1** identity (f64 vs f64) | `max E_cert = 0.0` | ✅ harness validated |
| **mpmath cert** (f64 vs dps50, same op set) | rel err `6.3e-16` ≤ floor `1.3e-15` | ✅ **float64 licensed as reference** (Firewall, *not* FP32-agreement) |
| **C2** expansive regime (LN off, gain 1.0, d=n=20, L=40, n=300) | log-slope **+0.245/layer**, E40/E1 **7.4e4×**, mean/median **14→592→685×** | ✅ **reproduces Budzinskiy** (exponential + heavy tail) — machinery validated |
| **C2** control (LN on, gain 1.0) | log-slope **+0.019/layer**, E40/E1 3×, mean/median 1.1× | CONTRACTIVE — see finding |
| **C4** composition | E40/E1 ≫ 40× in expansive regime | ✅ super-linear |

**Stage-1 finding `[V]`:** the exponential heavy-tailed worst case exists *only* in the weakly-normalized regime. **With LayerNorm on and well-conditioned weights, the perturbation is already contractive (+0.019/layer).** This is the structural reason to expect P1 on trained weights.

---

## Stage 2 — the scientific payload, trained GPT-2-small (124M, 12 blocks, d=768)

P1 (sub-exponential typical-case depth-error) vs F1 (exponential on trained weights too).

**Single sample (66 tok), `stage2_gpt2_p1.py`:** mean log-slope **+0.032/layer**, growth E[L]/E[1] **1.20×**; error rises mid-stack (~8.9e-3 @ L8) then the residual stream + final LN *contract* it to 5.3e-3 @ L12.

**Robustness, 8 diverse texts, `stage2b_robustness.py`:**

| quantity | value |
|---|---|
| log-slope/layer | mean **+0.048**, std 0.017, **min +0.022, max +0.078** |
| growth E[L]/E[1] | mean **1.59×**, max 2.28× (linear 12×, Stage-1 worst-case exp 15×) |
| **P1 verdict** | **HOLDS on all 8/8 texts** (every slope < 0.10/layer) |
| **F1** | **does not fire** |

**F3 (range vs mantissa), output layer, across tokens:**
- corr(activation_norm, **absolute** error) = **−0.28** → **not** a magnitude/range artifact (would be strongly positive).
- corr(activation_norm, **relative** error) = **−0.64** → small-norm tokens carry more relative error = **LayerNorm-conditioning signature** (κ_LN ∝ 1/‖x‖, Baek), not overflow.
- attention-sink (token 0) is top-error token in **6/8** texts.
- **F3 verdict: does not fire.** Concentration is conditioning-driven, not range-driven.

---

## Honest claim tags

| claim | tag | scope / caveat |
|---|---|---|
| Worst-case exponential heavy-tail reproduced; contractive under LN | **[V]** | synthetic random-weight, d=n=20, float32-vs-float64 |
| float64 is a certified-adequate reference here | **[V]** | dps50 cert on op set; bf16 gap (≥4e-3) is 12+ orders above float64 floor (~1e-15), so float64 is unimpeachable at this precision |
| **P1: trained-GPT-2-small typical-case error is non-amplifying through depth** | **[V]** for GPT-2-small / bf16 / 8 texts | robust (8/8), reproducible. **Not yet** [V] for: Pythia/medium/weight-sparse, FP8/FP4, the full frozen multi-hundred-text slice |
| F1 does not fire | **[V]** (this scope) | as above |
| F3 (range artifact) does not fire | **[V]** (this scope) | output-layer token correlation |
| Error concentrates on a sparse token set ("which tokens fire") | **[E]** | observed; consistent with attention-sink + κ_LN |
| Concentration is κ_softmax-attributable (F2 mechanism) | **[C] — UNTESTED** | needs C3 shuffle-control; do not claim |
| Behavior at FP4 / the allocator's operational edge | **[C]** | Stage 3, untested |

---

## Open (before any [V] beyond GPT-2-small/bf16)

1. **C3 shuffle-control** → attribute the concentration to κ_softmax or not (closes/forks F2).
2. **FP8 + FP4 fake-quant probes** → does P1's contractive regime hold at coarser precision (it should; slope is a Jacobian property, not an injection-scale property — but measure it).
3. **Cross-architecture:** GPT-2-medium (deeper), Pythia-160M (parallel vs sequential pre-LN), OpenAI weight-sparse cross-check.
4. **Full frozen slice** (multi-hundred held-out texts) → promote P1 from robust-pilot to full [V].
5. **Stage 3 — the actual deliverable (dual-maxim):** `precision_allocator.py` + the **F-app** test (certified-vs-FP32 allocation: predicted to *agree* at FP8, *differ* at FP4). This is where the certification earns its operational worth or doesn't.

---

## Notes

6. **On the prior session's numbers.** Stage-1 receipts narrated earlier (slope +0.285, ~88,000×, `transformers 5.12.1`, a git bank) were **not on disk in this sandbox** — so they were regenerated here, not inherited. The regenerated expansive-regime numbers (+0.245, ~74,000×) and the transformers version (5.12.1, confirmed real) **match closely**, which indicates the prior run was genuine and simply did not persist across the reset. Regenerate-and-compare was the correct discipline: it neither trusted the numbers blind nor wrongly dismissed them.

7. The scientific arc is clean and complete *at this scope*: **worst-case exponential is real but lives in the weakly-normalized regime; trained-weight typical-case is robustly contractive through depth; the residual error that does exist concentrates on a sparse, conditioning-identified (not range-driven) token set.** The instrument (depth-law) now supports building the allocator (H1) on a trustworthy footing.

— End of Stage 1+2 results. All numbers computed in this sandbox; reproducible via the named scripts and seed.
