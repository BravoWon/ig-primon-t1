# IG-PRIMON-T1 — Precision-Certification Map for Inference Primitives (v0.3, 2026-06-15)

**Track.** Hardware-execution / operationalization. Extends the Precision–Certification Firewall
(`ig_primon.firewall`) to the numerical primitives that dominate LLM inference: GEMM, softmax, LayerNorm,
attention. Receipts: `ig_primon/precision.py` (`igprimon precision-matrix`, cupy fp16/fp32/fp64) and
`ig_primon/torch_precision.py` (`igprimon precision-bf16`, torch fp32/tf32/fp16/bf16).

**v0.2 (this revision).** v0.1 was an **fp16-only** map. v0.2 adds the **bf16 pass** (real inference is
bf16, not fp16) and corrects two fp16-era findings, both surfaced by measurement against prediction:
(1) the LayerNorm fp16→overflow is a **range artifact** that bf16 (fp32 range) avoids, and its reduced
axis is the **hidden dim, not context**; (2) softmax fp16 error **rises with peakedness**, not with
diffuseness — the opposite of the first intuition. Recorded, not smoothed.

**v0.3 (this revision).** Adds the **fp8 pass** (`igprimon precision-fp8`): real Blackwell fp8 tensor-core
GEMM via `torch._scaled_mm`, and the range-vs-mantissa tradeoff that actually defines fp8 (§2b).

**Discipline.** **[V]** verified by a reproducible run, **[open]** named not built. **Scope wall:** every
claim is about the **numerical precision of inference primitives** — not speculative decoding (acceptance
is not exact certification) or LLM-as-judge (no numerical authority). The firewall is a numerical-precision
instrument and this is a numerical-precision problem.

---

## 0. Method

Per (operation, backend, dtype): run the op, compare to an **fp64 (Tier-C) reference**, report throughput
and relative error. The **noise floor** of a precision is the error of a *faithful* kernel — one that
accumulates reductions in fp32 (standard mixed-precision practice). Each reduction op carries a **near-miss
adversary**: the same kernel accumulating in the storage dtype. The fp64 certifier catches it because its
error exceeds the floor while passing a same-grade glance — the firewall, per inference op.

---

## 1. The map [V]

Hardware: RTX 5070 (Blackwell sm_120), GTX 1660 Ti (Turing sm_75), i9-9900K. Floor (faithful) error at
size 1024:

| op | fp32 | tf32 | fp16 | **bf16** |
|---|---|---|---|---|
| GEMM | 3.4e-7 | 2.9e-4 | 3.6e-4 | **2.9e-3** |
| softmax | 8.9e-8 | — | 6.5e-4 | **5.2e-3** |
| LayerNorm | 6.1e-8 | — | 4.6e-4 | **3.7e-3** |
| attention | 4.0e-7 | 4.3e-4 | 8.4e-4 | **6.6e-3** |

- **bf16 is ~8× coarser than fp16** (7 mantissa bits vs 10), uniformly. At a 1e-3 budget every bf16 op is
  a *near-miss*; at a realistic ~1e-2 inference budget they are all safe — bf16 *is* the deployed format.
  **fp16-safe does not imply bf16-safe**: the budget that matters has to be chosen per deployment.
- **tf32 is the GEMM sweet spot**: 2.9e-4 error at 25.5k GFLOP/s (vs fp32 17.4k) — fp32 storage, tensor-core
  math. (tf32 is a matmul mode; it does not apply to pure reductions.)
- **GEMM is the easy primitive.** fp16 floor 3.6e-4 is portable (algorithmic); only the *speed* is
  hardware-specific (5070 fp16 35 TFLOP/s; 1660 Ti fp16 runs *below* its fp32 — TU116 has no fp16 tensor
  cores). **GPU fp64 ≈ CPU fp64** (476 vs 326 GFLOP/s) — consumer Blackwell cripples fp64, so the GPU is a
  great explorer and a useless certifier: the hardware confirms Tier-C belongs on the CPU.

## 2. Reduction-accumulate fragility — and the bf16 inversion [V]

The reduction ops need **fp32-accumulate**; the cost of getting it wrong (storage-dtype accumulate)
depends on the *reduced axis* and the *dtype's range*:

- **LayerNorm — re-axed and inverted.** LayerNorm reduces over the **hidden dim (d_model ≤ ~16k)**, *not*
  context. The headline "overflows past ~64k" was a **synthetic reduction-width stress**, not a real
  inference failure — real d_model never reaches the overflow point. And the overflow itself is an **fp16
  range artifact** (the 65504 ceiling): fp16-accumulate sum-of-squares breaks past 64k (rel err
  5e-4→0.51→0.71 at 16k/64k/256k), but **bf16-accumulate is flat ~3.9e-3 at every width** (fp32 range, no
  overflow). So the most dramatic fp16 result mostly dissolves for the deployed dtype.
- **Attention — the genuine context-axis result.** Attention's softmax-sum and ×V-sum *are* over the
  sequence, so its fragility is correctly a **context** axis: fp16-accumulate crosses a 1e-3 budget by
  ~256 context — it compounds two matmuls and a softmax, the most precision-sensitive primitive.
- **Softmax — peakedness, corrected.** `igprimon precision-matrix --sweep` + the peakedness sweep show the
  fp16 error **rises with peakedness / logit magnitude** (3.9e-4 diffuse → 2.0e-3 peaked at width 4096),
  *refuting* the "diffuse spreads mass into the tails" intuition. Mechanism: peaked attention needs
  large-magnitude logits, fp16 stores those coarsely, and `exp` amplifies it — so **sharp heads
  (attention sinks), not diffuse ones, are where fp16 softmax degrades**. (Logit scaling confounds
  magnitude and entropy; the magnitude effect dominates.)

**Operational takeaway (real inference, bf16):** bf16 has no LayerNorm range failure and ~8× the mantissa
error of fp16; reductions still want fp32-accumulate for the floor; attention is the most fragile primitive
and the one whose fragility is genuinely context-scaling; sharp-logit softmax is the fp16 (and bf16) soft
spot. Per-op, per-precision, per-axis — not "we made LLMs better."

## 2b. fp8 — the range-vs-mantissa frontier [V]

fp8 is a storage / tensor-core-GEMM format, not a compute format (no `exp` in fp8); `igprimon
precision-fp8` measures it via `torch._scaled_mm` on the RTX 5070's fp8 tensor cores.

- **fp8 GEMM is fast and coarse.** E4M3 (3 mantissa, range ±448): **134 TFLOP/s** (2× bf16, ~280× the
  crippled fp64), rel err 3.8e-2 (~13× bf16). E5M2×E5M2 GEMM is *unsupported* by cublas ("Multiplication of
  two Float8_e5m2 matrices is not supported") — E5M2 is the gradient format; forward fp8 is E4M3, with
  mixed E4M3×E5M2 allowed.
- **The binding constraint is range, not mantissa.** E4M3 is finer at normal scale (2.7e-2 vs E5M2 5.3e-2)
  but **saturates to nan past magnitude ~100** (its ±448 ceiling) — precisely the LLM activation-outlier
  regime; E5M2 keeps bf16-like range (±57344) at every magnitude but is mantissa-starved. The fp8 lesson is
  that *range/outliers*, not raw mantissa count, is what binds — which is why fp8 inference lives or dies on
  per-tensor / per-channel scaling.

---

## 3. The open frontier [open] — NOT built

Per-op "safe" is **local**. Whether locally-certified ops stay **globally bounded** through N transformer
layers — softmax saturation feeding the next block, residual streams accumulating, attention amplifying
perturbations — is the unsolved question, and the reason bf16/fp8 inference sometimes silently degrades in
ways per-tensor checks never flag. Research, gated behind this map; named so the boundary is explicit.

---

## 4. Reproduce

```
pip install -e ".[dev,gpu]"; pip install "cupy-cuda12x[ctk]"            # fp16/fp32/fp64 (cupy)
pip install torch --index-url https://download.pytorch.org/whl/cu128   # + tf32/bf16 (torch, Blackwell)
igprimon precision-matrix --op all --size 1024     # the fp16/fp32 map
igprimon precision-matrix --op attention --sweep   # the genuine context-axis fragility
igprimon precision-bf16                            # the real-inference map + LayerNorm range inversion
igprimon precision-fp8                             # Blackwell fp8 GEMM + the range-vs-mantissa tradeoff
```
Tests: `pytest tests/test_precision.py` (CPU-portable; bf16 tests skip without torch+CUDA).
