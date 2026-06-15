# IG-PRIMON-T1 — Precision-Certification Map for Inference Primitives (v0.1, 2026-06-15)

**Track.** Hardware-execution / operationalization. Extends the Precision–Certification Firewall
(`ig_primon.firewall`, doctrine v0.3) from one scalar invariant to the numerical primitives that
dominate LLM inference: GEMM, softmax, LayerNorm, attention. Receipt: `ig_primon/precision.py`
(`igprimon precision-matrix`).

**Discipline.** HONEST_CLAIMS: **[V]** verified by a reproducible run, **[open]** named, not built.
**Scope wall (load-bearing):** every claim here is about the **numerical precision of inference
primitives** and nothing else. It does *not* speak to speculative decoding (whose "verification" is a
target model accepting tokens, not an exact authority certifying precision) or to LLM-as-judge (no exact
numerical authority for judgment). Those are analogies; the firewall is a numerical-precision instrument
and this is a numerical-precision problem.

---

## 0. The method

For each (operation, backend, dtype): run the op, compare to an **fp64 (Tier-C) reference**, report
throughput and relative error. The **noise floor** of a precision is the error of a *faithful* kernel —
one that accumulates reductions in fp32 (standard mixed-precision practice). Each reduction op carries a
**near-miss adversary**: the same kernel accumulating the reduction in the storage dtype (fp16). The fp64
certifier catches the fp16-accumulate kernel because its error exceeds the noise floor while passing an
fp16-grade glance — the firewall's "agreement is not verification," now per inference op.

GEMM's accumulation is internal to cuBLAS (not exposed), so its fp16 number *is* its floor; the
controllable-accumulate teeth live on the reduction ops, which is where precision actually bites.

---

## 1. The map (measured) [V]

Hardware: RTX 5070 (Blackwell sm_120), GTX 1660 Ti (Turing sm_75), i9-9900K. `igprimon precision-matrix`,
size 1024, safety budget rel_err ≤ 1e-3. Representative cells:

| op | backend | dtype | GFLOP/s | floor_err (faithful) | verdict |
|---|---|---|---|---|---|
| GEMM | RTX 5070 | fp16 | 43,498 | 3.6e-4 | SAFE (frontier) |
| GEMM | RTX 5070 | fp32 | 17,793 | 3.4e-7 | SAFE |
| GEMM | RTX 5070 | fp64 | 473 | 1.1e-15 | Tier-C |
| softmax | RTX 5070 | fp16 | — | 6.6e-4 | SAFE @1k |
| layernorm | RTX 5070 | fp16 | — | 4.5e-4 | SAFE @1k |
| attention | RTX 5070 | fp16 | 759 | 8.0e-4 | SAFE @1k |

**Findings:**
- **GEMM is the easy primitive.** fp16 lands at its floor (3.6e-4), portable across devices (the error is
  algorithmic); only the *speed* is hardware-specific (5070 fp16 43 TFLOP/s vs 1660 Ti 0.5 TFLOP/s — the
  TU116 has no fp16 tensor cores, so its fp16 runs *below* its fp32, a clean "speed is hw-specific" mark).
- **GPU fp64 is barely above CPU fp64** (476 vs 326 GFLOP/s on the 5070) — consumer Blackwell cripples
  fp64, so the GPU is a great explorer and a useless certifier. The hardware confirms why Tier-C is CPU.

## 2. Reduction-accumulate fragility is width-dependent [V]

`igprimon precision-matrix --sweep` (RTX 5070, fixed 64 rows, growing reduced axis = context length):

| width | softmax faithful / fp16-acc | layernorm faithful / fp16-acc | attention faithful / fp16-acc |
|---|---|---|---|
| 256 | 5.7e-4 / 6.4e-4 | 4.6e-4 / 5.1e-4 | 8.2e-4 / **1.0e-3** |
| 16384 | 7.7e-4 / 8.3e-4 | 4.4e-4 / 4.9e-4 | 9.0e-4 / 9.8e-4 |
| 65536 | 1.0e-3 / 1.1e-3 | 4.5e-4 / **OVERFLOW** | 1.1e-3 / 1.2e-3 |
| 131072 | 1.6e-3 / 1.6e-3 | 4.5e-4 / **OVERFLOW** | 1.7e-3 / 1.7e-3 |

- **LayerNorm — the sharpest result.** fp32-accumulate stays **flat-safe at every width** (~4.5e-4);
  fp16-accumulate **overflows to nan past ~64k context** (the sum-of-squares exceeds fp16's 65504 max). A
  *hard* failure, not soft degradation — the textbook reason norm reductions must accumulate in fp32.
- **Attention — the most fragile.** Compounding two matmuls and a softmax, its fp16-accumulate near-miss
  crosses the 1e-3 budget by **~256 context**, far earlier than softmax/layernorm.
- **Softmax — storage-dominated.** Mass concentration (post max-subtraction) protects its reduction, so
  the fp16-accumulate gap stays small; both faithful and near-miss cross ~64k from fp16 *storage*, not
  accumulation.

**Operational takeaway:** fp16 is safe for GEMM and for short-context reductions; reduction ops need
fp32-accumulate, and that requirement becomes *mandatory* (LayerNorm overflow, attention budget-break)
as context length grows. This is a per-op, per-precision, per-context certificate — not "we made LLMs
better."

---

## 3. The open frontier [open] — NOT built

A per-op "safe" certificate is **locally** valid and says nothing about **composition**. The unsolved,
genuinely valuable question: do locally-certified-safe ops stay **globally bounded** through N
transformer layers, or does the error *interact* — softmax saturation feeding the next block, residual
streams accumulating, attention amplifying small perturbations? This is *why* fp8/fp16 inference sometimes
silently degrades quality in ways per-tensor checks never flag. It is a **research question**, gated
behind this map; we do not scaffold for a composition result before the op-certificates exist. Named here
so the boundary is explicit, per the program's discipline.

---

## 4. Reproduce

```
pip install -e ".[dev,gpu]"; pip install "cupy-cuda12x[ctk]"
igprimon precision-matrix --op all --size 1024      # the map
igprimon precision-matrix --op layernorm --sweep    # the width-dependence + overflow
```
CPU-only (no GPU): the matrix runs on numpy backends; the reduction overflow reproduces on the CPU too
(precision error is portable). Tests: `pytest tests/test_precision.py`.
