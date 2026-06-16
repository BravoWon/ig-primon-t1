"""Precision-certification matrix tests.

CI has no GPU, so these assert backend-agnostic invariants and exercise the reduction
near-miss on the CPU (numpy fp16 reductions overflow/degrade exactly as the GPU's do,
since precision error is portable -- the hw:sw boundary the matrix is built to expose).
"""

import math

import pytest

from ig_primon import backends as B
from ig_primon import torch_precision as TP
from ig_primon.precision import build_matrix, sweep_reduction, sweep_softmax_peakedness


def test_matrix_runs_with_tier_c_and_safe_gemm():
    cells = build_matrix(["gemm"], size=128, budget=1e-3, iters=2)
    assert cells, "matrix produced no cells"
    # every fp64 cell is the Tier-C (exact) reference
    fp64 = [c for c in cells if c.dtype == "fp64"]
    assert fp64 and all(c.verdict == "Tier-C" for c in fp64)
    # fp32 GEMM is safe everywhere it ran
    g32 = [c for c in cells if c.op == "gemm" and c.dtype == "fp32" and math.isfinite(c.floor_err)]
    assert g32 and all(c.floor_err < 1e-3 for c in g32)


def test_layernorm_fp16_accumulate_overflows_at_synthetic_width():
    # NB: this is a SYNTHETIC reduction-width stress -- LayerNorm reduces over the hidden dim
    # (d_model <= ~16k), so real models never hit this. It demonstrates the mechanism: fp32-
    # accumulate stays flat-safe at large width; fp16-accumulate overflows (an fp16 RANGE artifact).
    be, data, crossover = sweep_reduction("layernorm", [256, 131072], budget=1e-3)
    (w_small, faith_small, naive_small), (w_big, faith_big, naive_big) = data
    assert math.isfinite(faith_big) and faith_big < 1e-2, "fp32-accumulate must stay bounded at large width"
    assert not math.isfinite(naive_big), "fp16-accumulate layernorm must overflow at 131072 width"
    assert crossover is not None


def test_softmax_fp16_error_rises_with_peakedness():
    # corrected finding: fp16 softmax error RISES with peakedness/logit-magnitude (sharp heads),
    # NOT with diffuseness -- the opposite of the 'tails carry the error' intuition.
    be, data = sweep_softmax_peakedness([0.25, 8.0], width=2048)
    (s_lo, ent_lo, err_lo), (s_hi, ent_hi, err_hi) = data
    assert ent_hi < ent_lo, "scale 8 must be peakier (lower entropy) than scale 0.25"
    assert err_hi > err_lo, "peaked softmax must have higher fp16 error than diffuse"


@pytest.mark.skipif(not TP.available(), reason="bf16 pass needs torch + CUDA")
def test_bf16_is_coarser_but_avoids_layernorm_overflow():
    cells = TP.bf16_matrix(ops=("gemm",), dtypes=("fp16", "bf16"), size=512, budget=1e-3, iters=2)
    by = {c.dtype: c for c in cells}
    assert by["bf16"].floor_err > by["fp16"].floor_err, "bf16 (7 mantissa bits) must be coarser than fp16"
    rows = TP.layernorm_range_check(widths=(16384, 131072))
    (_, _, _), (_, e16_big, ebf_big) = rows
    assert e16_big > 0.1, "fp16-accumulate layernorm must break at 131072 width (range overflow)"
    assert ebf_big < 1e-2, "bf16-accumulate (fp32 range) must stay bounded -- the inversion"


@pytest.mark.skipif(not TP.available(), reason="fp8 pass needs torch + CUDA")
def test_fp8_e4m3_finer_but_saturates_while_e5m2_holds():
    rows = TP.fp8_range_tradeoff(mags=(1, 1000))
    (_, (e4_lo, s4_lo), (e5_lo, _)), (_, (_, s4_hi), (_, s5_hi)) = rows
    assert e4_lo < e5_lo, "E4M3 (3 mantissa) must be finer than E5M2 (2 mantissa) at normal scale"
    assert s4_hi and not s5_hi, "E4M3 must saturate past its +-448 range while E5M2 (+-57344) holds"
    gemm = {n: err for n, gf, err in TP.fp8_gemm(size=512, iters=2)}
    assert math.isfinite(gemm["e4m3"]), "real E4M3 fp8 tensor-core GEMM must run"


def test_attention_is_most_fragile():
    # "Most fragile" is a RELATIVE claim: attention compounds two matmuls + a softmax reduction,
    # so its fp16-accumulate near-miss error is the largest of the reduction ops at any given
    # width. We assert that ordering (attn > softmax > layernorm), NOT an absolute budget-crossing:
    # attention's near-miss sits right at ~1e-3, so whether it "crosses" a fixed budget is
    # knife-edge and platform-dependent (CPU vs GPU BLAS -- it crosses on CUDA, not on CPU numpy),
    # whereas the ordering is structural and holds on both. Verified on cpu and cuda backends.
    widths = [256, 1024]
    err = {op: [naive for _, _, naive in sweep_reduction(op, widths, budget=1e-3)[1]]
           for op in ("attention", "softmax", "layernorm")}
    for i, n in enumerate(widths):
        assert err["attention"][i] > err["softmax"][i] > err["layernorm"][i], \
            f"attention must be the most fragile (largest fp16-accumulate error) at width {n}"
