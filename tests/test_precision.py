"""Precision-certification matrix tests.

CI has no GPU, so these assert backend-agnostic invariants and exercise the reduction
near-miss on the CPU (numpy fp16 reductions overflow/degrade exactly as the GPU's do,
since precision error is portable -- the hw:sw boundary the matrix is built to expose).
"""

import math

from ig_primon import backends as B
from ig_primon.precision import build_matrix, sweep_reduction


def test_matrix_runs_with_tier_c_and_safe_gemm():
    cells = build_matrix(["gemm"], size=128, budget=1e-3, iters=2)
    assert cells, "matrix produced no cells"
    # every fp64 cell is the Tier-C (exact) reference
    fp64 = [c for c in cells if c.dtype == "fp64"]
    assert fp64 and all(c.verdict == "Tier-C" for c in fp64)
    # fp32 GEMM is safe everywhere it ran
    g32 = [c for c in cells if c.op == "gemm" and c.dtype == "fp32" and math.isfinite(c.floor_err)]
    assert g32 and all(c.floor_err < 1e-3 for c in g32)


def test_layernorm_fp16_accumulate_overflows_at_width():
    # the sharp finding: fp32-accumulate stays flat-safe at large width; fp16-accumulate
    # overflows. This is exactly why production kernels accumulate fp16 reductions in fp32.
    be, data, crossover = sweep_reduction("layernorm", [256, 131072], budget=1e-3)
    (w_small, faith_small, naive_small), (w_big, faith_big, naive_big) = data
    assert math.isfinite(faith_big) and faith_big < 1e-2, "fp32-accumulate must stay bounded at large width"
    assert not math.isfinite(naive_big), "fp16-accumulate layernorm must overflow at 131072 width"
    assert crossover is not None, "the firewall must flag the fp16-accumulate failure"


def test_attention_is_most_fragile():
    # attention compounds two matmuls + a softmax reduction; its fp16-accumulate near-miss
    # crosses the budget at a much smaller context than softmax/layernorm.
    be, data, crossover = sweep_reduction("attention", [256, 1024], budget=1e-3)
    assert crossover is not None and crossover <= 1024
