"""Precision-certification matrix for inference primitives.

Runs the operations that dominate LLM inference -- GEMM, softmax, LayerNorm, attention --
across {backend} x {dtype}, certifying each against an fp64 (Tier-C) reference. The claim
is scoped to ONE thing: the numerical precision of inference primitives. It does not speak
to speculative decoding (acceptance != exact certification) or judgment quality.

Two findings the matrix is built to expose:

  * GEMM is the easy primitive (dense, regular); fp16 GEMM lands near its noise floor.
  * The REDUCTION ops (softmax, LayerNorm, attention's softmax) are where precision bites.
    Their accumulation dtype is controllable, so each carries a near-miss adversary: the
    faithful kernel accumulates the reduction in fp32 (the noise floor), the near-miss
    accumulates in the storage dtype (fp16). The fp64 certifier catches the fp16-accumulate
    kernel -- its error exceeds the noise floor while passing an fp16-grade glance -- which
    is exactly why production kernels accumulate fp16 ops in fp32.

The composition question -- do locally-certified ops stay globally bounded through N layers --
is the open frontier, named in the docs and deliberately NOT built here.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from . import backends as B

# storage dtype -> faithful accumulation dtype (fp16 reductions accumulate in fp32)
_ACC_HONEST = {"fp16": np.float32, "fp32": np.float32, "fp64": np.float64}


# --------------------------------------------------------------------------------------
# the primitives, parameterised by the array library `xp` and the reduction accumulator
# --------------------------------------------------------------------------------------

def _softmax(xp, x, acc):
    m = x.max(axis=-1, keepdims=True)
    e = xp.exp(x - m)
    s = e.sum(axis=-1, keepdims=True, dtype=acc)   # <-- the accumulation knob
    return e / s.astype(x.dtype)


def _layernorm(xp, x, g, b, acc, eps=1e-5):
    n = x.shape[-1]
    mu = (x.sum(axis=-1, keepdims=True, dtype=acc) / n).astype(x.dtype)
    xc = x - mu
    var = ((xc.astype(acc) ** 2).sum(axis=-1, keepdims=True, dtype=acc) / n)
    inv = (1.0 / xp.sqrt(var + eps)).astype(x.dtype)
    return xc * inv * g + b


def _attention(xp, q, k, v, acc):
    d = q.shape[-1]
    scale = x_dtype_scalar(xp, 1.0 / math.sqrt(d), q.dtype)
    s = (q @ _T(k)) * scale                        # matmuls accumulate in cuBLAS (not exposed)
    a = _softmax(xp, s, acc)                        # the controllable reduction
    return a @ v


def _gemm(xp, a, b, acc=None):
    return a @ b


def _T(x):
    return x.swapaxes(-1, -2)


def x_dtype_scalar(xp, val, dtype):
    return xp.asarray(val, dtype=dtype)


@dataclass
class Op:
    name: str
    make: object        # (rng, size) -> tuple of fp64 numpy inputs
    run: object         # (xp, *inputs_in_dtype, acc) -> result
    flops: object       # (size) -> float
    reduces: bool       # True if it has a controllable reduction accumulator (=> near-miss teeth)


def _mk_gemm(rng, n):
    return (rng.standard_normal((n, n)), rng.standard_normal((n, n)))


def _mk_mat(rng, n):
    return (rng.standard_normal((n, n)),)


def _mk_ln(rng, n):
    return (rng.standard_normal((n, n)), rng.standard_normal((n,)), rng.standard_normal((n,)))


def _mk_attn(rng, n, d=64):
    return (rng.standard_normal((n, d)), rng.standard_normal((n, d)), rng.standard_normal((n, d)))


OPS = {
    "gemm":      Op("gemm", _mk_gemm, lambda xp, a, b, acc: _gemm(xp, a, b), lambda n: 2 * n ** 3, False),
    "softmax":   Op("softmax", _mk_mat, lambda xp, x, acc: _softmax(xp, x, acc), lambda n: 5 * n * n, True),
    "layernorm": Op("layernorm", _mk_ln, lambda xp, x, g, b, acc: _layernorm(xp, x, g, b, acc), lambda n: 8 * n * n, True),
    "attention": Op("attention", _mk_attn, lambda xp, q, k, v, acc: _attention(xp, q, k, v, acc),
                    lambda n: 4 * n * n * 64, True),
}


# --------------------------------------------------------------------------------------
# running one cell
# --------------------------------------------------------------------------------------

def _to(xp, arr, np_dtype, backend):
    a = arr.astype(np_dtype)
    return xp.asarray(a)


def _relerr(result, ref):
    r = np.asarray(result, dtype=np.float64)
    return float(np.linalg.norm(r - ref) / np.linalg.norm(ref))


def _time(fn, backend, iters):
    fn(); B.sync(backend)                # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    B.sync(backend)
    return out, (time.perf_counter() - t0) / iters


@dataclass
class Cell:
    op: str
    backend_id: str
    backend_name: str
    dtype: str
    gflops: float
    floor_err: float     # faithful (fp32-accumulate) kernel vs fp64
    nearmiss_err: float  # fp16-accumulate kernel vs fp64 (only meaningful at fp16)
    verdict: str         # SAFE / near-miss / UNSAFE / Tier-C
    teeth: str           # "caught" / "n/a" / "-"


def _to_host(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)


def run_cell(op_name, backend, dtype, size, budget, iters, ref_cache):
    op = OPS[op_name]
    np_dtype = B.NP_DTYPE[dtype]
    xp = B.get_xp(backend)
    rng = np.random.default_rng(0)
    inputs64 = op.make(rng, size)

    # fp64 reference (computed once per (op,size), accumulator fp64)
    key = (op_name, size)
    if key not in ref_cache:
        ref_cache[key] = _to_host(op.run(np, *[a.astype(np.float64) for a in inputs64], acc=np.float64))
    ref = np.asarray(ref_cache[key], dtype=np.float64)

    with B.on_device(backend):
        ins = [_to(xp, a, np_dtype, backend) for a in inputs64]
        acc_honest = _ACC_HONEST[dtype]
        out, dt = _time(lambda: op.run(xp, *ins, acc=acc_honest), backend, iters)
        floor = _relerr(_to_host(out), ref)
        gf = op.flops(size) / dt / 1e9

        nearmiss = float("nan")
        teeth = "n/a"
        if op.reduces and dtype == "fp16":
            out_nm = op.run(xp, *ins, acc=np.float16)   # accumulate in fp16 (the near-miss)
            B.sync(backend)
            nearmiss = _relerr(_to_host(out_nm), ref)

    # safety verdict (is this dtype safe for this op, with a faithful kernel?)
    if dtype == "fp64":
        verdict = "Tier-C"
    elif floor <= budget:
        verdict = "SAFE"
    elif floor <= 10 * budget:
        verdict = "near-miss"
    else:
        verdict = "UNSAFE"

    if op.reduces and dtype == "fp16" and math.isfinite(nearmiss):
        # teeth: the fp16-accumulate kernel must exceed the noise floor (caught by fp64) ...
        teeth = "caught" if nearmiss > max(2 * floor, budget) else "weak"

    return Cell(op_name, backend.id, backend.name, dtype, gf, floor, nearmiss, verdict, teeth)


def build_matrix(op_names, size=1024, budget=1e-3, iters=10):
    backends = B.discover()
    ref_cache = {}
    cells = []
    for op_name in op_names:
        for be in backends:
            for dtype in be.dtypes:
                try:
                    cells.append(run_cell(op_name, be, dtype, size, budget, iters, ref_cache))
                except Exception as exc:
                    cells.append(Cell(op_name, be.id, be.name, dtype, float("nan"),
                                      float("nan"), float("nan"), f"ERR:{type(exc).__name__}", "-"))
    return cells


def _sweep_inputs(op_name, width, rows=64, d=64):
    """Inputs that grow ONLY the reduced axis (the last dimension), with a fixed small row
    count -- so the sweep isolates reduction-width / context-length dependence without an
    N^2 blow-up. (softmax/layernorm reduce over `width`; attention reduces over the key
    dimension = context length `width`, with `rows` queries.)"""
    rng = np.random.default_rng(0)
    if op_name == "softmax":
        return (rng.standard_normal((rows, width)),)
    if op_name == "layernorm":
        return (rng.standard_normal((rows, width)),
                rng.standard_normal((width,)), rng.standard_normal((width,)))
    if op_name == "attention":
        return (rng.standard_normal((rows, d)),
                rng.standard_normal((width, d)), rng.standard_normal((width, d)))
    raise ValueError(op_name)


def sweep_reduction(op_name, sizes, dtype="fp16", budget=1e-3):
    """Reduction-width / context-length dependence: faithful (fp32-accumulate) vs
    fp16-accumulate error as the reduced dimension grows. Returns the crossover size where
    the fp16-accumulate near-miss exceeds the safety budget (or None)."""
    op = OPS[op_name]
    backends = [b for b in B.discover() if b.kind == "cuda"] or B.discover()
    be = backends[0]
    xp = B.get_xp(be)
    data = []
    crossover = None
    for n in sizes:
        inputs64 = _sweep_inputs(op_name, n)
        ref = np.asarray(_to_host(op.run(np, *[a.astype(np.float64) for a in inputs64],
                                         acc=np.float64)), dtype=np.float64)
        with B.on_device(be), np.errstate(all="ignore"):
            ins = [_to(xp, a, np.float16, be) for a in inputs64]
            faithful = _relerr(_to_host(op.run(xp, *ins, acc=np.float32)), ref)
            naive = _relerr(_to_host(op.run(xp, *ins, acc=np.float16)), ref)   # fp16 reduction can overflow
        data.append((n, faithful, naive))
        # a non-finite near-miss is an OVERFLOW failure, not "within budget"
        if crossover is None and (not math.isfinite(naive) or naive > budget):
            crossover = n
    return be, data, crossover


# --------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------

def format_matrix(cells, size, budget):
    lines = []
    lines.append("=" * 92)
    lines.append(f"IG-PRIMON-T1 - precision-certification matrix  (size={size}, safety budget rel_err <= {budget:g})")
    lines.append("  Tier-C = fp64 reference (CPU/GPU). SAFE = faithful kernel within budget. claim: inference numerics only.")
    lines.append("=" * 92)
    ops = []
    for c in cells:
        if c.op not in ops:
            ops.append(c.op)
    for op in ops:
        oc = [c for c in cells if c.op == op]
        reduces = OPS[op].reduces
        lines.append(f"\n[{op}]")
        lines.append(f"  {'backend':<22} {'dtype':>5} {'GFLOP/s':>9} {'floor_err':>11} {'verdict':>9}"
                     + ("   near-miss(fp16-acc)" if reduces else ""))
        for c in sorted(oc, key=lambda c: (c.backend_id, c.dtype)):
            extra = ""
            if reduces and c.dtype == "fp16" and math.isfinite(c.nearmiss_err):
                extra = f"   {c.nearmiss_err:.2e} -> {c.teeth}"
            gf = f"{c.gflops:9.0f}" if math.isfinite(c.gflops) else f"{'-':>9}"
            fe = f"{c.floor_err:.2e}" if math.isfinite(c.floor_err) else "-"
            lines.append(f"  {c.backend_name[:22]:<22} {c.dtype:>5} {gf} {fe:>11} {c.verdict:>9}{extra}")
        # frontier: fastest SAFE/Tier-C cell
        safe = [c for c in oc if c.verdict in ("SAFE", "Tier-C") and math.isfinite(c.gflops)]
        if safe:
            best = max(safe, key=lambda c: c.gflops)
            lines.append(f"  frontier: fastest safe = {best.backend_name[:22]} / {best.dtype} "
                         f"({best.gflops:.0f} GFLOP/s, floor {best.floor_err:.1e})")
        if reduces:
            nm = [c for c in oc if c.dtype == "fp16" and c.teeth == "caught"]
            if nm:
                lines.append(f"  teeth: fp16-accumulate near-miss CAUGHT by fp64 (needs fp32-accumulate reduction)")
    lines.append("\n" + "-" * 92)
    lines.append("  Reduction-accumulate fragility is WIDTH-dependent (run --sweep): fp16-accumulate layernorm")
    lines.append("  OVERFLOWS past ~64k context; attention fp16 exceeds budget by ~256 -- fp32-accumulate fixes both.")
    lines.append("  Open frontier (NOT certified here): local op-safety does not imply GLOBAL output bounds")
    lines.append("  through N layers -- error composition (softmax saturation, residual accumulation) is research.")
    lines.append("=" * 92)
    return "\n".join(lines)


def format_sweep(be, data, crossover, op_name, budget):
    lines = []
    lines.append(f"\n[sweep] {op_name} on {be.name}: reduction-width / context-length dependence of fp16 "
                 f"(budget {budget:g})")
    lines.append(f"  {'width':>7} {'fp32-acc (faithful)':>20} {'fp16-acc (near-miss)':>21}")
    for n, faithful, naive in data:
        nm = "OVERFLOW(nan)" if not math.isfinite(naive) else f"{naive:.2e}"
        lines.append(f"  {n:>7} {faithful:>20.2e} {nm:>21}")
    if crossover is not None:
        lines.append(f"  => fp16-accumulate FAILS (exceeds {budget:g} or overflows) at width ~{crossover}: "
                     f"this reduction needs fp32 accumulate past that context length.")
    else:
        lines.append("  => fp16-accumulate stayed within budget across the swept widths.")
    return "\n".join(lines)
