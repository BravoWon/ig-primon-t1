"""bf16 pass (via torch -- cupy has no bf16). Real LLM inference is bf16, so this is the pass
that makes the precision map a map of real inference rather than of fp16.

bf16 = fp32's range (8 exponent bits) with 7 mantissa bits (vs fp16's 5 exponent / 10 mantissa).
Two consequences this pass measures:

  * The LayerNorm fp16->nan overflow is a RANGE artifact (fp16's 65504 ceiling). bf16 carries
    fp32's range, so a bf16 sum-of-squares over 64k -- or 64M -- never overflows. The most
    dramatic fp16 finding inverts.
  * Every accuracy verdict shifts ~8x coarser: bf16's 7 mantissa bits are ~3 bits below fp16's
    10, so an op certified at ~3.6e-4 in fp16 lands near ~3e-3 in bf16. fp16-safe != bf16-safe.

torch is imported lazily and isolated here; the rest of ig_primon does not depend on it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .precision import (_softmax, _layernorm, _attention, _mk_gemm, _mk_mat, _mk_ln, _mk_attn,
                        _relerr, _to_host)


def available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _dtype(torch, name):
    return {"fp32": torch.float32, "tf32": torch.float32,
            "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _ops(torch):
    def gemm(a, b, acc):
        return a @ b

    def softmax(x, acc):
        m = x.amax(dim=-1, keepdim=True)
        e = torch.exp(x - m)
        return e / e.sum(dim=-1, keepdim=True, dtype=acc).to(x.dtype)

    def layernorm(x, g, b, acc, eps=1e-5):
        n = x.shape[-1]
        mu = (x.sum(dim=-1, keepdim=True, dtype=acc) / n).to(x.dtype)
        xc = x - mu
        var = (xc.to(acc) ** 2).sum(dim=-1, keepdim=True, dtype=acc) / n
        return xc * (1.0 / torch.sqrt(var + eps)).to(x.dtype) * g + b

    def attention(q, k, v, acc):
        d = q.shape[-1]
        s = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(d))
        return softmax(s, acc) @ v

    return {"gemm": gemm, "softmax": softmax, "layernorm": layernorm, "attention": attention}


_MK = {"gemm": _mk_gemm, "softmax": _mk_mat, "layernorm": _mk_ln, "attention": _mk_attn}
_MATMUL_OPS = {"gemm", "attention"}


def _ref(op_name, inputs64):
    if op_name == "gemm":
        return inputs64[0] @ inputs64[1]
    if op_name == "softmax":
        return _softmax(np, inputs64[0], np.float64)
    if op_name == "layernorm":
        return _layernorm(np, *inputs64, np.float64)
    if op_name == "attention":
        return _attention(np, *inputs64, np.float64)
    raise ValueError(op_name)


@dataclass
class BCell:
    op: str
    dtype: str
    gflops: float
    floor_err: float
    verdict: str


def bf16_matrix(ops=("gemm", "softmax", "layernorm", "attention"),
                dtypes=("fp32", "tf32", "fp16", "bf16"), size=1024, budget=1e-3, device=0, iters=10):
    import torch
    torch.cuda.set_device(device)
    flops = {"gemm": lambda n: 2 * n ** 3, "softmax": lambda n: 5 * n * n,
             "layernorm": lambda n: 8 * n * n, "attention": lambda n: 4 * n * n * 64}
    OPS = _ops(torch)
    rng = np.random.default_rng(0)
    cells = []
    for op in ops:
        inputs64 = _MK[op](rng, size)
        ref = np.asarray(_to_host(_ref(op, [a.astype(np.float64) for a in inputs64])), dtype=np.float64)
        for dt in dtypes:
            if dt == "tf32" and op not in _MATMUL_OPS:
                continue   # tf32 is a matmul tensor-core mode; meaningless for pure reductions
            try:
                torch.backends.cuda.matmul.allow_tf32 = (dt == "tf32")
                torch.backends.cudnn.allow_tf32 = (dt == "tf32")
                tdt = _dtype(torch, dt)
                ins = [torch.as_tensor(a, dtype=tdt, device=f"cuda:{device}") for a in inputs64]
                acc = torch.float32   # faithful: fp32-accumulate reductions
                fn = lambda: OPS[op](*ins, acc=acc)
                fn(); torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    out = fn()
                torch.cuda.synchronize()
                dt_s = (time.perf_counter() - t0) / iters
                err = _relerr(out.float().cpu().numpy(), ref)
                gf = flops[op](size) / dt_s / 1e9
                verdict = ("SAFE" if err <= budget else "near-miss" if err <= 10 * budget else "UNSAFE")
                cells.append(BCell(op, dt, gf, err, verdict))
            except Exception as exc:
                cells.append(BCell(op, dt, float("nan"), float("nan"), f"ERR:{type(exc).__name__}"))
    torch.backends.cuda.matmul.allow_tf32 = False
    return cells


def layernorm_range_check(widths=(16384, 65536, 262144), device=0):
    """The inversion, head-on: the LayerNorm fp16 sum-of-squares OVERFLOWS past ~64k; bf16
    (fp32 range) does not. Returns rows (width, fp16_naive_err_or_nan, bf16_naive_err)."""
    import torch
    torch.cuda.set_device(device)
    OPS = _ops(torch)
    rows = []
    for w in widths:
        rng = np.random.default_rng(0)
        x64 = rng.standard_normal((64, w)); g64 = rng.standard_normal((w,)); b64 = rng.standard_normal((w,))
        ref = np.asarray(_to_host(_layernorm(np, x64, g64, b64, np.float64)), dtype=np.float64)
        out = {}
        for dt in ("fp16", "bf16"):
            tdt = _dtype(torch, dt)
            x = torch.as_tensor(x64, dtype=tdt, device=f"cuda:{device}")
            g = torch.as_tensor(g64, dtype=tdt, device=f"cuda:{device}")
            b = torch.as_tensor(b64, dtype=tdt, device=f"cuda:{device}")
            r = OPS["layernorm"](x, g, b, acc=tdt)   # in-dtype accumulate (the near-miss)
            torch.cuda.synchronize()
            out[dt] = _relerr(r.float().cpu().numpy(), ref)
        rows.append((w, out["fp16"], out["bf16"]))
    return rows


def format_bf16_matrix(cells, size, budget):
    lines = []
    lines.append("=" * 80)
    lines.append(f"IG-PRIMON-T1 - bf16 pass (torch, RTX 5070)  size={size}, budget rel_err <= {budget:g}")
    lines.append("  bf16 = fp32 range + 7 mantissa bits. The pass that makes this a REAL-inference map.")
    lines.append("=" * 80)
    ops = []
    for c in cells:
        if c.op not in ops:
            ops.append(c.op)
    for op in ops:
        lines.append(f"\n[{op}]")
        lines.append(f"  {'dtype':>6} {'GFLOP/s':>9} {'floor_err':>11} {'verdict':>9}")
        for c in [c for c in cells if c.op == op]:
            gf = f"{c.gflops:9.0f}" if math.isfinite(c.gflops) else f"{'-':>9}"
            fe = f"{c.floor_err:.2e}" if math.isfinite(c.floor_err) else "-"
            lines.append(f"  {c.dtype:>6} {gf} {fe:>11} {c.verdict:>9}")
    return "\n".join(lines)


_FP8 = {"e4m3": ("float8_e4m3fn", 448.0), "e5m2": ("float8_e5m2", 57344.0)}


def fp8_gemm(size=4096, device=0, iters=20):
    """Real fp8 GEMM on Blackwell fp8 tensor cores via torch._scaled_mm (plain a@b is unsupported
    for fp8). Per-tensor max-scaling into fp8 range. Returns rows (dtype, GFLOP/s, rel_err vs fp64)."""
    import torch
    torch.cuda.set_device(device)
    dev = f"cuda:{device}"
    a = torch.randn(size, size, device=dev); b = torch.randn(size, size, device=dev)
    ref = a.double() @ b.double()
    rows = []
    for name, (dt8, maxv) in _FP8.items():
        try:
            fdt = getattr(torch, dt8)
            sa = (a.abs().max() / maxv).to(torch.float32)
            sb = (b.abs().max() / maxv).to(torch.float32)
            a8 = (a / sa).to(fdt)
            b8 = (b / sb).to(fdt).t().contiguous().t()   # _scaled_mm wants col-major rhs
            fn = lambda: torch._scaled_mm(a8, b8, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            fn(); torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                out = fn()
            torch.cuda.synchronize()
            dt_s = (time.perf_counter() - t0) / iters
            err = float((out.double() - ref).norm() / ref.norm())
            rows.append((name, 2 * size ** 3 / dt_s / 1e9, err))
        except Exception as exc:
            rows.append((name, float("nan"), float("nan")))
    return rows


def fp8_range_tradeoff(mags=(1, 10, 100, 1000, 5000), device=0):
    """The range-vs-mantissa tradeoff head-on: quant-dequant rel_err vs input magnitude. E4M3
    (range +-448) is finer at normal scale but SATURATES on outliers; E5M2 (range +-57344) keeps
    bf16-like range but is mantissa-starved. Returns (mag, (e4m3_err, e4m3_sat), (e5m2_err, e5m2_sat))."""
    import torch
    torch.cuda.set_device(device)
    dev = f"cuda:{device}"
    out = []
    for m in mags:
        x = torch.randn(512, 512, device=dev) * m
        row = [m]
        for name, (dt8, maxv) in _FP8.items():
            fdt = getattr(torch, dt8)
            xb = x.to(fdt).to(torch.float32)
            err = float((xb - x).norm() / x.norm())
            sat = bool(torch.isinf(xb).any() or (xb.abs() >= maxv * 0.999).any())
            row.append((err, sat))
        out.append(tuple(row))
    return out


def format_fp8(gemm_rows, range_rows):
    lines = []
    lines.append("=" * 80)
    lines.append("IG-PRIMON-T1 - fp8 pass (torch _scaled_mm, RTX 5070 Blackwell fp8 tensor cores)")
    lines.append("  E4M3: 3 mantissa, range +-448 (finer, narrow). E5M2: 2 mantissa, range +-57344 (coarse, wide).")
    lines.append("=" * 80)
    lines.append("\n[fp8 GEMM, per-tensor scaled]")
    lines.append(f"  {'dtype':>6} {'GFLOP/s':>9} {'rel_err':>10}")
    for name, gf, err in gemm_rows:
        g = f"{gf:9.0f}" if math.isfinite(gf) else f"{'-':>9}"
        e = f"{err:.2e}" if math.isfinite(err) else "unsupported"
        lines.append(f"  {name:>6} {g} {e:>10}")
    lines.append("  (E5M2xE5M2 GEMM is unsupported by cublas: 'Multiplication of two Float8_e5m2 matrices is")
    lines.append("   not supported'. Forward fp8 GEMM is E4M3; mixed E4M3xE5M2 is allowed -- E5M2 is for gradients.)")
    lines.append("\n[range-vs-mantissa tradeoff: quant-dequant rel_err vs input magnitude]")
    lines.append(f"  {'magnitude':>9} {'E4M3 (+-448)':>18} {'E5M2 (+-57344)':>18}")
    for row in range_rows:
        m = row[0]; (e4, s4) = row[1]; (e5, s5) = row[2]
        a = f"{e4:.2e}{' SAT' if s4 else ''}"
        b = f"{e5:.2e}{' SAT' if s5 else ''}"
        lines.append(f"  {m:>9} {a:>18} {b:>18}")
    lines.append("  => E4M3 is finer at normal scale but SATURATES past +-448 (LLM activation outliers are")
    lines.append("     exactly this regime); E5M2 keeps bf16-like range but is mantissa-starved. That tension")
    lines.append("     -- not raw mantissa count -- is why fp8 inference needs per-tensor/-channel scaling.")
    return "\n".join(lines)


def format_range_check(rows):
    lines = []
    lines.append("\n[layernorm range check] in-dtype-accumulate sum-of-squares (the fp16 overflow, inverted)")
    lines.append(f"  {'width':>8} {'fp16-acc':>16} {'bf16-acc':>14}")
    for w, e16, e_bf in rows:
        s16 = "OVERFLOW(nan)" if not math.isfinite(e16) else f"{e16:.2e}"
        sbf = "OVERFLOW(nan)" if not math.isfinite(e_bf) else f"{e_bf:.2e}"
        lines.append(f"  {w:>8} {s16:>16} {sbf:>14}")
    lines.append("  => fp16 overflows past ~64k (range artifact, 65504 ceiling); bf16 (fp32 range) does NOT.")
    lines.append("     The single most dramatic fp16 finding is a range artifact the deployed dtype avoids.")
    return "\n".join(lines)
