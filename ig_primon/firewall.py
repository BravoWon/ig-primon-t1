"""The Precision-Certification Firewall (PCF) — the doctrine's H1 flagship, realized.

The doctrine's actual thesis is not "reject garbage" — it is **"agreement is not verification:
a reduced-precision result is exploratory until a higher-precision authority certifies it."**
This module demonstrates that thesis instead of merely asserting a reject switch works.

Three Tier-E (FP32) candidates for the same invariant, all the same failure mode (the
Gaussian integration domain cut short), at three magnitudes:

  * honest      span=12  -> rel_err ~6e-8   = 0.5 x float32-eps  (pure FP32 roundoff; algorithm correct)
  * near-miss   span=5.7 -> rel_err ~3.6e-7 = 3   x float32-eps  (a REAL error, camouflaged as FP32 noise)
  * gross       n=32     -> rel_err ~0.35                        (visibly wrong)

Two adjudicators:

  * an FP32 "certify-by-agreement" reference (float32 computation of the truth) whose OWN error
    is ~1 float32-eps. Its resolution floor == the camouflage band, so it CANNOT tell the
    near-miss from a correct kernel. This is the trap the doctrine names.
  * Tier-C: mpmath at dps>=50 (self-error ~1e-49). It certifies against the float32 noise floor:
    a deviation within ~1 eps is consistent with a correct FP32 kernel; a deviation exceeding it
    is a real error. Only this authority rejects the near-miss.

The firewall has PRECISION teeth iff: the honest candidate certifies, the near-miss passes the
FP32-grade tolerance but is REJECTED by Tier-C, and the gross candidate fails outright. The
near-miss is the load-bearing case: it is what FP32 agreement cannot catch and dps>=50 can.

Target invariant: Gardner capacity alpha_c(kappa) = 1 / E_t[(t+kappa)^2 ; t>=-kappa], alpha_c(0)=2.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

# float32 machine epsilon (2^-23). The FP32 noise floor: a correct FP32 kernel may deviate from
# the truth by O(eps); a deviation materially above eps is a real error, not roundoff.
FP32_EPS = float(np.finfo(np.float32).eps)   # ~1.1921e-07


# ----------------------------- Tier-C: certify (mpmath, the authority) -----------------------------

def certify_alpha_c(kappa=0.0, dps=50):
    """Exact (Tier-C) Gardner capacity via mpmath. The sole [V] authority (self-error ~1e-dps)."""
    from mpmath import mp, mpf, e, sqrt, pi, quad, inf
    mp.dps = dps
    k = mpf(kappa)
    f = lambda t: e ** (-t * t / 2) / sqrt(2 * pi) * (t + k) ** 2
    return 1 / quad(f, [-k, inf])


# ----------------------------- the FP32 "certify-by-agreement" trap -----------------------------

def certify_alpha_c_fp32(kappa=0.0, n=2_000_000, span=12.0):
    """An FP32 reference computation of the SAME truth — i.e. 'certification by FP32 agreement'.

    Returns (value, self_rel_err) where self_rel_err is its own deviation from the dps=50 truth
    (~1 float32-eps). Its resolution floor is exactly this self-error: it cannot adjudicate any
    candidate whose error is within a few eps — which is precisely the near-miss.
    """
    val = _trapz_fp32(np, kappa, n, span)
    truth = float(certify_alpha_c(kappa, dps=50))
    return val, abs(val - truth) / abs(truth)


# ----------------------------- Tier-E: explore (FP32 grid; GPU if available) -----------------------------

def _trapz_fp32(xp, kappa, n, span):
    """FP32 trapezoid quadrature of the Gardner integrand on [-kappa, -kappa+span]."""
    f32 = xp.float32
    t = xp.linspace(f32(-kappa), f32(-kappa + span), n, dtype=f32)
    dx = (t[-1] - t[0]) / (n - 1)
    integrand = ((t + f32(kappa)) ** 2
                 * xp.exp(-t * t / f32(2.0)) / f32(math.sqrt(2.0 * math.pi)))
    I = (integrand.sum() - f32(0.5) * integrand[0] - f32(0.5) * integrand[-1]) * dx
    return 1.0 / float(I)


def explore_alpha_c(kappa=0.0, n=2_000_000, span=12.0, backend="auto"):
    """Tier-E candidate. Returns (value, backend_label).

    backend: "auto" (GPU if available, else CPU FP32), "cuda" (require GPU), "cpu" (force CPU FP32).
    """
    if backend in ("auto", "cuda"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import cupy as cp
                cp.cuda.Device(0).use()
                dev = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            val = _trapz_fp32(cp, kappa, n, span)
            cp.cuda.Stream.null.synchronize()
            return val, f"cuda:{dev}"
        except Exception:
            if backend == "cuda":
                raise
    return _trapz_fp32(np, kappa, n, span), "cpu-fp32:numpy"


# ----------------------------- the firewall -----------------------------

# (n, span) for the three kernels — same failure mode (truncated domain), three magnitudes.
HONEST = (2_000_000, 12.0)      # ~0.5 eps : pure roundoff
NEAR_MISS = (2_000_000, 5.7)    # ~3   eps : real error inside the FP32 camouflage band
GROSS = (32, 2.0)               # ~0.35    : visibly wrong


@dataclass
class KernelRun:
    name: str
    backend: str
    n: int
    span: float
    value: float
    rel_err: float          # vs the dps=50 truth
    eps_multiple: float     # rel_err / float32-eps
    time_s: float
    explore_pass: bool      # within the loose FP32-grade tolerance
    certify_pass: bool      # within the float32 noise floor (Tier-C verdict)


@dataclass
class FirewallResult:
    invariant: str
    truth: str
    dps: int
    explore_budget: float
    certify_budget: float
    fp32_ref_value: float
    fp32_ref_self_err: float
    kernels: list = field(default_factory=list)
    firewall_intact: bool = False
    precision_teeth: bool = False
    tier_c_time_s: float = 0.0


def run_firewall(kappa=0.0, backend="auto", dps=50,
                 explore_budget=1e-5, certify_budget=None):
    """Run honest / near-miss / gross kernels and adjudicate at two precisions.

    ``certify_budget`` defaults to the float32 noise floor (~FP32_EPS): a deviation within it is
    consistent with a correct FP32 kernel; beyond it is a real error only Tier-C can see.
    ``precision_teeth`` is the load-bearing claim: the near-miss passes the FP32-grade tolerance
    but is REJECTED by Tier-C. ``firewall_intact`` additionally requires honest-certified and
    gross-rejected.
    """
    if certify_budget is None:
        certify_budget = FP32_EPS

    t0 = time.perf_counter()
    truth = certify_alpha_c(kappa, dps=dps)
    tc_time = time.perf_counter() - t0
    truth_f = float(truth)

    fp32_ref_val, fp32_ref_err = certify_alpha_c_fp32(kappa)

    res = FirewallResult(
        invariant=f"Gardner capacity alpha_c(kappa={kappa}) (exact = {truth_f:.6g})",
        truth=repr(truth)[:48], dps=dps,
        explore_budget=explore_budget, certify_budget=certify_budget,
        fp32_ref_value=fp32_ref_val, fp32_ref_self_err=fp32_ref_err,
        tier_c_time_s=tc_time,
    )

    for name, (n, span) in (("honest", HONEST), ("near-miss", NEAR_MISS), ("gross", GROSS)):
        t1 = time.perf_counter()
        val, be = explore_alpha_c(kappa, n=n, span=span, backend=backend)
        dt = time.perf_counter() - t1
        rel = abs(val - truth_f) / abs(truth_f)
        res.kernels.append(KernelRun(
            name, be, n, span, val, rel, rel / FP32_EPS, dt,
            explore_pass=rel <= explore_budget,
            certify_pass=rel <= certify_budget,
        ))

    honest, near, gross = res.kernels
    # PRECISION teeth: the near-miss is invisible at FP32 grade but caught by dps>=50.
    res.precision_teeth = near.explore_pass and (not near.certify_pass)
    res.firewall_intact = (honest.certify_pass and res.precision_teeth
                           and (not gross.explore_pass))
    return res


def format_firewall(res: FirewallResult):
    L = []
    L.append("=" * 88)
    L.append("IG-PRIMON-T1 - Precision-Certification Firewall  (agreement is not verification)")
    L.append("=" * 88)
    L.append(f"  invariant : {res.invariant}")
    L.append(f"  Tier-C    : mpmath dps={res.dps} (self-err ~1e-{res.dps})  ->  {res.truth}   ({res.tier_c_time_s:.2f}s)")
    L.append(f"  float32 eps (noise floor) : {FP32_EPS:.3e}")
    L.append(f"  FP32 'certify-by-agreement' reference: value={res.fp32_ref_value:.10f}  "
             f"self-err={res.fp32_ref_self_err:.2e} (~{res.fp32_ref_self_err/FP32_EPS:.1f} eps)")
    L.append(f"      -> its resolution floor IS the camouflage band; it cannot adjudicate the near-miss.")
    L.append(f"  budgets   : FP32-grade tolerance = {res.explore_budget:g}   |   "
             f"Tier-C certify = {res.certify_budget:.3e} (~1 eps)")
    L.append("")
    L.append(f"  {'kernel':<10} {'rel_err':>10} {'x eps':>7} {'FP32-tol':>9} {'Tier-C':>9}   verdict")
    for k in res.kernels:
        ftol = "pass" if k.explore_pass else "FAIL"
        tc = "pass" if k.certify_pass else "FAIL"
        if k.name == "near-miss":
            verdict = ("waved through by FP32, CAUGHT by dps>=50  <-- precision teeth"
                       if (k.explore_pass and not k.certify_pass) else "near-miss did not behave as designed")
        elif k.name == "honest":
            verdict = "certified: deviation within FP32 noise floor (algorithm correct)"
        else:
            verdict = "rejected by both (gross error)"
        L.append(f"  {k.name:<10} {k.rel_err:>10.2e} {k.eps_multiple:>7.1f} {ftol:>9} {tc:>9}   {verdict}")
    L.append("")
    L.append(f"  PRECISION TEETH : {'YES' if res.precision_teeth else 'NO'}  "
             f"(near-miss passes FP32 tolerance {res.explore_budget:g} but fails Tier-C)")
    if res.firewall_intact:
        L.append("  FIREWALL INTACT : honest certified; near-miss (FP32-invisible) caught by dps>=50; gross rejected.")
        L.append("                    => FP32 agreement is NOT verification; only Tier-C licenses [V].")
        L.append("                    (constructed demonstration of the mechanism on this Gardner near-miss:")
        L.append("                     it shows FP32 agreement CAN fail to verify, not that it always does.)")
    else:
        L.append("  FIREWALL BREACH : adjudication did not behave as specified - audit the harness.")
    L.append("=" * 88)
    return "\n".join(L)
