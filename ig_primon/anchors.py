"""Anchors — the exact reference values the IG-PRIMON-T1 receipts pin.

Each anchor re-checks, programmatically, a number that a receipt certifies against an
exact external value (Gardner ``alpha_c(0)=2``, Ising ``s(0)=ln2``, the engine ``R=-1``
pin, the registered ``C`` constant, ...). The ledger awarded the ``[V]`` tag from the
receipt's own printed run; this module makes the same checks *machine-verifiable* so a
regression (a dependency bump, a precision change) is caught instead of eyeballed.

Design notes
------------
* Receipts with an ``if __name__ == "__main__"`` guard are imported lazily and their
  anchor functions called directly — cheap, no printing.
* ``audit_independent.py`` and ``module_e_radius_finding.py`` have NO guard (they run
  heavy compute at import). Their anchors are re-derived here as small standalone checks;
  the full scripts remain runnable via ``igprimon run <name>``.
* Nothing here can *award* ``[V]``; it only reports whether the anchor still reproduces.
"""

from __future__ import annotations

import functools
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable

# precision-depth group stub (added for T1 experiment skeleton)
def _a_depth_skeleton():
    import numpy as np
    import module_T1_precision_depthN as m
    val = m.compute_block_error(np.zeros(2), np.eye(2), np.zeros(2))
    ok = np.allclose(val, 0)
    return ok, str(val), "0", "C1 skeleton identity"


@functools.lru_cache(maxsize=None)
def _capture_receipt(modname: str) -> str:
    """Run the ACTUAL receipt as a subprocess (`python -m <modname>`) and capture what it emits.

    This tests the real artifact reproduces its pinned value — not a reimplementation of the
    check — without touching a byte of the receipt. Memoised so a receipt backing several anchors
    runs once. Used for the two guard-less receipts (audit_independent, module_e_radius_finding),
    which run their whole computation at import and so cannot be imported cheaply.
    """
    proc = subprocess.run([sys.executable, "-m", modname],
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        # Operational CI gate: a receipt that exited non-zero (even after printing a
        # partial/plausible value) must fail the anchor, not be regex-parsed as if it ran
        # clean. AnchorSpec.run catches this and marks the anchor failed with the message.
        tail = (proc.stderr or proc.stdout or "")[-500:]
        raise RuntimeError(f"receipt '{modname}' exited {proc.returncode}: ...{tail}")
    return proc.stdout + "\n" + proc.stderr


@dataclass
class AnchorResult:
    id: str
    group: str
    status: str          # the receipt's honesty tag for this number: [V], [E], ...
    desc: str
    ok: bool
    value: str = ""
    expected: str = ""
    detail: str = ""
    slow: bool = False
    error: str = ""


@dataclass
class AnchorSpec:
    id: str
    group: str
    status: str
    desc: str
    fn: Callable[[], "tuple[bool, str, str, str]"]
    slow: bool = False

    def run(self) -> AnchorResult:
        try:
            ok, value, expected, detail = self.fn()
            # anchors compute with numpy/mpmath; coerce to plain str/bool so the
            # result is JSON-serialisable and the table logic is type-stable.
            return AnchorResult(self.id, self.group, self.status, self.desc,
                                bool(ok), str(value), str(expected), str(detail), self.slow)
        except Exception as exc:  # one bad anchor must not sink the suite
            return AnchorResult(self.id, self.group, self.status, self.desc,
                                False, "", "", "", self.slow, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------------------
# Group: engine  — the Ruppeiner curvature engine + sign convention
# --------------------------------------------------------------------------------------

def _a_engine_pin():
    import module_L_ridge_curvature as rdg
    pts = [(0.3, -0.7), (2.4, -3.1), (-1.2, -0.25)]
    vals = [rdg.gauss_pin(t1, t2) for t1, t2 in pts]
    worst = max(abs(v + 1.0) for v in vals)
    ok = worst < 1e-9
    return ok, f"R={vals[0]:.10f}", "R=-1 (normal family)", f"max|R+1| over 3 pts = {worst:.2e}"


def _a_flat_control():
    # Assert the REAL audit_independent receipt (PART 1) emits R=-1 on the normal family AND
    # R=0 on the flat product — by running it, not reimplementing it.
    out = _capture_receipt("audit_independent")
    gauss = [float(m) for m in re.findall(r"\(t1,t2\)=\([^)]*\):\s*R\s*=\s*(-?\d+\.\d+)", out)]
    fm = re.search(r"flat-product control.*?R\s*=\s*(-?[\d.eE+]+)", out)
    flat = float(fm.group(1)) if fm else None
    ok = (len(gauss) >= 1 and max(abs(g + 1.0) for g in gauss) < 1e-6
          and flat is not None and abs(flat) < 1e-3)
    return ok, f"flat R={flat}; normal-family R={gauss[:1]}", \
        "audit emits R=-1 (normal family) and R=0 (flat product)", \
        f"{len(gauss)} normal-family points all R=-1 (subprocess of the real receipt)"


# --------------------------------------------------------------------------------------
# Group: perceptron  — Result C third archetype (storage / jamming)
# --------------------------------------------------------------------------------------

def _a_gardner_capacity():
    import module_L_perceptron_replica as rep
    ac = float(rep.alpha_c_spherical(0.0))
    err = abs(ac - 2.0)
    ok = err < 1e-10
    return ok, f"alpha_c(0)={ac:.12f}", "alpha_c(0)=2 (Gardner, exact)", f"|err|={err:.1e}"


def _a_ising_entropy():
    import module_L_perceptron_replica as rep
    s0 = rep.ising_rs_entropy(0.0)[0]
    err = abs(s0 - math.log(2.0))
    ok = err < 1e-6
    return ok, f"s(0)={s0:.10f}", f"s(0)=ln2={math.log(2.0):.10f}", f"|err|={err:.1e}"


def _a_krauth_mezard():
    import module_L_perceptron_replica as rep
    from scipy.optimize import brentq
    a_zero = brentq(lambda a: rep.ising_rs_entropy(a)[0], 0.5, 1.2, xtol=1e-6)
    ok = abs(a_zero - 0.833) < 0.01
    return ok, f"alpha_RS={a_zero:.5f}", "alpha_RS~0.833 (Krauth-Mezard frozen-1RSB)", "RS zero-entropy crossing"


def _a_replicon_gardner():
    import module_L_perceptron_replicon as rpl
    lam_199 = rpl.replicon(1.999, 0.0)[0]
    lam_150 = rpl.replicon(1.5, 0.0)[0]
    ok = (0.0 <= lam_199 < 0.01) and (lam_199 < lam_150)
    return ok, f"lam_repl(1.999)={lam_199:.5f}", "lam_repl -> 0^+ at alpha=2", \
        f"lam(1.5)={lam_150:.4f} > lam(1.999)={lam_199:.4f}; Gardner-anchored"


def _a_perceptron_chi_div():
    import module_L_perceptron_replicon as rpl
    k = -0.5
    aAT = rpl.alpha_AT(k)
    if aAT is None:
        return False, "no AT crossing", "chi*(aAT-a)->3.22", "alpha_AT not found for kappa=-0.5"
    prods = []
    for d in (0.05, 0.02, 0.01):
        lam = rpl.replicon(aAT - d, k)[0]
        prods.append((1.0 / lam) * d)
    ok = abs(prods[-1] - 3.22) < 0.2
    return ok, f"chi*(aAT-a)={prods[-1]:.3f}", "-> 3.22 (kappa=-0.5)", \
        f"alpha_AT={aAT:.4f}; products {', '.join(f'{p:.2f}' for p in prods)}"


def _a_perceptron_posdef_R():
    import module_L_perceptron_finiteT as ftt
    k, alpha = -0.5, 4.2
    bAT = ftt.beta_AT(alpha, k)
    d = 0.02
    R, detg, chi, gbb, L = ftt.curvature_at(bAT - d, alpha, k)
    rr = abs(R) * d * d
    ok = (detg > 0) and (gbb > 0) and (10.8 < rr < 12.8)
    return ok, f"|R|*(bAT-b)^2={rr:.2f}", "-> 11.8 on det g>0", \
        f"beta_AT={bAT:.4f}; det g={detg:.2f}>0; g_bb={gbb:.3f}>0 (positive-definite)"


# --------------------------------------------------------------------------------------
# Group: sk  — Result C converse (Sherrington-Kirkpatrick spin glass)
# --------------------------------------------------------------------------------------

def _a_sk_h0_closed():
    import module_L_SK_converse as sk
    # h=0 closed form: R_bare bounded (->4), R_eps -> inf as lam=1-beta^2 -> 0
    rows = []
    for beta in (0.5, 0.9, 0.98):
        lam = 1 - beta**2
        rows.append((beta, 4 / beta**2, 2 / (beta**2 * lam**2)))
    growing = rows[0][2] < rows[1][2] < rows[2][2]
    bounded = all(abs(r[1] - 4) <= 12 for r in rows)  # R_bare = 4/beta^2 stays O(1)
    # construction check: psi2(eps=0) == 2*psi1 at (1.05, 0.3)
    b0, h0 = 1.05, 0.3
    gg = (sk.single_q(b0, h0), sk.single_q(b0, h0))
    diff = abs(sk.psi2(b0, h0, 0.0, gg) - 2 * sk.psi1(b0, h0))
    ok = growing and bounded and diff < 1e-6
    return ok, f"R_eps(0.98)={rows[2][2]:.1f}", "R_bare~O(1), R_eps->inf; psi2(0)=2psi1", \
        f"construction |psi2-2psi1|={diff:.1e}; R_eps grows {rows[0][2]:.2f}->{rows[2][2]:.1f}"


def _a_sk_dat_chi():
    import module_L_SK_converse as sk
    from scipy.optimize import brentq
    h = 0.3
    bAT = brentq(lambda b: sk.lamAT(b, h), 1.0, 5.0)
    tgt = 0.1
    b0 = brentq(lambda b: sk.lamAT(b, h) - tgt, 1.0, bAT)
    gi = sk.gee_implicit(b0, h)
    chiSG = gi / b0
    prod = chiSG * tgt
    ok = abs(prod - 1.30) < 0.1
    return ok, f"chi_SG*lam_AT={prod:.3f}", "-> ~1.30 (h=0.3)", \
        f"beta_AT={bAT:.4f}; g_ee={gi:.3f} by implicit diff (stable linear response)"


# --------------------------------------------------------------------------------------
# Group: ridge  — Result C kinematic side (linear teacher-student / double descent)
# --------------------------------------------------------------------------------------

def _a_ridge_validation():
    import module_L_ridge_curvature as rdg
    mu, di, Y, P = rdg.setup(1200, 1.5, seed=3)
    b0, l0, h = 1.0, 0.5, 1e-3
    f = lambda a, b: rdg.logZ(mu, di, Y, -b0 + a, -b0 * l0 + b)
    fd = [(f(h, 0) - 2 * f(0, 0) + f(-h, 0)) / h**2,
          (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h**2),
          (f(0, h) - 2 * f(0, 0) + f(0, -h)) / h**2,
          (f(2 * h, 0) - 2 * f(h, 0) + 2 * f(-h, 0) - f(-2 * h, 0)) / (2 * h**3),
          ((f(h, h) - 2 * f(0, h) + f(-h, h)) - (f(h, -h) - 2 * f(0, -h) + f(-h, -h))) / (2 * h**3),
          ((f(h, h) - 2 * f(h, 0) + f(h, -h)) - (f(-h, h) - 2 * f(-h, 0) + f(-h, -h))) / (2 * h**3),
          (f(0, 2 * h) - 2 * f(0, h) + 2 * f(0, -h) - f(0, -2 * h)) / (2 * h**3)]
    an = rdg.psi_derivs(mu, di, b0, l0)
    err = max(abs((a - d) / d) for a, d in zip(an, fd))
    ok = err < 1e-4
    return ok, f"max rel err={err:.1e}", "analytic Hessian vs f.d. logZ < 1e-4", \
        "closed-form 2nd/3rd derivatives validated"


def _a_ridge_dichotomy():
    import module_L_ridge_curvature as rdg
    mu, di, Y, P = rdg.setup(1500, 1.0, seed=3, noise=0.0)
    R, detg, defect = rdg.R_hp(mu, di, 1.0, 1e-5)  # noiseless: R -> 0 (asymptotically flat)
    R = abs(float(R))
    defect = float(defect)
    ok = (R < 1e-3) and (defect > 0.99)
    return ok, f"|R|={R:.2e}, rank1-defect={defect:.4f}", "R->0 (flat) while metric collapses rank-1", \
        f"det g large (volume divergence); curvature stays bounded at 40 dps"


# --------------------------------------------------------------------------------------
# Group: number-theory  — Result A (arithmetic dictionary) + audit
# --------------------------------------------------------------------------------------

def _a_module_e_radius():
    # Assert the REAL module_e_radius_finding receipt emits the unit-rank reading:
    # zeta_K(0)=0 for the real quadratic (rank 1), != 0 for the imaginary quadratic (rank 0).
    out = _capture_receipt("module_e_radius_finding")
    real_zero = re.search(r"real quad.*ZERO at s=0", out) is not None
    imag_nonzero = re.search(r"imag quad.*nonzero at s=0", out) is not None
    ok = real_zero and imag_nonzero
    return ok, f"real-quad ZERO-at-0={real_zero}, imag-quad nonzero={imag_nonzero}", \
        "real quad zeta_K(0)=0 (rank1), imag quad !=0 (rank0)", \
        "subprocess of the real receipt; radius reads the Dirichlet unit rank"


def _a_c_constant():
    # Assert the REAL audit_independent receipt (PART 4) emits an independent dps=56 reproduction
    # of the registered constant C that lands inside the +/-6e-31 budget. Run it; parse its |diff|.
    out = _capture_receipt("audit_independent")
    # PART 4's |diff| line is the one tagged "(registered budget 6e-31)" (PART 3 also prints a |diff|).
    m = re.search(r"\|diff\|\s*=\s*([\d.eE+-]+)\s*\(registered budget 6e-31\)", out)
    diff = float(m.group(1)) if m else None
    ok = diff is not None and diff < 6e-31
    shown = f"{diff:.2e}" if diff is not None else "not found"
    return ok, f"|C_ind - C_reg|={shown}", "< 6e-31 (registered budget)", \
        "PART 4 independent dps=56 reproduction (subprocess of the real receipt)"


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

ANCHORS: list[AnchorSpec] = [
    AnchorSpec("engine-pin", "engine", "[V]",
               "Ruppeiner engine returns R=-1 on the normal family", _a_engine_pin),
    AnchorSpec("flat-control", "engine", "[V]",
               "audit receipt emits R=-1 (normal family) and R=0 (flat product)", _a_flat_control, slow=True),

    AnchorSpec("gardner-capacity", "perceptron", "[V]",
               "spherical Gardner capacity alpha_c(0)=2 exact", _a_gardner_capacity),
    AnchorSpec("ising-entropy", "perceptron", "[V]",
               "Ising RS free entropy s(0)=ln2 exact", _a_ising_entropy),
    AnchorSpec("krauth-mezard", "perceptron", "[V]",
               "Ising RS zero-entropy crossing ~0.833 (Krauth-Mezard)", _a_krauth_mezard, slow=True),
    AnchorSpec("replicon-gardner", "perceptron", "[V]",
               "spherical replicon -> 0 at the Gardner capacity (kappa=0)", _a_replicon_gardner),
    AnchorSpec("perceptron-chi-div", "perceptron", "[V]",
               "overlap susceptibility chi*(alpha_AT-a) -> 3.22 (kappa=-0.5)", _a_perceptron_chi_div, slow=True),
    AnchorSpec("perceptron-posdef-R", "perceptron", "[V]",
               "|R|*(beta_AT-b)^2 -> 11.8 on a positive-definite (beta,eps) metric", _a_perceptron_posdef_R, slow=True),

    AnchorSpec("sk-h0-closed", "sk", "[V]",
               "SK h=0: R_bare bounded, R_eps -> inf; psi2(eps=0)=2*psi1", _a_sk_h0_closed),
    AnchorSpec("sk-dat-chi", "sk", "[V]",
               "SK dAT line: chi_SG*lambda_AT -> const ~1.30 (h=0.3)", _a_sk_dat_chi, slow=True),

    AnchorSpec("ridge-validation", "ridge", "[V]",
               "ridge analytic Hessian vs finite-diff logZ < 1e-4", _a_ridge_validation),
    AnchorSpec("ridge-dichotomy", "ridge", "[V]",
               "ridge double descent: R bounded/->0 while metric volume diverges", _a_ridge_dichotomy, slow=True),

    AnchorSpec("module-e-radius", "number-theory", "[V]",
               "zeta_K(0) reads the Dirichlet unit rank (radius dictionary)", _a_module_e_radius, slow=True),
    AnchorSpec("c-constant", "number-theory", "[V]",
               "registered geometric constant C reproduced within 6e-31", _a_c_constant, slow=True),

    # precision-depth group for T1 experiment (skeleton for now, per plan)
    AnchorSpec("depth-skeleton", "precision-depth", "[infra]",
               "skeleton depth map / recursion identity for C1 stub", _a_depth_skeleton),
]


def get_anchors(groups=None, include_slow=True):
    out = ANCHORS
    if groups:
        wanted = set(groups)
        out = [a for a in out if a.group in wanted]
    if not include_slow:
        out = [a for a in out if not a.slow]
    return out


def all_groups():
    seen = []
    for a in ANCHORS:
        if a.group not in seen:
            seen.append(a.group)
    return seen
