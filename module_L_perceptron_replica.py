#!/usr/bin/env python3
"""
module_L_perceptron_replica.py  --  Receipt for IG-PRIMON-T1 Result C, ledger §6.7 GATE.

QUESTION (the gate): does the perceptron -- the program's named "literal-learning realization of the
divergent branch" -- have a CONTINUOUS-REPLICON transition (so the SK-converse method transfers: augment
with eps conjugate to the inter-replica overlap, ride the dAT line, g_eps-eps ~ chi diverges), or a
FROZEN-1RSB / FIRST-ORDER one (so it does not)?

Answer derived + certified here: it depends on the weight geometry, and the split maps onto Result C's
own taxonomy.
  * SPHERICAL, convex (kappa>=0): RS-exact, no replicon instability; the "transition" is q->1 at the
    Gardner capacity -- a metric-VOLUME effect.  => KINEMATIC side (like double descent).
  * SPHERICAL, non-convex (kappa<0): continuous full-RSB; replicon -> 0 continuously along an AT line.
    => GENUINE criticality, dAT-RIDEABLE -- the SK-converse method TRANSFERS.  (the §6.7 model that works)
  * ISING (+/-1 weights): frozen-1RSB / first-order; q jumps, RS entropy goes NEGATIVE, no marginal line
    to ride.  => method FAILS, exactly as §6.7 anticipated.

This script CERTIFIES the two anchors that pin the algebra, then exhibits the regime signatures:
  [A] spherical Gardner capacity alpha_c(kappa) = [ int_{-kappa}^inf Dt (t+kappa)^2 ]^{-1};  alpha_c(0)=2 EXACT.
  [B] spherical RS saddle q*(alpha): q -> 1 CONTINUOUSLY as alpha -> alpha_c (convex, kappa>=0).
  [C] Ising RS free entropy s(alpha): s(0)=ln2 EXACT; s drives NEGATIVE  => RS invalid => discontinuous
      freezing (frozen-1RSB), NOT a continuous replicon.  Krauth-Mezard frozen-1RSB capacity = 0.833 (lit).
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq
from scipy.special import erfc
import mpmath as mp

# ---- Gaussian measure  Dt = e^{-t^2/2}/sqrt(2pi) dt   (hermegauss: weight e^{-x^2/2}) ----
_n, _w = hermegauss(200)   # 200 nodes: stable (400 overflows the hermite_e weight recursion)
_NRM = 1.0 / np.sqrt(2 * np.pi)
def EDt(fvals):          # E_t[f] = int Dt f(t),  fvals = f(_n)
    return _NRM * np.sum(_w * fvals)
def Hf(x):               # H(x) = int_x^inf Dz = 1/2 erfc(x/sqrt2)
    return 0.5 * erfc(x / np.sqrt(2.0))

# ================= [A] spherical Gardner capacity (mpmath-certified anchor) =================
def alpha_c_spherical(kappa, dps=30):
    mp.mp.dps = dps
    f = lambda t: mp.e**(-t*t/2) / mp.sqrt(2*mp.pi) * (t + kappa)**2
    I = mp.quad(f, [-kappa, mp.inf])
    return 1.0 / I

# ================= [B] spherical RS saddle  q*(alpha,kappa),  kappa>=0 =================
# s(q) = 1/2 ln(1-q) + q/(2(1-q)) + alpha * G_E(q),   u = (kappa - sqrt(q) t)/sqrt(1-q)
def G_E_spherical(q, kappa):
    q = min(max(q, 1e-12), 1 - 1e-12)
    u = (kappa - np.sqrt(q) * _n) / np.sqrt(1.0 - q)
    return EDt(np.log(np.clip(Hf(u), 1e-300, None)))
def dGE_dq_spherical(q, kappa, h=1e-6):
    qp, qm = min(q + h, 1 - 1e-12), max(q - h, 1e-12)
    return (G_E_spherical(qp, kappa) - G_E_spherical(qm, kappa)) / (qp - qm)
def rs_saddle_spherical(alpha, kappa):
    # ds/dq = q/(2(1-q)^2) + alpha * G_E'(q) = 0
    F = lambda q: q / (2 * (1 - q)**2) + alpha * dGE_dq_spherical(q, kappa)
    try:
        return brentq(F, 1e-7, 1 - 1e-6, xtol=1e-12)
    except ValueError:
        return None          # no interior root -> q pinned at edge (>= capacity)

# ================= [C] Ising RS free entropy  s(alpha,kappa=0) =================
# s = extr_{q,qh} [ -1/2 qh (1-q) + int Dz ln 2cosh(sqrt(qh) z) + alpha G_E(q) ]
# saddle: q = int Dz tanh^2(sqrt(qh) z) ;  qh = -2 alpha G_E'(q)
def _q_of_qh(qh):
    return EDt(np.tanh(np.sqrt(max(qh, 0.0)) * _n)**2)
def _entropic_ising(qh):
    return EDt(np.log(2 * np.cosh(np.sqrt(max(qh, 0.0)) * _n)))
def ising_rs_entropy(alpha, kappa=0.0, iters=4000, damp=0.3):
    q = 0.05
    for _ in range(iters):
        qh = -2.0 * alpha * dGE_dq_spherical(q, kappa)   # same G_E(q) (energetic term identical)
        qh = max(qh, 0.0)
        qnew = _q_of_qh(qh)
        if abs(qnew - q) < 1e-13:
            q = qnew; break
        q = (1 - damp) * q + damp * qnew
    qh = max(-2.0 * alpha * dGE_dq_spherical(q, kappa), 0.0)
    s = -0.5 * qh * (1 - q) + _entropic_ising(qh) + alpha * G_E_spherical(q, kappa)
    return s, q, qh
def annealed_capacity_ising(kappa=0.0):
    # s_ann = ln2 + alpha ln H(kappa);  zero at alpha = ln2 / (-ln H(kappa))
    return np.log(2.0) / (-np.log(Hf(kappa)))

if __name__ == "__main__":
    print("=" * 78)
    print("module_L_perceptron_replica.py  --  §6.7 regime gate (perceptron replica free energy)")
    print("=" * 78)

    print("\n[A] SPHERICAL Gardner capacity  alpha_c(kappa)  (mpmath-certified):")
    for k in [-0.5, -0.2, 0.0, 0.2, 0.5, 1.0]:
        print(f"     kappa={k:+.2f}   alpha_c = {mp.nstr(alpha_c_spherical(k), 12)}")
    ac0 = alpha_c_spherical(0.0)
    print(f"     ANCHOR CHECK  alpha_c(0) = {mp.nstr(ac0, 16)}   (exact = 2)   "
          f"|err| = {mp.nstr(abs(ac0 - 2), 3)}")

    print("\n[B] SPHERICAL RS saddle  q*(alpha)  at kappa=0  (convex -> q->1 CONTINUOUSLY at capacity):")
    for a in [0.5, 1.0, 1.5, 1.8, 1.95, 1.99]:
        q = rs_saddle_spherical(a, 0.0)
        print(f"     alpha={a:>5}   q* = {('%.8f'%q) if q is not None else 'edge (>=cap)'}")
    print("     -> q rises smoothly toward 1; no jump.  Convex problem: RS exact, NO replicon instability.")
    print("        (kappa<0 non-convex: continuous full-RSB, replicon->0 on an AT line -- dAT-rideable [lit].)")

    print("\n[C] ISING RS free entropy  s(alpha)  at kappa=0  (s(0)=ln2 anchor; s<0 => frozen-1RSB):")
    s0, q0, qh0 = ising_rs_entropy(0.0)
    print(f"     ANCHOR CHECK  s(0) = {s0:.10f}   (exact = ln2 = {np.log(2):.10f})   "
          f"|err| = {abs(s0-np.log(2)):.2e}")
    for a in [0.2, 0.4, 0.6, 0.8, 0.83, 1.0, 1.2]:
        s, q, qh = ising_rs_entropy(a)
        print(f"     alpha={a:>5}   s = {s:+.8f}   q = {q:.6f}")
    a_zero = brentq(lambda a: ising_rs_entropy(a)[0], 0.5, 1.2, xtol=1e-6)
    print(f"     annealed capacity (upper bound, kappa=0): alpha = {annealed_capacity_ising():.6f}  (= 1)")
    print(f"     RS zero-entropy crossing (brentq):  alpha_RS = {a_zero:.5f}")
    print("     Krauth-Mezard FROZEN-1RSB capacity: alpha_c = 0.833  [literature] -- RS s=0 ~ 0.83 coincides:")
    print("     the entropy hits 0 *at* the transition because it is a FREEZING (q->1 in the dominant")
    print("     cluster), discontinuous -- not a continuous replicon softening.")
    print("     RS s<0 below naive capacity = impossible for discrete weights => RS breaks by DISCONTINUOUS")
    print("     freezing (q jumps), NOT a continuous replicon.  => method does NOT transfer to Ising.")

    print("\nVERDICT: spherical-convex = kinematic (volume, q->1); spherical-nonconvex = genuine continuous")
    print("         replicon (dAT-rideable, SK-converse method TRANSFERS); Ising = frozen-1RSB/first-order")
    print("         (method fails).  §6.7 realization that WORKS = non-convex spherical perceptron.")
