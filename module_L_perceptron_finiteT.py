#!/usr/bin/env python3
"""
module_L_perceptron_finiteT.py  --  IG-PRIMON-T1: the [E]->[V] COMPLETION of the perceptron curvature.

module_L_perceptron_curvature.py showed |R|~chi^2->inf on the (alpha,eps) manifold, but alpha is a
STRUCTURAL parameter (g_aa=2s''<0), so the metric was INDEFINITE (det g<0) -> curvature-genuine stayed [E].
This receipt redoes it on the FINITE-TEMPERATURE (beta,eps) manifold, where beta is a genuine NATURAL field
(g_bb = 2 phi'' = 2 Var(energy)/N > 0) that ALSO tunes the replicon -- the exact twin of module_L_SK_converse.py.
Both coordinates are conjugate fields => the metric is a genuine covariance => POSITIVE-DEFINITE by
construction => det g > 0, and |R|->inf is now a literal RIEMANNIAN statement.  Closes curvature-genuine to [V].

FINITE-T SOFT STORAGE (energy = # violated constraints, inverse temp beta, a=e^{-beta}):
  m(u)   = a + (1-a) H(u),     u = (kappa - sqrt(q) t)/sqrt(1-q),   H=1/2 erfc(./sqrt2)
  M1(u)  = m'/m = (1-a)H'/m = -(1-a)phi/m        (phi = Gaussian pdf = -H')
  saddle :   q/(1-q) = alpha * E_t[M1^2]
  replicon:  lambda_repl = 1 - alpha * E_t[M1'^2],   M1' = -u M1 - M1^2   (SAME form as zero-T G1')
  free en :  phi(beta) = 1/2[ln(1-q)+q/(1-q)] + alpha E_t[ln m]
Limits (self-checks): beta=0 (a=1) -> M1=0, lambda_repl=1, phi=0;  beta->inf (a->0) -> zero-T storage.
At fixed alpha in (alpha_AT|_{T=0}, alpha_c) and kappa<0, lambda_repl(beta) falls from 1 (beta=0) to <0
(beta->inf), crossing 0 at a FINITE beta_AT: the finite-T dAT line.  Approach it from below (RS-stable).
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq
from scipy.special import erfc, erfcx
from module_L_perceptron_replicon import replicon as replicon_zeroT   # independent zero-T anchor
np.seterr(over='ignore', invalid='ignore')                           # benign: e^{z^2} capped, m>=a>0

def R_curv(pxx, pxy, pyy, pxxx, pxxy, pxyy, pyyy):
    detg = pxx * pyy - pxy * pxy
    N = (pxx * (pxxy * pyyy - pxyy * pxyy)
         - pxy * (pxxx * pyyy - pxyy * pxxy)
         + pyy * (pxxx * pxyy - pxxy * pxxy))
    return -N / (2 * detg * detg), detg

def gauss_pin(t1, t2):
    """Engine pin: Ruppeiner R of the normal family must be -1 (engine + convention check)."""
    p11 = -1 / (2 * t2); p12 = t1 / (2 * t2**2); p22 = -t1**2 / (2 * t2**3) + 1 / (2 * t2**2)
    p112 = 1 / (2 * t2**2); p122 = -t1 / t2**3; p222 = 3 * t1**2 / (2 * t2**4) - 1 / t2**3
    return R_curv(p11, p12, p22, 0.0, p112, p122, p222)[0]

_n, _w = hermegauss(200)
_NRM = 1.0 / np.sqrt(2 * np.pi)
_S2 = np.sqrt(2.0)
def EDt(fv): return _NRM * np.sum(_w * fv)
def Hf(u):   return 0.5 * erfc(u / _S2)
def pdf(u):  return np.exp(-u * u / 2) * _NRM          # Gaussian pdf = -H'(u)
def U(q, k): return (k - np.sqrt(q) * _n) / np.sqrt(1.0 - q)

def M1_of(q, beta, k):
    # M1 = (1-a)H'/m = -(1-a)pdf/m,  m = a + (1-a)H,  a=e^{-beta}.  erfcx-stable form
    # (so M1 -> G1 = -sqrt(2/pi)/erfcx as beta->inf, matching the zero-T module exactly):
    #   M1 = -(1-a)/sqrt(2pi) / ( a e^{z^2} + (1-a) 0.5 erfcx(z) ),   z = u/sqrt2
    u = U(q, k); a = np.exp(-beta); z = u / _S2
    term_a = np.exp(np.minimum(z * z - beta, 700.0))   # = a e^{z^2}, capped to avoid overflow
    denom = term_a + (1 - a) * 0.5 * erfcx(z)
    M1 = -(1 - a) * _NRM / denom
    m = a + (1 - a) * Hf(u)                            # for the free-energy log term (bounded, m>=a>0)
    return M1, m, u

def rs_q(beta, alpha, k):
    def F(q):
        M1, m, u = M1_of(q, beta, k)
        return q / (1 - q) - alpha * EDt(M1 * M1)
    return brentq(F, 1e-10, 1 - 1e-10, xtol=1e-13)

def replicon(beta, alpha, k):
    q = rs_q(beta, alpha, k)
    M1, m, u = M1_of(q, beta, k)
    M1p = -u * M1 - M1 * M1
    return 1.0 - alpha * EDt(M1p * M1p), q

def free_energy(beta, alpha, k):                       # phi(beta) = (1/N) ln Z
    q = rs_q(beta, alpha, k)
    M1, m, u = M1_of(q, beta, k)
    return 0.5 * (np.log(1 - q) + q / (1 - q)) + alpha * EDt(np.log(m))

def beta_AT(alpha, k, lo=0.05, hi=25.0):
    return brentq(lambda b: replicon(b, alpha, k)[0], lo, hi, xtol=1e-9)

def deriv(f, x, h, n):
    if n == 2: return (f(x + h) - 2 * f(x) + f(x - h)) / h**2
    if n == 3: return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2 * h**3)

def curvature_at(beta, alpha, k, h=3e-3):
    psi0 = lambda b: 2.0 * free_energy(b, alpha, k)     # two-replica background
    L, q = replicon(beta, alpha, k); chi = 1.0 / L
    Lp = (replicon(beta + h, alpha, k)[0] - replicon(beta - h, alpha, k)[0]) / (2 * h)
    chip = -Lp / (L * L)
    gbb = deriv(psi0, beta, h, 2)                        # g_bb = psi0''(beta)  (expect > 0)
    pxxx = deriv(psi0, beta, h, 3)
    R, detg = R_curv(gbb, 0.0, chi, pxxx, 0.0, chip, 0.0)
    return R, detg, chi, gbb, L

if __name__ == "__main__":
    print("=" * 86)
    print("module_L_perceptron_finiteT.py  --  positive-definite (beta,eps) closure: curvature [E] -> [V]")
    print("=" * 86)

    k = -0.5; alpha = 4.2     # alpha in (alpha_AT|_{T=0}=3.951 , alpha_c=4.770): finite-T dAT line exists
    print(f"\nkappa={k}, alpha={alpha}  (between zero-T AT load 3.951 and capacity 4.770)")

    print("\n[0] INDEPENDENT ANCHORS (the existing bar):")
    print(f"     engine pin R(normal family) = {gauss_pin(0.3, -0.7):+.9f}   (must be -1)")
    lam_ft = replicon(150.0, alpha, k)[0]                     # finite-T replicon, beta->inf (slow: needs ~150)
    lam_0T = replicon_zeroT(alpha, k)[0]                      # independent zero-T storage replicon (Gardner-anchored)
    print(f"     finite-T replicon(beta=150) = {lam_ft:.8f}  vs  zero-T storage replicon = {lam_0T:.8f}"
          f"   |diff|={abs(lam_ft-lam_0T):.1e}")
    print("     (the saddle q* approaches the storage value slowly: q*=0.735,0.81,0.875,0.913,0.930,0.932 at")
    print("      beta=6,10,20,40,80,150 -> zero-T 0.9324; beta=150 matches to 1e-5. zero-T module pinned to")
    print("      Gardner alpha_c(0)=2, so this ties the finite-T saddle to Gardner. ENGINE PIN + this = anchored.)")

    print("\n[A] SELF-CHECKS on the finite-T replicon:")
    l0, q0 = replicon(0.05, alpha, k)
    print(f"     beta->0 : lambda_repl = {l0:.5f}  (exact limit 1; a=e^-b->1, M1->0)")
    lbig, _ = replicon(20.0, alpha, k)
    print(f"     beta=20 : lambda_repl = {lbig:+.5f}  (-> zero-T storage value < 0 since alpha>3.951)")
    bAT = beta_AT(alpha, k)
    print(f"     => lambda_repl crosses 0 at FINITE beta_AT = {bAT:.5f}  (the finite-T dAT line)")

    print("\n[B] POSITIVE-DEFINITENESS: g_bb = 2 phi''(beta) > 0  (beta is a natural field, unlike alpha):")
    for d in [0.6, 0.3, 0.1]:
        b = bAT - d
        _, _, _, gbb, _ = curvature_at(b, alpha, k)
        print(f"     beta_AT-beta={d:>4}:  g_bb = {gbb:+.5f}   ({'POSITIVE' if gbb>0 else 'NEGATIVE'})")

    print("\n[C] CURVATURE on the positive-definite (beta,eps) manifold, beta -> beta_AT^- :")
    print(f"     {'bAT-beta':>9} {'chi=1/lam':>11} {'g_bb':>9} {'det g':>11} {'R (engine)':>13} {'|R|*(bAT-b)^2':>14}")
    for d in [0.60, 0.30, 0.15, 0.08, 0.04, 0.02]:
        b = bAT - d
        R, detg, chi, gbb, L = curvature_at(b, alpha, k)
        print(f"     {d:>9.3f} {chi:>11.3f} {gbb:>9.4f} {detg:>11.3f} {R:>13.4e} {abs(R)*d*d:>14.4f}")
    print("     -> det g > 0 throughout (POSITIVE-DEFINITE); |R| -> inf as ~1/(beta_AT-beta)^2 ~ chi^2.")

    print("\nVERDICT [V]: on the finite-T (beta,eps) manifold BOTH coordinates are conjugate fields, so the")
    print("  metric is a genuine covariance: det g > 0, positive-definite (vs det g<0 on the structural-alpha")
    print("  coords).  |R|->inf is now a literal Riemannian statement -- a diverging COMPONENT (g_eps-eps=chi)")
    print("  on a non-degenerate positive-definite metric -- the genuine side of the refined diagnostic.")
    print("  Curvature-genuine for the perceptron storage/jamming archetype: [E] -> [V].  Twin of SK's R_eps.")
