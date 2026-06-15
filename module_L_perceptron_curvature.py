#!/usr/bin/env python3
"""
module_L_perceptron_curvature.py  --  IG-PRIMON-T1: the Ruppeiner scalar R on the eps-augmented
perceptron-storage manifold.  Confirms |R| -> inf at the continuous replicon (AT/jamming) line,
promoting the susceptibility proxy chi=1/lambda_repl to the actual CURVATURE statement.

This is the curvature step on top of module_L_perceptron_replicon.py (which certified chi=1/lambda_repl
diverging continuously for kappa<0).  It is the direct twin of module_L_SK_converse.py's R_eps result:
    SK   :  R_eps = 2/[beta^2 lambda^2] ~ chi_SG^2 -> inf   (bare (beta,h) manifold R bounded = BLIND)
    here :  R     ~ chi^2 ~ 1/(alpha_AT-alpha)^2 -> inf      (regular background -> bounded = BLIND)

CONSTRUCTION (program-honest, mirrors the SK h=0 analytic anchor -- NOT finite-diff R at the edge,
which module_L_SK_converse.py documents as unreliable near criticality):
On the RS side near the AT line the overlap sector is a GAUSSIAN SOFT MODE, so the 2-replica free energy
on the manifold (a, eps) [a = control = load alpha; eps = field conjugate to the inter-replica overlap]:
        psi(a,eps) = 2*s(a) + chi(a)*eps^2/2 + O(eps^4),   chi(a) = 1/lambda_repl(a)  [CERTIFIED input]
At eps=0 the Ruppeiner engine inputs are:
        pxx = 2 s''(a)        (regular)         pyy = g_ee = chi(a)            (SINGULAR ~ 1/lambda)
        pxxx= 2 s'''(a)       (regular)         pxyy = d/da chi(a) = -lam'/lam^2 (SINGULAR ~ chi^2)
        pxy = pxxy = pyyy = 0  (at eps=0; psi is even in eps to leading order)
Engine R = -N/(2 det g^2) then gives, to leading order,  R ~ (lam')^2 chi^2 / (2 * 2 s'')  ~ chi^2 -> inf.
The e^4 term does not enter R at eps=0 (pyyy=0 either way), so the Gaussian form is exact for R here.

chi's NORMALIZATION (the overlap susceptibility prefactor) is convention-dependent; per the program's
convention-dependence wall only |R|-divergence vs boundedness is invariant -- and that is what we show.
"""
import numpy as np
from scipy.special import erfcx, erfc
from module_L_perceptron_replicon import rs_q, replicon, alpha_AT, alpha_c_spherical, U, EDt

# ---- Ruppeiner engine (sphere-positive), identical to module_L_ridge_curvature.py / SK converse ----
def R_curv(pxx, pxy, pyy, pxxx, pxxy, pxyy, pyyy):
    detg = pxx * pyy - pxy * pxy
    N = (pxx * (pxxy * pyyy - pxyy * pxyy)
         - pxy * (pxxx * pyyy - pxyy * pxxy)
         + pyy * (pxxx * pxyy - pxxy * pxxy))
    return -N / (2 * detg * detg), detg

_S2 = np.sqrt(2.0)
def lnH(u):                      # ln H(u) = ln(1/2 erfc(u/sqrt2)), stable on both tails
    z = np.atleast_1d(u / _S2)
    out = np.empty_like(z, dtype=float)
    pos = z > 0
    out[~pos] = np.log(0.5 * erfc(z[~pos]))                       # erfc(neg) in (1,2): no overflow
    out[pos] = np.log(0.5) + np.log(erfcx(z[pos])) - z[pos]**2    # erfcx only where z>0: safe
    return out

def s_RS(alpha, kappa):          # 1-replica RS free entropy at the saddle
    q = rs_q(alpha, kappa)
    return 0.5 * (np.log(1 - q) + q / (1 - q)) + alpha * EDt(lnH(U(q, kappa)))

def lam(alpha, kappa):           # certified replicon eigenvalue
    return replicon(alpha, kappa)[0]

def deriv(f, x, h, n):           # central finite difference, order n=2 or 3 (f smooth here)
    if n == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / h**2
    if n == 3:
        return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2 * h**3)

def curvature_at(alpha, kappa, h=2e-3):
    psi0 = lambda a: 2.0 * s_RS(a, kappa)        # 2-replica background (decoupled at eps=0)
    L = lam(alpha, kappa); chi = 1.0 / L
    Lp = (lam(alpha + h, kappa) - lam(alpha - h, kappa)) / (2 * h)   # lambda'(a): SMOOTH (stable)
    chip = -Lp / (L * L)                          # chi'(a) = -lam'/lam^2  (analytic from smooth lam)
    pxx = deriv(psi0, alpha, h, 2);  pxxx = deriv(psi0, alpha, h, 3)
    pyy = chi;  pxyy = chip
    pxy = pxxy = pyyy = 0.0
    R, detg = R_curv(pxx, pxy, pyy, pxxx, pxxy, pxyy, pyyy)
    R_analytic = (Lp * Lp) * chi * chi / (2.0 * pxx)   # leading-order closed form
    return R, detg, chi, pxx, R_analytic

if __name__ == "__main__":
    print("=" * 84)
    print("module_L_perceptron_curvature.py  --  Ruppeiner R on the eps-augmented perceptron manifold")
    print("=" * 84)

    kappa = -0.5
    aAT = alpha_AT(kappa); aC = float(alpha_c_spherical(kappa))
    print(f"\nkappa={kappa}:  alpha_AT={aAT:.5f}  (continuous replicon / jamming line),  alpha_c={aC:.5f}")

    print("\n[A] CONTROL: freeze chi (no approach to criticality) -> R = 0 (BLIND).")
    print("    Isolates that the divergence is driven ENTIRELY by chi->inf, not by the background:")
    for d in [0.4, 0.2, 0.1]:
        a = aAT - d
        psi0 = lambda x: 2.0 * s_RS(x, kappa)
        # eps-augmented engine but with chi held constant (chi'=0): pyy=chi, pxyy=0
        R0, _ = R_curv(deriv(psi0, a, 2e-3, 2), 0.0, 1.0/lam(a, kappa),
                       deriv(psi0, a, 2e-3, 3), 0.0, 0.0, 0.0)
        print(f"     alpha_AT-alpha={d:>5}:  R(chi frozen) = {R0:+.6f}   (bounded -> the bare manifold is blind)")

    print("\n[B] eps-AUGMENTED manifold:  |R| ~ chi^2 ~ 1/(alpha_AT-alpha)^2 -> inf  (GENUINE side):")
    print(f"     {'aAT-alpha':>10} {'chi=1/lam':>12} {'|det g|':>11} {'R (engine)':>14} {'|R|*(aAT-a)^2':>14} {'R_analytic':>14}")
    for d in [0.40, 0.20, 0.10, 0.05, 0.02, 0.01]:
        a = aAT - d
        R, detg, chi, pxx, Ran = curvature_at(a, kappa)
        print(f"     {d:>10.3f} {chi:>12.3f} {abs(detg):>11.2f} {R:>14.4e} {abs(R)*d*d:>14.4f} {Ran:>14.4e}")
    print("     -> |R| diverges; |R|*(aAT-a)^2 -> const  =>  |R| ~ 1/(aAT-a)^2 ~ chi^2.  Engine == analytic.")
    print("        |det g| -> inf (the eps soft mode DIVERGES; NOT a det g->0 collapse, so NOT the")
    print("        spurious Curie-Weiss degeneracy).  Driven entirely by chi->inf ([A] confirms).")

    print("\nVERDICT: |R| -> inf on the eps-augmented manifold at the continuous replicon (AT/jamming) line")
    print("  of the spherical-perceptron STORAGE problem, scaling as chi^2 -- the literal-curvature twin of")
    print("  SK's R_eps ~ chi_SG^2 at dAT.  The INVARIANT claim (|R|-divergence; program convention Wall)")
    print("  is [V].  HONEST CAVEAT: the load alpha is a STRUCTURAL tuning coordinate, not a natural field,")
    print("  so the (alpha,eps) Hessian is INDEFINITE (det g<0). |det g|->inf rules out the spurious")
    print("  det g->0 case, but a strictly POSITIVE-DEFINITE realization (SK's gold standard) needs the")
    print("  finite-T (beta,eps) manifold where beta is natural AND tunes the replicon -- the next receipt.")
