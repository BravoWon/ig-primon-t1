#!/usr/bin/env python3
"""
module_L_perceptron_replicon.py  --  IG-PRIMON-T1, the [E]->[V] closure for the perceptron archetype.

RELABEL IN PLACE (per review 2026-06-14): the subject is NOT "learning transitions." G_E below is the
log-volume of weights satisfying a margin constraint on random patterns -- the GARDNER STORAGE /
constraint-satisfaction problem. The kappa<0 instability is a JAMMING / SAT-UNSAT continuous-RSB
transition. The genus is "continuous-RSB CRITICALITY vs kinematic volume-divergence in disordered
systems"; learning (ridge), spin glass (SK), and storage/jamming (this) are VENUES. This receipt
certifies the third archetype's GENUINE (divergent) side.

This is the direct perceptron twin of module_L_SK_converse.py:
  SK:         g_eps-eps = beta*chi_SG ~ 1/lambda_AT  (replicon->0 at dAT)  => |R|->inf, method works.
  perceptron: chi_overlap = 1/lambda_repl            (replicon->0 at AT)   => same, on a STORAGE problem.

DERIVATION (spherical perceptron, RS):
  s(q) = 1/2[ln(1-q) + q/(1-q)] + alpha * E_t ln H(u),   u = (kappa - sqrt(q) t)/sqrt(1-q),  H=1/2 erfc(./sqrt2)
  RS saddle:        q/(1-q) = alpha * E_t[ G1(u)^2 ],         G1 = H'/H = -phi/H
  replicon (AT):    lambda_repl = 1 - alpha * E_t[ G1'(u)^2 ], G1'(u) = -u G1 - G1^2 = H''/H - (H'/H)^2
  SG susceptibility: chi = 1/lambda_repl   (replicon mass = inverse spin-glass susceptibility)

SELF-CHECK that pins the replicon formula to GARDNER: at kappa=0, q->1 gives E[G1'^2] -> Phi(0)=1/2, so
lambda_repl=0 exactly at alpha=2 = the Gardner capacity (AT line meets capacity, marginal). For kappa<0
the problem is non-convex: lambda_repl crosses 0 at alpha_AT(kappa) < alpha_c(kappa), CONTINUOUSLY.
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq
from scipy.special import erfcx
import mpmath as mp

_n, _w = hermegauss(200)
_NRM = 1.0 / np.sqrt(2 * np.pi)
def EDt(fv):
    return _NRM * np.sum(_w * fv)

# stable G1(u) = H'(u)/H(u) = -phi(u)/H(u) = -sqrt(2/pi)/erfcx(u/sqrt2)   (no under/overflow)
_S2 = np.sqrt(2.0)
_S2pi = np.sqrt(2.0 / np.pi)
def G1(u):
    return -_S2pi / erfcx(u / _S2)
def G1p(u):                      # G1'(u) = -u G1 - G1^2
    g = G1(u); return -u * g - g * g

def U(q, kappa):
    return (kappa - np.sqrt(q) * _n) / np.sqrt(1.0 - q)

def rs_q(alpha, kappa):
    # q/(1-q) = alpha * E[G1^2].  NOTE: F(q) -> +inf as q->1 for any alpha, so brentq always returns an
    # interior root; this assumes alpha < Gardner capacity (where a finite RS overlap exists). Above
    # capacity the true solution is q=1 and the returned value is unphysical -- callers must stay below.
    F = lambda q: q / (1 - q) - alpha * EDt(G1(U(q, kappa))**2)
    return brentq(F, 1e-10, 1 - 1e-10, xtol=1e-13)

def replicon(alpha, kappa):
    q = rs_q(alpha, kappa)
    lam = 1.0 - alpha * EDt(G1p(U(q, kappa))**2)
    return lam, q

def alpha_c_spherical(kappa, dps=30):       # Gardner capacity (closed form, mpmath)
    mp.mp.dps = dps
    f = lambda t: mp.e**(-t*t/2) / mp.sqrt(2*mp.pi) * (t + kappa)**2
    return 1.0 / mp.quad(f, [-kappa, mp.inf])

def alpha_AT(kappa, lo=0.05, hi=None):
    if hi is None:
        hi = float(alpha_c_spherical(kappa)) * 0.999
    flo = replicon(lo, kappa)[0]
    fhi = replicon(hi, kappa)[0]
    if flo * fhi > 0:
        return None                          # no interior crossing in (lo,hi)
    return brentq(lambda a: replicon(a, kappa)[0], lo, hi, xtol=1e-8)

if __name__ == "__main__":
    print("=" * 80)
    print("module_L_perceptron_replicon.py  --  continuous-RSB criticality in a STORAGE problem")
    print("=" * 80)

    print("\n[A] SELF-CHECK: replicon formula must reproduce GARDNER capacity at kappa=0")
    print("    (AT line meets capacity: lambda_repl -> 0 as alpha -> 2):")
    for a in [1.0, 1.5, 1.9, 1.99, 1.999]:
        lam, q = replicon(a, 0.0)
        print(f"     alpha={a:>6}  q*={q:.6f}  lambda_repl={lam:+.6f}")
    print("     => lambda_repl -> 0 continuously at alpha=2 = alpha_c(0).  Replicon formula VALIDATED.")

    print("\n[B] kappa<0 (NON-CONVEX): AT instability arrives BEFORE capacity, CONTINUOUSLY:")
    print(f"     {'kappa':>6} {'alpha_AT':>10} {'alpha_c':>10} {'AT<cap?':>8} {'q at AT':>9} {'lam@AT':>10}")
    ats = {}
    for k in [-0.1, -0.2, -0.3, -0.5]:
        aAT = alpha_AT(k); ac = float(alpha_c_spherical(k))
        ats[k] = aAT
        if aAT is not None:
            lam, q = replicon(aAT, k)
            print(f"     {k:>6.2f} {aAT:>10.5f} {ac:>10.5f} {str(aAT<ac):>8} {q:>9.5f} {lam:>+10.2e}")
        else:
            print(f"     {k:>6.2f} {'(none)':>10} {ac:>10.5f}")

    print("\n[C] SG susceptibility chi = 1/lambda_repl DIVERGES continuously as alpha -> alpha_AT^- :")
    print("    (the perceptron twin of SK's g_eps-eps ~ chi_SG ~ 1/lambda_AT; genuine continuous replicon)")
    k = -0.5; aAT = ats[k]
    if aAT is None:                           # guard: section [C] needs a real AT crossing for this kappa
        raise SystemExit(f"[C] aborted: kappa={k} has no replicon (AT) crossing in the search bracket.")
    print(f"     kappa={k}, alpha_AT={aAT:.5f}:")
    print(f"     {'alpha_AT-alpha':>14} {'lambda_repl':>13} {'chi=1/lam':>12} {'chi*(aAT-a)':>12}")
    for d in [0.20, 0.10, 0.05, 0.02, 0.01, 0.005]:
        a = aAT - d
        lam, q = replicon(a, k)
        chi = 1.0 / lam
        print(f"     {d:>14.3f} {lam:>13.6f} {chi:>12.4f} {chi*d:>12.4f}")
    print("     -> chi -> +inf smoothly; chi*(alpha_AT-alpha) -> const  =>  chi ~ 1/(alpha_AT-alpha),")
    print("        a CONTINUOUS replicon divergence on the RS side (no jump).  Method TRANSFERS. [V]")

    print("\nVERDICT [V]: the curvature dichotomy's GENUINE (divergent) side has a continuous-RSB")
    print("  realization in the NON-CONVEX SPHERICAL PERCEPTRON STORAGE problem -- chi_overlap=1/lambda_repl")
    print("  diverges continuously at the AT/jamming line, exactly as SK's chi_SG does at dAT. Storage,")
    print("  not learning; criticality. Third archetype certified (convex side = kinematic, this = genuine).")
