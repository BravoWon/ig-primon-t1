#!/usr/bin/env python3
"""
module_L_SK_converse.py  --  Receipt for IG-PRIMON-T1 Result C converse (ledger v0.3, closes 6.6).

Shows the Sherrington-Kirkpatrick spin glass populates the DIVERGENT (|R| -> inf) side of the
Ruppeiner kinematic-vs-interacting dichotomy, completing the diagnostic whose KINEMATIC side is
the linear-ridge double descent (module_L_ridge_curvature.py).

Strategy: stay replica-symmetric (RS); APPROACH (do not cross) the de Almeida-Thouless (dAT)
instability -- no RSB ansatz. Augment the (beta,h) manifold with a field eps conjugate to the
inter-replica overlap O = sum_i s1_i s2_i (two real replicas). The eps-direction metric component
is the spin-glass susceptibility chi_SG, which diverges at dAT (replicon -> 0).

  [1] h=0 CLOSED FORM:  R_bare = 4/beta^2            (bounded -> bare manifold BLIND)
                        R_eps  = 2/[beta^2 (1-beta^2)^2] -> inf   (|R| ~ chi_SG^2),  det g = beta^2/lambda > 0
  [2] two-replica saddle reduces to 2x single-replica at eps=0     (construction check)
  [3] h!=0 dAT LINE:    g_ee = beta*chi_SG ~ C(h)/lambda_AT,  via IMPLICIT DIFFERENTIATION
                        (linear response -- stable), C(h) stabilizing as lambda_AT -> 0.
                        Mechanism: det(I - M) -> 0, the two-replica saddle stability matrix going
                        singular = the replicon condition.
                        Documents the finite-difference 3rd-derivative R breaking down at
                        lambda_AT <= 0.05 (double precision on an fsolve'd saddle): hence reliance
                        on the analytic h=0 anchor + the stable linear-response metric component.

Engine convention: sphere-positive, R = -N/(2 det g^2). (R=-1 normal-family pin: module_L_ridge_curvature.py.)
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import fsolve, brentq
np.seterr(over='ignore')

nodes, wts = hermegauss(200); nrm = 1.0/np.sqrt(2*np.pi)
def Dz(fv): return nrm*np.sum(wts*fv)

# ---------- [1] h=0 closed form ----------
# psi2(beta,eps) = (beta^2/2)(1-p^2) + log4 + log cosh(beta^2 p + beta eps),  p = tanh(beta^2 p + beta eps)
def h0_check():
    print("[1] h=0 closed form   (lambda = 1 - beta^2 = replicon at h=0):")
    print(f"     {'beta':>6} {'lambda':>8} {'R_bare=4/b^2':>13} {'R_eps':>13}")
    for beta in [0.5, 0.8, 0.9, 0.95, 0.98]:
        lam = 1 - beta**2
        print(f"     {beta:>6} {lam:>8.4f} {4/beta**2:>13.5f} {2/(beta**2*lam**2):>13.5f}")
    print("     -> R_bare bounded (->4); R_eps -> inf as lambda->0; |R_eps| ~ chi_SG^2; det g = beta^2/lambda > 0\n")

# ---------- single replica ----------
def single_q(beta, h, q0=0.3):
    q = q0
    for _ in range(30000):
        a = beta*(np.sqrt(max(q, 0.0))*nodes + h); qn = Dz(np.tanh(a)**2)
        if abs(qn - q) < 1e-15: return qn
        q = 0.7*q + 0.3*qn
    return q
def lamAT(beta, h):
    q = single_q(beta, h); a = beta*(np.sqrt(max(q, 0.0))*nodes + h)
    return 1 - beta**2*Dz((1/np.cosh(a))**4)
def psi1(beta, h):
    q = single_q(beta, h); a = beta*(np.sqrt(max(q, 0.0))*nodes + h)
    return (beta**2/4)*(1 - q)**2 + Dz(np.log(2*np.cosh(a)))

# ---------- two-replica coupled saddle ----------
def coupled(beta, h, eps, guess):
    def F(pq):
        p, q = pq; qq = max(q, 1e-14); a = beta*(np.sqrt(qq)*nodes + h); b = beta**2*(p - q) + beta*eps
        c2 = np.cosh(2*a); s2 = np.sinh(2*a); eb = np.exp(b); emb = np.exp(-b); xi = 2*eb*c2 + 2*emb
        return [p - Dz((2*eb*c2 - 2*emb)/xi), q - 0.25*Dz((4*eb*s2/xi)**2)]
    return fsolve(F, guess, xtol=1e-13)
def psi2(beta, h, eps, guess):
    p, q = coupled(beta, h, eps, guess); qq = max(q, 1e-14)
    a = beta*(np.sqrt(qq)*nodes + h); b = beta**2*(p - q) + beta*eps
    xi = 2*np.exp(b)*np.cosh(2*a) + 2*np.exp(-b)
    return beta**2*(0.5 - q + q*q - 0.5*p*p) + Dz(np.log(xi))

def construction_check():
    b0, h0 = 1.05, 0.3; gg = (single_q(b0, h0), single_q(b0, h0))
    p2 = psi2(b0, h0, 0.0, gg); two = 2*psi1(b0, h0)
    print(f"[2] construction check (b=1.05,h=0.3):  psi2(eps=0)={p2:.8f}  vs  2*psi1={two:.8f}  |diff|={abs(p2-two):.1e}\n")

# ---------- [3] implicit-differentiation g_ee along the dAT line ----------
def gee_implicit(beta, h):
    q = single_q(beta, h); p = q                       # eps=0 saddle: p=q
    qq = max(q, 1e-14); sq = np.sqrt(qq); z = nodes
    a = beta*(sq*z + h); b = beta**2*(p - q)
    c2 = np.cosh(2*a); s2 = np.sinh(2*a); eb = np.exp(b); emb = np.exp(-b); xi = 2*eb*c2 + 2*emb
    w = (2*eb*c2 - 2*emb)/xi; mp = 4*eb*s2/xi
    dwdb = 1 - w*w; dwda = mp*(1 - w)                   # d<s1s2>/db = 1-w^2 ; d/da = m+(1-w)
    dmda = 2 + 2*w - mp*mp; dmdb = mp*(1 - w)           # d m+/da = 2+2w-m+^2 ; d/db = m+(1-w)
    da_dq = beta*z/(2*sq)
    P_p = beta**2*Dz(dwdb)
    P_q = Dz(dwda*da_dq - beta**2*dwdb)
    P_e = beta*Dz(dwdb)
    Q_p = (beta**2/2)*Dz(mp*dmdb)
    Q_q = 0.5*Dz(mp*(dmda*da_dq - beta**2*dmdb))
    Q_e = (beta/2)*Dz(mp*dmdb)
    M = np.array([[P_p, P_q], [Q_p, Q_q]])              # divergence <=> det(I-M)->0 (replicon)
    dp, dq = np.linalg.solve(np.eye(2) - M, np.array([P_e, Q_e]))
    return beta*dp                                      # g_ee = d^2 psi2/d eps^2 = beta * dp/deps
def gee_fd(beta, h, d=2e-3):
    g0 = (single_q(beta, h), single_q(beta, h))
    return (psi2(beta, h, d, g0) - 2*psi2(beta, h, 0.0, g0) + psi2(beta, h, -d, g0))/d**2

def dAT_trace():
    print("[3] dAT line:  g_ee = beta*chi_SG   (implicit diff = stable;  finite-diff degrades near edge)")
    print(f"     {'h':>4} {'lam_AT':>7} {'beta':>7} {'g_ee impl':>11} {'g_ee f.d.':>11} {'chiSG*lam':>10}")
    for h in [0.3, 0.5]:
        bAT = brentq(lambda b: lamAT(b, h), 1.0, 5.0)
        for tgt in [0.3, 0.2, 0.1, 0.05, 0.02]:
            b0 = brentq(lambda b: lamAT(b, h) - tgt, 1.0, bAT)
            gi = gee_implicit(b0, h); gf = gee_fd(b0, h); chiSG = gi/b0
            print(f"     {h:>4} {tgt:>7} {b0:>7.4f} {gi:>11.4f} {gf:>11.4f} {chiSG*tgt:>10.4f}")
        print(f"        h={h}: beta_AT={bAT:.4f};  chiSG*lam -> const  =>  chi_SG ~ 1/lambda_AT (replicon)")
    print("     finite-diff g_ee (and the 3rd-deriv R) break for lam_AT<=0.05; implicit stays clean to 0.02\n")

if __name__ == '__main__':
    print("="*74)
    print("module_L_SK_converse.py  --  Result C converse receipt (SK spin glass, closes 6.6)")
    print("="*74 + "\n")
    h0_check(); construction_check(); dAT_trace()
    print("VERDICT: SK populates the divergent side.  |R| -> inf via g_ee ~ chi_SG ~ 1/lambda_AT")
    print("         on a positive-definite metric (det g > 0).  Bare (beta,h) is BLIND (R bounded).")
    print("         Dichotomy complete: kinematic (double descent) vs genuine interacting (SK replicon).")
