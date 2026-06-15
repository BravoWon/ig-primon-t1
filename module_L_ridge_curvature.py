#!/usr/bin/env python3
"""
Module L - Geometric diagnostics of learning transitions  (IG-PRIMON-T1, Result C)
==================================================================================
Receipt for the [V] claim: the Ruppeiner curvature of the exactly-solvable linear
(Gaussian) teacher-student / ridge-regression statistical manifold SEPARATES a
kinematic/volume divergence (double descent) from genuine interacting criticality.

Manifold:   P(w) ~ exp(-beta E - beta*lam * 1/2 ||w||^2),  E = 1/2 ||y - Xw/sqrt(N)||^2.
Suff. stats (E, 1/2||w||^2); natural params (t1,t2)=(-beta,-beta*lam); metric=Hess logZ;
Ruppeiner R = -N/(2 det g^2)  (sphere-positive convention; normal family pins R=-1).
logZ is an exact Gaussian integral, diagonal in the eigenbasis of M = X^T X / N, so the
metric and its third derivatives are closed-form sums over the spectrum {mu_i} and the
teacher projections {d_i}.  The interpolation threshold alpha = P/N = 1 is the transition
(Marchenko-Pastur soft edge -> 0); the ridge lam -> 0 is the critical limit.

FINDINGS (beta=1, alpha=1, lam -> 0), confirmed at 40 dps because the metric goes rank-1
and float det g / numerator lose ~11 digits to cancellation (high precision is REQUIRED):
    * det g  diverges  ~ lam^{-3/2}          (metric-volume explosion, MP soft edge)
    * metric collapses to rank-1             (defect -> 1.000)
    * noiseless teacher:  R -> 0,  ~ lam     (asymptotically FLAT)
    * noisy teacher (sigma=0.5): R bounded ~ -1e-4, strictly negative, seed-robust
The scalar curvature NEVER diverges: the divergence is entirely in the metric volume.
R = -N/(2 det g^2) holds with N and det g^2 diverging at matching rates -> the singular
powers CANCEL, exactly the Paper-2 Hagedorn-flat mechanism (Lemma C.1), here operating on
a covariance spectrum rather than an arithmetic one.  Analytic Hessian validated against
finite differences of logZ (max rel err ~9e-6); engine pinned R=-1 on the normal family.
"""
import numpy as np, mpmath as mp
from math import factorial


def R_curv(pxx, pxy, pyy, pxxx, pxxy, pxyy, pyyy):
    """Paper-2 Ruppeiner engine (sphere-positive)."""
    detg = pxx * pyy - pxy * pxy
    N = (pxx * (pxxy * pyyy - pxyy * pxyy)
         - pxy * (pxxx * pyyy - pxyy * pxxy)
         + pyy * (pxxx * pxyy - pxxy * pxxy))
    return -N / (2 * detg * detg), detg


def gauss_pin(t1, t2):
    """Normal-family curvature; must return -1 (engine + convention check)."""
    p11 = -1 / (2 * t2); p12 = t1 / (2 * t2**2); p22 = -t1**2 / (2 * t2**3) + 1 / (2 * t2**2)
    p112 = 1 / (2 * t2**2); p122 = -t1 / t2**3; p222 = 3 * t1**2 / (2 * t2**4) - 1 / t2**3
    return R_curv(p11, p12, p22, 0.0, p112, p122, p222)[0]


def setup(N, alpha, seed=0, noise=0.5):
    """Linear/Gaussian teacher-student spectrum: eigenvalues of M=X^T X/N and teacher proj."""
    P = int(round(alpha * N)); rng = np.random.default_rng(seed)
    X = rng.normal(size=(P, N))
    ws = rng.normal(size=N); ws *= np.sqrt(N) / np.linalg.norm(ws)
    y = X @ ws / np.sqrt(N) + noise * rng.normal(size=P)
    M = X.T @ X / N
    mu, V = np.linalg.eigh(M); mu = np.clip(mu, 0, None)
    di = V.T @ (X.T @ y / np.sqrt(N)); Y = float(y @ y)
    return mu, di, Y, P


def psi_derivs(mu, di, beta, lam):
    """Analytic 2nd/3rd derivatives of logZ in (t1,t2)=(-beta,-beta*lam) (float)."""
    t = -beta; r = 1.0 / (beta * (mu + lam)); di2 = di * di
    A = lambda a, b: factorial(a + b - 1) / 2.0 * np.sum(mu**a * r**(a + b))
    Q = lambda a, b: factorial(a + b) * np.sum(di2 * mu**a * r**(a + b + 1))
    pxx = A(2, 0) + (Q(0, 0) + 2 * t * Q(1, 0) + 0.5 * t * t * Q(2, 0))
    pxy = A(1, 1) + (t * Q(0, 1) + 0.5 * t * t * Q(1, 1))
    pyy = A(0, 2) + (0.5 * t * t * Q(0, 2))
    pxxx = A(3, 0) + (3 * Q(1, 0) + 3 * t * Q(2, 0) + 0.5 * t * t * Q(3, 0))
    pxxy = A(2, 1) + (Q(0, 1) + 2 * t * Q(1, 1) + 0.5 * t * t * Q(2, 1))
    pxyy = A(1, 2) + (t * Q(0, 2) + 0.5 * t * t * Q(1, 2))
    pyyy = A(0, 3) + (0.5 * t * t * Q(0, 3))
    return pxx, pxy, pyy, pxxx, pxxy, pxyy, pyyy


def logZ(mu, di, Y, t1, t2):
    k = -(t1 * mu + t2)
    return (0.5 * len(mu) * np.log(2 * np.pi) - 0.5 * np.sum(np.log(k))
            + 0.5 * t1 * t1 * np.sum(di * di / k) + 0.5 * t1 * Y)


def R_hp(mu_np, di_np, beta, lam, dps=40):
    """High-precision R, det g, and rank-1 defect (mpmath)."""
    mp.mp.dps = dps
    b = mp.mpf(beta); l = mp.mpf(lam); t = -b
    mu = [mp.mpf(float(x)) for x in mu_np]; di2 = [mp.mpf(float(x))**2 for x in di_np]
    r = [1 / (b * (m + l)) for m in mu]; n = len(mu)
    A = lambda a, bb: mp.factorial(a + bb - 1) / 2 * mp.fsum(mu[i]**a * r[i]**(a + bb) for i in range(n))
    Q = lambda a, bb: mp.factorial(a + bb) * mp.fsum(di2[i] * mu[i]**a * r[i]**(a + bb + 1) for i in range(n))
    pxx = A(2, 0) + (Q(0, 0) + 2 * t * Q(1, 0) + t * t / 2 * Q(2, 0))
    pxy = A(1, 1) + (t * Q(0, 1) + t * t / 2 * Q(1, 1))
    pyy = A(0, 2) + (t * t / 2 * Q(0, 2))
    pxxx = A(3, 0) + (3 * Q(1, 0) + 3 * t * Q(2, 0) + t * t / 2 * Q(3, 0))
    pxxy = A(2, 1) + (Q(0, 1) + 2 * t * Q(1, 1) + t * t / 2 * Q(2, 1))
    pxyy = A(1, 2) + (t * Q(0, 2) + t * t / 2 * Q(1, 2))
    pyyy = A(0, 3) + (t * t / 2 * Q(0, 3))
    detg = pxx * pyy - pxy * pxy
    Nn = (pxx * (pxxy * pyyy - pxyy * pxyy) - pxy * (pxxx * pyyy - pxyy * pxxy)
          + pyy * (pxxx * pxyy - pxxy * pxxy))
    return -Nn / (2 * detg * detg), detg, 1 - (pxy * pxy) / (pxx * pyy)


if __name__ == "__main__":
    print("engine pin (normal family, expect -1):", round(gauss_pin(0.3, -0.7), 9))

    # --- validation: analytic derivatives vs finite differences of logZ ---
    mu, di, Y, P = setup(1200, 1.5, seed=3)
    b0, l0 = 1.0, 0.5; t1, t2 = -b0, -b0 * l0; h = 1e-3
    f = lambda a, b: logZ(mu, di, Y, t1 + a, t2 + b)
    fd = [(f(h, 0) - 2 * f(0, 0) + f(-h, 0)) / h**2,
          (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h**2),
          (f(0, h) - 2 * f(0, 0) + f(0, -h)) / h**2,
          (f(2 * h, 0) - 2 * f(h, 0) + 2 * f(-h, 0) - f(-2 * h, 0)) / (2 * h**3),
          ((f(h, h) - 2 * f(0, h) + f(-h, h)) - (f(h, -h) - 2 * f(0, -h) + f(-h, -h))) / (2 * h**3),
          ((f(h, h) - 2 * f(h, 0) + f(h, -h)) - (f(-h, h) - 2 * f(-h, 0) + f(-h, -h))) / (2 * h**3),
          (f(0, 2 * h) - 2 * f(0, h) + 2 * f(0, -h) - f(0, -2 * h)) / (2 * h**3)]
    an = psi_derivs(mu, di, b0, l0)
    print("analytic vs finite-diff logZ, max rel err:",
          max(abs((a - d) / d) for a, d in zip(an, fd)))

    # --- the dichotomy test, 40 dps ---
    for noise, tag in ((0.5, "noisy sigma=0.5"), (0.0, "noiseless")):
        mu, di, Y, P = setup(1500, 1.0, seed=3, noise=noise)
        print(f"\n[{tag}]  alpha=1, beta=1, 40 dps:")
        for lam in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            R, detg, defect = R_hp(mu, di, 1.0, lam)
            print(f"  lam={lam:.0e}: R={mp.nstr(R, 6):>13}  det g={mp.nstr(detg, 5):>11}  "
                  f"rank1-defect={mp.nstr(defect, 4)}")

    # --- seed robustness (noisy, lam=1e-4) ---
    print("\nseed robustness (alpha=1, lam=1e-4, noisy):")
    for s in (1, 2, 3, 7):
        mu, di, Y, P = setup(1500, 1.0, seed=s, noise=0.5)
        R, detg, _ = R_hp(mu, di, 1.0, 1e-4)
        print(f"  seed={s}: R={mp.nstr(R, 6):>13}  det g={mp.nstr(detg, 4)}")
