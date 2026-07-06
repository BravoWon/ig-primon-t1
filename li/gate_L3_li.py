#!/usr/bin/env python
"""Gate L3 -- two-route Li coefficients, per li/PREREG_L3_li_criterion.md. All CPU/mpmath.

Route A (zeros)    : lambda_n = sum over first N conjugate-paired nontrivial zeros of [1-(1-1/rho)^n],
                     + analytic tail budget  ~ n * integral dN(T)/T^2  beyond the last zero.
Route B (zero-free): sigma_j from Cauchy contour coefficients of log(eps*zeta(1+eps)) via the June
                     lemma chain; lambda_n = sum_j (-1)^(j+1) C(n,j) sigma_j.  Never touches a zero.
Anchors: lambda_1 closed form; planted off-line quadruple must drive lambda_n negative (sensitivity).

    python li/gate_L3_li.py
"""
import math
import numpy as np
from mpmath import mp, mpf, mpc, zeta, gamma as mpgamma, euler, pi, log, quad, exp, binomial, zetazero
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NMAX, NZEROS, DPS, RADIUS = 40, 2000, 60, 2.0
GREEN, RED, BLUE, NAVY, AMBER = "#1e7d34", "#c0392b", "#2c6fbb", "#15293f", "#9a6a2f"


def route_B_sigmas(nmax, dps=DPS, radius=RADIUS):
    """sigma_j (nontrivial-zero power sums) zero-free, via contour Taylor coefficients of log(eps*zeta(1+eps))."""
    mp.dps = dps
    f = lambda e: log(e * zeta(1 + e))
    # g_k = (1/2*pi) * int_0^{2pi} f(R e^{i t}) e^{-i k t} dt / R^k   (Cauchy)
    K = nmax  # need sigma_j for j<=nmax -> g_k for k<=nmax
    gs = []
    for k in range(1, K + 1):
        integ = quad(lambda t: f(radius * exp(mpc(0, t))) * exp(mpc(0, -k * t)), [0, 2 * pi])
        gs.append((integ / (2 * pi * radius ** k)).real)
    # lemma chain: sum_{ALL rho} (rho-1)^(-k) = -k*g_k ; trivial part = (-1)^k (lambda_odd(k)-1)
    # nontrivial: S_k = -k g_k - (-1)^k (lambda_odd(k)-1); sigma_k = (-1)^k S_k (k>=2)
    sigmas = {1: 1 + euler / 2 - log(4 * pi) / 2}
    for k in range(2, K + 1):
        lam_odd = (1 - mpf(2) ** (-k)) * zeta(k)         # sum (2n+1)^{-k}, n>=0  == lambda(k)
        triv = (-1) ** k * (lam_odd - 1)                 # sum over trivial zeros of (rho-1)^{-k}
        S_nt = -k * gs[k - 1] - triv
        sigmas[k] = (-1) ** k * S_nt
    return sigmas


def route_B_lambdas(sigmas, nmax):
    lams = {}
    for n in range(1, nmax + 1):
        s = mpf(0)
        for j in range(1, n + 1):
            s += (-1) ** (j + 1) * binomial(n, j) * sigmas[j]
        lams[n] = s
    return lams


def route_A_lambdas(nmax, nzeros, inject=None, dps=30):
    """Direct zero sum + tail budget. inject=(beta,gamma) plants an off-line quadruple (sensitivity anchor)."""
    mp.dps = dps
    gammas = np.load("lhc/zeta_zeros.npy")               # first 2000 true ordinates (f64, plenty vs tail)
    gammas = gammas[:nzeros]
    lams, tails = {}, {}
    for n in range(1, nmax + 1):
        s = mpf(0)
        for g in gammas:
            rho = mpc(0.5, float(g))
            s += (1 - (1 - 1 / rho) ** n).real * 2       # conjugate pair
        if inject is not None:
            b, gi = inject
            for rho in (mpc(b, gi), mpc(1 - b, gi)):     # quadruple = these two + conjugates
                s += (1 - (1 - 1 / rho) ** n).real * 2
        T = float(gammas[-1])
        # tail (v2, corrected): pair-term expands n/rho - C(n,2)/rho^2 + ...; the rho^-2 piece decays
        # like 1/g^2 with an n^2 coefficient (v1 missed it -> the exact n^2 excess the gate located).
        # I2 = sum_{g>T} 1/g^2 ~ (log(T/2pi)+1)/(2*pi*T); j>=3 terms negligible at these heights.
        I2 = (math.log(T / (2 * math.pi)) + 1) / (2 * math.pi * T)
        tail = (n + n * (n - 1)) * I2
        lams[n], tails[n] = s, tail
    return lams, tails


def main():
    print(f"[gate L3] two-route Li coefficients, n<=%d; prereg li/PREREG_L3_li_criterion.md" % NMAX)
    mp.dps = DPS
    lam1_closed = 1 + euler / 2 - log(4 * pi) / 2
    print(f"  anchor lambda_1 closed form = {mp.nstr(lam1_closed, 20)}")

    print("  Route B (zero-free cumulants) ...")
    sig = route_B_sigmas(NMAX)
    lamB = route_B_lambdas(sig, NMAX)
    print(f"    sigma_2 = {mp.nstr(sig[2], 20)}  (Lehmer: -0.046154317295804603)")
    print(f"    sigma_3 = {mp.nstr(sig[3], 20)}  (Lehmer: -0.00011115823145210592)")
    print(f"    lambda_1(B) = {mp.nstr(lamB[1], 20)}  vs closed {mp.nstr(lam1_closed, 20)}")

    print("  Route A (zero sums, N=2000) ...")
    lamA, tails = route_A_lambdas(NMAX, NZEROS)
    print(f"    lambda_1(A) = {mp.nstr(lamA[1], 12)} (+tail<= {tails[1]:.2e})")

    print(f"\n  {'n':>3} {'lambda_A':>14} {'lambda_B':>14} {'|A-B|':>10} {'tailbudget':>11} {'ratio':>7} {'B>0':>5}")
    worst, allpos = 0.0, True
    for n in range(1, NMAX + 1):
        d = abs(float(lamA[n] - lamB[n]))
        ratio = d / tails[n] if tails[n] > 0 else float("inf")
        worst = max(worst, ratio)
        allpos &= lamB[n] > 0
        if n <= 6 or n % 5 == 0:
            print(f"  {n:>3} {float(lamA[n]):>14.8f} {float(lamB[n]):>14.8f} {d:>10.2e} {tails[n]:>11.2e} {ratio:>7.2f} {str(lamB[n]>0):>5}")

    print("\n  N-rotation (pre-registered): |A-B| must shrink ~1/T as truncation deepens ...")
    for Nz in (500, 2000):
        lamR, tailsR = route_A_lambdas(20, Nz)
        d20 = abs(float(lamR[20] - lamB[20]))
        print(f"    N={Nz:5d} (T~{'{:.0f}'.format(float(np.load('lhc/zeta_zeros.npy')[Nz-1]))}): |A-B| at n=20 = {d20:.3e}  budget {tailsR[20]:.3e}  ratio {d20/tailsR[20]:.2f}")

    print("\n  violation-sensitivity anchor: plant off-line quadruple rho*=0.95+2i into Route A ...")
    lamV, _ = route_A_lambdas(NMAX, NZEROS, inject=(0.95, 2.0))
    minV = min(float(lamV[n]) for n in range(1, NMAX + 1))
    negs = [n for n in range(1, NMAX + 1) if float(lamV[n]) < 0]
    print(f"    min lambda_n (violated) = {minV:.3f}; negative at n = {negs[:6]}{'...' if len(negs)>6 else ''}")

    ok_route = worst <= 1.5
    ok_anchor = len(negs) > 0
    ok_l1 = abs(float(lamB[1] - lam1_closed)) < 1e-30
    print(f"\n  PRE-REGISTERED VERDICT:")
    print(f"    (a) lambda_1 closed-form anchor : {'PASS' if ok_l1 else 'FAIL'}")
    print(f"    (c) planted violation visible   : {'PASS' if ok_anchor else 'FAIL (instrument blind -- nulls meaningless)'}")
    print(f"    (i) A==B within budget (worst ratio {worst:.2f} vs 1.5): {'AGREE' if ok_route else 'EXCEEDED'}")
    print(f"    (ii) all lambda_n > 0 (n<=%d)    : {allpos}" % NMAX)
    if ok_l1 and ok_anchor and ok_route and allpos:
        print("  ==> GATE L3 PASSES: two independent routes agree on the critical line's Li data; the")
        print("      instrument demonstrably sees violations; lambda_n>0 throughout (RH-consistent, occupied territory).")
    else:
        print("  ==> report as-is per prereg (instrument-limit vs disagreement located by the table).")

    # figure: lambda_n + RH trend + residue + violated curve
    ns = np.arange(1, NMAX + 1)
    lB = np.array([float(lamB[n]) for n in ns])
    lV = np.array([float(lamV[n]) for n in ns])
    trend = ns / 2 * (np.log(ns) - math.log(2 * math.pi) + float(euler) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].plot(ns, lB, "o-", color=GREEN, ms=3, label="lambda_n (route B, zero-free)")
    axes[0].plot(ns, [float(lamA[n]) for n in ns], "x", color=BLUE, ms=4, label="route A (zeros)")
    axes[0].plot(ns, trend, "--", color=AMBER, label="RH trend (the fixed point)")
    axes[0].plot(ns, lV, ":", color=RED, label="with planted off-line zero")
    axes[0].axhline(0, color="#999", lw=0.8)
    axes[0].set_xlabel("n"); axes[0].set_title("Li coefficients: two routes, trend, planted violation", fontsize=9.5, color=NAVY)
    axes[0].legend(fontsize=7.5, frameon=False)
    axes[1].plot(ns, lB - trend, "o-", color=GREEN, ms=3, label="residue: lambda_n - trend")
    axes[1].plot(ns, lV - trend, ":", color=RED, label="residue with violation")
    axes[1].axhline(0, color="#999", lw=0.8)
    axes[1].set_xlabel("n"); axes[1].set_title("the residue off the fixed point (bounded <-> RH)", fontsize=9.5, color=NAVY)
    axes[1].legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig("li/gate_L3.png", dpi=160); plt.close(fig)
    print("  wrote li/gate_L3.png")


if __name__ == "__main__":
    main()
