#!/usr/bin/env python
"""L3 Phase 2 -- push n to 400, map the cancellation wall, chart violation sensitivity.
Per li/PREREG_L3p2_push.md. Route B re-engineered: DFT Taylor of the ENTIRE Z(1+eps)=eps*zeta(1+eps)
(no log-branch hazard), exact series-log recurrence, sigma_j, binomial lambda_n. Route A: 100k Odlyzko
zeros + semi-analytic RvM tail (an estimate with error budget, not a bound).

    python li/push_wall.py
"""
import math, json
import numpy as np
from mpmath import mp, mpf, mpc, zeta, euler, pi, log, exp, binomial
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NMAX, R_CONT, M_NODES, DPS_MAIN, DPS_CHECK = 400, 3.0, 4096, 130, 90
TABLE_N = [50, 100, 150, 200, 250, 300, 350, 400]
GREEN, RED, BLUE, NAVY, AMBER = "#1e7d34", "#c0392b", "#2c6fbb", "#15293f", "#9a6a2f"


# ---------------- Route B ----------------
def taylor_Z(K, dps, R, M):
    """Taylor coefficients a_0..a_K of Z(1+eps)=eps*zeta(1+eps) about eps=0, trapezoid-DFT on |eps|=R."""
    mp.dps = dps
    vals = []
    for m in range(M // 2 + 1):                                  # symmetry: f(conj) = conj(f)
        e = R * exp(mpc(0, 2 * pi * m / M))
        vals.append(e * zeta(1 + e))
    a = []
    for k in range(K + 1):
        s = mpf(0)
        for m in range(M):
            v = vals[m] if m <= M // 2 else vals[M - m].conjugate()
            s += (v * exp(mpc(0, -2 * pi * k * m / M))).real
        a.append(s / (M * mpf(R) ** k))
    return a


def series_log(a, K):
    """g with exp(sum g_k x^k) = sum a_k x^k, a_0=1: k g_k = k a_k - sum_{i<k} i g_i a_{k-i}."""
    g = [mpf(0)] * (K + 1)
    for k in range(1, K + 1):
        s = k * a[k]
        for i in range(1, k):
            s -= i * g[i] * a[k - i]
        g[k] = s / k
    return g


def sigmas_from_g(g, K):
    sig = {1: 1 + euler / 2 - log(4 * pi) / 2}
    for k in range(2, K + 1):
        lam_odd = (1 - mpf(2) ** (-k)) * zeta(k)
        triv = (-1) ** k * (lam_odd - 1)
        S_nt = -k * g[k] - triv
        sig[k] = (-1) ** k * S_nt
    return sig


def lambdas_from_sigmas(sig, nmax):
    lams = {}
    for n in range(1, nmax + 1):
        b = mpf(n)                                              # C(n,1)
        s = b * sig[1]
        for j in range(2, n + 1):
            b = b * (n - j + 1) / j                              # incremental binomial
            s += (-1) ** (j + 1) * b * sig[j]
        lams[n] = s
    return lams


def route_B(dps):
    a = taylor_Z(NMAX, dps, R_CONT, M_NODES)
    g = series_log(a, NMAX)
    sig = sigmas_from_g(g, NMAX)
    return lambdas_from_sigmas(sig, NMAX), sig


# ---------------- Route A ----------------
def route_A(nlist, gammas, inject=None):
    """f64 zero-sum + semi-analytic RvM tail integral + explicit error budget."""
    g = gammas
    theta = np.angle(1 - 1 / (0.5 + 1j * g))                    # arg(1-1/rho); |1-1/rho|=1 on-line
    lr = np.log(np.abs(1 - 1 / (0.5 + 1j * g)))                 # ~0 but keep exact
    T = float(g[-1])
    # tail integral: int_T^inf 2(1-cos(n*theta(t))) * log(t/2pi)/(2pi) dt on log grid to 1e10 + remainder
    ts = np.exp(np.linspace(math.log(T), math.log(1e10), 40000))
    th_t = np.angle(1 - 1 / (0.5 + 1j * ts))
    dens = np.log(ts / (2 * math.pi)) / (2 * math.pi)
    out = {}
    for n in nlist:
        core = float(np.sum(2 * (1 - np.exp(n * lr) * np.cos(n * theta))))
        integ = 2 * (1 - np.cos(n * th_t)) * dens
        tail = float(np.trapezoid(integ, ts))
        tail += (n ** 2) * (math.log(1e10 / (2 * math.pi)) + 1) / (2 * math.pi * 1e10)   # analytic remainder
        if inject is not None:
            b, gi = inject
            for rho in (b + 1j * gi, (1 - b) + 1j * gi):
                core += float(2 * (1 - ((1 - 1 / rho) ** n).real))
        # budget: S(T)-fluctuation at the cut + zero accuracy + f64 floor
        budget = 3 * 2 * (1 - math.cos(n * float(np.angle(1 - 1 / (0.5 + 1j * T))))) \
                 + 2 * n * 3e-9 * 0.025 + 1e-9
        out[n] = (core + tail, tail, budget)
    return out


def main():
    print(f"[L3p2] push n<={NMAX}; Route B: DFT Taylor of entire Z, R={R_CONT}, M={M_NODES}, dps={DPS_MAIN}")
    mp.dps = DPS_MAIN
    lam1_closed = 1 + euler / 2 - log(4 * pi) / 2

    print("  Route B @ main dps ...")
    lamB, sig = route_B(DPS_MAIN)
    print(f"    anchors: lambda_1 err = {mp.nstr(abs(lamB[1]-lam1_closed),3)}")
    print(f"             sigma_2  = {mp.nstr(sig[2], 20)} (Lehmer -0.046154317295804603)")
    print(f"             sigma_3  = {mp.nstr(sig[3], 20)} (Lehmer -0.00011115823145210592)")
    print(f"             lambda_40 = {mp.nstr(lamB[40], 15)} (phase-1 run: 30.47737542)")

    print("  Route B @ check dps (wall map via dps-pair agreement) ...")
    lamB2, _ = route_B(DPS_CHECK)
    ns_w, digits = [], []
    for n in range(10, NMAX + 1, 10):
        d = abs(lamB[n] - lamB2[n]) / abs(lamB[n])
        eff = -mp.log(d, 10) if d > 0 else mpf(DPS_CHECK)
        ns_w.append(n); digits.append(float(eff))
    lost = [DPS_CHECK - d for d in digits]
    A = np.polyfit(ns_w, lost, 1)
    pred_slope = math.log10(3.0 / R_CONT) + math.log10(1 + 1 / 14.1436)
    r2 = 1 - np.var(np.array(lost) - np.polyval(A, ns_w)) / max(np.var(lost), 1e-30)
    print(f"    WALL LAW: measured digits-lost slope {A[0]:.4f}/n  vs predicted {pred_slope:.4f}/n"
          f"  (R^2={r2:.4f})  -> n_wall(dps D) ~ (D-{A[1]:.0f})/{max(A[0],1e-9):.4f}")

    print("  Route A: 100k Odlyzko zeros + semi-analytic tail ...")
    gammas = np.loadtxt("li/odlyzko_zeros1.txt")
    lamA = route_A(TABLE_N, gammas)

    print(f"\n  {'n':>4} {'lambda_B':>16} {'A+tail':>16} {'|diff|':>10} {'budget':>10} {'ratio':>7} {'B>0':>5}")
    worst, allpos = 0.0, True
    rows = []
    for n in TABLE_N:
        Aval, tail, budget = lamA[n]
        d = abs(float(lamB[n]) - Aval)
        ratio = d / budget
        worst = max(worst, ratio); allpos &= lamB[n] > 0
        rows.append((n, float(lamB[n]), Aval, d, budget, ratio))
        print(f"  {n:>4} {float(lamB[n]):>16.8f} {Aval:>16.8f} {d:>10.2e} {budget:>10.2e} {ratio:>7.2f} {str(lamB[n]>0):>5}")

    print("\n  violation-sensitivity curve (planted quadruple at gamma*=14.13):")
    ns_all = list(range(1, NMAX + 1))
    sens = {}
    for beta in (0.55, 0.6, 0.7, 0.8, 0.9):
        lamV = route_A(ns_all, gammas[:20000], inject=(beta, 14.13))
        neg = next((n for n in ns_all if lamV[n][0] < 0), None)
        rho = (1 - beta) + 1j * 14.13
        rate = abs(1 - 1 / rho)
        sens[beta] = neg
        print(f"    beta={beta:.2f}: first lambda_n<0 at n={neg}   (growth |1-1/(1-rho*)| = {rate:.5f})")

    print(f"\n  PRE-REGISTERED VERDICTS:")
    v1 = worst <= 3.0
    v2 = abs(A[0] - pred_slope) / pred_slope < 0.15 and r2 > 0.99
    v3 = all(sens[b] is not None for b in sens) and all(
        sens[b1] >= sens[b2] for b1, b2 in zip(sorted(sens), sorted(sens)[1:]))
    print(f"    (1) two-route AGREE to n=400 (worst ratio {worst:.2f} vs 3.0): {'PASS' if v1 else 'FAIL'}")
    print(f"    (2) wall law (slope {A[0]:.4f} vs {pred_slope:.4f}, R2 {r2:.3f}): {'PASS' if v2 else 'FAIL -- measured law replaces prediction'}")
    print(f"    (3) sensitivity monotone in beta: {'PASS' if v3 else 'FAIL'}")
    print(f"    lambda_n > 0 for all n<=400: {allpos}  (RH-consistent; occupied territory, per prereg)")

    json.dump({"table": rows, "wall_slope": float(A[0]), "wall_pred": pred_slope,
               "sens": {str(k): v for k, v in sens.items()}},
              open("li/push_wall_results.json", "w"), indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    nn = np.array(TABLE_N); lb = np.array([r[1] for r in rows]); la = np.array([r[2] for r in rows])
    trend = nn / 2 * (np.log(nn) - math.log(2 * math.pi) + float(euler) - 1)
    axes[0].plot(nn, lb, "o-", color=GREEN, label="route B (zero-free)")
    axes[0].plot(nn, la, "x", color=BLUE, ms=7, label="route A + tail (100k zeros)")
    axes[0].plot(nn, trend, "--", color=AMBER, label="RH trend")
    axes[0].set_title("lambda_n to n=400: two routes", fontsize=9.5, color=NAVY); axes[0].legend(fontsize=7.5, frameon=False)
    axes[1].plot(ns_w, lost, "o", color=NAVY, ms=3)
    axes[1].plot(ns_w, np.polyval(A, ns_w), "-", color=RED,
                 label=f"fit {A[0]:.4f}n  (pred {pred_slope:.4f}n)")
    axes[1].set_title("the cancellation WALL: digits lost vs n", fontsize=9.5, color=NAVY)
    axes[1].set_xlabel("n"); axes[1].legend(fontsize=7.5, frameon=False)
    bs = sorted(sens); axes[2].plot(bs, [sens[b] for b in bs], "o-", color=RED)
    axes[2].set_xlabel("planted beta"); axes[2].set_ylabel("first n with lambda_n < 0")
    axes[2].set_title("violation-detection frontier", fontsize=9.5, color=NAVY)
    fig.tight_layout(); fig.savefig("li/push_wall.png", dpi=160); plt.close(fig)
    print("  wrote li/push_wall.png, li/push_wall_results.json")


if __name__ == "__main__":
    main()
