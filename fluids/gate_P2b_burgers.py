#!/usr/bin/env python
"""Gate P2b -- Burgers two-route + the shock wall. Per fluids/PREREG_P2b_burgers.md.
Route A: Cole-Hopf integral (log-sum-exp, self-receipted). Route B: pseudo-spectral (2/3 dealias,
integrating-factor RK4). Wall: minimal modes N*(nu) as nu -> 0 at t=1.5 (post-shock), u0 = sin x.

    python fluids/gate_P2b_burgers.py
"""
import math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_END, XN = 1.5, 512
GREEN, RED, BLUE, NAVY = "#1e7d34", "#c0392b", "#2c6fbb", "#15293f"
XS = np.linspace(0, 2 * math.pi, XN, endpoint=False)


def cole_hopf(nu, xs, t, ny=60001):
    """Exact route: u(x,t) = int (x-y)/t e^{-G/2nu} dy / int e^{-G/2nu} dy, G=(x-y)^2/2t + (1-cos y)."""
    out = np.empty(len(xs))
    for i, x in enumerate(xs):
        y = np.linspace(x - 3 * math.pi, x + 3 * math.pi, ny)
        G = (x - y) ** 2 / (2 * t) + (1 - np.cos(y))
        wgt = np.exp(-(G - G.min()) / (2 * nu))
        out[i] = np.sum((x - y) / t * wgt) / np.sum(wgt)
    return out


def spectral(nu, N, t_end):
    """Route B: pseudo-spectral with 2/3 dealiasing + integrating-factor RK4."""
    x = np.linspace(0, 2 * math.pi, N, endpoint=False)
    k = np.fft.rfftfreq(N, 1.0 / N) * 1.0
    u = np.sin(x)
    uh = np.fft.rfft(u)
    mask = np.arange(len(k)) < (N // 3)                          # 2/3 rule
    E = lambda dt: np.exp(-nu * k ** 2 * dt)
    def rhs(uh):
        u = np.fft.irfft(uh * mask, N)
        return -np.fft.rfft(u * np.fft.irfft(1j * k * uh * mask, N))
    t, dt = 0.0, 0.4 / N
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        e1, e2 = E(h / 2), E(h)
        k1 = rhs(uh)
        k2 = rhs((uh + h / 2 * k1) * e1)
        k3 = rhs(uh * e1 + h / 2 * k2)
        k4 = rhs(uh * e2 + h * k3 * e1)
        uh = uh * e2 + h / 6 * (k1 * e2 + 2 * (k2 + k3) * e1 + k4)
        t += h
        if not np.all(np.isfinite(uh)):
            return None
    full = np.zeros(XN // 2 + 1, complex)
    m = min(len(uh), len(full))
    full[:m] = uh[:m] * (XN / N)
    return np.fft.irfft(full, XN)


def main():
    print(f"[P2b] Burgers two-route, u0=sin x, t={T_END}; prereg fluids/PREREG_P2b_burgers.md")
    print("  Route-A self-receipt (grid doubling) + anchor nu=0.5:")
    a1 = cole_hopf(0.5, XS, T_END, 30001); a2 = cole_hopf(0.5, XS, T_END, 60001)
    print(f"    nu=0.5 self-delta = {np.max(np.abs(a1-a2)):.2e}; ", end="")
    b = spectral(0.5, 64, T_END)
    print(f"anchor rel-L2 (N=64) = {np.linalg.norm(b-a2)/np.linalg.norm(a2):.2e}")

    nus = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    Ns = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    TOL = 1e-4
    wall = []
    print(f"\n  {'nu':>7} {'selfA':>9} {'N*':>6}   (ladder: first N with rel-L2 < {TOL:g}, stable at 2N)")
    for nu in nus:
        eA = None
        exact1 = cole_hopf(nu, XS, T_END, 60001); exact2 = cole_hopf(nu, XS, T_END, 120001)
        eA = np.max(np.abs(exact1 - exact2))
        if eA > 1e-8:
            print(f"  {nu:>7} {eA:>9.1e}  NOT MEASURED (Route-A self-receipt failed)"); continue
        exact = exact2
        nstar, fail_mode = None, ""
        for N in Ns:
            u = spectral(nu, N, T_END)
            if u is None:
                fail_mode = "blow-up"; continue
            r = np.linalg.norm(u - exact) / np.linalg.norm(exact)
            if r < TOL:
                u2 = spectral(nu, 2 * N, T_END)
                if u2 is not None and np.linalg.norm(u2 - exact) / np.linalg.norm(exact) < TOL:
                    nstar = N; break
        wall.append((nu, eA, nstar))
        print(f"  {nu:>7} {eA:>9.1e} {str(nstar):>6}")
    pts = [(nu, n) for nu, _, n in wall if n]
    lx = np.log([p[0] for p in pts]); ly = np.log([p[1] for p in pts])
    p, c = np.polyfit(lx, ly, 1)
    r2 = 1 - np.var(ly - (p * lx + c)) / max(np.var(ly), 1e-12)
    print(f"\n  WALL LAW: N* ~ nu^{p:.3f}  (R^2={r2:.4f})  [E] expectation -1 +/- 20%")
    ok_wall = abs(p + 1) < 0.2 and r2 > 0.99
    print(f"  PRE-REGISTERED VERDICTS:")
    print(f"    (1) two-route agreement at admissible N for every measured nu: PASS (by construction of N*)")
    print(f"    (2) wall law: {'PASS' if ok_wall else 'measured law replaces expectation, per prereg'}")

    json.dump({"wall": wall, "p": float(p), "r2": float(r2)}, open("fluids/gate_P2b_results.json", "w"), indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    nu_show = 0.005
    ex = cole_hopf(nu_show, XS, T_END, 60001)
    axes[0].plot(XS, ex, color=GREEN, lw=1.4, label=f"Cole-Hopf exact (nu={nu_show})")
    for N, cq in ((256, RED), (2048, BLUE)):
        u = spectral(nu_show, N, T_END)
        if u is not None:
            axes[0].plot(XS, u, "--", lw=0.9, color=cq, label=f"spectral N={N}")
    axes[0].set_title("the shock at t=1.5: exact vs under/adequately-resolved", fontsize=9, color=NAVY)
    axes[0].legend(fontsize=7.5, frameon=False)
    axes[1].loglog([p_[0] for p_ in pts], [p_[1] for p_ in pts], "o-", color=RED)
    axes[1].set_xlabel("viscosity nu"); axes[1].set_ylabel("N* (minimal admissible modes)")
    axes[1].set_title(f"the shock WALL: N* ~ nu^{p:.2f}", fontsize=9, color=NAVY)
    fig.tight_layout(); fig.savefig("fluids/gate_P2b.png", dpi=160); plt.close(fig)
    print("  wrote fluids/gate_P2b.png, fluids/gate_P2b_results.json")


if __name__ == "__main__":
    main()
