#!/usr/bin/env python
"""Gate C1 -- Choptuik critical collapse, two-route. Per cosmos/PREREG_C1_choptuik.md.
Polar-areal massless-scalar collapse; constraints solved by vectorized Picard (mass function) +
cumulative log-integration (lapse); RK3 method-of-lines + Kreiss-Oliger dissipation.

    python cosmos/gate_C1_choptuik.py [--smoke]
"""
import sys, math, json
import numpy as np

R_OUT, N, CFL, KO = 60.0, 4096, 0.25, 0.02
R0, SIG = 20.0, 3.0
THRESH = 0.995                                  # 2m/r horizon-formation threshold
SMOKE = "--smoke" in sys.argv


def grid(n=N):
    dr = R_OUT / n
    r = (np.arange(n) + 0.5) * dr
    return r, dr


def constraints(r, dr, Phi, Pi):
    """dm/dr = S(1-2m/r) is LINEAR in m -> exact integrating-factor solution, log-space vectorized
    (Picard saturated near 2m/r -> 1: the plateau-at-0.8 bug caught in the smoke)."""
    S = 2 * math.pi * r * r * (Phi * Phi + Pi * Pi)
    g = 2 * S / r
    P = np.cumsum(g) * dr - 0.5 * g * dr                        # integrating-factor exponent
    with np.errstate(divide="ignore"):
        L = np.log(np.maximum(S * dr, 1e-300)) + P
    logI = np.logaddexp.accumulate(L)
    m = np.exp(logI - P)
    x = np.clip(2 * m / r, None, 0.999999)
    a = 1.0 / np.sqrt(1 - x)
    dlnal = (1 - a * a) / (2 * r) + 2 * math.pi * r * (Phi * Phi + Pi * Pi) + (a * a - 1) / r
    lna = np.cumsum(dlnal) * dr - 0.5 * dlnal * dr
    al = np.exp(lna)
    al *= 1.0 / (a[-1] * al[-1])                                # gauge: alpha(R) = 1/a(R)
    return m, a, al


def rhs(r, dr, Phi, Pi, a, al):
    F1 = al * Pi / a
    F2 = r * r * al * Phi / a
    dPhi = np.gradient(F1, dr)
    dPi = np.gradient(F2, dr) / (r * r)
    # outgoing BC at outer edge: d_t(r phi) + d_r(r phi) = 0 -> approximate on Pi
    dPi[-2:] = (-Phi[-2:] - Pi[-2:]) / (r[-2:] * 0 + 1) * 0 + dPi[-2:]  # keep interior scheme
    dPi[-1] = -(F2[-1] - F2[-2]) / dr / (r[-1] * r[-1]) - Pi[-1] / r[-1]
    # KO dissipation
    for X, dX in ((Phi, dPhi), (Pi, dPi)):
        d = np.zeros_like(X)
        d[2:-2] = X[:-4] - 4 * X[1:-3] + 6 * X[2:-2] - 4 * X[3:-1] + X[4:]
        dX -= KO / dr * d / 16.0
    return dPhi, dPi


def evolve(p, n=N, tmax=55.0, want_curv=False):
    """Return ('bh', M_BH) or ('disp', maxR_over_run)."""
    r, dr = grid(n)
    phi0 = p * np.exp(-((r - R0) / SIG) ** 2)
    Phi = np.gradient(phi0, dr)
    Pi = np.zeros_like(r)
    t, maxR = 0.0, 0.0
    while t < tmax:
        m, a, al = constraints(r, dr, Phi, Pi)
        x = 2 * m / r
        if x.max() >= THRESH:
            i = int(x.argmax())
            return "bh", float(m[i])
        if want_curv:
            Rsc = 8 * math.pi * np.abs(Phi * Phi - Pi * Pi) / (a * a)
            maxR = max(maxR, float(Rsc.max()))
        dt = CFL * dr / max(float((al / a).max()), 1e-9)
        k1 = rhs(r, dr, Phi, Pi, a, al)
        k2 = rhs(r, dr, Phi + dt * k1[0], Pi + dt * k1[1], a, al)
        k3 = rhs(r, dr, Phi + dt / 4 * (k1[0] + k2[0]), Pi + dt / 4 * (k1[1] + k2[1]), a, al)
        Phi = Phi + dt / 6 * (k1[0] + k2[0] + 4 * k3[0])
        Pi = Pi + dt / 6 * (k1[1] + k2[1] + 4 * k3[1])
        t += dt
    return "disp", maxR


def bisect(lo, hi, n=N, iters=45):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        out, _ = evolve(mid, n)
        if out == "bh":
            hi = mid
        else:
            lo = mid
        if (hi - lo) / hi < 3e-14:
            break
    return lo, hi


def main():
    if SMOKE:
        for p in (0.002, 0.05):
            out, v = evolve(p, n=1024, tmax=50.0, want_curv=True)
            print(f"SMOKE p={p}: {out}  value={v:.4e}")
        return
    print(f"[C1] Choptuik collapse; N={N}, R={R_OUT}; prereg cosmos/PREREG_C1_choptuik.md")
    print("  bracketing + bisection ...", flush=True)
    lo, hi = bisect(0.002, 0.05)
    pstar = 0.5 * (lo + hi)
    print(f"  p* = {pstar:.14f}   (bracket width {(hi-lo)/hi:.2e} -- the f64 bisection floor)", flush=True)

    print("\n  Route A: supercritical mass scaling M(p):")
    epsA, MA = [], []
    for k in range(2, 22):
        eps = 10 ** (-k / 2.0)
        p = pstar * (1 + eps)
        out, M = evolve(p)
        tag = "bh" if out == "bh" else "DISP(!)"
        if out == "bh" and M > 0:
            epsA.append(eps); MA.append(M)
        print(f"    eps={eps:.3e}: {tag}  M={M:.5e}", flush=True)
    lx, ly = np.log(epsA), np.log(MA)
    # pre-registered window rule: largest contiguous window with locally stable slope (drop tail where
    # local slope deviates >0.05 from the window median), R^2>0.99
    slopes = np.diff(ly) / np.diff(lx)
    med = float(np.median(slopes))
    keep = [i for i in range(len(slopes)) if abs(slopes[i] - med) < 0.05]
    i0, i1 = min(keep), max(keep) + 1
    gA, b = np.polyfit(lx[i0:i1 + 1], ly[i0:i1 + 1], 1)
    r2A = 1 - np.var(ly[i0:i1 + 1] - (gA * lx[i0:i1 + 1] + b)) / np.var(ly[i0:i1 + 1])
    print(f"  gamma_A = {gA:.4f}  (window {np.exp(lx[i0]):.1e}..{np.exp(lx[i1]):.1e}, R^2={r2A:.4f})")

    print("\n  Route B: subcritical curvature scaling |R|max(p):")
    epsB, RB = [], []
    for k in range(2, 20):
        eps = 10 ** (-k / 2.0)
        p = pstar * (1 - eps)
        out, Rm = evolve(p, want_curv=True)
        if out == "disp" and Rm > 0:
            epsB.append(eps); RB.append(Rm)
        print(f"    eps={eps:.3e}: {out}  Rmax={Rm:.5e}", flush=True)
    lxB, lyB = np.log(epsB), np.log(RB)
    slopesB = np.diff(lyB) / np.diff(lxB)
    medB = float(np.median(slopesB))
    keepB = [i for i in range(len(slopesB)) if abs(slopesB[i] - medB) < 0.1]
    j0, j1 = min(keepB), max(keepB) + 1
    sB, bB = np.polyfit(lxB[j0:j1 + 1], lyB[j0:j1 + 1], 1)
    gB = -sB / 2
    r2B = 1 - np.var(lyB[j0:j1 + 1] - (sB * lxB[j0:j1 + 1] + bB)) / np.var(lyB[j0:j1 + 1])
    print(f"  gamma_B = {gB:.4f}  (slope {sB:.3f} = -2*gamma, R^2={r2B:.4f})")

    print("\n  convergence control: N vs N/2 at eps=1e-3 (supercritical mass):")
    _, M1 = evolve(pstar * 1.001, N)
    _, M2 = evolve(pstar * 1.001, N // 2)
    print(f"    M(N)={M1:.5e}  M(N/2)={M2:.5e}  rel {abs(M1-M2)/abs(M1):.2e}")

    v1 = r2A > 0.99 and (lx[i1] - lx[i0]) / math.log(10) >= 2
    v2 = abs(gA - gB) <= 0.03
    v3 = abs(gA - 0.374) <= 0.02 and abs(gB - 0.374) <= 0.02
    print(f"\n  PRE-REGISTERED VERDICTS:")
    print(f"    (1) Route-A fit (>=2 decades, R^2>0.99): {'PASS' if v1 else 'FAIL'}")
    print(f"    (2) |gamma_A - gamma_B| = {abs(gA-gB):.4f} (<=0.03): {'PASS' if v2 else 'FAIL'}")
    print(f"    (3) both within 0.374 +- 0.02 (Gundlach anchor): {'PASS' if v3 else 'FAIL'}")
    json.dump({"pstar": pstar, "gamma_A": float(gA), "r2A": float(r2A), "gamma_B": float(gB),
               "r2B": float(r2B), "epsA": epsA, "MA": MA, "epsB": epsB, "RB": RB,
               "conv": [float(M1), float(M2)]},
              open(__file__.replace("gate_C1_choptuik.py", "gate_C1_results.json"), "w"), indent=1)
    print("  wrote cosmos/gate_C1_results.json")


if __name__ == "__main__":
    main()
