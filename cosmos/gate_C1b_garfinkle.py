#!/usr/bin/env python
"""Gate C1b -- Choptuik collapse in DOUBLE-NULL coordinates (Garfinkle-style scheme), the costed v2.
Same prereg (PREREG_C1_choptuik.md), escalation (b). ds^2 = -a^2 du dv + r^2 dOmega^2; the u-march:
  (r r,u),v = -a^2/4          (exact flat anchor: d/dv(-r/2) = -1/4)
  (r f,u),v = -r,u f,v        (wave)
  s,u(v) = s,u(axis) + int [ (r,u r,v + a^2/4)/r^2 - 4pi f,u f,v ] dv,   a = e^s, s,u(axis) = -s,v
Initial slice u=0: r = v/2, s from s,v = 2 pi v (f,v)^2, f = p exp(-((v-20)/3)^2).
Axis v=u advances each step (natural focusing). Horizon: r,v -> 0 (m = (r/2)(1 + 4 r,u r,v e^{-2s})).

    python cosmos/gate_C1b_garfinkle.py [--smoke]
"""
import sys, math, json
import numpy as np

VMAX, NV, UMAX = 60.0, 8192, 42.0
V0, SIG = 20.0, 3.0
SMOKE = "--smoke" in sys.argv
DV = VMAX / NV
V = (np.arange(NV) + 0.5) * DV


def initial(p):
    f = p * np.exp(-((V - V0) / SIG) ** 2)
    fv = np.gradient(f, DV)
    sv = 2 * math.pi * V * fv * fv
    s = np.cumsum(sv) * DV - 0.5 * sv * DV
    r = V / 2.0
    return r.copy(), s, f


def cumtrap_off(y, off):
    """int from the TRUE axis (v=u) to each grid point: leading sliver off*y[0] + trapezoid."""
    out = np.empty_like(y)
    out[0] = off * y[0]
    out[1:] = out[0] + np.cumsum((y[1:] + y[:-1]) * 0.5 * DV)
    return out


def uderivs(ia, u, r, s, f):
    """u-derivatives on the active slice [ia:]; integrals start at the moving axis v=u
    (the grid-aligned cumsum missed the axis sliver -> 1/r oscillation -> the NaN blow-up)."""
    off = V[ia] - u                                              # axis offset in (0, DV]
    rv = np.gradient(r[ia:], DV)
    fv = np.gradient(f[ia:], DV)
    sv = np.gradient(s[ia:], DV)
    a2 = np.exp(2 * np.clip(s[ia:], -60, 60))
    rr_u = -cumtrap_off(a2, off) / 4.0
    rsafe = np.maximum(r[ia:], 1e-12)
    ru = rr_u / rsafe
    g = -ru * fv
    fu = cumtrap_off(g, off) / rsafe
    integ = (ru * rv + a2 / 4.0) / (rsafe * rsafe) - 4 * math.pi * fu * fv
    integ[0] = integ[2]; integ[1] = integ[2]                     # tame the 0/0 axis limit
    su = -sv[0] + cumtrap_off(integ, off)
    return ru, su, fu, rv


def evolve(p, want_trace=False):
    r, s, f = initial(p)
    u, du = 0.0, 0.45 * DV
    maxR, trace = 0.0, []
    while u < UMAX:
        ia = min(int(u / DV) + 1, NV - 8)
        ru1, su1, fu1, rv = uderivs(ia, u, r, s, f)
        # horizon check: r,v -> 0 away from the axis edge
        j = np.argmin(rv[4:]) + 4
        if rv[j] <= 1e-3 * 0.5:
            a2 = math.exp(2 * s[ia + j])
            m = (r[ia + j] / 2) * (1 + 4 * ru1[j] * rv[j] / a2)
            return "bh", float(max(m, r[ia + j] / 2 * 0.5)), trace
        # curvature proxy near axis
        fv = np.gradient(f[ia:], DV)
        a2v = np.exp(2 * s[ia:])
        Rp = 16 * math.pi * np.abs(fu1 * fv) / a2v
        maxR = max(maxR, float(Rp[3:40].max()))
        if want_trace:
            trace.append((u, float(f[ia + 4])))
        # Heun step
        r2 = r.copy(); s2 = s.copy(); f2 = f.copy()
        r2[ia:] += du * ru1; s2[ia:] += du * su1; f2[ia:] += du * fu1
        ru2, su2, fu2, _ = uderivs(ia, u, r2, s2, f2)
        r[ia:] += du / 2 * (ru1 + ru2); s[ia:] += du / 2 * (su1 + su2); f[ia:] += du / 2 * (fu1 + fu2)
        r[ia:] = np.maximum(r[ia:], 0.0)
        u += du
        if not np.isfinite(r[ia:]).all():
            return "nan", maxR, trace
    return "disp", maxR, trace


def bisect(lo, hi, iters=46):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        out, _, _ = evolve(mid)
        if out == "bh":
            hi = mid
        else:
            lo = mid
        if (hi - lo) / hi < 3e-14:
            break
    return lo, hi


def main():
    global NV, DV, V
    if SMOKE:
        for p in (1e-6, 0.01, 0.05, 0.15):
            out, val, _ = evolve(p)
            print(f"SMOKE p={p}: {out}  value={val:.4e}")
        return
    print(f"[C1b] Garfinkle double-null; NV={NV}; prereg cosmos/PREREG_C1_choptuik.md (escalation b)")
    lo, hi = bisect(0.005, 0.15)
    pstar = 0.5 * (lo + hi)
    print(f"  p* = {pstar:.14f}  (bracket {(hi-lo)/hi:.2e})", flush=True)

    print("\n  Route A: supercritical M(p):")
    epsA, MA = [], []
    for k in range(2, 24):
        eps = 10 ** (-k / 2.0)
        out, M, _ = evolve(pstar * (1 + eps))
        if out == "bh" and M > 0:
            epsA.append(eps); MA.append(M)
        print(f"    eps={eps:.3e}: {out}  M={M:.5e}", flush=True)
    print("\n  Route B: subcritical Rmax(p):")
    epsB, RB = [], []
    for k in range(2, 24):
        eps = 10 ** (-k / 2.0)
        out, Rm, _ = evolve(pstar * (1 - eps))
        if out == "disp" and Rm > 0:
            epsB.append(eps); RB.append(Rm)
        print(f"    eps={eps:.3e}: {out}  Rmax={Rm:.5e}", flush=True)

    def fitwin(lx, ly, tol):
        sl = np.diff(ly) / np.diff(lx)
        med = float(np.median(sl))
        keep = [i for i in range(len(sl)) if abs(sl[i] - med) < tol]
        i0, i1 = min(keep), max(keep) + 1
        g, b = np.polyfit(lx[i0:i1 + 1], ly[i0:i1 + 1], 1)
        r2 = 1 - np.var(ly[i0:i1 + 1] - (g * lx[i0:i1 + 1] + b)) / np.var(ly[i0:i1 + 1])
        return g, r2, (lx[i1] - lx[i0]) / math.log(10)
    gA, r2A, decA = fitwin(np.log(epsA), np.log(MA), 0.05)
    sB, r2B, decB = fitwin(np.log(epsB), np.log(RB), 0.1)
    gB = -sB / 2
    print(f"\n  gamma_A = {gA:.4f} (R^2={r2A:.4f}, {decA:.1f} decades)")
    print(f"  gamma_B = {gB:.4f} (R^2={r2B:.4f}, {decB:.1f} decades)")

    # convergence control at NV/2
    NV2 = NV // 2
    NV, DV_old = NV2, DV
    DV = VMAX / NV; V = (np.arange(NV) + 0.5) * DV
    _, Mc, _ = evolve(pstar * 1.001)
    NV = NV2 * 2; DV = DV_old; V = (np.arange(NV) + 0.5) * DV
    _, Mf, _ = evolve(pstar * 1.001)
    print(f"  convergence: M(NV)={Mf:.5e} vs M(NV/2)={Mc:.5e}  rel {abs(Mf-Mc)/abs(Mf):.2e}")

    v1 = r2A > 0.99 and decA >= 2
    v2 = abs(gA - gB) <= 0.03
    v3 = abs(gA - 0.374) <= 0.02 and abs(gB - 0.374) <= 0.02
    print(f"\n  PRE-REGISTERED VERDICTS: (1) {'PASS' if v1 else 'FAIL'}  (2) {'PASS' if v2 else 'FAIL'}"
          f" |dg|={abs(gA-gB):.4f}  (3) {'PASS' if v3 else 'FAIL'} (anchor 0.374+-0.02)")
    json.dump({"pstar": pstar, "gamma_A": float(gA), "r2A": float(r2A), "decA": decA,
               "gamma_B": float(gB), "r2B": float(r2B), "decB": decB,
               "epsA": epsA, "MA": MA, "epsB": epsB, "RB": RB, "conv": [float(Mf), float(Mc)]},
              open(__file__.replace("gate_C1b_garfinkle.py", "gate_C1b_results.json"), "w"), indent=1)
    print("  wrote cosmos/gate_C1b_results.json")


if __name__ == "__main__":
    main()
