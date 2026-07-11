#!/usr/bin/env python
"""Gate C1c -- Garfinkle's ACTUAL scheme (gr-qc/9412008, read from the paper this time).
Christodoulou null formulation: evolve h with Phi = hbar; ds^2 = -2 g r' du dv + r^2 dOmega^2.
  fbar(v) = (1/r) int_0^r f dr~        (all bars are cumulative r-integrals along the slice)
  q = (h-hbar)^2 / r ;  g = exp(4 pi int_0^r q dr~) ;  gbar = (1/r) int_0^r g dr~
  hdot = (g-gbar)(h-hbar)/(2r) ;  rdot = -gbar/2
Axis: fit h ~ h0 + h1 r on first 4 pts; Taylor forms for the first 3 pts (paper eqs. 10-14).
MOTS: grad r . grad r = gbar/g -> 0; M = r/2 there. Rays drop at the axis; regrid by midpoint
insertion when half are lost. Initial data (paper eq. 15): Phi = p r^2 exp(-((r-r0)/sigma)^2),
h = d(r Phi)/dr.  Route B observable: max over run of h1^2 (axis curvature ~ 8 pi h1^2-ish).

    python cosmos/gate_C1c_garfinkle.py [--smoke]
"""
import sys, math, json
import numpy as np

N0, VMAX, R0, SIG = 800, 6.0, 2.0, 0.5
CFL, MOTS_THRESH, UEND = 0.4, 0.02, 14.0
SMOKE = "--smoke" in sys.argv


def init(p, n0=N0):
    r = np.linspace(VMAX / n0, VMAX, n0)
    z = (r - R0) / SIG
    e = np.exp(-z * z)
    h = p * e * (3 * r * r - 2 * r ** 3 * z / SIG)              # h = d(r Phi)/dr
    return r, h


def bars(r, h):
    A = np.vstack([np.ones(4), r[:4]]).T
    (h0, h1), *_ = np.linalg.lstsq(A, h[:4], rcond=None)
    Ih = np.empty_like(r)
    Ih[0] = h0 * r[0] + 0.5 * h1 * r[0] ** 2
    Ih[1:] = Ih[0] + np.cumsum(0.5 * (h[1:] + h[:-1]) * np.diff(r))
    hbar = Ih / r
    hbar[:3] = h0 + 0.5 * h1 * r[:3]
    q = (h - hbar) ** 2 / r
    q[:3] = 0.25 * h1 * h1 * r[:3]
    Iq = np.empty_like(r)
    Iq[0] = 0.125 * h1 * h1 * r[0] ** 2
    Iq[1:] = Iq[0] + np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(r))
    g = np.exp(np.clip(4 * math.pi * Iq, None, 600.0))          # clip: g overflow past MOTS is unphysical anyway
    g[:3] = 1 + 0.5 * math.pi * h1 * h1 * r[:3]
    Ig = np.empty_like(r)
    Ig[0] = r[0] * (1 + 0.125 * math.pi * h1 * h1 * r[0])
    Ig[1:] = Ig[0] + np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(r))
    gbar = Ig / r
    gbar[:3] = 1 + 0.25 * math.pi * h1 * h1 * r[:3]
    return h0, h1, hbar, g, gbar


def rates(r, h):
    h0, h1, hbar, g, gbar = bars(r, h)
    hdot = (g - gbar) * (h - hbar) / (2 * r)
    return hdot, -0.5 * gbar, g, gbar, h0, h1


def evolve(p, n0=N0, collect=False):
    r, h = init(p, n0)
    u, maxh1sq, trace = 0.0, 0.0, []
    m_init = None
    while u < UEND and len(r) > 8:
        hd1, rd1, g, gbar, h0, h1 = rates(r, h)
        mots = gbar / g
        j = int(np.argmin(mots))
        if mots[j] < MOTS_THRESH:
            return "bh", float(r[j] / 2), trace
        m_out = 0.5 * r[-1] * (1 - mots[-1])                     # Misner-Sharp mass at the outermost ray
        if m_init is None:
            m_init = max(m_out, 1e-30)
        elif m_out < 1e-3 * m_init:
            return "disp", maxh1sq, trace                        # pulse bounced and radiated out: done
        maxh1sq = max(maxh1sq, float(h1 * h1))
        du = CFL * float(np.quantile(np.diff(r), 0.05))
        r2 = r + du * rd1
        h2 = h + du * hd1
        keep2 = r2 > 1e-9
        if keep2.sum() < 8:
            break
        hd2 = np.zeros_like(h); rd2 = np.zeros_like(r)
        hd2[keep2], rd2[keep2], *_ = rates(r2[keep2], h2[keep2])
        hd2[~keep2] = hd1[~keep2]; rd2[~keep2] = rd1[~keep2]
        r = r + 0.5 * du * (rd1 + rd2)
        h = h + 0.5 * du * (hd1 + hd2)
        keep = r > 1e-9
        r, h = r[keep], h[keep]
        u += du
        if collect and len(r) > 8:
            trace.append((u, float(bars(r, h)[1])))              # h1(u) for the echo stretch
        if 8 < len(r) < n0 // 2:                                 # paper's regrid: midpoint insertion
            rm = 0.5 * (r[1:] + r[:-1]); hm = 0.5 * (h[1:] + h[:-1])
            rn = np.empty(2 * len(r) - 1); hn = np.empty_like(rn)
            rn[0::2] = r; rn[1::2] = rm
            hn[0::2] = h; hn[1::2] = hm
            r, h = rn, hn
    return "disp", maxh1sq, trace


def bisect(lo, hi, iters=46, n0=N0):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        out, _, _ = evolve(mid, n0)
        if out == "bh":
            hi = mid
        else:
            lo = mid
        if (hi - lo) / hi < 3e-14:
            break
    return lo, hi


def fitwin(lx, ly, tol):
    sl = np.diff(ly) / np.diff(lx)
    med = float(np.median(sl))
    keep = [i for i in range(len(sl)) if abs(sl[i] - med) < tol]
    i0, i1 = min(keep), max(keep) + 1
    gfit, b = np.polyfit(lx[i0:i1 + 1], ly[i0:i1 + 1], 1)
    r2 = 1 - np.var(ly[i0:i1 + 1] - (gfit * lx[i0:i1 + 1] + b)) / np.var(ly[i0:i1 + 1])
    return gfit, r2, (lx[i1] - lx[i0]) / math.log(10)


def main():
    if SMOKE:
        for p in (0.001, 0.02, 0.05, 0.1):
            out, v, _ = evolve(p)
            print(f"SMOKE p={p}: {out}  value={v:.4e}")
        return
    print(f"[C1c] Garfinkle scheme (from the paper); N={N0}; prereg cosmos/PREREG_C1_choptuik.md")
    lo, hi = bisect(0.001, 0.02)                                 # smoke: 0.001 disp / 0.02 bh
    pstar = 0.5 * (lo + hi)
    print(f"  p* = {pstar:.14f}  (bracket {(hi-lo)/hi:.2e})", flush=True)

    print("\n  Route A: supercritical M(p):")
    epsA, MA = [], []
    for k in range(2, 26):
        eps = 10 ** (-k / 2.0)
        out, M, _ = evolve(pstar * (1 + eps))
        if out == "bh" and M > 0:
            epsA.append(eps); MA.append(M)
        print(f"    eps={eps:.3e}: {out}  M={M:.5e}", flush=True)
    gA, r2A, decA = fitwin(np.log(epsA), np.log(MA), 0.05)
    print(f"  gamma_A = {gA:.4f} (R^2={r2A:.4f}, {decA:.1f} decades)")

    print("\n  Route B: subcritical max h1^2 (axis curvature):")
    epsB, RB = [], []
    for k in range(2, 26):
        eps = 10 ** (-k / 2.0)
        out, v, _ = evolve(pstar * (1 - eps))
        if out == "disp" and v > 0:
            epsB.append(eps); RB.append(v)
        print(f"    eps={eps:.3e}: {out}  maxh1^2={v:.5e}", flush=True)
    sB, r2B, decB = fitwin(np.log(epsB), np.log(RB), 0.1)
    gB = -sB / 2
    print(f"  gamma_B = {gB:.4f} (slope {sB:.3f}, R^2={r2B:.4f}, {decB:.1f} decades)")

    print("\n  convergence control (N vs N/2, eps=1e-3):")
    _, M1, _ = evolve(pstar * 1.001, N0)
    _, M2, _ = evolve(pstar * 1.001, N0 // 2)
    print(f"    M(N)={M1:.5e}  M(N/2)={M2:.5e}  rel {abs(M1-M2)/max(abs(M1),1e-30):.2e}")

    # echo stretch: h1(u) trace of the deepest subcritical run
    _, _, tr = evolve(pstar * (1 - 1e-12), collect=True)
    v1 = r2A > 0.99 and decA >= 2
    v2 = abs(gA - gB) <= 0.03
    v3 = abs(gA - 0.374) <= 0.02 and abs(gB - 0.374) <= 0.02
    print(f"\n  PRE-REGISTERED VERDICTS: (1) {'PASS' if v1 else 'FAIL'}  (2) {'PASS' if v2 else 'FAIL'}"
          f" |dg|={abs(gA-gB):.4f}  (3) {'PASS' if v3 else 'FAIL'} (Gundlach 0.374+-0.02)")
    json.dump({"pstar": pstar, "gamma_A": float(gA), "r2A": float(r2A), "decA": decA,
               "gamma_B": float(gB), "r2B": float(r2B), "decB": decB,
               "epsA": epsA, "MA": MA, "epsB": epsB, "RB": RB,
               "conv": [float(M1), float(M2)], "echo_trace": tr[-4000:]},
              open(__file__.replace("gate_C1c_garfinkle.py", "gate_C1c_results.json"), "w"), indent=1)
    print("  wrote cosmos/gate_C1c_results.json")


if __name__ == "__main__":
    main()
