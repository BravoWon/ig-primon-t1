#!/usr/bin/env python
"""Gate C1e -- the TRUE Ricci scalar, from the paper's equations (gr-qc/9412008 eqs 1,4,7,8,9).
R = 8 pi grad(Phi).grad(Phi) with Phi = hbar; metric ds^2 = -2 g r' du dv + r^2 dOmega^2 (eq 7)
  => R = -16 pi Phidot (h - hbar) / (g r),   Phi' = (h - hbar) r'/r  (from eq 4),
     Phidot = [ gbar hbar - h0 - int_0^r (g - gbar) hbar dr~/r~ ] / (2r)   (d/du of eq 4 at fixed
     v, using eqs 8,9, gbar' = (g-gbar) r'/r, and the moving-axis boundary term h0/2).
Axis limits: Phidot -> h1/4, (h-hbar)/r -> h1/2, g -> 1  =>  R_axis = -2 pi h1^2 -- i.e. the
C1c/C1d "proxy" IS Garfinkle-Duncan's central-observer curvature up to the constant. C1e's new
content: max |R| over the WHOLE SLICE (no 4-point axis fit involved), which stays resolved at the
deep echoes where the axis Taylor fit smears. Both observables tracked per run.

    python cosmos/gate_C1e_ricci.py --probe     (instrument characterization, 2 eps values)
    python cosmos/gate_C1e_ricci.py --smoke     (anchors: p^2 linearity + axis consistency)
    python cosmos/gate_C1e_ricci.py             (full gate per PREREG_C1e_ricci.md)
"""
import sys, math, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_C1c_garfinkle as G

PROBE = "--probe" in sys.argv
SMOKE = "--smoke" in sys.argv


def ricci(r, h):
    """|R| over the slice + axis-channel value, from the paper's equations (docstring)."""
    h0, h1, hbar, g, gbar = G.bars(r, h)
    integ = (g - gbar) * hbar / r
    integ[:3] = 0.25 * math.pi * h1 * h1 * h0                    # Taylor limit of (g-gbar)/r * hbar
    I3 = np.empty_like(r)
    I3[0] = integ[0] * r[0]
    I3[1:] = I3[0] + np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(r))
    Phid = (gbar * hbar - h0 - I3) / (2 * r)
    Phid[:3] = 0.25 * h1
    Rsc = -16 * math.pi * Phid * (h - hbar) / (g * r)
    Rsc[:3] = -2 * math.pi * h1 * h1
    return np.abs(Rsc), 2 * math.pi * h1 * h1


def evolve_R(p, n0=G.N0):
    """G.evolve with |R| tracking. Returns (outcome, maxR_slice, maxR_axis, argmax_r_over_run)."""
    r, h = G.init(p, n0)
    u, m_init = 0.0, None
    mxS, mxA, arg_r = 0.0, 0.0, 0.0
    while u < G.UEND and len(r) > 8:
        hd1, rd1, g, gbar, h0, h1 = G.rates(r, h)
        mots = gbar / g
        j = int(np.argmin(mots))
        if mots[j] < G.MOTS_THRESH:
            return "bh", mxS, mxA, arg_r
        m_out = 0.5 * r[-1] * (1 - mots[-1])
        if m_init is None:
            m_init = max(m_out, 1e-30)
        elif m_out < 1e-3 * m_init:
            return "disp", mxS, mxA, arg_r
        Rabs, Rax = ricci(r, h)
        k = int(np.argmax(Rabs))
        if Rabs[k] > mxS:
            mxS, arg_r = float(Rabs[k]), float(r[k])
        mxA = max(mxA, float(Rax))
        du = G.CFL * float(np.quantile(np.diff(r), 0.05))
        r2 = r + du * rd1
        h2 = h + du * hd1
        keep2 = r2 > 1e-9
        if keep2.sum() < 8:
            break
        hd2 = np.zeros_like(h); rd2 = np.zeros_like(r)
        hd2[keep2], rd2[keep2], *_ = G.rates(r2[keep2], h2[keep2])
        hd2[~keep2] = hd1[~keep2]; rd2[~keep2] = rd1[~keep2]
        r = r + 0.5 * du * (rd1 + rd2)
        h = h + 0.5 * du * (hd1 + hd2)
        keep = r > 1e-9
        r, h = r[keep], h[keep]
        u += du
        if 8 < len(r) < n0 // 2:
            rm = 0.5 * (r[1:] + r[:-1]); hm = 0.5 * (h[1:] + h[:-1])
            rn = np.empty(2 * len(r) - 1); hn = np.empty_like(rn)
            rn[0::2] = r; rn[1::2] = rm
            hn[0::2] = h; hn[1::2] = hm
            r, h = rn, hn
    return "disp", mxS, mxA, arg_r


def probe():
    res = json.load(open(os.path.join(HERE, "gate_C1c_results.json")))
    ps = res["pstar"]
    print("[C1e probe] slice-max vs axis-max |R| -- axis under-resolution characterization")
    for e in (1e-3, 1e-6):
        out, mS, mA, ar = evolve_R(ps * (1 - e))
        print(f"  eps={e:.1e}: {out}  maxR_slice={mS:.5e}  maxR_axis={mA:.5e}  "
              f"ratio={mS/max(mA,1e-30):.3f}  argmax at r={ar:.4e}")


def smoke():
    res = json.load(open(os.path.join(HERE, "gate_C1c_results.json")))
    ps = res["pstar"]
    print("[C1e smoke] in-run anchors (prereg: must pass before launch)")
    # anchor 1: weak-field linearity R_max ~ p^2 (exact as p->0)
    vals = []
    for p in (1e-6, 1e-5, 1e-4):
        out, mS, mA, _ = evolve_R(p)
        vals.append(mS / p ** 2)
        print(f"  p={p:.0e}: {out}  maxR_slice/p^2 = {mS/p**2:.6e}")
    dev = (max(vals) - min(vals)) / max(vals)
    a1 = dev < 0.01
    print(f"  anchor 1 (p^2 linearity <1%): {'PASS' if a1 else 'FAIL'}  (spread {dev:.2e})")
    # anchor 2 (amended pre-launch, disclosed in prereg): axis consistency on a synthetic
    # AXIS-ACTIVE slice h = c r exp(-r^2/2) (h1 = c exactly; the initial-slice version compared
    # noise -- the pulse sits at r=2 and the near-axis field is ~2e-7 of peak). Exercises the
    # full Phidot/I3 path with real signal; R(r->0) must approach 2 pi c^2.
    c = 0.05
    r = np.linspace(G.VMAX / G.N0, G.VMAX, G.N0)
    h = c * r * np.exp(-0.5 * r * r)
    Rabs, Rax = ricci(r, h)
    rel = abs(Rabs[3] - 2 * math.pi * c * c) / (2 * math.pi * c * c)
    relax = abs(Rax - 2 * math.pi * c * c) / (2 * math.pi * c * c)
    a2 = rel < 0.05 and relax < 0.01
    print(f"  anchor 2 (synthetic axis-active slice, R at ray 3 vs 2 pi c^2): "
          f"{'PASS' if a2 else 'FAIL'}  (slice rel {rel:.2e}, axis-fit rel {relax:.2e})")
    print(f"  ANCHORS: {'PASS' if a1 and a2 else 'FAIL'}")


def main():
    if PROBE:
        probe(); return
    if SMOKE:
        smoke(); return
    import gate_C1d_wiggle as W                                  # the calibrated C1d pipeline
    res = json.load(open(os.path.join(HERE, "gate_C1c_results.json")))
    conv = json.load(open(os.path.join(HERE, "gate_C1c_conv1600.json")))
    ps, ps16, gA = res["pstar"], conv["pstar1600"], res["refit"]["gamma_A"]
    print(f"[C1e] true-Ricci gate; prereg cosmos/PREREG_C1e_ricci.md; p*={ps:.14f}")
    epsB, yS, yA_ax, dropped = [], [], [], []
    for k in range(16, 61):                                      # eighth-decade, registered range
        e = 10 ** (-k / 8.0)
        out, mS, mA, ar = evolve_R(ps * (1 - e))
        if out == "disp" and mS > 0:
            epsB.append(e); yS.append(mS); yA_ax.append(mA)
        else:
            dropped.append((e, out))
        print(f"  eps={e:.4e}: {out}  R_slice={mS:.5e}  R_axis={mA:.5e}  argr={ar:.3e}", flush=True)
    if dropped:
        print(f"  dropped (listed per prereg): {dropped}")
    print(f"\n  PRIMARY pipeline (slice-max |R|, n={len(epsB)}):", flush=True)
    rS = W.pipeline(epsB, yS, -0.5)
    print(f"    gamma_B={rS['gamma']:.4f}  P={rS['P']:.3f}  Delta_B={rS['Delta']:.3f}  "
          f"amp={rS['amp1']:.3f}  SSE x{rS['reduction']:.3f}  p={rS['pval']:.4f}  "
          f"dphi={rS['dphi']:.2f}  detect={rS['detect']}")
    print(f"  GD-literal channel (axis-max, diagnostic):", flush=True)
    rX = W.pipeline(epsB, yA_ax, -0.5, do_surr=False)
    print(f"    gamma_B_axis={rX['gamma']:.4f}  Delta_axis={rX['Delta']:.3f}")
    print(f"\n  V5 control: N=1600, quarter-decade k'=8..30:", flush=True)
    e16, y16 = [], []
    for kq in range(8, 31):
        e = 10 ** (-kq / 4.0)
        out, mS, mA, _ = evolve_R(ps16 * (1 - e), 1600)
        if out == "disp" and mS > 0:
            e16.append(e); y16.append(mS)
        print(f"    eps={e:.4e}: {out}  R_slice={mS:.5e}", flush=True)
    r16 = W.pipeline(e16, y16, -0.5, do_surr=False)
    print(f"    gamma_B(1600)={r16['gamma']:.4f}  Delta_B(1600)={r16['Delta']:.3f}"
          f"  [signed drift dg={r16['gamma']-rS['gamma']:+.4f}]")
    v1 = rS["detect"]
    v2 = v1 and abs(rS["gamma"] - W.ANCH_G) <= 0.02 and abs(gA - rS["gamma"]) <= 0.03
    v3 = v1 and abs(rS["Delta"] - W.ANCH_D) <= 0.25
    v5 = v1 and abs(r16["gamma"] - rS["gamma"]) <= 0.01 and abs(r16["Delta"] - rS["Delta"]) <= 0.3
    print(f"\n  PRE-REGISTERED VERDICTS:")
    print(f"    (1) detection: {'PASS' if v1 else 'FAIL'}")
    print(f"    (2) |gamma_B-0.374|<=0.02 AND |gamma_A(C1c)-gamma_B|<=0.03: {'PASS' if v2 else 'FAIL'}"
          f"  (gB off {abs(rS['gamma']-W.ANCH_G):.4f}, |dg|={abs(gA-rS['gamma']):.4f})")
    print(f"    (3) Delta_B within 3.4453 +- 0.25: {'PASS' if v3 else 'FAIL'} "
          f"(off {abs(rS['Delta']-W.ANCH_D):.3f})")
    print(f"    (5) N=1600 control (|dgamma|<=0.01 TIGHTENED, |dDelta|<=0.3): "
          f"{'PASS' if v5 else 'FAIL'}  (dg={abs(r16['gamma']-rS['gamma']):.4f}, "
          f"dD={abs(r16['Delta']-rS['Delta']):.3f})")
    json.dump({"pstar": ps, "primary_slice": rS, "axis_channel": rX, "control1600": r16,
               "epsB": epsB, "R_slice": yS, "R_axis": yA_ax, "eps16": e16, "y16": y16,
               "dropped": dropped, "gamma_A_used": gA,
               "verdicts": {"v1": bool(v1), "v2": bool(v2), "v3": bool(v3), "v5": bool(v5)}},
              open(os.path.join(HERE, "gate_C1e_results.json"), "w"), indent=1)
    print("  wrote cosmos/gate_C1e_results.json")


if __name__ == "__main__":
    main()
