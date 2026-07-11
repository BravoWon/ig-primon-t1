#!/usr/bin/env python
"""Gate C1h -- the mp-float bisection. Per cosmos/PREREG_C1h_mpfloat.md (registered first).
Physics = the C1c/C1g scheme re-expressed in vectorized double-double (dd_kernel, ~31 digits):
state (r, h) as (hi, lo) pairs; integrals via DD prefix sums; g via DD exp; 4-point axis fit in
closed form. Detection (MOTS/r-floor, freeze, budget, crossings, du) reads f64 hi parts --
thresholds are coarse; only the state-update chain needs digits. u accumulates in DD (1M f64
adds would smear deep crossing times by ~1e-10). Configuration and every termination guard are
carried whole from the sealed C1g endpoint. Events/fit reuse C1g's A5c extraction and C1f's
fit_timing2 unchanged.

    python cosmos/gate_C1h_mpfloat.py --bench      (step-cost benchmark, ~200 steps)
    python cosmos/gate_C1h_mpfloat.py --calib      (calibration (b): continuity at eps=1e-9)
    python cosmos/gate_C1h_mpfloat.py              (full protocol)
"""
import sys, math, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dd_kernel as D
import gate_C1c_garfinkle as G
from gate_C1g_section3 import events_inbound
from gate_C1f_echoes import fit_timing2

VOUT = 4.134212319612277                                         # C1g sealed final config
PSTAR_F64 = 0.012793009036826557
N0, CFL = 800, 0.1
ANCH_D, PBAR, GA = 3.4453, 4.29, 0.3681
PI = (3.141592653589793, 1.2246467991473532e-16)
BENCH, CALIB = "--bench" in sys.argv, "--calib" in sys.argv


def _set(x, idx, val):                                           # functional slice-set on a DD pair
    x[0][idx], x[1][idx] = val[0], val[1]


def bars_dd(r, h):
    x, y = (r[0][:4], r[1][:4]), (h[0][:4], h[1][:4])
    xm = D.scale(D.dd(np.array(x[0].sum() + x[1].sum())), 0.25)  # cheap 4-elt means (exact adds
    ym = D.scale(D.dd(np.array(y[0].sum() + y[1].sum())), 0.25)  # in f64 would lose lo; keep DD)
    xm = (np.full(4, xm[0]), np.full(4, xm[1]))
    ym = (np.full(4, ym[0]), np.full(4, ym[1]))
    dx, dy = D.sub(x, xm), D.sub(y, ym)
    num = D.mul(dx, dy); num = (np.array(num[0].sum()), np.array(num[1].sum()))
    den = D.mul(dx, dx); den = (np.array(den[0].sum()), np.array(den[1].sum()))
    h1 = D.div(num, den)
    h0 = D.sub((ym[0][0], ym[1][0]), D.mul(h1, (xm[0][0], xm[1][0])))
    dr = D.diff(r)

    def cumtrap(f, f0):                                          # Ih-style integral, DD prefix
        avg = D.scale(D.add((f[0][1:], f[1][1:]), (f[0][:-1], f[1][:-1])), 0.5)
        inc = D.mul(avg, dr)
        ps = D.dd_sum_prefix(inc)
        out = (np.empty_like(f[0]), np.empty_like(f[1]))
        out[0][0], out[1][0] = f0[0], f0[1]
        tail = D.add(ps, (np.full_like(ps[0], f0[0]), np.full_like(ps[1], f0[1])))
        out[0][1:], out[1][1:] = tail[0], tail[1]
        return out

    r0 = (r[0][0], r[1][0])
    Ih0 = D.add(D.mul(h0, r0), D.scale(D.mul(D.mul(h1, r0), r0), 0.5))
    Ih = cumtrap(h, Ih0)
    hbar = D.div(Ih, r)
    tay = D.add((np.full(3, h0[0]), np.full(3, h0[1])),
                D.scale(D.mul((np.full(3, h1[0]), np.full(3, h1[1])), (r[0][:3], r[1][:3])), 0.5))
    _set(hbar, slice(0, 3), tay)
    dhh = D.sub(h, hbar)
    q = D.div(D.mul(dhh, dhh), r)
    h1sq = D.mul(h1, h1)
    qt = D.scale(D.mul((np.full(3, h1sq[0]), np.full(3, h1sq[1])), (r[0][:3], r[1][:3])), 0.25)
    _set(q, slice(0, 3), qt)
    Iq0 = D.scale(D.mul(D.mul(h1sq, r0), r0), 0.125)
    Iq = cumtrap(q, Iq0)
    garg = D.mul((np.full_like(r[0], 4 * PI[0]), np.full_like(r[0], 4 * PI[1])), Iq)
    over = garg[0] > 600.0                                       # physical clip (C1c)
    garg[0][over], garg[1][over] = 600.0, 0.0
    g = D.exp(garg)
    pih = D.scale(D.mul((np.full(3, PI[0]), np.full(3, PI[1])),
                        D.mul((np.full(3, h1sq[0]), np.full(3, h1sq[1])), (r[0][:3], r[1][:3]))), 0.5)
    _set(g, slice(0, 3), D.add(D.dd(np.full(3, 1.0)), pih))
    Ig0 = D.mul(r0, D.add(D.dd(np.array(1.0)),
                          D.scale(D.mul(D.mul((PI[0], PI[1]), h1sq), r0), 0.125)))
    Ig = cumtrap(g, Ig0)
    gbar = D.div(Ig, r)
    _set(gbar, slice(0, 3), D.add(D.dd(np.full(3, 1.0)), D.scale(pih, 0.5)))
    return h0, h1, hbar, g, gbar


def rates_dd(r, h):
    h0, h1, hbar, g, gbar = bars_dd(r, h)
    hdot = D.div(D.mul(D.sub(g, gbar), D.sub(h, hbar)), D.scale(r, 2.0))
    rdot = D.scale(gbar, -0.5)
    return hdot, rdot, g, gbar, h0, h1


def init_dd(p, vout=VOUT, n0=N0):
    rh = np.linspace(vout / n0, vout, n0)
    r = D.dd(rh)
    z = D.scale(D.sub(r, D.dd(np.full(n0, G.R0))), 1.0 / G.SIG)
    e = D.exp(D.neg(D.mul(z, z)))
    r2 = D.mul(r, r)
    poly = D.sub(D.scale(r2, 3.0), D.scale(D.mul(D.mul(r2, r), z), 2.0 / G.SIG))
    h = D.mul((np.full(n0, p[0]), np.full(n0, p[1])), D.mul(e, poly))
    return r, h


LAST = {}                                                        # termination forensics (diag aid)


def evolve_dd(p, vout=VOUT, cfl=CFL, n0=N0, collect=False):
    """C1g's evolve with every guard, state in DD. p is a DD scalar tuple."""
    r, h = init_dd(p, vout, n0)
    u = (0.0, 0.0)
    m_init, trace = None, []
    u_lc, s_prev, quiet, steps, mots_min = None, 0.0, 0, 0, 1.0
    last_flip = -10
    mots_trend_old, mots_trend_ref = 1.0, 1.0

    def _fin(cause, out):
        LAST.update(cause=cause, out=out, steps=steps, mots_min=mots_min, u=u[0],
                    n=len(r[0]), u_lc=u_lc)
        return out

    while u[0] < G.UEND and len(r[0]) > 8:
        steps += 1
        if steps > 2_000_000:
            return _fin("budget", "bh" if mots_min < 2 * G.MOTS_THRESH else "disp"), trace
        if not np.isfinite(h[0]).all():
            return _fin("overflow", "bh"), trace
        hd, rd, g, gbar, h0, h1 = rates_dd(r, h)
        mots = gbar[0] / g[0]                                    # detection: f64 hi parts
        j = int(np.argmin(mots))
        mj = float(mots[j])
        if mj > 0:                                               # amendment C1h-2: negative mots =
            mots_min = min(mots_min, mj)                         # ray-crossing pathology, never a
        if 0 < mj < G.MOTS_THRESH and r[0][j] > 5e-4:            # horizon (forensics: mots_min=-inf
            return _fin("mots", "bh"), trace                     # mislabeled a subcritical hover)
        h1f = float(h1[0])
        if u[0] > 1.0:
            s = math.copysign(1.0, h1f) if h1f != 0 else 0.0
            if s_prev != 0.0 and s != 0.0 and s != s_prev and steps - last_flip >= 5:
                u_lc, quiet = u[0], 0                            # C1h-2 guard-layer flicker rule:
            else:                                                # sub-resolution flips (every 1-2
                quiet += 1                                       # steps) must not reset the quiet
            if s_prev != 0.0 and s != 0.0 and s != s_prev:       # counter, else the freeze never
                last_flip = steps                                # engages and the budget burns
            s_prev = s
        if steps % 5000 == 0:                                    # mots trend sampler (amendment 3)
            mots_trend_old, mots_trend_ref = mots_trend_ref, mj
        m_out = 0.5 * r[0][-1] * (1 - mots[-1])
        if m_init is None:
            m_init = max(m_out, 1e-30)
        elif u_lc is None and m_out < 1e-3 * m_init:
            return _fin("m_out", "disp"), trace
        elif u_lc is not None and u[0] > u_lc + 1.5:
            return _fin("disp-clock", "disp"), trace
        du = cfl * float(np.quantile(np.diff(r[0]), 0.05))
        r2 = D.add(r, D.scale(rd, du))
        h2 = D.add(h, D.scale(hd, du))
        keep2 = r2[0] > 1e-9
        if keep2.sum() < 8:
            break
        hd2 = (hd[0].copy(), hd[1].copy()); rd2 = (rd[0].copy(), rd[1].copy())
        k2 = np.where(keep2)[0]
        hdk, rdk, *_ = rates_dd((r2[0][k2], r2[1][k2]), (h2[0][k2], h2[1][k2]))
        hd2[0][k2], hd2[1][k2] = hdk[0], hdk[1]
        rd2[0][k2], rd2[1][k2] = rdk[0], rdk[1]
        r = D.add(r, D.scale(D.add(rd, rd2), 0.5 * du))
        h = D.add(h, D.scale(D.add(hd, hd2), 0.5 * du))
        keep = r[0] > 1e-9
        r = (r[0][keep], r[1][keep]); h = (h[0][keep], h[1][keep])
        u = D.add(u, D.dd(np.array(du)))
        u = (float(u[0]), float(u[1]))
        if collect and len(r[0]) > 8:
            trace.append((u[0], float(bars_dd(r, h)[1][0]), float(r[0][-1])))
        # amendment 3 (the quiet-collapse robbery): post-departure the unstable mode grows
        # MONOTONICALLY -- no crossings, quiet accumulates, and the freeze would drain the grid
        # before the horizon resolves, mislabeling bh-side probes as disp (the entire first
        # descent converged above the true threshold this way; eps=1e-14 verdict run came back
        # bh). A collapsing run's mots trends DOWN: a downtrend now vetoes the freeze; the 2M
        # budget still catches any slow hover and classifies by its (positive-only) mots_min.
        frozen = (quiet > 20000 and mj > 1.05 * G.MOTS_THRESH and mj >= mots_trend_old - 1e-4)
        if not frozen and 8 < len(r[0]) < n0 // 2:
            rm = D.scale(D.add((r[0][1:], r[1][1:]), (r[0][:-1], r[1][:-1])), 0.5)
            hm = D.scale(D.add((h[0][1:], h[1][1:]), (h[0][:-1], h[1][:-1])), 0.5)
            n = len(r[0])
            rn = (np.empty(2 * n - 1), np.empty(2 * n - 1)); hn = (np.empty(2 * n - 1), np.empty(2 * n - 1))
            rn[0][0::2], rn[1][0::2] = r[0], r[1]; rn[0][1::2], rn[1][1::2] = rm[0], rm[1]
            hn[0][0::2], hn[1][0::2] = h[0], h[1]; hn[0][1::2], hn[1][1::2] = hm[0], hm[1]
            r, h = rn, hn
    return _fin("drain/UEND", "disp"), trace


def _probe_dd(args):
    ph, pl, cfl = args
    try:
        return evolve_dd((ph, pl), cfl=cfl)[0]
    except Exception:
        return "bh"


def trisect_dd(lo, hi, cfl, pool, target=1e-22, tag=""):
    """DD trisection: lo/hi are DD scalars with verified labels (lo disp, hi bh)."""
    lvl = 0
    while float(D.sub(hi, lo)[0]) / float(hi[0]) >= target and lvl < 26:
        span = D.sub(hi, lo)
        ms = [D.add(lo, D.scale(span, k / 6.0)) for k in (1, 2, 3, 4, 5)]
        outs = pool.map(_probe_dd, [(float(m[0]), float(m[1]), cfl) for m in ms])
        newlo, newhi = lo, hi
        for m, o in zip(ms, outs):
            if o == "bh":
                newhi = m
                break
            newlo = m
        lo, hi = newlo, newhi
        lvl += 1
        print(f"      dd-trisect[{lvl:02d}] rel={float(D.sub(hi, lo)[0]) / float(hi[0]):.2e}"
              f"  p_lo={float(lo[0]):.17g}+{float(lo[1]):.3g}", flush=True)
    return D.scale(D.add(lo, hi), 0.5)


def deep_dd(pstar, eps, cfl=CFL):
    p = D.mul(pstar, D.sub(D.dd(np.array(1.0)), D.dd(np.array(eps))))
    out, tr = evolve_dd((float(p[0]), float(p[1])), cfl=cfl, collect=True)
    arr = float(tr[-1][0]) if len(tr) else float("nan")
    if out != "disp":
        return None, out, 0, float("nan"), arr
    got = events_inbound(tr)
    if got is None or got[0] is None:
        return None, "few-events", 0 if got is None else -1, float("nan"), arr
    (F0, F1), ut, r_ut = got
    Dl, us, sse, D0, D1 = fit_timing2(F0, F1)
    return {"Delta": float(Dl), "n": [len(F0), len(F1)], "fam": [float(D0), float(D1)],
            "ustar": float(us), "F0": [float(x) for x in F0], "F1": [float(x) for x in F1]}, \
        "ok", len(F0) + len(F1), r_ut, arr


def bench():
    import time
    p = D.dd(np.array(PSTAR_F64 * 0.999))
    r, h = init_dd((float(p[0]), float(p[1])))
    t = time.time()
    for _ in range(200):
        rates_dd(r, h)
    dt = (time.time() - t) / 200
    print(f"[C1h bench] rates_dd: {dt * 1e3:.2f} ms/call  -> ~{2 * dt * 1e3:.1f} ms/step"
          f"  -> ~{2 * dt * 1e6 / 3600:.2f} h per 1M steps")


def calib():
    """Calibration (b): continuity vs f64 at eps=1e-9 (both fogs far below)."""
    import gate_C1g_section3 as GG
    eps = 1e-9
    print("[C1h calib] f64 side:", flush=True)
    outf, trf = GG.evolve_g(PSTAR_F64 * (1 - eps), VOUT, CFL, collect=True)
    gotf = events_inbound(trf)
    ucf = gotf[0][0] if gotf and gotf[0] else []
    print(f"  f64: {outf}, crossings {[round(float(x), 6) for x in ucf]}", flush=True)
    print("[C1h calib] DD side (slow):", flush=True)
    ps = D.dd(np.array(PSTAR_F64))
    fit, status, nev, r_ut, arr = deep_dd(ps, eps)
    ucd = fit["F0"] if fit else []
    print(f"  DD : {status}, crossings {[round(float(x), 6) for x in ucd]}", flush=True)
    if fit and gotf and gotf[0] is not None and len(ucf) and len(ucd):
        n = min(len(ucf), len(ucd))
        dev = max(abs(float(a) - float(b)) for a, b in zip(ucf[:n], ucd[:n]))
        print(f"  max |du| over {n} shared crossings = {dev:.2e}  "
              f"({'PASS' if dev < 1e-6 else 'FAIL'} at 1e-6)")


def main():
    if BENCH:
        bench(); return
    if CALIB:
        calib(); return
    import multiprocessing
    pool = multiprocessing.Pool(5)
    print("[C1h] mp-float bisection; prereg cosmos/PREREG_C1h_mpfloat.md")
    # seeded bracket, labels verified at DD (widen x10 to 1e-8, else nm); starts at 1e-9 --
    # the DD threshold displacement is MEASURED at (1e-10, 1e-9) above the f64 seed, so
    # narrower windows are known-inverted (banked receipts, first campaign launch)
    w = 1e-9
    lo = hi = None
    while w <= 1e-8:
        cl = D.mul(D.dd(np.array(PSTAR_F64)), D.dd(np.array(1.0 - w)))
        ch = D.mul(D.dd(np.array(PSTAR_F64)), D.dd(np.array(1.0 + w)))
        ol = _probe_dd((float(cl[0]), float(cl[1]), CFL))
        oh = _probe_dd((float(ch[0]), float(ch[1]), CFL))
        print(f"  bracket +-{w:.0e}: lo={ol} hi={oh}", flush=True)
        if ol == "disp" and oh == "bh":
            lo, hi = cl, ch
            break
        w *= 10
    if lo is None:
        print("  bracket verification nm -- DD threshold moved beyond 1e-8"); return
    ps = trisect_dd(lo, hi, CFL, pool)
    print(f"  p*(DD) = {float(ps[0]):.17g} + {float(ps[1]):.3g}", flush=True)

    print("\n  verdict runs, depth axis eps = 1e-14 / 1e-18 / 1e-22:", flush=True)
    Ds, diag = [], []
    for eps in (1e-14, 1e-18, 1e-22):
        fit, status, nev, r_ut, arr = deep_dd(ps, eps)
        if fit:
            Ds.append(fit["Delta"]); diag.append({"eps": eps, **fit})
            print(f"    eps={eps:.0e}: events {fit['n'][0]}+{fit['n'][1]}  Delta={fit['Delta']:.3f}"
                  f"  fams {fit['fam'][0]:.2f}/{fit['fam'][1]:.2f}", flush=True)
        else:
            print(f"    eps={eps:.0e}: {status} -> nm", flush=True)
    Dmed = float(np.median(Ds)) if Ds else float("nan")
    nev_22 = next((d["n"][0] + d["n"][1] for d in diag if d["eps"] == 1e-22), 0)

    print("\n  sampling control CFL=0.05 (own DD trisection, backoff enabled):", flush=True)
    w, lo05, hi05 = 1e-5, None, None
    while w <= 1e-3:
        cl = D.mul(ps, D.dd(np.array(1.0 - w)))
        ch = D.mul(ps, D.dd(np.array(1.0 + w)))
        if _probe_dd((float(cl[0]), float(cl[1]), 0.05)) == "disp" and \
           _probe_dd((float(ch[0]), float(ch[1]), 0.05)) == "bh":
            lo05, hi05 = cl, ch
            break
        w *= 10
    D05 = float("nan")
    if lo05 is not None:
        ps05 = trisect_dd(lo05, hi05, 0.05, pool)
        for eps in (1e-22, 1e-20, 1e-18):                        # backoff on the control too
            fit05, status05, *_ = deep_dd(ps05, eps, cfl=0.05)
            if fit05:
                D05 = fit05["Delta"]
                print(f"    CFL=0.05 eps={eps:.0e}: Delta={D05:.3f}", flush=True)
                break
            print(f"    (control eps={eps:.0e}: {status05} -> backoff)", flush=True)

    dspread = max(Ds) - min(Ds) if len(Ds) >= 2 else float("nan")
    gB = Dmed / (2 * PBAR)
    v1 = nev_22 >= 16
    v2 = len(Ds) >= 2 and abs(Dmed - ANCH_D) <= 0.25
    v3 = len(Ds) >= 2 and dspread <= 0.20 and (not math.isnan(D05)) and abs(D05 - Dmed) <= 0.20
    v4 = v2 and v3 and abs(gB - 0.374) <= 0.03 and abs(gB - GA) <= 0.03
    print(f"\n  PRE-REGISTERED VERDICTS:")
    print(f"    (1) >=16 events at eps=1e-22: {'PASS' if v1 else 'FAIL'} ({nev_22})")
    print(f"    (2) Delta={Dmed:.3f} in 3.4453+-0.25: {'PASS' if v2 else 'FAIL'} "
          f"(off {abs(Dmed - ANCH_D):.3f})")
    print(f"    (3) depth spread {dspread:.3f}<=0.20 and sampling |dD|="
          f"{abs(D05 - Dmed):.3f}<=0.20: {'PASS' if v3 else 'FAIL'}")
    print(f"    (4) implied gamma_B={gB:.4f}: {'PASS' if v4 else 'FAIL'}")
    if v2 and not v3:
        print("    NOTE: verdict 2 VOIDED by verdict-3 failure (registered rule)")
    json.dump({"pstar_dd": [float(ps[0]), float(ps[1])], "runs": diag, "Delta": Dmed,
               "spread": dspread, "D05": D05, "gamma_B_implied": float(gB),
               "verdicts": {"v1": bool(v1), "v2": bool(v2), "v3": bool(v3), "v4": bool(v4)}},
              open(os.path.join(HERE, "gate_C1h_results.json"), "w"), indent=1)
    print("  wrote cosmos/gate_C1h_results.json")


if __name__ == "__main__":
    main()
