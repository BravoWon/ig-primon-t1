#!/usr/bin/env python
"""Gate C1g -- Garfinkle Section III: the outermost gridpoint rides the marginal ray.
Per cosmos/PREREG_C1g_section3.md (registered first). Physics = gate_C1c_garfinkle bars/rates,
untouched. Instrument = truncated grid (0, v_out], v_out iterated onto the marginal ray per the
paper; p* re-bisected to the f64 floor per configuration; CFL=0.1 (C1f A2). Event extraction
INBOUND-ONLY (A5 re-registered); 2-family ladder fit imported from C1f (calibration banked there).

    python cosmos/gate_C1g_section3.py            (full protocol: iterate -> verdicts)
"""
import sys, math, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_C1c_garfinkle as G
from gate_C1f_echoes import fit_timing2, UTRIM

ANCH_D, PBAR, GA = 3.4453, 4.29, 0.3681


def init_trunc(p, vout, n0=800):
    r = np.linspace(vout / n0, vout, n0)
    z = (r - G.R0) / G.SIG
    h = p * np.exp(-z * z) * (3 * r * r - 2 * r ** 3 * z / G.SIG)
    return r, h


def evolve_g(p, vout, cfl=0.1, n0=800, collect=False):
    """C1c scheme on the truncated grid; trace = (u, h1, r_outer).
    Termination amended for truncation (disclosed in prereg): the C1c m_out early-exit fires
    DURING the echo phase here (bounce flux crosses the near-in outer boundary at u=4.857 <
    u* -- measured), clipping the ladder and mislabeling near-critical runs. Replacement:
    dispersal = no h1 sign change for 1.5 u-units after the last one (crossing-timing based,
    amplitude-free); the m_out exit survives only for runs with NO crossings (nothing to clip)."""
    r, h = init_trunc(p, vout, n0)
    u, m_init, trace = 0.0, None, []
    u_lc, s_prev, quiet = None, 0.0, 0
    while u < G.UEND and len(r) > 8:
        hd1, rd1, g, gbar, h0, h1 = G.rates(r, h)
        mots = gbar / g
        j = int(np.argmin(mots))
        if mots[j] < G.MOTS_THRESH:
            return "bh", trace
        if u > 1.0:                                              # same trim as the event rules
            s = math.copysign(1.0, h1) if h1 != 0 else 0.0
            if s_prev != 0.0 and s != 0.0 and s != s_prev:
                u_lc, quiet = u, 0
            else:
                quiet += 1
            s_prev = s
        m_out = 0.5 * r[-1] * (1 - mots[-1])
        if m_init is None:
            m_init = max(m_out, 1e-30)
        elif u_lc is None and m_out < 1e-3 * m_init:
            return "disp", trace
        elif u_lc is not None and u > u_lc + 1.5:
            return "disp", trace
        du = cfl * float(np.quantile(np.diff(r), 0.05))
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
        if collect and len(r) > 8:
            trace.append((u, float(G.bars(r, h)[1]), float(r[-1])))
        # post-echo regrid FREEZE (termination amendments 2+3, disclosed): once crossings have
        # stopped (echoes over) and no MOTS is imminent, stop refining -- the grid then drains
        # in O(N) steps instead of stalling on a geometrically shrinking du. Amendment 3: the
        # 5x-THRESH guard blocked freezing exactly in the near-critical hover window (mots
        # 0.04-0.1 post-closest-approach; 45 min on one bisection probe, measured) -> guard
        # loosened to 2x, plus a quiet-trigger: 20k steps with no crossing and mots > 1.5x
        # THRESH cannot be a collapse in progress (the echoes ARE crossings). Labels unchanged;
        # MOTS detection continues every step on the frozen grid.
        frozen = (u_lc is not None and u > u_lc + 0.2 and mots[j] > 2 * G.MOTS_THRESH) or \
                 (quiet > 20000 and mots[j] > 1.5 * G.MOTS_THRESH)
        if not frozen and 8 < len(r) < n0 // 2:
            rm = 0.5 * (r[1:] + r[:-1]); hm = 0.5 * (h[1:] + h[:-1])
            rn = np.empty(2 * len(r) - 1); hn = np.empty_like(rn)
            rn[0::2] = r; rn[1::2] = rm
            hn[0::2] = h; hn[1::2] = hm
            r, h = rn, hn
    return "disp", trace


def bisect_g(vout, cfl=0.1, lo=0.005, hi=0.1):
    while evolve_g(hi, vout, cfl)[0] != "bh":                    # registered bracket widening
        hi *= 2
        if hi > 0.4:
            return None
    while evolve_g(lo, vout, cfl)[0] == "bh":                    # lower-side escape guard (PR#13
        lo /= 2                                                  # CodeRabbit): lo must DISPERSE,
        if lo < 1e-4:                                            # else bisection returns silent
            return None                                          # garbage near lo
    for i in range(46):
        mid = 0.5 * (lo + hi)
        out = evolve_g(mid, vout, cfl)[0]
        print(f"      bisect[{i:02d}] p={mid:.12f} -> {out}", flush=True)
        if out == "bh":
            hi = mid
        else:
            lo = mid
        if (hi - lo) / hi < 3e-14:
            break
    return 0.5 * (lo + hi)


def events_inbound(trace):
    """A5 re-registered: strictly inbound events; turnaround = min crossing interval; events end
    BEFORE its second endpoint. Per family drop first + last (A6). Returns (F0, F1, ut, r_out_ut)."""
    tr = np.asarray(trace)
    if len(tr) < 4:
        return None
    u, h, rout = tr[:, 0], tr[:, 1], tr[:, 2]
    m = u > UTRIM
    u, h, rout = u[m], h[m], rout[m]
    s = np.sign(h)
    j = np.where(s[1:] * s[:-1] < 0)[0]
    if len(j) < 5:
        return None
    uc = u[j] + (u[j + 1] - u[j]) * np.abs(h[j]) / (np.abs(h[j]) + np.abs(h[j + 1]))
    ti = int(np.argmin(np.diff(uc)))
    ut = float(uc[ti])                                           # turnaround (first endpoint)
    r_ut = float(np.interp(ut, u, rout))
    ucin, jin = uc[:ti + 1], j[:ti + 1]                          # strictly inbound
    pks = []
    for a, b in zip(jin[:-1], jin[1:]):
        if b > a + 1:
            pks.append(u[a + 1 + int(np.argmax(np.abs(h[a + 1:b + 1])))])
    F0, F1 = ucin[1:-1], np.asarray(pks)[1:-1]
    if len(F0) < 4 or len(F1) < 3:
        return None, ut, r_ut
    return (F0, F1), ut, r_ut


def deep_delta(ps, vout, cfl, eps):
    out, tr = evolve_g(ps * (1 - eps), vout, cfl, collect=True)
    if out != "disp":
        return None, out, 0, float("nan"), float("nan")
    got = events_inbound(tr)
    if got is None or got[0] is None:
        n = 0 if (got is None) else -1
        ut, r_ut = (float("nan"), float("nan")) if got is None else (got[1], got[2])
        return None, "few-events", n, ut, r_ut
    (F0, F1), ut, r_ut = got
    D, us, sse, D0, D1 = fit_timing2(F0, F1)
    return {"Delta": float(D), "ustar": float(us), "n": [len(F0), len(F1)],
            "fam": [float(D0), float(D1)],
            "F0": [float(x) for x in F0], "F1": [float(x) for x in F1]}, "ok", \
        len(F0) + len(F1), ut, r_ut


def main():
    print("[C1g] Section III; prereg cosmos/PREREG_C1g_section3.md")
    vout, configs = 3.0, []
    for it in range(4):                                          # v_out^0 + up to 3 refinements
        ps = bisect_g(vout)
        if ps is None:
            print(f"  it{it}: v_out={vout:.4f} -- bracket failed, nm", flush=True)
            break
        fit, status, nev, ut, r_ut = deep_delta(ps, vout, 0.1, 1e-12)
        D = fit["Delta"] if fit else float("nan")
        print(f"  it{it}: v_out={vout:.4f}  p*={ps:.14f}  events={nev}  Delta={D:.3f}  "
              f"u_t={ut:.4f}  r_out(u_t)={r_ut:.4f}", flush=True)
        configs.append({"vout": vout, "pstar": ps, "n_events": nev, "ut": ut, "r_ut": r_ut,
                        "Delta": D})
        if not math.isnan(r_ut) and r_ut < 0.05:
            break
        if math.isnan(r_ut):
            print("    (no turnaround measured -- stopping refinement)", flush=True)
            break
        vout = vout - r_ut + max(0.1 * r_ut, 0.02)
    if len(configs) < 2:
        print("  protocol nm: fewer than 2 configurations"); return
    fin, pen = configs[-1], configs[-2]

    print(f"\n  FINAL config v_out={fin['vout']:.4f}: verdict runs eps in 1e-11/1e-12/3.16e-13:",
          flush=True)
    Ds, diag = [], []
    for e in (1e-11, 1e-12, 3.162e-13):
        fit, status, nev, ut, r_ut = deep_delta(fin["pstar"], fin["vout"], 0.1, e)
        if fit:
            Ds.append(fit["Delta"]); diag.append({"eps": e, **fit})
            print(f"    eps={e:.3e}: events {fit['n'][0]}+{fit['n'][1]}  Delta={fit['Delta']:.3f}"
                  f"  fams {fit['fam'][0]:.2f}/{fit['fam'][1]:.2f}", flush=True)
        else:
            print(f"    eps={e:.3e}: {status} -> nm", flush=True)
    Dmed = float(np.median(Ds)) if Ds else float("nan")
    nev_12 = next((d["n"][0] + d["n"][1] for d in diag if d["eps"] == 1e-12), 0)

    print(f"\n  V3 controls: CFL=0.05 (own bisection) + penultimate v_out={pen['vout']:.4f}:",
          flush=True)
    ps05 = bisect_g(fin["vout"], cfl=0.05)
    D05 = float("nan")
    if ps05:
        fit05, *_ = deep_delta(ps05, fin["vout"], 0.05, 1e-12)
        if fit05:
            D05 = fit05["Delta"]
            print(f"    CFL=0.05: p*={ps05:.14f}  Delta={D05:.3f}  "
                  f"events {fit05['n'][0]}+{fit05['n'][1]}", flush=True)
    Dpen = pen["Delta"]
    print(f"    penultimate: Delta={Dpen:.3f}  (from iteration record)")

    gB = Dmed / (2 * PBAR)
    v1 = nev_12 >= 10
    v2 = len(Ds) >= 2 and abs(Dmed - ANCH_D) <= 0.25
    v3 = (not math.isnan(D05)) and abs(D05 - Dmed) <= 0.15 and abs(Dpen - Dmed) <= 0.15
    v4 = v2 and v3 and abs(gB - 0.374) <= 0.03 and abs(gB - GA) <= 0.03
    print(f"\n  PRE-REGISTERED VERDICTS:")
    print(f"    (1) ladder >=10 inbound events at eps=1e-12: {'PASS' if v1 else 'FAIL'} ({nev_12})")
    print(f"    (2) Delta={Dmed:.3f} in 3.4453+-0.25: {'PASS' if v2 else 'FAIL'} "
          f"(off {abs(Dmed-ANCH_D):.3f})")
    print(f"    (3) controls |dD|<=0.15 both: {'PASS' if v3 else 'FAIL'} "
          f"(sampling {D05-Dmed:+.3f}, ray-alloc {Dpen-Dmed:+.3f})")
    print(f"    (4) implied gamma_B=Delta/(2*4.29)={gB:.4f} in 0.374+-0.03 and near gamma_A: "
          f"{'PASS' if v4 else 'FAIL'} (off {abs(gB-0.374):.4f}, |dgA|={abs(gB-GA):.4f})")
    if v2 and not v3:
        print("    NOTE: verdict 2 VOIDED by verdict-3 failure (registered rule)")
    json.dump({"configs": configs, "final": fin, "Delta": Dmed, "runs": diag,
               "D05": D05, "pstar05": ps05, "Dpen": Dpen, "gamma_B_implied": float(gB),
               "verdicts": {"v1": bool(v1), "v2": bool(v2), "v3": bool(v3), "v4": bool(v4)}},
              open(os.path.join(HERE, "gate_C1g_results.json"), "w"), indent=1)
    print("  wrote cosmos/gate_C1g_results.json")


if __name__ == "__main__":
    main()
