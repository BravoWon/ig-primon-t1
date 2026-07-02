#!/usr/bin/env python
"""LHC gates L1+L2 -- 'chaos / max-entropy / meaning / zeta' made falsifiable. Pre-registered.

Data: CERN ColliderML-Release-1 (SIMULATED 14 TeV pp: MadGraph->Pythia->Geant4; honest flag -- not beam
data), ttbar_pu0 truth particles, cut to hard-scatter primaries (primary & vertex_primary==1, pt>0.5,
|eta|<4), events with >=60 particles kept.

GATE L1 ("chaos = max entropy"): H = events are FAR from max entropy and the structure is rankable.
  Surrogates: S1 pooled-marginal resample (kills all within-event correlation, keeps single-particle
  marginals+multiplicity); S2 angular scramble (keeps pt & |eta| marginals, kills clustering).
  Measures: (a) |sum pT|/sum|pT| (conservation), (b) mean nearest-neighbor DeltaR (jet clustering),
  (c) zlib compressibility of quantized event streams. 'Meaning' = the measured gap vs max-ent -- the
  quarantined-but-operational form of 'observation assigns value'. Expected: huge gaps; value = the ladder.

GATE L2 (the zeta/spectral gate): per-event angular kernel matrix K_ij = exp(-DeltaR_ij^2/2) -> unfolded
  eigenvalue spacing statistics of the BULK + spike counts.
  H-bulk : bulk spacings are RMT-universal (Wigner-surmise family) = the 'chaos'.
  H-spike: spiked outliers vs the surrogate ensemble = the 'meaning' (jets/clusters).
  H-zeta : the PRE-REGISTERED discriminators for 'primes/zeta encode it':
           (i) beta: zeta zeros are GUE (beta=2); real-symmetric kernels generically GOE-like (beta=1).
               KS distance decides which surmise the bulk matches.
           (ii) long-range: zeta's number variance SATURATES (Berry; the prime/explicit-formula
               signature BEYOND generic GUE). Does the collider bulk show zeta-like saturation?
           Expected honestly: beta=1-ish bulk, no zeta saturation -> prime-encoding NULL; the bridge
           that survives is universality-class kinship, not prime-specific encoding.
  Anchors: synthetic GOE + GUE ensembles (the estimator must reproduce their surmises) and the first
  2000 TRUE zeta zeros (mpmath.zetazero, cached), unfolded by the Riemann-von-Mangoldt density.

    python lhc/gate_L1_L2.py
"""
import os, math, zlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_EV, CACHE_ZZ = "lhc/events_ttbar_pu0.npz", "lhc/zeta_zeros.npy"
N_EVENTS, N_ZEROS, MIN_PART = 250, 2000, 60
RED, GREEN, BLUE, NAVY, AMBER = "#c0392b", "#1e7d34", "#2c6fbb", "#15293f", "#9a6a2f"
rng = np.random.default_rng(0)


# ---------------- data ----------------
def get_events():
    if os.path.exists(CACHE_EV):
        d = np.load(CACHE_EV, allow_pickle=True)
        return list(d["events"])
    from datasets import load_dataset
    ds = load_dataset("CERN/ColliderML-Release-1", "ttbar_pu0_particles", split="train", streaming=True)
    events = []
    for i, e in enumerate(ds):
        if len(events) >= N_EVENTS:
            break
        px, py, pz = (np.asarray(e[k], np.float64) for k in ("px", "py", "pz"))
        pri = np.asarray(e["primary"], bool) & (np.asarray(e["vertex_primary"], np.int64) == 1)
        pt = np.hypot(px, py)
        p = np.sqrt(px ** 2 + py ** 2 + pz ** 2) + 1e-12
        eta = np.arctanh(np.clip(pz / p, -1 + 1e-10, 1 - 1e-10))
        m = pri & (pt > 0.5) & (np.abs(eta) < 4.0)
        if m.sum() >= MIN_PART:
            events.append(np.stack([px[m], py[m], pz[m], pt[m], eta[m], np.arctan2(py, px)[m]], 1))
    np.savez_compressed(CACHE_EV, events=np.array(events, dtype=object))
    return events


def get_zeta_zeros():
    if os.path.exists(CACHE_ZZ):
        return np.load(CACHE_ZZ)
    from mpmath import mp, zetazero
    mp.dps = 15
    z = np.array([float(zetazero(k).imag) for k in range(1, N_ZEROS + 1)])
    np.save(CACHE_ZZ, z)
    return z


# ---------------- gate L1 ----------------
def surrogate_pool(events):
    pool = np.concatenate(events, 0)
    out = []
    for ev in events:
        out.append(pool[rng.choice(len(pool), len(ev), replace=False)])
    return out


def surrogate_angular(events):
    out = []
    for ev in events:
        s = ev.copy()
        s[:, 4] = rng.permutation(s[:, 4])                      # permute eta across particles
        s[:, 5] = rng.uniform(-math.pi, math.pi, len(s))        # random phi
        s[:, 0] = s[:, 3] * np.cos(s[:, 5]); s[:, 1] = s[:, 3] * np.sin(s[:, 5])
        s[:, 2] = s[:, 3] * np.sinh(s[:, 4])
        out.append(s)
    return out


def L1_measures(events, name):
    bal, nnd, comp = [], [], []
    for ev in events:
        px, py, pt, eta, phi = ev[:, 0], ev[:, 1], ev[:, 3], ev[:, 4], ev[:, 5]
        bal.append(np.hypot(px.sum(), py.sum()) / pt.sum())
        dphi = np.abs(phi[:, None] - phi[None, :]); dphi = np.minimum(dphi, 2 * math.pi - dphi)
        dR = np.sqrt((eta[:, None] - eta[None, :]) ** 2 + dphi ** 2) + np.eye(len(ev)) * 1e9
        nnd.append(dR.min(1).mean())
        q = np.clip((ev[:, :3] / 50.0 * 127), -127, 127).astype(np.int8).tobytes()
        comp.append(len(zlib.compress(q, 9)) / len(q))
    return dict(name=name, bal=np.mean(bal), nnd=np.mean(nnd), comp=np.mean(comp))


# ---------------- gate L2 ----------------
def unfold_spacings(vals, trim=0.1):
    v = np.sort(vals)
    k = int(len(v) * trim); v = v[k:len(v) - k]                 # bulk only
    ranks = np.arange(len(v))
    coef = np.polyfit(v, ranks, 7)                             # smooth integrated density
    u = np.polyval(coef, v)
    s = np.diff(u)
    return s[s > 0]


def kernel_spacings(events):
    allS, spikes = [], []
    sur = surrogate_angular(events)
    for ev, sv in zip(events, sur):
        for E, coll in ((ev, allS), (sv, None)):
            eta, phi = E[:, 4], E[:, 5]
            dphi = np.abs(phi[:, None] - phi[None, :]); dphi = np.minimum(dphi, 2 * math.pi - dphi)
            dR2 = (eta[:, None] - eta[None, :]) ** 2 + dphi ** 2
            K = np.exp(-dR2 / 2.0)
            w = np.linalg.eigvalsh(K)
            if coll is not None:
                coll.append(unfold_spacings(w))
                ev_top = np.sort(w)[-12:]
            else:
                sur_top = np.sort(np.linalg.eigvalsh(np.exp(-((E[:, 4][:, None] - E[:, 4][None, :]) ** 2
                                    + np.minimum(np.abs(E[:, 5][:, None] - E[:, 5][None, :]),
                                                 2 * math.pi - np.abs(E[:, 5][:, None] - E[:, 5][None, :])) ** 2) / 2.0)))[-12:]
        spikes.append(int((ev_top > sur_top.max()).sum()))
    return np.concatenate(allS), np.array(spikes)


def synth_spacings(beta, n=200, reals=60):
    out = []
    for _ in range(reals):
        if beta == 1:
            A = rng.standard_normal((n, n)); H = (A + A.T) / 2
        else:
            A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)); H = (A + A.conj().T) / 2
        out.append(unfold_spacings(np.linalg.eigvalsh(H)))
    return np.concatenate(out)


def number_variance(unfolded_positions, Ls):
    x = np.sort(unfolded_positions)
    out = []
    for L in Ls:
        cnts = []
        starts = rng.uniform(x[0], x[-1] - L, 400)
        for s in starts:
            cnts.append(((x >= s) & (x < s + L)).sum())
        out.append(np.var(cnts))
    return np.array(out)


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b]); allv.sort()
    ca = np.searchsorted(a, allv, "right") / len(a)
    cb = np.searchsorted(b, allv, "right") / len(b)
    return np.abs(ca - cb).max()


def main():
    print("[LHC gates] ttbar_pu0 hard-scatter primaries; SIMULATED data (flagged); pre-registered L1+L2")
    events = get_events()
    print(f"  events kept: {len(events)}  (median particles/event: {int(np.median([len(e) for e in events]))})")

    print("\n=== GATE L1: distance from max entropy (the structure ladder) ===")
    rows = [L1_measures(events, "REAL"),
            L1_measures(surrogate_angular(events), "angular-scramble"),
            L1_measures(surrogate_pool(events), "pooled-marginal")]
    print(f"  {'ensemble':>18} {'pT-balance':>11} {'mean nnDR':>10} {'zlib ratio':>11}")
    for r in rows:
        print(f"  {r['name']:>18} {r['bal']:>11.4f} {r['nnd']:>10.4f} {r['comp']:>11.4f}")
    b0, b1 = rows[0], rows[1]
    print(f"  -> conservation structure: REAL pT-balance {b0['bal']:.4f} vs scrambled {b1['bal']:.4f} "
          f"({b1['bal']/max(b0['bal'],1e-9):.1f}x) | clustering: nnDR {b0['nnd']:.3f} vs {b1['nnd']:.3f}")
    print(f"  VERDICT L1: events are {'FAR from max entropy -- structure is real, localized, rankable' if (b1['bal']>2*b0['bal'] or b0['nnd']<0.8*b1['nnd']) else 'not clearly structured (unexpected!)'}")

    print("\n=== GATE L2: spectral bulk vs GOE/GUE/zeta + spikes ===")
    sR, spikes = kernel_spacings(events)
    sur_sp = kernel_spacings(surrogate_angular(events))[0]
    sGOE, sGUE = synth_spacings(1), synth_spacings(2)
    zz = get_zeta_zeros()
    g = zz; uz = g / (2 * math.pi) * np.log(g / (2 * math.pi * math.e))    # RvM unfolding
    sZ = np.diff(uz); sZ = sZ[sZ > 0]
    for nm, s in (("real-bulk", sR), ("surrogate-bulk", sur_sp), ("zeta", sZ)):
        print(f"  {nm:>15}: n={len(s):6d}  mean s={s.mean():.3f}")
    print(f"\n  KS distances of unfolded spacing distributions:")
    pairs = [("real vs GOE(b=1)", sR, sGOE), ("real vs GUE(b=2)", sR, sGUE), ("real vs zeta", sR, sZ),
             ("zeta vs GUE (sanity)", sZ, sGUE), ("surrogate vs GOE", sur_sp, sGOE)]
    kss = {}
    for nm, a, b in pairs:
        kss[nm] = ks(a / a.mean(), b / b.mean())
        print(f"    {nm:>22}: KS = {kss[nm]:.4f}")
    beta_match = "GOE (beta=1)" if kss["real vs GOE(b=1)"] < kss["real vs GUE(b=2)"] else "GUE (beta=2 = zeta class)"
    print(f"  -> bulk universality: closer to {beta_match}")
    print(f"  spikes (meaning): mean {spikes.mean():.1f} spiked eigenvalues/event above the surrogate max "
          f"(chaos bulk + physics spikes)")
    Ls = np.linspace(1, 15, 12)
    nvR = number_variance(np.cumsum(sR / sR.mean()), Ls)
    nvZ = number_variance(uz, Ls)
    nvGUE = number_variance(np.cumsum(sGUE / sGUE.mean()), Ls)
    sat_z = nvZ[-1] / nvGUE[-1]
    sat_r = nvR[-1] / nvGUE[-1]
    print(f"  long-range number variance at L=15: real {nvR[-1]:.2f}  GUE {nvGUE[-1]:.2f}  zeta {nvZ[-1]:.2f}")
    print(f"  -> zeta SATURATES vs GUE ({sat_z:.2f}x, the PRIME signature); real-bulk ratio {sat_r:.2f}x")
    zeta_specific = abs(sat_r - sat_z) < 0.25 * sat_z and kss["real vs zeta"] < kss["real vs GOE(b=1)"]
    print(f"  VERDICT L2 (pre-registered): H-bulk {'PASS' if min(kss['real vs GOE(b=1)'],kss['real vs GUE(b=2)'])<0.08 else 'weak'} | "
          f"H-spike {'PASS' if spikes.mean()>1.5 else 'FAIL'} | "
          f"H-zeta(prime-specific) {'SUPPORTED (!!)' if zeta_specific else 'NULL -- kinship is generic RMT universality, not prime encoding'}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    bins = np.linspace(0, 3.2, 40)
    for s, nm, c in ((sR, "real bulk", GREEN), (sZ, "zeta zeros", RED)):
        axes[0].hist(s / s.mean(), bins=bins, density=True, histtype="step", lw=1.6, label=nm, color=c)
    x = np.linspace(0, 3.2, 200)
    axes[0].plot(x, np.pi / 2 * x * np.exp(-np.pi * x ** 2 / 4), ":", color=BLUE, label="GOE surmise")
    axes[0].plot(x, 32 / np.pi ** 2 * x ** 2 * np.exp(-4 * x ** 2 / np.pi), "--", color=AMBER, label="GUE surmise")
    axes[0].set_title("unfolded spacings: collider bulk vs zeta vs surmises", fontsize=9, color=NAVY)
    axes[0].legend(fontsize=7, frameon=False); axes[0].set_xlabel("s")
    axes[1].plot(Ls, nvR, "o-", color=GREEN, label="real bulk")
    axes[1].plot(Ls, nvGUE, "--", color=AMBER, label="GUE")
    axes[1].plot(Ls, nvZ, "o-", color=RED, label="zeta (saturates = primes)")
    axes[1].set_title("number variance (long-range rigidity)", fontsize=9, color=NAVY)
    axes[1].legend(fontsize=7, frameon=False); axes[1].set_xlabel("L")
    axes[2].hist(spikes, bins=np.arange(-0.5, 12.5), color=GREEN)
    axes[2].set_title("spiked eigenvalues per event (the 'meaning')", fontsize=9, color=NAVY)
    axes[2].set_xlabel("# spikes above surrogate max")
    fig.tight_layout(); fig.savefig("lhc/gate_L1_L2.png", dpi=160); plt.close(fig)
    print("  wrote lhc/gate_L1_L2.png")


if __name__ == "__main__":
    main()
