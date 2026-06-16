"""genuine_sweep_v3.py -- the read-the-board run.

Composes the validated pieces: per-point thermodynamic-length ladders (genuine_ladder_tuning) +
likelihood-only tempering (prior as base measure) + MALA/dual-averaging/round-trip/init-independence
(genuine_tempered_sweep). For each (H, n):

  1. auto-detect beta_hot by the melt (descend until Var(E_lik) plateaus -> the top chain is at the prior).
  2. measure C(beta)=beta^2 Var(E_lik) over [beta_hot, beta_cold]; build the equal-thermodynamic-length ladder
     (K is an output); keep the C(beta) profile (a specific-heat peak = landscape-transition signature).
  3. run the likelihood-tempered sweep from TWO chain-inits on the SAME data (v2 bug fix: v2 used different
     data seeds, conflating init- with data-dependence).
  4. GATE: certified iff round-trips/replica > 1 AND fs_Rhat < 1.1 (both inits) AND |dq| < 0.03 (init-indep).
     Else INCONCLUSIVE. chi_F is admissible ONLY where certified; the K at which a corner still stalls is a
     quantitative barrier floor (glassy bedrock), making INCONCLUSIVE a proven bound.

NO branch verdict; the certified chi_F-vs-H trend is read in-the-loop. [E-hw] Torch/CUDA.
"""
import json, math, torch
import genuine_ladder_tuning as LT
import genuine_tempered_sweep as G

PRIOR = LT.PRIOR


def make_data(H, nd, d_in, dev, seed=0):
    g = torch.Generator(device=dev); g.manual_seed(seed)
    X = torch.randn(nd, d_in, generator=g, device=dev)
    tW1 = 1.5 * torch.randn(H, d_in, generator=g, device=dev) / d_in ** 0.5
    tW2 = 1.5 * torch.randn(H, generator=g, device=dev) / H ** 0.5
    y = torch.tanh(X @ tW1.t()) @ tW2
    D = torch.randn(512, d_in, generator=g, device=dev)
    return X, y, D


def var_lik(X, y, H, d_in, nd, beta, dev, gen, E=32, steps=1500):
    P = sum(G.mlp_dims(H, d_in)); W = 0.3 * torch.randn(E, P, generator=gen, device=dev)
    bch = torch.full((E,), beta, device=dev); ones = torch.ones(E, device=dev)
    Lk = lambda W: LT.lik(W, X, y, H, d_in, nd)
    Vb = lambda W: PRIOR * (W ** 2).sum(1) + bch * Lk(W)
    da = G.DualAvg(torch.log(torch.tensor([2e-3 / beta], device=dev)), 0.574); tau = 2e-3 / beta
    for _ in range(steps // 2):
        W, acc = G.mala_step(W, Vb, ones, torch.full((E,), tau, device=dev), gen)
        tau = float(da.update(acc.float().mean().view(1))[0])
    tau = float(da.final()[0]); Es = []
    for _ in range(steps // 2):
        W, _ = G.mala_step(W, Vb, ones, torch.full((E,), tau, device=dev), gen)
        Es.append(Lk(W).detach())
    return float(torch.stack(Es, 1).var(unbiased=True))


def auto_beta_hot(X, y, H, d_in, nd, dev, gen, b0=0.02, floor=2e-5):
    """Descend beta until Var(E_lik) plateaus (the chain has reached the prior)."""
    b = b0; v = var_lik(X, y, H, d_in, nd, b, dev, gen); trail = [(b, v)]
    while b > floor:
        b2 = b / 4; v2 = var_lik(X, y, H, d_in, nd, b2, dev, gen); trail.append((b2, v2))
        if v2 < 1.3 * v:
            return b2, "melted", trail
        b, v = b2, v2
    return floor, "floor(still-climbing)", trail


def lik_tempered(X, y, D, H, nd, ladder, dev, gen, init_scale, init_seed,
                 W_init=None, init_noise=0.02,
                 E=24, n_adapt=2000, n_burn=5000, n_meas=9000, swap_every=6, meas_every=25):
    d_in = 8; P = sum(G.mlp_dims(H, d_in)); L = len(ladder)
    rung = torch.arange(E * L, device=dev) % L; blad = torch.tensor(ladder, device=dev); bch = blad[rung]
    Lk = lambda W: LT.lik(W, X, y, H, d_in, nd)
    Vb = lambda W: PRIOR * (W ** 2).sum(1) + bch * Lk(W)
    gi = torch.Generator(device=dev); gi.manual_seed(init_seed)
    if W_init is None:                                   # original: random init at a scale
        W = init_scale * torch.randn(E * L, P, generator=gi, device=dev)
    else:                                                # v4: warm-start near a given solution (adversarial sep.)
        W = W_init.expand(E * L, P) + init_noise * torch.randn(E * L, P, generator=gi, device=dev)
    ones = torch.ones(E * L, device=dev)
    da = G.DualAvg(torch.log(2e-3 / blad), 0.574); tau_r = 2e-3 / blad
    for _ in range(n_adapt):
        W, acc = G.mala_step(W, Vb, ones, tau_r[rung], gen); tau_r = da.update(acc.float().view(E, L).mean(0))
    tau_r = da.final()
    perm = torch.arange(L, device=dev).repeat(E, 1); rt = G.RoundTrip(E, L, dev); sacc = []; snaps = []
    for t in range(n_burn + n_meas):
        W, acc = G.mala_step(W, Vb, ones, tau_r[rung], gen)
        if t % swap_every == 0:
            W, perm, s = G.pt_swap(W, perm, Lk, ladder, E, L, gen); sacc.append(s); rt.update(perm)
        if t >= n_burn and (t - n_burn) % meas_every == 0:
            snaps.append(G.phi_flat(W.view(E, L, -1)[:, L - 1, :], D, H, d_in))
    PHI = torch.stack(snaps, 1)
    d_ = G.chi_diag(PHI, H); d_.update(rt_per_replica=rt.round_trips() / E, swap_acc=sum(sacc) / len(sacc), K=L)
    return d_


def run_point(H, nd, dev, gen, beta_cold=20.0):
    d_in = 8; X, y, D = make_data(H, nd, d_in, dev)
    bhot, melt, trail = auto_beta_hot(X, y, H, d_in, nd, dev, gen)
    grid = [bhot * (beta_cold / bhot) ** (i / 23) for i in range(24)]
    prof = LT.measure_C(H, nd, grid, dev, gen)
    peak = max(prof, key=lambda p: p["C"])
    Ltot, K, ladder = LT.thermo_ladder(prof)
    A = lik_tempered(X, y, D, H, nd, ladder, dev, gen, 0.1, 0)
    B = lik_tempered(X, y, D, H, nd, ladder, dev, gen, 1.0, 17)
    dq = abs(A["q_mean"] - B["q_mean"])
    flow = min(A["rt_per_replica"], B["rt_per_replica"]) > 1.0
    mixed = A["fs_rhat"] < 1.1 and B["fs_rhat"] < 1.1
    certified = flow and mixed and dq < 0.03
    return dict(H=H, n_data=nd, beta_hot=bhot, melt=melt, K=K, L_total=round(Ltot, 2),
                peak_beta=peak["beta"], peak_C=round(peak["C"], 1), C_flat=(peak["C"] < 1.5 * min(p["C"] for p in prof)),
                certified=bool(certified), verdict="certified" if certified else "INCONCLUSIVE",
                dq=round(dq, 3), A=A, B=B)


def full(points=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"; gen = torch.Generator(device=dev)
    points = points or [(32, 24), (64, 24), (128, 24), (64, 96)]
    with open("genuine_v3_results.jsonl", "a") as f:
        for (H, nd) in points:
            r = run_point(H, nd, dev, gen)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"H{H} n{nd}: b_hot={r['beta_hot']:.1e}({r['melt']}) K={r['K']} peakC@{r['peak_beta']:.2f} "
                  f"flat={r['C_flat']} | chiF A={r['A']['chi_F']:.2f}/B={r['B']['chi_F']:.2f} "
                  f"rt={r['A']['rt_per_replica']:.1f} fsR={r['A']['fs_rhat']:.2f}/{r['B']['fs_rhat']:.2f} "
                  f"dq={r['dq']:.3f} -> {r['verdict']}")
    print("DONE -> genuine_v3_results.jsonl. chi_F admissible only where certified.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--one", action="store_true"); a = ap.parse_args()
    full(points=[(64, 24)] if a.one else None)
