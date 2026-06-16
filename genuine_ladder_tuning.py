"""genuine_ladder_tuning.py -- build the PT ladder from the posterior's heat-capacity profile.

The number of rungs is NOT a formula guess; it is set by thermodynamic length:
    L = integral sqrt(C(beta)) d(ln beta),   C(beta) = beta^2 * Var(E_lik)   [heat capacity]
Equal spacing in L gives constant adjacent swap acceptance; K = L_total / ell_target (ell~1-1.5 -> acc 0.2-0.4).
That is what Katzgraber-Trebst feedback converges to; we compute it directly. K is an OUTPUT.

Correction this requires: temper the LIKELIHOOD only, with the prior as the beta-independent base measure:
    pi_beta(w) ~ exp(-lambda||w||^2) * exp(-beta * n*MSE).
Then beta->0 -> prior (a PROPER melt), so 'hot enough' is verifiable (Var(E_lik) plateaus, autocorr short).
The v1/v2 code tempered prior+likelihood together, for which beta->0 is improper -- no defined melt.

Riders honored: (1) beta_hot is MEASURED (push down until the melt plateaus), not assumed 0.05.
(2) the C(beta) curve is kept -- a specific-heat PEAK is a landscape-transition signature, adjacent to the
branch question, not just plumbing. (3) if no feasible K with rungs bunched at the peak gets round-trips off
zero, the K at which it stalls is a quantitative floor on the barrier -> INCONCLUSIVE as a PROVEN result.

[E-hw] Torch/CUDA. Hardest point first: H128 n24.
"""
import math, torch
import genuine_tempered_sweep as G

PRIOR = 1e-2


def problem(H, nd, d_in, dev, gen, seed=0):
    gen.manual_seed(seed)
    X = torch.randn(nd, d_in, generator=gen, device=dev)
    tW1 = 1.5 * torch.randn(H, d_in, generator=gen, device=dev) / d_in ** 0.5
    tW2 = 1.5 * torch.randn(H, generator=gen, device=dev) / H ** 0.5
    y = torch.tanh(X @ tW1.t()) @ tW2
    return X, y


def lik(W, X, y, H, d_in, n):
    return n * ((G.forward_flat(W, X, H, d_in) - y[None, :]) ** 2).mean(1)


def tau_int(x, c=5.0):
    x = x - x.mean(); v = (x ** 2).mean()
    if float(v) == 0: return 1.0
    tau = 1.0
    for k in range(1, x.numel()):
        rho = float((x[:-k] * x[k:]).mean() / v); tau += 2 * rho
        if k >= c * tau: break
    return max(tau, 1.0)


def measure_C(H, nd, grid, dev, gen, E=32, n_adapt=1500, n_meas=2500):
    """Per-beta heat capacity C=beta^2 Var(E_lik), with likelihood-only tempering and dual-averaged tau."""
    d_in = 8; X, y = problem(H, nd, d_in, dev, gen); P = sum(G.mlp_dims(H, d_in))
    Gn = len(grid); betas = torch.tensor(grid, device=dev).repeat_interleave(E)   # (E*Gn,) grouped by beta
    grp = torch.arange(E * Gn, device=dev) // E
    Lk = lambda W: lik(W, X, y, H, d_in, nd)
    Vb = lambda W: PRIOR * (W ** 2).sum(1) + betas * Lk(W)                         # prior + beta*lik
    W = 0.3 * torch.randn(E * Gn, P, generator=gen, device=dev)
    ones = torch.ones(E * Gn, device=dev)
    da = G.DualAvg(torch.log(2e-3 / torch.tensor(grid, device=dev)), 0.574)
    tau_g = 2e-3 / torch.tensor(grid, device=dev)
    for _ in range(n_adapt):
        W, acc = G.mala_step(W, Vb, ones, tau_g[grp], gen)
        tau_g = da.update(acc.float().view(Gn, E).mean(1))
    tau_g = da.final()
    Es = []
    for _ in range(n_meas):
        W, acc = G.mala_step(W, Vb, ones, tau_g[grp], gen)
        Es.append(Lk(W).detach())
    E_t = torch.stack(Es, 1)                                                       # (E*Gn, n_meas)
    out = []
    for i, b in enumerate(grid):
        sel = E_t[grp == i]                                                        # (E, n_meas)
        varE = float(sel.var(unbiased=True)); meanE = float(sel.mean())
        ti = tau_int(sel.mean(0))
        out.append(dict(beta=b, meanE=meanE, varE=varE, C=b * b * varE, tau=ti))
    return out


def thermo_ladder(prof, ell=1.2):
    """L(beta)=int sqrt(C) dln(beta) = int sqrt(VarE) dbeta. Rungs equally spaced in L."""
    b = [p["beta"] for p in prof]; sv = [math.sqrt(max(p["varE"], 0)) for p in prof]
    Lcum = [0.0]
    for i in range(1, len(b)):
        Lcum.append(Lcum[-1] + 0.5 * (sv[i] + sv[i - 1]) * (b[i] - b[i - 1]))
    Ltot = Lcum[-1]; K = max(4, round(Ltot / ell))
    targets = [Ltot * j / (K - 1) for j in range(K)]
    ladder = []
    for tL in targets:                                                            # invert L(beta)
        for i in range(1, len(Lcum)):
            if Lcum[i] >= tL:
                f = (tL - Lcum[i - 1]) / (Lcum[i] - Lcum[i - 1] + 1e-12)
                ladder.append(b[i - 1] + f * (b[i] - b[i - 1])); break
        else:
            ladder.append(b[-1])
    return Ltot, K, ladder


def validate_flow(H, nd, ladder, dev, gen, E=24, n_adapt=2000, n_burn=4000, n_meas=6000, swap_every=10):
    d_in = 8; X, y = problem(H, nd, d_in, dev, gen); P = sum(G.mlp_dims(H, d_in)); L = len(ladder)
    rung = torch.arange(E * L, device=dev) % L; blad = torch.tensor(ladder, device=dev); bch = blad[rung]
    Lk = lambda W: lik(W, X, y, H, d_in, nd)
    Vb = lambda W: PRIOR * (W ** 2).sum(1) + bch * Lk(W)
    W = 0.1 * torch.randn(E * L, P, generator=gen, device=dev); ones = torch.ones(E * L, device=dev)
    da = G.DualAvg(torch.log(2e-3 / blad), 0.574); tau_r = 2e-3 / blad
    for t in range(n_adapt):
        W, acc = G.mala_step(W, Vb, ones, tau_r[rung], gen); tau_r = da.update(acc.float().view(E, L).mean(0))
    tau_r = da.final()
    perm = torch.arange(L, device=dev).repeat(E, 1); rt = G.RoundTrip(E, L, dev); sacc = []
    for t in range(n_burn + n_meas):
        W, acc = G.mala_step(W, Vb, ones, tau_r[rung], gen)
        if t % swap_every == 0:
            W, perm, s = G.pt_swap(W, perm, Lk, ladder, E, L, gen); sacc.append(s); rt.update(perm)
    return dict(rt_per_replica=rt.round_trips() / E, swap_acc=sum(sacc) / len(sacc), K=L)


def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; g = torch.Generator(device=dev)
    H, nd = 128, 24
    grid = [round(0.01 * (20 / 0.01) ** (i / 23), 4) for i in range(24)]            # log 0.01 -> 20
    print(f"[ladder tuning on {dev}]  hardest point H{H} n{nd}; likelihood-only tempering\n")
    prof = measure_C(H, nd, grid, dev, g)
    print(f"{'beta':>8}{'C=b^2VarE':>12}{'VarE':>10}{'tau(E)':>8}")
    for p in prof:
        print(f"{p['beta']:>8.3f}{p['C']:>12.2f}{p['varE']:>10.3f}{p['tau']:>8.1f}")
    peak = max(prof, key=lambda p: p["C"])
    hot = prof[0]
    melt = "MELTED (VarE plateau + short tau)" if prof[1]["varE"] < 1.3 * hot["varE"] and hot["tau"] < 10 else \
           "NOT melted -- push beta_hot lower"
    print(f"\nspecific-heat PEAK at beta={peak['beta']:.3f} (C={peak['C']:.1f})  <- rungs bunch here; landscape-transition marker")
    print(f"hot end beta={hot['beta']}: VarE={hot['varE']:.3f} tau={hot['tau']:.1f}  -> {melt}")
    Ltot, K, ladder = thermo_ladder(prof)
    print(f"\nthermodynamic length L_total={Ltot:.2f}  ->  K={K} rungs (ell~1.2)")
    print(f"ladder (equal-L, bunched at peak): {[round(x,3) for x in ladder]}")
    print(f"\nvalidating round-trip FLOW on H{H} n{nd} with the measured ladder (gate: rt/replica > 1)...")
    fl = validate_flow(H, nd, ladder, dev, g)
    verdict = "UNLOCKED -- ladder flows, re-run the sweep with it" if fl["rt_per_replica"] > 1.0 else \
              f"STILL STALLED at K={fl['K']} -- glassy-bedrock candidate; K-floor is a quantitative barrier bound"
    print(f"  K={fl['K']}  swap_acc={fl['swap_acc']:.3f}  round-trips/replica={fl['rt_per_replica']:.2f}  -> {verdict}")


if __name__ == "__main__":
    run()
