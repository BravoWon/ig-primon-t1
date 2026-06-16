"""genuine_tempered_sweep.py (v2) -- function-space chi_F with MALA + feedback-tuned parallel tempering.

Spec + the v2 refinement (round-trip flow and init-independence are the gate, NOT adjacent swap acceptance):

  - Function-space chi_F on a fixed holdout probe (D=512, in-distribution).  [guard-validated]
  - MALA (MH-correct, exact target) + REAL replica-exchange tempering.        [Gaussian-validated]
  - Per-rung dual-averaging step size -> MALA acceptance ~0.574 PER (H,n,rung) (kills the local mistuning
    artifact that faked the FSS trend in v1: chi_F 0.51->1.89->2.54 was collinear with acc 0.69->0.26->0.07).
  - ROUND-TRIP flow as the ladder metric. Local swap acceptance ~0.3 everywhere does NOT guarantee replicas
    travel hot->cold->hot; more rungs improve adjacent overlap but lengthen the round-trip path. We track and
    report actual round trips (feedback-optimized PT objective), because local acceptance is blind to the
    barrier-trapping that IS the global stuck-vs-glassy confound.
  - EQUILIBRATION GATE = init-independence: two very different initializations must give the SAME cold-rung
    q-distribution. fs_R-hat<1.1 in function space is NECESSARY; init-independence is what CERTIFIES it.
  - THREE terminal outcomes, all reported honestly: (a) trend SURVIVES under certified equilibration = physics;
    (b) trend VANISHES once acceptance is uniform = it was the artifact; (c) largest-H corner CANNOT be
    certified even tuned right = INCONCLUSIVE (the genuinely-glassy regime; bounds where a branch claim is
    reachable -- itself a result).

  --mode verify : perm guard + shuffle floor + probe sanity + Gaussian sampler-correctness + per-rung
                  acceptance uniformity (post-adaptation) + round-trip flow > 0 + init-independence plumbing.
  --mode full   : per (H,n) run TWO inits, certify equilibration, emit chi_F + flow + FLAGs. NO branch verdict.

[E-hw] Torch/CUDA. Empirical-Fisher preconditioner remains a documented default-OFF hook. NOT a [V] receipt.
"""
import argparse, json, math, torch
import phase2_posterior_sampler as S


# ---------------- flat MLP parameterisation ----------------
def mlp_dims(H, d_in): return H * d_in, H, H, 1
def unflat(W, H, d_in):
    s = mlp_dims(H, d_in); i = [0]
    def take(n):
        a = W[:, i[0]:i[0] + n]; i[0] += n; return a
    return take(s[0]).view(-1, H, d_in), take(s[1]), take(s[2]), take(s[3]).squeeze(-1)
def forward_flat(W, X, H, d_in):
    W1, b1, W2, b2 = unflat(W, H, d_in)
    h = torch.tanh(torch.einsum("nd,khd->knh", X, W1) + b1[:, None, :])
    return torch.einsum("knh,kh->kn", h, W2) + b2[:, None]
def phi_flat(W, D, H, d_in):
    f = forward_flat(W, D, H, d_in); f = f - f.mean(1, keepdim=True)
    return f / (f.norm(dim=1, keepdim=True) + 1e-12)
def mlp_energy_fn(X, y, H, d_in, n_data, prior):
    def U(W):
        return n_data * ((forward_flat(W, X, H, d_in) - y[None, :]) ** 2).mean(1) + prior * (W ** 2).sum(1)
    return U
def permute_hidden_flat(W, perm, H, d_in):
    W1, b1, W2, b2 = unflat(W, H, d_in)
    return torch.cat([W1[:, perm, :].reshape(W.shape[0], -1), b1[:, perm], W2[:, perm], b2[:, None]], 1)


# ---------------- MALA (MH-correct) ----------------
def mala_step(W, energy_fn, beta, tau, gen):
    W = W.detach().requires_grad_(True)
    U0 = energy_fn(W); g0, = torch.autograd.grad(U0.sum(), W)
    tb = (tau * beta)[:, None]; mu0 = W.detach() - tb * g0
    Wp = (mu0 + (2 * tau)[:, None].sqrt() * torch.randn(W.shape, generator=gen, device=W.device)).detach().requires_grad_(True)
    U1 = energy_fn(Wp); g1, = torch.autograd.grad(U1.sum(), Wp); mu1 = Wp.detach() - tb * g1
    logq_fwd = -((Wp.detach() - mu0) ** 2).sum(1) / (4 * tau)
    logq_rev = -((W.detach() - mu1) ** 2).sum(1) / (4 * tau)
    acc = torch.log(torch.rand(beta.shape, generator=gen, device=W.device)) < (-beta * (U1.detach() - U0.detach()) + logq_rev - logq_fwd)
    return torch.where(acc[:, None], Wp.detach(), W.detach()), acc


def pt_swap(W, perm, energy_fn, ladder, E, L, gen):
    """Replica exchange across adjacent rungs; tracks config-id perm for round-trip flow."""
    U = energy_fn(W).detach().view(E, L); Wv = W.view(E, L, -1).clone(); pm = perm.clone(); rate = []
    for parity in (0, 1):
        for r in range(parity, L - 1, 2):
            d = (ladder[r] - ladder[r + 1]) * (U[:, r] - U[:, r + 1])
            sw = torch.log(torch.rand(E, generator=gen, device=W.device)) < d
            rate.append(float(sw.float().mean()))
            a = Wv[:, r, :].clone(); Wv[:, r, :] = torch.where(sw[:, None], Wv[:, r + 1, :], Wv[:, r, :]); Wv[:, r + 1, :] = torch.where(sw[:, None], a, Wv[:, r + 1, :])
            pa = pm[:, r].clone(); pm[:, r] = torch.where(sw, pm[:, r + 1], pm[:, r]); pm[:, r + 1] = torch.where(sw, pa, pm[:, r + 1])
            U = energy_fn(Wv.reshape(E * L, -1)).detach().view(E, L)
    return Wv.reshape(E * L, -1), pm, (sum(rate) / len(rate))


class RoundTrip:
    """Count ladder traversals: a replica reaching the cold end having last been at the hot end (and vice
    versa). Round trips = traversals/2 -- the feedback-optimized-PT flow metric, blind-spot of local acc."""
    def __init__(self, E, L, dev):
        self.L = L; self.tag = torch.zeros(E, L, dtype=torch.int8, device=dev); self.trav = torch.zeros(E, device=dev)
        self.idx = torch.arange(E, device=dev)
    def update(self, perm):
        cid = perm[:, self.L - 1]; was = self.tag[self.idx, cid]; self.trav += (was == -1).float(); self.tag[self.idx, cid] = 1
        hid = perm[:, 0]; wash = self.tag[self.idx, hid]; self.trav += (wash == 1).float(); self.tag[self.idx, hid] = -1
    def round_trips(self): return float(self.trav.sum()) / 2.0


class DualAvg:
    """Per-rung dual-averaging (Hoffman-Gelman) to target MALA acceptance per rung."""
    def __init__(self, log_tau0, target):
        self.mu = log_tau0 + math.log(10); self.target = target; self.Hbar = torch.zeros_like(log_tau0)
        self.ltb = torch.zeros_like(log_tau0); self.m = 0
    def update(self, acc_rung):
        self.m += 1; m = self.m
        self.Hbar = (1 - 1 / (m + 10)) * self.Hbar + (1 / (m + 10)) * (self.target - acc_rung)
        lt = self.mu - (m ** 0.5 / 0.05) * self.Hbar; w = m ** (-0.75)
        self.ltb = w * lt + (1 - w) * self.ltb
        return torch.exp(lt)
    def final(self): return torch.exp(self.ltb)


def run_tempered(energy_fn, P, E, ladder, dev, gen, n_adapt=1500, n_burn=4000, n_meas=6000,
                 swap_every=10, meas_every=25, readout=None, init_scale=0.1, target_acc=0.574):
    L = len(ladder); rung = torch.arange(E * L, device=dev) % L
    blad = torch.tensor(ladder, device=dev); betas = blad[rung]
    W = init_scale * torch.randn(E * L, P, generator=gen, device=dev)
    da = DualAvg(torch.log(2e-3 / blad), target_acc); tau_r = 2e-3 / blad
    # --- adapt per-rung tau ---
    for t in range(n_adapt):
        W, acc = mala_step(W, energy_fn, betas, tau_r[rung], gen)
        tau_r = da.update(acc.float().view(E, L).mean(0))
        if t % swap_every == 0:
            W, _, _ = pt_swap(W, torch.arange(L, device=dev).repeat(E, 1), energy_fn, ladder, E, L, gen)
    tau_r = da.final(); tau = tau_r[rung]
    # --- burn + measure with round-trip flow ---
    perm = torch.arange(L, device=dev).repeat(E, 1); rt = RoundTrip(E, L, dev)
    macc, sacc, per_rung = [], [], []
    snaps = []
    for t in range(n_burn + n_meas):
        W, acc = mala_step(W, energy_fn, betas, tau, gen); macc.append(float(acc.float().mean()))
        per_rung.append(acc.float().view(E, L).mean(0))
        if t % swap_every == 0:
            W, perm, s = pt_swap(W, perm, energy_fn, ladder, E, L, gen); sacc.append(s); rt.update(perm)
        if t >= n_burn and (t - n_burn) % meas_every == 0 and readout is not None:
            snaps.append(readout(W.view(E, L, -1)[:, L - 1, :]))
    return W, snaps, dict(mala_acc=sum(macc) / len(macc), swap_acc=sum(sacc) / max(1, len(sacc)),
                          per_rung_acc=[round(x, 2) for x in torch.stack(per_rung).mean(0).tolist()],
                          round_trips=rt.round_trips(), rt_per_replica=rt.round_trips() / E,
                          final_tau=[float(x) for x in tau_r.tolist()])


def chi_diag(PHI, H):
    K, M, _ = PHI.shape; iu = torch.triu_indices(K, K, offset=1)
    qb = torch.stack([PHI[:, t, :] @ PHI[:, t, :].t() for t in range(M)], 0)[:, iu[0], iu[1]]
    chi = lambda q: float(H * q.var(unbiased=True))
    rs = torch.tensor([S.gelman_rubin(PHI[:, :, d]) for d in range(0, PHI.shape[2], max(1, PHI.shape[2] // 64))])
    return dict(q_mean=float(qb.mean()), chi_F=chi(qb.flatten()),
                chi_halves=[chi(qb[:M // 2].flatten()), chi(qb[M // 2:].flatten())],
                fs_rhat=float(rs[torch.isfinite(rs)].median()))


# ---------------- VERIFY ----------------
def verify(probe=512):
    dev = "cuda" if torch.cuda.is_available() else "cpu"; g = torch.Generator(device=dev); g.manual_seed(0)
    d_in, H = 8, 16; P = sum(mlp_dims(H, d_in)); fails = []
    print(f"[verify v2 on {dev}]  probe D={probe}\n")
    D = torch.randn(probe, d_in, generator=g, device=dev)
    W = torch.randn(1, P, generator=g, device=dev); perm = torch.randperm(H, generator=g, device=dev)
    qpp = float((phi_flat(W, D, H, d_in) * phi_flat(permute_hidden_flat(W, perm, H, d_in), D, H, d_in)).sum())
    print(f"1) permutation guard   q(W,permW)={qpp:.12f} (=1)"); fails += [] if abs(qpp - 1) < 1e-9 else ["perm guard"]
    W8 = torch.randn(8, P, generator=g, device=dev)
    PHI = torch.stack([phi_flat(W8, torch.randn(probe, d_in, generator=g, device=dev), H, d_in) for _ in range(40)], 1)
    fl = float(torch.tensor([S.gelman_rubin(PHI.reshape(320, -1)[torch.randperm(320, generator=g, device=dev)].reshape(8, 40, -1)[:, :, d]) for d in range(0, probe, max(1, probe // 64))]).median())
    print(f"2) shuffle floor       fs_Rhat(shuffled)={fl:.4f} (~1.0)"); fails += [] if fl < 1.1 else ["shuffle floor"]
    qs = float((phi_flat(W, D, H, d_in) ** 2).sum())
    qr = float((phi_flat(torch.randn(1, P, generator=g, device=dev), D, H, d_in) * phi_flat(torch.randn(1, P, generator=g, device=dev), D, H, d_in)).sum())
    print(f"3) probe sanity        q(self)={qs:.4f} q(rand,rand)={qr:+.3f}"); fails += [] if abs(qs - 1) < 1e-6 and qr < 0.95 else ["probe sanity"]
    # 4) Gaussian: sampler correctness + per-rung acc uniformity + round-trip flow + init-independence
    d = 10; A = torch.linspace(0.5, 2.0, d, device=dev); eg = lambda W: 0.5 * (W ** 2 * A).sum(1)
    lad = [0.25, 0.4, 0.63, 1.0]
    _, snaps, info = run_tempered(eg, d, E=64, ladder=lad, dev=dev, gen=g, n_adapt=1500, n_burn=1500,
                                  n_meas=3000, swap_every=8, meas_every=5, readout=lambda c: c.clone(), init_scale=0.3)
    samp = torch.stack(snaps, 0).reshape(-1, d)
    me = float(samp.mean(0).abs().max()); ve = float(((samp.var(0, unbiased=True) - 1.0 / A) / (1.0 / A)).abs().max())
    accs = info["per_rung_acc"]; unif = max(accs) < 0.75 and min(accs) > 0.35
    print(f"4) Gaussian correctness  max|mean|={me:.3f} max var-relerr={ve:.3f}")
    print(f"   per-rung MALA acc={accs}  uniform~0.574: {unif}")
    print(f"   round trips/replica={info['rt_per_replica']:.2f}  swap acc={info['swap_acc']:.2f}  (flow must be >0)")
    fails += [] if me < 0.1 and ve < 0.2 else ["gaussian correctness"]
    fails += [] if unif else ["acceptance not uniform"]
    fails += [] if info["rt_per_replica"] > 0.5 else ["no round-trip flow"]
    # init-independence plumbing: two inits -> same cold mean/var on the Gaussian
    g.manual_seed(7)
    _, sn2, _ = run_tempered(eg, d, E=64, ladder=lad, dev=dev, gen=g, n_adapt=1500, n_burn=1500, n_meas=3000,
                             swap_every=8, meas_every=5, readout=lambda c: c.clone(), init_scale=1.2)
    s2 = torch.stack(sn2, 0).reshape(-1, d); dvar = float(((s2.var(0) - samp.var(0)) / samp.var(0)).abs().max())
    print(f"   init-independence (Gaussian): max var diff between inits = {dvar:.3f} (->0)")
    fails += [] if dvar < 0.15 else ["init-independence plumbing"]
    print("\nVERIFY:", "PASS" if not fails else f"FAIL {fails}")
    return not fails


def full(probe=512, points=None, E=16, L=8, beta_hot=1.0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"; g = torch.Generator(device=dev); d_in = 8
    points = points or [(32, 24, 20.0), (64, 24, 20.0), (128, 24, 20.0), (64, 96, 20.0)]
    out = "genuine_tempered_results.jsonl"
    with open(out, "a") as fout:
        for (H, nd, bt) in points:
            P = sum(mlp_dims(H, d_in))
            recs = {}
            for tag, scale, seed in [("A", 0.1, 0), ("B", 1.0, 13)]:
                g.manual_seed(seed)
                Xtr = torch.randn(nd, d_in, generator=g, device=dev)
                D = torch.randn(probe, d_in, generator=g, device=dev)
                tW1 = 1.5 * torch.randn(H, d_in, generator=g, device=dev) / d_in ** 0.5
                tW2 = 1.5 * torch.randn(H, generator=g, device=dev) / H ** 0.5
                ytr = torch.tanh(Xtr @ tW1.t()) @ tW2
                U = mlp_energy_fn(Xtr, ytr, H, d_in, nd, 1e-2)
                ladder = [beta_hot * (bt / beta_hot) ** (r / (L - 1)) for r in range(L)]
                _, snaps, info = run_tempered(U, P, E, ladder, dev, g, n_adapt=2000, n_burn=5000, n_meas=6000,
                                              swap_every=10, meas_every=25, init_scale=scale,
                                              readout=lambda c: phi_flat(c, D, H, d_in))
                d_ = chi_diag(torch.stack(snaps, 1), H); d_.update(info)
                recs[tag] = d_
            # init-independence gate
            dq = abs(recs["A"]["q_mean"] - recs["B"]["q_mean"])
            both_mixed = recs["A"]["fs_rhat"] < 1.1 and recs["B"]["fs_rhat"] < 1.1
            flow = min(recs["A"]["rt_per_replica"], recs["B"]["rt_per_replica"]) > 1.0
            certified = both_mixed and flow and dq < 0.03
            rec = dict(H=H, n_data=nd, beta_target=bt, certified=bool(certified),
                       verdict=("certified" if certified else "INCONCLUSIVE"),
                       dq_between_inits=dq, A=recs["A"], B=recs["B"])
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"H{H} n{nd}: chiF A={recs['A']['chi_F']:.2f}/B={recs['B']['chi_F']:.2f} "
                  f"fsR A={recs['A']['fs_rhat']:.2f}/B={recs['B']['fs_rhat']:.2f} dq={dq:.3f} "
                  f"rt/rep={recs['A']['rt_per_replica']:.1f} -> {rec['verdict']}")
    print(f"DONE -> {out}. chi_F is admissible ONLY where verdict=certified; INCONCLUSIVE bounds reachability.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["verify", "full"], default="verify")
    ap.add_argument("--probe", type=int, default=512); a = ap.parse_args()
    (verify(a.probe) if a.mode == "verify" else full(a.probe))
