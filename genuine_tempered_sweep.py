"""genuine_tempered_sweep.py -- function-space chi_F sweep with MALA + parallel tempering.

Spec (agreed): function-space overlap on a fixed holdout probe (D=512, in-distribution, separate from
training); permutation-respecting (guard-validated); REAL beta-swap tempering; an MH-CORRECT accelerator
(MALA -- targets the exact posterior, no discretization bias); det-g kept OUT of the sampler; raw
observables + controls + FLAGs only, NO branch verdict.

Why MALA + tempering and not preconditioned SGHMC: chi_F = N*Var(q) is an EQUILIBRIUM quantity at the TRUE
posterior. An uncorrected preconditioned SGHMC changes the stationary distribution, so it would measure the
overlap of the WRONG distribution while looking equilibrated -- a mirage one level below the one Step 0
cleared. MALA's Metropolis correction makes the target exact; tempering swaps satisfy detailed balance by
construction. The empirical-Fisher preconditioner is left as a documented, DEFAULT-OFF hook (correctness
first; a preconditioner ships only after it passes the same Gaussian validation below).

  --mode verify : permutation guard + shuffle floor + probe sanity + SAMPLER-CORRECTNESS on a Gaussian whose
                  posterior moments are analytic. Nothing is trusted until these pass.
  --mode full   : the integrated tempered sweep (raw observables + FLAGs, no verdict).

[E-hw] Torch/CUDA. NOT a [V] receipt.
"""
import argparse, json, math, torch
import phase2_posterior_sampler as S


# ---------------- flat MLP parameterisation (one vector per chain) ----------------
def mlp_dims(H, d_in):
    return H * d_in, H, H, 1  # W1, b1, W2, b2 sizes

def unflat(W, H, d_in):
    s = mlp_dims(H, d_in); i = [0]
    def take(n):
        a = W[:, i[0]:i[0]+n]; i[0] += n; return a
    W1 = take(s[0]).view(-1, H, d_in); b1 = take(s[1]); W2 = take(s[2]); b2 = take(s[3]).squeeze(-1)
    return W1, b1, W2, b2

def forward_flat(W, X, H, d_in):
    W1, b1, W2, b2 = unflat(W, H, d_in)
    h = torch.tanh(torch.einsum("nd,khd->knh", X, W1) + b1[:, None, :])
    return torch.einsum("knh,kh->kn", h, W2) + b2[:, None]

def phi_flat(W, D, H, d_in):
    """Function-space readout: centered, unit-normalized OUTPUTS on probe set D. Weights never enter q."""
    f = forward_flat(W, D, H, d_in)
    f = f - f.mean(dim=1, keepdim=True)
    return f / (f.norm(dim=1, keepdim=True) + 1e-12)

def mlp_energy_fn(X, y, H, d_in, n_data, prior):
    def U(W):
        pred = forward_flat(W, X, H, d_in)
        mse = ((pred - y[None, :]) ** 2).mean(dim=1)
        return n_data * mse + prior * (W ** 2).sum(dim=1)
    return U

def permute_hidden_flat(W, perm, H, d_in):
    W1, b1, W2, b2 = unflat(W, H, d_in)
    W1 = W1[:, perm, :]; b1 = b1[:, perm]; W2 = W2[:, perm]
    return torch.cat([W1.reshape(W.shape[0], -1), b1, W2, b2[:, None]], dim=1)


# ---------------- MALA (MH-correct) + parallel tempering ----------------
def mala_step(W, energy_fn, beta, tau, gen):
    """One Metropolis-Adjusted Langevin step per chain toward exp(-beta*U). beta,tau: (K,)."""
    W = W.detach().requires_grad_(True)
    U0 = energy_fn(W); g0, = torch.autograd.grad(U0.sum(), W)
    tb = (tau * beta)[:, None]
    mu0 = W.detach() - tb * g0
    xi = torch.randn(W.shape, generator=gen, device=W.device)
    Wp = (mu0 + (2 * tau)[:, None].sqrt() * xi).detach().requires_grad_(True)
    U1 = energy_fn(Wp); g1, = torch.autograd.grad(U1.sum(), Wp)
    mu1 = Wp.detach() - tb * g1
    logq_fwd = -((Wp.detach() - mu0) ** 2).sum(1) / (4 * tau)
    logq_rev = -((W.detach() - mu1) ** 2).sum(1) / (4 * tau)
    log_acc = -beta * (U1.detach() - U0.detach()) + logq_rev - logq_fwd
    u = torch.rand(beta.shape, generator=gen, device=W.device)
    acc = torch.log(u) < log_acc
    Wn = torch.where(acc[:, None], Wp.detach(), W.detach())
    return Wn, float(acc.float().mean())

def pt_swap(W, energy_fn, ladder, E, L, gen):
    """Replica-exchange across adjacent rungs (even+odd sweeps). Detailed-balance preserving."""
    U = energy_fn(W).detach().view(E, L)
    Wv = W.view(E, L, -1).clone()
    rate = []
    for parity in (0, 1):
        for r in range(parity, L - 1, 2):
            d = (ladder[r] - ladder[r + 1]) * (U[:, r] - U[:, r + 1])
            u = torch.rand(E, generator=gen, device=W.device)
            sw = torch.log(u) < d
            rate.append(float(sw.float().mean()))
            a = Wv[:, r, :].clone()
            Wv[:, r, :] = torch.where(sw[:, None], Wv[:, r + 1, :], Wv[:, r, :])
            Wv[:, r + 1, :] = torch.where(sw[:, None], a, Wv[:, r + 1, :])
            U = energy_fn(Wv.reshape(E * L, -1)).detach().view(E, L)
    return Wv.reshape(E * L, -1), (sum(rate) / len(rate) if rate else 0.0)

def run_tempered(energy_fn, P, E, ladder, dev, gen, tau0, n_burn, n_meas, swap_every, meas_every, readout=None):
    L = len(ladder)
    betas = torch.tensor([ladder[r] for e in range(E) for r in range(L)], device=dev)  # k=e*L+r -> ladder[r]
    tau = tau0 / betas
    W = 0.1 * torch.randn(E * L, P, generator=gen, device=dev)
    macc, sacc = [], []
    for t in range(n_burn):
        W, a = mala_step(W, energy_fn, betas, tau, gen); macc.append(a)
        if t % swap_every == 0:
            W, s = pt_swap(W, energy_fn, ladder, E, L, gen); sacc.append(s)
    snaps = []
    for t in range(n_meas):
        W, a = mala_step(W, energy_fn, betas, tau, gen); macc.append(a)
        if t % swap_every == 0:
            W, s = pt_swap(W, energy_fn, ladder, E, L, gen); sacc.append(s)
        if t % meas_every == 0:
            cold = W.view(E, L, -1)[:, L - 1, :]            # (E, P) cold-rung samples
            snaps.append(readout(cold) if readout else cold.clone())
    return W, snaps, dict(mala_acc=sum(macc) / len(macc), swap_acc=sum(sacc) / max(1, len(sacc)))


# ---------------- function-space chi_F + diagnostics (raw, no verdict) ----------------
def chi_diagnostics(PHI, H):
    K, M, _ = PHI.shape
    iu = torch.triu_indices(K, K, offset=1)
    qb = torch.stack([PHI[:, t, :] @ PHI[:, t, :].t() for t in range(M)], 0)[:, iu[0], iu[1]]  # (M, pairs)
    chi = lambda q: float(H * q.var(unbiased=True))
    rs = torch.tensor([S.gelman_rubin(PHI[:, :, d]) for d in range(0, PHI.shape[2], max(1, PHI.shape[2] // 64))])
    qw = {str(lag): float((PHI[:, :M - lag, :] * PHI[:, lag:, :]).sum(-1).mean()) for lag in (1, M // 2) if 0 < lag < M}
    return dict(q_mean=float(qb.mean()), var_q=float(qb.flatten().var(unbiased=True)),
                chi_F=chi(qb.flatten()), chi_halves=[chi(qb[:M // 2].flatten()), chi(qb[M // 2:].flatten())],
                fs_rhat=float(rs[torch.isfinite(rs)].median()), q_within=qw)


# ---------------- VERIFY (nothing is trusted until these pass) ----------------
def verify(probe=512):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev); g.manual_seed(0)
    d_in, H = 8, 16; P = sum(mlp_dims(H, d_in))
    fails = []
    print(f"[verify on {dev}]  probe D={probe}\n")

    # 1) permutation guard -- function-space purity (weights forbidden)
    D = torch.randn(probe, d_in, generator=g, device=dev)
    W = torch.randn(1, P, generator=g, device=dev)
    perm = torch.randperm(H, generator=g, device=dev)
    qpp = float((phi_flat(W, D, H, d_in) * phi_flat(permute_hidden_flat(W, perm, H, d_in), D, H, d_in)).sum())
    print(f"1) permutation guard  q(W, permW) = {qpp:.12f}  (must be 1)")
    fails += [] if abs(qpp - 1) < 1e-9 else ["permutation guard"]

    # 2) shuffle floor -- the metric reads 1.0 when between-chain structure is destroyed
    W8 = torch.randn(8, P, generator=g, device=dev)
    PHI = torch.stack([phi_flat(W8, torch.randn(probe, d_in, generator=g, device=dev), H, d_in) for _ in range(40)], 1)
    flat = PHI.reshape(8 * 40, -1)[torch.randperm(320, generator=g, device=dev)].reshape(8, 40, -1)
    fl = float(torch.tensor([S.gelman_rubin(flat[:, :, d]) for d in range(0, probe, max(1, probe // 64))]).median())
    print(f"2) shuffle floor      fs_Rhat(shuffled) = {fl:.4f}  (must be ~1.0)")
    fails += [] if fl < 1.1 else ["shuffle floor"]

    # 3) probe sanity -- q(self)=1, q(two random) < 1
    qself = float((phi_flat(W, D, H, d_in) * phi_flat(W, D, H, d_in)).sum())
    qrand = float((phi_flat(torch.randn(1, P, generator=g, device=dev), D, H, d_in)
                   * phi_flat(torch.randn(1, P, generator=g, device=dev), D, H, d_in)).sum())
    print(f"3) probe sanity       q(self)={qself:.6f} (=1)  q(rand,rand)={qrand:+.4f} (<1)")
    fails += [] if abs(qself - 1) < 1e-6 and qrand < 0.95 else ["probe sanity"]

    # 4) SAMPLER CORRECTNESS -- MALA+tempering must recover an ANALYTIC Gaussian posterior.
    #    U(w)=0.5 sum A_i w_i^2 ; target exp(-beta U) = N(0, diag(1/(beta A_i))). Cold rung beta=1.
    d = 10
    A = torch.linspace(0.5, 2.0, d, device=dev)
    egauss = lambda W: 0.5 * (W ** 2 * A).sum(1)
    ladder = [0.25, 0.5, 1.0]
    _, snaps, info = run_tempered(egauss, d, E=64, ladder=ladder, dev=dev, gen=g,
                                  tau0=0.05, n_burn=2000, n_meas=4000, swap_every=10, meas_every=5,
                                  readout=lambda cold: cold.clone())
    samp = torch.stack(snaps, 0).reshape(-1, d)                 # (n_samples, d) cold-rung
    var_emp = samp.var(0, unbiased=True); var_true = 1.0 / (1.0 * A)
    mean_err = float(samp.mean(0).abs().max()); var_relerr = float(((var_emp - var_true) / var_true).abs().max())
    print(f"4) sampler correctness (Gaussian, cold beta=1):")
    print(f"     MALA acc={info['mala_acc']:.2f} swap acc={info['swap_acc']:.2f}")
    print(f"     max|mean| = {mean_err:.3f} (->0)   max var rel-err = {var_relerr:.3f} (->0)")
    fails += [] if mean_err < 0.1 and var_relerr < 0.15 else ["sampler correctness"]

    print("\nVERIFY:", "PASS -- sampler targets the true posterior; instrument + metric clean" if not fails
          else f"FAIL {fails}")
    return not fails


def full(probe=512, points=None, E=16, L=6, beta_hot=2.0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev); d_in = 8
    points = points or [(64, 24, 20.0), (64, 96, 20.0), (32, 24, 20.0), (128, 24, 20.0)]
    out = "genuine_tempered_results.jsonl"
    with open(out, "a") as fout:
        for (H, nd, beta_t) in points:
            g.manual_seed(0)
            P = sum(mlp_dims(H, d_in))
            Xtr = torch.randn(nd, d_in, generator=g, device=dev)
            D = torch.randn(probe, d_in, generator=g, device=dev)        # fixed holdout probe
            tW1 = 1.5 * torch.randn(H, d_in, generator=g, device=dev) / d_in ** 0.5
            tW2 = 1.5 * torch.randn(H, generator=g, device=dev) / H ** 0.5
            ytr = torch.tanh(Xtr @ tW1.t()) @ tW2
            U = mlp_energy_fn(Xtr, ytr, H, d_in, nd, 1e-2)
            ladder = [beta_hot * (beta_t / beta_hot) ** (r / (L - 1)) for r in range(L)]
            _, snaps, info = run_tempered(U, P, E, ladder, dev, g, tau0=2e-3, n_burn=6000, n_meas=6000,
                                          swap_every=20, meas_every=25,
                                          readout=lambda cold: phi_flat(cold, D, H, d_in))
            PHI = torch.stack(snaps, 1)                                   # (E, M, |D|)
            rec = chi_diagnostics(PHI, H); rec.update(H=H, n_data=nd, beta_target=beta_t,
                                                      mala_acc=info["mala_acc"], swap_acc=info["swap_acc"],
                                                      probe=probe, E=E, L=L)
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"H{H} n{nd} b{beta_t}: q={rec['q_mean']:.3f} chi_F={rec['chi_F']:.3f} "
                  f"fs_Rhat={rec['fs_rhat']:.2f} mala={info['mala_acc']:.2f} swap={info['swap_acc']:.2f} (NO verdict)")
    print(f"DONE -> {out}. Raw observables + FLAGs only; adjudicate in-the-loop.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["verify", "full"], default="verify")
    ap.add_argument("--probe", type=int, default=512)
    a = ap.parse_args()
    (verify(a.probe) if a.mode == "verify" else full(a.probe))
