"""Phase-1 section 2: the equilibration control (GPU SGLD), built BEFORE any scan ships a verdict.

Protocol: PHASE1_protocol_genuine_branch_placement_v0_1.md, section 2 (+ uses 1 and 3).

The killer confound: a stuck sampler reports low Var(q) -> fake "chi_F saturates -> vacuous", which
confirms the Branch-3 prior FOR THE WRONG REASON. So this module's job is NOT to read a branch -- it is
to MEASURE, and to refuse a reading when the chains are not equilibrated.

  K parallel SGLD chains on the GPU (teacher-student posterior at inverse-temperature beta).
  Per (H, beta) it returns: chi_F = N_eff*Var(q) (function-space overlap, protocol s.1),
  the equilibration verdict (Gelman-Rubin R-hat on per-chain loss; integrated autocorr time tau;
  first/second-half stability of chi_F), and -- HARD RULE -- flags INCONCLUSIVE (NOT Branch 3) when
  R-hat or tau or the halves fail.

  validate(): proves the diagnostic separates a KNOWN-equilibrated run (warm/small) from a KNOWN-stuck
  run (cold/large/short) -- the control on the control -- before any unattended sweep.

[E-hw] exploratory. Torch/CUDA. NOT a [V] receipt; ships measurements + control verdicts, never a branch.
"""
import torch


def make_params(K, d_in, H, device, gen):
    """K independent MLP replicas as batched leaf tensors. y = W2.tanh(W1 x + b1) + b2."""
    p = dict(
        W1=(torch.randn(K, H, d_in, generator=gen, device=device) / d_in**0.5).requires_grad_(True),
        b1=(0.1 * torch.randn(K, H, generator=gen, device=device)).requires_grad_(True),
        W2=(torch.randn(K, H, generator=gen, device=device) / H**0.5).requires_grad_(True),
        b2=torch.zeros(K, device=device, requires_grad=True),
    )
    return p


def forward(p, X):
    """X: (n, d_in) -> (K, n) outputs, one per replica."""
    h = torch.tanh(torch.einsum("nd,khd->knh", X, p["W1"]) + p["b1"][:, None, :])
    return torch.einsum("knh,kh->kn", h, p["W2"]) + p["b2"][:, None]


def per_chain_loss(p, X, y):
    return ((forward(p, X) - y[None, :]) ** 2).mean(dim=1)            # (K,)


def energy(p, X, y, n_data, prior):
    sq = sum((t**2).sum() for t in (p["W1"], p["b1"], p["W2"], p["b2"]))
    # energy per chain = n*MSE + prior*||w||^2 ; sum over independent chains for one backward
    return n_data * per_chain_loss(p, X, y).sum() + prior * sq


def phi(p, D):
    """Function-space readout (protocol s.1): centered, unit-normalized outputs on probe set D -> (K, |D|)."""
    f = forward(p, D)
    f = f - f.mean(dim=1, keepdim=True)
    return f / (f.norm(dim=1, keepdim=True) + 1e-12)


def pairwise_q(p, D):
    """All off-diagonal function-space overlaps between the K replicas -> (n_pairs,)."""
    F = phi(p, D)                                   # (K, |D|)
    Q = F @ F.t()                                   # (K, K), q(a,b)
    K = Q.shape[0]
    iu = torch.triu_indices(K, K, offset=1)
    return Q[iu[0], iu[1]]


def gelman_rubin(traj):
    """R-hat on a per-chain scalar trajectory traj: (K, M). ~1 = mixed; >1.1 = not."""
    K, M = traj.shape
    chain_mean = traj.mean(dim=1)
    chain_var = traj.var(dim=1, unbiased=True)
    W = chain_var.mean()
    B = M * chain_mean.var(unbiased=True)
    if W <= 0:
        return float("inf")
    var_hat = (M - 1) / M * W + B / M
    return float((var_hat / W).clamp(min=0).sqrt())


def tau_int(x, c=5.0):
    """Integrated autocorrelation time of a 1-D series (Sokal window)."""
    x = x - x.mean()
    n = x.numel()
    if float((x**2).mean()) == 0:
        return 1.0
    var = (x**2).mean()
    tau, M = 1.0, 1
    for k in range(1, n):
        rho = float((x[:-k] * x[k:]).mean() / var)
        tau += 2 * rho
        if k >= c * tau:
            break
        M = k
    return max(tau, 1.0)


def sample(H, beta, device, gen, K=8, d_in=8, n_data=512, n_probe=256,
           eps=2e-4, n_burn=4000, n_meas=4000, meas_every=20, prior=1e-2,
           teacher_scale=1.5, data=None):
    """K SGLD chains at (H, beta). Returns measurements + control diagnostics. No branch verdict."""
    if data is None:
        Xtr = torch.randn(n_data, d_in, generator=gen, device=device)
        D = torch.randn(n_probe, d_in, generator=gen, device=device)
        tW1 = teacher_scale * torch.randn(H, d_in, generator=gen, device=device) / d_in**0.5
        tW2 = teacher_scale * torch.randn(H, generator=gen, device=device) / H**0.5
        ytr = (torch.tanh(Xtr @ tW1.t()) @ tW2)
    else:
        Xtr, ytr, D = data
    p = make_params(K, d_in, H, device, gen)
    noise_scale = (2 * eps / beta) ** 0.5

    def step():
        for t in p.values():
            if t.grad is not None:
                t.grad = None
        energy(p, Xtr, ytr, n_data, prior).backward()
        with torch.no_grad():
            for t in p.values():
                t.add_(t.grad, alpha=-eps).add_(torch.randn(t.shape, generator=gen, device=device),
                                                alpha=noise_scale)

    for _ in range(n_burn):
        step()
    loss_traj, q_samples = [], []
    for i in range(n_meas):
        step()
        if i % meas_every == 0:
            with torch.no_grad():
                loss_traj.append(per_chain_loss(p, Xtr, ytr).detach())
                q_samples.append(pairwise_q(p, D).detach())
    L = torch.stack(loss_traj, dim=1)               # (K, M)
    Q = torch.stack(q_samples, dim=0).flatten()     # (M*n_pairs,)
    Qh = torch.stack(q_samples, dim=0)              # (M, n_pairs)
    M = Qh.shape[0]
    N_eff = H                                         # FSS-relevant size; absolute value calibrated vs controls
    chi = lambda q: float(N_eff * q.var(unbiased=True))
    chi_full = chi(Q)
    chi_1, chi_2 = chi(Qh[:M // 2].flatten()), chi(Qh[M // 2:].flatten())
    rhat = gelman_rubin(L)
    tau = tau_int(L.mean(dim=0))
    halves_ok = abs(chi_1 - chi_2) <= 0.25 * max(chi_full, 1e-9)
    equil = (rhat <= 1.1) and (M >= 50 * tau / meas_every) and halves_ok
    verdict = "ok" if equil else "INCONCLUSIVE"
    return dict(H=H, beta=float(beta), chi_F=chi_full, q_mean=float(Q.mean()),
                chi_halves=(chi_1, chi_2), rhat=rhat, tau=tau, M=M,
                equilibrated=equil, verdict=verdict)


def validate():
    """Control on the control: the diagnostic must call the easy run 'ok' and the stuck run INCONCLUSIVE."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev)
    print(f"[validate §2 equilibration diagnostic on {dev}]")
    print("  EASY (warm beta=2, small H=8, long run) -- should equilibrate -> 'ok':")
    g.manual_seed(0)
    e = sample(H=8, beta=2.0, device=dev, gen=g, eps=3e-4, n_burn=4000, n_meas=4000)
    print(f"    R-hat={e['rhat']:.3f} tau={e['tau']:.1f} chi_F={e['chi_F']:.3f} "
          f"halves={tuple(round(x,3) for x in e['chi_halves'])} -> {e['verdict']}")
    print("  STUCK (cold beta=200, large H=64, SHORT run) -- should NOT -> 'INCONCLUSIVE':")
    g.manual_seed(0)
    s = sample(H=64, beta=200.0, device=dev, gen=g, eps=5e-5, n_burn=200, n_meas=600, meas_every=10)
    print(f"    R-hat={s['rhat']:.3f} tau={s['tau']:.1f} chi_F={s['chi_F']:.3f} "
          f"halves={tuple(round(x,3) for x in s['chi_halves'])} -> {s['verdict']}")
    ok = e["verdict"] == "ok" and s["verdict"] == "INCONCLUSIVE"
    print("\n  §2 DIAGNOSTIC CONTROL:", "PASS" if ok else "FAIL (diagnostic cannot tell mixed from stuck)")
    return ok


if __name__ == "__main__":
    validate()
