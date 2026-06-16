"""All-day MAPPING run: collect the function-space overlap structure across the (H, n_data, beta) grid.

Protocol: PHASE1_protocol_genuine_branch_placement_v0_1.md.

This is the disciplined use of an unattended GPU: it SAMPLES (the expensive part) and MEASURES, and ships
NO branch verdict and NO equilibration verdict. Reason (the whole point of control-before-scan): a stuck
sampler fakes Branch 3 all day, so an unattended process must never *conclude* -- it collects measurements
+ their raw diagnostic inputs, and the judgment is adjudicated in-the-loop, clear-headed, when we're back.

What it discovered first (why this and not a scan): the branch question is only non-trivial in the
over-parametrized regime (data << params); realizable+identifiable gives q->1 trivially. So the grid spans
the degeneracy axis (n_data) as well as system size (H) and temperature (beta).

Per grid point it logs (JSONL, flushed per point): q_mean, Var(q), chi_F and its halves, q_between vs
q_within(lag) [the ergodicity signal], a function-space R-hat and a loss R-hat, tau -- and saves the K
chain-endpoint weights to a checkpoint for the connectivity cross-check (s.3) done in adjudication.

[E-hw] exploratory. Torch/CUDA. Incremental + interruption-safe. NO verdicts.
"""
import json, os, math, torch
import phase2_posterior_sampler as S

OUT = "phase2_mapping_results.jsonl"
CKPT_DIR = "phase2_ckpts"


def sample_with_snapshots(H, beta, device, gen, K=8, d_in=8, n_data=64,
                          eps=2e-4, n_burn=4000, n_meas=4000, meas_every=25, prior=1e-2,
                          teacher_scale=1.5, n_probe=256):
    Xtr = torch.randn(n_data, d_in, generator=gen, device=device)
    D = torch.randn(n_probe, d_in, generator=gen, device=device)
    tW1 = teacher_scale * torch.randn(H, d_in, generator=gen, device=device) / d_in**0.5
    tW2 = teacher_scale * torch.randn(H, generator=gen, device=device) / H**0.5
    ytr = torch.tanh(Xtr @ tW1.t()) @ tW2
    p = S.make_params(K, d_in, H, device, gen)
    noise = (2 * eps / beta) ** 0.5

    def step():
        for t in p.values():
            t.grad = None
        S.energy(p, Xtr, ytr, n_data, prior).backward()
        with torch.no_grad():
            for t in p.values():
                t.add_(t.grad, alpha=-eps).add_(torch.randn(t.shape, generator=gen, device=device), alpha=noise)

    for _ in range(n_burn):
        step()
    PHI, loss = [], []
    for i in range(n_meas):
        step()
        if i % meas_every == 0:
            with torch.no_grad():
                PHI.append(S.phi(p, D))                       # (K, |D|)
                loss.append(S.per_chain_loss(p, Xtr, ytr))
    PHI = torch.stack(PHI, dim=1)                              # (K, M, |D|)
    L = torch.stack(loss, dim=1)                               # (K, M)
    endpoints = {k: v.detach().cpu() for k, v in p.items()}
    return PHI, L, endpoints


def diagnostics(PHI, L, H, meas_every):
    K, M, _ = PHI.shape
    iu = torch.triu_indices(K, K, offset=1)
    # between-chain overlap, same time
    qb = []
    for t in range(M):
        Q = PHI[:, t, :] @ PHI[:, t, :].t()
        qb.append(Q[iu[0], iu[1]])
    qb = torch.stack(qb, dim=0)                                # (M, n_pairs)
    q_all = qb.flatten()
    chi = lambda q: float(H * q.var(unbiased=True))
    chi_F = chi(q_all)
    chi_1, chi_2 = chi(qb[:M // 2].flatten()), chi(qb[M // 2:].flatten())
    # within-chain overlap at a few lags (ergodicity signal)
    q_within = {}
    for lag in (1, M // 4, M // 2):
        if lag < 1 or lag >= M:
            continue
        s = sum(float((PHI[:, :M - lag, :] * PHI[:, lag:, :]).sum(-1).mean()) for _ in [0])
        q_within[lag] = s
    # function-space R-hat: per output-dim Gelman-Rubin on (K, M), median over dims
    rhats = []
    Dn = PHI.shape[2]
    for d in range(0, Dn, max(1, Dn // 64)):                    # subsample dims for speed
        rhats.append(S.gelman_rubin(PHI[:, :, d]))
    fs_rhat = float(torch.tensor(rhats).median())
    return dict(H=H, M=M, q_mean=float(q_all.mean()), var_q=float(q_all.var(unbiased=True)),
                chi_F=chi_F, chi_halves=[chi_1, chi_2],
                q_within={str(k): v for k, v in q_within.items()},
                fs_rhat=fs_rhat, loss_rhat=S.gelman_rubin(L), tau=S.tau_int(L.mean(0)))


def run(seeds=range(6), Hs=(16, 32, 64, 128), n_datas=(24, 48, 96), betas=(5.0, 20.0, 80.0)):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CKPT_DIR, exist_ok=True)
    g = torch.Generator(device=dev)
    idx, total = 0, len(list(seeds)) * len(Hs) * len(n_datas) * len(betas)
    with open(OUT, "a") as f:
        for seed in seeds:
            for H in Hs:
                for nd in n_datas:
                    for beta in betas:
                        g.manual_seed(seed)
                        PHI, L, end = sample_with_snapshots(H, beta, dev, g, n_data=nd)
                        rec = diagnostics(PHI, L, H, 25)
                        rec.update(seed=int(seed), n_data=nd, beta=beta, idx=idx)
                        torch.save(end, f"{CKPT_DIR}/pt_{idx:04d}.pt")
                        f.write(json.dumps(rec) + "\n"); f.flush()
                        idx += 1
                        print(f"[{idx}/{total}] seed{seed} H{H} n{nd} b{beta}: "
                              f"q={rec['q_mean']:.3f} chi_F={rec['chi_F']:.3f} "
                              f"fs_Rhat={rec['fs_rhat']:.2f} (NO verdict)", flush=True)
    print(f"DONE {idx} points -> {OUT} (+ {CKPT_DIR}/). No branch read; adjudicate in-the-loop.")


if __name__ == "__main__":
    run()
