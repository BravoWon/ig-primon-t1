"""Step 0 -- control the equilibration METRIC before the run-length sweep trusts it.

Question: is fs_R-hat > 1.1 at all 216 points REAL non-mixing, or a floor of the estimator
(finite M, K=8 chains, subsampled probe dims, unit-normalized phi)?

Decisive control = SHUFFLE TEST. Pool all K*M function-space snapshots and randomly reassign them
to K chains. This destroys between-chain structure, so the TRUE R-hat is exactly 1. Whatever the
estimator reports on shuffled data is its FLOOR at the run's real settings.
  observed >> floor  -> real between-chain structure = genuine non-mixing -> the run-length sweep is warranted.
  observed ~ floor ~ 1.3 -> the 1.1 gate is mis-set for this estimator -> recalibrate before burning GPU-hours.

Also reports the estimator with/without dim-subsampling and at K=8 vs K=16, and a length doubling on the
under-determined (non-trivial) config -- a first look at whether length pushes observed toward floor (mixing).

[E-hw] control on the diagnostic. Torch/CUDA.
"""
import torch
import phase2_posterior_sampler as S
import phase2_mapping_sweep as MS


def rhat_fs(PHI, subsample=True, reducer="median"):
    """Function-space R-hat: per-probe-dim Gelman-Rubin on (K,M), reduced over dims."""
    K, M, Dn = PHI.shape
    step = max(1, Dn // 64) if subsample else 1
    rs = torch.tensor([S.gelman_rubin(PHI[:, :, d]) for d in range(0, Dn, step)])
    rs = rs[torch.isfinite(rs)]
    return float(rs.median() if reducer == "median" else rs.mean())


def shuffle_floor(PHI, gen, n_rep=5, **kw):
    """Estimator floor: R-hat on data with between-chain structure destroyed (true R-hat=1)."""
    K, M, Dn = PHI.shape
    flat = PHI.reshape(K * M, Dn)
    vals = []
    for _ in range(n_rep):
        perm = torch.randperm(K * M, generator=gen, device=PHI.device)
        vals.append(rhat_fs(flat[perm].reshape(K, M, Dn), **kw))
    t = torch.tensor(vals)
    return float(t.mean()), float(t.std())


def collect(H, beta, nd, dev, gen, K, n_burn, n_meas):
    PHI, L, _ = MS.sample_with_snapshots(H, beta, dev, gen, K=K, n_data=nd,
                                         n_burn=n_burn, n_meas=n_meas, meas_every=25)
    return PHI


def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev)
    print(f"[Step 0: fs_R-hat calibration on {dev}]  gate currently = 1.1\n")
    print(f"{'config':<34}{'len':>6}{'observed':>10}{'floor(shuffle)':>16}{'obs/floor':>11}  verdict")

    def line(tag, PHI, nb, nm):
        obs = rhat_fs(PHI)
        fl, sd = shuffle_floor(PHI, g)
        ratio = obs / fl if fl > 0 else float("inf")
        v = "REAL non-mixing" if obs > fl + 3 * sd and ratio > 1.15 else "AT FLOOR (gate artifact)"
        print(f"{tag:<34}{f'{nb+nm}':>6}{obs:>10.3f}{f'{fl:.3f}+-{sd:.3f}':>16}{ratio:>11.2f}  {v}")
        return obs, fl

    # C1: unique-solution regime (high n_data) -- chains SHOULD converge to ~one function -> low between-chain.
    g.manual_seed(0)
    P = collect(64, 20.0, 96, dev, g, 16, 4000, 4000)
    line("unique-soln  H64 n96 b20", P, 4000, 4000)

    # C2: under-determined (the non-trivial regime) at mapping length and 2x.
    g.manual_seed(0)
    P1 = collect(64, 20.0, 24, dev, g, 16, 4000, 4000)
    o1, f1 = line("under-det    H64 n24 b20", P1, 4000, 4000)
    g.manual_seed(0)
    P2 = collect(64, 20.0, 24, dev, g, 16, 8000, 8000)
    o2, f2 = line("under-det    H64 n24 b20 (2x len)", P2, 8000, 8000)

    print("\nestimator sensitivity (under-det 1x):")
    print(f"  subsample dims + median : {rhat_fs(P1):.3f}")
    print(f"  ALL dims      + median : {rhat_fs(P1, subsample=False):.3f}")
    print(f"  ALL dims      + mean   : {rhat_fs(P1, subsample=False, reducer='mean'):.3f}")
    print(f"\nlength trend (under-det): observed {o1:.3f} (1x) -> {o2:.3f} (2x); "
          f"floor {f1:.3f} -> {f2:.3f}   "
          f"[{'moving toward floor = mixing' if o2 < o1 - 0.05 else 'NOT closing gap = stuck/glassy candidate'}]")


if __name__ == "__main__":
    run()
