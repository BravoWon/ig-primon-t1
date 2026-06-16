"""genuine_sweep_v4.py -- rigorous certificate: adversarially-separated inits + C-FSS + rt-spread.

Why v4: with beta_hot stuck at the floor (hot end NOT melted), the certificate rests entirely on
init-independence -- and two RANDOM inits can pass dq<0.03 by BASIN-COINCIDENCE. Worse, chi_F is
function-space (permutation-invariant), so two inits in symmetry-EQUIVALENT basins give identical chi_F
trivially (dq~0, zero ergodicity). That spuriously certifies a single-basin chi_F as a clean Branch signal,
hardest exactly at H128 (the multimodal regime).

Fix (no melt required): WARM-START the two ensembles from two INDEPENDENTLY-TRAINED solutions -- genuinely
different functions (verified: initial function-space overlap q_init < 0.9). Then dq<0.03 can pass ONLY if
the ladder actually carries the two distinct basins to the same cold distribution = real ergodicity =
direct connectivity probe. If they stay apart -> trapped/multimodal -> INCONCLUSIVE (correctly).

Two corroborating probes on the same 'whole posterior vs sub-region' question (both cheap, already in hand):
  - round-trip A/B SPREAD: on a shared ladder/landscape, two equilibrated inits should flow at similar rates;
    a persistent gap = init-dependent flow = the basin concern echoing. Gate also requires the spread closed.
  - C(beta) specific-heat PEAK FSS: energy-side criticality, complementary to overlap-side chi_F (different
    order parameters, not redundant). Logged with height/sharpness/location across H.

Certified(v4) := q_init<0.9 (separated) AND dq<0.03 (connected) AND min(rt_A,rt_B)>1 AND fs_Rhat<1.1 both
                 AND rt-spread |rtA-rtB|/max<0.5. Else INCONCLUSIVE. [E-hw] Torch/CUDA.
"""
import json, math, torch
import genuine_tempered_sweep as G
import genuine_ladder_tuning as LT
import genuine_sweep_v3 as V3

PRIOR = LT.PRIOR


def train_solution(X, y, H, d_in, nd, dev, seed, steps=4000, lr=0.05):
    """Full-batch GD to a low-loss interpolating solution (an adversarial warm-start point)."""
    g = torch.Generator(device=dev); g.manual_seed(seed)
    P = sum(G.mlp_dims(H, d_in)); W = (0.5 * torch.randn(1, P, generator=g, device=dev)).requires_grad_(True)
    for _ in range(steps):
        loss = (LT.lik(W, X, y, H, d_in, nd) / nd).sum()
        gr, = torch.autograd.grad(loss, W)
        with torch.no_grad():
            W -= lr * gr
        W.requires_grad_(True)
    Wd = W.detach()
    return Wd, float(LT.lik(Wd, X, y, H, d_in, nd) / nd)


def c_peak_fss(prof):
    """Extract specific-heat peak features: height, location, sharpness (peak/median, half-max width in ln-beta)."""
    Cs = [p["C"] for p in prof]; bs = [p["beta"] for p in prof]
    cmax = max(Cs); imax = Cs.index(cmax); cmed = sorted(Cs)[len(Cs) // 2]
    half = cmax - 0.5 * (cmax - min(Cs))
    lo = next((bs[i] for i in range(imax, -1, -1) if Cs[i] < half), bs[0])
    hi = next((bs[i] for i in range(imax, len(bs)) if Cs[i] < half), bs[-1])
    width = math.log(hi / lo) if lo > 0 and hi > lo else float("inf")
    return dict(peak_C=round(cmax, 2), peak_beta=round(bs[imax], 3),
                sharpness=round(cmax / cmed, 2), hm_width_lnbeta=round(width, 2))


def run_point(H, nd, dev, gen, beta_cold=20.0):
    d_in = 8; X, y, D = V3.make_data(H, nd, d_in, dev)
    bhot, melt, _ = V3.auto_beta_hot(X, y, H, d_in, nd, dev, gen)
    grid = [bhot * (beta_cold / bhot) ** (i / 23) for i in range(24)]
    prof = LT.measure_C(H, nd, grid, dev, gen)
    Ltot, K, ladder = LT.thermo_ladder(prof)
    cfss = c_peak_fss(prof)
    # adversarial inits: two independently-trained solutions
    WA, lossA = train_solution(X, y, H, d_in, nd, dev, seed=1)
    WB, lossB = train_solution(X, y, H, d_in, nd, dev, seed=99)
    q_init = float((G.phi_flat(WA, D, H, d_in) * G.phi_flat(WB, D, H, d_in)).sum())
    A = V3.lik_tempered(X, y, D, H, nd, ladder, dev, gen, 0.0, 0, W_init=WA)
    B = V3.lik_tempered(X, y, D, H, nd, ladder, dev, gen, 0.0, 1, W_init=WB)
    dq = abs(A["q_mean"] - B["q_mean"])
    rtA, rtB = A["rt_per_replica"], B["rt_per_replica"]
    spread = abs(rtA - rtB) / max(rtA, rtB, 1e-9)
    separated = q_init < 0.9
    flow = min(rtA, rtB) > 1.0
    mixed = A["fs_rhat"] < 1.1 and B["fs_rhat"] < 1.1
    certified = separated and (dq < 0.03) and flow and mixed and (spread < 0.5)
    return dict(H=H, n_data=nd, beta_hot=bhot, melt=melt, K=K, L_total=round(Ltot, 2),
                q_init=round(q_init, 3), separated=bool(separated), train_loss=[round(lossA, 4), round(lossB, 4)],
                dq=round(dq, 3), rt_A=round(rtA, 2), rt_B=round(rtB, 2), rt_spread=round(spread, 2),
                fs_rhat=[round(A["fs_rhat"], 2), round(B["fs_rhat"], 2)],
                chi_F=[round(A["chi_F"], 3), round(B["chi_F"], 3)], c_fss=cfss,
                certified=bool(certified), verdict="certified" if certified else "INCONCLUSIVE")


def full(points=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"; gen = torch.Generator(device=dev)
    points = points or [(32, 24), (64, 24), (128, 24), (64, 96)]
    with open("genuine_v4_results.jsonl", "a") as f:
        for (H, nd) in points:
            r = run_point(H, nd, dev, gen)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"H{H} n{nd}: q_init={r['q_init']:.2f}(sep={r['separated']}) "
                  f"chiF={r['chi_F']} dq={r['dq']:.3f} rt={r['rt_A']:.1f}/{r['rt_B']:.1f}(spread {r['rt_spread']:.2f}) "
                  f"fsR={r['fs_rhat']} | C-peak@{r['c_fss']['peak_beta']} sharp={r['c_fss']['sharpness']} "
                  f"-> {r['verdict']}")
    print("DONE -> genuine_v4_results.jsonl. Certified = inits separated AND connected AND mixed AND spread closed.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--one", action="store_true"); a = ap.parse_args()
    full(points=[(64, 24)] if a.one else None)
