"""T1_precision_map v0.2 -- Stage 3 (sim): the bake-off. Tests P2 and P3/F-app on trained GPT-2-small.

Weight-only mixed precision: every allocatable matmul weight starts at a BASE low tier (FP8 E4M3 or FP4 E2M1);
a budget of k tensors is promoted to the bf16 RECOMPUTE tier. float64 (un-quantized) is the CERTIFIED reference.
Three ways to spend the budget, all at matched k:
  uniform / no-info : a random k-subset promoted (averaged over seeds) -- spend the budget blindly.
  LAMP-style linear : the top-k tensors by the FP32 first-order sensitivity score.
  certified (A_cert): the top-k tensors by the certified (float64) true-KL sensitivity score.

P2  (pre-reg s.5): A_cert meets a fixed KL/flip budget at recompute cost <= uniform AND <= LAMP at matched k.
P3  (pre-reg s.5): the certified vs FP32-linear allocation DIFFERS at FP4, AGREES at FP8.
F-app (pre-reg s.6): the certified op-set EQUALS the FP32-linear op-set at matched budget -> certification adds
      nothing operational. PREDICTED to fire at FP8 (the redundancy control) and NOT at FP4 (the claim).

Admissible only after precision_allocator.gate() passes (control-before-scan). [E-hw] CPU, float64 reference.
This is a SIM (fake-quant, per-tensor, weight-only); it does NOT run the kappa_softmax/F2 attribution arm.
"""
import numpy as np
import torch
from transformers import GPT2TokenizerFast

import precision_allocator as PA

torch.set_grad_enabled(False)

TEXTS = [
    "The history of numerical analysis begins with the study of round-off error in finite-precision arithmetic.",
    "She walked into the room and immediately noticed that something was different about the arrangement.",
    "Large language models compose many layers, each performing matrix multiplications and nonlinear maps.",
    "In the morning the fishermen returned with their catch and sold it at the market by the harbor.",
    "The proof proceeds by induction on the number of layers, bounding the error contributed at each step.",
    "Quantization reduces memory but can amplify errors at sharp attention logits and saturated activations.",
]
K_GRID = [1, 2, 4, 8, 16, 24]
N_UNIFORM_SEEDS = 4


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.sqrt((ra @ ra) * (rb @ rb)) + 1e-30))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def run():
    print("[Stage-3 bake-off] running the gate first (control-before-scan)...\n")
    if not PA.gate():
        print("\nGATE FAILED -- bake-off NOT admissible. Stop."); return False
    print("\n" + "=" * 96 + "\n[Stage-3 bake-off] gate passed; weight-only mixed precision on trained GPT-2-small\n")

    tok = GPT2TokenizerFast.from_pretrained(PA.GPT2_PATH); tok.pad_token = tok.eos_token
    enc = tok(TEXTS, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    ids, mask = enc.input_ids, enc.attention_mask

    m = PA.load_model(torch.float64)
    mf32 = PA.load_model(torch.float32)
    tensors = PA.allocatable(m); tensors32 = PA.allocatable(mf32)
    originals = PA.snapshot(tensors)                                   # canonical fp64 originals
    nT = len(tensors)
    params = np.array([W.numel() for W in originals]); total_p = params.sum()
    ref = PA.logits_of(m, ids, mask)                                   # certified reference (un-quantized fp64)

    def measure(base, promoted):
        tiers = [base] * nT
        for j in promoted:
            tiers[j] = "bf16"
        PA.apply_assignment(tensors, originals, tiers)
        kl, fl = PA.kl_flip(ref, PA.logits_of(m, ids, mask), mask)
        PA.restore(tensors, originals)
        return kl, fl

    summary = {}
    for base in ("fp8", "fp4"):
        print(f"\n---------- base precision = {base.upper()} ----------")
        s_cert = PA.certified_scores(m, tensors, originals, base, ref, ids, mask)
        s_lin = PA.linear_scores(mf32, tensors32, originals, base, ids, mask)
        sp = spearman(s_cert, s_lin)

        kl0, fl0 = measure(base, set())                               # k=0, all-base (uniform precision)
        print(f"  k=0 (all-{base}, no recompute): KL={kl0:.3e}  flip={fl0:.3f}   [the uniform-precision baseline]")
        print(f"  {'k':>3} {'tens%':>6} {'param%':>7} {'random_KL':>11} {'lamp_KL':>11} {'cert_KL':>11} "
              f"{'cert<=lamp':>11} {'cert<=rand':>11} {'top-k Jacc':>11}")
        cle = 0; cre = 0
        for k in K_GRID:
            cert_set = PA.allocate(s_cert, k)
            lamp_set = PA.allocate(s_lin, k)
            rng = np.random.default_rng(0)
            rand_kls = []
            for _ in range(N_UNIFORM_SEEDS):
                rset = set(rng.choice(nT, size=k, replace=False).tolist())
                rand_kls.append(measure(base, rset)[0])
            rand_kl = float(np.mean(rand_kls))
            lamp_kl = measure(base, lamp_set)[0]
            cert_kl = measure(base, cert_set)[0]
            cle += cert_kl <= lamp_kl + 1e-15; cre += cert_kl <= rand_kl + 1e-15
            pp = params[list(cert_set)].sum() / total_p * 100
            print(f"  {k:>3} {100*k/nT:>5.0f}% {pp:>6.1f}% {rand_kl:>11.3e} {lamp_kl:>11.3e} {cert_kl:>11.3e} "
                  f"{str(cert_kl <= lamp_kl + 1e-15):>11} {str(cert_kl <= rand_kl + 1e-15):>11} "
                  f"{jaccard(cert_set, lamp_set):>11.2f}")
        # P3/F-app probe at a representative mid budget (k=8) + the full-vector rank correlation
        k_probe = 8
        cert8, lamp8 = PA.allocate(s_cert, k_probe), PA.allocate(s_lin, k_probe)
        fapp = (cert8 == lamp8)
        summary[base] = dict(spearman=sp, jacc8=jaccard(cert8, lamp8), fapp=fapp,
                             cle=cle, cre=cre, kl0=kl0)
        print(f"  P2({base.upper()}): cert<=lamp at {cle}/{len(K_GRID)} budgets, cert<=random at {cre}/{len(K_GRID)}")
        print(f"  scorer agreement: Spearman(cert,linear)={sp:+.3f} over all {nT} tensors; "
              f"top-{k_probe} sets {'IDENTICAL' if fapp else 'DIFFER'} (Jaccard={jaccard(cert8,lamp8):.2f})")

    # ---- verdicts
    print("\n" + "=" * 96 + "\n[VERDICTS]")
    p2 = all(summary[b]["cle"] >= len(K_GRID) - 1 and summary[b]["cre"] == len(K_GRID) for b in summary)
    print(f"P2 (allocator wins): certified KL <= LAMP-linear and <= random at matched budget, both bases "
          f"-> {'HOLDS' if p2 else 'PARTIAL/FAILS (see table)'}")
    f8, f4 = summary["fp8"], summary["fp4"]
    print(f"P3 (cert differs at FP4, agrees at FP8): "
          f"Spearman FP8={f8['spearman']:+.3f} (high=agree) vs FP4={f4['spearman']:+.3f} (lower=diverge) "
          f"-> {'HOLDS' if f8['spearman'] > f4['spearman'] + 1e-9 else 'NOT SHOWN'}")
    print(f"F-app (op-set identical -> certification adds nothing): "
          f"FP8 fires={f8['fapp']} (predicted True=redundancy control), "
          f"FP4 fires={f4['fapp']} (predicted False=the claim)")
    if f8["fapp"] and not f4["fapp"]:
        print("   -> F-app fires ONLY at FP8: certification has an operational edge exactly at FP4 (the build target).")
    elif not f4["fapp"] and not f8["fapp"]:
        print("   -> F-app fires at NEITHER: the certified pick differs from FP32-linear even at FP8 (stronger than predicted; report honestly).")
    else:
        print("   -> F-app fires at FP4: the certified instrument has NO operational edge there; retreat to the science (pre-reg s.7).")
    print("\n[E-hw] one model, 6 natural-text inputs, per-tensor weight-only fake-quant, float64 certified ref.")
    print("       Not the kappa_softmax/F2 arm, not hardware kernels, not >355M. No claim beyond measured.")
    return True


if __name__ == "__main__":
    run()
