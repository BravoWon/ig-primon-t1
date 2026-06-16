"""T1_precision_map v0.2 -- Stage 2: certified typical-case depth-error map on TRAINED GPT-2-small.

The climax test. Stage 1 certified that the WORST-CASE round-off is exponential in depth (random weights,
C2). H1/P1 predicts the MEASURED TYPICAL-CASE E_cert(L) on trained weights + natural text is
SUB-exponential -- the benign-direction / median<<mean mechanism (the §3.1 gate's sharpening).

float32 (the analyzed precision) vs float64 (the reference, licensed by the Stage-1 mpmath spot-cert on the
same op types). Per-token relative error of each hidden state, pooled over natural-text inputs -> a
typical-case distribution per layer. Verdict predicates:
  P1 holds  iff  log(median E_cert) vs L slope is SUB-exponential (<< Stage-1 random-weight +0.285/layer),
                 i.e. trained-weight typical-case attenuates where random-weight worst-case compounds.
  F1 fires  iff  the MEASURED typical-case curve is exponential (slope comparable to random weights).
Also reports mean vs median (the median<<mean slack) and per-layer growth. NO claim beyond what is measured.
[E-hw] CPU, torch float64 reference. Trained GPT-2-small (124M, 12 layers).
"""
import numpy as np, torch
from transformers import GPT2Model, GPT2TokenizerFast

torch.set_grad_enabled(False)

TEXTS = [
    "The history of numerical analysis begins with the study of round-off error in finite-precision arithmetic.",
    "She walked into the room and immediately noticed that something was different about the arrangement.",
    "Large language models compose many layers, each performing matrix multiplications and nonlinear maps.",
    "In the morning the fishermen returned with their catch and sold it at the market by the harbor.",
    "The proof proceeds by induction on the number of layers, bounding the error contributed at each step.",
    "Quantization reduces memory but can amplify errors at sharp attention logits and saturated activations.",
]


def hidden_states(model, ids, dtype):
    model.to(dtype)
    hs = model(ids, output_hidden_states=True).hidden_states          # tuple length L+1, each (1, T, d)
    return [h[0].to(torch.float64).numpy() for h in hs]


def run():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").eval()
    nL = model.config.n_layer
    per_layer = [[] for _ in range(nL + 1)]                            # per-token rel errors, pooled
    for txt in TEXTS:
        ids = tok(txt, return_tensors="pt").input_ids[:, :64]
        h64 = hidden_states(model, ids, torch.float64)
        h32 = hidden_states(model, ids, torch.float32)
        for L in range(nL + 1):
            num = np.linalg.norm(h32[L] - h64[L], axis=1)
            den = np.linalg.norm(h64[L], axis=1) + 1e-300
            per_layer[L].extend((num / den).tolist())                 # one sample per token
    mean_E = np.array([np.mean(e) for e in per_layer])
    med_E = np.array([np.median(e) for e in per_layer])

    print(f"[Stage-2] trained GPT-2-small, {nL} layers, {len(TEXTS)} texts, "
          f"{len(per_layer[0])} token-samples/layer; float32 vs float64 (certified ref)\n")
    print(f"  {'L':>3} {'mean E':>10} {'median E':>10} {'mean/median':>12}")
    for L in [0, 1, 3, 6, 9, nL]:
        print(f"  {L:>3} {mean_E[L]:>10.2e} {med_E[L]:>10.2e} {mean_E[L]/(med_E[L]+1e-300):>12.1f}")

    LL = np.arange(1, nL + 1)
    slope_mean = np.polyfit(LL, np.log(mean_E[1:] + 1e-300), 1)[0]
    slope_med = np.polyfit(LL, np.log(med_E[1:] + 1e-300), 1)[0]
    growth = med_E[nL] / (med_E[1] + 1e-300)
    print(f"\n  log(mean E)   vs L slope = {slope_mean:+.3f}/layer")
    print(f"  log(median E) vs L slope = {slope_med:+.3f}/layer   (Stage-1 random-weight was +0.285/layer)")
    print(f"  typical-case (median) growth over {nL} layers = {growth:.1f}x")

    RANDOM_SLOPE = 0.285
    p1 = slope_med < 0.5 * RANDOM_SLOPE                                # sub-exponential typical-case
    print("\n[P1] MEASURED typical-case E_cert(L) is",
          "SUB-EXPONENTIAL -> P1 HOLDS (trained-weight attenuation; worst-case exp does NOT govern typical inputs)"
          if p1 else
          "~exponential (slope comparable to random) -> F1 FIRES (attenuation does not transfer; important negative)")
    print(f"     median<<mean at L={nL}: {mean_E[nL]/(med_E[nL]+1e-300):.1f}x  (the benign-direction slack)")
    print("\n[E-hw] one model, float32-vs-float64; no claim beyond the measured curve. Stage-3 (allocator) is next.")
    return p1


if __name__ == "__main__":
    run()
