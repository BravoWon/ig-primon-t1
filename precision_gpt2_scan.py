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

GPT2_PATH = "C:/Users/JT-DEV1/Documents/gpt2-sm"   # locally downloaded gpt2-small (no HF network here)

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
    tok = GPT2TokenizerFast.from_pretrained(GPT2_PATH)
    model = GPT2Model.from_pretrained(GPT2_PATH).eval()
    nL = model.config.n_layer
    precisions = [("float32", torch.float32), ("bfloat16", torch.bfloat16)]
    pools = {name: [[] for _ in range(nL + 1)] for name, _ in precisions}
    for txt in TEXTS:
        ids = tok(txt, return_tensors="pt").input_ids[:, :64]
        h64 = hidden_states(model, ids, torch.float64)
        for name, dt in precisions:
            if name not in pools:
                continue
            try:
                hlow = hidden_states(model, ids, dt)
            except Exception as e:
                print(f"  [{name}] unavailable on this backend ({type(e).__name__}); skipped")
                pools.pop(name, None); continue
            for L in range(nL + 1):
                num = np.linalg.norm(hlow[L] - h64[L], axis=1)
                den = np.linalg.norm(h64[L], axis=1) + 1e-300
                pools[name][L].extend((num / den).tolist())
    model.to(torch.float64)                                            # leave model in a defined state

    print(f"\n[Stage-2] trained GPT-2-small, {nL} layers, {len(TEXTS)} texts; low-precision vs float64 (certified ref)")
    print(f"  (Stage-1 random-weight worst-case reference: +0.285/layer slope, 352x median<<mean)\n")
    RANDOM_SLOPE = 0.285
    for name in list(pools):
        per = pools[name]
        if not per[0]:
            continue
        mean_E = np.array([np.mean(e) for e in per]); med_E = np.array([np.median(e) for e in per])
        LL = np.arange(1, nL + 1)
        sl = np.polyfit(LL, np.log(med_E[1:] + 1e-300), 1)[0]
        growth = med_E[nL] / (med_E[1] + 1e-300); tail = mean_E[nL] / (med_E[nL] + 1e-300)
        p1 = sl < 0.5 * RANDOM_SLOPE
        print(f"  [{name:8s}] median E_cert: L1={med_E[1]:.2e} -> L{nL}={med_E[nL]:.2e}  (growth {growth:.2f}x over depth)")
        print(f"             log(median) slope = {sl:+.3f}/layer   mean/median@L{nL} = {tail:.1f}x   "
              f"-> P1 {'HOLDS (sub-exponential)' if p1 else 'FAILS -> F1 FIRES (~exponential)'}")
    print("\n[E-hw] one model, natural text. fp8/fp4 (the allocator regime, P3) is Stage 3. No claim beyond measured.")
    return True


if __name__ == "__main__":
    run()
