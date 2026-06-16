"""FORWARD (the use-aware axis): does activation-Hessian-aware quantization (GPTQ) analyze the weights better?

The refutation killed per-tensor ALLOCATION. The intuition: a weight's importance is relational to its USE --
which activation directions it is exercised along (the Hessian H = X^T X), not an isolated per-tensor scalar.
GPTQ is exactly that: it quantizes each weight to minimize OUTPUT error ||WX - WqX|| on the real token
activations, using the Hessian for error compensation.

Layer-level probe (tractable, correct, decisive for the PRINCIPLE before the full-model PPL build): for a set
of representative OPT layers, quantize W to W4 (per-output-channel) three ways and measure the RELATIVE OUTPUT
ERROR on calibration activations:
  RTN         : naive round-to-nearest (activation-blind).
  SmoothQuant : activation-magnitude scaling, then RTN (diagonal, activation-aware-lite).
  GPTQ        : full activation-Hessian, error-compensated (the use-aware method).
If GPTQ << RTN/SmoothQuant, the value is in use-aware QUANTIZATION (the intuition), not allocation.
[V-hw] OPT-2.7B, WikiText-2 calibration, FP4 weights.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; L = 64; N_CAL = 8
FP4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32, device=DEV)


def load_corpus(tok, n):
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * L:(i + 1) * L].unsqueeze(0) for i in range(n)]


def rtn_w4(W):
    s = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / 6.0
    mids = (FP4[:-1] + FP4[1:]) / 2
    return torch.sign(W) * FP4[torch.bucketize((W / s).abs(), mids)] * s


def smooth_w4(W, X):
    amax = X.abs().amax(0).clamp_min(1e-12); wmax = W.abs().amax(0).clamp_min(1e-12)
    s = ((amax ** 0.5) / (wmax ** 0.5)).clamp(1e-3, 1e3)
    Ws = W * s[None, :]
    return rtn_w4(Ws) / s[None, :]


def gptq_w4(W, X, damp=0.01):
    """Standard column-wise GPTQ to W4, per-output-channel scale, activation-Hessian error compensation."""
    W = W.clone().float(); out, inp = W.shape
    H = (X.t() @ X).float()                                            # (in, in)
    dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0
    H[torch.arange(inp), torch.arange(inp)] += damp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    s = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / 6.0           # per-output-channel scale (fixed)
    mids = (FP4[:-1] + FP4[1:]) / 2
    Q = torch.zeros_like(W)
    for i in range(inp):
        w = W[:, i]; d = Hinv[i, i]
        q = torch.sign(w) * FP4[torch.bucketize((w / s.squeeze(1)).abs(), mids)] * s.squeeze(1)
        Q[:, i] = q
        err = (w - q) / d
        W[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
    return Q


def run():
    print("[OPT-2.7B GPTQ layer probe]  is use-aware (Hessian) quantization better? relative output error, W4\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    cal = load_corpus(tok, N_CAL)
    m = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    layers = m.model.decoder.layers
    targets = [("L00.q_proj", layers[0].self_attn.q_proj), ("L00.fc1", layers[0].fc1),
               ("L02.fc2", layers[2].fc2), ("L08.fc1", layers[8].fc1),
               ("L15.q_proj", layers[15].self_attn.q_proj), ("L30.fc2", layers[30].fc2)]
    Xs = {n: [] for n, _ in targets}
    hk = []
    for n, mod in targets:
        hk.append(mod.register_forward_pre_hook((lambda n: (lambda m, i: Xs[n].append(
            i[0].reshape(-1, i[0].size(-1)).float())))(n)))
    for s in cal:
        m(s.to(DEV))
    for h in hk:
        h.remove()
    captured = [(n, mod.weight.detach().float().clone(), torch.cat(Xs[n], 0).clone()) for n, mod in targets]
    del m, layers, Xs; gc.collect(); torch.cuda.empty_cache()      # free the 10.9GB model -> room for fc2 Hessian

    print(f"  {'layer':>12} {'shape (out,in)':>16} {'RTN':>9} {'SmoothQuant':>12} {'GPTQ':>9} {'GPTQ/RTN':>9}")
    rr = []
    for n, W, X in captured:
        ref = W @ X.t(); rn = ref.norm()
        e_rtn = float(((rtn_w4(W) - W) @ X.t()).norm() / rn)
        e_sq = float(((smooth_w4(W, X) - W) @ X.t()).norm() / rn)
        e_gptq = float(((gptq_w4(W, X) - W) @ X.t()).norm() / rn)
        rr.append(e_gptq / e_rtn)
        print(f"  {n:>12} {str(tuple(W.shape)):>16} {e_rtn:>9.4f} {e_sq:>12.4f} {e_gptq:>9.4f} {e_gptq/e_rtn:>8.2f}x")

    print(f"\n[READ] mean GPTQ/RTN output-error ratio = {np.mean(rr):.2f}x  "
          f"(<<1 -> use-aware quantization is dramatically better at W4)")
    print("  If GPTQ collapses the output error vs RTN/SmoothQuant, the intuition holds: weights must be analyzed")
    print("  by their activation USE (the Hessian), not isolated per-tensor. Value is in the QUANTIZER, not the")
    print("  allocator -- the next test is whether GPTQ-W4 makes the full-model regime near-lossless (-> allocation moot).")
    print("\n[V-hw] RTX 5070 sm_120, OPT-2.7B, fp32 layers, WikiText-2 calibration. Per-layer W4 output error.")
    return rr


if __name__ == "__main__":
    run()
