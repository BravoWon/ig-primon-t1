"""DECISIVE: does use-aware (GPTQ) W4 quantization make the FULL MODEL near-lossless, closing the redirection?

Layer-level showed GPTQ ~7x better than RTN. The deployment question: does that compound to a near-lossless
full-model perplexity? Our naive RTN+SmoothQuant W4A8 gave +461% PPL (catastrophic). If GPTQ-W4 is near gold
PPL, then (a) the intuition is validated end-to-end, and (b) the lever is the use-aware QUANTIZER, not the
allocator -- a near-lossless quantizer leaves no room for allocation to matter (just like W8A8).

Parallel GPTQ: one calibration pass accumulates the activation Hessian H=X^T X per linear (big fc2 Hessians
offloaded to CPU), then each weight is GPTQ-W4 quantized using its H and baked in. Eval:
  GPTQ-W4A16 : GPTQ weights, full activations -- the clean test of weight-quant quality (lit: <1% on OPT).
  GPTQ-W4A8  : + per-token FP8 activations -- matches the sweep regime (RTN+SmoothQuant gave +461%).
[V-hw] OPT-2.7B, WikiText-2, fp32 gold, held-out.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; L = 64; N_CAL = 8; N_EVAL = 64
FP4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32, device=DEV)
FP8 = torch.tensor(sorted({0.0, *[(m / 8 if e == 0 else (1 + m / 8)) * 2.0 ** (e - 7 if e else -6)
                                  for e in range(16) for m in range(8)
                                  if (m / 8 if e == 0 else (1 + m / 8)) * 2.0 ** (e - 7 if e else -6) <= 448.0]}),
                  dtype=torch.float32, device=DEV)


def load_corpus(tok, n):
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * L:(i + 1) * L].unsqueeze(0) for i in range(n)]


def gptq_from_H(W, H, damp=0.01):
    W = W.clone().float(); out, inp = W.shape; H = H.float().to(DEV)
    dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0
    H[torch.arange(inp), torch.arange(inp)] += damp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    s = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / 6.0
    mids = (FP4[:-1] + FP4[1:]) / 2; ss = s.squeeze(1)
    for i in range(inp):
        w = W[:, i]; d = Hinv[i, i]
        q = torch.sign(w) * FP4[torch.bucketize((w / ss).abs(), mids)] * ss
        err = (w - q) / d
        W[:, i] = q
        W[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
    del H, Hinv; torch.cuda.empty_cache()
    return W


_A8 = {"on": False}
def patched(self, x):
    W = self.weight; b = self.bias
    if not _A8["on"]:
        return torch.nn.functional.linear(x, W, b)
    x2d = x.reshape(-1, x.size(-1))
    s = x2d.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / FP8[-1]
    mids = (FP8[:-1] + FP8[1:]) / 2
    xq = (torch.sign(x2d) * FP8[torch.bucketize((x2d.float() / s).abs(), mids)] * s).to(x2d.dtype)
    out = xq @ W.t()
    return (out + b if b is not None else out).view(*x.shape[:-1], W.size(0))


def ppl(model, ev, tgts):
    nll = np.zeros(len(ev))
    for j, s in enumerate(ev):
        lp = torch.log_softmax(model(s.to(DEV)).logits[0].float().double(), -1).cpu()
        nll[j] = float(-lp[:L - 1].gather(1, tgts[j].unsqueeze(1)).mean())
    return float(np.exp(nll.mean()))


def run():
    print("[OPT-2.7B full-model GPTQ-W4]  does use-aware quantization make the regime near-lossless?\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = load_corpus(tok, N_CAL + N_EVAL); cal, ev = seqs[:N_CAL], seqs[N_CAL:]
    tgts = [s[0, 1:] for s in ev]

    m32 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    gold = ppl(m32, ev, tgts); del m32; gc.collect(); torch.cuda.empty_cache()
    print(f"  fp32 gold PPL = {gold:.3f}")

    torch.nn.Linear.forward = patched
    mg = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.bfloat16).to(DEV).eval()
    U = OP.units(mg)
    H = {id(u): None for u in U}; n = {id(u): 0 for u in U}
    def mk(u):
        big = u.weight.shape[1] > 4096                                 # fc2: offload Hessian to CPU
        def hook(mod, inp):
            x = inp[0].reshape(-1, inp[0].size(-1)).float()
            h = (x.t() @ x)
            h = h.cpu() if big else h
            H[id(u)] = h if H[id(u)] is None else H[id(u)] + h
            n[id(u)] += x.shape[0]
        return u.register_forward_pre_hook(hook)
    hs = [mk(u) for u in U]
    for s in cal:
        mg(s.to(DEV))
    for h in hs:
        h.remove()

    for u in U:
        W4 = gptq_from_H(u.weight.detach(), H[id(u)] / max(n[id(u)], 1))
        u.weight.data = W4.to(u.weight.dtype)
        H[id(u)] = None
    gc.collect(); torch.cuda.empty_cache()

    _A8["on"] = False; p16 = ppl(mg, ev, tgts)
    _A8["on"] = True;  p8 = ppl(mg, ev, tgts)
    print(f"\n  {'config':>22} {'PPL':>9} {'dPPL vs gold':>13}")
    print(f"  {'fp32 gold':>22} {gold:>9.3f} {'--':>13}")
    print(f"  {'RTN+SmoothQuant W4A8':>22} {'(prior)':>9} {'+461% (broken)':>13}")
    print(f"  {'GPTQ-W4A16':>22} {p16:>9.3f} {100*(p16/gold-1):>+12.2f}%")
    print(f"  {'GPTQ-W4A8':>22} {p8:>9.3f} {100*(p8/gold-1):>+12.2f}%")

    print("\n[VERDICT]")
    if p16 / gold - 1 < 0.10:
        print(f"  use-aware (GPTQ) W4 is NEAR-LOSSLESS ({100*(p16/gold-1):+.1f}%) where naive RTN+SmoothQuant was +461%.")
        print("  The intuition is validated end-to-end: the lever is the use-aware QUANTIZER. And a near-lossless")
        print("  quantizer leaves no room for allocation -- closing the allocation arc for good (the value was never")
        print("  in WHICH weights to protect, but in HOW to quantize each given its activation use).")
    else:
        print(f"  GPTQ-W4 still degrades {100*(p16/gold-1):+.1f}% -- report as measured.")
    print("\n[V-hw] RTX 5070 sm_120, OPT-2.7B, parallel GPTQ, WikiText-2, fp32 gold. Held-out.")
    return gold, p16, p8


if __name__ == "__main__":
    run()
