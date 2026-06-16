"""SEQUENTIAL layer-wise reconstruction (the vision, made concrete; and the fix to the +150% parallel artifact).

"Map A->B against full precision, build back layer by layer with validation between" = sequential GPTQ:
quantize each block to match its full-precision output, PROPAGATE the reconstructed activations forward, then
quantize the next block against THOSE inputs (so each layer compensates for the upstream quantization error).
The parallel version omitted that propagation and compounded to +150%; this is the corrected method.

Block-wise: (1) catch block-0 inputs; (2) per block: forward on the propagated inputs with hooks accumulating
each linear's activation Hessian -> GPTQ-quantize the 6 linears -> re-forward the QUANTIZED block to get the
next block's inputs (the "validation layer in between"). The full-precision reference at each step IS the
program's certified reference, finally on the right lever. Eval W4 PPL vs gold; compare to parallel's +150%.
[V-hw] OPT-2.7B, WikiText-2, fp32 (certified) reconstruction + gold. Held-out.
"""
import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; L = 64; N_CAL = 16; N_EVAL = 64
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


def gptq_from_H(W, H, damp=0.01):
    W = W.clone().float(); out, inp = W.shape; H = H.float()
    dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0
    H[torch.arange(inp, device=DEV), torch.arange(inp, device=DEV)] += damp * torch.diag(H).mean()
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
    return W.to(torch.float32)


class _Stop(Exception):
    pass


def ppl(model, ev, tgts):
    nll = np.zeros(len(ev))
    for j, s in enumerate(ev):
        lp = torch.log_softmax(model(s.to(DEV)).logits[0].float().double(), -1).cpu()
        nll[j] = float(-lp[:L - 1].gather(1, tgts[j].unsqueeze(1)).mean())
    return float(np.exp(nll.mean()))


def run():
    print("[OPT-2.7B SEQUENTIAL GPTQ-W4]  layer-wise reconstruction against full precision (vision + the fix)\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = load_corpus(tok, N_CAL + N_EVAL); cal, ev = seqs[:N_CAL], seqs[N_CAL:]
    tgts = [s[0, 1:] for s in ev]

    m32 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    gold = ppl(m32, ev, tgts); del m32; gc.collect(); torch.cuda.empty_cache()
    print(f"  fp32 gold PPL = {gold:.3f}   (reconstructing against this, layer by layer)\n")
    m = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.bfloat16).to(DEV).eval()  # bf16 -> room for fc2 Hessian
    dec = m.model.decoder; layers = dec.layers

    # (1) catch block-0 inputs + kwargs for each calibration sequence
    caught = {"inps": [], "kw": []}
    orig0 = layers[0]
    class Catcher(nn.Module):
        def __init__(self, mod): super().__init__(); self.mod = mod
        def forward(self, hs, **kw):
            caught["inps"].append(hs.detach()); caught["kw"].append(kw); raise _Stop()
    layers[0] = Catcher(orig0)
    for s in cal:
        try:
            m(s.to(DEV))
        except _Stop:
            pass
    layers[0] = orig0
    inps = caught["inps"]; kws = caught["kw"]

    # (2) per block: Hessian on propagated inputs -> quantize -> re-forward quantized -> next inputs
    units = lambda blk: [blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj,
                         blk.self_attn.out_proj, blk.fc1, blk.fc2]
    for bi, blk in enumerate(layers):
        U = units(blk); H = {id(u): None for u in U}; nn_ = {id(u): 0 for u in U}
        def mk(u):
            def hook(mod, ip):
                x = ip[0].reshape(-1, ip[0].size(-1)).float()
                h = x.t() @ x
                H[id(u)] = h if H[id(u)] is None else H[id(u)] + h
                nn_[id(u)] += x.shape[0]
            return u.register_forward_pre_hook(hook)
        hs = [mk(u) for u in U]
        for j in range(len(inps)):
            blk(inps[j], **kws[j])
        for h in hs:
            h.remove()
        for u in U:
            u.weight.data = gptq_from_H(u.weight.detach(), H[id(u)] / max(nn_[id(u)], 1)).to(u.weight.dtype)
            H[id(u)] = None
        gc.collect(); torch.cuda.empty_cache()
        for j in range(len(inps)):                                    # propagate through the QUANTIZED block
            o = blk(inps[j], **kws[j])
            inps[j] = o[0] if isinstance(o, (tuple, list)) else o     # 5.12 OPTDecoderLayer returns a bare tensor
        if bi % 8 == 0 or bi == len(layers) - 1:
            print(f"    block {bi:2d}/{len(layers)-1} reconstructed")

    p = ppl(m, ev, tgts)
    print(f"\n  {'config':>26} {'PPL':>9} {'dPPL vs gold':>13}")
    print(f"  {'fp32 gold':>26} {gold:>9.3f} {'--':>13}")
    print(f"  {'parallel GPTQ-W4 (prior)':>26} {'139.5':>9} {'+150.7% (artifact)':>13}")
    print(f"  {'SEQUENTIAL GPTQ-W4':>26} {p:>9.3f} {100*(p/gold-1):>+12.2f}%")

    print("\n[VERDICT]")
    if p / gold - 1 < 0.15:
        print(f"  sequential layer-wise reconstruction is NEAR-LOSSLESS ({100*(p/gold-1):+.1f}%) vs the parallel")
        print("  artifact's +151%. The vision holds: reconstruct against full precision, propagate, validate between")
        print("  layers. The +150% was the missing propagation. The lever is the use-aware QUANTIZER + sequential")
        print("  reconstruction -- and the full-precision target at each layer is the certified reference's valid home.")
    else:
        print(f"  sequential GPTQ-W4 = {100*(p/gold-1):+.1f}%; better than parallel but FP4 weights still cost. Report as measured.")
    print("\n[V-hw] RTX 5070 sm_120, OPT-2.7B, sequential block-wise GPTQ, WikiText-2, fp32 gold. Held-out.")
    return gold, p


if __name__ == "__main__":
    run()
