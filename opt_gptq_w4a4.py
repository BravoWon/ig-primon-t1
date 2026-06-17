"""LAYER 3 / W4A4: attribute the activation-precision cost on top of the best W4 model.

The weight lever is closed (INT4-g128 + act-order = near-lossless). The remaining throughput win lives in
ACTIVATION precision: W4A16 -> W4A8 -> W4A4. This attributes the marginal cost of each activation-precision drop
on a FIXED best-W4 model (one reconstruction, then evals differing only in activation quant), isolating
activation cost from weight quant.

Activation quant = per-token dynamic absmax (each token row gets its own scale): A8 = FP8 E4M3 (max 448),
A4 = INT4 symmetric (16 levels). Per-token scaling handles per-token magnitude variation but NOT per-CHANNEL
outliers -- which are the known killer for 4-bit activations. So the honest prior: W4A8 ~ free, W4A4 hurts, and
the residual is the per-channel-outlier difficulty the program's SmoothQuant work (v0.2.4/2.5) was built to
migrate. If W4A4 is bad, that is the diagnosis -- and the re-entry point for outlier handling.

MERIT ATTRIBUTION: GPTQ weight quant uses FULL-PRECISION activations (Hessian from true acts; weight-only GPTQ,
standard); activation quant applied only at eval. Paired bootstrap on per-seq log-PPL for each precision drop.
[V-hw] OPT-2.7B, WikiText-2, fp32 master + fp32 gold, sequential GPTQ (INT4-g128 + act-order), held-out.
"""
import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; QDEV = "cuda:1"; L = 64; N_CAL = 192; N_EVAL = 256
WGRID = "fp4"; QMAX = 6.0; G = 128; ACT_ORDER = True                 # best weight config from Layer 2: FP4-g128 + act-order
FP4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
FP8 = torch.tensor(sorted({0.0, *[(m / 8 if e == 0 else (1 + m / 8)) * 2.0 ** (e - 7 if e else -6)
                                  for e in range(16) for m in range(8)
                                  if (m / 8 if e == 0 else (1 + m / 8)) * 2.0 ** (e - 7 if e else -6) <= 448.0]}),
                  dtype=torch.float32, device=DEV)
ACT = {"bits": None}                                                 # None | 8 | 4 -- global activation-quant mode


def load_corpus(tok, n):
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * L:(i + 1) * L].unsqueeze(0) for i in range(n)]


_ORIG = nn.Linear.forward
def _patched(self, x):
    W = self.weight; b = self.bias
    if not getattr(self, "_aq", False) or ACT["bits"] is None:
        return _ORIG(self, x)
    x2 = x.reshape(-1, x.size(-1)).float()
    if ACT["bits"] == 8:
        s = x2.abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        mids = (FP8[:-1] + FP8[1:]) / 2
        xq = torch.sign(x2) * FP8[torch.bucketize((x2 / s).abs(), mids)] * s
    else:                                                            # 4-bit INT per-token
        s = x2.abs().amax(1, keepdim=True).clamp_min(1e-12) / 7.0
        xq = torch.clamp(torch.round(x2 / s), -7, 7) * s
    xq = xq.to(x.dtype).view(*x.shape)
    return torch.nn.functional.linear(xq, W, b)


def gptq(W, H, grid, qmax, Gp, damp=0.01, act_order=False):
    dev = W.device
    W = W.clone().float(); out, inp = W.shape; H = H.float()
    fp4 = FP4.to(dev); mids = (fp4[:-1] + fp4[1:]) / 2
    dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0
    invperm = None
    if act_order:
        perm = torch.argsort(torch.diagonal(H), descending=True)
        W = W[:, perm]; H = H[perm][:, perm]; invperm = torch.argsort(perm)
    H[torch.arange(inp, device=dev), torch.arange(inp, device=dev)] += damp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Gp = Gp if Gp > 0 else inp
    ss = None
    for i in range(inp):
        if i % Gp == 0:
            ss = (W[:, i:min(i + Gp, inp)].abs().amax(1).clamp_min(1e-12) / qmax)
        w = W[:, i]; d = Hinv[i, i]
        if grid == "int4":
            q = torch.clamp(torch.round(w / ss), -qmax, qmax) * ss
        else:
            q = torch.sign(w) * fp4[torch.bucketize((w / ss).abs(), mids)] * ss
        err = (w - q) / d
        W[:, i] = q
        W[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
    if invperm is not None:
        W = W[:, invperm]
    del H, Hinv; torch.cuda.empty_cache()
    return W


class _Stop(Exception):
    pass


def ppl(model, ev, tgts):
    nll = np.zeros(len(ev))
    for j, s in enumerate(ev):
        lp = torch.log_softmax(model(s.to(DEV)).logits[0].float().double(), -1).cpu()
        nll[j] = float(-lp[:L - 1].gather(1, tgts[j].unsqueeze(1)).mean())
    return float(np.exp(nll.mean())), nll


def _catch_block0(m, cal):
    layers = m.model.decoder.layers; orig0 = layers[0]
    caught = {"inps": [], "kw": []}
    class Catcher(nn.Module):
        def __init__(s, mod): super().__init__(); s.mod = mod
        def forward(s, hs, **kw):
            caught["inps"].append(hs.detach())
            if not caught["kw"]: caught["kw"].append(kw)
            raise _Stop()
    layers[0] = Catcher(orig0)
    for sq in cal:
        try:
            m(sq.to(DEV))
        except _Stop:
            pass
    layers[0] = orig0
    kw0 = dict(caught["kw"][0])
    for _k in list(kw0):
        if "cache" in _k or "past_key" in _k: kw0[_k] = None
        if _k == "use_cache": kw0[_k] = False
    return caught["inps"], kw0


def _units(blk):
    return [blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj, blk.self_attn.out_proj, blk.fc1, blk.fc2]


def reconstruct_w4(cal):
    """Build the best-W4 model (INT4-g128 + act-order), tag units for activation quant, return resident model."""
    m = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    inps, kw0 = _catch_block0(m, cal)
    ACT["bits"] = None                                              # full-precision acts during reconstruction (weight-only GPTQ)
    for blk in m.model.decoder.layers:
        U = _units(blk); H = {id(u): None for u in U}; cnt = {id(u): 0 for u in U}
        def mk(u):
            def hook(mod, ip):
                x = ip[0].reshape(-1, ip[0].size(-1)).float().to(QDEV); h = x.t() @ x
                H[id(u)] = h if H[id(u)] is None else H[id(u)] + h; cnt[id(u)] += x.shape[0]
            return u.register_forward_pre_hook(hook)
        hs = [mk(u) for u in U]
        for j in range(len(inps)):
            blk(inps[j], **kw0)
        for h in hs:
            h.remove()
        for u in U:
            Hq = H[id(u)] / max(cnt[id(u)], 1)
            W4 = gptq(u.weight.detach().to(QDEV), Hq, WGRID, QMAX, G, act_order=ACT_ORDER)
            u.weight.data = W4.to(u.weight.device).to(u.weight.dtype)
            u._aq = True                                            # this linear gets activation quant at eval
            H[id(u)] = None; del Hq, W4
        gc.collect(); torch.cuda.empty_cache()
        for j in range(len(inps)):
            o = blk(inps[j], **kw0)
            inps[j] = o[0] if isinstance(o, (tuple, list)) else o
    return m


def paired_bootstrap(nll_a, nll_b, B=10000, seed=0):
    rng = np.random.default_rng(seed); n = len(nll_a); diff = nll_b - nll_a
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run():
    print("[OPT-2.7B W4A4]  activation-precision cost on the best W4 model (FP4-g128 + act-order, +0.70%)\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = load_corpus(tok, N_CAL + N_EVAL); cal, ev = seqs[:N_CAL], seqs[N_CAL:]
    tgts = [s[0, 1:] for s in ev]

    nn.Linear.forward = _patched                                   # install activation-quant-capable forward
    m0 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    ACT["bits"] = None
    gold_ppl, _ = ppl(m0, ev, tgts); del m0; gc.collect(); torch.cuda.empty_cache()

    m = reconstruct_w4(cal)
    ACT["bits"] = None; p16, nll16 = ppl(m, ev, tgts)
    ACT["bits"] = 8;    p8,  nll8  = ppl(m, ev, tgts)
    ACT["bits"] = 4;    p4,  nll4  = ppl(m, ev, tgts)
    del m; gc.collect(); torch.cuda.empty_cache()
    nn.Linear.forward = _ORIG

    d8 = paired_bootstrap(nll16, nll8)                             # A8 - A16
    d4 = paired_bootstrap(nll8, nll4)                             # A4 - A8
    print(f"  {'config':>14} {'PPL':>9} {'dPPL vs gold':>13}")
    print(f"  {'fp32 gold':>14} {gold_ppl:>9.3f} {'--':>13}")
    print(f"  {'W4A16':>14} {p16:>9.3f} {100*(p16/gold_ppl-1):>+12.2f}%")
    print(f"  {'W4A8':>14} {p8:>9.3f} {100*(p8/gold_ppl-1):>+12.2f}%")
    print(f"  {'W4A4':>14} {p4:>9.3f} {100*(p4/gold_ppl-1):>+12.2f}%")
    print(f"\n  marginal activation-precision cost (paired bootstrap, log-PPL):")
    print(f"    A16->A8 : mean {d8[0]:>+.5f}  95% CI [{d8[1]:+.5f}, {d8[2]:+.5f}]  (== {100*(np.exp(d8[0])-1):+.3f}% PPL)")
    print(f"    A8 ->A4 : mean {d4[0]:>+.5f}  95% CI [{d4[1]:+.5f}, {d4[2]:+.5f}]  (== {100*(np.exp(d4[0])-1):+.3f}% PPL)")

    print("\n[VERDICT]")
    c8 = 100 * (p8 / gold_ppl - 1); c4 = 100 * (p4 / gold_ppl - 1)
    print(f"  W4A8 = {c8:+.2f}% (per-token FP8 activations on near-lossless W4).")
    if c4 > 10:
        print(f"  W4A4 = {c4:+.2f}% -- per-token activation quant collapses at 4-bit. Diagnosis: per-CHANNEL outliers,")
        print(f"  which per-token scaling cannot reach. This is the re-entry point for SmoothQuant/rotation (Layer 3b).")
    else:
        print(f"  W4A4 = {c4:+.2f}% -- per-token activation quant survives 4-bit better than expected; report straight.")
    print("\n[V-hw] RTX 5070 sm_120 + GTX 1660 Ti, OPT-2.7B, fp32 master, sequential GPTQ FP4-g128+act-order, WikiText-2.")
    return gold_ppl, p16, p8, p4, d8, d4


if __name__ == "__main__":
    run()
