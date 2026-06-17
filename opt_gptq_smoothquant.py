"""LAYER 3b / SmoothQuant W4A4: does per-channel outlier migration recover the naive-4-bit-activation collapse?

Layer 3: naive per-token W4A4 = +37,332% (collapse), diagnosed as per-channel activation outliers. This is the
re-entry point for the program's SmoothQuant work (v0.2.4/2.5). SmoothQuant migrates outlier magnitude from
activations into weights: per input-channel j, s_j = a_j^alpha / w_j^(1-alpha) (a = activation amax, w = weight
amax); fold W'[:,j] = W[:,j]*s_j and divide x_j by s_j at inference. Flatter activations quantize at 4-bit; the
weights absorb the difficulty (and FP4-GPTQ handles weights well, as Layer 2 showed).

KEY EFFICIENCY: the smoothed Hessian needs no extra pass -- H'[i,j] = H[i,j]/(s_i s_j) exactly (since (x/s) outer
(x/s) scales H elementwise). So derive H' from the raw H.

CONTROL BEFORE SCAN: SmoothQuant is mathematically IDENTITY in full precision ((x/s)@(W*s)^T = x@W^T). So
smooth-A16 MUST reproduce +0.70%. If it doesn't, the fold is buggy -- do not trust the A4 number. Gate on it.

MERIT ATTRIBUTION: smooth-A4 vs naive-A4 (re-run here), paired bootstrap. Honest prior: SmoothQuant alone
recovers most of the collapse but is typically NOT sufficient for 4-bit activations (the frontier needs rotation
/ AWQ); report whatever it is. Per-linear smoothing (q/k/v get their own s -- valid, unfused variant), alpha=0.5.
[V-hw] OPT-2.7B, WikiText-2, fp32 master + fp32 gold, sequential GPTQ FP4-g128+act-order, held-out.
"""
import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; QDEV = "cuda:1"; L = 64; N_CAL = 192; N_EVAL = 256
WGRID = "fp4"; QMAX = 6.0; G = 128; ACT_ORDER = True; ALPHA = 0.5
FP4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
FP8 = torch.tensor(sorted({0.0, *[(m / 8 if e == 0 else (1 + m / 8)) * 2.0 ** (e - 7 if e else -6)
                                  for e in range(16) for m in range(8)
                                  if (m / 8 if e == 0 else (1 + m / 8)) * 2.0 ** (e - 7 if e else -6) <= 448.0]}),
                  dtype=torch.float32, device=DEV)
ACT = {"bits": None}


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
    s = getattr(self, "_sq_s", None)
    if s is not None:
        x = x / s                                                   # undo weight folding W'=W*s  =>  (x/s)@(W*s)^T = x@W^T
    if not getattr(self, "_aq", False) or ACT["bits"] is None:
        return torch.nn.functional.linear(x, W, b)
    x2 = x.reshape(-1, x.size(-1)).float()
    if ACT["bits"] == 8:
        sc = x2.abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        mids = (FP8[:-1] + FP8[1:]) / 2
        xq = torch.sign(x2) * FP8[torch.bucketize((x2 / sc).abs(), mids)] * sc
    else:
        sc = x2.abs().amax(1, keepdim=True).clamp_min(1e-12) / 7.0
        xq = torch.clamp(torch.round(x2 / sc), -7, 7) * sc
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


def reconstruct(smooth, cal):
    m = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    inps, kw0 = _catch_block0(m, cal)
    ACT["bits"] = None
    for blk in m.model.decoder.layers:
        U = _units(blk); H = {id(u): None for u in U}; cnt = {id(u): 0 for u in U}; amax = {id(u): None for u in U}
        def mk(u):
            def hook(mod, ip):
                x = ip[0].reshape(-1, ip[0].size(-1)).float().to(QDEV); h = x.t() @ x
                H[id(u)] = h if H[id(u)] is None else H[id(u)] + h; cnt[id(u)] += x.shape[0]
                a = x.abs().amax(0)                                  # per-input-channel activation amax
                amax[id(u)] = a if amax[id(u)] is None else torch.maximum(amax[id(u)], a)
            return u.register_forward_pre_hook(hook)
        hs = [mk(u) for u in U]
        for j in range(len(inps)):
            blk(inps[j], **kw0)
        for h in hs:
            h.remove()
        for u in U:
            Hq = H[id(u)] / max(cnt[id(u)], 1)
            W = u.weight.detach().to(QDEV)
            if smooth:
                a = amax[id(u)].clamp_min(1e-8)                     # (in,)
                w = W.abs().amax(0).clamp_min(1e-8)                 # (in,) per-input-channel weight amax
                s = (a.pow(ALPHA) / w.pow(1 - ALPHA)).clamp(1e-4, 1e4)   # SmoothQuant scale
                Hq = Hq / s.unsqueeze(0) / s.unsqueeze(1)           # H' = H/(s_i s_j), exact smoothed Hessian
                W = W * s.unsqueeze(0)                              # fold s into weight (out,in)*(1,in)
                u._sq_s = s.to(u.weight.device).to(u.weight.dtype)
            W4 = gptq(W, Hq, WGRID, QMAX, G, act_order=ACT_ORDER)
            u.weight.data = W4.to(u.weight.device).to(u.weight.dtype)
            u._aq = True
            H[id(u)] = None; del Hq, W, W4
        gc.collect(); torch.cuda.empty_cache()
        for j in range(len(inps)):
            o = blk(inps[j], **kw0)                                 # propagation: _sq_s set -> (x/s)@(W*s)^T exact
            inps[j] = o[0] if isinstance(o, (tuple, list)) else o
    return m


def paired_bootstrap(nll_a, nll_b, B=10000, seed=0):
    rng = np.random.default_rng(seed); n = len(nll_a); diff = nll_b - nll_a
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run():
    print("[OPT-2.7B SmoothQuant W4A4]  does outlier migration recover the naive 4-bit-activation collapse?\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = load_corpus(tok, N_CAL + N_EVAL); cal, ev = seqs[:N_CAL], seqs[N_CAL:]
    tgts = [s[0, 1:] for s in ev]

    nn.Linear.forward = _patched
    m0 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    ACT["bits"] = None; gold_ppl, _ = ppl(m0, ev, tgts); del m0; gc.collect(); torch.cuda.empty_cache()

    # smoothed arm: A16 (control), A8, A4
    ms = reconstruct(True, cal)
    ACT["bits"] = None; ps16, _ = ppl(ms, ev, tgts)
    ACT["bits"] = 8;    ps8, _ = ppl(ms, ev, tgts)
    ACT["bits"] = 4;    ps4, nll_s4 = ppl(ms, ev, tgts)
    del ms; gc.collect(); torch.cuda.empty_cache()

    # naive arm (re-run) for paired A4 attribution
    mn = reconstruct(False, cal)
    ACT["bits"] = 4; pn4, nll_n4 = ppl(mn, ev, tgts)
    del mn; gc.collect(); torch.cuda.empty_cache()
    nn.Linear.forward = _ORIG

    ctrl_ok = abs(100 * (ps16 / gold_ppl - 1) - 0.70) < 0.6        # smooth-A16 must reproduce +0.70% (fold is identity)
    md, lo, hi = paired_bootstrap(nll_n4, nll_s4)                   # smooth - naive at A4 (negative => recovery)
    print(f"  {'config':>20} {'PPL':>11} {'dPPL vs gold':>13}")
    print(f"  {'fp32 gold':>20} {gold_ppl:>11.3f} {'--':>13}")
    print(f"  {'smooth-A16 (control)':>20} {ps16:>11.3f} {100*(ps16/gold_ppl-1):>+12.2f}%   [identity check: {'OK' if ctrl_ok else 'FAIL'}]")
    print(f"  {'smooth-A8':>20} {ps8:>11.3f} {100*(ps8/gold_ppl-1):>+12.2f}%")
    print(f"  {'naive-A4 (Layer 3)':>20} {pn4:>11.3f} {100*(pn4/gold_ppl-1):>+12.2f}%")
    print(f"  {'smooth-A4':>20} {ps4:>11.3f} {100*(ps4/gold_ppl-1):>+12.2f}%")
    print(f"\n  recovery (smooth-A4 - naive-A4), log-PPL paired bootstrap:")
    print(f"    mean {md:>+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   (smooth {'<' if md < 0 else '>'} naive)")

    print("\n[VERDICT]")
    if not ctrl_ok:
        print(f"  CONTROL FAILED: smooth-A16 = {100*(ps16/gold_ppl-1):+.2f}% != +0.70%. The SmoothQuant fold is not")
        print(f"  identity in full precision -- a bug. Do NOT trust the A4 numbers; fix the fold first.")
    else:
        c4 = 100 * (ps4 / gold_ppl - 1)
        print(f"  CONTROL OK (smooth-A16 = +{100*(ps16/gold_ppl-1):.2f}%, fold verified identity).")
        if c4 < 10:
            print(f"  SmoothQuant RECOVERS W4A4 to {c4:+.2f}% (from +37,332% naive) -- outlier migration makes 4-bit")
            print(f"  activations deployable. The program's outlier tool is the key to the activation frontier.")
        elif ps4 < pn4 / 10:
            print(f"  SmoothQuant LARGELY recovers W4A4 ({c4:+.1f}%, from +37,332%) -- necessary, a huge improvement,")
            print(f"  but not yet near-lossless: the 4-bit activation frontier needs more (rotation/AWQ). Honest bound.")
        else:
            print(f"  SmoothQuant only partially helps W4A4 ({c4:+.1f}%) -- report straight; 4-bit acts need rotation.")
    print("\n[V-hw] RTX 5070 sm_120 + GTX 1660 Ti, OPT-2.7B, fp32 master, sequential GPTQ FP4-g128+act-order, WikiText-2.")
    return gold_ppl, ps16, ps8, ps4, pn4, (md, lo, hi)


if __name__ == "__main__":
    run()
