"""ROTATION on OPT-2.7B: does an orthogonal rotation rescue the SEVERE W4A4 collapse (+37,332%)?

GPT-2 (v0.2.23): rotation rescued W4A4 from +6582% to +44%. OPT-2.7B is the model where the collapse was
ESTABLISHED and severe (+37,332%, v0.2.11) -- driven by its famous per-channel activation outliers, exactly what
SmoothQuant couldn't fix (v0.2.12). This is the real test. Per-linear orthogonal rotation Q on each linear's
input, folded into the weight (W' = W Q for nn.Linear y=xW^T), so rotated activations have no per-channel
outliers while the full-precision output is unchanged; the Hessian rotates analytically H' = Q^T H Q.

Paired naive (Q=I) vs rotated (Q=orthogonal), same FP4-g128+act-order weights, per-token INT4 activations.
CONTROL: rotated-W4A16 ~ naive-W4A16 (rotation folds away under full-precision activations). bf16 master for
memory headroom (rotation Q for fc2 is 10240x10240).
[V-hw] OPT-2.7B, WikiText-2 held-out, bf16 master, sequential GPTQ FP4-g128+act-order + per-token INT4 acts.
"""
import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; L = 64; N_CAL = 192; N_EVAL = 128                     # 192x64=12288 > 10240 -> full-rank fc2 Hessian
WGRID = "fp4"; QMAX = 6.0; G = 128; ACT_ORDER = True
FP4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
ACT = {"bits": None}; ROT = {}                                       # ROT: in-dim -> orthogonal Q (float32, cuda:0)


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
    Q = getattr(self, "_rotQ", None)
    if Q is not None:
        x = (x.float() @ Q).to(x.dtype)                              # rotate the in-dim (folded into W' = W Q)
    if not getattr(self, "_aq", False) or ACT["bits"] is None:
        return torch.nn.functional.linear(x, W, b)
    x2 = x.reshape(-1, x.size(-1)).float()                            # per-token INT4 on the (rotated) activations
    s = x2.abs().amax(1, keepdim=True).clamp_min(1e-12) / 7.0
    xq = (torch.clamp(torch.round(x2 / s), -7, 7) * s).to(x.dtype).view(*x.shape)
    return torch.nn.functional.linear(xq, W, b)


def gptq(W, H, grid, qmax, Gp, damp=0.01, act_order=False):
    dev = W.device; W = W.clone().float(); out, inp = W.shape; H = H.float()
    fp4 = FP4.to(dev); mids = (fp4[:-1] + fp4[1:]) / 2
    dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0
    invperm = None
    if act_order:
        perm = torch.argsort(torch.diagonal(H), descending=True); W = W[:, perm]; H = H[perm][:, perm]; invperm = torch.argsort(perm)
    H[torch.arange(inp, device=dev), torch.arange(inp, device=dev)] += damp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    ss = None
    for i in range(inp):
        if i % Gp == 0:
            ss = (W[:, i:min(i + Gp, inp)].abs().amax(1).clamp_min(1e-12) / qmax)
        w = W[:, i]; d = Hinv[i, i]
        q = torch.sign(w) * fp4[torch.bucketize((w / ss).abs(), mids)] * ss
        err = (w - q) / d; W[:, i] = q
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
    return float(np.exp(nll.mean()))


def _catch0(m, cal):
    layers = m.model.decoder.layers; orig0 = layers[0]; caught = {"inps": [], "kw": []}
    class Catcher(nn.Module):
        def __init__(s, mod): super().__init__(); s.mod = mod
        def forward(s, hs, **kw):
            caught["inps"].append(hs.detach())
            if not caught["kw"]: caught["kw"].append(kw)
            raise _Stop()
    layers[0] = Catcher(orig0)
    for sq in cal:
        try: m(sq.to(DEV))
        except _Stop: pass
    layers[0] = orig0
    kw0 = dict(caught["kw"][0])
    for _k in list(kw0):
        if "cache" in _k or "past_key" in _k: kw0[_k] = None
        if _k == "use_cache": kw0[_k] = False
    return caught["inps"], kw0


def _units(blk):
    return [blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj, blk.self_attn.out_proj, blk.fc1, blk.fc2]


def _getQ(d):
    if d not in ROT:
        ROT[d] = torch.linalg.qr(torch.randn(d, d, device=DEV))[0].float()   # random orthogonal
    return ROT[d]


def reconstruct(rotate, cal):
    m = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.bfloat16).to(DEV).eval()
    inps, kw0 = _catch0(m, cal); ACT["bits"] = None
    for blk in m.model.decoder.layers:
        U = _units(blk); H = {id(u): None for u in U}; cnt = {id(u): 0 for u in U}
        def mk(u):
            def hook(mod, ip):                                       # H on the PRE-rotation input
                x = ip[0].reshape(-1, ip[0].size(-1)).float(); h = x.t() @ x
                H[id(u)] = h if H[id(u)] is None else H[id(u)] + h; cnt[id(u)] += x.shape[0]
            return u.register_forward_pre_hook(hook)
        hs = [mk(u) for u in U]
        for j in range(len(inps)):
            blk(inps[j], **kw0)
        for h in hs:
            h.remove()
        for u in U:
            Hq = H[id(u)] / max(cnt[id(u)], 1); W = u.weight.detach().float()
            if rotate:
                Q = _getQ(W.shape[1]); Hq = Q.t() @ Hq @ Q; W = W @ Q   # fold: W'=WQ, H'=Q^T H Q
                u._rotQ = Q
            W4 = gptq(W, Hq, WGRID, QMAX, G, act_order=ACT_ORDER)
            u.weight.data = W4.to(u.weight.dtype); u._aq = True
            H[id(u)] = None; del Hq, W, W4
        gc.collect(); torch.cuda.empty_cache()
        for j in range(len(inps)):
            o = blk(inps[j], **kw0); inps[j] = o[0] if isinstance(o, (tuple, list)) else o
    return m


def run():
    print("[OPT-2.7B ROTATION-W4A4]  does rotation rescue the severe +37,332% collapse?\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = load_corpus(tok, N_CAL + N_EVAL); cal, ev = seqs[:N_CAL], seqs[N_CAL:]
    tgts = [s[0, 1:] for s in ev]
    nn.Linear.forward = _patched
    m0 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.bfloat16).to(DEV).eval()
    ACT["bits"] = None; gold = ppl(m0, ev, tgts); del m0; gc.collect(); torch.cuda.empty_cache()
    print(f"  bf16 gold PPL = {gold:.3f}\n")

    res = {}
    for arm, rot in [("naive (Q=I)", False), ("rotated (Q=orth)", True)]:
        ROT.clear()
        m = reconstruct(rot, cal)
        ACT["bits"] = None; p16 = ppl(m, ev, tgts)
        ACT["bits"] = 4;    p4 = ppl(m, ev, tgts)
        res[arm] = (p16, p4); del m; gc.collect(); torch.cuda.empty_cache()
        print(f"  {arm:>18}: W4A16 {p16:>9.3f} ({100*(p16/gold-1):+.2f}%)   W4A4 {p4:>11.3f} ({100*(p4/gold-1):+.2f}%)")

    print("\n[VERDICT]")
    n16, n4 = res["naive (Q=I)"]; r16, r4 = res["rotated (Q=orth)"]
    ctrl_ok = abs(r16 / n16 - 1) < 0.05
    print(f"  CONTROL rotated-W4A16 {r16:.3f} ~ naive-W4A16 {n16:.3f}: fold-identity {'OK' if ctrl_ok else 'FAIL'}")
    print(f"  naive W4A4 = {100*(n4/gold-1):+.0f}%   rotated W4A4 = {100*(r4/gold-1):+.0f}%   (rescue factor {n4/r4:.0f}x)")
    if ctrl_ok and r4 < n4 * 0.5:
        print("  -> ROTATION RESCUES the severe OPT collapse: per-channel outliers spread into the bulk, 4-bit acts")
        print("     survive where SmoothQuant could not. The W4A4 frontier mechanism validated on the real outlier model.")
    print("\n[V-hw] OPT-2.7B, WikiText-2, bf16 master, sequential GPTQ FP4-g128+act-order + per-token INT4 acts, random rotation.")
    return gold, res


if __name__ == "__main__":
    run()
