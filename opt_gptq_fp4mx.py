"""LAYER 2 / FP4 hardware-native (MXFP4 block scaling): does the program's FP4 grid reach INT4 parity?

The program's original lever was FP4 E2M1 {0,.5,1,1.5,2,3,4,6} -- the hardware-native grid. FP4-per-channel gave
+58% (the rank-deficient artifact), and even FP4-g128 in that regime was +55-59% (grid-invariant => the GRID
wasn't the bottleneck, the Hessian rank was). With the bottleneck removed (full-rank H + sequential
reconstruction), this attributes the FP4 grid's TRUE cost: FP4-g128 vs INT4-g128 at matched group size and
matched 4-bit level count (both ~15 levels). MXFP4 = FP4 values with a per-128-column block scale -- the
"missing piece" the program suspected.

MERIT ATTRIBUTION (one knob = the grid): same fp32 master, same full-rank H, same group-128, same act-order
setting (ACT_ORDER, set from Layer 1's verdict), same eval. Arm A = INT4-g128; arm B = FP4-g128. Paired
bootstrap on per-seq log-PPL attributes the grid's marginal cost. If FP4-g128 ~ INT4-g128, the hardware-native
FP4 path is VIABLE with block scaling (the program's FP4 direction vindicated as an engineering, not a method,
gap). If FP4-g128 is materially worse, the 8-magnitude grid coarseness costs even with block scaling.
[V-hw] OPT-2.7B, WikiText-2, fp32 master + fp32 gold, sequential GPTQ, held-out.
"""
import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; QDEV = "cuda:1"; L = 64; N_CAL = 192; N_EVAL = 256  # powered eval (matches Layer 1) for clean attribution
G = 128
ACT_ORDER = True                                                     # Layer 1: act-order attributed -1.44% PPL (sig) -> best pipeline
FP4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
# Both grids are ~15-level 4-bit: INT4 symmetric -7..7 (15 vals); FP4 8 magnitudes x sign (15 vals).
CONFIGS = [("INT4-g128", "int4", 7.0), ("FP4-g128", "fp4", 6.0)]


def load_corpus(tok, n):
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * L:(i + 1) * L].unsqueeze(0) for i in range(n)]


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


def reconstruct(grid, qmax, cal, ev, tgts, want_gold=False):
    m = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    gold = ppl(m, ev, tgts) if want_gold else None
    inps, kw0 = _catch_block0(m, cal)
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
            W4 = gptq(u.weight.detach().to(QDEV), Hq, grid, qmax, G, act_order=ACT_ORDER)
            u.weight.data = W4.to(u.weight.device).to(u.weight.dtype)
            H[id(u)] = None; del Hq, W4
        gc.collect(); torch.cuda.empty_cache()
        for j in range(len(inps)):
            o = blk(inps[j], **kw0)
            inps[j] = o[0] if isinstance(o, (tuple, list)) else o
    p = ppl(m, ev, tgts); del m; gc.collect(); torch.cuda.empty_cache()
    return gold, p


def paired_bootstrap(nll_a, nll_b, B=10000, seed=0):
    rng = np.random.default_rng(seed); n = len(nll_a); diff = nll_b - nll_a
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run():
    print(f"[OPT-2.7B FP4 MXFP4]  hardware-native FP4 grid vs INT4 at matched g128 (act_order={ACT_ORDER})\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = load_corpus(tok, N_CAL + N_EVAL); cal, ev = seqs[:N_CAL], seqs[N_CAL:]
    tgts = [s[0, 1:] for s in ev]

    res = {}
    gold_ppl = None
    for k, (name, grid, qmax) in enumerate(CONFIGS):
        gold, (p, nll) = reconstruct(grid, qmax, cal, ev, tgts, want_gold=(k == 0))
        if k == 0:
            gold_ppl = gold[0]
        res[name] = (p, nll)

    p_int, nll_int = res["INT4-g128"]; p_fp4, nll_fp4 = res["FP4-g128"]
    md, lo, hi = paired_bootstrap(nll_int, nll_fp4)                  # fp4 - int4 (positive => FP4 costs)
    print(f"  {'config':>22} {'PPL':>9} {'dPPL vs gold':>13}")
    print(f"  {'fp32 gold':>22} {gold_ppl:>9.3f} {'--':>13}")
    print(f"  {'INT4-g128':>22} {p_int:>9.3f} {100*(p_int/gold_ppl-1):>+12.2f}%")
    print(f"  {'FP4-g128 (MXFP4)':>22} {p_fp4:>9.3f} {100*(p_fp4/gold_ppl-1):>+12.2f}%")
    print(f"\n  grid cost (FP4 - INT4), log-PPL paired bootstrap:")
    print(f"    mean {md:>+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]   (== {100*(np.exp(md)-1):+.3f}% PPL)")
    sig = (lo > 0) or (hi < 0)
    print(f"    significant at 95%: {sig}")

    print("\n[VERDICT]")
    if not sig:
        print(f"  FP4 PARITY: hardware-native FP4-g128 is statistically indistinguishable from INT4-g128. The FP4 path")
        print(f"  is VIABLE with block scaling -- block scaling was the missing engineering piece, not a method gap.")
    elif hi < 0:
        print(f"  FP4 BEATS INT4 ({100*(np.exp(md)-1):+.2f}%) -- the E2M1 grid's nonuniform levels fit the weights better.")
    else:
        print(f"  FP4 COSTS {100*(np.exp(md)-1):+.2f}% vs INT4 -- the 8-magnitude grid coarseness costs even with block")
        print(f"  scaling. INT4-g128 remains the near-lossless config; FP4 is a hardware-throughput tradeoff, not free.")
    print("\n[V-hw] RTX 5070 sm_120 + GTX 1660 Ti, OPT-2.7B, fp32 master, sequential GPTQ, WikiText-2. Held-out.")
    return gold_ppl, p_int, p_fp4, (md, lo, hi)


if __name__ == "__main__":
    run()
