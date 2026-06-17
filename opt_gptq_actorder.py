"""LAYER 1 / act-order: attribute the marginal merit of activation-ordered GPTQ (the known <2% refinement).

The validated pipeline gives INT4-g128 = +2.51% (certified-H, fp32 master). Literature reaches <2% on OPT; the
one refinement my hand-rolled GPTQ omits is ACT-ORDER (desc_act): quantize columns in order of DECREASING
diag(H) -- the highest-activation-energy columns first, so their quantization error is compensated against the
largest remaining weight budget. This isolates act-order's marginal contribution with everything else held fixed.

MERIT ATTRIBUTION (one knob, paired): same fp32 master, same full-rank Hessian (12,288 tokens), same INT4-g128,
same eval. Arm A = no act-order (reproduces +2.51%); arm B = + act-order. Paired bootstrap on per-seq log-PPL
gives a CI on the marginal improvement, so the value act-order adds is attributed, not asserted.
[V-hw] OPT-2.7B, WikiText-2, fp32 master + fp32 gold, sequential INT4-g128 GPTQ, held-out.
"""
import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; QDEV = "cuda:1"; L = 64; N_CAL = 192; N_EVAL = 256  # eval widened 64->256 to resolve a ~0.7% effect
QMAX = 7.0; G = 128


def load_corpus(tok, n):
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * L:(i + 1) * L].unsqueeze(0) for i in range(n)]


def gptq_int4(W, H, qmax, Gp, damp=0.01, act_order=False):
    """INT4 group-Gp GPTQ, optional act-order. Device-agnostic (runs where W/H live)."""
    dev = W.device
    W = W.clone().float(); out, inp = W.shape; H = H.float()
    dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0
    invperm = None
    if act_order:                                                    # desc_act: most-important columns first
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
        q = torch.clamp(torch.round(w / ss), -qmax, qmax) * ss
        err = (w - q) / d
        W[:, i] = q
        W[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
    if invperm is not None:
        W = W[:, invperm]                                            # back to original column order
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


def reconstruct(act_order, cal, ev, tgts, want_gold=False):
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
            W4 = gptq_int4(u.weight.detach().to(QDEV), Hq, QMAX, G, act_order=act_order)
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
    print("[OPT-2.7B act-order]  marginal merit of activation-ordered GPTQ (the <2% refinement)\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = load_corpus(tok, N_CAL + N_EVAL); cal, ev = seqs[:N_CAL], seqs[N_CAL:]
    tgts = [s[0, 1:] for s in ev]

    gold, (p_base, nll_base) = reconstruct(False, cal, ev, tgts, want_gold=True)
    gold_ppl, _ = gold
    _, (p_act, nll_act) = reconstruct(True, cal, ev, tgts, want_gold=False)

    md, lo, hi = paired_bootstrap(nll_base, nll_act)                 # act - base (negative => act-order improves)
    print(f"  {'config':>26} {'PPL':>9} {'dPPL vs gold':>13}")
    print(f"  {'fp32 gold':>26} {gold_ppl:>9.3f} {'--':>13}")
    print(f"  {'INT4-g128 (no act-order)':>26} {p_base:>9.3f} {100*(p_base/gold_ppl-1):>+12.2f}%")
    print(f"  {'INT4-g128 + act-order':>26} {p_act:>9.3f} {100*(p_act/gold_ppl-1):>+12.2f}%")
    print(f"\n  marginal (act-order - baseline), log-PPL paired bootstrap:")
    print(f"    mean {md:>+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]   (== {100*(np.exp(md)-1):+.3f}% PPL)")
    sig = (lo > 0) or (hi < 0)
    helps = sig and hi < 0
    print(f"    significant at 95%: {sig}  ({'act-order improves' if helps else ('act-order hurts' if sig else 'no significant effect')})")

    print("\n[VERDICT]")
    if helps:
        print(f"  ACT-ORDER ATTRIBUTED: marginal {100*(np.exp(md)-1):+.2f}% PPL, significant. Closes +{100*(p_base/gold_ppl-1):.2f}%")
        print(f"  -> +{100*(p_act/gold_ppl-1):.2f}% toward the literature <2%. The refinement's merit is measured, not assumed.")
    elif sig:
        print(f"  act-order significantly HURTS here ({100*(np.exp(md)-1):+.2f}%) -- report straight; investigate before adopting.")
    else:
        print(f"  act-order shows no significant marginal effect at this calibration ({100*(np.exp(md)-1):+.2f}%, CI straddles 0).")
    print("\n[V-hw] RTX 5070 sm_120 + GTX 1660 Ti, OPT-2.7B, fp32 master, sequential INT4-g128 GPTQ, WikiText-2. Held-out.")
    return gold_ppl, p_base, p_act, (md, lo, hi)


if __name__ == "__main__":
    run()
