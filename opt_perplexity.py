"""GAP #6: does the +30% KL edge translate into a DEPLOYMENT metric (perplexity / flip-rate), or is it
angels on a pin?

KL(fp32 || deployed) is a structural-distortion abstraction. This harness translates it into the currencies
a deployer reads, for the certified vs practical allocations on held-out WikiText-2:
  - perplexity  PPL = exp(mean NLL of true next tokens); and the GAP to the fp32 gold PPL.
  - flip rate vs fp32: fraction of positions where the deployed argmax != the fp32 argmax (decisions changed).
  - KL recomputed here too, so all three currencies sit side by side.
Certified vs practical compared with paired bootstrap CIs on each.

PRE-REGISTERED EXPECTATION (stated before the run): the absolute KLs are tiny (~2-4e-3 nats), so the +30%
RELATIVE KL edge is ~1.2e-3 nats ~= exp(1.2e-3) ~= +0.1% PPL. The PPL difference is likely OPERATIONALLY
MARGINAL; flip-rate may be the more sensitive translation. A small/null PPL result is a VALID finding:
"structurally real in KL, marginal in deployment metric." [V-hw] OPT-2.7B, SmoothQuant, WikiText-2.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; ALPHA = 0.5; L = 64; FP8 = torch.float8_e4m3fn
N_CAL = 6; N_EVAL = 64; K = 48


def load_corpus(tok, n):
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * L:(i + 1) * L].unsqueeze(0) for i in range(n)]


def kl_cached(logp_ref, q):
    lpr = logp_ref.double(); lpq = torch.log_softmax(q.double(), -1).cpu()
    return (lpr.exp() * (lpr - lpq)).sum(-1).mean().item()


def boot_diff(a, b, B=4000, pct=False):
    """CI of mean(b)-mean(a) (b=practical, a=certified); if pct, as percent of mean(b)."""
    n = len(a); rng = np.random.default_rng(0); st = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        d = b[idx].mean() - a[idx].mean()
        st.append(d / max(b[idx].mean(), 1e-30) * 100 if pct else d)
    return float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def run():
    print(f"[OPT-2.7B perplexity/flip]  does the KL edge translate to deployment metrics? k={K}, {N_EVAL} seqs\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = load_corpus(tok, N_CAL + N_EVAL)
    cal, ev = seqs[:N_CAL], seqs[N_CAL:]
    torch.nn.Linear.forward = OP.patched_linear

    # Phase A: fp32 anchor -> cal log-softmax (for scoring) + eval log-softmax (for KL/flip/PPL gold)
    m32 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    refs_cal = [torch.log_softmax(m32(s.to(DEV)).logits[0].float(), -1).half().cpu() for s in cal]
    refs_ev = [torch.log_softmax(m32(s.to(DEV)).logits[0].float(), -1).half().cpu() for s in ev]
    del m32; gc.collect(); torch.cuda.empty_cache()
    fp32_argmax = [r.argmax(-1) for r in refs_ev]
    tgts = [s[0, 1:] for s in ev]
    gold_nll = np.array([float(-refs_ev[j][:L - 1].double().gather(1, tgts[j].unsqueeze(1)).mean()) for j in range(len(ev))])

    mg = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.bfloat16).to(DEV).eval()
    U = OP.units(mg)
    store = {}; hs = []
    def mk(u):
        def hook(mod, inp):
            a = inp[0].reshape(-1, inp[0].size(-1)).float().abs().amax(0)
            store[id(u)] = a if id(u) not in store else torch.maximum(store[id(u)], a)
        return u.register_forward_pre_hook(hook)
    for u in U:
        u._prec = "hi"; hs.append(mk(u))
    for s in cal:
        mg(s.to(DEV))
    for h in hs:
        h.remove()
    for u in U:
        amax = store[id(u)].clamp_min(1e-12); wmax = u.weight.float().abs().amax(0).clamp_min(1e-12)
        u._s = ((amax ** ALPHA) / (wmax ** (1 - ALPHA))).clamp(1e-3, 1e3).to(u.weight.dtype)

    impC = np.zeros(len(U)); impP = np.zeros(len(U)); floorA = 0.0
    for ci, s in enumerate(cal):
        ids = s.to(DEV)
        for u in U:
            u._prec = "hi"
        bf = mg(ids).logits[0].float(); logp_bf = torch.log_softmax(bf.double(), -1).cpu()
        floorA += kl_cached(refs_cal[ci], bf)
        for i in range(len(U)):
            U[i]._prec = "smooth"; cfg = mg(ids).logits[0].float(); U[i]._prec = "hi"
            impC[i] += kl_cached(refs_cal[ci], cfg)
            lpq = torch.log_softmax(cfg.double(), -1).cpu()
            impP[i] += (logp_bf.exp() * (logp_bf - lpq)).sum(-1).mean().item()
    impC = impC / len(cal) - floorA / len(cal); impP /= len(cal)
    cert = set(np.argsort(-impC)[:K].tolist()); prac = set(np.argsort(-impP)[:K].tolist())

    def metrics(keep):
        nll = np.zeros(len(ev)); flip = np.zeros(len(ev)); kl = np.zeros(len(ev))
        for j, s in enumerate(ev):
            ids = s.to(DEV)
            for t, u in enumerate(U):
                u._prec = "hi" if t in keep else "smooth"
            lg = mg(ids).logits[0].float()
            logp = torch.log_softmax(lg.double(), -1).cpu()
            nll[j] = float(-logp[:L - 1].gather(1, tgts[j].unsqueeze(1)).mean())
            flip[j] = float((lg.argmax(-1).cpu() != fp32_argmax[j]).float().mean())
            kl[j] = float((refs_ev[j].double().exp() * (refs_ev[j].double() - logp)).sum(-1).mean())
        return nll, flip, kl

    allocs = {"all-SmoothQuant": set(), "practical(bf16)": prac, "certified(fp32)": cert, "bf16-floor": set(range(len(U)))}
    M = {n: metrics(k) for n, k in allocs.items()}
    gold_ppl = float(np.exp(gold_nll.mean()))

    print(f"  fp32 gold perplexity = {gold_ppl:.3f}\n")
    print(f"  {'allocation':>16} {'PPL':>9} {'dPPL vs gold':>13} {'flip vs fp32':>13} {'mean KL':>11}")
    for n in ("bf16-floor", "all-SmoothQuant", "practical(bf16)", "certified(fp32)"):
        nll, flip, kl = M[n]; ppl = float(np.exp(nll.mean()))
        print(f"  {n:>16} {ppl:>9.4f} {100*(ppl/gold_ppl-1):>+12.2f}% {flip.mean():>12.2%} {kl.mean():>11.3e}")

    cn, cf, ck = M["certified(fp32)"]; pn, pf, pk = M["practical(bf16)"]
    ppl_edge = (np.exp(pn.mean()) / np.exp(cn.mean()) - 1) * 100
    lo_p, hi_p = boot_diff(cn, pn)                         # NLL diff -> exp for PPL ratio
    lo_p, hi_p = (np.exp(lo_p) - 1) * 100, (np.exp(hi_p) - 1) * 100
    flip_edge = (pf.mean() - cf.mean()) * 100
    lo_f, hi_f = boot_diff(cf, pf); lo_f, hi_f = lo_f * 100, hi_f * 100
    kl_edge = (pk.mean() - ck.mean()) / pk.mean() * 100

    print(f"\n  certified vs practical (the edge in each currency):")
    print(f"    KL:    {kl_edge:+.1f}%   (the established structural edge)")
    print(f"    PPL:   {ppl_edge:+.3f}%  95% CI [{lo_p:+.3f}, {hi_p:+.3f}]   {'significant' if lo_p > 0 else 'NOT significant'}")
    print(f"    flip:  {flip_edge:+.2f} pts  95% CI [{lo_f:+.2f}, {hi_f:+.2f}]   {'significant' if lo_f > 0 else 'NOT significant'}")

    print("\n[VERDICT]")
    if lo_p > 0 and ppl_edge > 0.5:
        print(f"  the KL edge TRANSLATES: certified improves perplexity by {ppl_edge:.2f}% (CI>0). Real deployment value.")
    elif lo_f > 0:
        print(f"  PPL edge is marginal ({ppl_edge:+.3f}%) but FLIP rate is significantly better ({flip_edge:+.2f} pts):")
        print(f"  the edge changes real decisions even if aggregate perplexity barely moves. Partial translation.")
    else:
        print(f"  the +{kl_edge:.0f}% KL edge does NOT translate to a meaningful deployment metric (PPL {ppl_edge:+.3f}%,")
        print(f"  flip {flip_edge:+.2f} pts, CIs span 0). As pre-registered: structurally real, operationally marginal")
        print(f"  at W8A8 fidelity on OPT-2.7B. The KL tournament was, at this fidelity, angels on a pin.")
    print("\n[V-hw] RTX 5070 sm_120, OPT-2.7B, SmoothQuant, WikiText-2. fp32 gold. Held-out, paired bootstrap.")
    return M


if __name__ == "__main__":
    run()
