"""DECISIVE for W4A8: is the +1.8% edge a deployable win, or just 'better garbage' in a broken regime?

The sweep showed W4A8 all-low = +461% PPL (catastrophic) with a significant +1.8% cert-vs-practical edge at
k=48. But +1.8% between two broken models is still broken. This reports ABSOLUTE perplexity for certified,
practical, and all-low across a BUDGET sweep (k = 48 -> 176), against gold. The questions:
  (i) at what budget k does W4A8 cross into DEPLOYABLE territory (PPL near gold)?
  (ii) does the certified edge PERSIST into that deployable range, or only exist where the model is broken?
If W4A8 only becomes usable at k -> 192 (protect ~everything), the edge is an artifact of a broken regime and
there is no operational claim. If there is a usable budget where certified still beats practical -> real frontier.
[V-hw] OPT-2.7B, W4A8 (FP4 weights / FP8 activations) + SmoothQuant, WikiText-2, fp32 gold, held-out.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP
import opt_precision_sweep as SW

torch.set_grad_enabled(False)
DEV = "cuda:0"; ALPHA = 0.5; L = 64
N_CAL = 6; N_EVAL = 64; BUDGETS = (48, 96, 128, 160, 176)


def kl_c(lr, q):
    lq = torch.log_softmax(q.double(), -1).cpu()
    return (lr.double().exp() * (lr.double() - lq)).sum(-1).mean().item()


def boot(a, b, B=3000):
    n = len(a); rng = np.random.default_rng(0); st = []
    for _ in range(B):
        i = rng.integers(0, n, n); st.append((np.exp(b[i].mean()) / np.exp(a[i].mean()) - 1) * 100)
    return float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def run():
    print("[OPT-2.7B W4A8 budget sweep]  where does W4A8 become deployable, and does the cert edge survive it?\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = SW.load_corpus(tok, N_CAL + N_EVAL); cal, ev = seqs[:N_CAL], seqs[N_CAL:]
    torch.nn.Linear.forward = SW.patched
    SW.REGIME["w"], SW.REGIME["a"] = "fp4", "fp8"                      # W4A8

    m32 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    refs_cal = [torch.log_softmax(m32(s.to(DEV)).logits[0].float(), -1).half().cpu() for s in cal]
    refs_ev = [torch.log_softmax(m32(s.to(DEV)).logits[0].float(), -1).half().cpu() for s in ev]
    del m32; gc.collect(); torch.cuda.empty_cache()
    tgts = [s[0, 1:] for s in ev]
    gold = float(np.exp(np.mean([float(-refs_ev[j][:L - 1].double().gather(1, tgts[j].unsqueeze(1)).mean())
                                 for j in range(len(ev))])))

    mg = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.bfloat16).to(DEV).eval()
    U = OP.units(mg)
    store = {}; hs = [u.register_forward_pre_hook((lambda u: (lambda m, i: store.__setitem__(id(u),
              torch.maximum(store.get(id(u), i[0].reshape(-1, i[0].size(-1)).float().abs().amax(0)),
                            i[0].reshape(-1, i[0].size(-1)).float().abs().amax(0)))))(u)) for u in U]
    for u in U:
        u._prec = "hi"
    for s in cal:
        mg(s.to(DEV))
    for h in hs:
        h.remove()
    for u in U:
        amax = store[id(u)].clamp_min(1e-12); wmax = u.weight.float().abs().amax(0).clamp_min(1e-12)
        u._s = ((amax ** ALPHA) / (wmax ** (1 - ALPHA))).clamp(1e-3, 1e3).to(u.weight.dtype)

    impC = np.zeros(len(U)); impP = np.zeros(len(U)); fl = 0.0
    for ci, s in enumerate(cal):
        ids = s.to(DEV)
        for u in U:
            u._prec = "hi"
        bf = mg(ids).logits[0].float(); lpb = torch.log_softmax(bf.double(), -1).cpu()
        fl += kl_c(refs_cal[ci], bf)
        for i in range(len(U)):
            U[i]._prec = "low"; cfg = mg(ids).logits[0].float(); U[i]._prec = "hi"
            impC[i] += kl_c(refs_cal[ci], cfg)
            lq = torch.log_softmax(cfg.double(), -1).cpu()
            impP[i] += (lpb.exp() * (lpb - lq)).sum(-1).mean().item()
    impC = impC / len(cal) - fl / len(cal); impP /= len(cal)

    def ppl_nll(keep):
        nll = np.zeros(len(ev))
        for j, s in enumerate(ev):
            ids = s.to(DEV)
            for t, u in enumerate(U):
                u._prec = "hi" if t in keep else "low"
            lp = torch.log_softmax(mg(ids).logits[0].float().double(), -1).cpu()
            nll[j] = float(-lp[:L - 1].gather(1, tgts[j].unsqueeze(1)).mean())
        return nll

    print(f"  fp32 gold PPL = {gold:.2f}   (W4A8 all-192-low PPL below; protect top-k by score)\n")
    print(f"  {'k(bf16)':>7} {'%units':>7} {'cert PPL':>10} {'prac PPL':>10} {'cert dGold':>11} {'edge %':>17} {'deployable?':>12}")
    for k in BUDGETS:
        cset = set(np.argsort(-impC)[:k].tolist()); pset = set(np.argsort(-impP)[:k].tolist())
        cn = ppl_nll(cset); pn = ppl_nll(pset)
        cppl = float(np.exp(cn.mean())); pppl = float(np.exp(pn.mean()))
        edge = (pppl / cppl - 1) * 100; lo, hi = boot(cn, pn)
        dgold = (cppl / gold - 1) * 100
        depl = "yes" if dgold < 10 else ("marginal" if dgold < 30 else "broken")
        print(f"  {k:>7} {100*k/len(U):>6.0f}% {cppl:>10.2f} {pppl:>10.2f} {dgold:>+10.1f}% "
              f"{edge:>+7.2f}[{lo:+.1f},{hi:+.1f}] {depl:>12}")

    print("\n[VERDICT] read down the 'deployable?' column: at the FIRST k where cert PPL is near gold (<10-30%),")
    print("  is the edge still significant? If the model only becomes usable when the edge has vanished (or only")
    print("  near k=192), W4A8 has NO operational claim -- the +1.8% lived in the broken regime. If a usable k")
    print("  retains a significant edge, that k is the real operational frontier.")
    print("\n[V-hw] RTX 5070 sm_120, OPT-2.7B, W4A8+SmoothQuant, WikiText-2, fp32 gold. Held-out, paired bootstrap.")
    return True


if __name__ == "__main__":
    run()
