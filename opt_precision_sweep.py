"""GAP #6 (the missing axis): map the certification edge across the COMPUTE-METHOD axis (precision regime).

The W8A8 perplexity null was measured in the EASY CORNER: all-quant costs only +0.65% PPL, so allocation has
no operational room. The pre-reg's P3 says FP4 is the load-bearing regime. This sweep moves along the
precision axis -- W8A8 -> W4A8 -> W4A4 -- and asks at each regime: (i) how much room is there (all-low PPL
degradation), and (ii) does the certified-vs-practical edge cross from structural (KL) into operational (PPL)?

Hypothesis: as precision drops, the deployment error grows, and the +30% KL edge should start translating to a
significant PPL edge. If it stays null even at W4A4 -> the certification thesis is real-but-operationally-vacuous.
SmoothQuant migration applied in every regime. [V-hw] OPT-2.7B, WikiText-2, fp32 gold, held-out, bootstrap.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; ALPHA = 0.5; L = 64
N_CAL = 6; N_EVAL = 64; K = 48
FP8 = torch.tensor(sorted({0.0, *[(m / 8 if e == 0 else (1 + m / 8)) * 2.0 ** (e - 7 if e else -6)
                                  for e in range(16) for m in range(8) if (m / 8 if e == 0 else (1 + m / 8)) * 2.0 ** (e - 7 if e else -6) <= 448.0]}), dtype=torch.float32)
FP4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
GRIDS = {"fp8": FP8.to(DEV), "fp4": FP4.to(DEV)}
REGIME = {"w": "fp8", "a": "fp8"}
REGIMES = [("W8A8", "fp8", "fp8"), ("W4A8", "fp4", "fp8"), ("W4A4", "fp4", "fp4")]


def qgrid(x, grid, axis):
    s = x.abs().amax(axis, keepdim=True).clamp_min(1e-12) / grid[-1]
    a = (x / s).abs(); mids = (grid[:-1] + grid[1:]) / 2
    return torch.sign(x) * grid[torch.bucketize(a, mids)] * s


def patched(self, x):
    mode = getattr(self, "_prec", "hi"); W = self.weight; b = self.bias
    if mode == "hi":
        return torch.nn.functional.linear(x, W, b)
    x2d = x.reshape(-1, x.size(-1)); s = self._s
    xs = x2d / s; Ws = W * s[None, :]
    xq = qgrid(xs.float(), GRIDS[REGIME["a"]], 1).to(x2d.dtype)        # per-token activation
    wq = qgrid(Ws.float(), GRIDS[REGIME["w"]], 1).to(W.dtype)          # per-output-channel weight
    out = xq @ wq.t()
    return (out + b if b is not None else out).view(*x.shape[:-1], W.size(0))


def load_corpus(tok, n):
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * L:(i + 1) * L].unsqueeze(0) for i in range(n)]


def kl_c(lr, q):
    lq = torch.log_softmax(q.double(), -1).cpu()
    return (lr.double().exp() * (lr.double() - lq)).sum(-1).mean().item()


def boot(a, b, B=3000):
    n = len(a); rng = np.random.default_rng(0); st = []
    for _ in range(B):
        i = rng.integers(0, n, n); st.append((np.exp(b[i].mean()) / np.exp(a[i].mean()) - 1) * 100)
    return float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def run():
    print(f"[OPT-2.7B precision sweep]  does the cert edge cross structural->operational as precision drops? k={K}\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = load_corpus(tok, N_CAL + N_EVAL); cal, ev = seqs[:N_CAL], seqs[N_CAL:]
    torch.nn.Linear.forward = patched

    m32 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    refs_cal = [torch.log_softmax(m32(s.to(DEV)).logits[0].float(), -1).half().cpu() for s in cal]
    refs_ev = [torch.log_softmax(m32(s.to(DEV)).logits[0].float(), -1).half().cpu() for s in ev]
    del m32; gc.collect(); torch.cuda.empty_cache()
    tgts = [s[0, 1:] for s in ev]
    gold_ppl = float(np.exp(np.mean([float(-refs_ev[j][:L - 1].double().gather(1, tgts[j].unsqueeze(1)).mean())
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

    print(f"  fp32 gold PPL = {gold_ppl:.3f}\n")
    print(f"  {'regime':>6} {'all-low dPPL':>13} {'cert KL edge':>13} {'cert PPL edge':>22} {'verdict':>10}")
    for name, wg, ag in REGIMES:
        REGIME["w"], REGIME["a"] = wg, ag
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
        cset = set(np.argsort(-impC)[:K].tolist()); pset = set(np.argsort(-impP)[:K].tolist())

        def ev_metrics(keep):
            nll = np.zeros(len(ev)); kl = np.zeros(len(ev))
            for j, s in enumerate(ev):
                ids = s.to(DEV)
                for t, u in enumerate(U):
                    u._prec = "hi" if t in keep else "low"
                lg = mg(ids).logits[0].float(); lp = torch.log_softmax(lg.double(), -1).cpu()
                nll[j] = float(-lp[:L - 1].gather(1, tgts[j].unsqueeze(1)).mean())
                kl[j] = float((refs_ev[j].double().exp() * (refs_ev[j].double() - lp)).sum(-1).mean())
            return nll, kl
        an, _ = ev_metrics(set())                                       # all-low
        cn, ck = ev_metrics(cset); pn, pk = ev_metrics(pset)
        all_low_dppl = (np.exp(an.mean()) / gold_ppl - 1) * 100
        kl_edge = (pk.mean() - ck.mean()) / pk.mean() * 100
        ppl_edge = (np.exp(pn.mean()) / np.exp(cn.mean()) - 1) * 100
        lo, hi = boot(cn, pn); sig = lo > 0
        print(f"  {name:>6} {all_low_dppl:>+12.2f}% {kl_edge:>+12.1f}% {ppl_edge:>+8.3f}% [{lo:+.2f},{hi:+.2f}]"
              f"{'  SIG' if sig else '  ns':>10}")

    print("\n[VERDICT] If the PPL edge becomes significant (CI>0) as precision drops, the operational value lives in")
    print("  the harsh regime (P3 confirmed). If it stays ns even at W4A4, the certification edge is structurally")
    print("  real but operationally vacuous across the deployable precision range. [V-hw]")
    print("\n[V-hw] RTX 5070 sm_120, OPT-2.7B, SmoothQuant, WikiText-2, fp32 gold. Held-out, paired bootstrap.")
    return True


if __name__ == "__main__":
    run()
