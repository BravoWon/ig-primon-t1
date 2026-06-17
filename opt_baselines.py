"""GAP #5 -- the load-bearing one: does certified scoring beat a REAL allocator, not just bf16-reference scoring?

"+30% vs bf16-reference leave-one-low" is beats-naive, not beats-practice. Real deployed allocators mostly use
CHEAP, reference-free heuristics. We add four and ask whether the EXPENSIVE fp32-certified reference still wins:
  random          : no information (floor; averaged over seeds).
  magnitude        : ||dW_i||/||W_i|| -- RELATIVE weight quant error (size-fair). Reference-free.
  activation (AWQ) : mean |x_i| input-activation salience. AWQ's core idea. Reference-free.
  first-order      : ||(x/s) dWs^T||/||x W^T|| -- relative first-order OUTPUT perturbation (size-fair LAMP). Reference-free.
  NB the first run used raw Frobenius ||dW|| (size-biased, came out worse than random) -- fixed to size-fair here.
  practical        : leave-one-low KL vs the bf16 reference (our prior baseline).
  certified        : leave-one-low KL vs the fp32 anchor (the method under test -- the only one needing fp32).
Evaluate every allocation's true KL (vs fp32 anchor) on held-out WikiText; the verdict is whether certified
SIGNIFICANTLY beats the BEST non-certified allocator (paired bootstrap). If a cheap allocator matches certified,
the expensive reference is not worth it and we say so. [V-hw] OPT-2.7B, SmoothQuant configs, WikiText-2.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; ALPHA = 0.5; L = 64; FP8 = torch.float8_e4m3fn
N_CAL = 6; N_EVAL = 48; BUDGETS = (32, 48); NU = 192


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


def boot_ci(a, b, B=4000):
    """relative edge of a over b: (b-a)/b, bootstrap CI over paired sequences."""
    n = len(a); rng = np.random.default_rng(0); st = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        st.append((b[idx].mean() - a[idx].mean()) / max(b[idx].mean(), 1e-30) * 100)
    return float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def run():
    print(f"[OPT-2.7B gap-5 baselines]  certified vs cheap real allocators, WikiText-2, {N_EVAL} held-out\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = load_corpus(tok, N_CAL + N_EVAL)
    cal, ev = seqs[:N_CAL], seqs[N_CAL:]
    torch.nn.Linear.forward = OP.patched_linear

    m32 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    refs_cal = [torch.log_softmax(m32(s.to(DEV)).logits[0].float(), -1).half().cpu() for s in cal]
    refs_ev = [torch.log_softmax(m32(s.to(DEV)).logits[0].float(), -1).half().cpu() for s in ev]
    del m32; gc.collect(); torch.cuda.empty_cache()

    mg = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.bfloat16).to(DEV).eval()
    U = OP.units(mg)

    # calibration: per-channel act max (for s) + mean-abs activation salience (for AWQ/first-order)
    amax_store = {}; sal_store = {}; cnt = {}; hs = []
    def mk(u):
        def hook(mod, inp):
            x = inp[0].reshape(-1, inp[0].size(-1)).float()
            a = x.abs().amax(0); m = x.abs().mean().item()
            amax_store[id(u)] = a if id(u) not in amax_store else torch.maximum(amax_store[id(u)], a)
            sal_store[id(u)] = sal_store.get(id(u), 0.0) + m; cnt[id(u)] = cnt.get(id(u), 0) + 1
        return u.register_forward_pre_hook(hook)
    for u in U:
        u._prec = "hi"; hs.append(mk(u))
    for s in cal:
        mg(s.to(DEV))
    for h in hs:
        h.remove()
    for u in U:
        amax = amax_store[id(u)].clamp_min(1e-12); wmax = u.weight.float().abs().amax(0).clamp_min(1e-12)
        u._s = ((amax ** ALPHA) / (wmax ** (1 - ALPHA))).clamp(1e-3, 1e3).to(u.weight.dtype)

    # reference-free scores -- SIZE-FAIR (the first run's raw Frobenius ||dW|| was size-biased -> worse than
    # random; fixed here). magnitude = RELATIVE weight quant error; first_order = relative first-order OUTPUT
    # perturbation ||(x/s) dWs^T|| / ||x W^T|| (the proper reference-free sensitivity, computed in a 2nd pass).
    magnitude = np.zeros(NU); activation = np.zeros(NU)
    for i, u in enumerate(U):
        W = u.weight.float(); Ws = W * u._s.float()[None, :]
        sw = Ws.abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        Wq = (Ws / sw).to(FP8).float() * sw
        magnitude[i] = float((Wq - Ws).norm() / Ws.norm().clamp_min(1e-12))      # relative, size-fair
        activation[i] = sal_store[id(u)] / cnt[id(u)]
    idx_of = {id(u): i for i, u in enumerate(U)}
    fo_num = np.zeros(NU); fo_den = np.zeros(NU); hs2 = []
    def fo_hook(mod, inp):
        i = idx_of[id(mod)]
        W = mod.weight.float(); s = mod._s.float(); Ws = W * s[None, :]
        sw = Ws.abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        dWs = (Ws / sw).to(FP8).float() * sw - Ws
        x = inp[0].reshape(-1, inp[0].size(-1)).float()
        fo_num[i] += float(((x / s) @ dWs.t()).norm()); fo_den[i] += float((x @ W.t()).norm())
    for u in U:
        hs2.append(u.register_forward_pre_hook(fo_hook))
    for s in cal:
        mg(s.to(DEV))
    for h in hs2:
        h.remove()
    first_order = fo_num / np.maximum(fo_den, 1e-30)                              # relative output perturbation

    # reference-based scores: practical (bf16) + certified (fp32), leave-one-low
    impC = np.zeros(NU); impP = np.zeros(NU); floorA = 0.0
    for ci, s in enumerate(cal):
        ids = s.to(DEV)
        for u in U:
            u._prec = "hi"
        bf = mg(ids).logits[0].float(); logp_bf = torch.log_softmax(bf.double(), -1).cpu()
        floorA += kl_cached(refs_cal[ci], bf)
        for i in range(NU):
            U[i]._prec = "smooth"; cfg = mg(ids).logits[0].float(); U[i]._prec = "hi"
            impC[i] += kl_cached(refs_cal[ci], cfg)
            lpq = torch.log_softmax(cfg.double(), -1).cpu()
            impP[i] += (logp_bf.exp() * (logp_bf - lpq)).sum(-1).mean().item()
    impC = impC / len(cal) - floorA / len(cal); impP /= len(cal)

    scorers = {"magnitude": magnitude, "activation(AWQ)": activation, "first-order": first_order,
               "practical(bf16)": impP, "certified(fp32)": impC}

    def eval_alloc(keep):
        out = np.zeros(len(ev))
        for j, s in enumerate(ev):
            ids = s.to(DEV)
            for t, u in enumerate(U):
                u._prec = "hi" if t in keep else "smooth"
            out[j] = kl_cached(refs_ev[j], mg(ids).logits[0].float())
        return out

    for k in BUDGETS:
        print(f"  ---- budget k={k} (units kept bf16) ----")
        kls = {}
        for name, sc in scorers.items():
            kls[name] = eval_alloc(set(np.argsort(-sc)[:k].tolist()))
        # random baseline (avg 3 seeds)
        rkl = np.mean([eval_alloc(set(np.random.default_rng(r).choice(NU, k, replace=False).tolist()))
                       for r in range(3)], axis=0)
        kls["random"] = rkl
        order = sorted(kls, key=lambda n: kls[n].mean())
        for name in order:
            print(f"    {name:>16}  mean KL = {kls[name].mean():.4e}")
        # certified vs the best NON-certified allocator
        best_other = min((n for n in kls if n != "certified(fp32)"), key=lambda n: kls[n].mean())
        lo, hi = boot_ci(kls["certified(fp32)"], kls[best_other])
        edge = (kls[best_other].mean() - kls["certified(fp32)"].mean()) / max(kls[best_other].mean(), 1e-30) * 100
        sig = lo > 0
        print(f"    => certified vs BEST non-certified ({best_other}): {edge:+.1f}% [{lo:+.1f},{hi:+.1f}]  "
              f"{'SIGNIFICANT' if sig else 'NOT significant (cheap allocator matches certified)'}\n")

    print("[VERDICT] If certified significantly beats the best cheap allocator, the expensive fp32 reference earns")
    print("  its cost (operational edge over practice). If not, the edge was mostly 'beats bf16-reference-scoring'.")
    print("\n[V-hw] RTX 5070 sm_120, OPT-2.7B, SmoothQuant, WikiText-2. fp32 anchor. Held-out, paired bootstrap.")
    return True


if __name__ == "__main__":
    run()
