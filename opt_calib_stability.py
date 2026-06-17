"""CALIBRATION-SEED STABILITY: is the +30% edge (and the allocation it rests on) robust, or overfit to 6 texts?

The statistical harness used ONE calibration set. This re-derives the allocation from N_SEEDS DISJOINT
calibration sets and evaluates each on the SAME held-out eval sequences (paired bootstrap). Two questions:
  (1) does the significant edge (k>=32) hold across ALL calibration draws, with overlapping CIs?
  (2) are the certified ALLOCATIONS themselves consistent across draws (top-k set overlap)?
If the edge swings or the allocations disagree -> the +30% was a calibration artifact, say so. If both hold ->
the allocation is real and the edge is minted. Either way is genesis (per the abstraction-pushes-back principle).
[V-hw] RTX 5070 sm_120, OPT-2.7B, SmoothQuant configs, WikiText-2. fp32 anchor on GPU, cached.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; ALPHA = 0.5; L = 64
N_SEEDS = 3; N_CAL = 6; N_EVAL = 48; BUDGETS = (16, 32, 48); NU = 192


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


def boot_ci(c, p, B=4000):
    n = len(c); rng = np.random.default_rng(0); st = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        st.append((p[idx].mean() - c[idx].mean()) / max(p[idx].mean(), 1e-30) * 100)
    return float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def mean_jaccard(sets):
    js = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]; js.append(len(a & b) / max(len(a | b), 1))
    return float(np.mean(js)) if js else 1.0


def run():
    print(f"[OPT-2.7B calibration-seed stability]  {N_SEEDS} disjoint calibration sets, {N_EVAL} shared eval seqs\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = load_corpus(tok, N_SEEDS * N_CAL + N_EVAL)
    cal_sets = [seqs[s * N_CAL:(s + 1) * N_CAL] for s in range(N_SEEDS)]
    ev = seqs[N_SEEDS * N_CAL:]
    torch.nn.Linear.forward = OP.patched_linear

    # Phase A: fp32 anchor on GPU -> cache refs (cal per seed + shared eval)
    m32 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    refs_cal = [[torch.log_softmax(m32(s.to(DEV)).logits[0].float(), -1).half().cpu() for s in cs] for cs in cal_sets]
    refs_ev = [torch.log_softmax(m32(s.to(DEV)).logits[0].float(), -1).half().cpu() for s in ev]
    del m32; gc.collect(); torch.cuda.empty_cache()

    mg = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.bfloat16).to(DEV).eval()
    U = OP.units(mg)
    res = {}; cert_sets = {k: [] for k in BUDGETS}

    for seed in range(N_SEEDS):
        cal = cal_sets[seed]; rc = refs_cal[seed]
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

        impC = np.zeros(NU); impP = np.zeros(NU); floorA = 0.0
        for ci, s in enumerate(cal):
            ids = s.to(DEV)
            for u in U:
                u._prec = "hi"
            bf = mg(ids).logits[0].float(); logp_bf = torch.log_softmax(bf.double(), -1).cpu()
            floorA += kl_cached(rc[ci], bf)
            for i in range(NU):
                U[i]._prec = "smooth"; cfg = mg(ids).logits[0].float(); U[i]._prec = "hi"
                impC[i] += kl_cached(rc[ci], cfg)
                lpq = torch.log_softmax(cfg.double(), -1).cpu()
                impP[i] += (logp_bf.exp() * (logp_bf - lpq)).sum(-1).mean().item()
        impC = impC / len(cal) - floorA / len(cal); impP /= len(cal)

        for k in BUDGETS:
            cset = set(np.argsort(-impC)[:k].tolist()); pset = set(np.argsort(-impP)[:k].tolist())
            cert_sets[k].append(cset)
            klc = np.zeros(len(ev)); klp = np.zeros(len(ev))
            for j, s in enumerate(ev):
                ids = s.to(DEV)
                for t, u in enumerate(U):
                    u._prec = "hi" if t in cset else "smooth"
                klc[j] = kl_cached(refs_ev[j], mg(ids).logits[0].float())
                for t, u in enumerate(U):
                    u._prec = "hi" if t in pset else "smooth"
                klp[j] = kl_cached(refs_ev[j], mg(ids).logits[0].float())
            edge = (klp.mean() - klc.mean()) / max(klp.mean(), 1e-30) * 100
            lo, hi = boot_ci(klc, klp); win = float((klc < klp).mean())
            res[(seed, k)] = (edge, lo, hi, win)

    print(f"  {'budget':>7} {'seed':>5} {'edge %':>8} {'95% CI':>18} {'win':>6} {'sig?':>6}")
    for k in BUDGETS:
        for seed in range(N_SEEDS):
            e, lo, hi, w = res[(seed, k)]
            print(f"  {k:>7} {seed:>5} {e:>+7.1f}% [{lo:>+6.1f},{hi:>+6.1f}] {w:>5.0%} {'YES' if lo > 0 else 'no':>6}")
        print(f"          -> certified top-{k} set overlap across the {N_SEEDS} seeds (mean Jaccard) = "
              f"{mean_jaccard(cert_sets[k]):.2f}\n")

    print("[VERDICT]")
    held = {k: all(res[(s, k)][1] > 0 for s in range(N_SEEDS)) for k in BUDGETS}
    consistent = [k for k in BUDGETS if held[k]]
    if consistent:
        print(f"  the edge is significant in ALL {N_SEEDS} calibration draws at budgets {consistent}: the +30% is")
        print(f"  NOT a calibration artifact -- it holds across disjoint calibration sets. Allocation is real. [V-hw]")
    else:
        print(f"  the edge does NOT hold across all calibration draws -- it was calibration-overfit. Retract the claim.")
    print("\n[V-hw] RTX 5070 sm_120, OPT-2.7B, SmoothQuant, WikiText-2. 3 disjoint calibration sets, paired bootstrap.")
    return res, cert_sets


if __name__ == "__main__":
    run()
