"""STATISTICAL ROBUSTNESS: is the OPT-2.7B certification edge real, or a small-sample fluctuation?

The +8%/4-of-5 result was 6 sentences, no error bars -- suggestive, not significant. This harness clears the
honest floor: a real corpus (WikiText-2), the edge measured as a PAIRED per-sequence observation (certified
and practical allocations evaluated on the SAME held-out sequences), with bootstrap 95% CIs and a per-sequence
win rate. If the CI excludes 0, the edge is real; if it straddles 0, +8% was noise.

Design:
  - allocation scored ONCE on a small disjoint calibration set (SmoothQuant configs, fp32 vs bf16 reference).
  - evaluated on N_EVAL held-out WikiText sequences; per-sequence KL_certified, KL_practical (vs fp32 anchor).
  - statistic = relative edge (mean KL_prac - mean KL_cert)/mean KL_prac, bootstrap CI over eval sequences.
Memory: fp32 anchor on GPU alone (10.8GB), cache refs as fp16 log-softmax on CPU, free, then bf16 deployment.
[V-hw] RTX 5070 sm_120. OPT-2.7B, real SmoothQuant configs, WikiText-2. Honest significance test.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; ALPHA = 0.5
L = 64                      # tokens per sequence
N_CAL = 6                   # calibration sequences (score the allocation)
N_EVAL = 64                # held-out evaluation sequences (the CI)
BUDGETS = (16, 32, 48)
NU = 192


def load_corpus(tok):
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")   # parquet mirror, no script
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = N_CAL + N_EVAL
    chunks = [ids[i*L:(i+1)*L] for i in range(n)]
    return [c.unsqueeze(0) for c in chunks if c.numel() == L][:n]


def kl_cached(logp_ref, q_logits):
    lpr = logp_ref.double(); lpq = torch.log_softmax(q_logits.double(), -1).cpu()
    return (lpr.exp() * (lpr - lpq)).sum(-1).mean().item()


def boot_ci(cert, prac, B=4000):
    n = len(cert); rng = np.random.default_rng(0); stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        c = cert[idx].mean(); p = prac[idx].mean()
        stats.append((p - c) / max(p, 1e-30) * 100)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def run():
    print(f"[OPT-2.7B statistical robustness]  WikiText-2, {N_EVAL} held-out seqs x {L} tok, paired bootstrap\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = load_corpus(tok)
    cal, ev = seqs[:N_CAL], seqs[N_CAL:N_CAL + N_EVAL]
    print(f"  corpus: {len(cal)} calibration + {len(ev)} eval sequences\n")
    torch.nn.Linear.forward = OP.patched_linear

    # ---- Phase A: fp32 anchor on GPU (alone) -> cache log-softmax refs (fp16 CPU) ----
    m32 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    refs_cal = [torch.log_softmax(m32(s.to(DEV)).logits[0].float(), -1).half().cpu() for s in cal]
    refs_ev = [torch.log_softmax(m32(s.to(DEV)).logits[0].float(), -1).half().cpu() for s in ev]
    del m32; gc.collect(); torch.cuda.empty_cache()

    # ---- Phase B: bf16 deployment ----
    mg = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.bfloat16).to(DEV).eval()
    U = OP.units(mg)

    # calibrate SmoothQuant s over the calibration set
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

    # score impC (vs fp32) and impP (vs bf16) on the calibration set, SmoothQuant configs
    impC = np.zeros(NU); impP = np.zeros(NU); floorA = 0.0
    for ci, s in enumerate(cal):
        ids = s.to(DEV)
        for u in U:
            u._prec = "hi"
        bf_ref = mg(ids).logits[0].float()
        logp_bf = torch.log_softmax(bf_ref.double(), -1).cpu()
        floorA += kl_cached(refs_cal[ci], bf_ref)
        for i in range(NU):
            U[i]._prec = "smooth"; cfg = mg(ids).logits[0].float(); U[i]._prec = "hi"
            impC[i] += kl_cached(refs_cal[ci], cfg)
            lpq = torch.log_softmax(cfg.double(), -1).cpu()
            impP[i] += (logp_bf.exp() * (logp_bf - lpq)).sum(-1).mean().item()
    impC = impC / len(cal) - floorA / len(cal); impP /= len(cal)

    # evaluate per held-out sequence: KL_cert, KL_prac at each budget
    print(f"  {'k':>4} {'edge %':>8} {'95% CI':>20} {'win rate':>9} {'significant?':>12}")
    out = {}
    for k in BUDGETS:
        cset = set(np.argsort(-impC)[:k].tolist()); pset = set(np.argsort(-impP)[:k].tolist())
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
        lo, hi = boot_ci(klc, klp)
        win = float((klc < klp).mean())
        sig = lo > 0
        out[k] = (edge, lo, hi, win, sig)
        print(f"  {k:>4} {edge:>+7.1f}% [{lo:>+6.1f}, {hi:>+6.1f}]   {win:>8.0%} {('YES' if sig else 'no -- CI spans 0'):>12}")

    print("\n[VERDICT]")
    any_sig = any(out[k][4] for k in BUDGETS)
    if any_sig:
        sig_ks = [k for k in BUDGETS if out[k][4]]
        print(f"  certification edge is STATISTICALLY SIGNIFICANT at budgets {sig_ks} (95% CI excludes 0) on")
        print(f"  {N_EVAL} held-out WikiText sequences. The +8% was a finding, not 6-sentence noise. [V-hw]")
    else:
        print(f"  NO budget reaches significance -- every CI spans 0. The single-run +8% was within sampling")
        print(f"  noise on real text. The edge is NOT established at this size; do not claim it.")
    print(f"\n[V-hw] RTX 5070 sm_120, OPT-2.7B, SmoothQuant configs, WikiText-2 test. 1 calibration set, paired bootstrap.")
    print(f"       Caveats remaining: 1 calibration seed, alpha=0.5 fixed, fp8 fake-quant, KL metric. Not yet 'done'.")
    return out


if __name__ == "__main__":
    run()
