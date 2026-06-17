"""DIAGNOSTIC: why are the cheap reference-free allocators worse than RANDOM on OPT-2.7B?

Two hypotheses, decided by direct measurement (no held-out eval needed):
  (A) real anti-correlation: cheap magnitude/activation proxies key on LOCAL magnitude, which in this regime
      anti-correlates with FINAL sensitivity (certified leave-one-low KL) -> Spearman strongly NEGATIVE.
  (B) degenerate scores: a few extreme-outlier units dominate the cheap scores (max/median huge), so their
      top-k is ~arbitrary-but-outlier-biased -> Spearman ~0, not strongly negative.
We compute the certified leave-one-low KL (ground-truth sensitivity) and each cheap score, then report:
  - Spearman(cheap, certified)  - top-k set overlap with the certified pick  - score degeneracy (max/median).
[V-hw] OPT-2.7B, SmoothQuant, WikiText-2 calibration. Scoring only, no eval.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; ALPHA = 0.5; L = 64; FP8 = torch.float8_e4m3fn
N_CAL = 6; NU = 192


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


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.sqrt((ra @ ra) * (rb @ rb)) + 1e-30))


def run():
    print("[OPT-2.7B baseline diagnostic]  why are cheap allocators worse than random?\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    cal = load_corpus(tok, N_CAL)
    torch.nn.Linear.forward = OP.patched_linear
    m32 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    refs_cal = [torch.log_softmax(m32(s.to(DEV)).logits[0].float(), -1).half().cpu() for s in cal]
    del m32; gc.collect(); torch.cuda.empty_cache()
    mg = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.bfloat16).to(DEV).eval()
    U = OP.units(mg)

    amax_store = {}; sal_store = {}; cnt = {}; hs = []
    def mk(u):
        def hook(mod, inp):
            x = inp[0].reshape(-1, inp[0].size(-1)).float()
            a = x.abs().amax(0)
            amax_store[id(u)] = a if id(u) not in amax_store else torch.maximum(amax_store[id(u)], a)
            sal_store[id(u)] = sal_store.get(id(u), 0.0) + x.abs().mean().item(); cnt[id(u)] = cnt.get(id(u), 0) + 1
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

    magnitude = np.zeros(NU); activation = np.zeros(NU)
    for i, u in enumerate(U):
        W = u.weight.float(); Ws = W * u._s.float()[None, :]
        sw = Ws.abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        Wq = (Ws / sw).to(FP8).float() * sw
        magnitude[i] = float((Wq - Ws).norm() / Ws.norm().clamp_min(1e-12))
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
    first_order = fo_num / np.maximum(fo_den, 1e-30)

    # certified ground-truth sensitivity (leave-one-low KL vs fp32)
    cert = np.zeros(NU)
    for ci, s in enumerate(cal):
        ids = s.to(DEV)
        for u in U:
            u._prec = "hi"
        base = kl_cached(refs_cal[ci], mg(ids).logits[0].float())
        for i in range(NU):
            U[i]._prec = "smooth"; cfg = mg(ids).logits[0].float(); U[i]._prec = "hi"
            cert[i] += kl_cached(refs_cal[ci], cfg) - base
    cert /= len(cal)

    print(f"  ground truth = certified leave-one-low KL sensitivity (192 units)\n")
    print(f"  {'cheap score':>16} {'Spearman vs cert':>18} {'top-48 overlap':>15} {'max/median':>12}")
    for name, sc in (("magnitude", magnitude), ("activation(AWQ)", activation), ("first-order", first_order)):
        sp = spearman(sc, cert)
        ov = len(set(np.argsort(-sc)[:48].tolist()) & set(np.argsort(-cert)[:48].tolist())) / 48
        degen = float(np.max(sc) / (np.median(sc) + 1e-30))
        print(f"  {name:>16} {sp:>+17.3f} {ov:>14.0%} {degen:>12.1f}")

    print("\n[READ]")
    sps = [spearman(sc, cert) for sc in (magnitude, activation, first_order)]
    if all(s < -0.1 for s in sps):
        print("  (A) REAL ANTI-CORRELATION: cheap local-magnitude proxies negatively track true sensitivity on this")
        print("      outlier model -- they protect ROBUST high-magnitude units and quantize the fragile ones. Worse")
        print("      than random is a genuine finding: you cannot cheaply proxy the good allocation here; you must")
        print("      MEASURE the output effect (leave-one-low), which is the whole point of the certified instrument.")
    elif all(abs(s) < 0.1 for s in sps):
        print("  (B) DEGENERATE/UNINFORMATIVE: cheap scores ~uncorrelated with sensitivity (noise), not anti-correlated.")
    else:
        print("  MIXED: report per-score Spearman as-is; do not generalize.")
    print("\n[V-hw] RTX 5070 sm_120, OPT-2.7B, SmoothQuant, WikiText-2 calibration. Scoring-only diagnostic.")
    return cert, magnitude, activation, first_order


if __name__ == "__main__":
    run()
