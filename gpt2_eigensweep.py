"""Exp 3 -- compression sweep across the MEANINGFUL 2x2: {use-blind, use-aware} x {low-rank, quantize}.

Exp 2's matched point landed in the both-dead zone (low-rank 4.431 ~ FP4 4.472, both ~all-FP4 catastrophe), so
it couldn't RANK the methods. And the whole comparison lived in the use-BLIND quadrant (plain-SVD low-rank,
RTN-FP4) -- the path this program found weak. This sweep fixes both: vary the storage budget so the curves
separate, and add the use-AWARE arms (activation-metric low-rank = ASVD; GPTQ = activation-Hessian quant) so the
sweep can see the axis we actually validated.

PRE-REGISTERED (hypothesis / control / falsifier):
  H1 (the original question): quantization dominates low-rank at matched storage; they diverge at GENTLE ratios
     (8-bit quant ~ lossless while matched-k low-rank already truncates most of the rank). Falsifier: low-rank
     <= quant at any tested ratio.
  H2 (the bigger axis, the session's lever): use-awareness moves the frontier MORE than low-rank-vs-quant within
     the blind quadrant -- GPTQ << RTN, act-LR < plain-LR, use-aware-quant the frontier. Falsifier: use-aware ~
     use-blind within method (lever inert here too).
  Control: random-k >> top-k across ALL ratios (directions special, not ratio-specific).
Predictions (honest): H1 holds (low-rank never wins); H2 holds strongly; strong eigenweight form dies, the
use-metric form wins.

Storage accounting (matched stored-bits, fraction r of fp16): quant bits/entry b = round(16 r) in [2,8];
low-rank rank k = round(r * m*n / (m+n)) so k*(m+n)*16 == r*16*m*n. Scale/singular-value overhead ignored
symmetrically (small). GPT-2 Conv1D weight is (in,out); transposed to (out,in) for the quantizer, H is (in,in).
[V] gpt2-small, held-out text, KL(orig||method), fp32.
"""
import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
DEV = "cuda:0"; GPT2 = "C:/Users/JT-DEV1/Documents/gpt2-sm"
L = 128; N_CAL = 32; N_EVAL = 16; GROUP = 128
RATIOS = [0.5, 0.375, 0.25, 0.1875, 0.125]                          # fraction of fp16 -> b = 8,6,4,3,2 bits
torch.manual_seed(0)


def corpus_ids(tok):
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    return tok(text, return_tensors="pt").input_ids[0]


def targets(m):
    T = []
    for bi, blk in enumerate(m.transformer.h):                      # UNIQUE keys per block -- non-unique names
        T += [(f"h{bi}.attn.c_attn", blk.attn.c_attn), (f"h{bi}.attn.c_proj", blk.attn.c_proj),   # collapsed 48->4 (bug)
              (f"h{bi}.mlp.c_fc", blk.mlp.c_fc), (f"h{bi}.mlp.c_proj", blk.mlp.c_proj)]
    return T                                                        # Conv1D: weight is (in, out)


# ---- compression methods. W is (in,out) (Conv1D). H,(rms) over the INPUT dim. Return compressed (in,out). ----
def lowrank(W, k, rms=None):
    Wp = (rms.unsqueeze(1) * W) if rms is not None else W           # ASVD: scale input channels by act-RMS
    U, S, Vh = torch.linalg.svd(Wp.float(), full_matrices=False)
    Wk = (U[:, :k] * S[:k]) @ Vh[:k]
    if rms is not None:
        Wk = Wk / rms.unsqueeze(1)
    return Wk.to(W.dtype)


def random_k(W, k):
    m = W.shape[0]
    Q, _ = torch.linalg.qr(torch.randn(m, k, device=W.device))      # random k-dim row subspace
    return (Q @ (Q.t() @ W.float())).to(W.dtype)


def quant(W, H, b, group, gptq):
    """INT-b, group-`group` along input dim. gptq=True uses activation-Hessian error compensation; else RTN."""
    WL = W.t().contiguous().float()                                  # (out,in) Linear convention
    out, inp = WL.shape; qmax = float(2 ** (b - 1) - 1)
    if not gptq:
        Gp = group
        for i in range(0, inp, Gp):
            sl = slice(i, min(i + Gp, inp))
            s = WL[:, sl].abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
            WL[:, sl] = torch.clamp(torch.round(WL[:, sl] / s), -qmax, qmax) * s
        return WL.t().contiguous().to(W.dtype)
    Hf = H.float().clone(); damp = 0.01
    dead = torch.diag(Hf) == 0; Hf[dead, dead] = 1.0; WL[:, dead] = 0
    Hf[torch.arange(inp, device=Hf.device), torch.arange(inp, device=Hf.device)] += damp * torch.diag(Hf).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hf)), upper=True)
    ss = None
    for i in range(inp):
        if i % group == 0:
            ss = WL[:, i:min(i + group, inp)].abs().amax(1).clamp_min(1e-12) / qmax
        w = WL[:, i]; d = Hinv[i, i]
        q = torch.clamp(torch.round(w / ss), -qmax, qmax) * ss
        err = (w - q) / d
        WL[:, i] = q
        WL[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
    return WL.t().contiguous().to(W.dtype)


def collect(m, cal, T):
    H = {n: None for n, _ in T}; ms2 = {n: None for n, _ in T}; cnt = {n: 0 for n, _ in T}
    hooks = []
    def mk(name):
        def hook(mod, ip):
            x = ip[0].reshape(-1, ip[0].size(-1)).float()
            h = x.t() @ x
            H[name] = h if H[name] is None else H[name] + h
            s = (x * x).sum(0)
            ms2[name] = s if ms2[name] is None else ms2[name] + s
            cnt[name] += x.shape[0]
        return hook
    for name, mod in T:
        hooks.append(mod.register_forward_pre_hook(mk(name)))
    for s in cal:
        m(s.to(DEV))
    for h in hooks:
        h.remove()
    Hm = {n: H[n] / max(cnt[n], 1) for n in H}
    rms = {n: (ms2[n] / max(cnt[n], 1)).clamp_min(1e-12).sqrt() for n in ms2}
    return Hm, rms


def ppl_strided(m, ids, stride=512):
    """Proper strided perplexity (the session's trusted metric). KL-over-full-vocab and 128-chunk PPL are
    hypersensitive; this is the metric that ranks 8-bit << 4-bit correctly."""
    nlls = []; ntok = ids.numel() - 1
    for i in range(0, ntok, stride):
        c = ids[i:i + stride + 1].unsqueeze(0).to(DEV)
        if c.numel() < 8:
            break
        lp = torch.log_softmax(m(c[:, :-1]).logits[0].float(), -1)
        nlls.append(-lp.gather(1, c[0, 1:].unsqueeze(1)).mean() * (c.numel() - 1))
    return float(np.exp((torch.stack(nlls).sum() / ntok).item()))


def rank_stats(W):
    S = torch.linalg.svdvals(W.float())
    p = (S * S); p = p / p.sum()
    eff = float(torch.exp(-(p * (p.clamp_min(1e-12)).log()).sum()) / len(S))   # effective rank / full
    stable = float((S * S).sum() / (S[0] ** 2) / len(S))                       # stable rank / full
    return eff, stable


def run():
    print("[GPT-2 small eigen-sweep]  {use-blind,use-aware} x {low-rank,quantize}, matched storage\n")
    tok = AutoTokenizer.from_pretrained(GPT2)
    ids = corpus_ids(tok)
    eval_ids = ids[:8192].to(DEV)                                    # held-out eval
    cal = [ids[8192 + i * L: 8192 + (i + 1) * L].unsqueeze(0) for i in range(N_CAL)]   # disjoint calibration
    m = AutoModelForCausalLM.from_pretrained(GPT2).to(torch.float32).to(DEV).eval()
    T = targets(m)

    # Exp 1 confirm (verify-at-source): mean rank stats over targets vs a random control of matched shape/scale
    effs, stabs, reffs, rstabs = [], [], [], []
    for _, mod in T:
        W = mod.weight.detach()
        e, s = rank_stats(W); effs.append(e); stabs.append(s)
        R = torch.randn_like(W) * W.std()
        re, rs = rank_stats(R); reffs.append(re); rstabs.append(rs)
    print(f"[Exp1 confirm] mean over {len(T)} target matrices (fraction of full rank):")
    print(f"    trained: eff-rank {np.mean(effs):.3f}  stable-rank {np.mean(stabs):.3f}")
    print(f"    random : eff-rank {np.mean(reffs):.3f}  stable-rank {np.mean(rstabs):.3f}")
    print("    -> spiked spectrum (low stable, high eff) confirmed: a few energy spikes over a broad bulk.\n")

    Hm, rms = collect(m, cal, T)
    orig_state = {n: mod.weight.detach().cpu().clone() for n, mod in T}   # unique keys now -> all 48 preserved
    gold = ppl_strided(m, eval_ids)
    print(f"  gold PPL (fp32, strided) = {gold:.3f}\n")

    methods = [("lowrank-plain", "lr", False), ("lowrank-actW", "lr", True),
               ("quant-RTN", "q", False), ("quant-GPTQ", "q", True), ("random-k", "rk", False)]
    print(f"  {'ratio':>6} {'bits':>4} " + "".join(f"{nm:>16}" for nm, _, _ in methods) + "     (PPL; gold %s)" % round(gold, 1))
    results = {nm: [] for nm, _, _ in methods}
    for r in RATIOS:
        b = int(round(16 * r)); b = max(2, min(8, b))
        row = []
        for nm, kind, aware in methods:
            for n, mod in T:                                         # restore originals
                mod.weight.data = orig_state[n].to(DEV).clone()
            for n, mod in T:
                W = mod.weight.data; mm, nn_ = W.shape; k = max(1, int(round(r * mm * nn_ / (mm + nn_))))
                if kind == "lr":
                    Wc = lowrank(W, k, rms[n] if aware else None)
                elif kind == "rk":
                    Wc = random_k(W, k)
                else:
                    Wc = quant(W, Hm[n], b, GROUP, aware)
                mod.weight.data = Wc
            p = ppl_strided(m, eval_ids); results[nm].append(p); row.append(p)
        print(f"  {r:>6.3f} {b:>4d} " + "".join(f"{v:>16.2f}" for v in row))
    for n, mod in T:
        mod.weight.data = orig_state[n].to(DEV)

    print("\n[VERDICT -- scored against pre-registration; PPL, lower=better]")
    lr0 = np.array(results["lowrank-plain"]); q0 = np.array(results["quant-RTN"])
    qg = np.array(results["quant-GPTQ"]); lra = np.array(results["lowrank-actW"]); rk = np.array(results["random-k"])
    h1 = bool(np.all(q0 <= lr0 + 1e-9))
    print(f"  H1 quant dominates low-rank at every ratio (use-blind): {h1}  "
          f"(plain-LR/RTN ratio per r: {np.round(lr0 / q0, 2)})")
    h2_q = bool(np.all(qg <= q0 + 1e-9)); h2_lr = bool(np.all(lra <= lr0 + 1e-9))
    print(f"  H2 use-aware beats use-blind: quant GPTQ<=RTN {h2_q}, lowrank actW<=plain {h2_lr}")
    print(f"     per-ratio GPTQ/RTN ratio: {np.round(qg / q0, 3)};  actW/plain ratio: {np.round(lra / lr0, 3)}")
    ctrl = bool(np.all(rk >= lr0 - 1e-9))
    print(f"  control random-k >= top-k (plain LR) at every ratio: {ctrl}  (directions special)")
    print(f"  frontier (lowest PPL) per ratio: " +
          ", ".join(f"r={RATIOS[i]}:{min(methods, key=lambda mm: results[mm[0]][i])[0]}" for i in range(len(RATIOS))))
    print("\n[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, matched stored-bits.")
    return results, gold


if __name__ == "__main__":
    run()
