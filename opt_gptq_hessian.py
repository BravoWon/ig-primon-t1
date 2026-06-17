"""CERTIFIED HESSIAN: does a certified (fp32) vs cheap (bf16) activation Hessian change GPTQ's result?

The program's founding instrument is the certified reference. The operational arc REFUTED it on the wrong
lever (precision *allocation*: certified minimizes KL-from-fp32, the wrong deployment objective). This puts the
SAME instrument on the RIGHT lever -- use-aware *quantization* (GPTQ) -- where the lever actually moved the
needle (INT4-g128 = +2.81% near-lossless). The novel question: GPTQ's Hessian H = X^T X is the instrument that
decides quantization error-compensation. Does building it from CERTIFIED fp32 activations vs CHEAP bf16
activations change the deployed model's perplexity -- or is the Hessian, like the reference, robust to precision?

ISOLATION (the only knob is activation precision in the Hessian):
  * BOTH arms run the SAME fp32 master model (identical weights, identical forward, identical propagation given
    identical quantization decisions). This removes the confound of bf16 *weights* changing the activation
    VALUES -- it isolates activation PRECISION alone.
  * certified arm : H = sum x^T x with x in fp32.
  * cheap arm     : activations rounded to bf16 before the outer product (x -> bf16 -> fp32), underlying signal
    identical. This is exactly the precision a bf16 calibration forward would carry, with nothing else changed.
  * full-rank Hessian (N_CAL=192 -> 12288 tokens > 10240 fc2 in-dim), INT4-g128, sequential reconstruction.

CONTROL BEFORE SCAN: first quantify how much the INSTRUMENT moves (||dH||_F/||H||_F and the diag(H^-1) GPTQ
actually uses). If H barely moves, a null PPL is explained a priori, not mysterious.

Honest prior (pre-reg): GPTQ is known robust to Hessian perturbation (diagonal approximations work). This may be
a CLEAN NULL. Report PPL straight; paired-bootstrap the arm difference. Accept the null if that's the truth.
[V-hw] OPT-2.7B, WikiText-2, fp32 master + fp32 gold, sequential GPTQ, held-out.
"""
import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP

torch.set_grad_enabled(False)
DEV = "cuda:0"; QDEV = "cuda:1"; L = 64; N_CAL = 192; N_EVAL = 64   # 12288 cal tokens > 10240 -> full-rank H
QMAX = 7.0; G = 128                                                  # INT4 group-128: the decisive near-lossless config


def load_corpus(tok, n):
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * L:(i + 1) * L].unsqueeze(0) for i in range(n)]


def gptq_int4(W, H, qmax, Gp, damp=0.01):
    """INT4 group-Gp GPTQ. Device-agnostic (runs where W/H live, i.e. cuda:1)."""
    dev = W.device
    W = W.clone().float(); out, inp = W.shape; H = H.float()
    dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0
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
    del H, Hinv; torch.cuda.empty_cache()
    return W


def _hinv_diag(H, damp=0.01):
    """diag(H^-1) -- the per-column quantity GPTQ divides the error by. Returns it for instrument diagnostics."""
    dev = H.device; inp = H.shape[0]; H = H.float().clone()
    dead = torch.diag(H) == 0; H[dead, dead] = 1.0
    H[torch.arange(inp, device=dev), torch.arange(inp, device=dev)] += damp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    return torch.diagonal(Hinv).clone()


class _Stop(Exception):
    pass


def ppl(model, ev, tgts):
    """Returns (perplexity, per-sequence NLL array) so arms can be paired-bootstrapped."""
    nll = np.zeros(len(ev))
    for j, s in enumerate(ev):
        lp = torch.log_softmax(model(s.to(DEV)).logits[0].float().double(), -1).cpu()
        nll[j] = float(-lp[:L - 1].gather(1, tgts[j].unsqueeze(1)).mean())
    return float(np.exp(nll.mean())), nll


def _catch_block0(m, cal):
    """Run cal seqs up to decoder block 0; capture its inputs + (shared) kwargs with the KV cache nulled."""
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


def diagnose_instrument(cal):
    """CONTROL: how far does the Hessian instrument move between fp32 and bf16 activations (block 0, 6 linears)?"""
    print("[CONTROL] instrument shift on block 0 -- how much does the Hessian even move?")
    m = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    inps, kw0 = _catch_block0(m, cal)
    blk = m.model.decoder.layers[0]; U = _units(blk)
    Hc = {id(u): None for u in U}; Hb = {id(u): None for u in U}; names = {}
    for nm, u in [("q", U[0]), ("k", U[1]), ("v", U[2]), ("out", U[3]), ("fc1", U[4]), ("fc2", U[5])]:
        names[id(u)] = nm
    def mk(u):
        def hook(mod, ip):
            x = ip[0].reshape(-1, ip[0].size(-1)).float().to(QDEV)
            xb = x.to(torch.bfloat16).float()
            hc = x.t() @ x; hb = xb.t() @ xb
            Hc[id(u)] = hc if Hc[id(u)] is None else Hc[id(u)] + hc
            Hb[id(u)] = hb if Hb[id(u)] is None else Hb[id(u)] + hb
        return u.register_forward_pre_hook(hook)
    hs = [mk(u) for u in U]
    for j in range(len(inps)):
        blk(inps[j], **kw0)
    for h in hs:
        h.remove()
    print(f"    {'linear':>5} {'||dH||F/||H||F':>15} {'diag(H) reldiff':>16} {'diag(Hinv) reldiff':>18}")
    rows = {}
    for u in U:
        Hcc, Hbb = Hc[id(u)], Hb[id(u)]
        fro = float((Hbb - Hcc).norm() / Hcc.norm().clamp_min(1e-30))
        dc, db = torch.diagonal(Hcc), torch.diagonal(Hbb)
        ddiag = float(((db - dc).abs() / dc.abs().clamp_min(1e-30)).mean())
        ic, ib = _hinv_diag(Hcc), _hinv_diag(Hbb)
        dinv = float(((ib - ic).abs() / ic.abs().clamp_min(1e-30)).mean())
        rows[names[id(u)]] = (fro, ddiag, dinv)
        print(f"    {names[id(u)]:>5} {fro:>15.2e} {ddiag:>16.2e} {dinv:>18.2e}")
    del m; gc.collect(); torch.cuda.empty_cache()
    return rows


def reconstruct(hess_prec, cal, ev, tgts, want_gold=False):
    """Full sequential INT4-g128 GPTQ pass. hess_prec in {'fp32','bf16'} controls ONLY the Hessian activation prec."""
    m = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(DEV).eval()
    gold = ppl(m, ev, tgts) if want_gold else None
    inps, kw0 = _catch_block0(m, cal)
    layers = m.model.decoder.layers
    for blk in layers:
        U = _units(blk); H = {id(u): None for u in U}; cnt = {id(u): 0 for u in U}
        def mk(u):
            def hook(mod, ip):
                x = ip[0].reshape(-1, ip[0].size(-1)).float()
                if hess_prec == "bf16":
                    x = x.to(torch.bfloat16).float()          # cheap arm: activation precision = bf16, signal identical
                x = x.to(QDEV); h = x.t() @ x                  # Hessian accumulated on cuda:1
                H[id(u)] = h if H[id(u)] is None else H[id(u)] + h; cnt[id(u)] += x.shape[0]
            return u.register_forward_pre_hook(hook)
        hs = [mk(u) for u in U]
        for j in range(len(inps)):
            blk(inps[j], **kw0)
        for h in hs:
            h.remove()
        for u in U:
            Hq = H[id(u)] / max(cnt[id(u)], 1)
            W4 = gptq_int4(u.weight.detach().to(QDEV), Hq, QMAX, G)
            u.weight.data = W4.to(u.weight.device).to(u.weight.dtype)
            H[id(u)] = None; del Hq, W4
        gc.collect(); torch.cuda.empty_cache()
        for j in range(len(inps)):
            o = blk(inps[j], **kw0)
            inps[j] = o[0] if isinstance(o, (tuple, list)) else o
    p = ppl(m, ev, tgts); del m; gc.collect(); torch.cuda.empty_cache()
    return gold, p


def paired_bootstrap(nll_a, nll_b, B=10000, seed=0):
    """Paired bootstrap on log-PPL difference (cheap - cert). Mean NLL diff == log-PPL ratio."""
    rng = np.random.default_rng(seed); n = len(nll_a); diff = nll_b - nll_a
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run():
    print("[OPT-2.7B CERTIFIED HESSIAN]  fp32 vs bf16 activation Hessian -> does GPTQ's result change?\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = load_corpus(tok, N_CAL + N_EVAL); cal, ev = seqs[:N_CAL], seqs[N_CAL:]
    tgts = [s[0, 1:] for s in ev]

    rows = diagnose_instrument(cal)
    print()

    print("[SCAN] full sequential INT4-g128 reconstruction, certified (fp32) vs cheap (bf16) Hessian")
    gold, (p_cert, nll_cert) = reconstruct("fp32", cal, ev, tgts, want_gold=True)
    gold_ppl, _ = gold
    _, (p_cheap, nll_cheap) = reconstruct("bf16", cal, ev, tgts, want_gold=False)

    md, lo, hi = paired_bootstrap(nll_cert, nll_cheap)            # cheap - cert in log-PPL
    print(f"\n  {'config':>22} {'PPL':>9} {'dPPL vs gold':>13}")
    print(f"  {'fp32 gold':>22} {gold_ppl:>9.3f} {'--':>13}")
    print(f"  {'GPTQ / certified H':>22} {p_cert:>9.3f} {100*(p_cert/gold_ppl-1):>+12.2f}%")
    print(f"  {'GPTQ / cheap (bf16) H':>22} {p_cheap:>9.3f} {100*(p_cheap/gold_ppl-1):>+12.2f}%")
    print(f"\n  arm difference (cheap - certified), log-PPL paired bootstrap:")
    print(f"    mean {md:>+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]   (== {100*(np.exp(md)-1):+.3f}% PPL)")
    sig = (lo > 0) or (hi < 0)
    print(f"    significant at 95%: {sig}  (CI {'excludes' if sig else 'straddles'} zero)")

    print("\n[VERDICT]")
    if not sig:
        print("  CLEAN NULL: certified (fp32) and cheap (bf16) activation Hessians produce statistically")
        print("  indistinguishable deployed perplexity. The GPTQ Hessian -- like the certified reference before it")
        print("  -- is ROBUST to activation precision. The instrument's precision is not the lever; the use-aware")
        print("  STRUCTURE of the Hessian is. Honest, registered outcome: the novel certified question is a null.")
    elif hi < 0:
        print("  CERTIFIED WINS: the fp32 Hessian yields significantly better deployed PPL. The founding instrument")
        print("  vindicated on the RIGHT lever -- certification matters where it operates on use-aware quantization.")
    else:
        print("  CHEAP WINS: the bf16 Hessian is significantly better -- certification actively hurts even here.")
    print("  (Cross-check against [CONTROL]: a null PPL is consistent with a small instrument shift ||dH||/||H||.)")
    print("\n[V-hw] RTX 5070 sm_120 + GTX 1660 Ti, OPT-2.7B, fp32 master, sequential INT4-g128 GPTQ, WikiText-2. Held-out.")
    return rows, gold_ppl, p_cert, p_cheap, (md, lo, hi)


if __name__ == "__main__":
    run()
