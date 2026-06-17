"""FRONTIER: rotation-based W4A4 (QuaRot's core mechanism) -- does an orthogonal rotation rescue 4-bit activations?

The activation cliff (v0.2.11-12): naive 4-bit activations collapse on per-channel outliers; SmoothQuant is
necessary-but-insufficient. The named-not-built frontier is ROTATION: insert an orthogonal Q into each linear so
the *rotated* activations have no per-channel outliers (rotation spreads outlier energy across channels --
incoherence), while folding Q into the weight keeps the full-precision output unchanged. Then 4-bit activation
quant of the rotated activations survives. This is the spread-spectrum finding operationalized: you can't truncate
the outliers, but you can ROTATE them into the bulk.

Construction (per Conv1D linear y = x W, W is (in,out)):
  rotate input x' = x Q ; fold W' = Q^T W  (so x' W' = x Q Q^T W = x W, exact in fp32).
  Quantize W' via GPTQ (rotated Hessian H' = Q^T H Q); at eval quantize the rotated activation x' = xQ to 4-bit.
  Naive arm = same with Q = I. Q = random orthogonal (the incoherence effect; deployable version uses Hadamard).

PRE-REG: rotation rescues W4A4 (rotated << naive). CONTROL: rotated-W4A16 ~ gold (rotation exact in fp -> gates
the fold). Falsifier: rotated ~ naive -> rotation doesn't help, OR gpt2-small lacks severe outliers (then the
honest demo needs OPT, where the collapse is established at +37,332%).
[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, GPTQ-4bit weights + per-token 4-bit acts, random rotation.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import gpt2_eigensweep as E

torch.set_grad_enabled(False)
DEV = E.DEV; GPT2 = E.GPT2; L = E.L; GROUP = E.GROUP; N_CAL = 32
A4 = {"on": False}; Qs = {}
torch.manual_seed(0)


def mk_hook(Q):
    def hook(mod, inp):
        x = inp[0]; xr = x @ Q                                       # rotate the in-dim
        if A4["on"]:
            x2 = xr.reshape(-1, xr.size(-1)).float()
            s = x2.abs().amax(1, keepdim=True).clamp_min(1e-12) / 7.0
            xr = (torch.clamp(torch.round(x2 / s), -7, 7) * s).to(xr.dtype).view(*xr.shape)
        return (xr,) + tuple(inp[1:])
    return hook


def install(T, Hm, rotate):
    handles = []
    for n, mod in T:
        W = mod.weight.data; d = W.shape[0]                          # (in,out)
        if rotate:
            if d not in Qs:
                Qs[d] = torch.linalg.qr(torch.randn(d, d, device=DEV))[0]
            Q = Qs[d]
        else:
            Q = torch.eye(d, device=DEV)
        Hr = Q.t() @ Hm[n].to(DEV) @ Q                              # rotated Hessian
        Wr = Q.t() @ W                                              # folded weight (in,out)
        mod.weight.data = E.quant(Wr, Hr, 4, GROUP, True)          # GPTQ-4bit on the rotated weight
        handles.append(mod.register_forward_pre_hook(mk_hook(Q)))
    return handles


def run():
    print("[FRONTIER: rotation-based W4A4]  does an orthogonal rotation rescue 4-bit activations?\n")
    tok = AutoTokenizer.from_pretrained(GPT2); ids = E.corpus_ids(tok)
    eval_ids = ids[:8192].to(DEV)
    cal = [ids[8192 + i * L: 8192 + (i + 1) * L].unsqueeze(0) for i in range(N_CAL)]
    m = AutoModelForCausalLM.from_pretrained(GPT2).to(torch.float32).to(DEV).eval()
    T = E.targets(m); orig = {n: mod.weight.detach().cpu().clone() for n, mod in T}
    Hm, _ = E.collect(m, cal, T)
    gold = E.ppl_strided(m, eval_ids); print(f"  gold PPL = {gold:.3f}\n")

    def restore(handles):
        for h in handles:
            h.remove()
        for n, mod in T:
            mod.weight.data = orig[n].to(DEV).clone()

    res = {}
    for arm, rot in [("naive (Q=I)", False), ("rotated (Q=orthogonal)", True)]:
        h = install(T, Hm, rot)
        A4["on"] = False; p16 = E.ppl_strided(m, eval_ids)
        A4["on"] = True;  p4 = E.ppl_strided(m, eval_ids)
        res[arm] = (p16, p4); restore(h); A4["on"] = False

    print(f"  {'arm':>24} {'W4A16':>10} {'W4A4':>12}")
    print(f"  {'gold':>24} {gold:>10.3f} {'--':>12}")
    for arm in ["naive (Q=I)", "rotated (Q=orthogonal)"]:
        p16, p4 = res[arm]
        print(f"  {arm:>24} {p16:>10.3f} {p4:>12.2f}    (W4A16 {100*(p16/gold-1):+.2f}%, W4A4 {100*(p4/gold-1):+.2f}%)")

    print("\n[VERDICT -- scored against pre-registration]")
    ctrl = res["rotated (Q=orthogonal)"][0]
    ctrl_ok = abs(ctrl / gold - 1) < 0.05
    nv4 = res["naive (Q=I)"][1]; rt4 = res["rotated (Q=orthogonal)"][1]
    print(f"  CONTROL rotated-W4A16 = {ctrl:.3f} vs gold {gold:.3f}: fold-is-identity {'OK' if ctrl_ok else 'FAIL (rotation buggy)'}")
    if not ctrl_ok:
        print("  -> control failed; do not trust the A4 numbers.")
    elif nv4 > gold * 2 and rt4 < nv4 * 0.5:
        print(f"  -> ROTATION RESCUES W4A4: naive {100*(nv4/gold-1):+.0f}% -> rotated {100*(rt4/gold-1):+.0f}%. The")
        print(f"     spread-spectrum operationalized: rotate outliers into the bulk, 4-bit activations survive.")
    elif nv4 < gold * 1.5:
        print(f"  -> gpt2-small W4A4 did NOT collapse (naive {100*(nv4/gold-1):+.0f}%): too few outliers at 124M to")
        print(f"     show the rescue. The honest demo needs OPT-2.7B (collapse established at +37,332%). Rotation")
        print(f"     effect here: {100*(rt4/nv4-1):+.1f}% vs naive.")
    else:
        print(f"  -> mixed: naive {100*(nv4/gold-1):+.0f}%, rotated {100*(rt4/gold-1):+.0f}% -- report straight.")
    print("\n[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, GPTQ-4bit wt + per-token 4-bit act, random rotation.")
    return gold, res


if __name__ == "__main__":
    run()
