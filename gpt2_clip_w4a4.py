"""CLIPPING test for rotated W4A4: can a cheap (storage-free) per-token clip close the +44% residual?

v0.2.24-26: rotation rescues W4A4 (GPT-2 +44% unpadded; padding closes it but costs storage; basis is inert).
The remaining storage-free fidelity knob is CLIPPING: after rotation, set the per-token INT4 scale from a
fraction alpha of the absmax (s = alpha*absmax/7), trading a little clipping of residual outliers for finer
quantization of the bulk. Sweep alpha on the unpadded random-orthogonal rotation (NO storage cost). If a good
alpha closes +44% toward the padded +20% (or near-lossless) for free, it's a cheap win; if it barely moves, the
residual is the per-token INT4 floor on this small model and near-lossless needs a bigger model / the full scheme.

PRE-REG (honest): clipping helps (optimal alpha < 1, bias-variance) but rotation already spread the outliers, so
limited gains -- I expect it shaves the residual modestly (maybe +44% -> ~+25-35%), not to near-lossless.
[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, FP4-g128+act-order wt + per-token INT4 acts, random rotation.
"""
import torch
import gpt2_eigensweep as E

torch.set_grad_enabled(False)
DEV = E.DEV; GPT2 = E.GPT2; L = E.L; GROUP = E.GROUP; N_CAL = 32
ACT = {"bits": None, "clip": 1.0}; QC = {}; CLIPS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
torch.manual_seed(0)


def get_Q(in_d):
    if in_d not in QC:
        QC[in_d] = torch.linalg.qr(torch.randn(in_d, in_d, device=DEV))[0].float()
    return QC[in_d]


def mk_hook(Q):
    def hook(mod, args):
        x = args[0]
        if Q is not None:
            x = x.float() @ Q
        if ACT["bits"] == 4:
            x2 = x.reshape(-1, x.size(-1))
            s = (ACT["clip"] * x2.abs().amax(1, keepdim=True)).clamp_min(1e-12) / 7.0
            x = (torch.clamp(torch.round(x2 / s), -7, 7) * s).view(*x.shape)
        return (x.to(args[0].dtype),) + tuple(args[1:])
    return hook


def reconstruct(rotate, T, Hm, orig):
    hs = []
    for n, mod in T:
        mod.weight.data = orig[n].to(DEV).clone()
    for n, mod in T:
        W = mod.weight.data.float()
        if rotate:
            Q = get_Q(W.shape[0]); W4 = E.quant(Q.t() @ W, Q.t() @ Hm[n].to(DEV) @ Q, 4, GROUP, True)
            hs.append(mod.register_forward_pre_hook(mk_hook(Q)))
        else:
            W4 = E.quant(W, Hm[n].to(DEV), 4, GROUP, True); hs.append(mod.register_forward_pre_hook(mk_hook(None)))
        mod.weight.data = W4
    return hs


def run():
    print("[GPT-2 CLIPPING for rotated W4A4]  can a storage-free clip close the +44% residual?\n")
    tr = __import__("transformers"); tok = tr.AutoTokenizer.from_pretrained(GPT2)
    ids = E.corpus_ids(tok); eval_ids = ids[:8192].to(DEV)
    cal = [ids[8192 + i * L:8192 + (i + 1) * L].unsqueeze(0) for i in range(N_CAL)]
    m = tr.AutoModelForCausalLM.from_pretrained(GPT2).to(torch.float32).to(DEV).eval()
    T = E.targets(m); orig = {n: mod.weight.detach().cpu().clone() for n, mod in T}
    Hm, _ = E.collect(m, cal, T)
    gold = E.ppl_strided(m, eval_ids); print(f"  gold PPL = {gold:.3f}\n")

    QC.clear(); hs = reconstruct(False, T, Hm, orig)
    ACT["bits"] = 4; ACT["clip"] = 1.0; n4 = E.ppl_strided(m, eval_ids)
    for h in hs:
        h.remove()
    print(f"  naive W4A4 (no rotation, no clip): {n4:.1f} ({100*(n4/gold-1):+.0f}%)\n")

    QC.clear(); hs = reconstruct(True, T, Hm, orig)
    print(f"  rotated (random-ortho, unpadded) W4A4 -- clip sweep:")
    print(f"    {'clip a':>7} {'W4A4 PPL':>10} {'dPPL vs gold':>13}")
    best = None
    for a in CLIPS:
        ACT["bits"] = 4; ACT["clip"] = a; p = E.ppl_strided(m, eval_ids)
        print(f"    {a:>7.2f} {p:>10.3f} {100*(p/gold-1):>+12.1f}%")
        if best is None or p < best[1]:
            best = (a, p)
    for h in hs:
        h.remove()
    for n, mod in T:
        mod.weight.data = orig[n].to(DEV)

    print(f"\n[VERDICT]")
    print(f"  best clip a={best[0]:.2f}: W4A4 +{100*(best[1]/gold-1):.1f}%  (no-clip a=1.0 was the first row)")
    print(f"  reference: padded rotation (1.33x storage) was +20%; near-lossless would be sub-few-%.")
    print("  -> if best-clip is well below +44% and approaching the padded +20% FOR FREE, clipping is the cheap")
    print("     storage-free fidelity knob. If it barely moves, +44% is the per-token-INT4 floor on this small model")
    print("     and near-lossless W4A4 needs a bigger model / the full residual-stream scheme.")
    print("\n[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, FP4-g128+act-order wt + per-token INT4 acts.")
    return gold, n4, best


if __name__ == "__main__":
    run()
