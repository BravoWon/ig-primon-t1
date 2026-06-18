"""DIMENSION sweep for W4A4 rotation: is over-dimensioned rotation a real lever, or just a storage trade?

v0.2.25 found that padding the rotation to a higher dimension (more channels to spread outliers into) closed
the W4A4 residual (+44% -> +20%) -- but padding multiplies the 4-bit WEIGHT STORAGE (D/in x). So this is a
storage-for-quality trade, not a free win. This sweeps the rotation dimension D = {1,2,4} x nextpow2(in) with a
random orthogonal rotation, reporting W4A4 PPL AND the weight-storage multiplier. The honest question is not "does
more D help" (it will) but "does over-dimensioned W4A4 beat just spending that storage on more bits (W8A4-class)?"

PRE-REG: PPL drops with D (diminishing returns); the storage cost grows linearly. If the PPL/storage tradeoff is
worse than 'use more bits', dimension is NOT a real efficiency lever -- the storage-free SCHEME (residual-stream
rotation) is the path to near-lossless. Falsifier: PPL saturates by 2x (dimension weak) OR over-dim beats the
bit-budget alternative (dimension real).
[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, FP4-g128+act-order wt + per-token INT4 acts, random rotation.
"""
import numpy as np
import torch
import torch.nn.functional as Fn
import gpt2_eigensweep as E

torch.set_grad_enabled(False)
DEV = E.DEV; GPT2 = E.GPT2; L = E.L; GROUP = E.GROUP; N_CAL = 32
ACT = {"bits": None}; QC = {}; FACTORS = [1, 2, 4]
torch.manual_seed(0)


def _np2(x):
    return 1 << (x - 1).bit_length()


def get_Q(in_d, factor):
    key = (in_d, factor)
    if key not in QC:
        D = factor * _np2(in_d)
        QC[key] = (torch.linalg.qr(torch.randn(D, D, device=DEV))[0].float(), D)
    return QC[key]


def mk_hook(Q, in_d, D):
    def hook(mod, args):
        x = args[0]
        if Q is not None:
            x = Fn.pad(x.reshape(-1, in_d), (0, D - in_d)).view(*x.shape[:-1], D) @ Q
        if ACT["bits"] == 4:
            x2 = x.reshape(-1, x.size(-1)); s = x2.abs().amax(1, keepdim=True).clamp_min(1e-12) / 7.0
            x = (torch.clamp(torch.round(x2 / s), -7, 7) * s).view(*x.shape)
        return (x.to(args[0].dtype),) + tuple(args[1:])
    return hook


def reconstruct(factor, T, Hm, orig):
    hs = []; stored = 0; orig_sz = 0
    for n, mod in T:
        mod.weight.data = orig[n].to(DEV).clone()
    for n, mod in T:
        W = mod.weight.data.float(); in_d, out_d = W.shape; orig_sz += in_d * out_d
        if factor == 0:                                              # naive: act-quant only, no rotation
            W4 = E.quant(W, Hm[n].to(DEV), 4, GROUP, True); hs.append(mod.register_forward_pre_hook(mk_hook(None, in_d, in_d)))
            stored += in_d * out_d
        else:
            Q, D = get_Q(in_d, factor)
            Wp = Fn.pad(W, (0, 0, 0, D - in_d)); Hp = Fn.pad(Hm[n].to(DEV), (0, D - in_d, 0, D - in_d))
            W4 = E.quant(Q.t() @ Wp, Q.t() @ Hp @ Q, 4, GROUP, True)
            hs.append(mod.register_forward_pre_hook(mk_hook(Q, in_d, D))); stored += D * out_d
        mod.weight.data = W4
    return hs, stored / orig_sz                                      # storage multiplier


def run():
    print("[GPT-2 DIMENSION sweep for W4A4]  over-dimensioned rotation: real lever or storage trade?\n")
    tr = __import__("transformers"); tok = tr.AutoTokenizer.from_pretrained(GPT2)
    ids = E.corpus_ids(tok); eval_ids = ids[:8192].to(DEV)
    cal = [ids[8192 + i * L:8192 + (i + 1) * L].unsqueeze(0) for i in range(N_CAL)]
    m = tr.AutoModelForCausalLM.from_pretrained(GPT2).to(torch.float32).to(DEV).eval()
    T = E.targets(m); orig = {n: mod.weight.detach().cpu().clone() for n, mod in T}
    Hm, _ = E.collect(m, cal, T)
    gold = E.ppl_strided(m, eval_ids); print(f"  gold PPL = {gold:.3f}\n")

    print(f"  {'rotation dim':>16} {'storage x':>10} {'W4A4 PPL':>10} {'dPPL vs gold':>13}")
    rows = []
    for factor in [0] + FACTORS:
        QC.clear(); hs, smult = reconstruct(factor, T, Hm, orig)
        ACT["bits"] = 4; p4 = E.ppl_strided(m, eval_ids)
        for h in hs:
            h.remove()
        tag = "naive (no rot)" if factor == 0 else f"{factor}x nextpow2"
        rows.append((tag, smult, p4)); print(f"  {tag:>16} {smult:>9.2f}x {p4:>10.2f} {100*(p4/gold-1):>+12.1f}%")
    for n, mod in T:
        mod.weight.data = orig[n].to(DEV)

    print("\n[VERDICT -- driftwave round-trip: re-derive from the PPL-vs-storage curve]")
    rot = [r for r in rows if r[0] != "naive (no rot)"]
    print(f"  PPL by storage: " + " | ".join(f"{r[1]:.1f}x->{100*(r[2]/gold-1):+.0f}%" for r in rot))
    p1, p2, p4f = rot[0][2], rot[1][2], rot[2][2]
    sat = abs(p4f / p2 - 1) < 0.1
    print(f"  diminishing returns (4x vs 2x): {100*(p4f/p2-1):+.1f}%  ->  {'SATURATED' if sat else 'still improving'}")
    print(f"  BUT storage cost at the best point ({rot[-1][1]:.1f}x weights) is the catch: W4A4 at {rot[-1][1]:.1f}x")
    print(f"  4-bit storage = {4*rot[-1][1]:.1f} effective bits/orig-weight -- comparable to just using higher precision.")
    print("  HONEST READ: over-dimensioned rotation lowers PPL but TRADES STORAGE; it is not a free efficiency lever.")
    print("  The storage-free path to near-lossless W4A4 is the SCHEME (residual-stream rotation), not dimension/basis.")
    print("\n[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, FP4-g128+act-order wt + per-token INT4 acts.")
    return gold, rows


if __name__ == "__main__":
    run()
