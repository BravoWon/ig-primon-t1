"""HADAMARD vs random rotation for W4A4: is the residual the BASIS or the SCHEME?

v0.2.23-24: a per-linear RANDOM orthogonal rotation rescues W4A4 (GPT-2 +44%, OPT +24%) but leaves a residual.
Claim under test: swapping the random rotation for a structured/randomized HADAMARD closes the residual toward
near-lossless. PRE-REG (honest, against the claim): a randomized Hadamard is the FAST version of a random
orthogonal rotation, not the BETTER one -- both achieve incoherence. I predict Hadamard ~ random in PPL (the
residual is the SIMPLIFIED SCHEME -- per-linear rotation only, no residual-stream rotation -- not the basis).
The Hadamard win is SPEED (O(d log d) via fast WHT), not quality. Falsifier for my prediction: Hadamard << random.

Three arms at W4A4 (FP4-g128+act-order weights + per-token INT4 acts): naive (Q=I), random-orthogonal,
randomized-Hadamard (sign-diagonal x Sylvester Hadamard, padded to next power of 2). Control: rotated-W4A16 ~
naive-W4A16 (fold identity).
[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import gpt2_eigensweep as E

torch.set_grad_enabled(False)
DEV = E.DEV; GPT2 = E.GPT2; L = E.L; GROUP = E.GROUP; N_CAL = 32
ACT = {"bits": None}; QC = {}
torch.manual_seed(0)


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def _hadamard(D):
    H = torch.ones(1, 1, device=DEV)
    while H.shape[0] < D:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (D ** 0.5)


def get_Q(in_d, mode):
    key = (in_d, mode)
    if key not in QC:
        if mode == "ortho":                                         # random orthogonal, UNPADDED (d x d)
            QC[key] = (torch.linalg.qr(torch.randn(in_d, in_d, device=DEV))[0].float(), in_d)
        elif mode == "ortho_pad":                                   # random orthogonal, PADDED to next pow2 (control)
            D = _next_pow2(in_d); QC[key] = (torch.linalg.qr(torch.randn(D, D, device=DEV))[0].float(), D)
        else:                                                        # randomized Hadamard, padded to next pow2
            D = _next_pow2(in_d); H = _hadamard(D)
            S = (torch.randint(0, 2, (D,), device=DEV).float() * 2 - 1)
            QC[key] = ((S.unsqueeze(1) * H).float(), D)
    return QC[key]


def mk_hook(Q, in_d, D, pad):
    def hook(mod, args):
        x = args[0]
        if Q is not None:
            if pad:
                x = Fn.pad(x.reshape(-1, in_d), (0, D - in_d)).view(*x.shape[:-1], D)
            x = x.float() @ Q                                        # rotate
        if ACT["bits"] == 4:                                         # per-token INT4 (applies in EVERY arm, incl. naive)
            x2 = x.reshape(-1, x.size(-1))
            s = x2.abs().amax(1, keepdim=True).clamp_min(1e-12) / 7.0
            x = (torch.clamp(torch.round(x2 / s), -7, 7) * s).view(*x.shape)
        return (x.to(args[0].dtype),) + tuple(args[1:])
    return hook


def reconstruct(mode, T, Hm, orig):
    hs = []
    for n, mod in T:
        mod.weight.data = orig[n].to(DEV).clone()
    for n, mod in T:
        W = mod.weight.data.float(); in_d = W.shape[0]              # Conv1D (in, out)
        if mode == "none":
            W4 = E.quant(W, Hm[n].to(DEV), 4, GROUP, True)
            hs.append(mod.register_forward_pre_hook(mk_hook(None, in_d, in_d, False)))   # act-quant only, no rotation
        else:
            Q, D = get_Q(in_d, mode); pad = (D != in_d)
            Wp = Fn.pad(W, (0, 0, 0, D - in_d)) if pad else W        # (D, out)
            Hp = Fn.pad(Hm[n].to(DEV), (0, D - in_d, 0, D - in_d)) if pad else Hm[n].to(DEV)
            Hr = Q.t() @ Hp @ Q; Wr = Q.t() @ Wp                    # fold: W'=Q^T W, H'=Q^T H Q
            W4 = E.quant(Wr, Hr, 4, GROUP, True)
            hs.append(mod.register_forward_pre_hook(mk_hook(Q, in_d, D, pad)))
        mod.weight.data = W4
    return hs


def run():
    print("[GPT-2 HADAMARD vs random rotation, W4A4]  is the residual the BASIS or the SCHEME?\n")
    tok = __import__("transformers").AutoTokenizer.from_pretrained(GPT2)
    ids = E.corpus_ids(tok); eval_ids = ids[:8192].to(DEV)
    cal = [ids[8192 + i * L:8192 + (i + 1) * L].unsqueeze(0) for i in range(N_CAL)]
    m = __import__("transformers").AutoModelForCausalLM.from_pretrained(GPT2).to(torch.float32).to(DEV).eval()
    T = E.targets(m); orig = {n: mod.weight.detach().cpu().clone() for n, mod in T}
    Hm, _ = E.collect(m, cal, T)
    gold = E.ppl_strided(m, eval_ids); print(f"  gold PPL = {gold:.3f}\n")

    res = {}
    nm = {"none": "naive (Q=I)", "ortho": "random-ortho (unpad)", "ortho_pad": "random-ortho (padded)",
          "hadamard": "rand-Hadamard (padded)"}
    for mode in ["none", "ortho", "ortho_pad", "hadamard"]:
        QC.clear()
        hs = reconstruct(mode, T, Hm, orig)
        ACT["bits"] = None; p16 = E.ppl_strided(m, eval_ids)
        ACT["bits"] = 4;    p4 = E.ppl_strided(m, eval_ids)
        for h in hs:
            h.remove()
        res[mode] = (p16, p4)
        print(f"  {nm[mode]:>22}: W4A16 {p16:>9.3f} ({100*(p16/gold-1):+.2f}%)   W4A4 {p4:>11.3f} ({100*(p4/gold-1):+.2f}%)")
    for n, mod in T:
        mod.weight.data = orig[n].to(DEV)

    print("\n[VERDICT -- scored against pre-registration]")
    n4 = res["none"][1]; o4 = res["ortho"][1]; op4 = res["ortho_pad"][1]; h4 = res["hadamard"][1]
    print(f"  W4A4: naive +{100*(n4/gold-1):.0f}%  |  ortho-unpad +{100*(o4/gold-1):.0f}%  |  "
          f"ortho-PADDED +{100*(op4/gold-1):.0f}%  |  Hadamard +{100*(h4/gold-1):.0f}%")
    print(f"  padding effect (ortho-pad vs ortho-unpad): {100*(op4/o4-1):+.1f}%")
    print(f"  Hadamard-structure effect (Hadamard vs ortho-PADDED, matched dims): {100*(h4/op4-1):+.1f}%")
    if h4 < op4 * 0.9:
        print("  -> HADAMARD STRUCTURE genuinely helps beyond padding (prediction WRONG): the structured basis closes")
        print("     more residual at matched dimensionality.")
    elif abs(h4 / op4 - 1) < 0.1:
        print("  -> Hadamard ~ padded-random (prediction HOLDS): the win is the PADDING (more channels to smear into),")
        print("     not the Hadamard structure. The basis isn't the lever; near-lossless needs the full scheme.")
    else:
        print("  -> mixed -- report straight.")
    print("\n[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, FP4-g128+act-order weights + per-token INT4 acts.")
    return gold, res


if __name__ == "__main__":
    run()
