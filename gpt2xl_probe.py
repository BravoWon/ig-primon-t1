"""SCALE-UP validity probe: is GPT-2-XL (1.5B) a valid testbed for the certification/SmoothQuant test?

Control-before-scan. Before re-running the certification edge at scale, confirm three things:
  (L) fp32 anchor is LICENSED. fp64 of a 1.5B model is intractable, so fp32 becomes the certified anchor.
      License it the Firewall way: per-op fp32-vs-fp64 matmul agreement on real weights + real activations
      (a few layers). If fp32 ~= fp64 at the op level, fp32 is the trustworthy reference here.
  (O) the OUTLIER pathology is STRONGER than at 124M. gpt2-small had median per-channel activation ratio
      9.7x (max 119x), too mild for SmoothQuant. Does 12x scale (same family) awaken it?
  (S) SmoothQuant now CRUSHES the floor. If the pathology is real, migration should reduce all-W8A8 error
      (unlike 124M, where it made it worse). If it still doesn't, even 1.5B GPT-2 lacks the pathology and the
      meaningful SmoothQuant test needs OPT-class.

Only if (L) passes and (O)+(S) show a real, mitigable pathology do we run the full certification re-test.
[V-hw] RTX 5070 sm_120. fp32 anchor (spot-licensed vs fp64). No claim until these gates pass.
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

torch.set_grad_enabled(False)
DEV = "cuda:0"; FP8 = torch.float8_e4m3fn
GPT2XL_PATH = "C:/Users/JT-DEV1/Documents/gpt2-xl"
ALPHA = 0.5
TEXTS = [
    "The history of numerical analysis begins with the study of round-off error in finite-precision arithmetic.",
    "Large language models compose many layers, each performing matrix multiplications and nonlinear maps.",
    "Quantization reduces memory but can amplify errors at sharp attention logits and saturated activations.",
    "In the morning the fishermen returned with their catch and sold it at the market by the harbor.",
]


def units(model):
    out = []
    for blk in model.transformer.h:
        out += [blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj]
    return out


def kl(ref, q):
    rl = ref.double(); ql = q.double()
    lpr = torch.log_softmax(rl, -1); lpq = torch.log_softmax(ql, -1)
    return (lpr.exp() * (lpr - lpq)).sum(-1).mean().item()


def patched(self, x):
    size_out = x.size()[:-1] + (self.nf,)
    x2d = x.reshape(-1, x.size(-1)); W = self.weight; b = self.bias
    mode = getattr(self, "_prec", "hi")
    if mode == "hi":
        out = torch.addmm(b, x2d, W)
    elif mode == "fp8t":
        sx = x2d.float().abs().amax().clamp_min(1e-12) / 448.0
        sw = W.float().abs().amax().clamp_min(1e-12) / 448.0
        out = torch.addmm(b, ((x2d.float()/sx).to(FP8).float()*sx).to(x2d.dtype),
                          ((W.float()/sw).to(FP8).float()*sw).to(W.dtype))
    elif mode == "fp8pc":
        sx = x2d.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        sw = W.float().abs().amax(0, keepdim=True).clamp_min(1e-12) / 448.0
        out = torch.addmm(b, ((x2d.float()/sx).to(FP8).float()*sx).to(x2d.dtype),
                          ((W.float()/sw).to(FP8).float()*sw).to(W.dtype))
    else:  # smooth
        s = self._s
        xs = x2d / s; Ws = W * s[:, None]
        sx = xs.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        sw = Ws.float().abs().amax(0, keepdim=True).clamp_min(1e-12) / 448.0
        out = torch.addmm(b, ((xs.float()/sx).to(FP8).float()*sx).to(x2d.dtype),
                          ((Ws.float()/sw).to(FP8).float()*sw).to(W.dtype))
    return out.view(size_out)


def run():
    import os
    if not os.path.exists(GPT2XL_PATH):
        print(f"[probe] model not found at {GPT2XL_PATH} -- download openai-community/gpt2-xl there first.")
        return False
    print("[GPT-2-XL probe] validity gates before any certification re-test\n")
    tok = GPT2TokenizerFast.from_pretrained(GPT2XL_PATH)
    Conv1D.forward = patched
    mg = GPT2LMHeadModel.from_pretrained(GPT2XL_PATH).to(torch.bfloat16).to(DEV).eval()
    m32 = GPT2LMHeadModel.from_pretrained(GPT2XL_PATH).to(torch.float32).to(DEV).eval()
    U = units(mg); nU = len(U)
    print(f"  loaded: {mg.config.n_layer} layers, d={mg.config.n_embd}, {nU} matmul units\n")

    def ids_of(t):
        return tok(t, return_tensors="pt").input_ids[:, :32]

    # (L) license fp32 anchor: per-op fp32 vs fp64 on real weights + captured activations (from m32 itself)
    U32 = units(m32)
    store = {}; hs = []
    def mk(u):
        def hook(mod, inp):
            store.setdefault(id(u), inp[0].reshape(-1, inp[0].size(-1)).detach())
        return u.register_forward_pre_hook(hook)
    for u in U32:
        hs.append(mk(u))
    _ = m32(ids_of(TEXTS[0]).to(DEV))
    for h in hs:
        h.remove()
    rels = []
    for u in (U32[0], U32[len(U32)//2], U32[-1]):
        X = store[id(u)].float()
        W = u.weight.float()
        o32 = (X @ W); o64 = (X.double() @ W.double())
        rels.append(float((o32.double() - o64).norm() / o64.norm().clamp_min(1e-30)))
    print(f"(L) fp32 anchor license: per-op fp32-vs-fp64 rel err = {max(rels):.1e} (<= ~1e-5 -> fp32 licensed)")
    licensed = max(rels) < 1e-5

    # (O) outlier pathology
    store2 = {}; hs = []
    def mk2(u):
        def hook(mod, inp):
            a = inp[0].reshape(-1, inp[0].size(-1)).float().abs().amax(0)
            store2[id(u)] = a if id(u) not in store2 else torch.maximum(store2[id(u)], a)
        return u.register_forward_pre_hook(hook)
    for u in U:
        hs.append(mk2(u))
    for t in TEXTS:
        mg(ids_of(t).to(DEV))
    for h in hs:
        h.remove()
    ratios = np.array([(store2[id(u)].max() / store2[id(u)].median().clamp_min(1e-12)).item() for u in U])
    print(f"(O) outlier ratio (max/median per channel): median={np.median(ratios):.1f}  mean={ratios.mean():.1f}  "
          f"max={ratios.max():.1f}   (gpt2-small was median 9.7 / max 119)")
    stronger = np.median(ratios) > 9.7

    # build fp32-reference floor + SmoothQuant
    def mean_kl(mode):
        for u in U:
            u._prec = mode
        ks = []
        for t in TEXTS:
            ids = ids_of(t); ref = m32(ids.to(DEV)).logits[0].float().cpu()
            ks.append(kl(ref, mg(ids.to(DEV)).logits[0].float().cpu()))
        return float(np.mean(ks))
    # calibrate s for smooth
    for u in U:
        amax = store2[id(u)].clamp_min(1e-12)
        wmax = u.weight.float().abs().amax(1).clamp_min(1e-12)
        u._s = ((amax ** ALPHA) / (wmax ** (1 - ALPHA))).clamp(1e-3, 1e3).to(u.weight.dtype)

    kl_bf16 = mean_kl("hi"); kl_pc = mean_kl("fp8pc"); kl_sq = mean_kl("smooth")
    print(f"(S) all-W8A8 KL vs fp32 anchor: bf16={kl_bf16:.3e}  per-channel={kl_pc:.3e}  SmoothQuant={kl_sq:.3e}")
    crushes = kl_sq < kl_pc
    print(f"    SmoothQuant {'CRUSHES the floor (%.2fx)' % (kl_pc/kl_sq) if crushes else 'does NOT help'} vs per-channel\n")

    print("[VALIDITY]")
    print(f"  (L) fp32 licensed: {licensed}   (O) outliers stronger than 124M: {stronger}   (S) SmoothQuant helps: {crushes}")
    if licensed and (stronger or crushes):
        print("  -> GPT-2-XL is a VALID testbed: fp32 anchor trustworthy, real(er) pathology. Proceed to certification re-test.")
    elif licensed:
        print("  -> fp32 licensed but the outlier pathology is STILL mild at 1.5B (same-family GPT-2). Honest finding:")
        print("     a meaningful SmoothQuant test needs OPT-class outliers; GPT-2 lacks them even at 1.5B.")
    else:
        print("  -> fp32 anchor NOT licensed at this scale: need a higher-precision anchor before any claim.")
    print("\n[V-hw] RTX 5070 sm_120, fp32 anchor (op-licensed vs fp64). GPT-2-XL 1.5B. Validity gates only.")
    return licensed and (stronger or crushes)


if __name__ == "__main__":
    run()
