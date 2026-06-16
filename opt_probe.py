"""SCALE-UP validity probe for OPT-2.7B -- the open GPT-3 stand-in with REAL activation outliers.

GPT-2-XL failed the validity gate (outliers still mild at 1.5B). OPT is the family the outlier pathology
(LLM.int8 / SmoothQuant) was characterized on. This probe ports the three gates to OPT (nn.Linear, y=x.W^T):
  (L) license the fp32 anchor (per-op fp32-vs-fp64). fp64 of 2.7B is intractable; fp32 must earn 'certified'.
  (O) OUTLIER pathology: per-input-channel activation max/median. Expect STRONGER than GPT-2 (median 9-10).
  (S) does SmoothQuant now CRUSH the floor (unlike GPT-2, where it made W8A8 worse)?

Memory: fp32 (10.8GB) + bf16 (5.4GB) > 12.8GB card, so the fp32 ANCHOR runs on CPU (cache its few reference
logits), the bf16 deployment + FP8 configs on the GPU. Models loaded SEQUENTIALLY (peak ~one model in RAM).
Only if (L) passes and (O)+(S) show a real, mitigable pathology do we run the full certification re-test.
[V-hw] RTX 5070 sm_120 (deployment) + CPU fp32 anchor. OPT-2.7B. Validity gates only.
"""
import gc
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, OPTForCausalLM

torch.set_grad_enabled(False)
DEV = "cuda:0"; FP8 = torch.float8_e4m3fn; ALPHA = 0.5
OPT_PATH = "C:/Users/JT-DEV1/Documents/opt-2.7b"
TEXTS = [
    "The history of numerical analysis begins with the study of round-off error in finite-precision arithmetic.",
    "Large language models compose many layers, each performing matrix multiplications and nonlinear maps.",
    "Quantization reduces memory but can amplify errors at sharp attention logits and saturated activations.",
    "In the morning the fishermen returned with their catch and sold it at the market by the harbor.",
]


def units(model):
    out = []
    for layer in model.model.decoder.layers:
        a = layer.self_attn
        out += [a.q_proj, a.k_proj, a.v_proj, a.out_proj, layer.fc1, layer.fc2]
    return out


def kl(ref, q):
    rl = ref.double(); ql = q.double()
    lpr = torch.log_softmax(rl, -1); lpq = torch.log_softmax(ql, -1)
    return (lpr.exp() * (lpr - lpq)).sum(-1).mean().item()


def patched_linear(self, x):
    """nn.Linear forward, y = x @ W^T + b, with optional FP8 fake-quant gated on self._prec.
    W is (out, in); input channel = column of W; per-output-channel scale = row of W."""
    mode = getattr(self, "_prec", None)
    W = self.weight; b = self.bias
    if mode is None or mode == "hi":
        return F.linear(x, W, b)
    x2d = x.reshape(-1, x.size(-1))
    if mode == "fp8t":
        sx = x2d.float().abs().amax().clamp_min(1e-12) / 448.0
        sw = W.float().abs().amax().clamp_min(1e-12) / 448.0
        xq = ((x2d.float()/sx).to(FP8).float()*sx).to(x2d.dtype)
        wq = ((W.float()/sw).to(FP8).float()*sw).to(W.dtype)
    elif mode == "fp8pc":
        sx = x2d.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0   # per token
        sw = W.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0     # per output channel
        xq = ((x2d.float()/sx).to(FP8).float()*sx).to(x2d.dtype)
        wq = ((W.float()/sw).to(FP8).float()*sw).to(W.dtype)
    else:                                                                       # smooth (SmoothQuant)
        s = self._s                                                             # (in,) per input channel
        xs = x2d / s; Ws = W * s[None, :]
        sx = xs.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        sw = Ws.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        xq = ((xs.float()/sx).to(FP8).float()*sx).to(x2d.dtype)
        wq = ((Ws.float()/sw).to(FP8).float()*sw).to(W.dtype)
    out = xq @ wq.t()
    if b is not None:
        out = out + b
    return out.view(*x.shape[:-1], W.size(0))


def run():
    import os
    if not os.path.exists(OPT_PATH):
        print(f"[probe] model not found at {OPT_PATH} -- download facebook/opt-2.7b there first.")
        return False
    print("[OPT-2.7B probe] validity gates (CPU fp32 anchor + GPU bf16 deployment)\n")
    tok = AutoTokenizer.from_pretrained(OPT_PATH)

    def ids_of(t):
        return tok(t, return_tensors="pt").input_ids[:, :32]

    # ---- Phase A: fp32 anchor on CPU -- reference logits + license ----
    torch.nn.Linear.forward = patched_linear                                    # safe: only _prec-tagged units quantize
    m32 = OPTForCausalLM.from_pretrained(OPT_PATH).to(torch.float32).eval()      # CPU
    U32 = units(m32)
    store = {}; hs = []
    def mk(u):
        def hook(mod, inp):
            store.setdefault(id(u), inp[0].reshape(-1, inp[0].size(-1)).detach())
        return u.register_forward_pre_hook(hook)
    for u in U32:
        hs.append(mk(u))
    ref_cache = [m32(ids_of(t)).logits[0].float() for t in TEXTS]               # cached fp32 references
    for h in hs:
        h.remove()
    rels = []
    for u in (U32[0], U32[len(U32)//2], U32[-1]):
        X = store[id(u)].float(); W = u.weight.float()
        o32 = X @ W.t(); o64 = X.double() @ W.double().t()
        rels.append(float((o32.double() - o64).norm() / o64.norm().clamp_min(1e-30)))
    print(f"(L) fp32 anchor license: per-op fp32-vs-fp64 rel err = {max(rels):.1e} (<= ~1e-5 -> fp32 licensed)")
    licensed = max(rels) < 1e-5
    n_layer = m32.config.num_hidden_layers; d = m32.config.hidden_size
    del m32, store, U32; gc.collect()

    # ---- Phase B: bf16 deployment on GPU ----
    mg = OPTForCausalLM.from_pretrained(OPT_PATH).to(torch.bfloat16).to(DEV).eval()
    U = units(mg); nU = len(U)
    print(f"  loaded: {n_layer} layers, d={d}, {nU} matmul units\n")

    # (O) outlier pathology + calibrate s
    store2 = {}; hs = []
    def mk2(u):
        def hook(mod, inp):
            a = inp[0].reshape(-1, inp[0].size(-1)).float().abs().amax(0)
            store2[id(u)] = a if id(u) not in store2 else torch.maximum(store2[id(u)], a)
        return u.register_forward_pre_hook(hook)
    for u in U:
        u._prec = "hi"; hs.append(mk2(u))
    for t in TEXTS:
        mg(ids_of(t).to(DEV))
    for h in hs:
        h.remove()
    ratios = np.array([(store2[id(u)].max() / store2[id(u)].median().clamp_min(1e-12)).item() for u in U])
    print(f"(O) outlier ratio (max/median per channel): median={np.median(ratios):.1f}  mean={ratios.mean():.1f}  "
          f"max={ratios.max():.1f}   (GPT-2 was median ~9-10)")
    stronger = np.median(ratios) > 15.0
    for u in U:
        amax = store2[id(u)].clamp_min(1e-12)
        wmax = u.weight.float().abs().amax(0).clamp_min(1e-12)                   # per input channel (column)
        u._s = ((amax ** ALPHA) / (wmax ** (1 - ALPHA))).clamp(1e-3, 1e3).to(u.weight.dtype)

    # (S) floor: bf16 / per-channel / SmoothQuant, vs cached fp32 reference
    def mean_kl(mode):
        for u in U:
            u._prec = mode
        return float(np.mean([kl(ref_cache[i], mg(ids_of(t).to(DEV)).logits[0].float().cpu())
                              for i, t in enumerate(TEXTS)]))
    kl_bf16 = mean_kl("hi"); kl_pc = mean_kl("fp8pc"); kl_sq = mean_kl("smooth")
    print(f"(S) all-W8A8 KL vs fp32 anchor: bf16={kl_bf16:.3e}  per-channel={kl_pc:.3e}  SmoothQuant={kl_sq:.3e}")
    crushes = kl_sq < kl_pc
    print(f"    SmoothQuant {'CRUSHES the floor (%.2fx)' % (kl_pc/kl_sq) if crushes else 'does NOT help'} vs per-channel\n")

    print("[VALIDITY]")
    print(f"  (L) fp32 licensed: {licensed}   (O) outliers (median>15): {stronger}   (S) SmoothQuant helps: {crushes}")
    if licensed and (stronger or crushes):
        print("  -> OPT-2.7B is a VALID testbed: fp32 anchor trustworthy, REAL pathology. Proceed to certification re-test.")
    elif licensed:
        print("  -> fp32 licensed but pathology weaker than expected; inspect ratios before claiming. Report as-is.")
    else:
        print("  -> fp32 anchor NOT licensed at this scale: need a higher-precision anchor first.")
    print("\n[V-hw] RTX 5070 sm_120 (bf16 GPU) + CPU fp32 anchor. OPT-2.7B. Validity gates only.")
    return licensed and (stronger or crushes)


if __name__ == "__main__":
    run()
