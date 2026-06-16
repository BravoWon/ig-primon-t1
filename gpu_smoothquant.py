"""HARDWARE: real SmoothQuant migration on the 5070 -- crush the activation floor, then re-ask certification.

SmoothQuant (Xiao et al. 2022): outlier ACTIVATION channels are what make W8A8 hard. Migrate the difficulty
into the weights via a per-input-channel smoothing factor s (calibrated):
    s_j = max|X_j|^alpha / max|W_j|^(1-alpha)
    X_hat = X . diag(s)^-1 ,  W_hat = diag(s) . W   ->  X_hat @ W_hat = X @ W  (exact; outliers balanced)
then FP8-quantize X_hat, W_hat. This is the real mitigation (not just finer scales).

Controls (control-before-scan):
  C-exact : (X/s) @ (s*W) == X @ W in float64 (migration is a no-op without quant).
  C-floor : all-W8A8 KL must DROP: per-tensor > per-channel > SmoothQuant. If SmoothQuant doesn't lower the
            floor, the premise fails and the certification re-test is meaningless.
Then: certification P2 (held-out) on the SmoothQuant baseline -- does the +6% certified edge survive, vanish,
or become the dominant differentiator once the activation floor is crushed?
[V-hw] RTX 5070 sm_120, FP8 e4m3 fake-quant (rowwise kernel unsupported here). float64 ref. score A, eval B.
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

import precision_allocator as PA
import gpu_fapp_hardware as H

torch.set_grad_enabled(False)
DEV = "cuda:0"; FP8 = torch.float8_e4m3fn; ALPHA = 0.5
TEXTS_A = H.TEXTS
TEXTS_B = [
    "A river carved the canyon over millions of years, exposing layers of ancient sediment.",
    "The committee postponed the vote until the auditors finished reviewing the quarterly accounts.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose inside the chloroplast.",
    "He tuned the old radio carefully, searching the static for a station broadcasting the match.",
    "Differential equations describe how quantities change continuously with respect to one another.",
    "The bakery on the corner sells warm sourdough every morning before the commuters arrive.",
]


def fp8pc_mm(xs, Ws, b, odt, wdt):
    sx = xs.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
    sw = Ws.float().abs().amax(0, keepdim=True).clamp_min(1e-12) / 448.0
    xq = ((xs.float() / sx).to(FP8).float() * sx).to(odt)
    wq = ((Ws.float() / sw).to(FP8).float() * sw).to(wdt)
    return torch.addmm(b, xq, wq)


def patched(self, x):
    size_out = x.size()[:-1] + (self.nf,)
    x2d = x.reshape(-1, x.size(-1)); W = self.weight; b = self.bias
    mode = getattr(self, "_prec", "bf16")
    if mode == "bf16":
        out = torch.addmm(b, x2d, W)
    elif mode == "fp8pc":                                            # per-channel, NO smoothing
        out = fp8pc_mm(x2d, W, b, x2d.dtype, W.dtype)
    else:                                                            # smooth: SmoothQuant migration + per-channel FP8
        s = self._s                                                  # (K,) per input channel
        out = fp8pc_mm(x2d / s, W * s[:, None], b, x2d.dtype, W.dtype)
    return out.view(size_out)


def calibrate(mg, U, tok):
    """Per-input-channel activation absmax (bf16 forward), then s_j per unit."""
    store = {}
    def mk(u):
        def hook(mod, inp):
            a = inp[0].reshape(-1, inp[0].size(-1)).float().abs().amax(0)
            store[id(u)] = a if id(u) not in store else torch.maximum(store[id(u)], a)
        return u.register_forward_pre_hook(hook)
    hs = [mk(u) for u in U]
    for u in U:
        u._prec = "bf16"
    for t in TEXTS_A:
        mg(tok(t, return_tensors="pt").input_ids[:, :32].to(DEV))
    for h in hs:
        h.remove()
    for u in U:
        amax = store[id(u)].clamp_min(1e-12)
        wmax = u.weight.float().abs().amax(1).clamp_min(1e-12)        # per input channel
        s = (amax ** ALPHA) / (wmax ** (1 - ALPHA))
        u._s = s.clamp(1e-3, 1e3).to(u.weight.dtype)


def run():
    print(f"[SmoothQuant hardware]  alpha={ALPHA}, RTX 5070 sm_120, crush the activation floor then re-test P3\n")

    # C-exact: migration is a no-op without quant (float64)
    g = torch.Generator().manual_seed(0)
    X = torch.randn(20, 16, generator=g, dtype=torch.float64); Wt = torch.randn(16, 24, generator=g, dtype=torch.float64)
    s = (X.abs().amax(0) ** 0.5) / (Wt.abs().amax(1) ** 0.5)
    err = ((X / s) @ (s[:, None] * Wt) - X @ Wt).abs().max().item()
    print(f"(C-exact) migration identity (X/s)(sW)=XW : max abs err = {err:.1e}  (must be ~0)")

    tok = GPT2TokenizerFast.from_pretrained(PA.GPT2_PATH)
    m64 = GPT2LMHeadModel.from_pretrained(PA.GPT2_PATH).double().eval()
    Conv1D.forward = patched
    mg = GPT2LMHeadModel.from_pretrained(PA.GPT2_PATH).to(torch.bfloat16).to(DEV).eval()
    U = H.units(mg); nU = len(U)
    calibrate(mg, U, tok)

    def ids_of(t):
        return tok(t, return_tensors="pt").input_ids[:, :32]

    def mean_kl_B(mode_all):
        for u in U:
            u._prec = mode_all
        ks = []
        for t in TEXTS_B:
            ids = ids_of(t); ref = m64(ids).logits[0].float()
            ks.append(H.kl(ref, mg(ids.to(DEV)).logits[0].float().cpu()))
        return float(np.mean(ks))

    # C-floor: does SmoothQuant lower the all-W8A8 floor?
    kl_bf16 = mean_kl_B("bf16"); kl_pc = mean_kl_B("fp8pc"); kl_sq = mean_kl_B("smooth")
    print(f"\n(C-floor) all-W8A8 KL vs float64 (held-out B):")
    print(f"   bf16 floor = {kl_bf16:.3e}   per-channel (no smooth) = {kl_pc:.3e}   SmoothQuant = {kl_sq:.3e}")
    print(f"   -> SmoothQuant cuts W8A8 error {kl_pc/kl_sq:.2f}x vs per-channel; "
          f"gap above bf16 floor: {kl_pc-kl_bf16:.2e} -> {kl_sq-kl_bf16:.2e}")
    floor_crushed = kl_sq < kl_pc
    print(f"   floor {'CRUSHED (SmoothQuant helps; re-test is meaningful)' if floor_crushed else 'NOT reduced -- premise fails'}\n")

    # Part 2: certification P2 on the SmoothQuant baseline (held-out)
    impC = np.zeros(nU); impP = np.zeros(nU); floorA = 0.0
    for t in TEXTS_A:
        ids = ids_of(t)
        for u in U:
            u._prec = "bf16"
        refC = m64(ids).logits[0].float(); refP = mg(ids.to(DEV)).logits[0].float().cpu()
        floorA += H.kl(refC, refP)
        for i in range(nU):
            U[i]._prec = "smooth"; cfg = mg(ids.to(DEV)).logits[0].float().cpu(); U[i]._prec = "bf16"
            impC[i] += H.kl(refC, cfg); impP[i] += H.kl(refP, cfg)
    impC = impC / len(TEXTS_A) - floorA / len(TEXTS_A); impP /= len(TEXTS_A)

    refsB = [(ids_of(t), m64(ids_of(t)).logits[0].float()) for t in TEXTS_B]
    def true_kl_B(keep):
        for j, u in enumerate(U):
            u._prec = "bf16" if j in keep else "smooth"
        return float(np.mean([H.kl(rc, mg(ids.to(DEV)).logits[0].float().cpu()) for ids, rc in refsB]))

    print("Part 2 -- certification P2 on the SmoothQuant baseline (held-out B): certified vs practical-bf16")
    print(f"   {'k(bf16)':>7} {'certified KL':>13} {'practical KL':>13} {'cert<=prac?':>11}")
    wins = 0; gains = []
    for k in (4, 8, 12, 16, 24):
        cset = set(np.argsort(-impC)[:k].tolist()); pset = set(np.argsort(-impP)[:k].tolist())
        klC = true_kl_B(cset); klP = true_kl_B(pset)
        wins += klC <= klP + 1e-12; gains.append((klP - klC) / klP * 100)
        print(f"   {k:>7} {klC:>13.4e} {klP:>13.4e} {str(klC <= klP + 1e-12):>11}")
    gmean = float(np.mean(gains))

    print(f"\n[VERDICT] SmoothQuant baseline: certified <= practical {wins}/5; mean certified gain = {gmean:+.1f}% KL.")
    print(f"  (per-tensor gave +6.1%/5-of-5; naive per-channel +4.0%/3-of-5.)")
    if not floor_crushed:
        print("  -> premise failed (floor not reduced); the certification re-test is not interpretable.")
    elif wins >= 4 and gmean > 3:
        print("  -> with the activation floor crushed, certification BECOMES the dominant differentiator: when the")
        print("     easy error is gone, the certified reference's subtle picks are what is left. (your hypothesis)")
    elif gmean <= 1.5 or wins <= 2:
        print("  -> the certified edge VANISHES into the lowered baseline: SmoothQuant removes the room certification")
        print("     exploited. Certification mattered only while quantization was crude. (the deflation prior)")
    else:
        print("  -> edge persists at reduced magnitude; report as-is, claim nothing larger than measured.")
    print("\n[V-hw] RTX 5070 sm_120, SmoothQuant FP8 fake-quant. float64 ref. score A, eval disjoint B. No timing claim.")
    return gmean, wins, kl_sq


if __name__ == "__main__":
    run()
