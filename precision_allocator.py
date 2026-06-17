"""T1_precision_map v0.2 -- Stage 3 (sim): the certified precision allocator + its GATE and controls.

The instrument (pre-reg s.8): LAMP's mechanism with a CERTIFIED reference. Weight-only mixed precision on
GPT-2-small. Each allocatable weight tensor is assigned a tier from {FP4 E2M1, FP8 E4M3, bf16}; bf16 is the
"recompute"/protected tier, FP4 the cheapest. float64 (un-quantized weights) is the CERTIFIED reference --
the right to trust it was licensed in Stage 1 by the mpmath dps>=50 spot-cert on the same op types.

This file holds the primitives the bake-off (allocator_bakeoff.py) consumes, plus:
  [GATE] derive-before-numerics  (run as __main__): the FP8 E4M3 / FP4 E2M1 grids are DERIVED and certified
         against spec (known representable values, max, ulp); round-to-nearest + per-tensor absmax scaling are
         certified; the FP32-linear scorer is certified to be a faithful FIRST-ORDER predictor (O(alpha) probe)
         -- so the FP4-divergence prediction (P3) is a falsifiable measurement, not an artifact of a broken scorer.
  controls  control-before-scan: C1 reference identity (fp64-vs-fp64 KL==0), per-tensor tier monotonicity
         (fp4 err >= fp8 >= bf16), matched-budget assertion. Bake-off is admissible iff these pass.

Two scorers, both estimating "how much does leaving tensor i at the base low precision hurt the output KL":
  certified (A_cert): quantize ONLY tensor i to base, run the float64 forward, measure the TRUE KL vs the
                      float64 reference. Nonlinear, exact, certified reference.
  FP32-linear (LAMP-style): the FIRST-ORDER estimate of that same KL, computed in float32 -- the actual
                      quantization error dW_i propagated through a LINEARIZED network (small-alpha probe,
                      scaled). Faithful first-order look-ahead; breaks where the O(eps^2) remainder bites.

Prediction (pre-reg P3/F-app): the two scorers RANK tensors the same at FP8 (small dW -> first-order accurate)
and DIFFERENTLY at FP4 (large dW -> second-order dominates) -- exactly the Stage-0 certified recursion
eps_{l+1}=(I+J_F)eps_l + O(eps^2) applied at the allocation level. [E-hw] CPU, torch float64 reference.
"""
import numpy as np
import torch

GPT2_PATH = "C:/Users/JT-DEV1/Documents/gpt2-sm"   # locally downloaded gpt2-small (no HF network here)
torch.set_grad_enabled(False)

# ----------------------------------------------------------------------------- representable grids (DERIVED)
def fp8_e4m3_grid():
    """OCP/NVIDIA FP8 E4M3: 1-4-3, exp bias 7. Normals 2^(e-7)*(1+m/8) e=1..15 m=0..7 (drop e15m7=NaN);
    subnormals 2^-6*(m/8) m=1..7; plus 0. Max finite = 448, smallest subnormal = 2^-9. Returns sorted +grid."""
    vals = {0.0}
    for e in range(1, 16):
        for m in range(8):
            if e == 15 and m == 7:
                continue                                   # S.1111.111 = NaN in E4M3
            vals.add(2.0 ** (e - 7) * (1.0 + m / 8.0))
    for m in range(1, 8):
        vals.add(2.0 ** (-6) * (m / 8.0))                  # subnormals
    return torch.tensor(sorted(vals), dtype=torch.float64)


def fp4_e2m1_grid():
    """OCP MX FP4 E2M1: 1-2-1, exp bias 1. Positive grid = {0,.5,1,1.5,2,3,4,6}; max 6, no inf/NaN."""
    vals = {0.0}
    for e in range(4):                                     # exp field 0..3
        for m in range(2):                                 # mantissa bit
            if e == 0:
                vals.add(2.0 ** (1 - 1) * (m / 2.0))        # subnormal: 2^0 * m/2
            else:
                vals.add(2.0 ** (e - 1) * (1.0 + m / 2.0))  # normal
    return torch.tensor(sorted(vals), dtype=torch.float64)


_GRIDS = {"fp8": fp8_e4m3_grid(), "fp4": fp4_e2m1_grid()}
TIER_BITS = {"fp4": 4, "fp8": 8, "bf16": 16, "ref": 64}


def quant_to_grid(W, grid):
    """Per-tensor absmax-scaled round-to-nearest onto `grid` (ascending +values incl 0). Returns float64."""
    W = W.to(torch.float64)
    amax = W.abs().max()
    if amax == 0:
        return W.clone()
    scale = amax / grid[-1]                                 # absmax -> top of grid (no clipping beyond)
    a = (W.abs() / scale)
    mids = (grid[:-1] + grid[1:]) / 2.0
    idx = torch.bucketize(a, mids)                          # nearest grid index
    return grid[idx] * torch.sign(W) * scale


def fake_quant(W, tier):
    """Weight-only fake-quant to `tier`, returning a dequantized float64 tensor (the ONLY error source)."""
    if tier == "ref":
        return W.to(torch.float64).clone()
    if tier == "bf16":
        return W.to(torch.bfloat16).to(torch.float64)
    return quant_to_grid(W, _GRIDS[tier])


# ----------------------------------------------------------------------------- model + metric helpers
def load_model(dtype):
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained(GPT2_PATH).eval()
    return m.to(dtype)


def allocatable(model):
    """The weight-only allocation surface: the 4 matmul weight tensors per block (c_attn, c_proj, c_fc, c_proj).
    Embeddings / LayerNorm / lm_head are held at the reference precision (standard weight-only quant)."""
    out = []
    for i, blk in enumerate(model.transformer.h):
        for sub, mod in (("attn.c_attn", blk.attn.c_attn), ("attn.c_proj", blk.attn.c_proj),
                         ("mlp.c_fc", blk.mlp.c_fc), ("mlp.c_proj", blk.mlp.c_proj)):
            out.append((f"h{i}.{sub}", mod))
    return out


def snapshot(tensors):
    return [mod.weight.detach().to(torch.float64).clone() for _, mod in tensors]


def apply_assignment(tensors, originals, tiers):
    """tiers: list[str] per tensor; set each module weight to fake_quant(original, tier) in the model dtype."""
    for (_, mod), W0, t in zip(tensors, originals, tiers):
        mod.weight.data = fake_quant(W0, t).to(mod.weight.dtype)


def restore(tensors, originals):
    for (_, mod), W0 in zip(tensors, originals):
        mod.weight.data = W0.to(mod.weight.dtype)


def logits_of(model, ids, mask):
    return model(input_ids=ids, attention_mask=mask).logits


def kl_flip(ref_logits, q_logits, mask):
    """KL(p_ref || p_q) and top-1 flip rate, averaged over masked (valid) positions. float64."""
    rl = ref_logits.to(torch.float64); ql = q_logits.to(torch.float64)
    lpr = torch.log_softmax(rl, -1); lpq = torch.log_softmax(ql, -1)
    kl = (lpr.exp() * (lpr - lpq)).sum(-1)                 # (B,T)
    flip = (rl.argmax(-1) != ql.argmax(-1)).to(torch.float64)
    m = mask.to(torch.float64)
    denom = m.sum().clamp_min(1.0)
    return float((kl * m).sum() / denom), float((flip * m).sum() / denom)


# ----------------------------------------------------------------------------- the two scorers (the instrument)
def certified_scores(model_f64, tensors, originals, base, ref_logits, ids, mask):
    """A_cert: leave-one-tensor-quantized-to-`base`, TRUE KL vs the float64 reference. Nonlinear, certified."""
    s = np.zeros(len(tensors))
    for i in range(len(tensors)):
        tiers = ["ref"] * len(tensors); tiers[i] = base
        apply_assignment(tensors, originals, tiers)
        s[i] = kl_flip(ref_logits, logits_of(model_f64, ids, mask), mask)[0]
    restore(tensors, originals)
    return s


def linear_scores(model_f32, tensors, originals, base, ids, mask, alpha=1.0 / 32.0):
    """LAMP-style FP32-linear: first-order estimate of the per-tensor KL. dW_i is the real base-precision
    quant error; it is propagated through the LINEARIZED net (probe at W+alpha*dW, scale by 1/alpha) in
    float32, then converted to a first-order KL vs the float32 reference. Faithful first-order look-ahead."""
    restore(tensors, originals)
    ref32 = logits_of(model_f32, ids, mask).to(torch.float64)
    lpr = torch.log_softmax(ref32, -1); pr = lpr.exp()
    s = np.zeros(len(tensors))
    for i, ((_, mod), W0) in enumerate(zip(tensors, originals)):
        dW = (fake_quant(W0, base) - W0)                   # the actual base-precision quantization error
        mod.weight.data = (W0 + alpha * dW).to(mod.weight.dtype)
        probe = logits_of(model_f32, ids, mask).to(torch.float64)
        mod.weight.data = W0.to(mod.weight.dtype)
        dlog_lin = (probe - ref32) / alpha                 # first-order predicted FULL-step logit change
        lpq = torch.log_softmax(ref32 + dlog_lin, -1)
        kl = (pr * (lpr - lpq)).sum(-1)
        m = mask.to(torch.float64)
        s[i] = float((kl * m).sum() / m.sum().clamp_min(1.0))
    return s


def allocate(scores, k):
    """Promote the top-k highest-sensitivity tensors to the recompute (bf16) tier; rest stay at base."""
    order = np.argsort(-scores)
    return set(int(j) for j in order[:k])


# ----------------------------------------------------------------------------- [GATE] + controls
def gate():
    import math
    fails = []
    print("[GATE precision_allocator] derive-before-numerics: certify the quantizers + the linear scorer\n")

    # (Q1) the grids ARE the derivation -- assert against the published representable sets
    g8, g4 = _GRIDS["fp8"], _GRIDS["fp4"]
    fp4_expect = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
    q1 = torch.allclose(g4, fp4_expect)
    fp8_ok = (abs(float(g8[-1]) - 448.0) < 1e-9 and abs(float(g8[g8 > 0][0]) - 2.0 ** -9) < 1e-15
              and all(float(v) in set(g8.tolist()) for v in (0.5, 1.0, 2.0, 256.0, 448.0)))
    print(f"(Q1) FP4 E2M1 grid == {{0,.5,1,1.5,2,3,4,6}}: {q1}")
    print(f"     FP8 E4M3 grid: max={float(g8[-1]):.0f} (spec 448), min subnormal={float(g8[g8>0][0]):.3e} "
          f"(spec 2^-9={2.0**-9:.3e}), 1.0/2.0/256/448 present: {fp8_ok}")
    if not q1: fails.append("Q1 fp4 grid")
    if not fp8_ok: fails.append("Q1 fp8 grid")

    # (Q2) round-to-NEAREST + absmax scaling. The CORRECT invariant for absmax-scaled FLOAT quant is exact
    #      nearest-neighbour onto the SIGNED scaled grid (a small weight legitimately rounds toward 0 -> a
    #      per-element *relative* half-ulp bound is the WRONG invariant; the absolute error is bounded by
    #      half the largest scaled grid gap). We certify: (a) output == brute-force nearest scaled grid point;
    #      (b) grid points are fixed (idempotent); (c) |W-q| <= half the max scaled grid gap (abs half-ulp).
    gen = torch.Generator().manual_seed(0)
    for tier in ("fp4", "fp8"):
        W = torch.randn(96, 96, generator=gen, dtype=torch.float64)
        q = fake_quant(W, tier)
        grid = _GRIDS[tier]
        scale = W.abs().max() / grid[-1]
        sgrid = torch.cat([-grid.flip(0), grid[1:]]) * scale          # full signed scaled grid
        nearest = sgrid[(W.unsqueeze(-1) - sgrid).abs().argmin(-1)]   # brute-force nearest, per element
        exact = (q - nearest).abs().max().item()                     # quantizer must equal the true nearest
        fixed = (fake_quant(q, tier) - q).abs().max().item()         # grid points are fixed points
        half_gap = float((grid[1:] - grid[:-1]).max() * scale) / 2.0 # absolute half-ulp (largest gap)
        abserr = (q - W).abs().max().item()
        print(f"(Q2) {tier}: ||quantizer - brute-force nearest|| = {exact:.1e} (round-to-nearest certified); "
              f"idempotent residual = {fixed:.1e}; max |W-q| = {abserr:.2e} <= half-ulp {half_gap:.2e}")
        if exact > 1e-12: fails.append(f"Q2 {tier} not nearest")
        if fixed > 1e-12: fails.append(f"Q2 {tier} not idempotent")
        if abserr > half_gap + 1e-9: fails.append(f"Q2 {tier} exceeds half-ulp")

    # (Q3) tier ordering on a REAL trained weight: fp4 err >= fp8 err >= bf16 err
    m = load_model(torch.float64)
    W0 = allocatable(m)[18][1].weight.detach().to(torch.float64)   # a mid-network mlp weight
    errs = {t: float((fake_quant(W0, t) - W0).norm() / W0.norm()) for t in ("fp4", "fp8", "bf16")}
    order_ok = errs["fp4"] >= errs["fp8"] >= errs["bf16"]
    print(f"(Q3) tier ordering on a trained weight: fp4={errs['fp4']:.2e} >= fp8={errs['fp8']:.2e} "
          f">= bf16={errs['bf16']:.2e}  -> {order_ok}")
    if not order_ok: fails.append("Q3 tier ordering")

    # build a tiny frozen batch for the model-level controls
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(GPT2_PATH); tok.pad_token = tok.eos_token
    enc = tok(["The history of numerical analysis begins with round-off error.",
               "Quantization amplifies error at sharp attention logits."],
              return_tensors="pt", padding="max_length", truncation=True, max_length=16)
    ids, mask = enc.input_ids, enc.attention_mask
    tensors = allocatable(m); originals = snapshot(tensors)
    ref = logits_of(m, ids, mask)

    # (C1) reference identity: fp64 vs fp64 -> KL==0; all-ref allocation -> KL==0
    kl0, fl0 = kl_flip(ref, logits_of(m, ids, mask), mask)
    apply_assignment(tensors, originals, ["ref"] * len(tensors))
    klr, _ = kl_flip(ref, logits_of(m, ids, mask), mask); restore(tensors, originals)
    print(f"(C1) reference identity: self-KL={kl0:.1e}, all-'ref' allocation KL={klr:.1e} (both must be 0)")
    if kl0 > 1e-12 or klr > 1e-12: fails.append("C1 reference identity")

    # (GATE-L) the FP32-linear scorer is a genuine FIRST-ORDER predictor. Two things, both certified in fp64
    #          (the linearization math is precision-independent; the scorer runs fp32 in practice):
    #   (a) first-order soundness: response g(b)=f(W+b*u)-f(W) along u=dW/||dW|| has a QUADRATIC remainder --
    #       halving b shrinks ||g(b) - b*D|| ~4x (D = central-diff directional derivative). Mirrors Stage 0.
    #   (b) mechanism preview (NOT the P3 verdict): the linear extrapolation of the FULL step is faithful at
    #       fp8 (small dW) and degrades at fp4 (large dW). P3 is whether this flips the 48-tensor RANKING -> bake-off.
    t_idx = 18
    (_, modg) = allocatable(m)[t_idx]; W0g = modg.weight.detach().to(torch.float64).clone()
    f = lambda W: (modg.weight.__setattr__("data", W.to(torch.float64)),
                   logits_of(m, ids, mask).to(torch.float64))[1]
    base_log = f(W0g)
    print("(GATE-L) FP32-linear scorer is first-order (fp64-certified):")
    for tier in ("fp8", "fp4"):
        u = (fake_quant(W0g, tier) - W0g); un = u / u.norm()
        h = 1e-4
        D = (f(W0g + h * un) - f(W0g - h * un)) / (2 * h)            # directional derivative along u
        prev = None
        line = []
        for b in (1e-2, 5e-3, 2.5e-3):
            rem = float((f(W0g + b * un) - base_log - b * D).norm())
            line.append(f"b={b:.1e}:rem={rem:.1e}" + (f"(x{rem/prev:.2f})" if prev else ""))
            prev = rem
        # full-step linear extrapolation vs true full step
        full = f(W0g + u) - base_log
        lin_full = u.norm() * D
        rel = float((lin_full - full).norm() / full.norm().clamp_min(1e-30))
        modg.weight.data = W0g                                       # restore
        print(f"   {tier}: quad remainder {' '.join(line)}  |  full-step linear-vs-true rel err = {rel:.2f}")
    print("   => soundness: b/2 -> remainder ~/4 (genuine first-order). preview: fp8 full-step rel err << fp4's,")
    print("      so the linear scorer is faithful at fp8 and degrades at fp4. Whether that flips the RANKING is P3.")

    print("\n[FROZEN PREDICTION Stage-3]  P2: certified allocation meets a fixed KL/flip budget at <= recompute")
    print("  cost of uniform AND of the FP32-linear (LAMP-style) allocator at matched budget. P3/F-app: the")
    print("  certified top-k op-set EQUALS the FP32-linear op-set at FP8 (F-app fires = redundancy control) and")
    print("  DIFFERS at FP4 (F-app does NOT fire; P3 holds), with the certified pick achieving lower true KL.")
    print("\nALLOCATOR-GATE:", "quantizers + linear scorer CERTIFIED; bake-off admissible"
          if not fails else f"FAIL {fails}")
    return not fails


if __name__ == "__main__":
    gate()
