"""
stage2_gpt2_p1.py  —  IG-PRIMON-T1 Stage 2 (the scientific payload)

REAL test on TRAINED GPT-2-small (124M, 12 blocks, d=768). UNTUNED.

Question (P1 vs F1, from T1_precision_map_v0_2.md §5/§6):
  Is the TYPICAL-CASE certified error E_cert(L) sub-exponential / near-linear
  through depth (P1, residual attenuation dominates on trained weights) —
  despite the worst-case being exponential (Budzinskiy, reproduced in Stage 1)?

  P1 holds  -> log-slope of typical-case error vs L is small/near-zero.
  F1 fires  -> log-slope is large & positive (exponential on trained weights too).

Reference = float64 (certified adequate on this op set at dps=50 in Stage 1).
Low precision = bf16 (real torch ops; weights+activations quantized, as in
                low-precision inference). This is the round-off injection.
Headline (per §2 confound #2 / arXiv 2505.24187): the signal is WHICH tokens
fire, not the curve magnitude -> we also report median vs mean and per-token
concentration at the output.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(20260616)
dev = "cpu"

print("="*72)
print("IG-PRIMON-T1  Stage 2  —  trained GPT-2-small, certified depth-error")
print("="*72)

tok = AutoTokenizer.from_pretrained("gpt2")
text = ("The numerical analysis of deep transformer networks requires careful "
        "attention to how round-off errors injected at each layer either "
        "accumulate or attenuate as signals propagate through the residual "
        "stream. In practice, trained weights and layer normalization tend to "
        "keep activations well conditioned, which is precisely the regime this "
        "experiment is designed to characterize on real model weights.")
ids = tok(text, return_tensors="pt").input_ids.to(dev)
seq = ids.shape[1]
print(f"\ninput: 1 x {seq} tokens of natural English (typical-case arm)")

# ---- reference forward in float64 ----
m64 = AutoModelForCausalLM.from_pretrained("gpt2").to(dev).double().eval()
with torch.no_grad():
    hs64 = m64(ids, output_hidden_states=True).hidden_states  # tuple (L+1) of (1,seq,d)
L = len(hs64) - 1
print(f"reference: float64 forward, {L} residual-stream checkpoints captured")

# ---- low-precision forward (bf16) ----
try:
    m_lo = AutoModelForCausalLM.from_pretrained("gpt2").to(dev).to(torch.bfloat16).eval()
    with torch.no_grad():
        hs_lo = m_lo(ids, output_hidden_states=True).hidden_states
    lo_name = "bf16"
except Exception as e:
    print(f"  bf16 path failed ({type(e).__name__}); falling back to fp16")
    m_lo = AutoModelForCausalLM.from_pretrained("gpt2").to(dev).half().eval()
    with torch.no_grad():
        hs_lo = m_lo(ids, output_hidden_states=True).hidden_states
    lo_name = "fp16"

# ---- per-layer certified relative error (per token, then mean/median) ----
print(f"\nlow precision: {lo_name}.  E_cert[l] = ||h_lo - h_64|| / ||h_64|| per token\n")
print(f"{'layer':>5} {'mean E':>11} {'median E':>11} {'max E':>11} {'mean/med':>9}")
mean_e, med_e = [], []
per_layer_tokerr = []
for l in range(L+1):
    a = hs64[l][0].double()                       # (seq,d) reference
    b = hs_lo[l][0].double()                       # (seq,d) low precision -> f64 for compare
    num = torch.linalg.vector_norm(b - a, dim=-1)  # per-token error
    den = torch.linalg.vector_norm(a, dim=-1) + 1e-300
    te = (num/den).numpy()                         # (seq,)
    per_layer_tokerr.append(te)
    me, md, mx = te.mean(), np.median(te), te.max()
    mean_e.append(me); med_e.append(md)
    if l == 0 or l == L or l % 2 == 0:
        print(f"{l:>5} {me:>11.3e} {md:>11.3e} {mx:>11.3e} {me/max(md,1e-300):>8.2f}x")

mean_e = np.array(mean_e); med_e = np.array(med_e)
# slope over the BLOCK checkpoints (l=1..L; l=0 is the embedding)
Ls = np.arange(1, L+1)
slope_mean = np.polyfit(Ls, np.log(mean_e[1:]), 1)[0]
slope_med  = np.polyfit(Ls, np.log(med_e[1:]),  1)[0]
# linear-vs-exponential discriminator: compare to a pure linear accumulation
lin_ref = mean_e[1] * Ls            # if error were L * (single-layer error)
exp_ratio = mean_e[-1] / mean_e[1]  # actual growth factor over the stack

print("\n" + "-"*72)
print("VERDICT — P1 (sub-exponential typical case) vs F1 (exponential)")
print("-"*72)
print(f"  mean   E_cert: L1 {mean_e[1]:.3e} -> L{L} {mean_e[-1]:.3e}   "
      f"log-slope {slope_mean:+.3f}/layer")
print(f"  median E_cert: L1 {med_e[1]:.3e} -> L{L} {med_e[-1]:.3e}   "
      f"log-slope {slope_med:+.3f}/layer")
print(f"  growth factor E[L]/E[1] = {exp_ratio:.2f}x   (linear would be {L}x; "
      f"exponential at Stage-1 worst-case slope +0.245 would be {np.exp(0.245*(L-1)):.0f}x)")

# Stage-1 expansive worst case was slope ~ +0.245/layer. Define the gate:
if slope_mean < 0.05:
    verdict = ("P1 HOLDS (and then some): typical-case error is NON-amplifying "
               "through depth on trained weights.\n  Residual + LayerNorm + trained "
               "weights sit firmly in the CONTRACTIVE regime — the worst-case\n  "
               "exponential (Stage 1) does NOT transfer to the typical case. F1 does "
               "not fire.")
elif slope_mean < 0.10:
    verdict = ("P1 holds (weakly): mild sub-exponential growth, far below the "
               "Stage-1 worst case. F1 does not fire.")
else:
    verdict = ("F1 FIRES: trained-weight typical-case error grows exponentially "
               "(slope > 0.10/layer).\n  Residual attenuation does NOT save typical "
               "inputs. This would be a major negative result.")
print(f"\n  => {verdict}")

# ---- headline: which tokens fire? concentration at the output layer ----
print("\n" + "-"*72)
print("HEADLINE (§2 #2 / arXiv 2505.24187): is error CONCENTRATED on few tokens?")
print("-"*72)
out = per_layer_tokerr[-1]
order = np.argsort(out)[::-1]
total = out.sum()
top1_share = out[order[0]] / total
top10pct_n = max(1, seq//10)
top10pct_share = out[order[:top10pct_n]].sum() / total
toks = tok.convert_ids_to_tokens(ids[0].tolist())
print(f"  output-layer per-token error: top-1 token carries {top1_share*100:.1f}% of total")
print(f"  top {top10pct_n} tokens (~10%) carry {top10pct_share*100:.1f}% of total error")
print(f"  highest-error tokens: " +
      ", ".join(f"{repr(toks[i])}({out[i]:.1e})" for i in order[:5]))
conc = "CONCENTRATED (heavy-tailed across tokens)" if top10pct_share > 0.3 else "diffuse"
print(f"  -> error distribution across tokens is {conc}")

print("\n" + "="*72)
print("Real run, this sandbox. seed 20260616. Reference float64 (dps50-certified op set).")
print("="*72)
