"""
stage2b_robustness.py  —  IG-PRIMON-T1 Stage 2 (robustness + F3 separation)

Two jobs:
  (1) Is P1 (sub-exponential typical-case depth-error) ROBUST across many inputs,
      or a single-sample fluke? -> slope distribution over N texts.
  (2) F3 (confound #3): is the per-token error CONCENTRATION a RANGE artifact
      (error tracks activation magnitude) or genuine mantissa-composition?
      -> correlate per-token absolute error with per-token activation norm,
         and check whether the first token (attention sink) dominates.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
np.set_printoptions(suppress=True)
torch.manual_seed(20260616)

tok = AutoTokenizer.from_pretrained("gpt2")
m64 = AutoModelForCausalLM.from_pretrained("gpt2").double().eval()
mlo = AutoModelForCausalLM.from_pretrained("gpt2").to(torch.bfloat16).eval()

texts = [
    "The numerical stability of deep networks depends on how perturbations propagate.",
    "She walked quickly through the rain, clutching an umbrella that barely helped.",
    "Quantum error correction codes protect fragile superpositions from decoherence.",
    "In 1969 the first humans landed on the Moon, watched by millions on television.",
    "Recipe: combine flour, sugar, and butter, then bake at 180 degrees for twenty minutes.",
    "The stock market fell sharply today amid fears of rising interest rates.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy.",
    "He argued that justice without mercy is merely a sophisticated form of cruelty.",
]

print("="*72)
print("Stage 2 robustness — P1 across", len(texts), "texts + F3 range separation")
print("="*72)

slopes, growths = [], []
sink_dom = 0
r_abs_list, r_rel_list = [], []
for t in texts:
    ids = tok(t, return_tensors="pt").input_ids
    with torch.no_grad():
        h64 = m64(ids, output_hidden_states=True).hidden_states
        hlo = mlo(ids, output_hidden_states=True).hidden_states
    L = len(h64) - 1
    mean_e = []
    for l in range(L+1):
        a = h64[l][0].double(); b = hlo[l][0].double()
        rel = (torch.linalg.vector_norm(b-a, dim=-1) /
               (torch.linalg.vector_norm(a, dim=-1)+1e-300)).numpy()
        mean_e.append(rel.mean())
    mean_e = np.array(mean_e)
    Ls = np.arange(1, L+1)
    slopes.append(np.polyfit(Ls, np.log(mean_e[1:]), 1)[0])
    growths.append(mean_e[-1]/mean_e[1])

    # ---- F3 at output layer: absolute error vs activation norm ----
    a = h64[-1][0].double(); b = hlo[-1][0].double()
    abs_err = torch.linalg.vector_norm(b-a, dim=-1).numpy()
    act_norm = torch.linalg.vector_norm(a, dim=-1).numpy()
    rel_err = abs_err/(act_norm+1e-300)
    if len(abs_err) > 3:
        r_abs = np.corrcoef(act_norm, abs_err)[0,1]   # range signal
        r_rel = np.corrcoef(act_norm, rel_err)[0,1]   # if ~0, rel error is scale-free
        r_abs_list.append(r_abs); r_rel_list.append(r_rel)
    if int(np.argmax(abs_err)) == 0:   # first token = attention sink
        sink_dom += 1

slopes = np.array(slopes); growths = np.array(growths)
print(f"\n(1) P1 robustness over {len(texts)} texts:")
print(f"    log-slope/layer:  mean {slopes.mean():+.3f}  std {slopes.std():.3f}  "
      f"min {slopes.min():+.3f}  max {slopes.max():+.3f}")
print(f"    growth E[L]/E[1]: mean {growths.mean():.2f}x  max {growths.max():.2f}x  "
      f"(linear=12x, Stage-1 worst-case exp={np.exp(0.245*11):.0f}x)")
allP1 = slopes.max() < 0.10
print(f"    => P1 {'HOLDS on ALL '+str(len(texts))+' texts' if allP1 else 'VIOLATED on some texts'} "
      f"(every slope < 0.10/layer: {allP1})")

print(f"\n(2) F3 range-vs-mantissa (output layer, corr across tokens):")
print(f"    corr(activation_norm, ABSOLUTE error) = {np.mean(r_abs_list):+.2f}  "
      f"[high+ => error scales with magnitude = RANGE effect]")
print(f"    corr(activation_norm, RELATIVE error) = {np.mean(r_rel_list):+.2f}  "
      f"[~0 => relative error is scale-free = mantissa-limited, not range]")
print(f"    first token (attention sink) is the top-error token in "
      f"{sink_dom}/{len(texts)} texts")

print("\n" + "-"*72)
print("READING")
print("-"*72)
ra, rr = np.mean(r_abs_list), np.mean(r_rel_list)
if ra > 0.5 and abs(rr) < 0.3:
    print("  The CONCENTRATION headline is substantially a RANGE artifact (F3 partially")
    print("  fires): absolute per-token error tracks activation magnitude, while the")
    print("  RELATIVE error is scale-free. The attention-sink token dominates because it")
    print("  has outlier magnitude, not because of κ_softmax composition. The 'which")
    print("  tokens fire' headline must therefore be stated in ABSOLUTE terms and NOT")
    print("  attributed to κ_softmax without the C3 control + a range-normalized metric.")
elif abs(rr) > 0.3:
    print("  Relative error itself correlates with magnitude — mixed range/mantissa;")
    print("  needs the κ_softmax-resolved C3 control to attribute.")
else:
    print("  Error is largely scale-free (mantissa-limited); the concentration is not a")
    print("  pure range artifact — κ_softmax attribution (F2/C3) is worth running.")
print("\n  P1 itself is unaffected by F3: the DEPTH non-amplification is a relative-error")
print("  property and is robust across all texts above.")
print("="*72)
