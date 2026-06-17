"""Control for gpu_fapp_hardware.py: is the 'P3 revives' divergence REAL signal or GPU noise?

Two kill-shots for the artifact hypothesis, before any P3 claim is allowed to stand:
  (1) DETERMINISM: run the all-bf16 GPU forward twice. KL between the two runs = the noise floor. If it is
      comparable to the FP8 demotion marginals (~e-3), impP is noise-corrupted and 'P3 revives' is an artifact.
  (2) REPRODUCIBILITY: compute the practical-bf16-reference importance ranking (impP) TWICE. If the top-k
      protected set is NOT stable across identical runs, the divergence from impC is noise, not the reference.
The hardware P3 result only survives if: noise floor << marginals AND impP self-overlap ~100%.
[V-hw] RTX 5070 sm_120. Same regime as gpu_fapp_hardware.py.
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

import precision_allocator as PA
import gpu_fapp_hardware as H            # reuse q_fp8 / patched / units / kl / spearman / TEXTS

torch.set_grad_enabled(False)
DEV = "cuda:0"


def run():
    print("[control] is the hardware P3 divergence real signal or GPU noise?\n")
    tok = GPT2TokenizerFast.from_pretrained(PA.GPT2_PATH)
    Conv1D.forward = H.patched
    mg = GPT2LMHeadModel.from_pretrained(PA.GPT2_PATH).to(torch.bfloat16).to(DEV).eval()
    U = H.units(mg); nU = len(U)
    for u in U:
        u._prec = "bf16"

    # (1) determinism / noise floor: all-bf16 forward twice
    noise = []
    for txt in H.TEXTS:
        ids = tok(txt, return_tensors="pt").input_ids[:, :32].to(DEV)
        a = mg(ids).logits[0].float().cpu(); b = mg(ids).logits[0].float().cpu()
        noise.append((H.kl(a, b), (a - b).abs().max().item()))
    nkl = float(np.mean([n[0] for n in noise])); nmax = float(np.max([n[1] for n in noise]))
    print(f"(1) DETERMINISM: all-bf16 forward run-twice  mean KL = {nkl:.2e}  max abs logit diff = {nmax:.2e}")

    # (2) reproducibility of impP (practical bf16-reference importance), two independent passes
    def score_impP():
        imp = np.zeros(nU)
        for txt in H.TEXTS:
            ids = tok(txt, return_tensors="pt").input_ids[:, :32].to(DEV)
            for u in U:
                u._prec = "bf16"
            refP = mg(ids).logits[0].float().cpu()
            for i in range(nU):
                U[i]._prec = "fp8"; cfg = mg(ids).logits[0].float().cpu(); U[i]._prec = "bf16"
                imp[i] += H.kl(refP, cfg)
        return imp / len(H.TEXTS)

    p1 = score_impP(); p2 = score_impP()
    sp = H.spearman(p1, p2)
    med_marg = float(np.median(p1)); top_marg = float(np.max(p1))
    print(f"    median FP8 demotion marginal = {med_marg:.2e}   top marginal = {top_marg:.2e}")
    print(f"    noise floor / median marginal = {nkl/ (med_marg+1e-30):.2f}   "
          f"(<<1 needed: marginals must dominate noise)")
    print(f"\n(2) REPRODUCIBILITY of impP ranking across two identical passes: Spearman = {sp:+.4f}")
    for k in (4, 8, 16):
        s1 = set(np.argsort(-p1)[:k].tolist()); s2 = set(np.argsort(-p2)[:k].tolist())
        print(f"    top-{k} self-overlap = {len(s1 & s2)/k:.0%}")

    ok = (nkl < 0.1 * med_marg) and (sp > 0.95)
    print("\n[CONTROL VERDICT]",
          "PASS -- noise floor << marginals AND impP ranking reproducible; the impC-vs-impP divergence is REAL."
          if ok else
          "FAIL -- GPU noise is comparable to the signal; the 'P3 revives' result is NOT trustworthy, retract it.")
    return ok


if __name__ == "__main__":
    run()
