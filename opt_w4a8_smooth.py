"""RESIDUAL FIX (driftwave R1, drift 0.96): certify the W4A8 smoothing delta with a PAIRED in-harness A/B.

The v0.2.11 ledger claimed "A8 +1.01%->+0.76% with smoothing" by subtracting naive-A8 (from opt_gptq_w4a4.py)
and smooth-A8 (from opt_gptq_smoothquant.py) ACROSS two runs -- no paired bootstrap, so the 0.25% was
uncertified. A fresh-context round-trip verifier flagged it. This runs naive-W4A8 vs smooth-W4A8 in ONE harness
(reusing opt_gptq_smoothquant's exact reconstruct/eval), evals BOTH at A8 on the same held-out set, and
paired-bootstraps smooth-A8 - naive-A8. Now the delta has an in-file CI and significance, or it doesn't.

Control retained: smooth-A16 must reproduce +0.67% (the fold is identity in full precision) before trusting A8.
[V-hw] OPT-2.7B, WikiText-2, fp32 master + fp32 gold, sequential GPTQ FP4-g128+act-order, held-out, paired.
"""
import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM

import opt_probe as OP
import opt_gptq_smoothquant as SQ                                    # reuse the SAME machinery -> directly comparable


def run():
    print("[OPT-2.7B W4A8 smoothing -- PAIRED]  certify the cross-file A8 delta in one harness\n")
    tok = AutoTokenizer.from_pretrained(OP.OPT_PATH)
    seqs = SQ.load_corpus(tok, SQ.N_CAL + SQ.N_EVAL); cal, ev = seqs[:SQ.N_CAL], seqs[SQ.N_CAL:]
    tgts = [s[0, 1:] for s in ev]

    nn.Linear.forward = SQ._patched
    m0 = OPTForCausalLM.from_pretrained(OP.OPT_PATH).to(torch.float32).to(SQ.DEV).eval()
    SQ.ACT["bits"] = None; gold, _ = SQ.ppl(m0, ev, tgts); del m0; gc.collect(); torch.cuda.empty_cache()

    ms = SQ.reconstruct(True, cal)                                  # smooth arm
    SQ.ACT["bits"] = None; ps16, _ = SQ.ppl(ms, ev, tgts)          # fold control
    SQ.ACT["bits"] = 8;    ps8, nll_s8 = SQ.ppl(ms, ev, tgts)
    del ms; gc.collect(); torch.cuda.empty_cache()

    mn = SQ.reconstruct(False, cal)                                 # naive arm
    SQ.ACT["bits"] = 8;    pn8, nll_n8 = SQ.ppl(mn, ev, tgts)
    del mn; gc.collect(); torch.cuda.empty_cache()
    nn.Linear.forward = SQ._ORIG

    ctrl_ok = abs(100 * (ps16 / gold - 1) - 0.67) < 0.6
    md, lo, hi = SQ.paired_bootstrap(nll_n8, nll_s8)               # smooth - naive at A8 (negative => smoothing helps)
    print(f"  {'config':>22} {'PPL':>9} {'dPPL vs gold':>13}")
    print(f"  {'fp32 gold':>22} {gold:>9.3f} {'--':>13}")
    print(f"  {'smooth-A16 (control)':>22} {ps16:>9.3f} {100*(ps16/gold-1):>+12.2f}%   [identity: {'OK' if ctrl_ok else 'FAIL'}]")
    print(f"  {'W4A8 naive':>22} {pn8:>9.3f} {100*(pn8/gold-1):>+12.2f}%")
    print(f"  {'W4A8 smoothed':>22} {ps8:>9.3f} {100*(ps8/gold-1):>+12.2f}%")
    print(f"\n  smoothing effect at A8 (smooth - naive), log-PPL paired bootstrap:")
    print(f"    mean {md:>+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]   (== {100*(np.exp(md)-1):+.3f}% PPL)")
    sig = (lo > 0) or (hi < 0)
    helps = sig and hi < 0
    print(f"    significant at 95%: {sig}  ({'smoothing improves A8' if helps else ('smoothing hurts A8' if sig else 'no significant A8 effect')})")

    print("\n[VERDICT]")
    if not ctrl_ok:
        print(f"  CONTROL FAILED (smooth-A16 = {100*(ps16/gold-1):+.2f}% != +0.67%) -- fold bug; A8 result untrustworthy.")
    elif helps:
        print(f"  CERTIFIED: smoothing improves W4A8 by {100*(np.exp(md)-1):+.2f}% PPL (paired, significant). The cross-file")
        print(f"  splice the verifier flagged is now an in-harness paired result. Deployable W4A8 = {100*(ps8/gold-1):+.2f}%.")
    elif sig:
        print(f"  REVERSED: smoothing significantly HURTS W4A8 ({100*(np.exp(md)-1):+.2f}%) -- the cross-file delta was wrong-signed.")
    else:
        print(f"  NULL: no significant A8 smoothing effect ({100*(np.exp(md)-1):+.2f}%, CI straddles 0). The cross-file 0.25%")
        print(f"  was noise; deployable W4A8 is reported as a range. Verifier was right to flag it.")
    print("\n[V-hw] RTX 5070 sm_120 + GTX 1660 Ti, OPT-2.7B, fp32 master, sequential GPTQ FP4-g128+act-order, WikiText-2.")
    return gold, ps16, pn8, ps8, (md, lo, hi)


if __name__ == "__main__":
    run()
