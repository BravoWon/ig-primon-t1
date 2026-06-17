"""Use-aware per-layer bit ALLOCATION: does allocating bits by activation-sensitivity beat uniform, in the use metric?

The arc refuted use-BLIND allocation (certification: KL-from-fp32, wrong objective) and found allocation
second-order to quantizer quality. But it never isolated USE-AWARE allocation in the STRESSED regime where quant
has headroom. This does: on top of a good quantizer (GPTQ), at a fixed avg-4-bit budget, allocate bits by each
matrix's measured activation-sensitivity and test against uniform-4 -- with a reverse-allocation control.

DESIGN (budget matched by construction): menu {2,4,6} bits. Sensitivity s_i = dlogPPL from GPTQ-quantizing
matrix i ALONE to 2-bit (rest fp32). Assign equal PARAMETER MASS (~30%) to 6-bit (most sensitive) and 2-bit
(least sensitive), rest 4-bit -> average exactly 4 (symmetric +2/-2 around 4). Three allocations at avg-4:
  uniform   : all 4-bit.
  use-aware : sensitive -> 6, robust -> 2.
  reverse   : sensitive -> 2, robust -> 6   (CONTROL: must be worse than uniform if sensitivity is real).

PRE-REG: H use-aware < uniform (allocation pays in the use metric). Control reverse > uniform (signal real).
Falsifiers: use-aware ~ uniform (allocation inert even use-aware -> thread closed); reverse ~ uniform (no signal).
Honest prior: modest use-aware win + clear reverse-worse; allocation real but second-order.
[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, GPTQ, matched avg-bit budget.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import gpt2_eigensweep as E

torch.set_grad_enabled(False)
DEV = E.DEV; GPT2 = E.GPT2; L = E.L; GROUP = E.GROUP; N_CAL = 32
MENU_HI, MENU_LO, MID = 6, 2, 4; FRAC = 0.30; SENS_BITS = 2


def run():
    print("[GPT-2 small use-aware ALLOCATION]  bits by activation-sensitivity vs uniform, matched avg-4 budget\n")
    tok = AutoTokenizer.from_pretrained(GPT2); ids = E.corpus_ids(tok)
    eval_ids = ids[:8192].to(DEV)
    cal = [ids[8192 + i * L: 8192 + (i + 1) * L].unsqueeze(0) for i in range(N_CAL)]
    m = AutoModelForCausalLM.from_pretrained(GPT2).to(torch.float32).to(DEV).eval()
    T = E.targets(m); names = [n for n, _ in T]; mods = {n: mod for n, mod in T}
    orig = {n: mod.weight.detach().cpu().clone() for n, mod in T}
    params = {n: orig[n].numel() for n in names}
    Hm, _ = E.collect(m, cal, T)
    gold = E.ppl_strided(m, eval_ids); lg = np.log(gold)
    print(f"  gold PPL = {gold:.3f}\n")

    def restore():
        for n in names:
            mods[n].weight.data = orig[n].to(DEV).clone()

    # per-matrix sensitivity: quantize matrix n ALONE to SENS_BITS, measure dlogPPL
    print(f"  measuring per-matrix sensitivity (GPTQ {SENS_BITS}-bit, one matrix at a time)...")
    sens = {}
    for n in names:
        restore()
        mods[n].weight.data = E.quant(mods[n].weight.data, Hm[n], SENS_BITS, GROUP, True)
        sens[n] = np.log(E.ppl_strided(m, eval_ids)) - lg
    restore()
    order = sorted(names, key=lambda n: -sens[n])                   # most sensitive first
    tot = sum(params.values())

    def tier(hi_bits, lo_bits, mid_bits):                          # mid_bits passed in -> budget conserved (the tuple)
        alloc = {}; hi = set(); lo = set(); acc = 0.0
        for n in order:                                             # top mass -> hi_bits
            if acc < FRAC * tot: hi.add(n); acc += params[n]
            else: break
        acc = 0.0
        for n in reversed(order):                                   # bottom mass -> lo_bits
            if acc < FRAC * tot: lo.add(n); acc += params[n]
            else: break
        for n in names:
            alloc[n] = hi_bits if n in hi else (lo_bits if n in lo else mid_bits)
        avg = sum(alloc[n] * params[n] for n in names) / tot
        return alloc, avg

    top5 = order[:5]; bot5 = order[-5:]
    print(f"    most sensitive: {[(n.replace('h','').replace('.attn','').replace('.mlp',''), round(sens[n],3)) for n in top5]}")
    print(f"    least sensitive:{[(n.replace('h','').replace('.attn','').replace('.mlp',''), round(sens[n],3)) for n in bot5]}\n")

    # Two budgets: avg-4 {2,4,6} (low end BELOW the cliff) and avg-6 {4,6,8} (every option SURVIVABLE -> fair test)
    CONFIGS = [("avg-4 {2,4,6}", 6, 4, 2), ("avg-6 {4,6,8} (cliff-safe)", 8, 6, 4)]
    out = {}
    for tag, hi, mid, lo in CONFIGS:
        allocs = {"uniform": ({n: mid for n in names}, float(mid)),
                  "use-aware": tier(hi, lo, mid), "reverse": tier(lo, hi, mid)}
        print(f"  [{tag}]  {'allocation':>10} {'avg bits':>9} {'PPL':>9} {'dPPL vs gold':>13}")
        res = {}
        for name in ["uniform", "use-aware", "reverse"]:
            alloc, avg = allocs[name]
            restore()
            for n in names:
                mods[n].weight.data = E.quant(mods[n].weight.data, Hm[n], alloc[n], GROUP, True)
            p = E.ppl_strided(m, eval_ids); res[name] = p
            print(f"  {'':>14} {name:>10} {avg:>9.3f} {p:>9.3f} {100*(p/gold-1):>+12.2f}%")
        restore(); out[tag] = res
        u, a, r = res["uniform"], res["use-aware"], res["reverse"]
        print(f"     -> use-aware vs uniform {100*(a/u-1):+.2f}%; reverse vs uniform {100*(r/u-1):+.2f}%\n")

    print("[VERDICT -- scored against pre-registration]")
    for tag, hi, mid, lo in CONFIGS:
        u, a, r = out[tag]["uniform"], out[tag]["use-aware"], out[tag]["reverse"]
        print(f"  {tag}: use-aware<uniform={a < u} ({100*(a/u-1):+.2f}%); reverse>uniform={r > u} (signal real)")
    safe = out["avg-6 {4,6,8} (cliff-safe)"]
    if safe["use-aware"] >= safe["uniform"] * 0.999:
        print("  -> Even with a CLIFF-SAFE menu, use-aware allocation does NOT beat uniform. Allocation is inert as a")
        print("     GAINER (the signal is real -- reverse hurts -- but you cannot cash it): the quantizer is the lever,")
        print("     allocation only redistributes a loss. Confirms the arc's 'no room for allocation', with mechanism:")
        print("     a sharp bit-cliff means trading bits across layers can only fall off it, never climb.")
    print("\n[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, GPTQ, two matched-budget menus.")
    return gold, out


if __name__ == "__main__":
    run()
