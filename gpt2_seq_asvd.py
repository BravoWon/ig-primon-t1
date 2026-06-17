"""Sequential ASVD: does propagated (non-stale) calibration close the all-layers low-rank catastrophe?

In the eigen-sweep, one-shot ASVD over all layers was WORSE than plain SVD (act-weighting computed on the clean
model, but applied after every layer is already compressed -> stale stats). That is the SAME defect that made
parallel GPTQ +150%. The fix that worked for GPTQ: SEQUENTIAL reconstruction -- calibrate each layer on the
PROPAGATED (already-compressed-upstream) activations, compress, propagate. This applies that frame to low-rank.

Per block, in order: (1) forward on the propagated inputs with hooks collecting each module's *current* input RMS;
(2) ASVD-compress the 4 weights at matched rank using those fresh stats; (3) re-forward the compressed block to
get the next block's inputs. Compared against: parallel ASVD (stale), sequential PLAIN low-rank (isolates the
act-weighting), and the quant-GPTQ frontier at matched bits.

PRE-REG: H = sequential >> parallel (staleness was the cause); act-weighting helps with fresh stats; but
sequential low-rank STILL loses to quantization at matched storage (spectrum too spread -> strong eigenweight
form stays dead). Falsifiers: seq ~ parallel (staleness not the cause); seq-ASVD <= quant (strong form resurrected).
[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, matched stored-bits.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

import gpt2_eigensweep as E

torch.set_grad_enabled(False)
DEV = E.DEV; GPT2 = E.GPT2; L = E.L; N_CAL = 48; GROUP = E.GROUP
RATIOS = [0.5, 0.25]


class _Stop(Exception):
    pass


def _block_units(blk):
    return [blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj]


def _catch0(m, cal):
    blocks = m.transformer.h; orig0 = blocks[0]; caught = {"h": [], "kw": None}
    class Catcher(nn.Module):
        def __init__(s, mod): super().__init__(); s.mod = mod
        def forward(s, hs, *a, **kw):
            caught["h"].append(hs.detach())
            if caught["kw"] is None: caught["kw"] = dict(kw)
            raise _Stop()
    blocks[0] = Catcher(orig0)
    for sq in cal:
        try:
            m(sq.to(DEV))
        except _Stop:
            pass
    blocks[0] = orig0
    kw0 = caught["kw"] or {}
    for k in list(kw0):
        if "cache" in k or "past" in k: kw0[k] = None
        if k == "use_cache": kw0[k] = False
    return caught["h"], kw0


def seq_compress(m, cal, r, aware):
    """Sequential low-rank: propagate inputs, calibrate each layer on its CURRENT input, compress, propagate."""
    inps, kw0 = _catch0(m, cal)
    for blk in m.transformer.h:
        U = _block_units(blk); ms2 = {id(u): None for u in U}; cnt = {id(u): 0 for u in U}
        def mk(u):
            def hook(mod, ip):
                x = ip[0].reshape(-1, ip[0].size(-1)).float()
                s = (x * x).sum(0)
                ms2[id(u)] = s if ms2[id(u)] is None else ms2[id(u)] + s; cnt[id(u)] += x.shape[0]
            return u.register_forward_pre_hook(hook)
        hk = [mk(u) for u in U]
        for j in range(len(inps)):
            blk(inps[j], **kw0)
        for h in hk:
            h.remove()
        for u in U:
            W = u.weight.data; mm, nn_ = W.shape; k = max(1, int(round(r * mm * nn_ / (mm + nn_))))
            rms = (ms2[id(u)] / max(cnt[id(u)], 1)).clamp_min(1e-12).sqrt() if aware else None
            u.weight.data = E.lowrank(W, k, rms)
        for j in range(len(inps)):
            o = blk(inps[j], **kw0)
            inps[j] = o[0] if isinstance(o, (tuple, list)) else o
    return m


def run():
    print("[GPT-2 small SEQUENTIAL ASVD]  does propagated calibration close the low-rank catastrophe?\n")
    tok = AutoTokenizer.from_pretrained(GPT2); ids = E.corpus_ids(tok)
    eval_ids = ids[:8192].to(DEV)
    cal = [ids[8192 + i * L: 8192 + (i + 1) * L].unsqueeze(0) for i in range(N_CAL)]
    m = AutoModelForCausalLM.from_pretrained(GPT2).to(torch.float32).to(DEV).eval()
    T = E.targets(m); orig = {n: mod.weight.detach().cpu().clone() for n, mod in T}
    Hm, rms_clean = E.collect(m, cal, T)
    gold = E.ppl_strided(m, eval_ids); print(f"  gold PPL = {gold:.3f}\n")

    def restore():
        for n, mod in T:
            mod.weight.data = orig[n].to(DEV).clone()

    rows = []
    for r in RATIOS:
        b = max(2, min(8, int(round(16 * r))))
        # parallel (stale) low-rank, plain + actW
        restore()
        for n, mod in T:
            W = mod.weight.data; mm, nn_ = W.shape; k = max(1, int(round(r * mm * nn_ / (mm + nn_))))
            mod.weight.data = E.lowrank(W, k, None)
        par_plain = E.ppl_strided(m, eval_ids)
        restore()
        for n, mod in T:
            W = mod.weight.data; mm, nn_ = W.shape; k = max(1, int(round(r * mm * nn_ / (mm + nn_))))
            mod.weight.data = E.lowrank(W, k, rms_clean[n])
        par_actw = E.ppl_strided(m, eval_ids)
        # sequential, plain + actW
        restore(); seq_compress(m, cal, r, False); seq_plain = E.ppl_strided(m, eval_ids)
        restore(); seq_compress(m, cal, r, True);  seq_actw = E.ppl_strided(m, eval_ids)
        # quant-GPTQ frontier at matched bits
        restore()
        for n, mod in T:
            mod.weight.data = E.quant(mod.weight.data, Hm[n], b, GROUP, True)
        qg = E.ppl_strided(m, eval_ids)
        restore()
        rows.append((r, b, par_plain, par_actw, seq_plain, seq_actw, qg))

    print(f"  {'ratio':>6} {'bits':>4} {'par-plain':>10} {'par-actW':>10} {'seq-plain':>10} {'seq-actW':>10} {'quant-GPTQ':>11}")
    for r, b, pp, pa, sp, sa, qg in rows:
        print(f"  {r:>6.3f} {b:>4d} {pp:>10.2f} {pa:>10.2f} {sp:>10.2f} {sa:>10.2f} {qg:>11.2f}")

    print("\n[VERDICT -- scored against pre-registration]")
    for r, b, pp, pa, sp, sa, qg in rows:
        seq_better = sa < pa
        actw_helps = sa < sp
        beats_quant = sa <= qg
        print(f"  r={r} ({b}-bit): seq-ASVD {sa:.1f} vs parallel-ASVD {pa:.1f} -> staleness was the cause: {seq_better} "
              f"({pa/sa:.1f}x better); act-weighting helps (seq): {actw_helps}; beats quant ({qg:.1f}): {beats_quant}")
    print("\n  Reading: if seq>>parallel but seq-ASVD still > quant, the sequential FRAME is right for low-rank too,")
    print("  yet the strong eigenweight form stays dead -- quantization-in-the-use-metric remains the frontier.")
    print("\n[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, matched stored-bits, sequential propagation.")
    return rows, gold


if __name__ == "__main__":
    run()
