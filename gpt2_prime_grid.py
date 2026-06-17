"""Prime-grid quantization: does encoding weights on a PRIME grid {0,1,2,3,5,7,11,13} beat FP4/INT4 at 4-bit?

The WEAK (literal) form of "encode weights on primes": use a prime-valued quantization grid. This is the only
form testable on the harness now -- it is a pure grid choice, not the compositional/factorization claim (which
lives in routing, not dense weights). All three grids are 8-magnitude (4-bit with sign), GPTQ + act-order +
group-128, normalized per-group by the grid max, compared at matched 4-bit on GPT-2-small.

PRE-REG (honest): a prime grid is denser-near-zero like FP4 (4 of its 8 magnitudes lie in [0,3]) -> I predict
prime ~ FP4, NO special structural advantage. Falsifier for the conjecture's weak form: prime <= FP4 (no prime
grid advantage). Support: prime < FP4 significantly. A null here does NOT refute the STRONG (factorization-as-
role-code) form -- that is a different experiment (gating/routing, not weight grids).
[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, GPTQ act-order g128, matched 4-bit.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import gpt2_eigensweep as E

torch.set_grad_enabled(False)
DEV = E.DEV; GPT2 = E.GPT2; L = E.L; GROUP = E.GROUP; N_CAL = 32

GRIDS = {
    "INT4 (uniform)": torch.tensor([0., 1, 2, 3, 4, 5, 6, 7]),
    "FP4 (E2M1)":     torch.tensor([0., .5, 1, 1.5, 2, 3, 4, 6]),
    "PRIME":          torch.tensor([0., 1, 2, 3, 5, 7, 11, 13]),
}


def gptq_grid(Wc, H, grid, group=GROUP, damp=0.01, act_order=True):
    """GPTQ with an arbitrary nonuniform magnitude grid. Wc is Conv1D (in,out); returns (in,out)."""
    dt = Wc.dtype
    W = Wc.t().contiguous().float()                                  # (out,in) Linear convention
    dev = W.device; out, inp = W.shape; H = H.float()
    g = grid.to(dev); mids = (g[:-1] + g[1:]) / 2; qmax = float(g[-1])
    dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0
    invperm = None
    if act_order:
        perm = torch.argsort(torch.diagonal(H), descending=True)
        W = W[:, perm]; H = H[perm][:, perm]; invperm = torch.argsort(perm)
    H[torch.arange(inp, device=dev), torch.arange(inp, device=dev)] += damp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    ss = None
    for i in range(inp):
        if i % group == 0:
            ss = W[:, i:min(i + group, inp)].abs().amax(1).clamp_min(1e-12) / qmax
        w = W[:, i]; d = Hinv[i, i]
        q = torch.sign(w) * g[torch.bucketize((w / ss).abs(), mids)] * ss
        err = (w - q) / d
        W[:, i] = q
        W[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
    if invperm is not None:
        W = W[:, invperm]
    return W.t().contiguous().to(dt)


def run():
    print("[GPT-2 small PRIME-GRID]  prime {0,1,2,3,5,7,11,13} vs FP4 vs INT4 at 4-bit (GPTQ act-order g128)\n")
    tok = AutoTokenizer.from_pretrained(GPT2); ids = E.corpus_ids(tok)
    eval_ids = ids[:8192].to(DEV)
    cal = [ids[8192 + i * L: 8192 + (i + 1) * L].unsqueeze(0) for i in range(N_CAL)]
    m = AutoModelForCausalLM.from_pretrained(GPT2).to(torch.float32).to(DEV).eval()
    T = E.targets(m); orig = {n: mod.weight.detach().cpu().clone() for n, mod in T}
    Hm, _ = E.collect(m, cal, T)
    gold = E.ppl_strided(m, eval_ids); print(f"  gold PPL = {gold:.3f}\n")

    def restore():
        for n, mod in T:
            mod.weight.data = orig[n].to(DEV).clone()

    print(f"  {'grid':>16} {'levels':>7} {'PPL':>9} {'dPPL vs gold':>13}")
    res = {}
    for name, grid in GRIDS.items():
        restore()
        for n, mod in T:
            mod.weight.data = gptq_grid(mod.weight.data, Hm[n], grid)
        p = E.ppl_strided(m, eval_ids); res[name] = p
        print(f"  {name:>16} {2*len(grid)-1:>7d} {p:>9.3f} {100*(p/gold-1):>+12.2f}%")
    restore()

    print("\n[VERDICT -- scored against pre-registration]")
    fp4 = res["FP4 (E2M1)"]; pr = res["PRIME"]; it = res["INT4 (uniform)"]
    print(f"  PRIME vs FP4: {100*(pr/fp4-1):+.2f}%   PRIME vs INT4: {100*(pr/it-1):+.2f}%")
    if pr < fp4 * 0.995:
        print("  -> PRIME GRID BEATS FP4 -- a real, surprising grid-level advantage. Weak-form conjecture supported;")
        print("     worth probing WHY (is it the spacing, or something structural?).")
    elif pr <= fp4 * 1.02:
        print("  -> PRIME ~ FP4 (within noise): prime grid is just another denser-near-zero grid, NO special advantage.")
        print("     Weak form null. Does NOT touch the STRONG form (factorization-as-role-code in routing).")
    else:
        print("  -> PRIME worse than FP4: the prime spacing (coarse 7->11->13 mid-range) costs. Weak form refuted.")
    print("\n  NOTE: this tests the GRID only. The compositional claim (0=pathway, 1=coherence, primes=decodable")
    print("  amplification roles via unique factorization) is a routing/gating experiment, not a weight grid.")
    print("\n[V] gpt2-small, WikiText-2 held-out, strided PPL, fp32, GPTQ act-order g128, matched 4-bit.")
    return gold, res


if __name__ == "__main__":
    run()
