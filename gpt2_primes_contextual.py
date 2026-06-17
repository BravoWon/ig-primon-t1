"""TOKENIZATION CONTROL for the semantic-prime probe: re-run on CONTEXTUAL hidden states, not token embeddings.

The wte version handicapped the thesis: token embeddings are frequency-dominated, and the NSM primes are the
highest-frequency function words -> atypical clustered geometry -> they spanned BELOW random for tokenization
reasons, not semantic ones. And multi-token words were averaged into bag-of-fragments. This removes both
confounds: each word is embedded by its DEEP hidden state (layer L), which integrates its subwords through
attention and is far less frequency-dominated -- a fairer proxy for 'meaning'. If the spanning gap (prime vs
random) closes here, the wte refutation was substantially a tokenization artifact.

PRE-REG: bias direction says contextual should HELP primes vs the wte result (close the prime-below-random gap).
Honest prior: primes likely reach ~random or modestly above (artifact removed) but still UNDER-span a small
basis (meaning stays distributed -- the spread spectrum is real). Composition should stay >= chance, hopefully
cleaner. Falsifier for the thesis (now fair): prime still not above random AND still <80% of optimal.
[V] gpt2-small, layer-L last-token hidden state of each word in isolation, fp32.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.optimize import nnls

import gpt2_semantic_primes as SP                                   # reuse PRIMES, COMPOSITES

torch.set_grad_enabled(False)
DEV = "cuda:0"; GPT2 = "C:/Users/JT-DEV1/Documents/gpt2-sm"; LAYERS = [4, 8, 11]; NT = 1200


def main():
    tok = AutoTokenizer.from_pretrained(GPT2)
    model = AutoModelForCausalLM.from_pretrained(GPT2).to(torch.float32).to(DEV).eval()
    V = model.transformer.wte.weight.shape[0]

    # whole-word target vocabulary (same selection rule as the wte probe)
    tgt_words = []
    for tid in range(V):
        s = tok.decode([tid])
        if s.startswith(" ") and s[1:].isalpha() and len(s) >= 4:
            tgt_words.append(s[1:])
        if len(tgt_words) >= NT:
            break

    def ctx_embed(words, layer):
        out = []
        for w in words:
            ids = torch.tensor([tok.encode(" " + w)], device=DEV)
            hs = model(ids, output_hidden_states=True).hidden_states[layer][0]   # (n_tok,768)
            out.append(hs[-1].float())                              # last-token hidden state integrates the word
        return torch.stack(out)

    def span_r2(B, Xt):
        Bf = B.float(); P = Bf.t() @ torch.linalg.pinv(Bf @ Bf.t()) @ Bf
        Xh = Xt @ P
        return float(1 - ((Xt - Xh) ** 2).sum() / (Xt ** 2).sum())

    print("[TOKENIZATION CONTROL]  semantic-prime projection on CONTEXTUAL hidden states (vs wte)\n")
    print(f"  targets T={len(tgt_words)}, primes nP={len(SP.PRIMES)}; comparing layers {LAYERS}\n")
    print(f"  {'layer':>5} {'prime R2':>9} {'random R2':>16} {'PCA R2':>8} {'prime vs rand (z)':>18} {'%opt':>6}")
    rng = np.random.default_rng(0)
    ctx_for_comp = None
    for layer in LAYERS:
        Xt = ctx_embed(tgt_words, layer); mu = Xt.mean(0); Xt = Xt - mu
        Bp = ctx_embed(SP.PRIMES, layer) - mu
        nP = Bp.shape[0]
        U, S, Vh = torch.linalg.svd(Xt, full_matrices=False)
        r_prime = span_r2(Bp, Xt); r_pca = float((S[:nP] ** 2).sum() / (S ** 2).sum())
        rr = np.array([span_r2(Xt[rng.choice(len(tgt_words), nP, replace=False)], Xt) for _ in range(30)])
        z = (r_prime - rr.mean()) / (rr.std() + 1e-9)
        print(f"  {layer:>5} {r_prime:>9.4f} {rr.mean():>11.4f}+/-{rr.std():.4f} {r_pca:>8.4f} {z:>+18.2f} {100*r_prime/r_pca:>5.0f}%")
        if layer == 8:
            ctx_for_comp = (Bp, mu)

    # composition on layer-8 contextual
    print("\n  [composition @ layer 8, contextual]")
    Bp, mu = ctx_for_comp; Bp_np = Bp.cpu().numpy().T
    aligns, chance = [], []
    for w, expect in SP.COMPOSITES.items():
        v = (ctx_embed([w], 8)[0] - mu).cpu().numpy()
        coef, _ = nnls(Bp_np, v); order = np.argsort(-coef); tot = coef.sum() + 1e-9
        top = [SP.PRIMES[i] for i in order[:6] if coef[i] > 1e-6]
        a = sum(coef[SP.PRIMES.index(p)] for p in expect if p in SP.PRIMES) / tot
        rset = [SP.PRIMES[i] for i in rng.choice(len(SP.PRIMES), len(expect), replace=False)]
        c = sum(coef[SP.PRIMES.index(p)] for p in rset) / tot
        aligns.append(a); chance.append(c)
        print(f"    {w:>9} -> {top}   exp-mass {a:.2f} (chance {c:.2f}) hits:{[p for p in top if p in expect]}")
    print(f"    mean exp-mass {np.mean(aligns):.3f} vs chance {np.mean(chance):.3f}")

    print("\n[VERDICT]")
    print("  Compare the prime-vs-random z here to the wte probe's z=-13.5. If z moved up toward/above 0, the wte")
    print("  'primes span worse than random' was substantially a TOKENIZATION/frequency artifact. If primes still")
    print("  under-span the optimal subspace (%opt < ~80), meaning stays distributed regardless -- the strong")
    print("  'small discrete basis' thesis fails on a FAIR representation too, while composition (weak form) holds.")
    print("\n[V] gpt2-small contextual hidden states (isolation, last token), fp32.")


if __name__ == "__main__":
    main()
