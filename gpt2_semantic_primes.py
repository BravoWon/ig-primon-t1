"""META-THESIS PROBE: does meaning PROJECT onto a discrete semantic-prime basis, and how big must the basis be?

The thesis: represent meaning by PROJECTION onto discrete structured atoms (NSM semantic primes), not float-
operated. The strong claim predicts the ~65 Natural-Semantic-Metalanguage primes (Wierzbicka/Goddard) form a
PRIVILEGED spanning basis for a model's meaning geometry, and composites DECODE into their defining primes.
Our spread-spectrum prior predicts the opposite extreme: meaning is high-dim/distributed, 65 atoms under-span,
the basis must grow large. This measures exactly where reality lands between those two.

THREE measurements on GPT-2 token-embedding geometry (768-d), each with the control that makes it decisive:
  (1) SPANNING  -- variance of a broad word sample captured by span(prime-65), vs PCA-65 (OPTIMAL 65-d subspace,
      the upper bound) and vs RANDOM-65 (distribution over 30 seeds: is prime an outlier above random?).
  (2) SCALING   -- reconstruction R^2 vs basis size {16..512}: prime-seeded vs random vs PCA. The crux number:
      how large must a discrete basis grow to span -- and does prime-seeding ever beat random at any size?
  (3) COMPOSITION (illustrative) -- NNLS-decompose composite words onto the prime basis; do top loadings match
      the NSM-intuitive primes (kill->DIE/DO/BECAUSE) above a chance baseline?

PRE-REG (honest): thesis -> prime ~ PCA, prime >> random, composites align. Spread-prior -> prime ~ random << PCA,
need a big basis, alignment ~ chance. Falsifier for the thesis: prime not above the random-65 distribution.
[V] gpt2-small token embeddings, fp32. Scope: input-embedding geometry (a proxy for 'meaning'); noted, not hidden.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.optimize import nnls

torch.set_grad_enabled(False)
DEV = "cpu"; GPT2 = "C:/Users/JT-DEV1/Documents/gpt2-sm"

PRIMES = ["i", "you", "someone", "something", "people", "body", "kind", "part", "this", "same", "other",
          "one", "two", "some", "all", "many", "few", "good", "bad", "big", "small", "know", "think", "want",
          "feel", "see", "hear", "say", "words", "true", "do", "happen", "move", "be", "there", "mine",
          "live", "die", "time", "now", "before", "after", "moment", "where", "place", "here", "above",
          "below", "far", "near", "side", "inside", "touch", "not", "maybe", "can", "because", "if", "very",
          "more", "like"]

COMPOSITES = {                                                       # illustrative NSM-intuitive expected primes
    "kill": ["die", "do", "because", "happen", "body"],
    "destroy": ["do", "bad", "not", "happen"],
    "happy": ["feel", "good"],
    "sad": ["feel", "bad"],
    "remember": ["think", "before", "know"],
    "love": ["feel", "good", "want", "very"],
    "fear": ["feel", "bad", "can", "happen"],
    "murder": ["die", "do", "bad", "because"],
}


def main():
    tok = AutoTokenizer.from_pretrained(GPT2)
    model = AutoModelForCausalLM.from_pretrained(GPT2).to(torch.float32).eval()
    Wte = model.transformer.wte.weight.detach().float()             # (V, 768)
    V, D = Wte.shape
    mu = Wte.mean(0)                                                 # global center

    def emb(word):                                                  # mean of subword token embeddings, centered
        ids = tok.encode(" " + word)
        return (Wte[ids].mean(0) - mu)

    Bp = torch.stack([emb(w) for w in PRIMES])                      # (nP, 768) prime basis (centered)
    nP = Bp.shape[0]

    # broad target sample: whole-word, alphabetic, >=3 chars, most-frequent (low id) tokens
    tgt_ids = []
    for tid in range(V):
        s = tok.decode([tid])
        if s.startswith(" ") and s[1:].isalpha() and len(s) >= 4:
            tgt_ids.append(tid)
        if len(tgt_ids) >= 2000:
            break
    X = (Wte[tgt_ids] - mu)                                          # (T,768) centered targets

    def span_r2(B, Xt):                                              # variance of Xt captured by span(rows of B)
        Bf = B.float(); G = Bf @ Bf.t()
        P = Bf.t() @ torch.linalg.pinv(G) @ Bf                      # (768,768) projector onto span
        Xh = Xt @ P
        return float(1 - ((Xt - Xh) ** 2).sum() / (Xt ** 2).sum())

    # PCA-k optimal subspace (top-k right singular vectors of X)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    def pca_r2(k):
        return float((S[:k] ** 2).sum() / (S ** 2).sum())

    rng = np.random.default_rng(0)
    def rand_basis(k):
        idx = rng.choice(len(tgt_ids), size=k, replace=False)
        return X[idx]

    print("[META-THESIS PROBE]  does meaning project onto the NSM semantic-prime basis?\n")
    print(f"  GPT-2 token-embedding geometry: V={V}, D={D}; prime basis nP={nP}; targets T={len(tgt_ids)}\n")

    # (1) SPANNING at matched size nP
    r_prime = span_r2(Bp, X); r_pca = pca_r2(nP)
    rr = np.array([span_r2(rand_basis(nP), X) for _ in range(30)])
    z = (r_prime - rr.mean()) / (rr.std() + 1e-9)
    print("[1] SPANNING (variance of 2000 words captured by a size-%d subspace):" % nP)
    print(f"    prime-{nP}      R^2 = {r_prime:.4f}")
    print(f"    random-{nP}     R^2 = {rr.mean():.4f} +/- {rr.std():.4f}  (30 seeds)")
    print(f"    PCA-{nP} (best) R^2 = {r_pca:.4f}   <- optimal upper bound")
    print(f"    prime vs random: z = {z:+.2f}  ({'prime ABOVE random' if z > 2 else 'prime ~ random (not privileged)' if abs(z) <= 2 else 'prime BELOW random'})")
    frac_of_opt = r_prime / r_pca
    print(f"    prime captures {100*frac_of_opt:.1f}% of the optimal-65 subspace's variance\n")

    # (2) SCALING
    print("[2] SCALING (R^2 vs basis size):  how big must a discrete basis grow to span?")
    print(f"    {'size':>5} {'prime-seeded':>13} {'random':>9} {'PCA(opt)':>9}")
    for k in [16, 32, nP, 128, 256, 512]:
        if k <= nP:
            pb = Bp[:k]
        else:                                                       # prime-seeded: all primes + random fill
            extra = rand_basis(k - nP); pb = torch.cat([Bp, extra], 0)
        rp = span_r2(pb, X)
        rr_k = np.mean([span_r2(rand_basis(k), X) for _ in range(8)])
        print(f"    {k:>5} {rp:>13.4f} {rr_k:>9.4f} {pca_r2(k):>9.4f}")
    print()

    # (3) COMPOSITION (NNLS decompose onto prime basis)
    print("[3] COMPOSITION (NNLS onto prime basis; does it load the NSM-intuitive primes?)")
    Bp_np = Bp.numpy().T                                            # (768, nP) columns = prime atoms
    aligns = []; chance = []
    for w, expect in COMPOSITES.items():
        v = emb(w).numpy()
        coef, _ = nnls(Bp_np, v)                                    # non-negative coeffs over primes
        order = np.argsort(-coef)
        top = [PRIMES[i] for i in order[:6] if coef[i] > 1e-6]
        tot = coef.sum() + 1e-9
        a = sum(coef[PRIMES.index(p)] for p in expect if p in PRIMES) / tot
        # chance: random expected-set of same size
        rand_set = [PRIMES[i] for i in rng.choice(nP, size=len(expect), replace=False)]
        c = sum(coef[PRIMES.index(p)] for p in rand_set) / tot
        aligns.append(a); chance.append(c)
        hit = [p for p in top if p in expect]
        print(f"    {w:>9} -> top: {top}   | expected-prime mass {a:.2f} (chance {c:.2f})  hits:{hit}")
    print(f"    mean expected-prime mass {np.mean(aligns):.3f}  vs chance {np.mean(chance):.3f}  "
          f"-> {'ALIGNED above chance' if np.mean(aligns) > np.mean(chance)+0.05 else 'NOT above chance'}\n")

    print("[VERDICT -- scored against pre-registration]")
    privileged = z > 2
    spans = frac_of_opt > 0.8
    print(f"  prime basis privileged over random: {privileged} (z={z:+.2f})")
    print(f"  prime-65 spans (>80% of optimal-65): {spans} ({100*frac_of_opt:.0f}%)")
    if privileged and spans:
        print("  -> THESIS SUPPORTED: meaning projects onto the semantic-prime basis; discrete projection is real.")
    elif privileged and not spans:
        print("  -> PARTIAL: primes are a real (privileged) structure but UNDER-span -- the discrete basis must grow")
        print("     (structure real, basis large): consistent with the spread spectrum. The leg-up survives only if")
        print("     a LARGER discrete simplicial basis keeps decodability.")
    else:
        print("  -> THESIS NOT SUPPORTED in this geometry: prime basis is not privileged over random words; meaning")
        print("     is distributed, not projectable onto NSM primes. The elegance is ours to describe, not the model's.")
    print("\n[V] gpt2-small token embeddings, fp32. Proxy = input-embedding geometry (noted scope, not hidden states).")


if __name__ == "__main__":
    main()
