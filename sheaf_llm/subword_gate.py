#!/usr/bin/env python
"""subword-backoff gate -- HARDEN brick 3'. Does grounding beat (and ADD to) a realistic char/subword
fallback, instead of just beating the broken single-UNK baseline?

Brick 3' starved held-out nouns to ONE shared UNK, so flat got zero signal there. A real subword/BPE LM
gets PARTIAL signal from spelling. So four arms, identical backbone, differing only in how a word's INPUT
is encoded -- decisive at HELD-OUT (embedding-starved) positions:
  flat          : whole-word emb;  held-out -> UNK            (brick-3' floor, zero signal)
  char          : fastText hashed char-n-gram composition;    held-out keeps its SPELLING (real subword LM)
  grounded      : whole-word + WordNet (supersense,depth);    held-out -> UNK + CATEGORY (brick 3')
  char+grounded : spelling AND category                       (use both)

All arms predict the SAME masked targets (held-out -> UNK as target), so PPL is comparable; only the
conditioning backoff differs.

PRE-REGISTERED.
  Hypothesis: char recovers much of the UNK gap (morphology is informative). The decisive number is
              char+grounded vs char at held-out positions -- does WordNet category add what spelling does
              not already encode?
  Outcomes  : (a) grounding still helps on top of char -> semantics beyond morphology, earns its place;
              (b) char+grounded ~= char -> grounding largely redundant with subword, and brick 3's -27%
                  was mostly beating broken UNK. Either is honest.
  Falsifier (for "grounding is worth it"): char+grounded shows no gain over char at held-out positions.

    python subword_gate.py
"""
import math
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
try:
    wn.ensure_loaded()
except LookupError:
    nltk.download("wordnet", quiet=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
V, DM, DG, NH, NL, FF, BLK = 6000, 192, 16, 6, 3, 512, 96
DT = DM - 2 * DG                              # token/char width so concat with grounding == DM
BSUB, KMAX = 8000, 24                         # char-ngram hash buckets, max ngrams/word
STEPS, BS, LR, DROP, SEED = 3000, 64, 3e-4, 0.1, 0
RED, BLUE, GREEN, PURPLE, NAVY = "#c0392b", "#2c6fbb", "#1e7d34", "#6f3d9a", "#15293f"
LEX = sorted({s.lexname() for s in wn.all_synsets("n")})
SS_NONE, D_NONE, NB = len(LEX), 8, 8


def fnv(s):
    x = 2166136261
    for c in s:
        x = ((x ^ ord(c)) * 16777619) & 0xffffffff
    return x % BSUB


def ngram_ids(word):
    w = "<" + word + ">"
    grams = {w}
    for n in (3, 4, 5):
        for i in range(len(w) - n + 1):
            grams.add(w[i:i + n])
    ids = sorted({fnv(g) for g in grams})[:KMAX]
    return ids if ids else [fnv("<unk>")]


def build():
    nltk.download("brown", quiet=True); nltk.download("wordnet", quiet=True)
    toks = [w.lower() for w in brown.words() if w[0].isalnum()]
    cnt = Counter(toks)
    vocab = [w for w, _ in cnt.most_common(V - 1)]
    stoi = {w: i + 1 for i, w in enumerate(vocab)}; UNK = 0
    itos = {i + 1: w for i, w in enumerate(vocab)}; itos[UNK] = "<unk>"
    feat = {}
    for w in vocab:
        syn = wn.synsets(w, pos="n")
        if syn:
            feat[w] = [LEX.index(syn[0].lexname()), syn[0].min_depth()]
    qs = np.quantile([d for _, d in feat.values()], np.linspace(0, 1, NB + 1)[1:-1])
    for w in feat:
        feat[w][1] = int(sum(feat[w][1] > q for q in qs))
    rng = np.random.default_rng(SEED)
    groundable = [w for w in feat if cnt[w] >= 20]; rng.shuffle(groundable)
    H = set(groundable[: int(0.30 * len(groundable))])
    ids = np.array([stoi.get(w, UNK) for w in toks], dtype=np.int64)
    masked = np.array([UNK if w in H else stoi.get(w, UNK) for w in toks], dtype=np.int64)
    g_ss = np.array([feat[w][0] if w in feat else SS_NONE for w in toks], dtype=np.int64)
    g_d = np.array([feat[w][1] if w in feat else D_NONE for w in toks], dtype=np.int64)
    buck = np.array([2 if w in H else (1 if (w in feat and w not in H) else 0) for w in toks], dtype=np.int64)
    # char-ngram table over all ids (incl UNK=0)
    NG = np.full((V, KMAX), BSUB, dtype=np.int64); LEN = np.ones(V, dtype=np.int64)
    for i in range(V):
        g = ngram_ids(itos.get(i, "<unk>")); NG[i, :len(g)] = g; LEN[i] = len(g)
    n = len(toks); cut = int(0.9 * n)
    print(f"[subword gate]  dev={DEV}  brown {n:,} toks  V={V}  groundable={len(feat)}  H(starved)={len(H)}  "
          f"char-buckets={BSUB}")
    return dict(ids=ids, masked=masked, g_ss=g_ss, g_d=g_d, buck=buck, cut=cut, n=n,
                NG=torch.from_numpy(NG), LEN=torch.from_numpy(LEN))


class CharBag(nn.Module):
    def __init__(self, dim, NG, LEN):
        super().__init__()
        self.emb = nn.Embedding(BSUB + 1, dim, padding_idx=BSUB)
        self.register_buffer("NG", NG); self.register_buffer("LEN", LEN)
    def forward(self, x):                       # x [B,T] token ids -> [B,T,dim] mean of ngram embeddings
        g = self.NG[x]                          # [B,T,KMAX]
        e = self.emb(g)                         # [B,T,KMAX,dim]
        m = (g != BSUB).float()[..., None]
        return (e * m).sum(2) / self.LEN[x][..., None].float()


class LM(nn.Module):
    def __init__(self, mode, NG, LEN):
        super().__init__(); self.mode = mode
        self.use_g = "grounded" in mode; self.use_char = "char" in mode
        if self.use_char:
            self.char = CharBag(DT if self.use_g else DM, NG, LEN)
        else:
            self.tok = nn.Embedding(V, DT if self.use_g else DM)
        if self.use_g:
            self.ss = nn.Embedding(SS_NONE + 1, DG); self.dp = nn.Embedding(D_NONE + 1, DG)
        self.pos = nn.Embedding(BLK, DM)
        layer = nn.TransformerEncoderLayer(DM, NH, FF, DROP, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, NL)
        self.ln = nn.LayerNorm(DM); self.head = nn.Linear(DM, V)
        self.register_buffer("mask", torch.triu(torch.full((BLK, BLK), float("-inf")), 1))
    def emb(self, xm, xu, ss, dp):
        base = self.char(xu) if self.use_char else self.tok(xm)
        if self.use_g:
            base = torch.cat([base, self.ss(ss), self.dp(dp)], -1)
        return base
    def forward(self, xm, xu, ss, dp):
        T = xm.size(1)
        h = self.emb(xm, xu, ss, dp) + self.pos(torch.arange(T, device=xm.device))[None]
        h = self.enc(h, mask=self.mask[:T, :T])
        return self.head(self.ln(h))


def run(mode, d):
    torch.manual_seed(SEED)
    m = LM(mode, d["NG"].to(DEV), d["LEN"].to(DEV)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    tmk = torch.from_numpy(d["masked"]).to(DEV); tid = torch.from_numpy(d["ids"]).to(DEV)
    tgs = torch.from_numpy(d["g_ss"]).to(DEV); tgd = torch.from_numpy(d["g_d"]).to(DEV)
    cut = d["cut"]; g = torch.Generator(device=DEV).manual_seed(SEED); m.train()
    for _ in range(STEPS):
        ix = torch.randint(0, cut - BLK - 1, (BS,), device=DEV, generator=g)
        idx = ix[:, None] + torch.arange(BLK, device=DEV)[None]
        logits = m(tmk[idx], tid[idx], tgs[idx], tgd[idx]); y = tmk[idx + 1]
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); buck = torch.from_numpy(d["buck"]).to(DEV)
    starts = np.arange(cut, d["n"] - BLK - 1, BLK)
    sums = torch.zeros(3, device=DEV); counts = torch.zeros(3, device=DEV)
    tot_s = torch.zeros((), device=DEV); tot_c = torch.zeros((), device=DEV)
    with torch.no_grad():
        for w0 in range(0, len(starts), BS):
            ix = torch.from_numpy(starts[w0:w0 + BS]).to(DEV)
            idx = ix[:, None] + torch.arange(BLK, device=DEV)[None]
            nll = F.cross_entropy(m(tmk[idx], tid[idx], tgs[idx], tgd[idx]).reshape(-1, V),
                                  tmk[idx + 1].reshape(-1), reduction="none")
            bf = buck[idx].reshape(-1); tot_s += nll.sum(); tot_c += nll.numel()
            for k in range(3):
                msk = bf == k; sums[k] += nll[msk].sum(); counts[k] += msk.sum()
    ppl = lambda s, c: math.exp((s / c).item())
    return sum(t.numel() for t in m.parameters()), ppl(tot_s, tot_c), [ppl(sums[k], counts[k]) for k in range(3)]


def main():
    d = build()
    arms = ["flat", "char", "grounded", "char+grounded"]
    print(f"\n{'model':16}{'params':>9}{'overall':>10}{'post-none':>11}{'post-seen-noun':>16}{'post-H(starved)':>17}")
    res = {}
    for mode in arms:
        p, ov, bk = run(mode, d); res[mode] = (p, ov, bk)
        print(f"{mode:16}{p/1e6:>7.2f}M{ov:>10.2f}{bk[0]:>11.2f}{bk[1]:>16.2f}{bk[2]:>17.2f}")

    H = lambda k: res[k][2][2]                      # post-H-noun PPL
    print(f"\nVERDICT at HELD-OUT (starved) positions -- PPL lower=better:")
    print(f"  flat(UNK)={H('flat'):.1f}  char={H('char'):.1f}  grounded={H('grounded'):.1f}  char+grounded={H('char+grounded'):.1f}")
    print(f"  subword recovers the UNK gap: char vs flat = {(H('char')-H('flat'))/H('flat')*100:+.1f}%")
    print(f"  grounding vs subword alone:   grounded vs char = {(H('grounded')-H('char'))/H('char')*100:+.1f}%")
    g_on_char = (H('char+grounded') - H('char')) / H('char') * 100
    print(f"  DECISIVE -- grounding ON TOP of subword: char+grounded vs char = {g_on_char:+.1f}%  -> "
          f"{'grounding ADDS beyond spelling (earns its place)' if g_on_char < -3 else 'grounding REDUNDANT with subword (FALSIFIED on top of char)'}")

    fig, ax = plt.subplots(figsize=(8.4, 4.8)); x = np.arange(4); w = 0.2
    cols = {"flat": RED, "char": BLUE, "grounded": GREEN, "char+grounded": PURPLE}
    labels = ["overall", "post-none", "post-seen\nnoun", "post-H-noun\n(starved)"]
    for i, k in enumerate(arms):
        vals = [res[k][1]] + res[k][2]
        ax.bar(x + (i - 1.5) * w, vals, w, color=cols[k], label=k)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("perplexity (lower better)")
    ax.set_title("Subword gate: does grounding add beyond a real char/subword backoff?", color=NAVY, fontsize=10.5)
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig("sheaf_llm/subword_gate.png", dpi=160); plt.close(fig)
    print("  wrote sheaf_llm/subword_gate.png")


if __name__ == "__main__":
    main()
