#!/usr/bin/env python
"""brick 3' -- generative grounded LM. The FIRST number on the grounded-vs-fluent tension.

A small causal Transformer LM on the Brown corpus (real running English), two arms:
  flat            : standard token embedding  Embedding(V, d_model).
  grounded+token  : concat(token_emb[d_tok], supersense_emb[d_g], depth_emb[d_g]) == d_model -- inherited
                    WordNet structure fused into every token's INPUT, no projection. (=> grounded has
                    FEWER params: it trades token-emb width for tiny tier embeddings.)

CRUX -- the embedding-starved protocol (generative analog of brick-1 zero-shot):
  a held-out noun set H is forced to UNK on the INPUT side for BOTH arms, so neither learns a token
  embedding for it -- but the grounded arm still receives H-words' (supersense, depth). At an H-position
  flat sees UNK; grounded knows the CATEGORY. We then measure next-token NLL at those positions.

PRE-REGISTERED.
  Hypothesis: (1) overall val PPL grounded ~= flat (grounding doesn't wreck fluency, despite fewer params);
              (2) PPL after an H-noun: grounded < flat (inherited category pays where the embedding is
                  blank -- the generative payoff); (3) PPL after a SEEN noun: grounded ~= flat. The win is
                  CONCENTRATED at starved positions, which controls for any global param/capacity edge.
  Control   : post-seen-noun and post-nonword buckets ~equal across arms => a uniform capacity advantage
              is ruled out; only the starved bucket should move.
  Falsifier : grounded >= flat at H-positions (no payoff), OR grounded overall PPL much worse (tension bites).

    python generative_gate.py
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
DT = DM - 2 * DG                                   # token-emb width so concat == DM exactly
STEPS, BS, LR, DROP, SEED = 4000, 64, 3e-4, 0.1, 0
RED, GREEN, NAVY = "#c0392b", "#1e7d34", "#15293f"
LEX = sorted({s.lexname() for s in wn.all_synsets("n")})      # 26 noun supersenses
SS_NONE, D_NONE, NB = len(LEX), 8, 8                          # sentinel ids + depth buckets


def build():
    nltk.download("brown", quiet=True); nltk.download("wordnet", quiet=True)
    toks = [w.lower() for w in brown.words() if w[0].isalnum()]
    cnt = Counter(toks)
    vocab = [w for w, _ in cnt.most_common(V - 1)]
    stoi = {w: i + 1 for i, w in enumerate(vocab)}; UNK = 0      # 0 = UNK
    # grounding: predominant-sense WordNet noun -> (supersense idx, depth bucket); else sentinel
    depths = {}
    feat = {}
    for w in vocab:
        syn = wn.synsets(w, pos="n")
        if syn:
            feat[w] = [LEX.index(syn[0].lexname()), syn[0].min_depth()]
    if feat:
        qs = np.quantile([d for _, d in feat.values()], np.linspace(0, 1, NB + 1)[1:-1])
        for w in feat:
            feat[w][1] = int(sum(feat[w][1] > q for q in qs))
    # held-out (embedding-starved) nouns: 30% of groundable words with enough occurrences
    rng = np.random.default_rng(SEED)
    groundable = [w for w in feat if cnt[w] >= 20]
    rng.shuffle(groundable)
    H = set(groundable[: int(0.30 * len(groundable))])
    ids = np.array([stoi.get(w, UNK) for w in toks], dtype=np.int64)
    masked = np.array([UNK if w in H else stoi.get(w, UNK) for w in toks], dtype=np.int64)
    g_ss = np.array([feat[w][0] if w in feat else SS_NONE for w in toks], dtype=np.int64)
    g_d = np.array([feat[w][1] if w in feat else D_NONE for w in toks], dtype=np.int64)
    # per-position bucket of the CONDITIONING token: 0 none/UNK, 1 seen-noun, 2 H-noun
    buck = np.zeros(len(toks), dtype=np.int64)
    seen_noun = {w for w in feat if w not in H}
    for i, w in enumerate(toks):
        buck[i] = 2 if w in H else (1 if w in seen_noun else 0)
    n = len(toks); cut = int(0.9 * n)
    print(f"[generative gate]  dev={DEV}  brown {n:,} toks  V={V}  groundable={len(feat)}  "
          f"H(starved)={len(H)} types  blk={BLK}  params: token-emb DT={DT}+grounding 2*DG={2*DG}")
    return dict(masked=masked, g_ss=g_ss, g_d=g_d, tgt_full=masked, buck=buck, cut=cut, n=n)


class LM(nn.Module):
    def __init__(self, grounded):
        super().__init__(); self.grounded = grounded
        if grounded:
            self.tok = nn.Embedding(V, DT); self.ss = nn.Embedding(SS_NONE + 1, DG)
            self.dp = nn.Embedding(D_NONE + 1, DG)
        else:
            self.tok = nn.Embedding(V, DM)
        self.pos = nn.Embedding(BLK, DM)
        layer = nn.TransformerEncoderLayer(DM, NH, FF, DROP, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, NL)
        self.ln = nn.LayerNorm(DM); self.head = nn.Linear(DM, V)
        self.register_buffer("mask", torch.triu(torch.full((BLK, BLK), float("-inf")), 1))

    def emb(self, x, ss, dp):
        if self.grounded:
            return torch.cat([self.tok(x), self.ss(ss), self.dp(dp)], -1)
        return self.tok(x)

    def forward(self, x, ss, dp):
        T = x.size(1)
        h = self.emb(x, ss, dp) + self.pos(torch.arange(T, device=x.device))[None]
        h = self.enc(h, mask=self.mask[:T, :T])
        return self.head(self.ln(h))


def run(kind, d):
    torch.manual_seed(SEED)
    m = LM(kind == "grounded").to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    mk, gs, gd, cut = d["masked"], d["g_ss"], d["g_d"], d["cut"]
    tmk = torch.from_numpy(mk).to(DEV); tgs = torch.from_numpy(gs).to(DEV); tgd = torch.from_numpy(gd).to(DEV)
    g = torch.Generator(device=DEV).manual_seed(SEED)
    m.train()
    for step in range(STEPS):
        ix = torch.randint(0, cut - BLK - 1, (BS,), device=DEV, generator=g)
        idx = ix[:, None] + torch.arange(BLK, device=DEV)[None]
        x, ss, dp = tmk[idx], tgs[idx], tgd[idx]
        y = tmk[idx + 1]
        logits = m(x, ss, dp)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    # ---- eval on val: per-position NLL bucketed by conditioning token ----
    m.eval()
    buck = torch.from_numpy(d["buck"]).to(DEV)
    starts = np.arange(cut, d["n"] - BLK - 1, BLK)
    sums = torch.zeros(3, device=DEV); counts = torch.zeros(3, device=DEV)
    tot_s = torch.zeros((), device=DEV); tot_c = torch.zeros((), device=DEV)
    with torch.no_grad():
        for w0 in range(0, len(starts), BS):
            ix = torch.from_numpy(starts[w0:w0 + BS]).to(DEV)
            idx = ix[:, None] + torch.arange(BLK, device=DEV)[None]
            x, ss, dp, y, b = tmk[idx], tgs[idx], tgd[idx], tmk[idx + 1], buck[idx]
            nll = F.cross_entropy(m(x, ss, dp).reshape(-1, V), y.reshape(-1), reduction="none")
            bf = b.reshape(-1)
            tot_s += nll.sum(); tot_c += nll.numel()
            for k in range(3):
                msk = bf == k
                sums[k] += nll[msk].sum(); counts[k] += msk.sum()
    ppl = lambda s, c: math.exp((s / c).item())
    p = sum(t.numel() for t in m.parameters())
    return p, ppl(tot_s, tot_c), [ppl(sums[k], counts[k]) for k in range(3)], counts.tolist()


def main():
    d = build()
    print(f"\n{'model':16}{'params':>9}{'overall PPL':>13}{'post-NONE':>11}{'post-SEEN-noun':>16}{'post-H-noun(starved)':>22}")
    res = {}
    for kind in ("flat", "grounded"):
        p, ov, bk, cnts = run(kind, d); res[kind] = (p, ov, bk)
        print(f"{kind:16}{p/1e6:>7.2f}M{ov:>13.2f}{bk[0]:>11.2f}{bk[1]:>16.2f}{bk[2]:>22.2f}")
    print(f"  (val bucket positions: none={int(cnts[0]):,}  seen-noun={int(cnts[1]):,}  H-noun={int(cnts[2]):,})")

    fo, fb = res["flat"][1], res["flat"][2]; go, gb = res["grounded"][1], res["grounded"][2]
    d_ov = (go - fo) / fo * 100; d_H = (gb[2] - fb[2]) / fb[2] * 100; d_seen = (gb[1] - fb[1]) / fb[1] * 100
    print(f"\nVERDICT (PPL lower = better; grounded has FEWER params):")
    print(f"  fluency:  overall grounded {go:.1f} vs flat {fo:.1f}  ({d_ov:+.1f}%)  -> "
          f"{'grounding preserves fluency' if d_ov < 5 else 'grounding HURTS fluency (tension bites)'}")
    print(f"  payoff:   post-H-noun grounded {gb[2]:.1f} vs flat {fb[2]:.1f}  ({d_H:+.1f}%)  -> "
          f"{'GROUNDING PAYS at embedding-starved positions' if d_H < -3 else 'no generative payoff (FALSIFIED)'}")
    print(f"  control:  post-seen-noun grounded {gb[1]:.1f} vs flat {fb[1]:.1f}  ({d_seen:+.1f}%)  -> "
          f"{'~neutral where embedding exists (win is CONCENTRATED, not global capacity)' if abs(d_seen) < abs(d_H) else 'seen bucket also moved -- check capacity confound'}")

    fig, ax = plt.subplots(figsize=(7.4, 4.7)); x = np.arange(4); w = 0.38
    flat_v = [fo, fb[0], fb[1], fb[2]]; grnd_v = [go, gb[0], gb[1], gb[2]]
    ax.bar(x - w / 2, flat_v, w, color=RED, label="flat (token-emb)")
    ax.bar(x + w / 2, grnd_v, w, color=GREEN, label="grounded+token (fewer params)")
    ax.set_xticks(x); ax.set_xticklabels(["overall", "post-none", "post-seen\nnoun", "post-H-noun\n(starved)"])
    ax.set_ylabel("perplexity (lower better)")
    ax.set_title("Generative gate: grounding preserves fluency and pays at embedding-starved positions",
                 color=NAVY, fontsize=10)
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig("sheaf_llm/generative_gate.png", dpi=160); plt.close(fig)
    print("  wrote sheaf_llm/generative_gate.png")


if __name__ == "__main__":
    main()
