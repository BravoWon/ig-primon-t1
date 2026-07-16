#!/usr/bin/env python
"""Terminus training -- flat vs grounded vs grounded-random causal LM, local (RTX 5070).

The terminus question: does grounding's PPL edge (concentrated at rare nouns) SURVIVE as training tokens
grow (rare-word embeddings get better-learned)?  One architecture, swept over token budgets; the trend
Delta(budget) is the result.

Arms (identical transformer; input embedding differs; matched d_model so grounded is slightly LEANER):
  flat            : tok_emb(d_model)
  grounded        : concat(tok_emb(d_model-32), ss_emb(24), dp_emb(8))   -- inherited WordNet dims
  grounded-random : same shape, but each token's (ss,dp) reassigned to a FIXED RANDOM one (placebo:
                    same density/capacity, semantics destroyed)

Metric: val PPL overall + bucketed by TARGET-token frequency (rare/mid/common), restricted to grounded-noun
targets for the payoff bucket. Grounding should help most on RARE grounded targets.

    python train.py --arm grounded --budget 30000000 --data data124M --tag s1
"""
import argparse, json, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SS_VOCAB, DP_VOCAB = 27, 5
D_SS, D_DP = 24, 8

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(D, dim=2)
        q = q.view(B, T, s.h, D // s.h).transpose(1, 2); k = k.view(B, T, s.h, D // s.h).transpose(1, 2)
        v = v.view(B, T, s.h, D // s.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + s.proj(a); x = x + s.mlp(s.ln2(x)); return x

class GPT(nn.Module):
    def __init__(s, vocab, d, nlayer, nhead, block, grounded=False):
        super().__init__(); s.grounded = grounded; s.block = block
        if grounded:
            s.tok = nn.Embedding(vocab, d - D_SS - D_DP)
            s.ss = nn.Embedding(SS_VOCAB, D_SS); s.dp = nn.Embedding(DP_VOCAB, D_DP)
        else:
            s.tok = nn.Embedding(vocab, d)
        s.pos = nn.Embedding(block, d)
        s.blocks = nn.ModuleList([Block(d, nhead) for _ in range(nlayer)])
        s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, vocab, bias=False)
    def forward(s, idx, ssq=None, dpq=None):
        T = idx.shape[1]; pos = torch.arange(T, device=idx.device)
        e = torch.cat([s.tok(idx), s.ss(ssq), s.dp(dpq)], -1) if s.grounded else s.tok(idx)
        x = e + s.pos(pos)
        for b in s.blocks: x = b(x)
        return s.head(s.lnf(x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["flat", "grounded", "grounded-random"], required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--data", default="data124M"); ap.add_argument("--tag", default="")
    ap.add_argument("--d", type=int, default=320); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--val_tokens", type=int, default=4_000_000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    ids = np.load(f"terminus/{a.data}_ids.npy"); ss = np.load(f"terminus/{a.data}_ss.npy"); dp = np.load(f"terminus/{a.data}_dp.npy")
    N = len(ids); vsz = int(ids.max()) + 1
    val_lo = N - a.val_tokens
    assert a.budget <= val_lo, f"budget {a.budget} overlaps val (train must be < {val_lo})"

    # grounded-random: fixed random (ss,dp) per token-id, applied only where truly grounded
    if a.arm == "grounded-random":
        rng = np.random.default_rng(1234)
        rss = rng.integers(1, SS_VOCAB, size=vsz).astype(np.uint8); rdp = rng.integers(1, DP_VOCAB, size=vsz).astype(np.uint8)
        ss = np.where(ss > 0, rss[ids], 0).astype(np.uint8); dp = np.where(dp > 0, rdp[ids], 0).astype(np.uint8)

    ids_t = torch.from_numpy(ids.astype(np.int64)); ss_t = torch.from_numpy(ss.astype(np.int64)); dp_t = torch.from_numpy(dp.astype(np.int64))
    # target-token frequency (over the train prefix) for rare/mid/common buckets
    freq = np.bincount(ids[:a.budget], minlength=vsz)
    grounded_tok = np.zeros(vsz, bool)
    gmask = ss[:a.budget] > 0
    grounded_tok[np.unique(ids[:a.budget][gmask])] = True

    grounded_model = a.arm in ("grounded", "grounded-random")
    m = GPT(vsz, a.d, a.layers, a.heads, a.block, grounded=grounded_model).to(DEV)
    nparams = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps = a.budget // (a.bs * a.block)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

    def batch(lo, hi):
        ix = torch.randint(lo, hi - a.block - 1, (a.bs,))
        x = torch.stack([ids_t[i:i+a.block] for i in ix]).to(DEV)
        y = torch.stack([ids_t[i+1:i+1+a.block] for i in ix]).to(DEV)
        sx = torch.stack([ss_t[i:i+a.block] for i in ix]).to(DEV)
        dx = torch.stack([dp_t[i:i+a.block] for i in ix]).to(DEV)
        return x, y, sx, dx

    m.train(); t0 = time.time()
    for step in range(steps):
        x, y, sx, dx = batch(0, a.budget)
        logits = m(x, sx, dx) if grounded_model else m(x)
        loss = F.cross_entropy(logits.reshape(-1, vsz), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"    [{a.arm} b{a.budget//10**6}M step {step+1}/{steps}] loss {loss.item():.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- eval: mean-NLL + PPL bucketed by TARGET-token train frequency (grounded-noun targets) ----
    # payoff bucket = LEARNABLE-but-rare nouns (freq 3..50); hapaxes (<=2) are hopeless for everyone -> separate.
    m.eval()
    nll_sum = {"all": 0.0, "g_hapax": 0.0, "g_rare": 0.0, "g_common": 0.0, "nongrounded": 0.0}
    cnt = {k: 0 for k in nll_sum}
    with torch.no_grad():
        pos = val_lo
        while pos + a.block + 1 < N:
            xb = ids_t[pos:pos+a.block].unsqueeze(0).to(DEV)
            yb = ids_t[pos+1:pos+1+a.block].to(DEV)
            sb = ss_t[pos:pos+a.block].unsqueeze(0).to(DEV); db = dp_t[pos:pos+a.block].unsqueeze(0).to(DEV)
            logits = m(xb, sb, db) if grounded_model else m(xb)
            nll = F.cross_entropy(logits[0], yb, reduction="none").cpu().numpy()   # (block,)
            yt = ids[pos+1:pos+1+a.block]; y_ss = ss[pos+1:pos+1+a.block]; yf = freq[yt]
            for i in range(a.block):
                nll_sum["all"] += nll[i]; cnt["all"] += 1
                if y_ss[i] > 0 and grounded_tok[yt[i]]:
                    b = "g_hapax" if yf[i] <= 2 else ("g_rare" if yf[i] <= 50 else "g_common")
                    nll_sum[b] += nll[i]; cnt[b] += 1
                else:
                    nll_sum["nongrounded"] += nll[i]; cnt["nongrounded"] += 1
            pos += a.block
    nll_mean = {k: float(nll_sum[k] / max(1, cnt[k])) for k in nll_sum}   # float() -> valid JSON (was numpy float32)
    ppl = {k: float(math.exp(min(nll_mean[k], 20.0))) for k in nll_sum}
    out = {"arm": a.arm, "budget": a.budget, "tag": a.tag, "seed": int(a.seed), "nparams": int(nparams), "steps": int(steps),
           "ppl": ppl, "nll": nll_mean, "counts": {k: int(v) for k, v in cnt.items()}, "vocab": int(vsz)}
    print(f"  RESULT {a.arm} b{a.budget//10**6}M params={nparams/1e6:.1f}M  "
          f"PPL all={ppl['all']:.1f}  g_rare={ppl['g_rare']:.1f} (n={cnt['g_rare']})  "
          f"g_common={ppl['g_common']:.1f}  nll_rare={nll_mean['g_rare']:.3f}", flush=True)
    tagn = f"{a.tag}_" if a.tag else ""
    json.dump(out, open(f"terminus/metrics_{tagn}{a.arm}_{a.budget//10**6}M_s{a.seed}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
