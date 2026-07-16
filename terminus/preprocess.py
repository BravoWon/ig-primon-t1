#!/usr/bin/env python
"""Terminus preprocessing -- build token stream + WordNet grounding streams aligned to GPT-2 BPE.

Load-bearing risk (flagged in PREREG_terminus_scale_gate.md): a WordNet noun spans several BPE subword
tokens; misaligning the grounding stream silently nulls the signal. This script builds parallel streams
  token_ids   (uint16)  -- GPT-2 BPE ids
  ss_ids      (uint8)   -- WordNet supersense id (0=none, 1..26)  per token
  dp_ids      (uint8)   -- hypernym-depth bucket (0=none, 1..4)   per token
and, in --validate mode, PRINTS the decoded word<->grounding alignment for a fresh-eyes round-trip check
BEFORE any training consumes it.

Grounding rule (noun-sense heuristic, matching brick 3'): a word is grounded iff it has a WordNet noun
synset; the first synset's lexname -> supersense, its min_depth -> bucket; assigned to ALL BPE tokens of
that word. Non-nouns / OOV / non-alpha -> (0,0).

    python preprocess.py                 # validate: process a small sample, print alignment
    python preprocess.py --tokens 100000000 --out data_100M   # generate a real dataset
"""
import sys, argparse, string
from functools import lru_cache
import numpy as np

STOP = set("""the a an and or but if then else of in on at to for from by with as is are was were be been
being have has had do does did will would shall should can could may might must not no nor so than that this
these those it its it's he she they them him her his their theirs we you i me my mine your yours our ours us
who whom which what when where why how all any both each few more most other some such only own same too very
s t d ll re ve m o just don now here there off out up down over under again further once about into through
during before after above below between out against because while what who```""".split())
from tokenizers import Tokenizer
import nltk
from nltk.corpus import wordnet as wn
try: wn.ensure_loaded()
except LookupError: nltk.download("wordnet", quiet=True); wn.ensure_loaded()

TOK = Tokenizer.from_pretrained("gpt2")
SS_LIST = sorted({s.lexname() for s in wn.all_synsets("n")})          # 26 noun.* supersenses
SS_ID = {name: i + 1 for i, name in enumerate(SS_LIST)}              # 0 = none
def depth_bucket(d): return 1 if d <= 3 else 2 if d <= 6 else 3 if d <= 9 else 4

@lru_cache(maxsize=None)
def ground_word(w):
    w2 = w.strip().lower().strip(string.punctuation)      # strip edge punctuation ("love," -> "love")
    if len(w2) < 3 or not w2.isalpha() or w2 in STOP: return (0, 0)   # stoplist kills function-word noun senses
    syns = wn.synsets(w2, pos="n")
    if not syns: return (0, 0)
    s = syns[0]
    return (SS_ID[s.lexname()], depth_bucket(s.min_depth()))

def clean_piece(p): return p.replace("Ġ", "").replace("Ċ", "")

def align_text(text):
    """Return (ids, ss, dp, words) where words is list of (word, span_start, span_len, (ss,dp)) for validation."""
    enc = TOK.encode(text)
    ids, pieces = enc.ids, enc.tokens
    n = len(ids)
    ss = np.zeros(n, np.uint8); dp = np.zeros(n, np.uint8)
    words = []
    start = 0
    for i in range(1, n + 1):
        boundary = (i == n) or pieces[i].startswith("Ġ") or pieces[i].startswith("Ċ")
        if boundary:
            word = "".join(clean_piece(p) for p in pieces[start:i])
            g = ground_word(word)
            if g != (0, 0):
                ss[start:i] = g[0]; dp[start:i] = g[1]
            words.append((word, start, i - start, g))
            start = i
    return np.array(ids, np.uint16), ss, dp, words

def stream_texts():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for ex in ds:
        yield ex["text"]

def validate(n_words=60):
    print(f"[validate] supersenses={len(SS_LIST)} (0=none, 1..{len(SS_LIST)})  depth buckets 1..4\n")
    text = next(stream_texts())[:1500]
    ids, ss, dp, words = align_text(text)
    inv = {v: k for k, v in SS_ID.items()}
    print(f"  {'word':22}{'toks':>5}  {'supersense':22}{'depth':>6}")
    shown = 0
    for word, st, ln, (s, d) in words:
        if shown >= n_words: break
        tag = inv.get(s, "-") if s else "-"
        safe = word[:22].encode("ascii", "replace").decode()          # cp1252-safe console print
        print(f"  {safe:22}{ln:>5}  {tag:22}{d if d else '-':>6}")
        shown += 1
    grounded = int((ss > 0).sum()); tot = len(ids)
    print(f"\n  grounded tokens: {grounded}/{tot} = {grounded/tot:.1%}  (rest = function words / punctuation / OOV)")
    print(f"  ROUND-TRIP CHECK: do the nouns above carry sensible supersenses and non-nouns show '-'? (eyeball)")

def generate(n_tokens, out):
    ids = np.empty(n_tokens, np.uint16); ss = np.empty(n_tokens, np.uint8); dp = np.empty(n_tokens, np.uint8)
    pos = 0; nextmark = 5_000_000
    for text in stream_texts():
        i, s, d, _ = align_text(text)
        take = min(len(i), n_tokens - pos)
        ids[pos:pos+take] = i[:take]; ss[pos:pos+take] = s[:take]; dp[pos:pos+take] = d[:take]
        pos += take
        if pos >= n_tokens: break
        if pos >= nextmark:
            print(f"  ... {pos/1e6:.0f}M / {n_tokens/1e6:.0f}M tokens", flush=True); nextmark += 5_000_000
    np.save(f"terminus/{out}_ids.npy", ids); np.save(f"terminus/{out}_ss.npy", ss); np.save(f"terminus/{out}_dp.npy", dp)
    g = int((ss > 0).sum())
    print(f"  wrote terminus/{out}_[ids|ss|dp].npy : {len(ids):,} tokens, {g/len(ids):.1%} grounded, "
          f"tok-vocab {int(ids.max())+1}, ss-vocab {int(ss.max())+1}, dp-vocab {int(dp.max())+1}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=0)
    ap.add_argument("--out", type=str, default="data")
    a = ap.parse_args()
    if a.tokens > 0: generate(a.tokens, a.out)
    else: validate()
