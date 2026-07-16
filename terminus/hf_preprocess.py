# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.19",
#   "tokenizers>=0.15",
#   "nltk>=3.8",
#   "numpy>=1.26",
#   "huggingface_hub>=0.23",
# ]
# ///
"""Terminus Job 0 (HF Jobs) -- build the token + WordNet grounding streams for the PARAM-scale ladder,
run the MANDATORY alignment round-trip check, and push to a Hub dataset. Blocks all training jobs.

Ported verbatim from terminus/preprocess.py (same ground_word / align_text logic, proven locally on the
124M-token run). Additions for cloud: PEP-723 deps, streaming build to a target token count, a Hub push,
and an ALWAYS-ON alignment report (printed to the Job log AND saved into the dataset) so the round-trip
check is auditable BEFORE any GPU job consumes the stream -- the pre-reg's gating prerequisite.

The training jobs read a PREFIX of this one dataset per scale point (S0=first 200M, S1=first 1B, S2=first
4B tokens), and a fixed held-out tail as val -- exactly the local design (train.py used ids[:budget]).

    # cheap dry run: alignment check on a small sample, no build, no push (~free, cpu-basic)
    hf jobs uv run --flavor cpu-basic --secrets HF_TOKEN=$HF_TOKEN terminus/hf_preprocess.py -- --validate

    # real build + push (size to the ladder max; S2 needs ~4B train + val)
    hf jobs uv run --flavor cpu-xl --timeout 6h --secrets HF_TOKEN=$HF_TOKEN \
        terminus/hf_preprocess.py -- --tokens 4020000000 --repo rOGUEgRINGO/terminus-grounded-fineweb
"""
import argparse, json, os, string, sys
from functools import lru_cache
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

STOP = set("""the a an and or but if then else of in on at to for from by with as is are was were be been
being have has had do does did will would shall should can could may might must not no nor so than that this
these those it its it's he she they them him her his their theirs we you i me my mine your yours our ours us
who whom which what when where why how all any both each few more most other some such only own same too very
s t d ll re ve m o just don now here there off out up down over under again further once about into through
during before after above below between out against because while what who""".split())

from tokenizers import Tokenizer
import nltk
from nltk.corpus import wordnet as wn
try:
    wn.ensure_loaded()
except LookupError:
    nltk.download("wordnet", quiet=True); wn.ensure_loaded()

TOK = Tokenizer.from_pretrained("gpt2")
SS_LIST = sorted({s.lexname() for s in wn.all_synsets("n")})       # 26 noun.* supersenses
SS_ID = {name: i + 1 for i, name in enumerate(SS_LIST)}            # 0 = none
def depth_bucket(d): return 1 if d <= 3 else 2 if d <= 6 else 3 if d <= 9 else 4


@lru_cache(maxsize=None)
def ground_word(w):
    w2 = w.strip().lower().strip(string.punctuation)
    if len(w2) < 3 or not w2.isalpha() or w2 in STOP: return (0, 0)
    syns = wn.synsets(w2, pos="n")
    if not syns: return (0, 0)
    s = syns[0]
    return (SS_ID[s.lexname()], depth_bucket(s.min_depth()))


def clean_piece(p): return p.replace("Ġ", "").replace("Ċ", "")


def align_text(text):
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


def alignment_report(n_words=80):
    """The gating round-trip check: decode a fresh sample, show word<->grounding, return the text."""
    inv = {v: k for k, v in SS_ID.items()}
    text = next(stream_texts())[:2000]
    ids, ss, dp, words = align_text(text)
    lines = [f"[alignment round-trip check] supersenses={len(SS_LIST)} (0=none, 1..{len(SS_LIST)}), depth 1..4",
             f"  {'word':22}{'toks':>5}  {'supersense':22}{'depth':>6}"]
    shown = 0
    for word, st, ln, (s, d) in words:
        if shown >= n_words: break
        tag = inv.get(s, "-") if s else "-"
        safe = word[:22].encode("ascii", "replace").decode()
        lines.append(f"  {safe:22}{ln:>5}  {tag:22}{(d if d else '-'):>6}")
        shown += 1
    grounded = int((ss > 0).sum()); tot = len(ids)
    lines.append(f"\n  grounded tokens in sample: {grounded}/{tot} = {grounded/max(1,tot):.1%}")
    lines.append("  ROUND-TRIP: do the nouns carry sensible supersenses and non-nouns show '-'?  (eyeball before training)")
    report = "\n".join(lines)
    print(report, flush=True)
    return report


def generate(n_tokens):
    ids = np.empty(n_tokens, np.uint16); ss = np.empty(n_tokens, np.uint8); dp = np.empty(n_tokens, np.uint8)
    pos = 0; nextmark = 50_000_000
    for text in stream_texts():
        i, s, d, _ = align_text(text)
        take = min(len(i), n_tokens - pos)
        ids[pos:pos+take] = i[:take]; ss[pos:pos+take] = s[:take]; dp[pos:pos+take] = d[:take]
        pos += take
        if pos >= n_tokens: break
        if pos >= nextmark:
            print(f"  ... {pos/1e6:.0f}M / {n_tokens/1e6:.0f}M tokens", flush=True); nextmark += 50_000_000
    return ids[:pos], ss[:pos], dp[:pos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=0, help="total tokens to build (0 => --validate only)")
    ap.add_argument("--repo", type=str, default="rOGUEgRINGO/terminus-grounded-fineweb")
    ap.add_argument("--validate", action="store_true", help="alignment check only; no build, no push")
    a = ap.parse_args()

    # The gating check always runs first, and is cheap.
    report = alignment_report()
    if a.validate or a.tokens <= 0:
        print("\n[validate-only] alignment check done; not building or pushing.", flush=True)
        return

    print(f"\n[build] generating {a.tokens/1e9:.2f}B tokens ...", flush=True)
    ids, ss, dp, = generate(a.tokens)
    stats = {"tokens": int(len(ids)), "grounded_frac": float((ss > 0).mean()),
             "tok_vocab": int(ids.max()) + 1, "ss_vocab": int(ss.max()) + 1, "dp_vocab": int(dp.max()) + 1,
             "supersenses": len(SS_LIST), "source": "HuggingFaceFW/fineweb-edu sample-10BT",
             "grounding": "WordNet noun 1st-synset lexname->supersense, min_depth->bucket; STOP+OOV+<3char=none"}
    print(f"  built {stats['tokens']:,} tokens, {stats['grounded_frac']:.1%} grounded, "
          f"tok-vocab {stats['tok_vocab']}, ss-vocab {stats['ss_vocab']}, dp-vocab {stats['dp_vocab']}", flush=True)

    os.makedirs("out", exist_ok=True)
    np.save("out/ids.npy", ids); np.save("out/ss.npy", ss); np.save("out/dp.npy", dp)
    with open("out/stats.json", "w") as f: json.dump(stats, f, indent=2)
    with open("out/alignment_check.txt", "w", encoding="utf-8") as f: f.write(report)

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.repo, repo_type="dataset", exist_ok=True)
    print(f"[push] uploading ids/ss/dp + stats + alignment_check to dataset {a.repo} ...", flush=True)
    api.upload_folder(folder_path="out", repo_id=a.repo, repo_type="dataset")
    print(f"[done] dataset ready: https://huggingface.co/datasets/{a.repo}", flush=True)


if __name__ == "__main__":
    main()
