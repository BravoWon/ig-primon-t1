#!/usr/bin/env python
"""DeltaSheaf-v0.2 Phase-2 — ensemble scoring harness (one model at a time; sequential to fit 12 GB).

For each item: prompt the model for a BRIEF rationale + a final lettered choice (never a bare letter, per
SPEC §2). Record the chosen letter, correctness, and the full reply text (the stalk source).

RESUMABLE at the item level: writes each batch to <out> immediately (append + flush), and on restart skips
the items already present. So an external kill loses zero work; repeated relaunches converge. (Needed
because slow unauthenticated HF downloads can eat a background window and the job gets killed mid-model.)

    python score.py --model Qwen/Qwen2.5-3B-Instruct --n 12 --smoke
    python score.py --model <id> --n 2500 --out data/raw/<tag>.jsonl
"""
import argparse, json, os, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

LETTERS = ["A", "B", "C", "D"]
PROMPT = ("Answer the multiple-choice question. First give a brief one- or two-sentence reason, then on a "
          "new line write exactly 'Answer: X' where X is the letter of the correct option.\n\n"
          "Question: {q}\n{opts}\n")

def parse_letter(text):
    m = re.findall(r"Answer:\s*([ABCD])", text, flags=re.I)
    if m: return m[-1].upper()
    m = re.findall(r"\b([ABCD])\b", text)
    return m[-1].upper() if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--pool", default="deltasheaf-v02/data/mmlu_pool.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--big", action="store_true")   # device_map=auto (GPU+CPU offload) for 7B+ on 12GB
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    done = 0
    if a.out and os.path.exists(a.out):
        done = sum(1 for _ in open(a.out, encoding="utf-8"))
        if done >= a.n:
            print(f"  DONE-ALREADY {a.model} ({done}/{a.n})"); return
    if a.out: os.makedirs(os.path.dirname(a.out), exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if a.big:
        mdl = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map="auto").eval()
    else:
        mdl = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to(dev).eval()
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    # read MMLU items from the LOCAL pool (zero network); skip first `done` items for resume.
    # Same deterministic order the streaming used, so resume alignment with prior runs holds.
    with open(a.pool, encoding="utf-8") as f:
        pool = [json.loads(l) for l in f]
    items = pool[done:a.n]
    if done: print(f"  RESUME {a.model} from item {done}/{a.n} ({len(items)} to go)", flush=True)

    fout = open(a.out, "a", encoding="utf-8") if a.out else None
    n_ok = n_correct = 0; rows = []; t0 = time.time()
    for s in range(0, len(items), a.batch):
        chunk = items[s:s + a.batch]
        msgs = [[{"role": "user", "content": PROMPT.format(
            q=it["question"], opts="\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(it["choices"])))}]
            for it in chunk]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, padding=True).to(dev)
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        for i, it in enumerate(chunk):
            reply = tok.decode(out[i, plen:], skip_special_tokens=True).strip()
            letter = parse_letter(reply); gold = LETTERS[it["answer"]]; correct = (letter == gold)
            n_ok += letter is not None; n_correct += correct
            row = {"q": it["question"], "gold": gold, "letter": letter, "correct": correct, "reply": reply}
            rows.append(row)
            if fout: fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        if fout: fout.flush()                                   # checkpoint each batch — kill-safe
        if not a.smoke and (s // a.batch) % 10 == 0:
            print(f"    {done+s+len(chunk)}/{a.n}  ({time.time()-t0:.0f}s)", flush=True)
    if fout: fout.close()

    print(f"[{a.model}]  scored {len(items)} this run  parsed={n_ok}/{len(items)}  correct={n_correct}/{len(items)}"
          f"  (file now {done+len(items)}/{a.n})")
    if a.smoke:
        for r in rows[:3]:
            print(f"  gold={r['gold']} pred={r['letter']} ok={r['correct']}  reply: {r['reply'][:110].replace(chr(10),' ')}")

if __name__ == "__main__":
    main()
