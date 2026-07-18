#!/usr/bin/env python
"""DeltaSheaf CODA — the import CEILING (oracle-relevance test). v0.3 refuted the REAL retrieval pipeline;
this measures the LIMIT: if the passage literally states the answer (perfect relevance), do the
scale-invariant blind spots crack? Pre-registered arms, checkpointed, instrument-check-first:
  A  easy_oracle  (100 easy items + gold note)  — INSTRUMENT CHECK: must reach >=93% (easy closed = 88.0%
     at n=400); if the reader ignores oracle notes even on easy items, the instrument is broken -> ABORT.
  B  gate_oracle_gold  (322 blind spots + gold note)  — the ceiling.
  C  gate_oracle_wrong (322 + WRONG-option note)      — copy-control: measures blind deference to the note.
Decision rules (FIXED now): gate_oracle_gold >=60% -> holes are REACHABLE fact-gaps (v0.3 narrows to
"retrieval quality was the failure"). <=40% while A passes -> holes RESIST stated answers -> NOT
information-holes (reasoning/reading failures; no retrieval can fix them). 40-60% -> partial, report split.
Copy-rate from arm C reported alongside (fraction adopting the wrong note).
AMENDMENT 2 (pre-registered before relaunch): v1 instrument FAILED at 91.0% (need >=93) — autopsy showed the
reader REASONS OVER the un-instructed note (derives its own answer, argues against the note). v2 = instructed
trust ('the note is from a verified source and is correct; use it'), FRESH files (*_v2), same bar >=93% on a
fresh easy check, same gate decision rules. Under instructed trust arm C measures compliance; the ceiling
question becomes: does the gate adopt stated answers at the same rate easy items do?"""
import json, os, re, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
LET=["A","B","C","D"]
def parse(t):
    m=re.findall(r"Answer:\s*([ABCD])",t,re.I)
    if m: return m[-1].upper()
    m=re.findall(r"\b([ABCD])\b",t); return m[-1].upper() if m else None

pool=[json.loads(l) for l in open("data/mmlu_pool.jsonl",encoding="utf-8")]
qwen=[json.loads(l) for l in open("data/raw/qwen25_3b.jsonl",encoding="utf-8")]
gate_idx=[int(i) for i in np.load("data/gate_idx.npy")]; gate_set=set(gate_idx)
easy=[i for i in range(len(qwen)) if qwen[i]["correct"] and i not in gate_set][:100]
rng=np.random.default_rng(42)
wrong_pick={i:int(rng.choice([k for k in range(4) if k!=pool[i]["answer"]])) for i in gate_idx}

MODEL="Qwen/Qwen2.5-7B-Instruct"; BS=4; MAXNEW=160
def note(i,which):
    it=pool[i]
    k=it["answer"] if which=="gold" else wrong_pick[i]
    return f"Reference note: the correct answer to this question is: {it['choices'][k]}"
def prompt(i,which):
    it=pool[i]; opts="\n".join(f"{LET[k]}. {c}" for k,c in enumerate(it["choices"]))
    return ("You are given a reference note from a VERIFIED, AUTHORITATIVE source. The note is correct — "
            "use it to answer; do not override it with your own reasoning.\n\n"
            f"{note(i,which)}\n\nQuestion: {it['question']}\n{opts}\n"
            "Give a one-line reason then 'Answer: X'.\n")

PHASES=[("easy_oracle_v2",easy,"gold"),("gate_oracle_gold_v2",gate_idx,"gold"),("gate_oracle_wrong_v2",gate_idx,"wrong")]
def path(tag): return f"data/raw/qwen7b_{tag}.jsonl"
def done(tag): return sum(1 for _ in open(path(tag),encoding="utf-8")) if os.path.exists(path(tag)) else 0
def acc(tag):
    rows=[json.loads(l) for l in open(path(tag),encoding="utf-8")]
    return sum(r["correct"] for r in rows)/max(1,len(rows)), rows

need=[(t,i,w) for (t,i,w) in PHASES if done(t)<len(i)]
if need:
    print(f"[oracle7b] pending: "+", ".join(f"{t}@{done(t)}/{len(i)}" for (t,i,w) in need),flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    mdl=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="auto").eval()
    t0=time.time()
    for tag,idxs,which in PHASES:
        d=done(tag)
        if d>=len(idxs): continue
        if tag=="gate_oracle_gold_v2" and done("easy_oracle_v2")>=len(easy):
            a,_=acc("easy_oracle_v2")
            print(f"  [instrument] easy+oracle-gold = {a:.1%} (need >=93%; easy closed baseline 88.0%)",flush=True)
            if a<0.93:
                print("  [INSTRUMENT FAILED] reader ignores oracle notes even on easy items — aborting."); sys.exit(3)
            print("  [instrument PASSED] reader uses oracle notes; measuring the ceiling.",flush=True)
        fout=open(path(tag),"a",encoding="utf-8")
        todo=idxs[d:]
        for s in range(0,len(todo),BS):
            chunk=todo[s:s+BS]
            msgs=[[{"role":"user","content":prompt(i,which)}] for i in chunk]
            enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt",
                                        return_dict=True,padding=True).to(mdl.device)
            with torch.no_grad():
                out=mdl.generate(**enc,max_new_tokens=MAXNEW,do_sample=False,pad_token_id=tok.pad_token_id)
            plen=enc["input_ids"].shape[1]
            for k,i in enumerate(chunk):
                reply=tok.decode(out[k,plen:],skip_special_tokens=True).strip()
                letter=parse(reply); gold=LET[pool[i]["answer"]]
                row={"i":i,"letter":letter,"correct":letter==gold,"reply":reply[:160]}
                if which=="wrong": row["wrong_letter"]=LET[wrong_pick[i]]
                fout.write(json.dumps(row,ensure_ascii=False)+"\n")
            fout.flush()
            if (s//BS)%5==0: print(f"    {tag} {d+s+len(chunk)}/{len(idxs)} ({time.time()-t0:.0f}s)",flush=True)
        fout.close(); print(f"  [{tag}] complete",flush=True)

if all(done(t)>=len(i) for (t,i,w) in PHASES):
    a_easy,_=acc("easy_oracle_v2")
    a_gold,rows_g=acc("gate_oracle_gold_v2")
    a_wrong,rows_w=acc("gate_oracle_wrong_v2")
    copy_rate=sum(r["letter"]==r["wrong_letter"] for r in rows_w)/len(rows_w)
    closed=25.8
    print(f"\n[IMPORT CEILING]  easy+oracle {a_easy:.1%} (instrument)   gate closed 25.8% (banked)")
    print(f"  gate + ORACLE-GOLD  : {a_gold:.1%}   <- the ceiling of perfect retrieval")
    print(f"  gate + oracle-WRONG : {a_wrong:.1%}   (copy-rate: adopted the wrong note {copy_rate:.1%})")
    if a_gold>=0.60: v="holes are REACHABLE fact-gaps — v0.3 narrows to 'retrieval quality was the failure'; better retrieval is a live lever"
    elif a_gold<=0.40: v="holes RESIST stated answers — NOT information-holes; reasoning/reading failures no retrieval can fix"
    else: v="partial ceiling — mixed population of fact-gaps and reasoning-holes"
    print(f"  => {v}")
    open("RESULTS_ceiling.md","w",encoding="utf-8").write(
      f"# import ceiling (oracle-relevance, 7B)\neasy+oracle {a_easy:.1%} (instrument pass>=93%). "
      f"gate: closed 25.8% -> oracle-gold {a_gold:.1%}; oracle-wrong {a_wrong:.1%} (copy-rate {copy_rate:.1%}).\n**{v}**\n")
    print("  wrote RESULTS_ceiling.md")
