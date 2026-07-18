#!/usr/bin/env python
"""DeltaSheaf CODA-2 — FACT-FORM import (the rung where import and copying come apart), v2 instrument.
The oracle ceiling (89.8%) used answer-STATING notes (copy-rate 90.4% -> mostly mechanical). Here the note
is an encyclopedia-style FACT sentence (3B-generated from the gold choice; leak-filtered: no answer/letters),
read by the 7B under a neutral reliable-source prompt.

INSTRUMENT (corrected — the v1 easy>=93% bar was saturation-invalid: fact notes can't lift already-known easy
items). The integration check is the DOWNWARD, unsaturated direction: give WRONG facts on easy items the model
knows. Integration is real iff wrong facts pull easy accuracy DOWN vs gold facts. Proceed to the gate only if
gap = acc(easy_fact_gold) - acc(easy_fact_wrong) >= 0.10; else the reader ignores fact-form context under a
neutral prompt and the rung is unmeasurable with this reader (bank that).
Gate decision (FIXED): gate_fact_gold >=60% -> genuine fact-import works. <=40% (integration confirmed) ->
holes need answer-stating content, fact-form insufficient. 40-60% -> partial. fact-wrong copy-rate reported."""
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
wrong_pick={i:int(rng.choice([k for k in range(4) if k!=pool[i]["answer"]])) for i in list(easy)+gate_idx}

NOTES_PATH="data/fact_notes.jsonl"
def leak_ok(s):
    if re.search(r"\b(answer|correct|option|choice)\b",s,re.I): return False
    if re.search(r"\b[ABCD]\)\s|^[ABCD][.)]",s): return False
    return len(s.strip())>10
def gen_notes():
    need=[("easy",i,pool[i]["answer"]) for i in easy]+[("easy_wrong",i,wrong_pick[i]) for i in easy]+ \
         [("gold",i,pool[i]["answer"]) for i in gate_idx]+[("wrong",i,wrong_pick[i]) for i in gate_idx]
    done={}
    if os.path.exists(NOTES_PATH):
        for l in open(NOTES_PATH,encoding="utf-8"):
            r=json.loads(l); done[(r["kind"],r["i"])]=r
    todo=[(k,i,c) for (k,i,c) in need if (k,i) not in done]
    if todo:
        print(f"[notes] generating {len(todo)} fact sentences with Qwen-3B",flush=True)
        mid="Qwen/Qwen2.5-3B-Instruct"
        tok=AutoTokenizer.from_pretrained(mid); tok.padding_side="left"
        if tok.pad_token_id is None: tok.pad_token=tok.eos_token
        mdl=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.float16).to("cuda").eval()
        fout=open(NOTES_PATH,"a",encoding="utf-8"); BS=12; t0=time.time()
        for s in range(0,len(todo),BS):
            chunk=todo[s:s+BS]
            msgs=[[{"role":"user","content":
                ("Rewrite the information below as ONE standalone encyclopedia-style factual sentence. "
                 "Do not mention any question, options, letters, or the words 'answer' or 'correct'. "
                 "Just state the fact plainly.\n\n"
                 f"Topic: {pool[i]['question']}\nInformation to state as fact: {pool[i]['choices'][c]}")}]
                for (k,i,c) in chunk]
            enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt",
                                        return_dict=True,padding=True).to("cuda")
            with torch.no_grad(): out=mdl.generate(**enc,max_new_tokens=80,do_sample=False,pad_token_id=tok.pad_token_id)
            plen=enc["input_ids"].shape[1]
            for j,(k,i,c) in enumerate(chunk):
                note=tok.decode(out[j,plen:],skip_special_tokens=True).strip().split("\n")[0]
                if not leak_ok(note): note=f"{pool[i]['choices'][c]}."
                fout.write(json.dumps({"kind":k,"i":i,"note":note},ensure_ascii=False)+"\n")
            fout.flush()
            if (s//BS)%10==0: print(f"    notes {s+len(chunk)}/{len(todo)} ({time.time()-t0:.0f}s)",flush=True)
        fout.close(); del mdl; torch.cuda.empty_cache()
        done={}
        for l in open(NOTES_PATH,encoding="utf-8"):
            r=json.loads(l); done[(r["kind"],r["i"])]=r
    return done

notes=gen_notes()
def prompt(i,kind):
    it=pool[i]; opts="\n".join(f"{LET[k]}. {c}" for k,c in enumerate(it["choices"]))
    nt=notes[(kind,i)]["note"]
    return ("A reference excerpt from a reliable encyclopedia is provided. Consider it carefully when "
            "answering.\n\n"
            f"Reference excerpt: {nt}\n\nQuestion: {it['question']}\n{opts}\n"
            "Give a one-line reason then 'Answer: X'.\n")

PHASES=[("easy_fact",easy,"easy"),("easy_fact_wrong",easy,"easy_wrong"),
        ("gate_fact_gold",gate_idx,"gold"),("gate_fact_wrong",gate_idx,"wrong")]
def path(tag): return f"data/raw/qwen7b_{tag}.jsonl"
def done_n(tag): return sum(1 for _ in open(path(tag),encoding="utf-8")) if os.path.exists(path(tag)) else 0
def acc(tag):
    rows=[json.loads(l) for l in open(path(tag),encoding="utf-8")]
    return sum(r["correct"] for r in rows)/max(1,len(rows)), rows

need=[(t,i,k) for (t,i,k) in PHASES if done_n(t)<len(i)]
if need:
    print(f"[factform7b v2] pending: "+", ".join(f"{t}@{done_n(t)}/{len(i)}" for (t,i,k) in need),flush=True)
    MODEL="Qwen/Qwen2.5-7B-Instruct"; BS=4
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    mdl=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="auto").eval()
    t0=time.time()
    for tag,idxs,kind in PHASES:
        d=done_n(tag)
        if d>=len(idxs): continue
        if tag=="gate_fact_gold" and done_n("easy_fact")>=len(easy) and done_n("easy_fact_wrong")>=len(easy):
            ag,_=acc("easy_fact"); aw,_=acc("easy_fact_wrong"); gap=ag-aw
            print(f"  [instrument] easy fact-gold {ag:.1%} - fact-wrong {aw:.1%} = integration gap {gap:+.1%} (need >=+10%)",flush=True)
            if gap<0.10:
                print("  [INSTRUMENT FAILED] reader does not integrate fact-form context under neutral prompt "
                      "-> fact-form rung UNMEASURABLE with this reader. (Not a statement about the holes.)"); sys.exit(3)
            print("  [instrument PASSED] reader integrates fact-form notes; measuring the fact-import ceiling.",flush=True)
        fout=open(path(tag),"a",encoding="utf-8")
        todo=idxs[d:]
        for s in range(0,len(todo),BS):
            chunk=todo[s:s+BS]
            msgs=[[{"role":"user","content":prompt(i,kind)}] for i in chunk]
            enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt",
                                        return_dict=True,padding=True).to(mdl.device)
            with torch.no_grad(): out=mdl.generate(**enc,max_new_tokens=160,do_sample=False,pad_token_id=tok.pad_token_id)
            plen=enc["input_ids"].shape[1]
            for k2,i in enumerate(chunk):
                reply=tok.decode(out[k2,plen:],skip_special_tokens=True).strip()
                letter=parse(reply); gold=LET[pool[i]["answer"]]
                row={"i":i,"letter":letter,"correct":letter==gold,"reply":reply[:160]}
                if kind in ("wrong","easy_wrong"): row["wrong_letter"]=LET[wrong_pick[i]]
                fout.write(json.dumps(row,ensure_ascii=False)+"\n")
            fout.flush()
            if (s//BS)%5==0: print(f"    {tag} {d+s+len(chunk)}/{len(idxs)} ({time.time()-t0:.0f}s)",flush=True)
        fout.close(); print(f"  [{tag}] complete",flush=True)

if all(done_n(t)>=len(i) for (t,i,k) in PHASES):
    ag,_=acc("easy_fact"); aw,rows_ew=acc("easy_fact_wrong"); gap=ag-aw
    e_copy=sum(r["letter"]==r["wrong_letter"] for r in rows_ew)/len(rows_ew)
    a_gold,_=acc("gate_fact_gold"); a_wrong,rows_w=acc("gate_fact_wrong")
    g_copy=sum(r["letter"]==r["wrong_letter"] for r in rows_w)/len(rows_w)
    print(f"\n[FACT-FORM IMPORT]  integration gap (easy gold {ag:.1%} - wrong {aw:.1%}) = {gap:+.1%}  "
          f"(easy wrong-fact copy {e_copy:.1%})")
    print(f"  gate closed 25.8% (banked)   answer-form ceiling 89.8% (copy 90.4%)")
    print(f"  gate + FACT-GOLD  : {a_gold:.1%}")
    print(f"  gate + fact-WRONG : {a_wrong:.1%}   (adopted wrong fact {g_copy:.1%})")
    if a_gold>=0.60: v="GENUINE fact-import works — retrieval with good CONTENT recovers the holes (89.8% was not just copying)"
    elif a_gold<=0.40: v="fact-form INSUFFICIENT where answer-form worked — recovery needs answer-stating content; the retrieval bar is high"
    else: v="partial — mixed population of importable fact-gaps and answer-only holes"
    print(f"  => {v}")
    open("RESULTS_factform.md","w",encoding="utf-8").write(
      f"# fact-form import (CODA-2, 7B, corrected instrument)\nintegration gap easy {ag:.1%}-{aw:.1%}={gap:+.1%} "
      f"(easy copy {e_copy:.1%}). gate: closed 25.8% -> fact-gold {a_gold:.1%} (answer-form 89.8%); "
      f"fact-wrong {a_wrong:.1%} (copy {g_copy:.1%} vs 90.4% answer-form).\n**{v}**\n")
    print("  wrote RESULTS_factform.md")
