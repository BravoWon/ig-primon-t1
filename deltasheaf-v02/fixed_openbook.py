#!/usr/bin/env python
"""DeltaSheaf-v0.3 — fixed reader re-test. Recurse on the failing pair (the degrading open-book reader).
Fix = 'use context ONLY if it directly answers; else ignore' prompt. Re-verify on EASY (did the fix remove
the -7.3% degradation?) and re-test GATE (does a non-degrading reader recover?), split by passage RELEVANCE
(passage contains the gold answer text)."""
import json, os, sys, re
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
LET=["A","B","C","D"]
def parse(t): m=re.findall(r"Answer:\s*([ABCD])",t,re.I); return m[-1].upper() if m else (re.findall(r"\b([ABCD])\b",t) or [None])[-1]

pool=[json.loads(l) for l in open("data/mmlu_pool.jsonl",encoding="utf-8")]
qwen=[json.loads(l) for l in open("data/raw/qwen25_3b.jsonl",encoding="utf-8")]
ret=[json.loads(l)["passage"] for l in open("data/retrieval.jsonl",encoding="utf-8")]
gate_idx=[int(i) for i in np.load("data/gate_idx.npy")]; gate_set=set(gate_idx)
easy=[i for i in range(min(len(qwen),len(ret))) if qwen[i]["correct"] and i not in gate_set and len(ret[i])>20][:150]

dev="cuda"
BIG = "7b" in sys.argv
MODEL = "Qwen/Qwen2.5-7B-Instruct" if BIG else "Qwen/Qwen2.5-3B-Instruct"
print(f"[reader] {MODEL}")
tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
if BIG:
    mdl=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="auto").eval()
else:
    mdl=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).to(dev).eval()
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
FIX=("You get a multiple-choice question and, optionally, a reference passage. Use the passage ONLY if it "
     "directly contains the answer; if it is not clearly relevant, IGNORE it and answer from your own knowledge.")
def prompt(i,openbook):
    it=pool[i]; opts="\n".join(f"{LET[k]}. {c}" for k,c in enumerate(it["choices"]))
    ctx=f"Passage: {ret[i][:1000]}\n\n" if openbook else ""
    return f"{FIX}\n\n{ctx}Question: {it['question']}\n{opts}\nGive a one-line reason then 'Answer: X'.\n"
def run(idxs,openbook,bs=(4 if BIG else 12)):
    preds=[]
    for s in range(0,len(idxs),bs):
        chunk=idxs[s:s+bs]; msgs=[[{"role":"user","content":prompt(i,openbook)}] for i in chunk]
        enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt",return_dict=True,padding=True).to(dev)
        with torch.no_grad(): out=mdl.generate(**enc,max_new_tokens=160,do_sample=False,pad_token_id=tok.pad_token_id)
        for k in range(out.shape[0]): preds.append(parse(tok.decode(out[k,enc["input_ids"].shape[1]:],skip_special_tokens=True)))
    return preds
def acc(idxs,preds): g=[LET[pool[i]["answer"]] for i in idxs]; return sum(preds[k]==g[k] for k in range(len(idxs)))/max(1,len(idxs))
def relevant(i):     # passage contains the gold option's text (rough relevance proxy)
    gtxt=pool[i]["choices"][pool[i]["answer"]].lower().strip()
    return len(gtxt)>=3 and gtxt in ret[i].lower()

# EASY: did the fix remove the degradation?
e_cb=run(easy,False); e_ob=run(easy,True)
ea_cb,ea_ob=acc(easy,e_cb),acc(easy,e_ob)
print(f"[fixed reader]  EASY: closed {ea_cb:.1%} -> open {ea_ob:.1%}  Δ={ea_ob-ea_cb:+.1%}  ({'FIX WORKS (no degradation)' if ea_ob-ea_cb>-0.03 else 'STILL DEGRADES -> escalate to 7B reader'})")
# GATE: does a non-degrading reader recover? split by relevance
g_ob=run(gate_idx,True); g_cb=run(gate_idx,False)
rel=[k for k,i in enumerate(gate_idx) if relevant(i)]; nrel=[k for k in range(len(gate_idx)) if k not in set(rel)]
def sub(preds,ks): g=[LET[pool[gate_idx[k]]["answer"]] for k in ks]; return sum(preds[k]==g[j] for j,k in enumerate(ks))/max(1,len(ks))
print(f"  GATE all: closed {sub(g_cb,list(range(len(gate_idx)))):.1%} -> open {sub(g_ob,list(range(len(gate_idx)))):.1%}  (chance 25%)")
print(f"  GATE passage-RELEVANT ({len(rel)}): closed {sub(g_cb,rel):.1%} -> open {sub(g_ob,rel):.1%}   <-- does relevant retrieval recover?")
print(f"  GATE not-relevant ({len(nrel)}):     closed {sub(g_cb,nrel):.1%} -> open {sub(g_ob,nrel):.1%}")
law = ea_ob-ea_cb>-0.03 and len(rel)>=15 and sub(g_ob,rel)-sub(g_cb,rel)>0.10 and sub(g_ob,rel)>0.35
print(f"\n  => LAW {'CONFIRMED (fixed reader; relevant retrieval recovers gold on blind spots)' if law else 'still not supported / need stronger reader — see rows'}")
open("RESULTS_fixed.md","w",encoding="utf-8").write(
  f"# v0.3 fixed reader\nEASY {ea_cb:.1%}->{ea_ob:.1%} (fix {'OK' if ea_ob-ea_cb>-0.03 else 'FAILS'}).\n"
  f"GATE relevant({len(rel)}) {sub(g_cb,rel):.1%}->{sub(g_ob,rel):.1%}; not-rel {sub(g_cb,nrel):.1%}->{sub(g_ob,nrel):.1%}.\n"
  f"**LAW {'CONFIRMED' if law else 'undecided/negative'}.**\n")
print("  wrote RESULTS_fixed.md")
