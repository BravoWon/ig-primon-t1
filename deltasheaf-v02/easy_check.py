#!/usr/bin/env python
"""DeltaSheaf-v0.3 driftwave round-trip: is the open-book 'hurt' REAL (retrieval can't fill reasoning-holes)
or an ARTIFACT (open-book prompt degrades the 3B generally)? Decisive control: run open-book vs closed-book
on EASY items (Qwen-3B correct closed-book, passage present). If the passage hurts there too -> artifact,
v0.3 null is void. If neutral/helpful on easy but hurts only on the hard gate items -> the finding is real."""
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
gate=set(int(i) for i in np.load("data/gate_idx.npy"))
# EASY = Qwen-3B correct closed-book, NOT a gate item, passage present
easy=[i for i in range(min(len(qwen),len(ret))) if qwen[i]["correct"] and i not in gate and len(ret[i])>20][:150]
print(f"[easy-check] {len(easy)} easy items (Qwen-3B correct closed-book, passage present)")

dev="cuda"; MODEL="Qwen/Qwen2.5-3B-Instruct"
tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
mdl=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).to(dev).eval()
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
def prompt(i,openbook):
    it=pool[i]; opts="\n".join(f"{LET[k]}. {c}" for k,c in enumerate(it["choices"]))
    ctx=f"Context (may help):\n{ret[i][:1000]}\n\n" if openbook else ""
    return f"{ctx}Answer the multiple-choice question. Give a one-line reason then 'Answer: X'.\n\nQuestion: {it['question']}\n{opts}\n"
def run(openbook,bs=12):
    preds=[]
    for s in range(0,len(easy),bs):
        chunk=easy[s:s+bs]
        msgs=[[{"role":"user","content":prompt(i,openbook)}] for i in chunk]
        enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt",return_dict=True,padding=True).to(dev)
        with torch.no_grad(): out=mdl.generate(**enc,max_new_tokens=160,do_sample=False,pad_token_id=tok.pad_token_id)
        for k in range(out.shape[0]): preds.append(parse(tok.decode(out[k,enc["input_ids"].shape[1]:],skip_special_tokens=True)))
    return preds
gold=[LET[pool[i]["answer"]] for i in easy]
cb=run(False); ob=run(True)
cb_a=sum(cb[k]==gold[k] for k in range(len(easy)))/len(easy)
ob_a=sum(ob[k]==gold[k] for k in range(len(easy)))/len(easy)
print(f"\n  EASY items (chance 0.25):  closed-book {cb_a:.1%}  ->  open-book {ob_a:.1%}   Δ = {ob_a-cb_a:+.1%}")
print(f"  HARD gate items (from openbook.py):  closed 17.9% -> open 12.2%   Δ = -5.7%")
artifact = (cb_a-ob_a) > 0.07     # passage hurts easy items too, by a comparable margin
print(f"\n  => open-book 'hurt' is {'AN ARTIFACT (passage degrades even EASY items -> v0.3 null is VOID, needs a fixed reader)' if artifact else 'REAL (passage does NOT hurt easy items; the hurt is specific to reasoning-hard gate items)'}")
open("RESULTS_easycheck.md","w",encoding="utf-8").write(
  f"# v0.3 instrument round-trip\nEASY: closed {cb_a:.1%} -> open {ob_a:.1%} (Δ{ob_a-cb_a:+.1%}); HARD: 17.9%->12.2%.\n"
  f"**open-book hurt is {'ARTIFACT (v0.3 VOID)' if artifact else 'REAL'}.**\n")
print("  wrote RESULTS_easycheck.md")
