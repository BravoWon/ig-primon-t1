#!/usr/bin/env python
"""DeltaSheaf-v0.3 — the DECISIVE law test with the RIGHT reader. The sheaf embedding-cosine readout is
degenerate (topic vector can't pick the gold option). Instead: give a small model (Qwen-3B, which failed
these items closed-book as part of the 0-of-5 ensemble) the RETRIEVED PASSAGE and let it READ. If it
recovers gold open-book on the items it (and all LLMs to 7B) missed closed-book, the law is confirmed:
import from a complementary system, READ, recovers the hole. Split by passage-present vs absent."""
import json, os, sys, re
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
LET=["A","B","C","D"]
def parse(t):
    m=re.findall(r"Answer:\s*([ABCD])",t,re.I); return m[-1].upper() if m else (re.findall(r"\b([ABCD])\b",t) or [None])[-1]

gate_idx=np.load("data/gate_idx.npy")
gpool=[json.loads(l) for l in open("data/gate_pool.jsonl",encoding="utf-8")]
allret=[json.loads(l)["passage"] for l in open("data/retrieval.jsonl",encoding="utf-8")]
passages=[allret[int(i)] for i in gate_idx]                         # aligned to gate items
has=[len(p)>20 for p in passages]
print(f"[openbook] {len(gpool)} blind-spot items, {sum(has)} with a passage ({sum(has)/len(gpool):.0%})")

dev="cuda"; MODEL="Qwen/Qwen2.5-3B-Instruct"
tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
mdl=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).to(dev).eval()
if tok.pad_token_id is None: tok.pad_token=tok.eos_token

def prompt(it,passage,openbook):
    opts="\n".join(f"{LET[i]}. {c}" for i,c in enumerate(it["choices"]))
    ctx=f"Context (may help):\n{passage[:1000]}\n\n" if (openbook and len(passage)>20) else ""
    return (f"{ctx}Answer the multiple-choice question. Give a one-line reason then 'Answer: X'.\n\n"
            f"Question: {it['question']}\n{opts}\n")

def run(openbook, bs=12):
    preds=[]
    for s in range(0,len(gpool),bs):
        msgs=[[{"role":"user","content":prompt(gpool[j],passages[j],openbook)}] for j in range(s,min(s+bs,len(gpool)))]
        enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt",return_dict=True,padding=True).to(dev)
        with torch.no_grad(): out=mdl.generate(**enc,max_new_tokens=200,do_sample=False,pad_token_id=tok.pad_token_id)
        for k in range(out.shape[0]):
            preds.append(parse(tok.decode(out[k,enc["input_ids"].shape[1]:],skip_special_tokens=True)))
    return preds

gold=[LET[it["answer"]] for it in gpool]
cb=run(False); ob=run(True)
def acc(preds,mask=None):
    idx=range(len(gold)) if mask is None else [i for i in range(len(gold)) if mask[i]]
    return sum(preds[i]==gold[i] for i in idx)/max(1,len(idx)), len(idx)
cb_a,_=acc(cb); ob_a,_=acc(ob)
ob_has,nh=acc(ob,has); ob_no,nn_=acc(ob,[not h for h in has])
cb_has,_=acc(cb,has)
print(f"\n  Qwen-3B on the 322 blind-spot items (chance 0.25):")
print(f"    CLOSED-book (no passage):            {cb_a:.1%}")
print(f"    OPEN-book  (with retrieved passage): {ob_a:.1%}   Δ = {ob_a-cb_a:+.1%}")
print(f"    open-book, passage PRESENT ({nh}):   {ob_has:.1%}   (closed-book same items: {cb_has:.1%})")
print(f"    open-book, passage ABSENT  ({nn_}):   {ob_no:.1%}")
law = ob_has>0.35 and ob_has-cb_has>0.10
print(f"\n  => LAW {'CONFIRMED' if law else 'not supported'}: reading an imported passage {'RECOVERS' if law else 'does not recover'} gold "
      f"on items every LLM (26M-7B) missed closed-book. The reading IS the map.")
open("RESULTS_openbook.md","w",encoding="utf-8").write(
  f"# v0.3 open-book (the real reader)\nclosed {cb_a:.1%} -> open {ob_a:.1%}; passage-present {ob_has:.1%} vs closed {cb_has:.1%}; absent {ob_no:.1%}.\n"
  f"**LAW {'CONFIRMED' if law else 'not supported'}.**\n")
print("  wrote RESULTS_openbook.md")
