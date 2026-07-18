#!/usr/bin/env python
"""Heterogeneity go/no-go: does the MATH SPECIALIST (Qwen2.5-Math-7B-Instruct) recover the DeltaSheaf
blind spots that the generalist ensemble (incl. Qwen2.5-7B, 25.8%) failed? If Math-7B >> 25.8%, the pair is
epistemically heterogeneous (specialist holds what generalist lacks) and the Procrustes transplant is worth
running. If Math-7B is also ~chance, even the specialist doesn't hold them -> abort. Checkpointed, offline."""
import json, os, re, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
DS="../deltasheaf-v02"; LET=["A","B","C","D"]
def parse(t):
    m=re.findall(r"Answer:\s*\(?([ABCD])\)?",t,re.I)
    if m: return m[-1].upper()
    m=re.findall(r"\\boxed\{\(?([ABCD])\)?\}",t)          # math models love \boxed{}
    if m: return m[-1].upper()
    m=re.findall(r"\b([ABCD])\b",t); return m[-1].upper() if m else None
PROMPT=("Answer the multiple-choice question. First give a brief reason, then on a new line write exactly "
        "'Answer: X' where X is the letter (A, B, C, or D) of the correct option.\n\nQuestion: {q}\n{opts}\n")

pool=[json.loads(l) for l in open(f"{DS}/data/mmlu_pool.jsonl",encoding="utf-8")]
gate_idx=[int(i) for i in np.load(f"{DS}/data/gate_idx.npy")]
out=f"{DS}/data/raw/math7b_gate.jsonl"
done=sum(1 for _ in open(out,encoding="utf-8")) if os.path.exists(out) else 0
if done>=len(gate_idx):
    rows=[json.loads(l) for l in open(out,encoding="utf-8")]
    acc=sum(r["correct"] for r in rows)/len(rows)
    print(f"[DONE] Math-7B on {len(rows)} blind spots = {acc:.1%}  (Qwen-7B generalist 25.8%, chance 25%)")
    sys.exit(0)

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"; BS=6; MAXNEW=512
print(f"[score_math] {MODEL} (4-bit nf4, on-GPU)  resume@{done}/{len(gate_idx)}",flush=True)
tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
mdl=AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb,device_map={"":0}).eval()
fout=open(out,"a",encoding="utf-8"); t0=time.time()
todo=gate_idx[done:]
for s in range(0,len(todo),BS):
    chunk=todo[s:s+BS]
    msgs=[[{"role":"user","content":PROMPT.format(q=pool[i]["question"],
           opts="\n".join(f"{LET[k]}. {c}" for k,c in enumerate(pool[i]["choices"])))}] for i in chunk]
    enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt",
                                return_dict=True,padding=True).to(mdl.device)
    with torch.no_grad(): o=mdl.generate(**enc,max_new_tokens=MAXNEW,do_sample=False,pad_token_id=tok.pad_token_id)
    plen=enc["input_ids"].shape[1]
    for k,i in enumerate(chunk):
        reply=tok.decode(o[k,plen:],skip_special_tokens=True).strip()
        letter=parse(reply); gold=LET[pool[i]["answer"]]
        fout.write(json.dumps({"i":i,"letter":letter,"correct":letter==gold,"reply":reply[:400]},ensure_ascii=False)+"\n")
    fout.flush()
    if (s//BS)%5==0:
        cur=[json.loads(l) for l in open(out,encoding="utf-8")]
        print(f"    {done+s+len(chunk)}/{len(gate_idx)}  running acc {sum(r['correct'] for r in cur)/len(cur):.1%}  ({time.time()-t0:.0f}s)",flush=True)
fout.close()
rows=[json.loads(l) for l in open(out,encoding="utf-8")]; acc=sum(r["correct"] for r in rows)/len(rows)
print(f"\n[HETEROGENEITY]  Math-7B specialist on {len(rows)} blind spots = {acc:.1%}   (Qwen-7B generalist 25.8%, chance 25%)")
print(f"  => {'HETEROGENEOUS — specialist recovers the holes; Procrustes transplant is worth running' if acc>0.45 else ('weak gap — marginal' if acc>0.35 else 'NO — even the specialist fails the blind spots; abort the transplant')}")
