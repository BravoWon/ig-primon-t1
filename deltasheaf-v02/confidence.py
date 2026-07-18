#!/usr/bin/env python
"""DeltaSheaf-v0.5 — within-model CONFIDENCE via teacher-forcing on each model's OWN stored reasoned reply
(data/raw/*.jsonl). ONE forward pass per (model,item), no generation. From it, at the model's actual answer
position (post-reasoning), read the 4-way letter distribution; over the reasoning tokens, read perplexity.
Features per item: p[4] (renorm letter probs), amrg (answer logit-margin), rnll/rmax (reasoning mean/max NLL).
Item-level resume (append+flush) — kill-safe. Offline + local. Batch=1 for index-correctness."""
import json, os, re, sys, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
LET=["A","B","C","D"]
PROMPT=("Answer the multiple-choice question. First give a brief one- or two-sentence reason, then on a "
        "new line write exactly 'Answer: X' where X is the letter of the correct option.\n\nQuestion: {q}\n{opts}\n")
MODELS=[("qwen3b","Qwen/Qwen2.5-3B-Instruct","qwen25_3b"),
        ("phi","microsoft/Phi-3.5-mini-instruct","phi35_mini"),
        ("smollm","HuggingFaceTB/SmolLM2-1.7B-Instruct","smollm2_17b"),
        ("falcon","tiiuae/Falcon3-3B-Instruct","falcon3_3b"),
        ("olmo","allenai/OLMo-2-0425-1B-Instruct","olmo2_1b")]
pool=[json.loads(l) for l in open("data/mmlu_pool.jsonl",encoding="utf-8")]
dev="cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("data/conf",exist_ok=True)
ONLY=sys.argv[1] if len(sys.argv)>1 else None      # optional: run one model tag per foreground call

def letter_ids(tok):
    d={}
    for L in LET:
        c=[]
        for s in (L," "+L):
            t=tok(s,add_special_tokens=False)["input_ids"]
            if t: c.append(t[0])
        d[L]=list(dict.fromkeys(c))
    return d

for tag,mid,rawtag in MODELS:
    if ONLY and tag!=ONLY: continue
    out=f"data/conf/{tag}.jsonl"
    rows=[json.loads(l) for l in open(f"data/raw/{rawtag}.jsonl",encoding="utf-8")]
    N=min(len(pool),len(rows))                      # align to items that have replies + embeddings
    done=sum(1 for _ in open(out,encoding="utf-8")) if os.path.exists(out) else 0
    if done>=N: print(f"  DONE {tag} ({done})",flush=True); continue
    print(f"[{tag}] {mid}  resume@{done}",flush=True)
    tok=AutoTokenizer.from_pretrained(mid)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    mdl=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.float16).to(dev).eval()
    LID=letter_ids(tok); fout=open(out,"a",encoding="utf-8"); t0=time.time()
    for i in range(done,N):
        it=pool[i]; r=rows[i]; reply=r["reply"]
        opts="\n".join(f"{LET[k]}. {c}" for k,c in enumerate(it["choices"]))
        pref=tok.apply_chat_template([{"role":"user","content":PROMPT.format(q=it["question"],opts=opts)}],
              tokenize=False,add_generation_prompt=True)
        full_ids=tok(pref+reply,add_special_tokens=False)["input_ids"]
        plen=len(tok(pref,add_special_tokens=False)["input_ids"])
        m=list(re.finditer(r"Answer:",reply,re.I))
        T=len(full_ids); ids=torch.tensor([full_ids]).to(dev)
        with torch.no_grad(): lg=mdl(ids).logits[0]                       # [T,V]
        # reasoning perplexity over reply tokens (positions plen-1 .. T-2 predict reply tokens)
        if T>plen:
            lp=F.log_softmax(lg[plen-1:T-1].float(),-1)
            tgt=torch.tensor(full_ids[plen:T]).to(dev)
            nll=(-lp.gather(1,tgt[:,None]).squeeze(1)).cpu().numpy()
            rnll,rmax=float(nll.mean()),float(nll.max())
        else: rnll=rmax=0.0
        # answer-position letter distribution
        if m:
            cut=m[-1].end(); apos=len(tok(pref+reply[:cut],add_special_tokens=False)["input_ids"])
            apos=min(max(apos,1),T)
            al=lg[apos-1].float()
            pr=torch.softmax(al,-1)
            praw=[float(max(pr[t].item() for t in LID[L])) for L in LET]
            lgl=[float(max(al[t].item() for t in LID[L])) for L in LET]
            s=sum(praw) or 1.0; p=[x/s for x in praw]
            ch=int(np.argmax(lgl)); amrg=lgl[ch]-max(lgl[j] for j in range(4) if j!=ch)
            ans=int(np.argmax(p))
        else:
            p=[0.25,0.25,0.25,0.25]; amrg=0.0; ans=LET.index(r["letter"]) if r.get("letter") in LET else 0
        fout.write(json.dumps({"i":i,"ans":ans,"correct":bool(r["correct"]),
                   "p":[round(x,5) for x in p],"amrg":round(amrg,4),
                   "rnll":round(rnll,4),"rmax":round(rmax,4)},ensure_ascii=False)+"\n")
        if i%200==0: fout.flush(); print(f"    {tag} {i}/{N} ({time.time()-t0:.0f}s)",flush=True)
    fout.flush(); fout.close(); del mdl; torch.cuda.empty_cache()
    print(f"[{tag}] done ({time.time()-t0:.0f}s)",flush=True)
print("ALL DONE",flush=True)
