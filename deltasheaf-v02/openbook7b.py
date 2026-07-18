#!/usr/bin/env python
"""DeltaSheaf-v0.3 resolution — Qwen2.5-7B as the open-book reader, CHECKPOINTED (kill-safe).
Pre-registered protocol (control FIRST, hard abort):
  Phase A  easy_closed — 3B-easy items, closed book.
  Phase B  easy_open   — same items + retrieved passage. CONTROL: if open-closed < -3%
           the 7B reader ALSO degrades on context -> reader invalid -> STOP (do not run the gate).
  AMENDMENT (pre-registered before relaunch, 2026-07-17): first pass at n=150 fired the abort at
  Δ-4.7% but McNemar p=0.19 (14 harmed / 7 helped) — the CONTROL was underpowered. Extended to
  n=400 (same deterministic order; first 150 rows already banked are reused). Decision rule FIXED
  now: verdict on ALL 400; Δ<=-3% -> reader invalid at both scales (bank); else control passes ->
  run the gate. No further extensions.
  Phase C  gate_open   (322)  — blind-spot items + passage. Compare vs the existing
           data/raw/qwen7b_gate.jsonl closed-book scoring (25.8% = chance).
Each phase appends per-item to its own jsonl (flush every batch; resume by line count), so ~1h
window kills lose nothing — relaunches converge. Offline, local pool, device_map=auto (CPU offload)."""
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
ret=[json.loads(l)["passage"] for l in open("data/retrieval.jsonl",encoding="utf-8")]
gate_idx=[int(i) for i in np.load("data/gate_idx.npy")]; gate_set=set(gate_idx)
easy=[i for i in range(min(len(qwen),len(ret))) if qwen[i]["correct"] and i not in gate_set and len(ret[i])>20][:400]

MODEL="Qwen/Qwen2.5-7B-Instruct"; BS=4; MAXNEW=160
def prompt(i,openbook):
    it=pool[i]; opts="\n".join(f"{LET[k]}. {c}" for k,c in enumerate(it["choices"]))
    ctx=f"Passage: {ret[i][:1000]}\n\n" if openbook else ""
    return (f"{ctx}Question: {it['question']}\n{opts}\n"
            "Give a one-line reason then 'Answer: X'.\n")

PHASES=[("easy_closed",easy,False),("easy_open",easy,True),("gate_open",gate_idx,True)]
def path(tag): return f"data/raw/qwen7b_{tag}.jsonl"
def done(tag): return sum(1 for _ in open(path(tag),encoding="utf-8")) if os.path.exists(path(tag)) else 0
def acc(tag,idxs):
    rows=[json.loads(l) for l in open(path(tag),encoding="utf-8")]
    return sum(r["correct"] for r in rows)/max(1,len(rows)), len(rows)

# early exit without loading the model if all phases (or the abort condition) are already resolved
need=[(t,i,o) for (t,i,o) in PHASES if done(t)<len(i)]
if need and done("easy_closed")==len(easy) and done("easy_open")==len(easy):
    a_c,_=acc("easy_closed",easy); a_o,_=acc("easy_open",easy)
    if a_o-a_c<-0.03:
        print(f"[CONTROL FAILED] easy closed {a_c:.1%} -> open {a_o:.1%} (Δ{a_o-a_c:+.1%}). "
              "7B reader ALSO degrades on context. Gate NOT run. v0.3 verdict: reader-invalid at 7B too.")
        sys.exit(3)
if not need:
    print("[all phases complete]")
else:
    print(f"[openbook7b] pending: "+", ".join(f"{t}@{done(t)}/{len(i)}" for (t,i,o) in need),flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    mdl=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="auto").eval()
    t0=time.time()
    for tag,idxs,ob in PHASES:
        d=done(tag)
        if d>=len(idxs): continue
        # control gate before starting the expensive gate phase
        if tag=="gate_open":
            a_c,_=acc("easy_closed",easy); a_o,_=acc("easy_open",easy)
            print(f"  [control] easy closed {a_c:.1%} -> open {a_o:.1%}  Δ{a_o-a_c:+.1%}",flush=True)
            if a_o-a_c<-0.03:
                print("  [CONTROL FAILED] aborting gate phase — reader invalid."); sys.exit(3)
            print("  [control PASSED] 7B reader does not degrade on context; running the gate.",flush=True)
        fout=open(path(tag),"a",encoding="utf-8")
        todo=idxs[d:]
        for s in range(0,len(todo),BS):
            chunk=todo[s:s+BS]
            msgs=[[{"role":"user","content":prompt(i,ob)}] for i in chunk]
            enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt",
                                        return_dict=True,padding=True).to(mdl.device)
            with torch.no_grad():
                out=mdl.generate(**enc,max_new_tokens=MAXNEW,do_sample=False,pad_token_id=tok.pad_token_id)
            plen=enc["input_ids"].shape[1]
            for k,i in enumerate(chunk):
                reply=tok.decode(out[k,plen:],skip_special_tokens=True).strip()
                letter=parse(reply); gold=LET[pool[i]["answer"]]
                fout.write(json.dumps({"i":i,"letter":letter,"correct":letter==gold,
                                       "reply":reply[:200]},ensure_ascii=False)+"\n")
            fout.flush()
            if (s//BS)%5==0:
                print(f"    {tag} {d+s+len(chunk)}/{len(idxs)} ({time.time()-t0:.0f}s)",flush=True)
        fout.close(); print(f"  [{tag}] complete",flush=True)

# final report if everything is in
if all(done(t)>=len(i) for (t,i,o) in PHASES):
    a_c,_=acc("easy_closed",easy); a_o,_=acc("easy_open",easy)
    g_rows=[json.loads(l) for l in open(path("gate_open"),encoding="utf-8")]
    g_open=sum(r["correct"] for r in g_rows)/len(g_rows)
    g_closed=sum(json.loads(l)["correct"] for l in open("data/raw/qwen7b_gate.jsonl",encoding="utf-8"))/322
    print(f"\n[v0.3 RESOLUTION]  easy: closed {a_c:.1%} -> open {a_o:.1%} (control Δ{a_o-a_c:+.1%})")
    print(f"  GATE (322 blind spots, chance 25%): closed {g_closed:.1%} -> open {g_open:.1%}  Δ{g_open-g_closed:+.1%}")
    verdict = "IMPORT RECOVERS (law supported)" if g_open-g_closed>0.10 and g_open>0.35 else \
              ("weak/partial import effect" if g_open-g_closed>0.04 else "import does NOT recover (law refuted on this instrument)")
    print(f"  => {verdict}")
    open("RESULTS_v3_7b.md","w",encoding="utf-8").write(
      f"# v0.3 resolution (7B reader, checkpointed)\neasy {a_c:.1%}->{a_o:.1%} (control ok). "
      f"gate closed {g_closed:.1%} -> open {g_open:.1%}.\n**{verdict}**\n")
    print("  wrote RESULTS_v3_7b.md")
