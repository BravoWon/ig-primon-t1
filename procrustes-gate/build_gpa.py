#!/usr/bin/env python
"""GPA — "the map is the terrain between." Generalized Procrustes co-registration of a dim-matched
heterogeneous trio (Qwen2.5-3B / SmolLM2-1.7B / OLMo-2-1B, native hidden dim 2048) into ONE shared consensus
frame U, then test whether the aligned terrain holds task-live signal the un-aligned individual bases obscure.
NOTE: distinct substrate from v0.4-v0.10 (which used nomic reply-EMBEDDINGS in a common space); here we align
each model's NATIVE final-layer answer-position hidden state, in its own arbitrary rotation.
  Test 1 UNIVERSAL READ: does sender m's aligned rep decode to m's own answer through reader r's head? (3x3)
  Test 2 CONSENSUS SIGNAL: does the centroid U decode to the GOLD answer above best-single & majority?
    control: aligned-consensus vs RAW-average-consensus (different frames -> should be garbage) = alignment matters.
Instrument-first: per-model readout validity. Offline; hiddens cached; small models (no offload)."""
import json, os, re, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
DS="../deltasheaf-v02"; LET=["A","B","C","D"]; N_ITEMS=1500
PROMPT=("Answer the multiple-choice question. First give a brief one- or two-sentence reason, then on a "
        "new line write exactly 'Answer: X' where X is the letter of the correct option.\n\nQuestion: {q}\n{opts}\n")
TRIO=[("qwen3b","Qwen/Qwen2.5-3B-Instruct","qwen25_3b"),
      ("smollm","HuggingFaceTB/SmolLM2-1.7B-Instruct","smollm2_17b"),
      ("olmo","allenai/OLMo-2-0425-1B-Instruct","olmo2_1b")]
pool=[json.loads(l) for l in open(f"{DS}/data/mmlu_pool.jsonl",encoding="utf-8")]
gold=np.array([pool[i]["answer"] for i in range(N_ITEMS)])

def rmsnorm(x,w,eps): return x/np.sqrt((x*x).mean(-1,keepdims=True)+eps)*w
def extract(tag,mid,rawtag):
    cache=f"data/gpa_{tag}.npz"
    if os.path.exists(cache):
        z=np.load(cache); return z["H"],z["correct"],z["ans"],z["normw"],float(z["eps"]),z["urows"]
    rows=[json.loads(l) for l in open(f"{DS}/data/raw/{rawtag}.jsonl",encoding="utf-8")]
    print(f"[extract {tag}] {N_ITEMS} native hiddens",flush=True)
    tok=AutoTokenizer.from_pretrained(mid)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    mdl=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.float16).to("cuda").eval()
    lid=[tok(s,add_special_tokens=False)["input_ids"][0] for s in LET]   # one id per letter
    urows=mdl.lm_head.weight.data[lid].float().cpu().numpy()             # [4,2048]
    normw=mdl.model.norm.weight.data.float().cpu().numpy(); eps=float(getattr(mdl.model.norm,"variance_epsilon",1e-6))
    H=np.zeros((N_ITEMS,mdl.config.hidden_size),np.float32); cor=np.zeros(N_ITEMS); ans=np.zeros(N_ITEMS,int)
    t0=time.time()
    for i in range(N_ITEMS):
        it=pool[i]; reply=rows[i]["reply"]; m=list(re.finditer(r"Answer:",reply,re.I))
        opts="\n".join(f"{LET[k]}. {c}" for k,c in enumerate(it["choices"]))
        pref=tok.apply_chat_template([{"role":"user","content":PROMPT.format(q=it["question"],opts=opts)}],
              tokenize=False,add_generation_prompt=True)
        cut=m[-1].end() if m else len(reply)
        ids=torch.tensor([tok(pref+reply[:cut],add_special_tokens=False)["input_ids"]]).to("cuda")
        with torch.no_grad(): o=mdl(ids,output_hidden_states=True)
        H[i]=o.hidden_states[-1][0,-1,:].float().cpu().numpy()
        cor[i]=rows[i]["correct"]; ans[i]=LET.index(rows[i]["letter"]) if rows[i].get("letter") in LET else 0
        if i%300==0: print(f"    {tag} {i}/{N_ITEMS} ({time.time()-t0:.0f}s)",flush=True)
    np.savez(cache,H=H,correct=cor,ans=ans,normw=normw,eps=eps,urows=urows)
    del mdl; torch.cuda.empty_cache()
    return H,cor,ans,normw,eps,urows

Hs,cors,anss,normws,epss,urowss=[],[],[],[],[],[]
for tag,mid,rt in TRIO:
    H,c,a,nw,ep,ur=extract(tag,mid,rt); Hs.append(H); cors.append(c); anss.append(a); normws.append(nw); epss.append(ep); urowss.append(ur)
cor=np.stack(cors); ans=np.stack(anss)                 # [3,N]
def decode(x,m):                                       # x [.,2048] in model-m frame -> letter idx
    return np.argmax(rmsnorm(x,normws[m],epss[m])@urowss[m].T,-1)

# instrument: each model's own hidden decodes to its own answer
print("\n[instrument] readout validity (model hidden -> own answer):",
      "  ".join(f"{TRIO[m][0]} {(decode(Hs[m],m)==ans[m]).mean():.0%}" for m in range(3)))
print(f"[per-model acc] "+"  ".join(f"{TRIO[m][0]} {cor[m].mean():.1%}" for m in range(3)))

# ---- Generalized Procrustes ----
mus=[H.mean(0) for H in Hs]; Xc=[H-mu for H,mu in zip(Hs,mus)]
Rs=[np.eye(Hs[0].shape[1],dtype=np.float32) for _ in Hs]; aligned=[X.copy() for X in Xc]
U=np.mean(aligned,0)
for _ in range(12):
    for m in range(3):
        W,_,Vt=np.linalg.svd(U.T@Xc[m]); Rs[m]=(W@Vt).astype(np.float32); aligned[m]=Xc[m]@Rs[m].T
    U=np.mean(aligned,0)
disp=np.mean([np.linalg.norm(aligned[m]-U,axis=1).mean() for m in range(3)])
print(f"[GPA] converged; mean residual dispersion to consensus {disp:.3f}")

# Test 1 — universal read: sender m -> reader r frame -> reader head -> sender's own answer?
print("\n[Test 1] UNIVERSAL READ  (sender aligned rep decoded through reader's head -> sender's own answer)")
M1=np.zeros((3,3))
for sm in range(3):
    for rd in range(3):
        xr = (Xc[sm]@Rs[sm].T)@Rs[rd] + mus[rd]        # sender -> consensus -> reader frame
        M1[sm,rd]=(decode(xr,rd)==ans[sm]).mean()
    print("   sender "+TRIO[sm][0].ljust(7)+" -> readers "+"  ".join(f"{TRIO[r][0]}:{M1[sm,r]:.0%}" for r in range(3)))
offdiag=(M1.sum()-np.trace(M1))/6
print(f"   mean OFF-diagonal cross-read {offdiag:.0%}  (universal frame if high; identity diag ~readout-valid)")

# Test 2 — consensus centroid signal vs the parts
best=int(cor.mean(1).argmax()); best_acc=cor[best].mean()
maj=np.array([np.bincount(ans[:,i],minlength=4).argmax() for i in range(N_ITEMS)]); maj_acc=(maj==gold).mean()
# decode consensus U mapped into each model frame; take plurality across the 3 head-reads
cons_reads=np.stack([decode(U@Rs[m]+mus[m],m) for m in range(3)])       # [3,N]
cons=np.array([np.bincount(cons_reads[:,i],minlength=4).argmax() for i in range(N_ITEMS)])
cons_acc=(cons==gold).mean()
raw_cons=np.array([np.bincount([decode((np.mean(Hs,0))[i:i+1],m)[0] for m in range(3)],minlength=4).argmax() for i in range(N_ITEMS)])
raw_acc=(raw_cons==gold).mean()
print(f"\n[Test 2] CONSENSUS SIGNAL")
print(f"   best single model ({TRIO[best][0]}) {best_acc:.1%}   majority vote {maj_acc:.1%}")
print(f"   ALIGNED consensus U (plurality of head-reads) {cons_acc:.1%}   RAW-average consensus {raw_acc:.1%}")
beats = cons_acc>best_acc+0.01 and cons_acc>maj_acc+0.01
align_matters = cons_acc>raw_acc+0.05
print(f"\n  => TERRAIN {'is TASK-LIVE — aligned consensus beats best-single AND majority (more than the sum of its maps)' if beats else 'is an inert average — aligned consensus does not beat best-single/majority (reduces to banked null)'};"
      f"  co-registration {'MATTERS (aligned >> raw-average)' if align_matters else 'does not help vs raw'}.")
open("RESULTS_gpa.md","w",encoding="utf-8").write(
  f"# GPA terrain-between  trio {[t[0] for t in TRIO]}  N={N_ITEMS}\n"
  f"universal cross-read off-diag {offdiag:.0%}; GPA dispersion {disp:.3f}.\n"
  f"best-single {best_acc:.1%}, majority {maj_acc:.1%}, ALIGNED-consensus {cons_acc:.1%}, raw-avg {raw_acc:.1%}.\n"
  f"**terrain {'TASK-LIVE' if beats else 'inert average'}; alignment {'matters' if align_matters else 'no help'}.**\n")
print("  wrote RESULTS_gpa.md")
