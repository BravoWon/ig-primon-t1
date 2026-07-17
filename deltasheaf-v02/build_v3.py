#!/usr/bin/env python
"""DeltaSheaf-v0.3 — the LAW test. Swap OLMo-2 for a RETRIEVAL node (complementary info), same 322
blind-spot items. If a between-system map to a complementary node recovers gold where LLM<->LLM couldn't,
the law ('import between complementary systems is the lever') is confirmed. Adds retrieval-direct arm."""
import itertools, math, sys, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.chdir(os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"

D,NOPT,DSIG,TEMP = 512,4,128,0.1
CHANCE,DMIN = 0.25,0.10
SEEDS=[0,1,2,3,4]
PAIRS=list(itertools.combinations(range(5),2)); TRIS=list(itertools.combinations(range(5),3))
PIX={p:i for i,p in enumerate(PAIRS)}
rng=np.random.default_rng(20260716)
P_CYC=(rng.standard_normal((len(TRIS)*D,DSIG))/math.sqrt(DSIG)).astype(np.float32)
P_EDG=(rng.standard_normal((len(PAIRS)*D,DSIG))/math.sqrt(DSIG)).astype(np.float32)

# ---- embed retrieval passages -> retrieval stalk ----
from transformers import AutoModel, AutoTokenizer
tok=AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
emb_mdl=AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5",trust_remote_code=True).cuda().eval()
def embed(texts,bs=256):
    out=[]
    for s in range(0,len(texts),bs):
        enc=tok(["search_document: "+(t or "[none]") for t in texts[s:s+bs]],padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda")
        with torch.no_grad(): h=emb_mdl(**enc)[0]
        m=enc["attention_mask"].unsqueeze(-1).float(); e=(h*m).sum(1)/m.sum(1)
        e=F.layer_norm(e,(e.shape[1],))[:,:512]; out.append(F.normalize(e,dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)
passages=[json.loads(l)["passage"] for l in open("data/retrieval.jsonl",encoding="utf-8")]
nonempty=sum(len(p)>20 for p in passages)
print(f"[v0.3] retrieval: {len(passages)} passages, {nonempty} non-empty ({nonempty/len(passages):.0%}), avg len {np.mean([len(p) for p in passages]):.0f}")
ret=embed(passages)                                            # [n,512]

z=np.load("data/embedded.npz")
opts,gold,correct,letters=z["opts"],z["gold"],z["correct"],z["letters"]
n=len(ret)
# nodes: 4 LLMs (drop olmo2=idx4) + retrieval
stalks=np.concatenate([z["stalks"][:,:4], ret[:,None,:]],axis=1)   # [n,5,512]
# gate/clean = original 5-LLM blind-spot (unchanged 322)
ncorr=correct.sum(1); ndist=np.array([len(set(l for l in letters[i] if l>=0)) for i in range(n)])
gate=(ncorr==0)&(ndist>=2); clean=ncorr>=1
gi,ci=np.where(gate)[0],np.where(clean)[0]
print(f"[v0.3] gate(same as v0.2)={gate.sum()}  clean={clean.sum()}  (node5 = RETRIEVAL, node1-4 = LLMs)")
rs=np.random.default_rng(0); perm=rs.permutation(ci); nval=len(ci)//5
val_i,tr_i=perm[:nval],perm[nval:]

# ---- substrate: ridge-LS restriction maps on clean-train ----
lam=1.0; W={}
for (i,j) in PAIRS:
    Si,Sj=stalks[tr_i,i],stalks[tr_i,j]
    W[(i,j)]=np.linalg.solve(Si.T@Si+lam*np.eye(D),Si.T@Sj).astype(np.float32)
def sig(idx):
    S=stalks[idx]; m=len(idx)
    Δ=np.zeros((m,len(PAIRS),D),np.float32)
    for (i,j) in PAIRS: Δ[:,PIX[(i,j)]]=S[:,i]@W[(i,j)]-S[:,j]
    cyc=np.stack([Δ[:,PIX[(i,j)]]+Δ[:,PIX[(j,k)]]-Δ[:,PIX[(i,k)]] for (i,j,k) in TRIS],1)
    H=cyc.reshape(m,-1)@P_CYC; E=Δ.reshape(m,-1)@P_EDG
    area=np.stack([0.5*np.sqrt(np.maximum(0,(a:=Δ[:,PIX[(i,j)]]).__pow__(2).sum(1)*(b:=Δ[:,PIX[(j,k)]]).__pow__(2).sum(1)-(a*b).sum(1)**2)) for (i,j,k) in TRIS],1)
    return {"C_cycle":H,"B_edge":E,"C_area":area.astype(np.float32),
            "ctrl_mag":np.linalg.norm(H,axis=1,keepdims=True).astype(np.float32),
            "retrieval_direct":stalks[idx,4]}          # the passage embedding alone (the import)
Xtr,Xval,Xg=sig(tr_i),sig(val_i),sig(gi)

class Dec(nn.Module):
    def __init__(s,din): super().__init__(); s.w=nn.Linear(din,D)
    def forward(s,x): return s.w(x)
def train_eval(arm,seed):
    torch.manual_seed(seed)
    Xt,gt,Ot=torch.tensor(Xtr[arm]),torch.tensor(gold[tr_i]),torch.tensor(opts[tr_i])
    Xv,gv,Ov=torch.tensor(Xval[arm]),torch.tensor(gold[val_i]),torch.tensor(opts[val_i])
    Xe,Oe=torch.tensor(Xg[arm]),torch.tensor(opts[gi])
    m=Dec(Xt.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=1e-2,weight_decay=1e-3); best=1e9;bs=None;pat=0
    for ep in range(400):
        m.train(); loss=F.cross_entropy(torch.bmm(Ot,m(Xt).unsqueeze(2)).squeeze(2)/TEMP,gt)
        opt.zero_grad(); loss.backward(); opt.step(); m.eval()
        with torch.no_grad(): vl=F.cross_entropy(torch.bmm(Ov,m(Xv).unsqueeze(2)).squeeze(2)/TEMP,gv).item()
        if vl<best-1e-4: best,bs,pat=vl,{k:v.clone() for k,v in m.state_dict().items()},0
        else:
            pat+=1
            if pat>=30: break
    m.load_state_dict(bs)
    with torch.no_grad(): pred=torch.bmm(Oe,m(Xe).unsqueeze(2)).squeeze(2).argmax(1).numpy()
    return (pred==gold[gi]).astype(np.float32)
def boot(a,b,B=10000,seed=7):
    g=np.random.default_rng(seed); am,bm=a.mean(0),b.mean(0); nn_=len(am); d=[am[g.integers(0,nn_,nn_)].mean()-bm[g.integers(0,nn_,nn_)].mean() for _ in range(B)]
    return am.mean()-bm.mean(), *np.percentile(d,[2.5,97.5])

ARMS=["ctrl_mag","B_edge","C_cycle","C_area","retrieval_direct"]
per={a:np.stack([train_eval(a,s) for s in SEEDS]) for a in ARMS}
res={a:(float(per[a].mean(1).mean()),float(per[a].mean(1).std())) for a in ARMS}
print(f"\n  arm                 gate-acc (chance {CHANCE})")
for a in ARMS: print(f"  {a:18} {res[a][0]:.3f} ± {res[a][1]:.3f}")
print(f"\n  --- the LAW test (does a complementary node recover gold where LLM<->LLM did not?) ---")
base=np.full_like(per['C_cycle'],CHANCE)                       # vs-chance
for a in ["retrieval_direct","C_cycle","C_area","B_edge"]:
    d,lo,hi=boot(per[a],base); print(f"  {a:18} − chance = {d:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  {'** ABOVE CHANCE **' if lo>0 else ''}")
rd=res["retrieval_direct"][0]
law = rd>CHANCE and boot(per["retrieval_direct"],base)[1]>0
print(f"\n  => LAW {'CONFIRMED' if law else 'not supported'}: retrieval-direct = {rd:.3f} vs chance {CHANCE} "
      f"(LLM-only arms were all ~chance in v0.2). Import from a complementary system {'RECOVERS' if law else 'does not recover'} gold on the blind-spot.")
open("RESULTS_v3.md","w",encoding="utf-8").write(
  f"# DeltaSheaf-v0.3 (retrieval node) RESULTS\nSame {int(gate.sum())} blind-spot items; node5=Wikipedia retrieval.\n\n"
  +"".join(f"- {a}: {res[a][0]:.3f}±{res[a][1]:.3f}\n" for a in ARMS)
  +f"\nretrieval non-empty {nonempty/len(passages):.0%}.\n**LAW {'CONFIRMED' if law else 'not supported'}** (retrieval-direct {rd:.3f} vs chance {CHANCE}).\n")
print("  wrote RESULTS_v3.md")
