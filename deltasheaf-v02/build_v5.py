#!/usr/bin/env python
"""DeltaSheaf-v0.5 — ORTHOGONAL construct: within-model CONFIDENCE as the node-importance signal.
(1) Instrument: within each model, does confidence separate its own correct from wrong answers? (AUC)
(2) Headline (non-learned): route to the MOST-CONFIDENT model per item — beat best-single & majority?
(3) Learned confidence router (5 seeds) vs shuffle. (4) UNION with the v0.4 geometry map — does confidence
ADD over the between-model delta (the complementary-import law)? Merit frame: routable items (>=1 correct),
held-out split, oracle ceiling = 100%. CPU, ~1 min."""
import itertools, json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
np.random.seed(0); torch.manual_seed(0)
TAGS=["qwen3b","phi","smollm","falcon","olmo"]; M=len(TAGS)
pool=[json.loads(l) for l in open("data/mmlu_pool.jsonl",encoding="utf-8")]
conf=[[json.loads(l) for l in open(f"data/conf/{t}.jsonl",encoding="utf-8")] for t in TAGS]
N=min(len(c) for c in conf)
gold=np.array([pool[i]["answer"] for i in range(N)])
correct=np.array([[conf[m][i]["correct"] for m in range(M)] for i in range(N)],dtype=np.float32)
ans    =np.array([[conf[m][i]["ans"]     for m in range(M)] for i in range(N)])
P      =np.array([[conf[m][i]["p"]       for m in range(M)] for i in range(N)],dtype=np.float32)   # [N,M,4]
amrg   =np.array([[conf[m][i]["amrg"]    for m in range(M)] for i in range(N)],dtype=np.float32)
rnll   =np.array([[conf[m][i]["rnll"]    for m in range(M)] for i in range(N)],dtype=np.float32)
rmax   =np.array([[conf[m][i]["rmax"]    for m in range(M)] for i in range(N)],dtype=np.float32)

# confidence signals (higher = more confident)
pmax  = P.max(2)
negent= (P*np.log(P+1e-9)).sum(2)          # negative entropy
SIG={"pmax":pmax,"neg_entropy":negent,"logit_margin":amrg,"neg_reason_ppl":-rnll,"neg_reason_max":-rmax}

recov=correct.sum(1)>=1; idx=np.where(recov)[0]
rng=np.random.default_rng(0); perm=rng.permutation(idx); cut=len(idx)*3//4; tr,te=perm[:cut],perm[cut:]
print(f"[v0.5 within-model confidence]  routable (>=1 correct) = {len(idx)}  (train {len(tr)} / test {len(te)})")
print(f"  per-model acc: "+"  ".join(f"{TAGS[m]} {correct[:,m].mean():.1%}" for m in range(M)))

def auc(s,y):
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    n1=y.sum(); n0=len(y)-n1
    return 0.5 if n1==0 or n0==0 else (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

# (1) instrument: within-model correct-vs-wrong separation (pooled AUC over all model,item)
print("\n  (1) within-model confidence -> correctness AUC (0.5=no signal):")
for name,S in SIG.items():
    aucs=[auc(S[:,m],correct[:,m]) for m in range(M)]
    pooled=auc(S.reshape(-1),correct.reshape(-1))
    print(f"    {name:16s} pooled AUC {pooled:.3f}   per-model[{' '.join(f'{a:.2f}' for a in aucs)}]")

# baselines
best=int(correct[tr].mean(0).argmax()); acc_best=correct[te,best].mean()
maj=np.array([np.bincount(ans[i],minlength=4).argmax() for i in te]); acc_maj=(maj==gold[te]).mean()
print(f"\n  baselines: best-single ({TAGS[best]}) {acc_best:.1%}   majority {acc_maj:.1%}   oracle 100%")

# (2) non-learned: trust the most-confident model
print("\n  (2) route to MOST-CONFIDENT model (non-learned):")
best_route=("",0.0)
for name,S in SIG.items():
    pick=S[te].argmax(1); a=correct[te,pick].mean()
    tag="  <-- beats best-single" if a>acc_best+0.01 else ""
    print(f"    by {name:16s}: {a:.1%}{tag}")
    if a>best_route[1]: best_route=(name,a)

# (3) learned routers
FEAT=np.concatenate([P.reshape(N,-1),amrg[...,None].reshape(N,M),rnll[...,None].reshape(N,M),
                     rmax[...,None].reshape(N,M),pmax,negent],1).astype(np.float32)  # conf features
# geometry (v0.4 map): PCA(64) projected pairwise stalk deltas
z=np.load("data/embedded.npz"); stalks=z["stalks"][:N]; PAIRS=list(itertools.combinations(range(M),2)); DP=64
Sf=stalks[tr].reshape(-1,stalks.shape[2]); mu=Sf.mean(0)
_,_,Vt=np.linalg.svd(Sf-mu,full_matrices=False); PR=Vt[:DP].T.astype(np.float32)
Pr=((stalks-mu)@PR); GEO=np.stack([Pr[:,i]-Pr[:,j] for (i,j) in PAIRS],1).reshape(N,-1).astype(np.float32)
UNION=np.concatenate([FEAT,GEO],1).astype(np.float32)
def stdz(X): m=X[tr].mean(0); s=X[tr].std(0)+1e-6; return ((X-m)/s).astype(np.float32)
BANK={"confidence":stdz(FEAT),"geometry(v0.4)":stdz(GEO),"union":stdz(UNION)}
class R(nn.Module):
    def __init__(s,d): super().__init__(); s.f=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,M))
    def forward(s,x): return s.f(x)
def route(X,seed,shuffle=False):
    torch.manual_seed(seed)
    Xtr=torch.tensor(X[tr]); Ytr=torch.tensor(correct[tr]); Xte=torch.tensor(X[te])
    if shuffle: Ytr=Ytr[torch.randperm(len(Ytr))]
    m=R(Xtr.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=3e-3,weight_decay=1e-3)
    nv=len(Xtr)//5; vi,ti=torch.arange(nv),torch.arange(nv,len(Xtr)); bv=1e9;bs=None;pat=0
    for ep in range(300):
        m.train(); l=F.binary_cross_entropy_with_logits(m(Xtr[ti]),Ytr[ti]); opt.zero_grad(); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl=F.binary_cross_entropy_with_logits(m(Xtr[vi]),Ytr[vi]).item()
        if vl<bv-1e-4: bv,bs,pat=vl,{k:v.clone() for k,v in m.state_dict().items()},0
        else:
            pat+=1
            if pat>=25: break
    m.load_state_dict(bs)
    with torch.no_grad(): pick=m(Xte).argmax(1).numpy()
    return correct[te,pick].mean()
SEEDS=[0,1,2,3,4]
print("\n  (3) learned routers (route by predicted reliability, 5 seeds):")
learned={}
for name,X in BANK.items():
    a=[route(X,s) for s in SEEDS]; learned[name]=(float(np.mean(a)),float(np.std(a)))
    print(f"    {name:16s}: {learned[name][0]:.1%} ± {learned[name][1]:.1%}")
shuf=float(np.mean([route(BANK["confidence"],s,shuffle=True) for s in SEEDS]))
print(f"    {'shuffle control':16s}: {shuf:.1%}")

conf_learned=learned["confidence"][0]; geo=learned["geometry(v0.4)"][0]; uni=learned["union"][0]
win = max(best_route[1],conf_learned) > max(acc_best,acc_maj)+0.02 and max(best_route[1],conf_learned)>shuf+0.03
adds= uni > max(conf_learned,geo)+0.01
print(f"\n  => VERDICT: within-model confidence {'BEATS' if win else 'does NOT beat'} best-single/majority "
      f"(best non-learned {best_route[0]} {best_route[1]:.1%}; learned {conf_learned:.1%}; shuffle {shuf:.1%}).")
print(f"     union vs parts: {'confidence ADDS over geometry' if adds else 'no complementary gain (union <= best part)'} "
      f"(conf {conf_learned:.1%}, geo {geo:.1%}, union {uni:.1%}).")
open("RESULTS_v5.md","w",encoding="utf-8").write(
  f"# v0.5 within-model confidence (orthogonal construct)\nroutable n={len(idx)}. best-single {acc_best:.1%}, "
  f"majority {acc_maj:.1%}. non-learned best: {best_route[0]} {best_route[1]:.1%}. learned conf {conf_learned:.1%} "
  f"(shuffle {shuf:.1%}). union {uni:.1%} vs geo {geo:.1%}.\n"
  f"**Confidence {'BEATS' if win else 'does NOT beat'} baselines; union {'ADDS' if adds else 'no add'} over geometry.**\n")
print("  wrote RESULTS_v5.md")
