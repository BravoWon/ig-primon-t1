#!/usr/bin/env python
"""DeltaSheaf-v0.8 — "geometry in the inverse." v7 used only the FORWARD residual R_ij*s_i - s_j (one
direction per edge). This adds the discarded half: the REVERSE map R_ji, the ROUND-TRIP non-invertibility
(||R_ij R_ji s_j - s_j||, how far there-and-back is from identity, PER ITEM), the forward<->inverse ASYMMETRY,
and the map's own local GAIN. Target = detection (predict phi's errors), difficulty-controlled, held-out AUC
15 splits (same harness as v7). Merit: inverse/asymmetry features add over difficulty AND over forward, CI>0.
Prior: LOW (forward geometry null at generation v0.2 and detection v0.7) — but this direction is untested."""
import itertools, json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
np.random.seed(0); torch.manual_seed(0)
TAGS=["qwen3b","phi","smollm","falcon","olmo"]; M=5; DP=64; LAM=10.0
pool=[json.loads(l) for l in open("data/mmlu_pool.jsonl",encoding="utf-8")]
conf=[[json.loads(l) for l in open(f"data/conf/{t}.jsonl",encoding="utf-8")] for t in TAGS]
N=min(len(c) for c in conf)
correct=np.array([[conf[m][i]["correct"] for m in range(M)] for i in range(N)],dtype=np.float32)
ans =np.array([[conf[m][i]["ans"] for m in range(M)] for i in range(N)])
P   =np.array([[conf[m][i]["p"]   for m in range(M)] for i in range(N)],dtype=np.float32)
amrg=np.array([[conf[m][i]["amrg"] for m in range(M)] for i in range(N)],dtype=np.float32)
pmax=P.max(2); stalks=np.load("data/embedded.npz")["stalks"][:N]
phi=int(correct.mean(0).argmax()); y=correct[:,phi]
PAIRS=list(itertools.combinations(range(M),2))
def vote_ent(a):
    out=np.zeros(len(a))
    for n in range(len(a)): c=np.bincount(a[n],minlength=4)/M; out[n]=-(c*np.log(c+1e-9)).sum()
    return out
agree=np.array([np.bincount(ans[n],minlength=4).max() for n in range(N)])/M
DIFF=np.stack([agree,vote_ent(ans),amrg.mean(1),amrg.max(1),amrg.min(1),amrg.std(1),pmax.mean(1),pmax.min(1)],1).astype(np.float32)

def geom(tr):   # returns FWD (forward residual norms) and INV (reverse + round-trip + asymmetry + gain)
    Sf=stalks[tr].reshape(-1,stalks.shape[2]); mu=Sf.mean(0)
    _,_,Vt=np.linalg.svd(Sf-mu,full_matrices=False); PR=Vt[:DP].T.astype(np.float32)
    Pr=((stalks-mu)@PR).astype(np.float32)
    fwd=[]; rev=[]; rt=[]; asym=[]; gain=[]
    for (i,j) in PAIRS:
        Xi,Xj=Pr[tr,i],Pr[tr,j]
        Wij=np.linalg.solve(Xi.T@Xi+LAM*np.eye(DP,dtype=np.float32),Xi.T@Xj)   # i->j
        Wji=np.linalg.solve(Xj.T@Xj+LAM*np.eye(DP,dtype=np.float32),Xj.T@Xi)   # j->i (the inverse direction)
        dfwd=Pr[:,i]@Wij-Pr[:,j]                       # forward residual (space j)
        drev=Pr[:,j]@Wji-Pr[:,i]                       # REVERSE residual (space i) — discarded in v7
        rtj =(Pr[:,j]@Wji)@Wij-Pr[:,j]                 # round-trip R_ij(R_ji(s_j)) - s_j : non-invertibility
        rti =(Pr[:,i]@Wij)@Wji-Pr[:,i]
        nf=np.linalg.norm(dfwd,axis=1); nr=np.linalg.norm(drev,axis=1)
        fwd.append(nf); rev.append(nr)
        rt.append(np.linalg.norm(rtj,axis=1)); rt.append(np.linalg.norm(rti,axis=1))
        asym.append(np.abs(nf-nr))                     # forward<->inverse asymmetry
        gain.append(np.linalg.norm(Pr[:,i]@Wij,axis=1)/(np.linalg.norm(Pr[:,i],axis=1)+1e-6))
        gain.append(np.linalg.norm(Pr[:,j]@Wji,axis=1)/(np.linalg.norm(Pr[:,j],axis=1)+1e-6))
    FWD=np.stack(fwd,1).astype(np.float32)
    INV=np.concatenate([np.stack(rev,1),np.stack(rt,1),np.stack(asym,1),np.stack(gain,1)],1).astype(np.float32)
    return FWD,INV

def auc(sc,yy):
    o=np.argsort(sc); r=np.empty(len(sc)); r[o]=np.arange(1,len(sc)+1); n1=yy.sum(); n0=len(yy)-n1
    return 0.5 if n1==0 or n0==0 else (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)
class Net(nn.Module):
    def __init__(s,d): super().__init__(); s.f=nn.Sequential(nn.Linear(d,96),nn.GELU(),nn.Dropout(0.1),nn.Linear(96,1))
    def forward(s,x): return s.f(x).squeeze(1)
def cauc(FEAT,tr,te,seed):
    m0=FEAT[tr].mean(0); s0=FEAT[tr].std(0)+1e-6; X=((FEAT-m0)/s0).astype(np.float32)
    torch.manual_seed(seed); net=Net(X.shape[1]); opt=torch.optim.Adam(net.parameters(),lr=3e-3,weight_decay=1e-3)
    Xtr=torch.tensor(X[tr]); Ytr=torch.tensor(y[tr]); nv=len(Xtr)//5; vi,ti=torch.arange(nv),torch.arange(nv,len(Xtr)); bv=1e9;bs=None;pat=0
    for ep in range(300):
        net.train(); l=F.binary_cross_entropy_with_logits(net(Xtr[ti]),Ytr[ti]); opt.zero_grad(); l.backward(); opt.step(); net.eval()
        with torch.no_grad(): vl=F.binary_cross_entropy_with_logits(net(Xtr[vi]),Ytr[vi]).item()
        if vl<bv-1e-4: bv,bs,pat=vl,{k:v.clone() for k,v in net.state_dict().items()},0
        else:
            pat+=1
            if pat>=25: break
    net.load_state_dict(bs)
    with torch.no_grad(): return auc(net(torch.tensor(X[te])).numpy(),y[te])

ARMS=["difficulty","inverse_only","diff+fwd","diff+inv","diff+fwd+inv"]; A={k:[] for k in ARMS}
for s in range(15):
    rng=np.random.default_rng(s); perm=rng.permutation(N); cut=N*3//4; tr,te=perm[:cut],perm[cut:]
    FWD,INV=geom(tr)
    A["difficulty"].append(cauc(DIFF,tr,te,s))
    A["inverse_only"].append(cauc(INV,tr,te,s))
    A["diff+fwd"].append(cauc(np.concatenate([DIFF,FWD],1),tr,te,s))
    A["diff+inv"].append(cauc(np.concatenate([DIFF,INV],1),tr,te,s))
    A["diff+fwd+inv"].append(cauc(np.concatenate([DIFF,FWD,INV],1),tr,te,s))
A={k:np.array(v) for k,v in A.items()}
def ci(x): return np.percentile(x,[2.5,97.5])
print(f"[v0.8 geometry-in-the-inverse @ detection]  target=predict {TAGS[phi]}'s errors (held-out AUC, 15 splits):")
for k in ARMS: print(f"    {k:16s} AUC {A[k].mean():.3f} ± {A[k].std():.3f}")
def inc(hi,lo,name):
    d=A[hi]-A[lo]; c=ci(d)
    print(f"    {name:40s} Δ {d.mean():+.3f}  95%CI[{c[0]:+.3f},{c[1]:+.3f}]  frac>0={np.mean(d>0):.0%}  {'PAYS' if c[0]>0 else 'ns'}")
    return c[0]>0
print("\n  increments:")
i1=inc("diff+inv","difficulty","inverse/asymmetry OVER difficulty")
i2=inc("diff+fwd+inv","diff+fwd","inverse OVER (difficulty+forward)")
raw=A["inverse_only"].mean()
print(f"    inverse-only raw AUC {raw:.3f} (0.5=chance)")
print(f"\n  => VERDICT: geometry-in-the-inverse {'ADDS real signal (the map direction matters!)' if (i1 or i2) else 'null too — the inverse/non-invertibility carries nothing beyond difficulty (whole map, both directions, is inert)'}.")
open("RESULTS_v8.md","w",encoding="utf-8").write(
  f"# v0.8 geometry in the inverse @ detection\nAUC: difficulty {A['difficulty'].mean():.3f}, inverse_only {raw:.3f}, "
  f"diff+fwd {A['diff+fwd'].mean():.3f}, diff+inv {A['diff+inv'].mean():.3f}, diff+fwd+inv {A['diff+fwd+inv'].mean():.3f}.\n"
  f"inverse-over-difficulty Δ{(A['diff+inv']-A['difficulty']).mean():+.3f} CI{list(np.round(ci(A['diff+inv']-A['difficulty']),3))}; "
  f"inverse-over-forward Δ{(A['diff+fwd+inv']-A['diff+fwd']).mean():+.3f}.\n"
  f"**geometry-in-the-inverse {'ADDS' if (i1 or i2) else 'null'} — map direction/non-invertibility {'matters' if (i1 or i2) else 'inert'}.**\n")
print("  wrote RESULTS_v8.md")
