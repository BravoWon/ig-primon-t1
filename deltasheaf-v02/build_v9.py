#!/usr/bin/env python
"""DeltaSheaf-v0.9 — the inverse of the CHANCE geometry on the FULL cocycle complex (the user's real "inverse").
v7/v8 used raw cocycle NORMS -> dominated by overall magnitude = difficulty = the chance common-mode, so they
washed out. Here:
  (1) "geometry of chance": estimate the null distribution of the FULL cochain vectors (all 10 edge deltas +
      all 10 triangle coboundaries) by permuting the cross-model registration (random correspondence). -> mu0, Sigma0.
  (2) "the inverse is a geometric subclass": WHITEN observed cochains by Sigma0^{-1/2} (the inverse of the chance
      geometry) -> deviation-from-chance coordinates, with the difficulty common-mode divided OUT.
  (3) test at detection (predict phi's errors), difficulty-controlled, held-out AUC 15 splits. Fairest test yet.
Everything fit on TRAIN only. Merit: whitened-cochain increment OVER difficulty, CI excludes 0."""
import itertools, json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
np.random.seed(0); torch.manual_seed(0)
TAGS=["qwen3b","phi","smollm","falcon","olmo"]; M=5; DP=16; LAM=10.0; KSH=10
pool=[json.loads(l) for l in open("data/mmlu_pool.jsonl",encoding="utf-8")]
conf=[[json.loads(l) for l in open(f"data/conf/{t}.jsonl",encoding="utf-8")] for t in TAGS]
N=min(len(c) for c in conf)
correct=np.array([[conf[m][i]["correct"] for m in range(M)] for i in range(N)],dtype=np.float32)
ans =np.array([[conf[m][i]["ans"] for m in range(M)] for i in range(N)])
P   =np.array([[conf[m][i]["p"]   for m in range(M)] for i in range(N)],dtype=np.float32)
amrg=np.array([[conf[m][i]["amrg"] for m in range(M)] for i in range(N)],dtype=np.float32)
pmax=P.max(2); stalks=np.load("data/embedded.npz")["stalks"][:N].astype(np.float32)
phi=int(correct.mean(0).argmax()); y=correct[:,phi]
PAIRS=list(itertools.combinations(range(M),2)); TRIP=list(itertools.combinations(range(M),3))
def vote_ent(a):
    o=np.zeros(len(a))
    for n in range(len(a)): c=np.bincount(a[n],minlength=4)/M; o[n]=-(c*np.log(c+1e-9)).sum()
    return o
agree=np.array([np.bincount(ans[n],minlength=4).max() for n in range(N)])/M
DIFF=np.stack([agree,vote_ent(ans),amrg.mean(1),amrg.max(1),amrg.min(1),amrg.std(1),pmax.mean(1),pmax.min(1)],1).astype(np.float32)

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

def whitened_cochain(tr):
    Sf=stalks[tr].reshape(-1,stalks.shape[2]); mu=Sf.mean(0)
    _,_,Vt=np.linalg.svd(Sf-mu,full_matrices=False); PR=Vt[:DP].T.astype(np.float32)
    Pr=((stalks-mu)@PR).astype(np.float32)                       # [N,M,DP]
    W={}
    for (i,j) in PAIRS:
        Xi,Xj=Pr[tr,i],Pr[tr,j]; W[(i,j)]=np.linalg.solve(Xi.T@Xi+LAM*np.eye(DP,dtype=np.float32),Xi.T@Xj)
    def cochain(Prm):
        E=[Prm[:,i]@W[(i,j)]-Prm[:,j] for (i,j) in PAIRS]         # 10 edge deltas
        T=[]
        for (i,j,k) in TRIP:
            T.append((Prm[:,i]@W[(i,j)]-Prm[:,j])+(Prm[:,j]@W[(j,k)]-Prm[:,k])-(Prm[:,i]@W[(i,k)]-Prm[:,k]))
        return np.concatenate([np.concatenate(E,1),np.concatenate(T,1)],1)   # [n, 20*DP]
    X=cochain(Pr)                                                # observed [N, 320]
    # (1) geometry of chance: permute cross-model registration on TRAIN
    rng=np.random.default_rng(12345); Ptr=Pr[tr]; nulls=[]
    for _ in range(KSH):
        Psh=Ptr.copy()
        for m in range(M): Psh[:,m]=Ptr[rng.permutation(len(tr)),m]
        nulls.append(cochain(Psh))
    Xn=np.concatenate(nulls,0); mu0=Xn.mean(0); Xc=Xn-mu0
    S0=(Xc.T@Xc)/len(Xc); S0+=(np.trace(S0)/S0.shape[0])*0.10*np.eye(S0.shape[0],dtype=np.float32)  # shrinkage
    w,U=np.linalg.eigh(S0); Wh=(U*(1.0/np.sqrt(np.maximum(w,1e-8))))@U.T    # Sigma0^{-1/2} = inverse chance geometry
    Z=((X-mu0)@Wh).astype(np.float32)                            # (2) whitened observed [N,320]
    ed=20*DP//2                                                  # edges span first 10*DP dims
    D2=(Z**2).sum(1); D2e=(Z[:,:10*DP]**2).sum(1); D2t=(Z[:,10*DP:]**2).sum(1)   # Mahalanobis (total/edge/tri)
    muZ=Z[tr].mean(0); _,_,VtZ=np.linalg.svd(Z[tr]-muZ,full_matrices=False); Ppca=VtZ[:32].T
    Zp=((Z-muZ)@Ppca).astype(np.float32)                         # structured deviation directions
    chance_dev=D2.mean()/Z.shape[1]                              # ~1 if observed == chance; >1 if deviates
    return np.concatenate([np.stack([D2,D2e,D2t],1),Zp],1).astype(np.float32), chance_dev

ARMS=["difficulty","whitened_only","diff+whitened"]; A={k:[] for k in ARMS}; devs=[]
for s in range(15):
    rng=np.random.default_rng(s); perm=rng.permutation(N); cut=N*3//4; tr,te=perm[:cut],perm[cut:]
    WHIT,dev=whitened_cochain(tr); devs.append(dev)
    A["difficulty"].append(cauc(DIFF,tr,te,s))
    A["whitened_only"].append(cauc(WHIT,tr,te,s))
    A["diff+whitened"].append(cauc(np.concatenate([DIFF,WHIT],1),tr,te,s))
A={k:np.array(v) for k,v in A.items()}
def ci(x): return np.percentile(x,[2.5,97.5])
print(f"[v0.9 inverse-chance-geometry on full cocycle complex @ detection]  target=predict {TAGS[phi]}'s errors:")
print(f"  observed cochain deviates from chance by {np.mean(devs):.2f}x isotropic (1.0=identical to chance)")
for k in ARMS: print(f"    {k:16s} AUC {A[k].mean():.3f} ± {A[k].std():.3f}")
d=A["diff+whitened"]-A["difficulty"]; c=ci(d)
print(f"\n  whitened-cochain OVER difficulty:  Δ {d.mean():+.3f}  95%CI[{c[0]:+.3f},{c[1]:+.3f}]  frac>0={np.mean(d>0):.0%}  {'PAYS' if c[0]>0 else 'ns'}")
print(f"  whitened-only raw AUC {A['whitened_only'].mean():.3f} (0.5=chance)")
pays=c[0]>0
print(f"\n  => VERDICT: inverse-chance geometry on the full cocycle complex "
      f"{'PAYS — structured non-chance signal exists once you divide the chance geometry out!' if pays else 'still null — even whitened against chance across the whole complex, the cocycles carry nothing beyond difficulty'}.")
open("RESULTS_v9.md","w",encoding="utf-8").write(
  f"# v0.9 inverse-chance geometry, full cocycle complex @ detection\nobserved deviates from chance {np.mean(devs):.2f}x. "
  f"AUC difficulty {A['difficulty'].mean():.3f}, whitened_only {A['whitened_only'].mean():.3f}, diff+whitened {A['diff+whitened'].mean():.3f}. "
  f"whitened-over-difficulty Δ{d.mean():+.3f} CI[{c[0]:+.3f},{c[1]:+.3f}].\n"
  f"**{'PAYS' if pays else 'null'} — full whitened cocycle complex {'adds structured signal' if pays else 'inert beyond difficulty'}.**\n")
print("  wrote RESULTS_v9.md")
