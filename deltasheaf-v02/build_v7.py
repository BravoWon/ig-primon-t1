#!/usr/bin/env python
"""DeltaSheaf-v0.7 — the closure: point v0.2's DEAD geometry at the DETECTION target (predict when the
strongest model phi is wrong), controlling for item DIFFICULTY. Every v0.2 signature — restriction-map
residuals (EDGE, first-order), cycle residuals + AREA-between-maps (higher-order, the H1 obstruction) — was
built and died against GENERATION. Here they predict ERRORS instead. Questions:
  (1) does geometry predict phi-errors ABOVE a rich label-free difficulty control?  (area-between-maps pays?)
  (2) does HIGHER-order (cycle/area) add over FIRST-order (edge)+difficulty?  (the H1 question, at detection)
Held-out AUC, 15 splits, increments with split CIs. Merit: increment over difficulty CI excludes 0."""
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
pmax=P.max(2)
stalks=np.load("data/embedded.npz")["stalks"][:N]
phi=int(correct.mean(0).argmax()); y=correct[:,phi]
PAIRS=list(itertools.combinations(range(M),2)); TRIP=list(itertools.combinations(range(M),3))

# ---- label-free DIFFICULTY control (consensus / spread; no gold, no correctness) ----
def vote_ent(a):
    out=np.zeros(len(a))
    for n in range(len(a)):
        c=np.bincount(a[n],minlength=4)/M; out[n]=-(c*np.log(c+1e-9)).sum()
    return out
agree=np.array([np.bincount(ans[n],minlength=4).max() for n in range(N)])/M
DIFF=np.stack([agree,vote_ent(ans),amrg.mean(1),amrg.max(1),amrg.min(1),amrg.std(1),pmax.mean(1),pmax.min(1)],1).astype(np.float32)
CROSS=np.concatenate([np.concatenate([P[:,m,:],amrg[:,m,None],pmax[:,m,None]],1) for m in range(M) if m!=phi],1).astype(np.float32)

def geom(tr):   # fit PCA + ridge-LS restriction maps on TRAIN; return EDGE (first-order), HIGH (cycle+area)
    Sf=stalks[tr].reshape(-1,stalks.shape[2]); mu=Sf.mean(0)
    _,_,Vt=np.linalg.svd(Sf-mu,full_matrices=False); PR=Vt[:DP].T.astype(np.float32)
    Pr=((stalks-mu)@PR).astype(np.float32)                     # [N,M,DP]
    D={}
    for (i,j) in PAIRS:
        Xi,Xj=Pr[tr,i],Pr[tr,j]
        W=np.linalg.solve(Xi.T@Xi+LAM*np.eye(DP,dtype=np.float32),Xi.T@Xj)
        D[(i,j)]=Pr[:,i]@W-Pr[:,j]                             # restriction-map residual [N,DP]
    edge=np.stack([np.linalg.norm(D[(i,j)],axis=1) for (i,j) in PAIRS],1)
    cyc=[]; area=[]
    for (i,j,k) in TRIP:
        dij,djk,dik=D[(i,j)],D[(j,k)],D[(i,k)]
        cyc.append(np.linalg.norm(dij+djk-dik,axis=1))         # Cech H1 coboundary norm
        a,b=dij,dik
        area.append(0.5*np.sqrt(np.maximum((a*a).sum(1)*(b*b).sum(1)-(a*b).sum(1)**2,0)))  # area between maps
    return edge.astype(np.float32), np.concatenate([np.stack(cyc,1),np.stack(area,1)],1).astype(np.float32)

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

ARMS=["phi_own","difficulty","edge","cycle+area","diff+edge","diff+edge+cyc","diff+cross","all"]
A={k:[] for k in ARMS}
for s in range(15):
    rng=np.random.default_rng(s); perm=rng.permutation(N); cut=N*3//4; tr,te=perm[:cut],perm[cut:]
    EDGE,HIGH=geom(tr)
    A["phi_own"].append(auc(amrg[te,phi],y[te]))
    A["difficulty"].append(cauc(DIFF,tr,te,s))
    A["edge"].append(cauc(EDGE,tr,te,s))
    A["cycle+area"].append(cauc(HIGH,tr,te,s))
    A["diff+edge"].append(cauc(np.concatenate([DIFF,EDGE],1),tr,te,s))
    A["diff+edge+cyc"].append(cauc(np.concatenate([DIFF,EDGE,HIGH],1),tr,te,s))
    A["diff+cross"].append(cauc(np.concatenate([DIFF,CROSS],1),tr,te,s))
    A["all"].append(cauc(np.concatenate([DIFF,EDGE,HIGH,CROSS],1),tr,te,s))
A={k:np.array(v) for k,v in A.items()}
def ci(x): return np.percentile(x,[2.5,97.5])
print(f"[v0.7 geometry@detection]  target=predict {TAGS[phi]}'s errors (held-out AUC, 15 splits):")
for k in ARMS: print(f"    {k:16s} AUC {A[k].mean():.3f} ± {A[k].std():.3f}")
def inc(hi,lo,name):
    d=A[hi]-A[lo]; c=ci(d)
    print(f"    {name:34s} Δ {d.mean():+.3f}  95%CI[{c[0]:+.3f},{c[1]:+.3f}]  frac>0={np.mean(d>0):.0%}  {'PAYS' if c[0]>0 else 'ns'}")
    return c[0]>0
print("\n  increments:")
g1=inc("diff+edge","difficulty","first-order geometry OVER difficulty")
g2=inc("diff+edge+cyc","diff+edge","HIGHER-order (cycle+AREA) OVER first-order+diff")
g3=inc("diff+edge+cyc","difficulty","ALL geometry OVER difficulty")
c1=inc("diff+cross","difficulty","cross-model confidence OVER difficulty")
gp=inc("diff+edge+cyc","phi_own","geometry-detector OVER phi's own confidence")
print(f"\n  => VERDICT: v0.2 geometry {'PAYS at detection (area-between-maps finally earns its keep)' if g3 else 'still null even at detection (it was the difficulty proxy)'}; "
      f"higher-order {'adds over first-order' if g2 else 'inert vs first-order (H1 still flat)'}; "
      f"cross-model confidence {'pays over difficulty' if c1 else 'reduces to difficulty'}.")
open("RESULTS_v7.md","w",encoding="utf-8").write(
  f"# v0.7 geometry @ detection\nAUC: difficulty {A['difficulty'].mean():.3f}, edge {A['edge'].mean():.3f}, "
  f"cycle+area {A['cycle+area'].mean():.3f}, diff+edge+cyc {A['diff+edge+cyc'].mean():.3f}, all {A['all'].mean():.3f}, phi_own {A['phi_own'].mean():.3f}.\n"
  f"geometry-over-difficulty Δ{(A['diff+edge+cyc']-A['difficulty']).mean():+.3f} CI[{ci(A['diff+edge+cyc']-A['difficulty'])[0]:+.3f},{ci(A['diff+edge+cyc']-A['difficulty'])[1]:+.3f}]; "
  f"higher-over-first Δ{(A['diff+edge+cyc']-A['diff+edge']).mean():+.3f}; cross-over-diff Δ{(A['diff+cross']-A['difficulty']).mean():+.3f}.\n"
  f"**geometry {'PAYS' if g3 else 'null'} at detection; higher-order {'adds' if g2 else 'inert'}; cross-conf {'pays' if c1 else '=difficulty'}.**\n")
print("  wrote RESULTS_v7.md")
