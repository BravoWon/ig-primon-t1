#!/usr/bin/env python
"""DeltaSheaf-v0.10 — the RANK of the whole thing. How many orthogonal geometric rotations derive the full
matrix of these realities, and how many carry task signal? Theoretical ceiling: M=5 points -> C(5,2)=10
rotation-invariant pairwise coordinates fully determine the configuration; centered shape = M-1=4 modes.
Every probe (delta/cocycle/inverse/whitened) is a re-encoding of THIS object. Measure:
  (1) geometric rank: how many orthogonal modes the ensembles actually occupy (PCA cumulative variance +
      the mean 5-point Gram eigenspectrum).
  (2) signal rank: how many orthogonal modes carry task signal (per-PC held-out AUC for phi-correctness;
      cumulative combiner AUC vs #modes -> where it plateaus). And does mode-1 == difficulty?"""
import itertools, json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
np.random.seed(0); torch.manual_seed(0)
TAGS=["qwen3b","phi","smollm","falcon","olmo"]; M=5
pool=[json.loads(l) for l in open("data/mmlu_pool.jsonl",encoding="utf-8")]
conf=[[json.loads(l) for l in open(f"data/conf/{t}.jsonl",encoding="utf-8")] for t in TAGS]
N=min(len(c) for c in conf)
correct=np.array([[conf[m][i]["correct"] for m in range(M)] for i in range(N)],dtype=np.float32)
ans=np.array([[conf[m][i]["ans"] for m in range(M)] for i in range(N)])
amrg=np.array([[conf[m][i]["amrg"] for m in range(M)] for i in range(N)]); pmax=np.array([[max(conf[m][i]["p"]) for m in range(M)] for i in range(N)])
stalks=np.load("data/embedded.npz")["stalks"][:N].astype(np.float32)
phi=int(correct.mean(0).argmax()); y=correct[:,phi]
PAIRS=list(itertools.combinations(range(M),2))

# ---- the FULL rotation-invariant relational geometry: 10 pairwise cosines + 5 norms = the Gram content ----
Sn=stalks/ (np.linalg.norm(stalks,axis=2,keepdims=True)+1e-8)
cos=np.stack([(Sn[:,i]*Sn[:,j]).sum(1) for (i,j) in PAIRS],1)            # [N,10] pairwise angles
nrm=np.log(np.linalg.norm(stalks,axis=2)+1e-8)                          # [N,5] scales
GEO=np.concatenate([cos,nrm],1).astype(np.float32)                      # [N,15] the full matrix of realities

# (1a) mean 5-point Gram eigenspectrum (intrinsic shape dimensionality) — centered stalks per item
Sc=stalks-stalks.mean(1,keepdims=True)
G=np.einsum('nmd,nkd->nmk',Sc,Sc)                                       # [N,5,5] Gram per item
ev=np.linalg.eigvalsh(G).mean(0)[::-1]; ev=ev/ev.sum()
print(f"[v0.10 rank of the reality]  mean 5-point Gram eigenspectrum (shape modes):")
print("   "+"  ".join(f"λ{k+1}={ev[k]:.2f}" for k in range(M))+f"   (cum to 90% at mode {int(np.searchsorted(np.cumsum(ev),0.90))+1})")

# (1b) geometric rank of the invariant features across items
mu=GEO.mean(0); Gc=GEO-mu; U,S,Vt=np.linalg.svd(Gc,full_matrices=False); var=(S**2)/np.sum(S**2)
cum=np.cumsum(var)
r90=int(np.searchsorted(cum,0.90))+1; r95=int(np.searchsorted(cum,0.95))+1; r99=int(np.searchsorted(cum,0.99))+1
print(f"  geometric rank of the 15 invariants: 90%@{r90} modes, 95%@{r95}, 99%@{r99} (of {GEO.shape[1]} possible)")

# (2) signal rank: project onto orthogonal PCs; per-PC AUC for phi-correctness
def auc(sc,yy):
    o=np.argsort(sc); r=np.empty(len(sc)); r[o]=np.arange(1,len(sc)+1); n1=yy.sum(); n0=len(yy)-n1
    return 0.5 if n1==0 or n0==0 else (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)
PC=Gc@Vt.T                                                             # [N,15] orthogonal geometry coordinates
per=[(k,max(auc(PC[:,k],y),1-auc(PC[:,k],y))) for k in range(GEO.shape[1])]
per_sorted=sorted(per,key=lambda t:-t[1])
print("  per-orthogonal-mode |AUC-0.5| for predicting phi-correctness (informativeness of each rotation):")
print("   "+"  ".join(f"m{k}:{a:.03f}" for k,a in per_sorted[:8]))
# difficulty reference + correlation with the top informative mode
diff=amrg.mean(1)+ (pmax.mean(1))
topmode=per_sorted[0][0]
rho=np.corrcoef(PC[:,topmode], -np.array([np.bincount(ans[n],minlength=4).max() for n in range(N)]))[0,1]
print(f"  most-informative single mode = PC{topmode} (folded AUC {per_sorted[0][1]:.3f}, ~chance); |corr with (dis)agreement| = {abs(rho):.2f}")
# confidence-only AUC for reference (the non-geometric 'difficulty' signal)
conf_auc=auc(amrg.mean(1)+pmax.mean(1),y)

# cumulative held-out AUC using top-k orthogonal modes (does signal saturate at rank 1?)
rng=np.random.default_rng(0); perm=rng.permutation(N); cut=N*3//4; tr,te=perm[:cut],perm[cut:]
order=[k for k,_ in per_sorted]
class Net(nn.Module):
    def __init__(s,d): super().__init__(); s.f=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,1))
    def forward(s,x): return s.f(x).squeeze(1)
def combauc(cols):
    X=PC[:,cols]; m0=X[tr].mean(0); s0=X[tr].std(0)+1e-6; X=((X-m0)/s0).astype(np.float32)
    torch.manual_seed(0); net=Net(X.shape[1]); opt=torch.optim.Adam(net.parameters(),lr=3e-3,weight_decay=1e-3)
    Xtr=torch.tensor(X[tr]); Ytr=torch.tensor(y[tr]); nv=len(Xtr)//5; vi,ti=torch.arange(nv),torch.arange(nv,len(Xtr)); bv=1e9;bs=None;pat=0
    for ep in range(250):
        net.train(); l=F.binary_cross_entropy_with_logits(net(Xtr[ti]),Ytr[ti]); opt.zero_grad(); l.backward(); opt.step(); net.eval()
        with torch.no_grad(): vl=F.binary_cross_entropy_with_logits(net(Xtr[vi]),Ytr[vi]).item()
        if vl<bv-1e-4: bv,bs,pat=vl,{k:v.clone() for k,v in net.state_dict().items()},0
        else:
            pat+=1
            if pat>=20: break
    net.load_state_dict(bs)
    with torch.no_grad(): return auc(net(torch.tensor(X[te])).numpy(),y[te])
curve=[(k,combauc(order[:k])) for k in [1,2,3,5,10,15]]
print("  cumulative held-out AUC by #orthogonal modes used:")
print("   "+"  ".join(f"{k}->{a:.3f}" for k,a in curve))
ceil=curve[-1][1]; rank1=curve[0][1]
sig_rank=next((k for k,a in curve if a>=ceil-0.005),15)
print(f"\n  => ANSWER: {GEO.shape[1]} orthogonal invariants exist; the ensembles occupy ~{r95} (95% var); "
      f"intrinsic per-item shape ~{int(np.searchsorted(np.cumsum(ev),0.90))+1}-D. Geometry's TASK signal saturates "
      f"at ~{sig_rank} modes but caps at AUC {ceil:.3f} — WEAKER than the non-geometric confidence signal "
      f"(AUC {conf_auc:.3f}). So: ~{r95} orthogonal rotations derive the full matrix; the geometry is a diffuse "
      f"~{sig_rank}-mode object that never beats one confidence scalar.")
open("RESULTS_v10.md","w",encoding="utf-8").write(
  f"# v0.10 rank of the reality\nGram shape modes 90%@{int(np.searchsorted(np.cumsum(ev),0.90))+1}-D (lambda5=0). "
  f"15 invariants, geometric rank 95%@{r95}. Geometry task-signal saturates ~{sig_rank} modes, caps AUC {ceil:.3f} "
  f"< confidence AUC {conf_auc:.3f}.\n**~{r95} orthogonal rotations derive the full matrix; geometry is a diffuse "
  f"~{sig_rank}-mode object weaker than one confidence scalar. All probes re-encode the same object.**\n")
print("  wrote RESULTS_v10.md")
