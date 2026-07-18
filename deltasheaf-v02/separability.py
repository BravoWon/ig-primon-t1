#!/usr/bin/env python
"""Gemini claim 2/3 test — do 0-of-5 BLIND items (shared delusion) and 5-of-5 CLEAN items (truth) form
linearly-separable classes in the UNCOLLAPSED pairwise-residual space (the 5120-D "twisted cohomology"),
ABOVE a label-free difficulty baseline? If geometry adds nothing over difficulty, the "distinct cohomology
classes" are just the difficulty extremes (0-of-5 = hardest, 5-of-5 = easiest), not a geometric seam.
Held-out AUC, 15 splits. CPU."""
import itertools, json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
np.random.seed(0); torch.manual_seed(0)
TAGS=["qwen3b","phi","smollm","falcon","olmo"]; M=5; PAIRS=list(itertools.combinations(range(M),2))
z=np.load("data/embedded.npz"); stalks=z["stalks"]; N=stalks.shape[0]
conf=[[json.loads(l) for l in open(f"data/conf/{t}.jsonl",encoding="utf-8")] for t in TAGS]
Nc=min(len(c) for c in conf); N=min(N,Nc); stalks=stalks[:N]
correct=np.array([[conf[m][i]["correct"] for m in range(M)] for i in range(N)],dtype=np.float32)
ans=np.array([[conf[m][i]["ans"] for m in range(M)] for i in range(N)])
amrg=np.array([[conf[m][i]["amrg"] for m in range(M)] for i in range(N)],dtype=np.float32)
pmax=np.array([[max(conf[m][i]["p"]) for m in range(M)] for i in range(N)],dtype=np.float32)

s=correct.sum(1)
blind=np.where(s==0)[0]; clean=np.where(s==5)[0]
idx=np.concatenate([blind,clean]); y=np.concatenate([np.zeros(len(blind)),np.ones(len(clean))]).astype(np.float32)
print(f"[separability]  blind(0-of-5)={len(blind)}  clean(5-of-5)={len(clean)}  total={len(idx)}")

# UNCOLLAPSED geometry: all 10 pairwise stalk deltas, 512 each = 5120-D
GEO=np.concatenate([stalks[idx,i]-stalks[idx,j] for (i,j) in PAIRS],1).astype(np.float32)   # [n,5120]
# label-free DIFFICULTY baseline
agree=np.array([np.bincount(ans[i],minlength=4).max() for i in idx])/M
vent=np.array([-(np.bincount(ans[i],minlength=4)/M*np.log(np.bincount(ans[i],minlength=4)/M+1e-9)).sum() for i in idx])
DIFF=np.stack([agree,vent,amrg[idx].mean(1),amrg[idx].std(1),pmax[idx].mean(1),pmax[idx].min(1)],1).astype(np.float32)

def auc(sc,yy):
    o=np.argsort(sc); r=np.empty(len(sc)); r[o]=np.arange(1,len(sc)+1); n1=yy.sum(); n0=len(yy)-n1
    return 0.5 if n1==0 or n0==0 else (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)
class Net(nn.Module):
    def __init__(s,d): super().__init__(); s.f=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Dropout(0.2),nn.Linear(64,1))
    def forward(s,x): return s.f(x).squeeze(1)
def cauc(X,tr,te,seed,pca=None):
    if pca:
        m0=X[tr].mean(0); Xc=X-m0; _,_,Vt=np.linalg.svd(Xc[tr],full_matrices=False); X=(Xc@Vt[:pca].T)
    m0=X[tr].mean(0); s0=X[tr].std(0)+1e-6; Xn=((X-m0)/s0).astype(np.float32)
    torch.manual_seed(seed); net=Net(Xn.shape[1]); opt=torch.optim.Adam(net.parameters(),lr=3e-3,weight_decay=2e-3)
    Xtr=torch.tensor(Xn[tr]); Ytr=torch.tensor(y[tr]); nv=len(Xtr)//5; vi,ti=torch.arange(nv),torch.arange(nv,len(Xtr)); bv=1e9;bs=None;pat=0
    for ep in range(250):
        net.train(); l=F.binary_cross_entropy_with_logits(net(Xtr[ti]),Ytr[ti]); opt.zero_grad(); l.backward(); opt.step(); net.eval()
        with torch.no_grad(): vl=F.binary_cross_entropy_with_logits(net(Xtr[vi]),Ytr[vi]).item()
        if vl<bv-1e-4: bv,bs,pat=vl,{k:v.clone() for k,v in net.state_dict().items()},0
        else:
            pat+=1
            if pat>=20: break
    net.load_state_dict(bs)
    with torch.no_grad(): return auc(net(torch.tensor(Xn[te])).numpy(),y[te])

A={k:[] for k in ["difficulty","geometry(5120→64)","geom+diff"]}
for sd in range(15):
    rng=np.random.default_rng(sd); perm=rng.permutation(len(idx)); cut=len(idx)*3//4; tr,te=perm[:cut],perm[cut:]
    A["difficulty"].append(cauc(DIFF,tr,te,sd))
    A["geometry(5120→64)"].append(cauc(GEO,tr,te,sd,pca=64))
    A["geom+diff"].append(cauc(np.concatenate([DIFF, (GEO-GEO[tr].mean(0))@np.linalg.svd(GEO[tr]-GEO[tr].mean(0),full_matrices=False)[2][:64].T],1),tr,te,sd))
A={k:np.array(v) for k,v in A.items()}
for k in A: print(f"    {k:20s} AUC {A[k].mean():.3f} ± {A[k].std():.3f}")
d=A["geom+diff"]-A["difficulty"]; lo,hi=np.percentile(d,[2.5,97.5])
print(f"\n  geometry increment OVER difficulty: Δ {d.mean():+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  frac>0={np.mean(d>0):.0%}")
pays=lo>0
print(f"  => {'GEOMETRIC SEAM REAL — uncollapsed residual separates delusion from truth beyond difficulty (Gemini claim CONFIRMED)' if pays else 'NO geometric seam beyond difficulty — the two classes are the difficulty extremes, not a distinct cohomology (claim refuted; geometry ≈ difficulty)'}")
open("RESULTS_separability.md","w",encoding="utf-8").write(
  f"# blind-vs-clean separability (Gemini cohomology claim)\nblind {len(blind)}, clean {len(clean)}. "
  f"difficulty {A['difficulty'].mean():.3f}, geometry {A['geometry(5120→64)'].mean():.3f}, geom+diff {A['geom+diff'].mean():.3f}. "
  f"increment {d.mean():+.3f} CI[{lo:+.3f},{hi:+.3f}].\n**{'geometric seam real' if pays else 'no seam beyond difficulty'}.**\n")
print("  wrote RESULTS_separability.md")
