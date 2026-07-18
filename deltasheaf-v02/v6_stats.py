#!/usr/bin/env python
"""DeltaSheaf-v0.6 significance recurse. Harden the two borderline positives against the v5 failure mode.
 FUSION: I tried 3 weightings and reported the best -> multiple-comparison risk. Fix: HELD-OUT weighting
   selection — pick the best weighting on train-half, evaluate on test-half, over 20 splits. Real gain must
   survive out-of-sample selection. Plus bootstrap CI on the full-set logit-margin fusion vs best-single.
 SELECTIVE: is the learned-combiner's +0.021 AUC over phi-own stable across splits, or split luck?"""
import json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
try: from scipy.stats import binomtest; HAVE_SCIPY=True
except Exception: HAVE_SCIPY=False
TAGS=["qwen3b","phi","smollm","falcon","olmo"]; M=5
pool=[json.loads(l) for l in open("data/mmlu_pool.jsonl",encoding="utf-8")]
conf=[[json.loads(l) for l in open(f"data/conf/{t}.jsonl",encoding="utf-8")] for t in TAGS]
N=min(len(c) for c in conf)
gold=np.array([pool[i]["answer"] for i in range(N)])
correct=np.array([[conf[m][i]["correct"] for m in range(M)] for i in range(N)],dtype=np.float32)
ans =np.array([[conf[m][i]["ans"] for m in range(M)] for i in range(N)])
P   =np.array([[conf[m][i]["p"]   for m in range(M)] for i in range(N)],dtype=np.float32)
amrg=np.array([[conf[m][i]["amrg"] for m in range(M)] for i in range(N)],dtype=np.float32)
pmax=P.max(2)
phi=int(correct.mean(0).argmax())
def mcnemar(a,b):
    n01=int(np.sum(a&~b)); n10=int(np.sum(~a&b))
    if n01+n10==0: return 1.0
    if HAVE_SCIPY: return binomtest(min(n01,n10),n01+n10,0.5).pvalue
    from math import comb; k,n=min(n01,n10),n01+n10
    return min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n)

# ---------- FUSION: held-out weighting selection ----------
recov=correct.sum(1)>=1; R=np.where(recov)[0]
def softmax_w(x,T): e=np.exp((x-x.max(1,keepdims=True))/T); return e/e.sum(1,keepdims=True)
WEIGHTS={"soft":np.ones_like(pmax),"pmax":pmax,"amrg_clamp":np.maximum(amrg,0.0),
         "amrg_sm2":softmax_w(amrg,2),"amrg_sm4":softmax_w(amrg,4)}
def fuse_acc(w,items):
    p=(w[items][...,None]*P[items]).sum(1).argmax(1); return (p==gold[items]).mean(), (p==gold[items])
diffs=[]; sigs=[]; chosen={}
for s in range(20):
    rng=np.random.default_rng(1000+s); pr=rng.permutation(R); h=len(R)//2; A,Bset=pr[:h],pr[h:]
    bestw=max(WEIGHTS, key=lambda k: fuse_acc(WEIGHTS[k],A)[0])       # pick weighting on half A
    chosen[bestw]=chosen.get(bestw,0)+1
    accB,okB=fuse_acc(WEIGHTS[bestw],Bset); base_ok=correct[Bset,phi].astype(bool)
    diffs.append(accB-correct[Bset,phi].mean()); sigs.append(mcnemar(okB,base_ok)<0.05)
diffs=np.array(diffs); lo,hi=np.percentile(diffs,[2.5,97.5])
print("FUSION — held-out weighting selection (pick on half A, score on half B), 20 splits:")
print(f"  Δ(fusion - best-single) mean {diffs.mean():+.2%}  95%CI[{lo:+.2%},{hi:+.2%}]  frac>0={np.mean(diffs>0):.0%}  frac sig={np.mean(sigs):.0%}")
print(f"  weighting chosen out-of-sample: {chosen}")
# full-set bootstrap on the specific logit-margin weighting (reference, uncorrected)
w=np.maximum(amrg,0.0); okF=((w[R][...,None]*P[R]).sum(1).argmax(1)==gold[R]); okB=correct[R,phi].astype(bool)
bd=[]
for _ in range(3000):
    bi=np.random.randint(0,len(R),len(R)); bd.append(okF[bi].mean()-okB[bi].mean())
blo,bhi=np.percentile(bd,[2.5,97.5])
print(f"  [ref] full-set logit-margin fusion vs best-single: Δ{okF.mean()-okB.mean():+.2%} boot95%CI[{blo:+.2%},{bhi:+.2%}] (uncorrected, 1-of-5 weightings)")

# ---------- SELECTIVE: learned-combiner AUC advantage stability ----------
def auc(sc,y):
    o=np.argsort(sc); r=np.empty(len(sc)); r[o]=np.arange(1,len(sc)+1); n1=y.sum(); n0=len(y)-n1
    return 0.5 if n1==0 or n0==0 else (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
FEAT=np.concatenate([P.reshape(N,-1),amrg,pmax],1).astype(np.float32); y=correct[:,phi]
class Net(nn.Module):
    def __init__(s,d): super().__init__(); s.f=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,1))
    def forward(s,x): return s.f(x).squeeze(1)
adv=[]
for s in range(15):
    rng=np.random.default_rng(s); perm=rng.permutation(N); cut=N*3//4; tr,te=perm[:cut],perm[cut:]
    m0=FEAT[tr].mean(0); s0=FEAT[tr].std(0)+1e-6; X=((FEAT-m0)/s0).astype(np.float32)
    torch.manual_seed(s); net=Net(X.shape[1]); opt=torch.optim.Adam(net.parameters(),lr=3e-3,weight_decay=1e-3)
    Xtr=torch.tensor(X[tr]); Ytr=torch.tensor(y[tr]); nv=len(Xtr)//5; vi,ti=torch.arange(nv),torch.arange(nv,len(Xtr)); bv=1e9;bs=None;pat=0
    for ep in range(300):
        net.train(); l=F.binary_cross_entropy_with_logits(net(Xtr[ti]),Ytr[ti]); opt.zero_grad(); l.backward(); opt.step(); net.eval()
        with torch.no_grad(): vl=F.binary_cross_entropy_with_logits(net(Xtr[vi]),Ytr[vi]).item()
        if vl<bv-1e-4: bv,bs,pat=vl,{k:v.clone() for k,v in net.state_dict().items()},0
        else:
            pat+=1
            if pat>=25: break
    net.load_state_dict(bs)
    with torch.no_grad(): sc=net(torch.tensor(X[te])).numpy()
    adv.append(auc(sc,y[te])-auc(amrg[te,phi],y[te]))
adv=np.array(adv); alo,ahi=np.percentile(adv,[2.5,97.5])
print(f"\nSELECTIVE — learned combiner AUC advantage over phi-own, 15 splits:")
print(f"  ΔAUC mean {adv.mean():+.3f}  95%CI[{alo:+.3f},{ahi:+.3f}]  frac>0={np.mean(adv>0):.0%}")

fuse_real = lo>0 and np.mean(sigs)>=0.5
sel_real  = alo>0
print(f"\n=> HARDENED VERDICT: fusion {'REAL (survives out-of-sample weighting selection)' if fuse_real else 'FRAGILE (does not survive held-out selection)'}; "
      f"selective {'REAL (stable AUC gain)' if sel_real else 'FRAGILE (split-dependent)'}.")
open("RESULTS_v6_stats.md","w",encoding="utf-8").write(
  f"# v0.6 hardening\nFUSION held-out selection: Δ{diffs.mean():+.2%} 95%CI[{lo:+.2%},{hi:+.2%}] frac-sig {np.mean(sigs):.0%} "
  f"(chosen {chosen}). SELECTIVE ΔAUC {adv.mean():+.3f} 95%CI[{alo:+.3f},{ahi:+.3f}].\n"
  f"**fusion {'REAL' if fuse_real else 'fragile'}; selective {'REAL' if sel_real else 'fragile'}.**\n")
print("  wrote RESULTS_v6_stats.md")
