#!/usr/bin/env python
"""DeltaSheaf-v0.6 ablation — the decisive recurse (per verifier). Claim A's selective-prediction gain used
phi's OWN multi-signal confidence features, so it may be phi RECALIBRATING ITSELF, not the ENSEMBLE adding
anything. Split the gain: (i) phi-self-combiner - phi-single-amrg = recalibration; (ii) all-combiner -
phi-self-combiner = the CROSS-MODEL increment (the only DeltaSheaf-relevant quantity). If (ii)'s CI includes
0, the between-model relation still carries nothing and the 'win' is single-model calibration.
Also: fix Claim B baseline (verify phi is strongest on routable) + Bonferroni-correct the 5 weightings."""
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
rnll=np.array([[conf[m][i]["rnll"] for m in range(M)] for i in range(N)],dtype=np.float32)
rmax=np.array([[conf[m][i]["rmax"] for m in range(M)] for i in range(N)],dtype=np.float32)
pmax=P.max(2); negent=(P*np.log(P+1e-9)).sum(2)
phi=int(correct.mean(0).argmax())
y=correct[:,phi]
def block(m): return np.concatenate([P[:,m,:],amrg[:,m,None],rnll[:,m,None],rmax[:,m,None],pmax[:,m,None],negent[:,m,None]],1)
F_phi   = block(phi).astype(np.float32)                                   # 9 phi-only self features
F_all   = np.concatenate([block(m) for m in range(M)],1).astype(np.float32) # 45 all-model
F_others= np.concatenate([block(m) for m in range(M) if m!=phi],1).astype(np.float32) # 36 others-only
def auc(sc,yy):
    o=np.argsort(sc); r=np.empty(len(sc)); r[o]=np.arange(1,len(sc)+1); n1=yy.sum(); n0=len(yy)-n1
    return 0.5 if n1==0 or n0==0 else (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)
class Net(nn.Module):
    def __init__(s,d): super().__init__(); s.f=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(d if False else 128,1))
    def forward(s,x): return s.f(x).squeeze(1)
def combiner_auc(FEAT,seed):
    rng=np.random.default_rng(seed); perm=rng.permutation(N); cut=N*3//4; tr,te=perm[:cut],perm[cut:]
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
    with torch.no_grad(): sc=net(torch.tensor(X[te])).numpy()
    return auc(sc,y[te]), auc(amrg[te,phi],y[te])       # combiner AUC, phi-single-amrg AUC (same te)
S=15; res={"phi_self":[],"all":[],"others":[]}; base=[]
for s in range(S):
    a_self,ab=combiner_auc(F_phi,s);  res["phi_self"].append(a_self)
    a_all,_ =combiner_auc(F_all,s);   res["all"].append(a_all)
    a_oth,_ =combiner_auc(F_others,s);res["others"].append(a_oth)
    base.append(ab)
for k in res: res[k]=np.array(res[k])
base=np.array(base)
recal = res["phi_self"]-base                       # phi self-recalibration gain over single amrg
incr  = res["all"]-res["phi_self"]                 # THE cross-model increment
def ci(x): return np.percentile(x,[2.5,97.5])
print(f"[v0.6 ablation]  target = predict phi's own errors; phi-single-amrg AUC {base.mean():.3f}")
print(f"  phi-self combiner AUC   {res['phi_self'].mean():.3f}   (recalibration Δ over amrg {recal.mean():+.3f}  CI[{ci(recal)[0]:+.3f},{ci(recal)[1]:+.3f}])")
print(f"  all-model combiner AUC  {res['all'].mean():.3f}")
print(f"  others-only combiner AUC {res['others'].mean():.3f}   (can OTHER models alone predict phi's errors?)")
print(f"  ==> CROSS-MODEL INCREMENT (all - phi_self): {incr.mean():+.3f}  95%CI[{ci(incr)[0]:+.3f},{ci(incr)[1]:+.3f}]  frac>0={np.mean(incr>0):.0%}")
ensemble_real = ci(incr)[0]>0
print(f"      => {'ENSEMBLE genuinely adds over phi self-recalibration (between-model info PAYS)' if ensemble_real else 'NO cross-model increment — the gain is phi RECALIBRATING ITSELF; between-model delta still null'}")

# ---- Claim B fixes ----
recov=correct.sum(1)>=1; R=np.where(recov)[0]
permeas=[(correct[R,m].mean(),TAGS[m]) for m in range(M)]
strongest=max(permeas); print(f"\n[fusion baseline] strongest single model on routable R = {strongest[1]} ({strongest[0]:.1%}); phi={TAGS[phi]} ({correct[R,phi].mean():.1%})")
def mcp(a,b):
    n01=int(np.sum(a&~b)); n10=int(np.sum(~a&b))
    if n01+n10==0: return 1.0
    return binomtest(min(n01,n10),n01+n10,0.5).pvalue if HAVE_SCIPY else 1.0
w=np.maximum(amrg,0.0); okF=((w[R][...,None]*P[R]).sum(1).argmax(1)==gold[R])
okBest=correct[R,strongest[1]==np.array(TAGS)].astype(bool).ravel() if False else correct[R,[TAGS.index(strongest[1])]].astype(bool).ravel()
p_raw=mcp(okF,okBest); print(f"[fusion] logit-margin fusion {okF.mean():.1%} vs strongest {okBest.mean():.1%}: McNemar p={p_raw:.4g}, Bonferroni×5={min(1,p_raw*5):.4g} ({'sig after correction' if p_raw*5<0.05 else 'NOT sig after ×5 correction'})")
open("RESULTS_v6_ablation.md","w",encoding="utf-8").write(
  f"# v0.6 ablation (decisive)\nSelective: phi-amrg AUC {base.mean():.3f}; phi-self combiner {res['phi_self'].mean():.3f} "
  f"(recal {recal.mean():+.3f}); all combiner {res['all'].mean():.3f}. **CROSS-MODEL INCREMENT {incr.mean():+.3f} "
  f"CI[{ci(incr)[0]:+.3f},{ci(incr)[1]:+.3f}] -> {'ensemble PAYS' if ensemble_real else 'phi self-recalibration only'}.**\n"
  f"Fusion: strongest-on-R {strongest[1]} {okBest.mean():.1%}; fusion {okF.mean():.1%}; McNemar p={p_raw:.4g} Bonf×5 {min(1,p_raw*5):.4g}.\n")
print("  wrote RESULTS_v6_ablation.md")
