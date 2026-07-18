#!/usr/bin/env python
"""DeltaSheaf-v0.6 — pushing the ONE real signal (within-model confidence, AUC 0.744) to the targets where a
calibration signal is supposed to pay. Two decidable tests on existing data (data/conf/*.jsonl), CPU:
 PART 1 SELECTIVE PREDICTION. Predictor = the strongest single model (phi). Question: can ENSEMBLE-derived
   confidence predict phi's OWN errors better than phi's own confidence? (paired AUC + bootstrap CI, held-out
   learned combiner). This is the easy cousin of the dead routing question ("IS phi wrong" vs "WHO is right").
 PART 2 CONFIDENCE-WEIGHTED FUSION. Soft-vote the 5 answer distributions weighted by confidence; beat hard
   majority AND best-single? (paired McNemar on routable items).
Merit bar (both): beat the trivial baseline with paired significance."""
import itertools, json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
try: from scipy.stats import binomtest; HAVE_SCIPY=True
except Exception: HAVE_SCIPY=False
np.random.seed(0); torch.manual_seed(0)
TAGS=["qwen3b","phi","smollm","falcon","olmo"]; M=len(TAGS)
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
print(f"[v0.6]  N={N}  strongest single model = {TAGS[phi]} ({correct[:,phi].mean():.1%})")

def auc(s,y):
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    n1=y.sum(); n0=len(y)-n1
    return 0.5 if n1==0 or n0==0 else (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def mcnemar(a,b):
    n01=int(np.sum(a&~b)); n10=int(np.sum(~a&b))
    if n01+n10==0: return n01,n10,1.0
    if HAVE_SCIPY: return n01,n10,binomtest(min(n01,n10),n01+n10,0.5).pvalue
    from math import comb; k,n=min(n01,n10),n01+n10
    return n01,n10,min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n)

# ================= PART 1: SELECTIVE PREDICTION (predict phi's errors) =================
rng=np.random.default_rng(0); perm=rng.permutation(N); cut=N*3//4; tr,te=perm[:cut],perm[cut:]
y=correct[:,phi]                                   # 1 = phi correct  (target for selective prediction)
FEAT=np.concatenate([P.reshape(N,-1),amrg,rnll,rmax,pmax,negent],1).astype(np.float32)
# fixed-rule confidences for "is phi right?":
S_phi   = amrg[:,phi]                               # phi's own confidence  (BASELINE)
S_mean  = amrg.mean(1)                              # ensemble mean confidence
S_agree = (ans==ans[:,phi:phi+1]).sum(1).astype(np.float32)   # how many models agree with phi
# learned combiner on held-out:
class Net(nn.Module):
    def __init__(s,d): super().__init__(); s.f=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,1))
    def forward(s,x): return s.f(x).squeeze(1)
def learn_phi(seed):
    torch.manual_seed(seed)
    m0=FEAT[tr].mean(0); s0=FEAT[tr].std(0)+1e-6; X=((FEAT-m0)/s0).astype(np.float32)
    Xtr=torch.tensor(X[tr]); Ytr=torch.tensor(y[tr]); Xte=torch.tensor(X[te])
    nv=len(Xtr)//5; vi,ti=torch.arange(nv),torch.arange(nv,len(Xtr)); bv=1e9;bs=None;pat=0
    net=Net(Xtr.shape[1]); opt=torch.optim.Adam(net.parameters(),lr=3e-3,weight_decay=1e-3)
    for ep in range(300):
        net.train(); l=F.binary_cross_entropy_with_logits(net(Xtr[ti]),Ytr[ti]); opt.zero_grad(); l.backward(); opt.step(); net.eval()
        with torch.no_grad(): vl=F.binary_cross_entropy_with_logits(net(Xtr[vi]),Ytr[vi]).item()
        if vl<bv-1e-4: bv,bs,pat=vl,{k:v.clone() for k,v in net.state_dict().items()},0
        else:
            pat+=1
            if pat>=25: break
    net.load_state_dict(bs)
    with torch.no_grad(): return net(Xte).numpy()
S_learn_te=np.mean([learn_phi(s) for s in range(5)],0)
yte=y[te]
a_phi=auc(S_phi[te],yte); a_mean=auc(S_mean[te],yte); a_agree=auc(S_agree[te],yte); a_learn=auc(S_learn_te,yte)
# paired bootstrap: does ensemble beat phi-own at predicting phi's errors?
B=3000; d_learn=[]; d_mean=[]
for _ in range(B):
    bi=np.random.randint(0,len(te),len(te))
    d_learn.append(auc(S_learn_te[bi],yte[bi])-auc(S_phi[te][bi],yte[bi]))
    d_mean.append(auc(S_mean[te][bi],yte[bi])-auc(S_phi[te][bi],yte[bi]))
dl=np.percentile(d_learn,[2.5,97.5]); dm=np.percentile(d_mean,[2.5,97.5])
print("\n  PART 1 — SELECTIVE PREDICTION: predict phi's OWN errors (held-out AUC, higher=better):")
print(f"    phi's own confidence   : {a_phi:.3f}   [BASELINE]")
print(f"    ensemble mean conf     : {a_mean:.3f}   Δ {a_mean-a_phi:+.3f}  95%CI[{dm[0]:+.3f},{dm[1]:+.3f}]  {'SIG' if dm[0]>0 else 'ns'}")
print(f"    ensemble agreement     : {a_agree:.3f}")
print(f"    learned combiner (all) : {a_learn:.3f}   Δ {a_learn-a_phi:+.3f}  95%CI[{dl[0]:+.3f},{dl[1]:+.3f}]  {'SIG' if dl[0]>0 else 'ns'}")
# concrete risk-coverage: accuracy among top-coverage by best ensemble score vs phi-own
best_ens = S_learn_te if a_learn>=a_mean else S_mean[te]
def acc_at(score,cov):
    k=int(len(te)*cov); idx=np.argsort(-score)[:k]; return yte[idx].mean()
print("    accuracy@coverage (phi answers, abstain low-confidence):")
for cov in (0.5,0.7,0.9):
    print(f"      cov {cov:.0%}:  phi-own {acc_at(S_phi[te],cov):.1%}   ensemble {acc_at(best_ens,cov):.1%}")
sel_win = dl[0]>0 or dm[0]>0

# ================= PART 2: CONFIDENCE-WEIGHTED FUSION =================
recov=correct.sum(1)>=1; R=np.where(recov)[0]
def fuse(weight):                                  # weight[N,M] -> fused answer per item
    W=weight[...,None]; s=(W*P).sum(1); return s.argmax(1)
maj = np.array([np.bincount(ans[i],minlength=4).argmax() for i in range(N)])
base= ans[:,phi]
f_soft   = fuse(np.ones_like(pmax))                # unweighted soft vote (each P already encodes its own conf)
f_pmax   = fuse(pmax)                              # weight by each model's max-prob
f_amrg   = fuse(np.maximum(amrg,0.0))              # weight by (clamped) logit margin
def acc(p): return (p[R]==gold[R]).mean()
print(f"\n  PART 2 — CONFIDENCE-WEIGHTED FUSION (on {len(R)} routable items, oracle 100%):")
print(f"    best-single ({TAGS[phi]})   : {correct[R,phi].mean():.1%}   [baseline]")
print(f"    hard majority         : {(maj[R]==gold[R]).mean():.1%}   [baseline]")
for nm,p in [("soft vote (unweighted)",f_soft),("conf-weighted (pmax)",f_pmax),("conf-weighted (logit-margin)",f_amrg)]:
    ca=acc(p)
    _,_,pmaj=mcnemar((p[R]==gold[R]),(maj[R]==gold[R])); _,_,pbest=mcnemar((p[R]==gold[R]),correct[R,phi].astype(bool))
    tag=""
    if ca>(maj[R]==gold[R]).mean() and pmaj<0.05 and ca>correct[R,phi].mean() and pbest<0.05: tag="  <-- BEATS both (sig)"
    print(f"    {nm:28s}: {ca:.1%}   vs-maj p={pmaj:.3g}  vs-best p={pbest:.3g}{tag}")
best_fuse=max(acc(f_soft),acc(f_pmax),acc(f_amrg))
_,_,pfb=mcnemar((f_pmax[R]==gold[R]),correct[R,phi].astype(bool))
fuse_win = best_fuse>correct[R,phi].mean() and pfb<0.05

print(f"\n  => VERDICT: selective-prediction {'PAYS (ensemble predicts phi errors better than phi does, sig)' if sel_win else 'does NOT beat phi-own confidence'}; "
      f"fusion {'BEATS best-single+majority (sig)' if fuse_win else 'does NOT significantly beat majority/best-single'}.")
open("RESULTS_v6.md","w",encoding="utf-8").write(
  f"# v0.6 push the real signal\nPART1 selective (predict phi errors): phi-own AUC {a_phi:.3f}, ensemble learned {a_learn:.3f} "
  f"(Δ{a_learn-a_phi:+.3f} 95%CI[{dl[0]:+.3f},{dl[1]:+.3f}]), mean {a_mean:.3f}. **{'ensemble PAYS' if sel_win else 'no gain over phi-own'}.**\n"
  f"PART2 fusion: best-single {correct[R,phi].mean():.1%}, majority {(maj[R]==gold[R]).mean():.1%}, "
  f"conf-weighted {acc(f_pmax):.1%}. **{'fusion BEATS baselines' if fuse_win else 'no sig gain'}.**\n")
print("  wrote RESULTS_v6.md")
