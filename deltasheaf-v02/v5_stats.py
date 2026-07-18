#!/usr/bin/env python
"""DeltaSheaf-v0.5 significance recurse (per the verifier's failing-pairs). Replace seed-stability std with:
(A) McNemar PAIRED test of the deterministic 'trust most-confident (logit-margin)' router vs always-best-model,
    on ALL routable items (max power, no split/training noise).
(B) Multi-split CI for the LEARNED confidence router: 20 random splits, distribution of (router - best-single).
Decides whether the +1.6% is real signal or split/seed luck."""
import itertools, json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
try: from scipy.stats import binomtest; HAVE_SCIPY=True
except Exception: HAVE_SCIPY=False
TAGS=["qwen3b","phi","smollm","falcon","olmo"]; M=len(TAGS)
pool=[json.loads(l) for l in open("data/mmlu_pool.jsonl",encoding="utf-8")]
conf=[[json.loads(l) for l in open(f"data/conf/{t}.jsonl",encoding="utf-8")] for t in TAGS]
N=min(len(c) for c in conf)
gold=np.array([pool[i]["answer"] for i in range(N)])
correct=np.array([[conf[m][i]["correct"] for m in range(M)] for i in range(N)],dtype=np.float32)
P   =np.array([[conf[m][i]["p"]    for m in range(M)] for i in range(N)],dtype=np.float32)
amrg=np.array([[conf[m][i]["amrg"] for m in range(M)] for i in range(N)],dtype=np.float32)
rnll=np.array([[conf[m][i]["rnll"] for m in range(M)] for i in range(N)],dtype=np.float32)
rmax=np.array([[conf[m][i]["rmax"] for m in range(M)] for i in range(N)],dtype=np.float32)
recov=correct.sum(1)>=1; idx=np.where(recov)[0]
gbest=int(correct.mean(0).argmax())    # globally strongest model = phi (unambiguous: report margin)
print(f"routable={len(idx)}  strongest model={TAGS[gbest]} ({correct[:,gbest].mean():.1%}); "
      f"next={sorted((correct[:,m].mean(),TAGS[m]) for m in range(M))[-2][1]}")

def mcnemar(a,b):   # a,b boolean correctness arrays (paired); test a vs b
    n01=int(np.sum(a & ~b)); n10=int(np.sum(~a & b))   # a-right/b-wrong ; a-wrong/b-right
    if HAVE_SCIPY:
        p=binomtest(min(n01,n10),n01+n10,0.5).pvalue if (n01+n10)>0 else 1.0
    else:
        from math import comb
        k,n=min(n01,n10),n01+n10
        p=min(1.0,2*sum(comb(n,i) for i in range(k+1))/(2**n)) if n>0 else 1.0
    return n01,n10,p

# (A) deterministic non-learned router: trust most-confident by logit-margin, on ALL routable items
r_conf = correct[idx, amrg[idx].argmax(1)].astype(bool)
r_best = correct[idx, gbest].astype(bool)
n01,n10,p = mcnemar(r_conf,r_best)
print(f"\n(A) trust-most-confident (logit-margin) vs always-{TAGS[gbest]}, ALL {len(idx)} routable items:")
print(f"    acc: confidence-route {r_conf.mean():.1%}  vs best-model {r_best.mean():.1%}   (Δ={r_conf.mean()-r_best.mean():+.1%})")
print(f"    discordant: conf-right/best-wrong={n01}, conf-wrong/best-right={n10}   McNemar p={p:.4g}  "
      f"({'SIGNIFICANT' if p<0.05 else 'NOT significant'} at .05)")
for nm,S in {"pmax":P.max(2),"neg_entropy":(P*np.log(P+1e-9)).sum(2)}.items():
    rc=correct[idx,S[idx].argmax(1)].astype(bool); a,b,pp=mcnemar(rc,r_best)
    print(f"    [{nm}] route {rc.mean():.1%} vs best {r_best.mean():.1%}  McNemar p={pp:.4g}")

# (B) learned router: 20 random splits, distribution of (router_acc - best_acc)
FEAT=np.concatenate([P.reshape(N,-1),amrg,rnll,rmax,P.max(2),(P*np.log(P+1e-9)).sum(2)],1).astype(np.float32)
class R(nn.Module):
    def __init__(s,d): super().__init__(); s.f=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,M))
    def forward(s,x): return s.f(x)
def one_split(split_seed):
    rng=np.random.default_rng(split_seed); perm=rng.permutation(idx); cut=len(idx)*3//4; tr,te=perm[:cut],perm[cut:]
    m0=FEAT[tr].mean(0); s0=FEAT[tr].std(0)+1e-6; X=((FEAT-m0)/s0).astype(np.float32)
    torch.manual_seed(split_seed)
    Xtr=torch.tensor(X[tr]); Ytr=torch.tensor(correct[tr]); Xte=torch.tensor(X[te])
    nv=len(Xtr)//5; vi,ti=torch.arange(nv),torch.arange(nv,len(Xtr)); bv=1e9;bs=None;pat=0
    net=R(Xtr.shape[1]); opt=torch.optim.Adam(net.parameters(),lr=3e-3,weight_decay=1e-3)
    for ep in range(300):
        net.train(); l=F.binary_cross_entropy_with_logits(net(Xtr[ti]),Ytr[ti]); opt.zero_grad(); l.backward(); opt.step(); net.eval()
        with torch.no_grad(): vl=F.binary_cross_entropy_with_logits(net(Xtr[vi]),Ytr[vi]).item()
        if vl<bv-1e-4: bv,bs,pat=vl,{k:v.clone() for k,v in net.state_dict().items()},0
        else:
            pat+=1
            if pat>=25: break
    net.load_state_dict(bs)
    with torch.no_grad(): pick=net(Xte).argmax(1).numpy()
    bsel=int(correct[tr].mean(0).argmax())
    return correct[te,pick].mean()-correct[te,bsel].mean(), correct[te,pick].mean()
diffs=[]; accs=[]
for s in range(20):
    d,a=one_split(s); diffs.append(d); accs.append(a)
diffs=np.array(diffs);
lo,hi=np.percentile(diffs,[2.5,97.5])
print(f"\n(B) learned router across 20 splits:  router acc {np.mean(accs):.1%}   "
      f"Δ(router - best-single) mean {diffs.mean():+.2%}  95%CI [{lo:+.2%}, {hi:+.2%}]  "
      f"frac(Δ>0)={np.mean(diffs>0):.0%}")
verdict = (p<0.05 and r_conf.mean()>r_best.mean()) or (diffs.mean()>0 and lo>0)
print(f"\n=> HONEST VERDICT: within-model confidence-routing is "
      f"{'a REAL (if small) signal — significant over best-single' if verdict else 'NOT significantly > best-single (within noise)'}.")
open("RESULTS_v5_stats.md","w",encoding="utf-8").write(
  f"# v0.5 significance recurse\n(A) logit-margin route {r_conf.mean():.1%} vs best {r_best.mean():.1%} on all {len(idx)}: "
  f"McNemar p={p:.4g} ({'sig' if p<0.05 else 'ns'}).\n(B) learned Δ mean {diffs.mean():+.2%} 95%CI[{lo:+.2%},{hi:+.2%}] "
  f"frac>0={np.mean(diffs>0):.0%}.\n**{'REAL small signal' if verdict else 'not significant over best-single'}.**\n")
print("  wrote RESULTS_v5_stats.md")
