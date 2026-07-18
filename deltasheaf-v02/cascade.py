#!/usr/bin/env python
"""CASCADE — push the detector from abstention to FULL-COVERAGE gains: answer with phi when the
(validated) error-detector trusts it, fall back to MAJORITY when it doesn't. Note this is NOT the dead
routing target (5-way argmax): it is a binary defer decision driven by the one ablation-proven signal.
Protocol: 15 splits; threshold tau chosen on TRAIN ONLY (grid over train-score quantiles, maximizing train
cascade accuracy); evaluated on held-out; paired McNemar vs always-phi across the pooled held-out decisions.
Bar: beat always-phi (69.7%) AND always-majority at full coverage with CI-clean paired significance."""
import json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
try: from scipy.stats import binomtest; HAVE=True
except Exception: HAVE=False
np.random.seed(0); torch.manual_seed(0)
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
phi=int(correct.mean(0).argmax()); y=correct[:,phi]
maj=np.array([np.bincount(ans[i],minlength=4).argmax() for i in range(N)])
maj_ok=(maj==gold).astype(np.float32)
FEAT=np.concatenate([P.reshape(N,-1),amrg,pmax],1).astype(np.float32)

class Net(nn.Module):
    def __init__(s,d): super().__init__(); s.f=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,1))
    def forward(s,x): return s.f(x).squeeze(1)
def fit(tr,seed):
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
    with torch.no_grad(): return torch.sigmoid(net(torch.tensor(X))).numpy()

pol_ok_all=[]; phi_ok_all=[]; maj_ok_all=[]; taus=[]; defer_rates=[]
for s in range(15):
    rng=np.random.default_rng(s); perm=rng.permutation(N); cut=N*3//4; tr,te=perm[:cut],perm[cut:]
    score=fit(tr,s)
    # tau on TRAIN only: grid over quantiles, maximize train cascade accuracy
    qs=np.quantile(score[tr],np.linspace(0.05,0.95,19))
    best_t,best_a=None,-1
    for t in qs:
        a=np.where(score[tr]>=t,y[tr],maj_ok[tr]).mean()
        if a>best_a: best_a,best_t=a,t
    taus.append(float(best_t))
    pol=np.where(score[te]>=best_t,y[te],maj_ok[te])
    pol_ok_all.append(pol); phi_ok_all.append(y[te]); maj_ok_all.append(maj_ok[te])
    defer_rates.append(float((score[te]<best_t).mean()))
pol=np.concatenate(pol_ok_all); ph=np.concatenate(phi_ok_all); mj=np.concatenate(maj_ok_all)
def mcn(a,b):
    n01=int(np.sum((a>0.5)&(b<0.5))); n10=int(np.sum((a<0.5)&(b>0.5)))
    if n01+n10==0: return 1.0
    return binomtest(min(n01,n10),n01+n10,0.5).pvalue if HAVE else 1.0
accs=np.array([p.mean() for p in pol_ok_all]); dphi=np.array([pol_ok_all[k].mean()-phi_ok_all[k].mean() for k in range(15)])
lo,hi=np.percentile(dphi,[2.5,97.5])
print(f"[cascade]  policy: phi if detector-score>=tau else MAJORITY  (tau on train only; defer rate {np.mean(defer_rates):.0%})")
print(f"  always-phi      : {ph.mean():.1%}")
print(f"  always-majority : {mj.mean():.1%}")
print(f"  CASCADE         : {pol.mean():.1%}   Δ vs phi {pol.mean()-ph.mean():+.1%}  split-CI[{lo:+.1%},{hi:+.1%}]  "
      f"McNemar-vs-phi p={mcn(pol,ph):.4g}")
win = pol.mean()>ph.mean() and lo>0 and mcn(pol,ph)<0.05 and pol.mean()>mj.mean()
print(f"  => {'CASCADE WINS at full coverage (paired-sig, CI-clean)' if win else 'cascade does NOT beat always-phi cleanly'}")
open("RESULTS_cascade.md","w",encoding="utf-8").write(
  f"# cascade (phi -> majority by detector)\nalways-phi {ph.mean():.1%}, majority {mj.mean():.1%}, "
  f"cascade {pol.mean():.1%} (Δ{pol.mean()-ph.mean():+.1%}, CI[{lo:+.1%},{hi:+.1%}], p={mcn(pol,ph):.4g}, "
  f"defer {np.mean(defer_rates):.0%}).\n**{'WINS' if win else 'no clean win'}**\n")
print("  wrote RESULTS_cascade.md")
