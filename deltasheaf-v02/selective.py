#!/usr/bin/env python
"""SELECTIVE — the v0.6 signal productized: selective prediction (abstention) for the strongest model.
Fit the ensemble-confidence error-detector for phi (the honest, ablation-verified signal: cross-model
increment +0.026 AUC over phi-self), then deliver the operational artifact:
  - risk-coverage curves: phi answers the top-q fraction by detector score, abstains on the rest;
    compare vs phi's OWN confidence (the trivial baseline) and random abstention.
  - AURC (area under risk-coverage; lower better) with split CIs, both arms.
  - calibrated operating thresholds (target coverages 30..95%) saved to selective_thresholds.json
    with achieved held-out accuracy at each — the numbers a deployment actually needs.
Held-out protocol: 15 splits; curves/thresholds reported as split means. CPU, ~1 min."""
import json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
np.random.seed(0); torch.manual_seed(0)
TAGS=["qwen3b","phi","smollm","falcon","olmo"]; M=5
pool=[json.loads(l) for l in open("data/mmlu_pool.jsonl",encoding="utf-8")]
conf=[[json.loads(l) for l in open(f"data/conf/{t}.jsonl",encoding="utf-8")] for t in TAGS]
N=min(len(c) for c in conf)
correct=np.array([[conf[m][i]["correct"] for m in range(M)] for i in range(N)],dtype=np.float32)
P   =np.array([[conf[m][i]["p"]   for m in range(M)] for i in range(N)],dtype=np.float32)
amrg=np.array([[conf[m][i]["amrg"] for m in range(M)] for i in range(N)],dtype=np.float32)
pmax=P.max(2)
phi=int(correct.mean(0).argmax()); y=correct[:,phi]
FEAT=np.concatenate([P.reshape(N,-1),amrg,pmax],1).astype(np.float32)

class Net(nn.Module):
    def __init__(s,d): super().__init__(); s.f=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,1))
    def forward(s,x): return s.f(x).squeeze(1)

def fit_detector(tr,seed):
    m0=FEAT[tr].mean(0); s0=FEAT[tr].std(0)+1e-6
    X=((FEAT-m0)/s0).astype(np.float32)
    torch.manual_seed(seed); net=Net(X.shape[1]); opt=torch.optim.Adam(net.parameters(),lr=3e-3,weight_decay=1e-3)
    Xtr=torch.tensor(X[tr]); Ytr=torch.tensor(y[tr])
    nv=len(Xtr)//5; vi,ti=torch.arange(nv),torch.arange(nv,len(Xtr)); bv=1e9;bs=None;pat=0
    for ep in range(300):
        net.train(); l=F.binary_cross_entropy_with_logits(net(Xtr[ti]),Ytr[ti]); opt.zero_grad(); l.backward(); opt.step(); net.eval()
        with torch.no_grad(): vl=F.binary_cross_entropy_with_logits(net(Xtr[vi]),Ytr[vi]).item()
        if vl<bv-1e-4: bv,bs,pat=vl,{k:v.clone() for k,v in net.state_dict().items()},0
        else:
            pat+=1
            if pat>=25: break
    net.load_state_dict(bs)
    with torch.no_grad(): score=torch.sigmoid(net(torch.tensor(X))).numpy()
    return score, (m0,s0)

COVS=np.arange(0.30,1.001,0.05)
def risk_cov(score,te):
    s=score[te]; yy=y[te]; order=np.argsort(-s)
    accs=[]
    for c in COVS:
        k=max(1,int(len(te)*c)); accs.append(yy[order[:k]].mean())
    return np.array(accs)
def aurc(accs): return float(np.trapezoid(1-accs,COVS)/(COVS[-1]-COVS[0]))

curves_e=[]; curves_p=[]; thr_rows={f"{c:.2f}":[] for c in COVS}
for s in range(15):
    rng=np.random.default_rng(s); perm=rng.permutation(N); cut=N*3//4; tr,te=perm[:cut],perm[cut:]
    score,_=fit_detector(tr,s)
    curves_e.append(risk_cov(score,te)); curves_p.append(risk_cov(amrg[:,phi],te))
    st=np.sort(score[te])[::-1]
    for c in COVS: thr_rows[f"{c:.2f}"].append(float(st[max(1,int(len(te)*c))-1]))
E=np.array(curves_e); Pc=np.array(curves_p)
au_e=[aurc(E[k]) for k in range(15)]; au_p=[aurc(Pc[k]) for k in range(15)]
d=np.array(au_p)-np.array(au_e); lo,hi=np.percentile(d,[2.5,97.5])
base=float(y.mean())
print(f"[selective]  predictor={TAGS[phi]} (base acc {base:.1%})  detector=ensemble confidence (45 feats)")
print(f"  AURC (lower=better): ensemble {np.mean(au_e):.4f}   phi-own {np.mean(au_p):.4f}   "
      f"Δ {np.mean(d):+.4f}  CI[{lo:+.4f},{hi:+.4f}]  {'ensemble better' if lo>0 else 'ns'}")
print( "  accuracy @ coverage (held-out, 15-split mean)   [random abstention = base at all coverages]")
print( "    cov   ensemble   phi-own    Δ")
for k,c in enumerate(COVS):
    if abs(c*100-round(c*100/10)*10)<1e-6 or c in (0.95,):
        print(f"    {c:.0%}   {E[:,k].mean():.1%}     {Pc[:,k].mean():.1%}    {E[:,k].mean()-Pc[:,k].mean():+.1%}")
out={"predictor":TAGS[phi],"base_acc":base,
     "aurc":{"ensemble":float(np.mean(au_e)),"phi_own":float(np.mean(au_p)),
             "delta_ci":[float(lo),float(hi)]},
     "operating_points":[{"coverage":float(c),
                          "threshold_mean":float(np.mean(thr_rows[f"{c:.2f}"])),
                          "acc_ensemble":float(E[:,k].mean()),"acc_phi_own":float(Pc[:,k].mean())}
                         for k,c in enumerate(COVS)]}
json.dump(out,open("selective_thresholds.json","w"),indent=1)
try:
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.4))
    ax.plot(COVS*100,E.mean(0)*100,"-o",ms=3,label="ensemble detector",color="#2a9df4")
    ax.fill_between(COVS*100,(E.mean(0)-E.std(0))*100,(E.mean(0)+E.std(0))*100,alpha=.18,color="#2a9df4")
    ax.plot(COVS*100,Pc.mean(0)*100,"-s",ms=3,label="phi's own confidence",color="#f4a02a")
    ax.axhline(base*100,ls="--",lw=1,color="#888",label="no abstention / random")
    ax.set_xlabel("coverage (% of items answered)"); ax.set_ylabel("accuracy on answered (%)")
    ax.set_title("Selective prediction: abstain by ensemble confidence (phi, MMLU)")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig("selective_risk_coverage.png",dpi=140)
    print("  wrote selective_risk_coverage.png")
except Exception as e:
    print(f"  (plot skipped: {type(e).__name__})")
print("  wrote selective_thresholds.json")
open("RESULTS_selective.md","w",encoding="utf-8").write(
  f"# selective prediction tool (v0.6 productized)\npredictor {TAGS[phi]} base {base:.1%}. "
  f"AURC ensemble {np.mean(au_e):.4f} vs phi-own {np.mean(au_p):.4f} (Δ{np.mean(d):+.4f} CI[{lo:+.4f},{hi:+.4f}]).\n"
  f"acc@50% cov: ensemble {E[:,4].mean():.1%} vs phi-own {Pc[:,4].mean():.1%}. Thresholds: selective_thresholds.json\n")
print("  wrote RESULTS_selective.md")
