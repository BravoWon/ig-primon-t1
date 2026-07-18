#!/usr/bin/env python
"""DeltaSheaf-v0.4 — NODES OF IMPORTANCE (the decidable target). On items where >=1 model is right, does a
map from the boundary-layer projections + their deltas identify WHICH model to trust (per-item routing)?
Decidable ground truth (a correct model exists). Bar: beat best-single-model, majority, a raw-stalks router,
and a shuffle control. Round-trip: held-out split; report the oracle ceiling. CPU, ~1 min."""
import itertools, math, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
np.random.seed(0); torch.manual_seed(0)

z=np.load("data/embedded.npz")
stalks,correct,gold,letters,opts = z["stalks"],z["correct"],z["gold"],z["letters"],z["opts"]
N,M,D = stalks.shape          # 4200, 5 models, 512
TAGS=["qwen3b","phi","smollm","falcon","olmo"]
recov=correct.sum(1)>=1        # items where routing CAN win (>=1 model right)
idx=np.where(recov)[0]
rng=np.random.default_rng(0); perm=rng.permutation(idx); cut=len(idx)*3//4
tr,te=perm[:cut],perm[cut:]
print(f"[v0.4 nodes-of-importance]  routable items (>=1 correct) = {len(idx)}  (train {len(tr)} / test {len(te)})")

# ---- boundary-layer projection: PCA(64) fit on TRAIN stalks (pooled), then pairwise projected deltas ----
DP=64; PAIRS=list(itertools.combinations(range(M),2))
Sflat=stalks[tr].reshape(-1,D); mu=Sflat.mean(0)
_,_,Vt=np.linalg.svd(Sflat-mu,full_matrices=False); P=Vt[:DP].T.astype(np.float32)   # [512,64]
def proj(S): return (S-mu)@P                                   # [.,M,64]
def feats(idxs):
    S=stalks[idxs]; Pr=proj(S)                                 # [n,M,64] projected stalks (the projections)
    dl=np.stack([Pr[:,i]-Pr[:,j] for (i,j) in PAIRS],1).reshape(len(idxs),-1)  # projected DELTAS = the map
    st=S.reshape(len(idxs),-1)                                 # raw stalks (control)
    return {"map_deltas":dl.astype(np.float32), "raw_stalks":st.astype(np.float32),
            "both":np.concatenate([Pr.reshape(len(idxs),-1),dl],1).astype(np.float32)}
Ftr,Fte=feats(tr),feats(te)

# ---- non-learned baselines ----
best=int(correct[tr].mean(0).argmax())                        # globally strongest model on train
acc_best=correct[te,best].mean()
maj=np.array([np.bincount(letters[i][letters[i]>=0],minlength=4).argmax() for i in te])
acc_maj=(maj==gold[te]).mean()
oracle=1.0                                                     # >=1 correct by construction

# ---- learned routers: predict per-model reliability, route to argmax ----
class R(nn.Module):
    def __init__(s,din): super().__init__(); s.f=nn.Sequential(nn.Linear(din,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,M))
    def forward(s,x): return s.f(x)
def route(arm,seed,shuffle=False):
    torch.manual_seed(seed)
    Xtr=torch.tensor(Ftr[arm]); Ytr=torch.tensor(correct[tr].astype(np.float32))
    if shuffle: Ytr=Ytr[torch.randperm(len(Ytr))]             # break input<->reliability pairing
    Xte=torch.tensor(Fte[arm])
    m=R(Xtr.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=3e-3,weight_decay=1e-3)
    nval=len(Xtr)//5; vi,ti=torch.arange(nval),torch.arange(nval,len(Xtr)); best_v=1e9;bs=None;pat=0
    for ep in range(300):
        m.train(); loss=F.binary_cross_entropy_with_logits(m(Xtr[ti]),Ytr[ti])
        opt.zero_grad(); loss.backward(); opt.step(); m.eval()
        with torch.no_grad(): vl=F.binary_cross_entropy_with_logits(m(Xtr[vi]),Ytr[vi]).item()
        if vl<best_v-1e-4: best_v,bs,pat=vl,{k:v.clone() for k,v in m.state_dict().items()},0
        else:
            pat+=1
            if pat>=25: break
    m.load_state_dict(bs)
    with torch.no_grad(): pick=m(Xte).argmax(1).numpy()       # the node of importance for each item
    return correct[te,pick].mean()                            # did the routed model get it right?
SEEDS=[0,1,2,3,4]
res={a:(float(np.mean([route(a,s) for s in SEEDS])),float(np.std([route(a,s) for s in SEEDS]))) for a in ["raw_stalks","map_deltas","both"]}
shuf=float(np.mean([route("map_deltas",s,shuffle=True) for s in SEEDS]))

print(f"\n  routing accuracy on held-out routable items (oracle ceiling = 100%):")
print(f"    best-single-model ({TAGS[best]}) : {acc_best:.1%}   [non-learned baseline]")
print(f"    majority vote                    : {acc_maj:.1%}   [non-learned baseline]")
print(f"    router: raw stalks               : {res['raw_stalks'][0]:.1%} ± {res['raw_stalks'][1]:.1%}")
print(f"    router: MAP (boundary deltas)     : {res['map_deltas'][0]:.1%} ± {res['map_deltas'][1]:.1%}   <-- the boundary map")
print(f"    router: both                     : {res['both'][0]:.1%} ± {res['both'][1]:.1%}")
print(f"    control: shuffled map             : {shuf:.1%}")
mp=res["map_deltas"][0]
beats_base = mp>acc_best+0.02 and mp>acc_maj+0.02
beats_raw  = mp>res["raw_stalks"][0]+0.01
real = mp>shuf+0.03
print(f"\n  => VERDICT: node-routing {'BEATS baselines' if beats_base else 'does NOT beat best-single/majority'} "
      f"({'and the boundary delta adds over raw stalks' if beats_raw else 'but adds nothing over raw stalks'}); "
      f"{'not a shuffle artifact' if real else 'AT SHUFFLE LEVEL — artifact'}.")
print(f"     headroom: best-single {acc_best:.1%} -> oracle 100%; the map captured "
      f"{(mp-acc_best)/(1-acc_best)*100:+.0f}% of the gap." if beats_base else "")
open("RESULTS_v4.md","w",encoding="utf-8").write(
  f"# v0.4 nodes-of-importance (routing)\nroutable n={len(idx)}. best-single {acc_best:.1%}, majority {acc_maj:.1%}, "
  f"MAP {mp:.1%}, raw {res['raw_stalks'][0]:.1%}, shuffle {shuf:.1%}.\n"
  f"**{'MAP BEATS baselines' if beats_base else 'MAP does not beat baselines'}; boundary-delta {'adds over raw' if beats_raw else 'no add over raw'}.**\n")
print("  wrote RESULTS_v4.md")
