#!/usr/bin/env python
"""DeltaSheaf-v0.2 — substrate + arms + verdict (v0.2.1 chance-relative). Reads data/embedded.npz.
Substrate: R_ij = ridge-LS linear alignment map (stalk_i -> stalk_j) fit on clean; Δ_ij = R_ij(s_i)-s_j.
Signature = cycle residuals only (C_cycle). Arms: map(B_edge) · displacement(C_cycle) · AREA(C_area) ·
volume(ctrl_mag) · blind · shuffle. Decoder -> nomic ℝ⁵¹² -> cosine-nearest option. 5 seeds. H3/H4 gates.
Verdict is CHANCE-relative (majority A=0 on a 0-of-5 gate set)."""
import itertools, math, sys, json
import numpy as np, torch, torch.nn as nn
MLP = "mlp" in sys.argv                                  # SPEC §4 secondary decoder (1-hidden-layer MLP)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

D, NOPT, DSIG, TEMP = 512, 4, 128, 0.1
CHANCE, DMIN, EPS = 1.0/NOPT, 0.10, 0.02
SEEDS = [0,1,2,3,4]
PAIRS = list(itertools.combinations(range(5), 2))      # 10
TRIS  = list(itertools.combinations(range(5), 3))       # 10
PIX = {p:i for i,p in enumerate(PAIRS)}
rng = np.random.default_rng(20260716)
P_CYC = (rng.standard_normal((len(TRIS)*D, DSIG))/math.sqrt(DSIG)).astype(np.float32)
P_EDG = (rng.standard_normal((len(PAIRS)*D, DSIG))/math.sqrt(DSIG)).astype(np.float32)

z = np.load("data/embedded.npz")
stalks, opts, gold, correct, letters = z["stalks"], z["opts"], z["gold"], z["correct"], z["letters"]
n = len(stalks)
# gate = 0-of-5 correct AND >=2 distinct emitted options; clean = >=1 correct
ncorr = correct.sum(1)
ndist = np.array([len(set(l for l in letters[i] if l>=0)) for i in range(n)])
gate = (ncorr==0) & (ndist>=2)
clean = ncorr>=1
gi, ci = np.where(gate)[0], np.where(clean)[0]
print(f"[build] n={n}  gate(0-of-5,>=2opt)={gate.sum()}  clean={clean.sum()}")
# clean train/val split
rs = np.random.default_rng(0); perm = rs.permutation(ci); nval = max(200, len(ci)//5)
val_i, tr_i = perm[:nval], perm[nval:]

# ---- substrate: ridge-LS restriction maps on clean-train ----
lam = 1.0
W = {}
for (i,j) in PAIRS:
    Si, Sj = stalks[tr_i,i], stalks[tr_i,j]
    W[(i,j)] = np.linalg.solve(Si.T@Si + lam*np.eye(D), Si.T@Sj).astype(np.float32)   # s_i @ W ≈ s_j
def deltas(idx):
    S = stalks[idx]                                   # [m,5,512]
    Δ = np.zeros((len(idx), len(PAIRS), D), np.float32)
    for (i,j) in PAIRS: Δ[:, PIX[(i,j)]] = S[:,i]@W[(i,j)] - S[:,j]
    return Δ
def signatures(idx):
    Δ = deltas(idx); m = len(idx)
    cyc = np.stack([Δ[:,PIX[(i,j)]]+Δ[:,PIX[(j,k)]]-Δ[:,PIX[(i,k)]] for (i,j,k) in TRIS],1)  # [m,10,512]
    H = cyc.reshape(m,-1)@P_CYC
    E = Δ.reshape(m,-1)@P_EDG
    area = np.stack([0.5*np.sqrt(np.maximum(0,(a:=Δ[:,PIX[(i,j)]]).__pow__(2).sum(1)*(b:=Δ[:,PIX[(j,k)]]).__pow__(2).sum(1)-(a*b).sum(1)**2))
                     for (i,j,k) in TRIS],1)                                                 # [m,10]
    return {"C_cycle":H, "B_edge":E, "C_area":area.astype(np.float32),
            "ctrl_mag":np.linalg.norm(H,axis=1,keepdims=True).astype(np.float32),
            "ctrl_blind":np.ones((m,DSIG),np.float32)}

Xtr, Xval, Xg = signatures(tr_i), signatures(val_i), signatures(gi)
# ctrl_shuffle = C_cycle with row permutation (break H<->item)
gsh = np.random.default_rng(4242)
for X,idx in [(Xtr,tr_i),(Xval,val_i),(Xg,gi)]:
    X["ctrl_shuffle"] = X["C_cycle"][gsh.permutation(len(idx))]

class Dec(nn.Module):
    def __init__(s,din):
        super().__init__()
        s.w = (nn.Sequential(nn.Linear(din,256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256,D)) if MLP
               else nn.Linear(din,D))
    def forward(s,x): return s.w(x)

def train_eval(arm, seed):
    torch.manual_seed(seed)
    Xt,gt,Ot = torch.tensor(Xtr[arm]), torch.tensor(gold[tr_i]), torch.tensor(opts[tr_i])
    Xv,gv,Ov = torch.tensor(Xval[arm]), torch.tensor(gold[val_i]), torch.tensor(opts[val_i])
    Xe,ge,Oe = torch.tensor(Xg[arm]), torch.tensor(gold[gi]), torch.tensor(opts[gi])
    m=Dec(Xt.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=1e-2,weight_decay=1e-3)
    best=1e9; bs=None; pat=0
    for ep in range(400):
        m.train(); lo=torch.bmm(Ot,m(Xt).unsqueeze(2)).squeeze(2)/TEMP
        loss=nn.functional.cross_entropy(lo,gt); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl=nn.functional.cross_entropy(torch.bmm(Ov,m(Xv).unsqueeze(2)).squeeze(2)/TEMP,gv).item()
        if vl<best-1e-4: best,bs,pat=vl,{k:v.clone() for k,v in m.state_dict().items()},0
        else:
            pat+=1
            if pat>=30: break
    m.load_state_dict(bs)
    with torch.no_grad():
        pred=torch.bmm(Oe,m(Xe).unsqueeze(2)).squeeze(2).argmax(1).numpy()
    return (pred==gold[gi]).astype(np.float32)      # per-item correct on gate

ARMS=["A","ctrl_blind","ctrl_mag","B_edge","C_cycle","C_area","ctrl_shuffle"]
majority = np.array([np.bincount(letters[i][letters[i]>=0],minlength=NOPT).argmax() for i in gi])
res={"A":(float((majority==gold[gi]).mean()),0.0,None)}
per={}
for arm in ARMS[1:]:
    hits=np.stack([train_eval(arm,s) for s in SEEDS])           # [5, n_gate]
    per[arm]=hits
    accs=hits.mean(1); res[arm]=(float(accs.mean()),float(accs.std()),hits)

def boot_ci(a,b,seed=7,B=10000):                                # paired bootstrap CI of (a-b) mean-acc
    g=np.random.default_rng(seed); am=a.mean(0); bm=b.mean(0); n=len(am); d=[]
    for _ in range(B):
        ix=g.integers(0,n,n); d.append(am[ix].mean()-bm[ix].mean())
    lo,hi=np.percentile(d,[2.5,97.5]); return am.mean()-bm.mean(), lo, hi

print(f"\n  arm            gate-acc (mean±sd over 5 seeds)   [decoder={'MLP' if MLP else 'linear'}]")
for arm in ARMS:
    if arm=="A": print(f"  {arm:12}  {res[arm][0]:.3f}   (majority; 0 by construction on 0-of-5)")
    else: print(f"  {arm:12}  {res[arm][0]:.3f} ± {res[arm][1]:.3f}")

cyc=per["C_cycle"]; verdicts={}
for opp in ["B_edge","ctrl_mag","C_area","ctrl_shuffle","ctrl_blind"]:
    d,lo,hi=boot_ci(cyc,per[opp]); verdicts[opp]=(d,lo,hi)
    print(f"    C_cycle − {opp:12} = {d:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]")
# area as the user's focal claim
for opp in ["B_edge","ctrl_mag","C_cycle"]:
    d,lo,hi=boot_ci(per["C_area"],per[opp]); print(f"    C_area  − {opp:12} = {d:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]")

blind=res["ctrl_blind"][0]; leak = blind > CHANCE+EPS
cyc_m=res["C_cycle"][0]
beats_edge = verdicts["B_edge"][0]>=DMIN and verdicts["B_edge"][1]>0
beats_mag  = verdicts["ctrl_mag"][0]>=DMIN and verdicts["ctrl_mag"][1]>0
above_ch   = cyc_m>CHANCE+EPS
print(f"\n  H3 hardness: gate n={int(gate.sum())} (>=300 required) -> {'PASS' if gate.sum()>=300 else 'ABORT'}")
print(f"  H4 leak sentinel: ctrl_blind={blind:.3f} vs chance+eps={CHANCE+EPS:.3f} -> {'LEAK (VOID)' if leak else 'clean'}")
if leak: verdict="VOID(leak)"
elif beats_edge and beats_mag and above_ch and cyc_m>res["ctrl_shuffle"][0]: verdict="PASS"
else: verdict="FALSIFIED"
print(f"\n  => VERDICT (C_cycle, v0.2.1 chance-relative): {verdict}")
print(f"     (chance={CHANCE:.2f}; A degenerate at {res['A'][0]:.2f})")

open("RESULTS_mlp.md" if MLP else "RESULTS.md","w",encoding="utf-8").write(
  f"# DeltaSheaf-v0.2 RESULTS\n\nGate n={int(gate.sum())} (H3 {'PASS' if gate.sum()>=300 else 'ABORT'}), clean={int(clean.sum())}, 5 seeds.\n"
  f"Substrate: ridge-LS restriction maps (freeze-compliant instantiation).\n\n"
  + "".join(f"- {a}: {res[a][0]:.3f}±{res[a][1]:.3f}\n" for a in ARMS if a!="A")
  + f"- A (majority): {res['A'][0]:.3f}\n\n"
  + "".join(f"- C_cycle − {o} = {verdicts[o][0]:+.3f} CI[{verdicts[o][1]:+.3f},{verdicts[o][2]:+.3f}]\n" for o in verdicts)
  + f"\nH4 leak: ctrl_blind={blind:.3f} ({'VOID' if leak else 'clean'}).\n**VERDICT (C_cycle): {verdict}**\n")
print("  wrote RESULTS.md")
