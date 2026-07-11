"""Localize the grounded-sheaf training failure: does it fail at depth-1 (single composition, trivial)?
Try cell variants. Reuses recursion_gate's vocab/target via copy to stay self-contained."""
from collections import defaultdict
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from nltk.corpus import wordnet as wn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SS = ["noun.animal","noun.artifact","noun.food","noun.plant","noun.body","noun.location","noun.person","noun.substance"]
S,Bd,NV,D,HID = len(SS),4,8,64,128

def build():
    cand=defaultdict(set)
    for syn in wn.all_synsets("n"):
        if syn.lexname() in SS:
            for lem in syn.lemmas():
                w=lem.name()
                if w.isalpha() and len(w)>2: cand[w].add((SS.index(syn.lexname()),syn.min_depth()))
    vocab,by={},defaultdict(list)
    for w,st in cand.items():
        if len({x[0] for x in st})==1:
            ss=next(iter(st))[0]; vocab[w]=(ss,int(np.median([x[1] for x in st]))); by[ss].append(w)
    rng=np.random.default_rng(0); ch={}
    for ss,ws in by.items():
        ws=sorted(ws); rng.shuffle(ws)
        for w in ws[:40]: ch[w]=vocab[w]
    qs=np.quantile([d for _,d in ch.values()],[.25,.5,.75])
    return {w:(ss,int(sum(d>q for q in qs))) for w,(ss,d) in ch.items()}

feat=build(); words=sorted(feat); rng=np.random.default_rng(0)
ss_of={w:feat[w][0] for w in words}; dp_of={w:feat[w][1] for w in words}

def gen(depth,n):
    subj=[[words[i] for i in rng.integers(0,len(words),n)] for _ in range(depth)]
    verb=rng.integers(0,NV,(depth,n)); obj=[words[i] for i in rng.integers(0,len(words),n)]
    val=np.array([ss_of[w] for w in obj])
    for i in reversed(range(depth)):
        val=(np.array([ss_of[w] for w in subj[i]])+2*val+verb[i])%S
    T=lambda a:torch.tensor(a,device=DEV)
    sinp=([T([ss_of[w] for w in l]) for l in subj],[T([dp_of[w] for w in l]) for l in subj],
          [T(v) for v in verb],T([ss_of[w] for w in obj]),T([dp_of[w] for w in obj]))
    return sinp,T(val)

class Sheaf(nn.Module):
    def __init__(self,cell):
        super().__init__(); self.cell=cell
        self.ess,self.edp,self.everb=nn.Embedding(S,D),nn.Embedding(Bd,D),nn.Embedding(NV,D)
        self.leaf=nn.Sequential(nn.Linear(2*D,D),nn.GELU())
        self.Rs,self.Rv,self.Rc=nn.Linear(2*D,D),nn.Linear(D,D),nn.Linear(D,D)
        self.compose=nn.Sequential(nn.Linear(3*D,HID),nn.GELU(),nn.Linear(HID,D))
        self.ln=nn.LayerNorm(D); self.head=nn.Linear(D,S)
    def forward(self,s_ss,s_dp,vb,o_ss,o_dp):
        val=self.leaf(torch.cat([self.ess(o_ss),self.edp(o_dp)],-1))
        for i in reversed(range(len(vb))):
            s=torch.cat([self.ess(s_ss[i]),self.edp(s_dp[i])],-1)
            c=self.compose(torch.cat([self.Rs(s),self.Rv(self.everb(vb[i])),self.Rc(val)],-1))
            val = self.ln(c) if self.cell=="ln" else (self.ln(val+c) if self.cell=="resln" else c)
        return self.head(val)

def trial(cell,depths,steps=4000):
    torch.manual_seed(0); m=Sheaf(cell).to(DEV); opt=torch.optim.AdamW(m.parameters(),2e-3); L=[]
    for _ in range(steps):
        d=depths[rng.integers(0,len(depths))]; (si,y)=gen(d,128)
        loss=F.cross_entropy(m(*si),y); loss.backward(); opt.step(); opt.zero_grad(); L.append(loss.item())
    def acc(d):
        (si,y)=gen(d,2000)
        with torch.no_grad(): return (m(*si).argmax(-1)==y).float().mean().item()
    print(f"  cell={cell:6} train={depths}  finalloss={np.mean(L[-300:]):.3f}  acc d1={acc(1):.2f} d3={acc(3):.2f} d5={acc(5):.2f}")

print("depth-1 ONLY (single composition -- must be ~1.0 if the cell can fit at all):")
for c in ["plain","ln","resln"]: trial(c,[1])
print("depths 1-3 (mixed):")
for c in ["plain","ln","resln"]: trial(c,[1,2,3])
