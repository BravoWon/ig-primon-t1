#!/usr/bin/env python
"""Procrustes-Import Gate v1 (per PREREG.md). Heterogeneous pair, matched hidden dim (2048):
  S = Qwen2.5-3B-Instruct (strong)  ->  G = OLMo-2-0425-1B-Instruct (weak).
Align the two models' FINAL-layer answer-position hidden states via orthogonal Procrustes (fit on
BOTH-CORRECT anchors = shared-truth geometry), then decode R*h_S through G's OWN LM head. Test set =
items S-correct & G-wrong. Question: does the rotation carry S's correctness into G's readout frame,
beyond random-R / identity / wrong-item — recovering answers G fails?
Hiddens extracted by teacher-forcing each model's stored reply to its answer position (as confidence.py);
cached to npz. Offline. Instrument checks first: heterogeneity, readout validity, alignment validity."""
import json, os, re, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
np.random.seed(0); torch.manual_seed(0)
DS="../deltasheaf-v02"; LET=["A","B","C","D"]
PROMPT=("Answer the multiple-choice question. First give a brief one- or two-sentence reason, then on a "
        "new line write exactly 'Answer: X' where X is the letter of the correct option.\n\nQuestion: {q}\n{opts}\n")
S=("qwen3b","Qwen/Qwen2.5-3B-Instruct","qwen25_3b"); G=("olmo","allenai/OLMo-2-0425-1B-Instruct","olmo2_1b")

pool=[json.loads(l) for l in open(f"{DS}/data/mmlu_pool.jsonl",encoding="utf-8")]
def load_raw(tag): return [json.loads(l) for l in open(f"{DS}/data/raw/{tag}.jsonl",encoding="utf-8")]
Sr, Gr = load_raw(S[2]), load_raw(G[2])
N=min(len(Sr),len(Gr))
Sc=np.array([Sr[i]["correct"] for i in range(N)]); Gc=np.array([Gr[i]["correct"] for i in range(N)])
print(f"[heterogeneity]  S={S[0]} acc {Sc.mean():.1%}   G={G[0]} acc {Gc.mean():.1%}   gap {Sc.mean()-Gc.mean():+.1%}")
both=np.where(Sc & Gc)[0]; test=np.where(Sc & ~Gc)[0]
rng=np.random.default_rng(0); both=rng.permutation(both); test=rng.permutation(test)
anch=both[:400]; anch_ho=both[400:520]; test=test[:600]
print(f"[sets]  anchors(both-correct) {len(anch)} (+{len(anch_ho)} holdout)   test(S-right,G-wrong) {len(test)}")

def letter_ids(tok):
    ids=[]
    for L in LET:
        c=[tok(s,add_special_tokens=False)["input_ids"][0] for s in (L," "+L) if tok(s,add_special_tokens=False)["input_ids"]]
        ids.append(c[0])
    return ids

def extract(tag,mid,rawrows,idxs):
    cache=f"data/hid_{tag}.npz"; os.makedirs("data",exist_ok=True)
    have={}
    if os.path.exists(cache):
        z=np.load(cache); have={int(i):z["H"][k] for k,i in enumerate(z["idx"])}
    need=[i for i in idxs if i not in have]
    if need:
        print(f"[extract {tag}] {len(need)} hiddens",flush=True)
        tok=AutoTokenizer.from_pretrained(mid)
        if tok.pad_token_id is None: tok.pad_token=tok.eos_token
        mdl=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.float16).to("cuda").eval()
        t0=time.time()
        for c,i in enumerate(need):
            it=pool[i]; reply=rawrows[i]["reply"]; m=list(re.finditer(r"Answer:",reply,re.I))
            opts="\n".join(f"{LET[k]}. {ch}" for k,ch in enumerate(it["choices"]))
            pref=tok.apply_chat_template([{"role":"user","content":PROMPT.format(q=it["question"],opts=opts)}],
                  tokenize=False,add_generation_prompt=True)
            cut=m[-1].end() if m else len(reply)
            ids=torch.tensor([tok(pref+reply[:cut],add_special_tokens=False)["input_ids"]]).to("cuda")
            with torch.no_grad(): out=mdl(ids,output_hidden_states=True)
            have[i]=out.hidden_states[-1][0,-1,:].float().cpu().numpy()
            if c%200==0: print(f"    {tag} {c}/{len(need)} ({time.time()-t0:.0f}s)",flush=True)
        idx=np.array(sorted(have)); H=np.stack([have[i] for i in idx])
        np.savez(cache,idx=idx,H=H)
        if tag==G[0]:  # keep G's head+norm for readout
            return have, tok, mdl
        del mdl; torch.cuda.empty_cache()
    return have, None, None

allidx=sorted(set(list(anch)+list(anch_ho)+list(test)))
HS,_,_ = extract(*S[:2],Sr,allidx)
HG, gtok, gmdl = extract(*G[:2],Gr,allidx)
if gmdl is None:
    gtok=AutoTokenizer.from_pretrained(G[1]);
    if gtok.pad_token_id is None: gtok.pad_token=gtok.eos_token
    gmdl=AutoModelForCausalLM.from_pretrained(G[1],dtype=torch.float16).to("cuda").eval()
GLID=letter_ids(gtok)
gnorm=gmdl.model.norm; ghead=gmdl.lm_head

def stack(H,idxs): return np.stack([H[i] for i in idxs]).astype(np.float32)
Xs_a,Xg_a=stack(HS,anch),stack(HG,anch)
muS,muG=Xs_a.mean(0),Xg_a.mean(0)
def procrustes(A,B):  # R: A->B (rotation), on centered
    M=(B-muG).T@(A-muS); U,_,Vt=np.linalg.svd(M); return (U@Vt).astype(np.float32)
R=procrustes(Xs_a,Xg_a)                       # S->G
Rgg=procrustes(Xg_a,Xg_a)                      # G->G self (≈I)
Rrand=np.linalg.svd(rng.standard_normal((2048,2048)))[0].astype(np.float32)

def decode_G(vecs):                            # vecs [n,2048] in G final-hidden space -> letter idx
    x=torch.tensor(np.asarray(vecs),dtype=torch.float16,device="cuda")
    with torch.no_grad(): lg=ghead(gnorm(x)).float().cpu().numpy()
    return np.array([int(np.argmax([lg[k,t] for t in GLID])) for k in range(len(vecs))])
def apply_R(Rm,Xs): return (muG + (Xs-muS)@Rm.T).astype(np.float32)   # map S-hidden into G space

Xs_t=stack(HS,test); Xg_t=stack(HG,test)
gold=np.array([pool[i]["answer"] for i in test])
# instrument: readout validity — G's own hidden must decode to G's own (wrong) letter
g_self=decode_G(Xg_t); g_self_acc=(g_self==gold).mean()
s_own=np.array([LET.index(Sr[i]["letter"]) if Sr[i]["letter"] in LET else -1 for i in test])
readout_ok=(g_self==np.array([LET.index(Gr[i]["letter"]) if Gr[i]["letter"] in LET else -2 for i in test])).mean()
print(f"\n[instrument] readout reproduces G's own letter on test: {readout_ok:.1%} (want high)")
print(f"[instrument] G-own decoded acc on test {g_self_acc:.1%} (≈0 by construction)   S-own known acc 100%")
# alignment validity on holdout both-correct
Xs_h,Xg_h=stack(HS,anch_ho),stack(HG,anch_ho)
al=lambda Rm: float(np.mean([np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
        for a,b in zip(apply_R(Rm,Xs_h)-muG, Xg_h-muG)]))
print(f"[instrument] holdout aligned cosine: Procrustes {al(R):+.3f} vs random-R {al(Rrand):+.3f}")

# ---- arms on test ----
arms={
 "transplant  R*h_S":  decode_G(apply_R(R,   Xs_t)),
 "random-R    Rr*h_S": decode_G(apply_R(Rrand,Xs_t)),
 "identity    h_S":    decode_G(muG+(Xs_t-muS)),
 "wrong-item  R*hSp":  decode_G(apply_R(R,   Xs_t[rng.permutation(len(test))])),
 "self G->G   Rgg*h_G":decode_G(apply_R(Rgg, Xg_t)),
}
print(f"\n[Procrustes-import v1]  S={S[0]}->G={G[0]}  test={len(test)} (S-right,G-wrong; G floor 0%, S ceiling 100%)")
acc={k:(v==gold).mean() for k,v in arms.items()}
for k in arms: print(f"    {k:22s} recovery {acc[k]:.1%}")
tp=acc["transplant  R*h_S"]
ctl=max(acc["random-R    Rr*h_S"],acc["identity    h_S"],acc["wrong-item  R*hSp"])
mech = tp>acc["random-R    Rr*h_S"]+0.05 and tp>acc["identity    h_S"]+0.05 and tp>acc["wrong-item  R*hSp"]+0.05
print(f"\n  => MECHANISM {'CONFIRMED — Procrustes transfers S-correctness into G beyond all controls' if mech else 'NOT confirmed — transplant does not beat controls'}"
      f"  (transplant {tp:.1%} vs best control {ctl:.1%})   NOTE: final-layer h_S ≈ answer -> this is the routing-trivial regime; v2/v4 test capability transfer where h is not the answer.")
print(f"     value bar (CODA-2 text-import, 7B ref) 81.7%; local olmo+fact baseline = v2.")
open("RESULTS_v1.md","w",encoding="utf-8").write(
  f"# Procrustes-import v1  {S[0]}->{G[0]}\nhet gap {Sc.mean()-Gc.mean():+.1%}; test {len(test)}; readout-valid {readout_ok:.1%}; "
  f"holdout cos Procrustes {al(R):+.3f} vs random {al(Rrand):+.3f}.\n"
  +"; ".join(f"{k.split()[0]} {acc[k]:.1%}" for k in arms)+
  f"\n**MECHANISM {'CONFIRMED' if mech else 'not confirmed'}** (transplant {tp:.1%}).\n")
print("  wrote RESULTS_v1.md")
