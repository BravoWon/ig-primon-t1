#!/usr/bin/env python
"""DeltaSheaf-v0.3 — retrieval node (robust). For each MMLU item, fetch a Wikipedia passage via the
QUESTION (never the answer). v2 fixes the empty-node failure: descriptive UA, inter-request delay,
retries+backoff, and EXPLICIT error accounting (so throttling can't hide as 'empty'). Checkpointed."""
import json, os, sys, time, urllib.request, urllib.parse
from collections import Counter
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
os.chdir(os.path.dirname(os.path.abspath(__file__)))

POOL="data/mmlu_pool.jsonl"; OUT="data/retrieval.jsonl"; N=4200
UA="DeltaSheaf-research/0.3 (https://github.com/deltasheaf; contact deltasheaf@example.org)"
DELAY=0.25            # polite inter-request pause
ERRS=Counter()

def wiki(query, retries=4):
    q=urllib.parse.quote(query[:250].replace("\n"," ").strip())
    url=(f"https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={q}"
         f"&gsrlimit=1&prop=extracts&exintro&explaintext&format=json&formatversion=2")
    for k in range(retries):
        try:
            req=urllib.request.Request(url, headers={"User-Agent":UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                d=json.load(r)
            pages=d.get("query",{}).get("pages",[])
            ext=(pages[0].get("extract","") if pages else "")[:1200]
            ERRS["ok" if ext else "empty_extract"]+=1
            return ext
        except Exception as e:
            code=getattr(e,"code",None)
            if k==retries-1: ERRS[f"fail:{type(e).__name__}{'/'+str(code) if code else ''}"]+=1; return ""
            time.sleep(1.2*(k+1))
    return ""

def main():
    smoke="--smoke" in sys.argv
    pool=[json.loads(l) for l in open(POOL,encoding="utf-8")][:N]
    if smoke:
        idxs=list(range(0,1000,10))[:100]                 # 100 spread items
        for i in idxs: wiki(pool[i]["question"]); time.sleep(DELAY)
        tot=sum(ERRS.values()); ne=ERRS["ok"]
        print(f"  SMOKE {len(idxs)} items -> non-empty {ne}/{tot} ({ne/tot:.0%})")
        print(f"  error/outcome breakdown: {dict(ERRS)}")
        print(f"  -> {'FIX WORKS (high non-empty)' if ne/tot>0.5 else 'STILL FAILING — see breakdown'}")
        return
    done=sum(1 for _ in open(OUT,encoding="utf-8")) if os.path.exists(OUT) else 0
    if done>=N: print(f"  DONE-ALREADY ({done}/{N})"); return
    f=open(OUT,"a",encoding="utf-8")
    for i in range(done,N):
        f.write(json.dumps({"passage":wiki(pool[i]["question"])},ensure_ascii=False)+"\n"); f.flush()
        time.sleep(DELAY)
        if i%100==0: print(f"  {i}/{N}  non-empty-so-far {ERRS['ok']}  {dict(ERRS)}",flush=True)
    print(f"  RETRIEVAL DONE ({N})  final {dict(ERRS)}")

if __name__=="__main__":
    main()
