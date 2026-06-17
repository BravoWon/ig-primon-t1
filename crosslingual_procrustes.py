"""CROSS-LINGUAL PROCRUSTES: is meaning an invariant geometry across languages, models, AND tokenizers?

The confound-free test of the surviving gem. v0.2.17 showed same-tokenizer alignment was mostly shared vocab.
OPT (GPT-2 BPE) and Qwen2.5 (its own multilingual tokenizer) share NO vocabulary, so any alignment is genuinely
semantic. Embed ~100 English words and their Chinese translations; align via orthogonal Procrustes (rotation+
reflection only -- the strict 'same geometry, different orientation' test) and free-linear, PCA-64 (no N<d
overfit). Three arms decompose the invariances:
  (1) Qwen-EN <-> Qwen-ZH   : language invariance within one multilingual model (cleanest).
  (2) OPT-EN  <-> Qwen-EN   : model invariance, same language, DIFFERENT tokenizers (confound-free PRH).
  (3) OPT-EN  <-> Qwen-ZH   : model AND language AND tokenizer invariance (the full universal-manifold test).

PRE-REG (honest): multilingual models map translations nearby -> arm (1) should align strongly (kNN>>chance,
Procrustes high). Cross-model arms weaker (v0.2.16: Procrustes went negative cross-model in the interior). The
'universal manifold' holds to the extent kNN >> chance and Procrustes stays positive. Translations are author-
provided common-usage pairs, not a verified bilingual lexicon (noted).
[V] OPT-2.7B (local) + Qwen2.5-0.5B, contextual hidden states ~2/3 depth, ~100 EN/ZH pairs, fp32.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
DEV = "cuda:0"; OPT = "C:/Users/JT-DEV1/Documents/opt-2.7b"; QWEN = "Qwen/Qwen2.5-0.5B"; K = 8; PCA_K = 16  # PCA-16: well-posed for N=128 (102 train >> 16-d map); PCA-64 overfit (test R2 swung to -56)

PAIRS = [("water","水"),("fire","火"),("mountain","山"),("river","河"),("sky","天空"),("sun","太阳"),
("moon","月亮"),("star","星星"),("tree","树"),("flower","花"),("grass","草"),("dog","狗"),("cat","猫"),
("bird","鸟"),("fish","鱼"),("horse","马"),("cow","牛"),("pig","猪"),("sheep","羊"),("tiger","老虎"),
("person","人"),("man","男人"),("woman","女人"),("child","孩子"),("mother","母亲"),("father","父亲"),
("friend","朋友"),("teacher","老师"),("doctor","医生"),("king","国王"),("hand","手"),("foot","脚"),
("eye","眼睛"),("ear","耳朵"),("mouth","嘴"),("head","头"),("heart","心"),("hair","头发"),("food","食物"),
("rice","米饭"),("meat","肉"),("bread","面包"),("milk","牛奶"),("tea","茶"),("wine","酒"),("fruit","水果"),
("house","房子"),("door","门"),("window","窗户"),("room","房间"),("road","路"),("bridge","桥"),
("city","城市"),("village","村庄"),("country","国家"),("school","学校"),("hospital","医院"),("book","书"),
("paper","纸"),("pen","笔"),("word","词"),("language","语言"),("name","名字"),("year","年"),("month","月"),
("week","星期"),("time","时间"),("money","钱"),("gold","金子"),("iron","铁"),("stone","石头"),("wood","木头"),
("wind","风"),("rain","雨"),("snow","雪"),("cloud","云"),("sea","海"),("lake","湖"),("ice","冰"),
("light","光"),("color","颜色"),("music","音乐"),("war","战争"),("peace","和平"),("love","爱"),("hate","恨"),
("life","生命"),("death","死亡"),("dream","梦"),("fear","恐惧"),("joy","快乐"),("anger","愤怒"),("hot","热"),
("cold","冷"),("big","大"),("small","小"),("good","好"),("bad","坏"),("new","新"),("old","老"),
("long","长"),("high","高"),("fast","快"),("slow","慢"),("strong","强"),("white","白色"),("black","黑色"),
("red","红色"),("green","绿色"),("blue","蓝色"),("eat","吃"),("drink","喝"),("sleep","睡觉"),("run","跑"),
("walk","走"),("fly","飞"),("see","看"),("hear","听"),("speak","说"),("read","读"),("write","写"),
("know","知道"),("think","想"),("give","给"),("buy","买"),("sell","卖"),("come","来"),("go","去")]


def embed(path, items, leading_space):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path).to(torch.float32).to(DEV).eval()
    nl = m.config.num_hidden_layers; layer = int(round(0.66 * nl))
    X = []
    for w in items:
        s = (" " + w) if leading_space else w
        ids = torch.tensor([tok.encode(s)], device=DEV)
        hs = m(ids, output_hidden_states=True).hidden_states[layer]
        X.append(hs[0, -1].float().cpu())
    X = torch.stack(X); X = X - X.mean(0); X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
    del m; gc.collect(); torch.cuda.empty_cache()
    return X


def _pca(X, k):
    Xc = X - X.mean(0); U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vh[:k].t()


def mutual_knn(A, B, k):
    sa = A @ A.t(); sb = B @ B.t(); n = A.shape[0]
    sa.fill_diagonal_(-2); sb.fill_diagonal_(-2)
    na = sa.topk(k, 1).indices; nb = sb.topk(k, 1).indices
    return float(np.mean([len(set(na[i].tolist()) & set(nb[i].tolist())) / k for i in range(n)]))


def _split(n):
    idx = np.arange(n); return idx[:int(.8 * n)], idx[int(.8 * n):]


def linear_r2(A, B):
    A = _pca(A, PCA_K); B = _pca(B, PCA_K); n = A.shape[0]; tr, te = _split(n)
    A1 = torch.cat([A, torch.ones(n, 1)], 1); W = torch.linalg.lstsq(A1[tr], B[tr]).solution
    pred = A1[te] @ W; return float(1 - ((B[te] - pred) ** 2).sum() / ((B[te] - B[tr].mean(0)) ** 2).sum())


def procrustes_r2(A, B):
    A = _pca(A, PCA_K); B = _pca(B, PCA_K); n = A.shape[0]; tr, te = _split(n)
    M = A[tr].t() @ B[tr]; U, S, Vh = torch.linalg.svd(M); R = U @ Vh
    pred = A[te] @ R; return float(1 - ((B[te] - pred) ** 2).sum() / ((B[te] - B[tr].mean(0)) ** 2).sum())


def run():
    print("[CROSS-LINGUAL PROCRUSTES]  is meaning an invariant geometry across language/model/tokenizer?\n")
    en = [e for e, _ in PAIRS]; zh = [z for _, z in PAIRS]
    print(f"  {len(PAIRS)} EN/ZH translation pairs\n  embedding (OPT local + Qwen2.5-0.5B download)...")
    opt_en = embed(OPT, en, True)
    qwen_en = embed(QWEN, en, True)
    qwen_zh = embed(QWEN, zh, False)
    print("  embedded.\n")

    n = len(PAIRS); perm = torch.tensor(np.random.default_rng(0).permutation(n))
    arms = [("Qwen-EN <-> Qwen-ZH  (language)", qwen_en, qwen_zh),
            ("OPT-EN  <-> Qwen-EN  (model)", opt_en, qwen_en),
            ("OPT-EN  <-> Qwen-ZH  (full)", opt_en, qwen_zh)]
    print(f"  {'arm':>34} {'mutual-kNN':>11} {'shuf':>6} {'procrustes-R2':>14} {'linear-R2':>10}")
    for name, A, B in arms:
        print(f"  {name:>34} {mutual_knn(A,B,K):>11.3f} {mutual_knn(A,B[perm],K):>6.3f} "
              f"{procrustes_r2(A,B):>14.3f} {linear_r2(A,B):>10.3f}")

    print("\n[READING]")
    print("  arm (1) = does ONE multilingual model place translations in the same geometry? (language invariance)")
    print("  arm (2) = do two DIFFERENT models with DIFFERENT tokenizers agree in English? (confound-free PRH)")
    print("  arm (3) = the full test: shared meaning across model + language + tokenizer.")
    print("  procrustes-R2 > 0 and >> shuffled kNN  =>  meaning is invariant geometry up to ROTATION.")
    print(f"  (chance kNN ~ {K/n:.3f})")
    print("\n[V] OPT-2.7B + Qwen2.5-0.5B, contextual ~2/3 depth, ~100 EN/ZH pairs, PCA-64, fp32.")


if __name__ == "__main__":
    run()
