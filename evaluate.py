import json
import re
import jieba
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ========== 加载文本块 ==========
print("正在加载文本块...")
chunks = []
with open("data/chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

def normalize_id(s):
    return re.sub(r'[^\u4e00-\u9fff0-9a-zA-Z]', '', s)

chunk_id_to_idx = {}
for i, c in enumerate(chunks):
    chunk_id_to_idx[c["id"]] = i
    chunk_id_to_idx[normalize_id(c["id"])] = i

corpus = [c["text"] for c in chunks]
print(f"共加载 {len(corpus)} 个文本块")

# ========== BM25 ==========
print("正在建立 BM25 索引...")
tokenized_corpus = [list(jieba.cut(text)) for text in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# ========== BGE + FAISS ==========
print("正在加载 BGE 模型...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print("正在编码文本块...")
embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")
print("正在建立 FAISS 索引...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# ========== Reranker ==========
print("正在加载 bge-reranker...")
reranker = CrossEncoder("BAAI/bge-reranker-base")

# ========== 检索函数 ==========
def bm25_search(query, top_k=5):
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return top_indices

def vector_search(query, top_k=10):
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype("float32")
    scores, indices = index.search(query_vec, top_k)
    return list(indices[0])

def rrf_fusion(bm25_idx, vector_idx, k=60):
    rrf_scores = {}
    for rank, idx in enumerate(bm25_idx, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
    for rank, idx in enumerate(vector_idx, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
    return [idx for idx, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)]

def hybrid_search(query, top_k=5, cand=20, k=60):
    bm25_idx = bm25_search(query, top_k=cand)
    vector_idx = vector_search(query, top_k=cand)
    fused = rrf_fusion(bm25_idx, vector_idx, k=k)
    return fused[:top_k]

def rerank_search(query, top_k=5, cand=20, k=60):
    candidates = hybrid_search(query, top_k=cand, cand=cand, k=k)
    pairs = [[query, chunks[i]["text"]] for i in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:top_k]]

# ========== 加载测评集 ==========
eval_set = []
for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
    try:
        with open("data/eval_set.jsonl", "r", encoding=enc) as f:
            eval_set = [json.loads(line) for line in f if line.strip()]
        print(f"用 {enc} 编码成功读取")
        break
    except Exception as e:
        print(f"{enc} 读取失败: {e}")
        continue
print(f"共加载 {len(eval_set)} 条测评集")

# ========== 评测 ==========
def evaluate(search_fn, name, top_k=5, print_miss=False):
    recall = 0
    mrr = 0
    missing = 0
    miss_questions = []
    for item in eval_set:
        q = item["question"]
        correct_ids = item["evidence_ids"]
        correct_idx = []
        for cid in correct_ids:
            if cid in chunk_id_to_idx:
                correct_idx.append(chunk_id_to_idx[cid])
            elif normalize_id(cid) in chunk_id_to_idx:
                correct_idx.append(chunk_id_to_idx[normalize_id(cid)])
        if not correct_idx:
            missing += 1
            continue
        results = search_fn(q, top_k=top_k)
        hit = any(idx in results for idx in correct_idx)
        if hit:
            recall += 1
        else:
            miss_questions.append(q)
        for rank, idx in enumerate(results, start=1):
            if idx in correct_idx:
                mrr += 1 / rank
                break
    n = len(eval_set) - missing
    print(f"\n{name}:")
    print(f"  有效问题数: {n}")
    print(f"  Recall@5 = {recall/n:.4f}")
    print(f"  MRR      = {mrr/n:.4f}")
    if print_miss and miss_questions:
        print(f"  未命中的问题：")
        for mq in miss_questions:
            print(f"    - {mq}")

# ========== 对比多种配置 ==========
print("\n" + "="*60)
print("配置对比")
print("="*60)

# 基线
evaluate(bm25_search, "BM25 (Top-5)")

# RRF 不同候选数和 k 值
for cand in [10, 20, 30]:
    for k in [20, 60]:
        def make_hybrid(c=cand, kk=k):
            return lambda q, top_k=5: hybrid_search(q, top_k=top_k, cand=c, k=kk)
        evaluate(make_hybrid(), f"Hybrid + RRF (候选={cand}, k={k})")

# Rerank 不同候选数
for cand in [10, 20, 30]:
    def make_rerank(c=cand):
        return lambda q, top_k=5: rerank_search(q, top_k=top_k, cand=c, k=20)
    evaluate(make_rerank(), f"Hybrid + RRF + Rerank (候选={cand}, k=20)")