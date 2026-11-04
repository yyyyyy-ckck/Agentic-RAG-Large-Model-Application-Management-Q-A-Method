import json
import jieba
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ========== 加载文本块 ==========
print("正在加载文本块...")
chunks = []
with open("data/chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

corpus = [c["text"] for c in chunks]
print(f"共加载 {len(corpus)} 个文本块")

# ========== BM25 索引 ==========
print("正在建立 BM25 索引...")
tokenized_corpus = [list(jieba.cut(text)) for text in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# ========== BGE + FAISS 索引 ==========
print("正在加载 BGE 模型...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

print("正在编码文本块...")
embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

print("正在建立 FAISS 索引...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# ========== BM25 检索 ==========
def bm25_search(query, top_k=10):
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(idx, scores[idx]) for idx in top_indices]

# ========== 向量检索 ==========
def vector_search(query, top_k=10):
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype("float32")
    scores, indices = index.search(query_vec, top_k)
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]

# ========== RRF 融合 ==========
def rrf_fusion(bm25_results, vector_results, k=60):
    """
    RRF 公式：score = 1/(k + rank)
    rank 从 1 开始
    """
    rrf_scores = {}
    
    # BM25 排名
    for rank, (idx, _) in enumerate(bm25_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
    
    # 向量排名
    for rank, (idx, _) in enumerate(vector_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
    
    # 按 RRF 分数排序
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items

# ========== 混合检索 ==========
def hybrid_search(query, top_k=5):
    bm25_results = bm25_search(query, top_k=10)
    vector_results = vector_search(query, top_k=10)
    fused = rrf_fusion(bm25_results, vector_results)
    
    results = []
    for idx, score in fused[:top_k]:
        results.append({
            "score": score,
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"]
        })
    return results

# ========== 测试 ==========
questions = [
    "地震发生后应该如何响应？",
    "低温雨雪冰冻灾害的预警分级是什么？",
    "中央企业发生事故后多久上报？",
    "养老机构发生突发事件怎么办？"
]

for q in questions:
    print("\n" + "="*60)
    print(f"问题: {q}")
    results = hybrid_search(q, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n--- 结果 {i} (RRF得分: {r['score']:.4f}) ---")
        print(f"来源: {r['source'][:30]}... 第 {r['page']} 页")
        print(f"内容: {r['text'][:200]}...")