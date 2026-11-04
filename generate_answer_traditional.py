import os
import json
import re
import jieba
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ========== 初始化大模型客户端（阿里云百炼 Qwen） ==========
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

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

# ========== Reranker ==========
print("正在加载 bge-reranker...")
reranker = CrossEncoder("BAAI/bge-reranker-base")

# ========== 检索函数 ==========
def bm25_search(query, top_k=30):
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(idx, scores[idx]) for idx in top_indices]

def vector_search(query, top_k=30):
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype("float32")
    scores, indices = index.search(query_vec, top_k)
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]

def rrf_fusion(bm25_results, vector_results, k=20):
    """RRF 融合：候选=30，k=20（最佳配置）"""
    rrf_scores = {}
    for rank, (idx, _) in enumerate(bm25_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
    for rank, (idx, _) in enumerate(vector_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

def retrieve(query, top_k=3):
    """检索流程：BM25(30) + 向量(30) → RRF(k=20) → 取前10 → reranker精排 → 取Top-K"""
    bm25_results = bm25_search(query, top_k=10)
    vector_results = vector_search(query, top_k=10)
    fused = rrf_fusion(bm25_results, vector_results, k=20)

    # 取 RRF 前 10 个候选，送进 reranker
    candidates = []
    for idx, score in fused[:10]:
        candidates.append({
            "idx": idx,
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "rrf_score": score
        })

    # Reranker 精排
    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    for c, s in zip(candidates, rerank_scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]

# ========== 生成答案 ==========
def generate_answer(query, top_k=3):
    # 1. 检索
    results = retrieve(query, top_k=top_k)

    # 2. 如果最高分太低，拒答
    if not results or results[0]["rerank_score"] < 0.3:
        return "根据现有资料无法确定。", []

    # 3. 拼接证据
    context = ""
    for i, r in enumerate(results, 1):
        context += f"[{i}] 来源：{r['source']} 第{r['page']}页\n{r['text']}\n\n"

    # 4. 构造 Prompt
    prompt = f"""你是应急管理问答助手。请严格根据以下证据回答问题。
要求：
1. 每句话后标注来源编号，如[1]、[2]。
2. 如果证据不足以回答，请回答“根据现有资料无法确定”。
3. 不要编造证据中没有的信息。

证据：
{context}

问题：{query}

回答："""

    # 5. 调用大模型
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    answer = response.choices[0].message.content
    return answer, results

# ========== 测试 ==========
if __name__ == "__main__":
    questions = [
        "地震发生后应该如何响应？",
        "低温雨雪冰冻灾害的预警分级是什么？",
        "中央企业发生事故后多久上报？",
        "养老机构发生突发事件怎么办？"
    ]

    for q in questions:
        print("\n" + "="*70)
        print(f"问题: {q}")
        answer, sources = generate_answer(q, top_k=3)
        print(f"\n答案:\n{answer}")
        print("\n参考来源:")
        for i, s in enumerate(sources, 1):
            print(f"  [{i}] {s['source'][:40]} 第{s['page']}页 (rerank: {s['rerank_score']:.4f})")