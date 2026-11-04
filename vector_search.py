import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. 加载文本块
print("正在加载文本块...")
chunks = []
with open("data/chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

corpus = [c["text"] for c in chunks]
print(f"共加载 {len(corpus)} 个文本块")

# 2. 加载 BGE 模型
print("正在加载 BGE 模型...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 3. 编码所有文本块
print("正在编码文本块...")
embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# 4. 建立 FAISS 索引
print("正在建立 FAISS 索引...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# 5. 检索函数
def search(query, top_k=3):
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype("float32")
    scores, indices = index.search(query_vec, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"]
        })
    return results

# 6. 测试
questions = [
    "地震发生后应该如何响应？",
    "低温雨雪冰冻灾害的预警分级是什么？",
    "中央企业发生事故后多久上报？",
    "养老机构发生突发事件怎么办？"
]

for q in questions:
    print("\n" + "="*60)
    print(f"问题: {q}")
    results = search(q, top_k=2)
    for i, r in enumerate(results, 1):
        print(f"\n--- 结果 {i} (相似度: {r['score']:.4f}) ---")
        print(f"来源: {r['source'][:30]}... 第 {r['page']} 页")
        print(f"内容: {r['text'][:200]}...")
