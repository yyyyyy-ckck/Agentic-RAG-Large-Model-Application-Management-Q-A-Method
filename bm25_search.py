import json
import jieba
from rank_bm25 import BM25Okapi

# 1. 加载切好的文本块
print("正在加载文本块...")
chunks = []
with open("data/chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

corpus = [c["text"] for c in chunks]
print(f"共加载 {len(corpus)} 个文本块")

# 2. 用 jieba 分词
print("正在分词...")
tokenized_corpus = [list(jieba.cut(text)) for text in corpus]

# 3. 建立 BM25 索引
print("正在建立 BM25 索引...")
bm25 = BM25Okapi(tokenized_corpus)

# 4. 测试检索
def search(query, top_k=3):
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    
    # 找出分数最高的 top_k 个
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "score": scores[idx],
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"]
        })
    return results

# 5. 测试几个问题
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
        print(f"\n--- 结果 {i} (得分: {r['score']:.2f}) ---")
        print(f"来源: {r['source'][:30]}... 第 {r['page']} 页")
        print(f"内容: {r['text'][:200]}...")
