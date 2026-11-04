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

# ========== 检索函数（复用现有逻辑） ==========
def bm25_search(query, top_k=10):
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(idx, scores[idx]) for idx in top_indices]

def vector_search(query, top_k=10):
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype("float32")
    scores, indices = index.search(query_vec, top_k)
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]

def rrf_fusion(bm25_results, vector_results, k=20):
    rrf_scores = {}
    for rank, (idx, _) in enumerate(bm25_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
    for rank, (idx, _) in enumerate(vector_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

def retrieve(query, top_k=3):
    bm25_results = bm25_search(query, top_k=10)
    vector_results = vector_search(query, top_k=10)
    fused = rrf_fusion(bm25_results, vector_results, k=20)
    candidates = []
    for idx, score in fused[:10]:
        candidates.append({
            "idx": idx,
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "rrf_score": score
        })
    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    for c, s in zip(candidates, rerank_scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]

# ========== LLM 决策函数 ==========
def llm_judge_need_retrieval(query):
    """判断问题是否需要检索知识库"""
    prompt = f"""判断以下问题是否需要检索专业知识库才能准确回答。
问题：{query}
只输出 JSON，格式：{{"need_retrieval": true}} 或 {{"need_retrieval": false}}
不需要检索的情况包括：打招呼、闲聊、通用常识。"""
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    text = response.choices[0].message.content.strip()
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(match.group())["need_retrieval"]
    except:
        return True

def llm_judge_sufficiency(query, results):
    """判断证据是否充分，不足则给出改写查询"""
    context = "\n".join([f"[{i+1}] {r['text'][:300]}" for i, r in enumerate(results)])
    prompt = f"""问题：{query}

检索到的证据：
{context}

判断以上证据是否足以回答该问题（不要求100%完整，能回答主要部分即可）。
如果不够，给出一个更好的检索查询，注意不要改变原问题的核心意图。
只输出 JSON：
{{"sufficient": true, "rewritten_query": ""}}
或
{{"sufficient": false, "rewritten_query": "新的查询"}}"""
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    text = response.choices[0].message.content.strip()
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        data = json.loads(match.group())
        return data.get("sufficient", True), data.get("rewritten_query", "")
    except:
        return True, ""

def llm_direct_answer(query):
    """不检索时直接回答"""
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": query}],
        temperature=0.3
    )
    return response.choices[0].message.content

def llm_generate_with_evidence(query, results):
    """基于证据生成答案"""
    context = ""
    for i, r in enumerate(results, 1):
        context += f"[{i}] 来源：{r['source']} 第{r['page']}页\n{r['text']}\n\n"
    prompt = f"""你是应急管理问答助手。请严格根据以下证据回答问题。
要求：
1. 每句话后标注来源编号，如[1]、[2]。
2. 如果证据不足以回答，请回答“根据现有资料无法确定”。
3. 不要编造证据中没有的信息。

证据：
{context}

问题：{query}

回答："""
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

# ========== Agentic 检索主流程（加阈值兜底） ==========
HIGH_CONFIDENCE_THRESHOLD = 0.98  # 高置信度阈值

def agentic_retrieve(query, max_rounds=3, verbose=True):
    """多轮检索：第1轮高置信度直接通过，否则进入多轮改写"""
    history = []

    # 第1轮：用原查询检索
    results = retrieve(query, top_k=3)
    top1 = results[0]["rerank_score"] if results else 0
    if verbose:
        print(f"  [轮次 1] 检索查询：{query}")
        print(f"  [轮次 1] Top-1 rerank 得分：{top1:.4f}")

    # 高置信度兜底：Top-1 得分 > 0.95，直接返回
    if top1 > HIGH_CONFIDENCE_THRESHOLD:
        if verbose:
            print(f"  [轮次 1] 高置信度（>{HIGH_CONFIDENCE_THRESHOLD}），直接通过，不再改写")
        history.append({"round": 1, "query": query, "sufficient": True, "top1_score": top1})
        return results, history

    # 判断证据是否充分
    sufficient, rewritten = llm_judge_sufficiency(query, results)
    if verbose:
        print(f"  [轮次 1] 证据是否充分：{sufficient}")
    history.append({
        "round": 1, "query": query, "sufficient": sufficient,
        "rewritten_query": rewritten, "top1_score": top1
    })
    if sufficient:
        return results, history

    # 多轮改写（最多 max_rounds - 1 次）
    current_query = rewritten
    for round_num in range(2, max_rounds + 1):
        if not current_query:
            break

        # 判断是否需要检索
        need = llm_judge_need_retrieval(current_query)
        if verbose:
            print(f"  [轮次 {round_num}] 是否需要检索：{need}")
        if not need:
            return None, history

        # 检索
        results = retrieve(current_query, top_k=3)
        top1 = results[0]["rerank_score"] if results else 0
        if verbose:
            print(f"  [轮次 {round_num}] 检索查询：{current_query}")
            print(f"  [轮次 {round_num}] Top-1 rerank 得分：{top1:.4f}")

        # 高置信度兜底
        if top1 > HIGH_CONFIDENCE_THRESHOLD:
            if verbose:
                print(f"  [轮次 {round_num}] 高置信度，直接通过")
            history.append({"round": round_num, "query": current_query, "sufficient": True, "top1_score": top1})
            return results, history

        # 判断是否充分
        sufficient, rewritten = llm_judge_sufficiency(query, results)
        if verbose:
            print(f"  [轮次 {round_num}] 证据是否充分：{sufficient}")
            if not sufficient:
                print(f"  [轮次 {round_num}] 改写查询：{rewritten}")
        history.append({
            "round": round_num, "query": current_query, "sufficient": sufficient,
            "rewritten_query": rewritten, "top1_score": top1
        })

        if sufficient:
            return results, history

        # 限制改写长度，不超过原问题的2倍
        if rewritten and len(rewritten) <= len(query) * 2:
            current_query = rewritten
        else:
            break

    return results, history

def agentic_generate_answer(query, max_rounds=3, verbose=True):
    """Agentic RAG 主入口"""
    if verbose:
        print(f"\n问题：{query}")
        print("=" * 70)

    results, history = agentic_retrieve(query, max_rounds=max_rounds, verbose=verbose)

    if results is None:
        if verbose:
            print("  → 不需要检索，直接回答")
        answer = llm_direct_answer(query)
        return answer, [], history
    else:
        answer = llm_generate_with_evidence(query, results)
        return answer, results, history

# ========== 测试 ==========
questions = [
    "地震发生后应该如何响应？",
    "中央企业发生突发事件后，信息报告的时限和内容要求是什么？如果事件可能演化为特别重大、重大突发事件，报告时限有什么特殊规定？",
    "对比《国家地震应急预案》和《国家防汛抗旱应急预案》在应急响应分级上的异同。",
    "城市轨道交通运营突发事件中，如果同时发生列车脱轨和突发大客流，应急处置措施有哪些？涉及哪些部门？"
    
]

for q in questions:
        answer, sources, history = agentic_generate_answer(q, max_rounds=3, verbose=True)
        print(f"\n答案：\n{answer}")
        print(f"\n检索轮数：{len(history)}")
        if sources:
            print("参考来源：")
            for i, s in enumerate(sources, 1):
                print(f"  [{i}] {s['source'][:40]} 第{s['page']}页 (rerank: {s['rerank_score']:.4f})")
        print("\n" + "=" * 70)