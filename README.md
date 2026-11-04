# Agentic-RAG-Large-Model-Application-Management-Q-A-Method
针对通用大模型在应急法规问答中检索噪声敏感、生成易幻觉、答案不可溯源的问题，重构 RAG 检索与生成链路，提升大模型的事实召回率、引用准确性与拒答可靠性。
# Agentic RAG 应急管理问答系统

基于 RAG（检索增强生成）与 Agentic 智能体技术构建的应急法规问答系统，旨在解决通用大模型在垂直领域存在的“幻觉”与长文本理解不足问题。

##  核心指标
在自建的 82 条测试集上，混合检索 + Reranker 重排将 **Recall@5 从 0.8415 提升至 0.9390**，MRR 提升至 **0.7457**。
*(插入你的 evaluation_chart.png 图片)*

##  技术栈
- **向量模型**：BAAI/bge-small-zh-v1.5
- **重排模型**：BAAI/bge-reranker-base
- **检索框架**：BM25 + FAISS + RRF 融合
- **大模型**：Qwen API (DashScope)
- **可视化**：Matplotlib

##  文件说明
- `ingest_docx.py`: 数据清洗与 400 字切块
- `hybrid_answer.py`: 双通道混合检索与 RRF 融合
- `agentic_rag.py`: Agentic 多轮决策代码（含 0.95 免检与 0.3 拒答阈值）
- `evaluate.py`: 82 条测试集自动化评测脚本
- `plot_result.py`: 性能对比柱状图生成代码
- `data/eval_set.jsonl`: 人工构建的 82 条问答评测集

##  运行说明
1. 安装依赖：`pip install -r requirements.txt`
2. 配置 `.env` 文件，填入 `DASHSCOPE_API_KEY`
3. 运行 `python agentic_rag.py` 即可体验多轮检索问答。
