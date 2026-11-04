import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，防止中文乱码（Windows环境常用 SimHei 或 Microsoft YaHei）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 1. 准备数据（基于你之前的真实评测结果） =====
models = ['BM25 (单路检索)', '混合检索 (BM25+BGE+RRF)', '混合检索 + Reranker重排']

# Recall@5 数据
recall = [0.8415, 0.9268, 0.9390]
# MRR 数据
mrr = [0.6850, 0.6752, 0.7457]

# ===== 2. 设置柱状图参数 =====
x = np.arange(len(models))  # X轴刻度位置
width = 0.35                # 柱子宽度

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制两组柱子
rects1 = ax.bar(x - width/2, recall, width, label='Recall@5', color='#5B9BD5')
rects2 = ax.bar(x + width/2, mrr, width, label='MRR', color='#ED7D31')

# ===== 3. 添加标签、标题和图例 =====
ax.set_ylabel('分数 (Score)', fontsize=12)
ax.set_title('应急管理问答系统：不同检索策略性能对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)

# 设置Y轴范围，让图表更好看
ax.set_ylim(0, 1.1)

# 在柱子上显示具体数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移3个像素
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

# 添加网格线，方便阅读
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# 调整布局并保存图片
fig.tight_layout()
plt.savefig('evaluation_chart.png', dpi=300) # 保存为高清图
print("图表已成功生成并保存为 evaluation_chart.png")

# 显示图表（运行后会弹出一个窗口展示图表）
plt.show()