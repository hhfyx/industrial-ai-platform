# `make` 入口。**所有目标都转发给 scripts/tasks.py** —— 命令只在那一处定义。
#
# 为什么不是把命令直接写在这里：实测本机没有 make（Windows 默认不带），
# 而验收要求"一条命令出指标"。如果命令只写在 Makefile 里，
# 没装 make 的人就跑不了。所以：
#
#     scripts/tasks.py   ← 唯一真值（python 一定能跑）
#     Makefile           ← 装了 make 的人用的转发层
#
# 两套定义必然漂移（"make test" 与 "tasks.py test" 参数慢慢就不一样了），
# 所以这里一个命令都不重写，全部转发。

PY := python
UTF8 := -X utf8

.DEFAULT_GOAL := help
.PHONY: help test lint fmt ci eval-retrieval eval-retrieval-hashing topk token-audit grid filter-eval threshold refusal-demo strategy-compare strategy-significance threshold-fine diagnose-semantic answer-quality

help:  ## 列出所有任务
	@$(PY) $(UTF8) scripts/tasks.py

test:  ## 全套测试（断网也能跑）
	@$(PY) $(UTF8) scripts/tasks.py test

lint:  ## ruff 检查
	@$(PY) $(UTF8) scripts/tasks.py lint

fmt:  ## ruff 自动修复 + 格式化
	@$(PY) $(UTF8) scripts/tasks.py fmt

ci: lint test  ## 提交前两条：lint + test

# ------------------------------------------------------------------ 评测
eval-retrieval:  ## 50 条检索评测：Recall@k / MRR / 拒答准确率
	@$(PY) $(UTF8) scripts/tasks.py eval-retrieval

eval-retrieval-hashing:  ## 同上但用 hashing（仅验证流程）
	@$(PY) $(UTF8) scripts/tasks.py eval-retrieval-hashing

# ------------------------------------------------------------------ 调参实验
topk:  ## TopK 扫描：k 3/5/8/10
	@$(PY) $(UTF8) scripts/tasks.py topk

token-audit:  ## bge tokenizer 审计
	@$(PY) $(UTF8) scripts/tasks.py token-audit

grid:  ## chunk × overlap 网格 + 缓存验证
	@$(PY) $(UTF8) scripts/tasks.py grid

filter-eval:  ## 三场景过滤前后命中率
	@$(PY) $(UTF8) scripts/tasks.py filter-eval

threshold:  ## 阈值分布
	@$(PY) $(UTF8) scripts/tasks.py threshold

refusal-demo:  ## 拒答可演示
	@$(PY) $(UTF8) scripts/tasks.py refusal-demo

# ------------------------------------------------------------------ W5 周六主交付
strategy-compare:  ## 三策略 × 各自最优参数（主表）
	@$(PY) $(UTF8) scripts/tasks.py strategy-compare

strategy-significance:  ## 三策略差异显著性
	@$(PY) $(UTF8) scripts/tasks.py strategy-significance

threshold-fine:  ## 阈值精标（50 条集）
	@$(PY) $(UTF8) scripts/tasks.py threshold-fine

diagnose-semantic:  ## 诊断语义切分
	@$(PY) $(UTF8) scripts/tasks.py diagnose-semantic

answer-quality:  ## 答案质量抽检（会花 LLM 钱）
	@$(PY) $(UTF8) scripts/tasks.py answer-quality

# ------------------------------------------------------------------ W6 周一：BM25
build-bm25:  ## 重建 BM25 索引并落盘（记录耗时/体积）
	@$(PY) $(UTF8) scripts/tasks.py build-bm25

bm25-audit:  ## BM25 管线体检（词典/语料/索引一致性）
	@$(PY) $(UTF8) scripts/tasks.py bm25-audit

bm25-compare:  ## 术语型 query：BM25 vs 纯向量对照
	@$(PY) $(UTF8) scripts/tasks.py bm25-compare

# ------------------------------------------------------------------ W6 周二：混合检索
hybrid-compare:  ## 八组对照（纯向量/RRF/加权）+ 显著性检验
	@$(PY) $(UTF8) scripts/tasks.py hybrid-compare

# ------------------------------------------------------------------ W6 周三：Reranker
fetch-reranker:  ## 从 ModelScope 拉 bge-reranker-v2-m3（2.3 GB）
	@$(PY) $(UTF8) scripts/tasks.py fetch-reranker

rerank-probe:  ## 精排延迟/批量/截断/分数跨度（很慢，约 10 分钟）
	@$(PY) $(UTF8) scripts/tasks.py rerank-probe

rerank-eval:  ## 10 条 query 精排前后 top5（约 4 分钟）
	@$(PY) $(UTF8) scripts/tasks.py rerank-eval

# ------------------------------------------------------------------ W6 周四：两段式 + 召回上限
recall-ceiling:  ## 只算 Recall@N（精排天花板，秒级）
	@$(PY) $(UTF8) scripts/tasks.py recall-ceiling

rerank-pipeline:  ## N×K 权衡 + 混合vs纯向量→精排（很慢，40~60 分钟）
	@$(PY) $(UTF8) scripts/tasks.py rerank-pipeline

# ------------------------------------------------------------------ W6 周五：上下文压缩
compress-eval:  ## 压缩：成本 + 检索 + 答案质量三张表
	@$(PY) $(UTF8) scripts/tasks.py compress-eval
