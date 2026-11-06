# W8 周一执行版：多轮改写层（吸纳审查意见后定稿）

> 2026-11-06。基于审查报告（引用位置已逐条核实）修订。
> 红线不变：`retrieve()` 契约零改动，改写层是 `pipeline.answer()` **调用方**的一层。

---

## 一、采纳的四个决策（审查报告 §二）

### 1. 摘要：同一次 structured_call 顺带产出（省一半调用）

改写调用发现历史 > 3 轮时，schema 加 `rolling_summary` 字段，
**一次调用同时产出** standalone_question + rolling_summary。
不做"每次落盘调一次摘要"——10 条对照实验成本直接翻倍，不值。

### 2. token 预算用字符近似（不引 tiktoken）

仓库现有预算全是字符口径（`pipeline.max_context_chars`，pipeline.py:52）。
第一天：改写历史预算 = `总输入预算 × 30%` 折算字符数，超出从最旧截起。
**口径注明"字符近似"**，写进 rewrite prompt 模板的自述块。
引入 tokenizer 的时机：等真实工业语料进来、字符近似出现误判再议。

### 3. needs_clarification 在非交互路径 = 用原问 + 记标记

| 路径 | 行为 |
| --- | --- |
| UI（streamlit） | 真正反问：渲染澄清话术，本轮不检索 |
| CLI / eval / multi_turn_eval | **跳过改写、用原问**，结果行记 `needs_clarification=true` |
| 对照实验计分 | 歧义样本单独一行统计（不计入改写收益/损失的分母） |

不提前定义的话 21:15 的对照实验跑到歧义样本必然卡死——这是审查报告
抓出的最贵的坑。

### 4. UI 恢复：message dict 与 SQLite 行一一对应

`st.session_state` 每次刷新重建，持久化后必须能从 session_id 灌回。
现有 message dict 已含 role/content/citations/contexts/chunk_id
（streamlit_app.py:127），只加 `rewrite_of?` / `created_at` 两字段；
SQLite turns 表列结构与 message dict **字段同名同义**，恢复零转换。

---

## 二、落位（采纳审查 §四）

```
app/rag/
  query_rewrite.py       # rewrite(question, history) → structured_call
                         #   fallback=RewriteResult(standalone_question=原问,
                         #     needs_clarification=False) —— 降级即用原问
  session_store.py       # sessions / turns 两表，sqlite（仿 cache.py:121 写法，
                         #   check_same_thread=False，独立文件 data/sessions.sqlite）
app/rag/prompts/rewrite/
  schema.py + v1.jinja2  # 与 qa/extract 同构；自述块写清字符预算口径
scripts/multi_turn_eval.py  # 10 条对照，复用 retrieval_harness 的 hit 判定口径
tests/test_query_rewrite.py # 离线三边界：首轮不改写 / 降级用原问 / 歧义反问
```

放 `app/rag/` 不放 `app/core/`：检索质量层，参数将来进 RagConfig，
与 settings.py 分工一致（审查判断正确）。

### 改写层与链路的边界（架构立场有据）

retrieval.py 模块 docstring 自己写了："检索段不产生新信息，重写 query
属于检索段之外的事"（retrieval.py:4-5）。落点：调用方在
`pipeline.answer()` 的 `retrieve()` **之前**改写 question；
`retrieve()` / `answer()` 签名零改动（answer.py:133 已核）。

---

## 三、今晚时间盒（修订版）

| 时间 | 任务 | 验收 |
| --- | --- | --- |
| 20:00–20:30 | schema + prompt 模板（含 rolling_summary、字符预算口径） | prompts/rewrite/ 三件套 |
| 20:30–21:10 | query_rewrite.py + 三条离线边界测试 | 离线测试绿（不发请求） |
| 21:10–21:40 | session_store.py + UI 恢复路径 | UI 刷新后历史还在 |
| 21:40–22:10 | multi_turn_eval.py 10 条对照（含歧义标记行） | 对照表落盘 |

## 四、回归保护（审查 §一第 2 行的验证）

改完第一件事：`make eval-smoke`（slice20 检索-only）必须绿——
run_eval 走 pipeline.answer → retrieve，改写不进 retrieve 内部，
回归天然绿。**如果红了，说明改写泄漏进了检索段，立刻回滚。**

## 五、本周新增一条纪律

改写层产生的**每一次调用**（含降级 fallback、歧义标记）都落盘
`eval/results/rewrite-*.json`——W7 的教训：归因要看文本不看标志位，
改写层的"改了什么/为什么没改"必须可回放。
