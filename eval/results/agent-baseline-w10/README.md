# Agent 基线指纹（W10 · v1.2.0 · W11 回归子集的对照凭据）

> 本目录是 **W10 判据基线的入库快照**（源产物在 `qa_runs_w10*/`，被 ignore 是
> 因为轨迹会随复跑更新；这里的是 v1.2.0 时点的冻结凭据，不随复跑变）。
> W11 的 20 任务里，这 10 个任务是回归子集——加可靠性机制后先跑这 10 个，
> 判据成功率不得低于本表，否则第一怀疑对象是 budget/早停而不是模型。

## 冻结坐标（引用任何数字必须带上这三行）

- **tag**：`v1.2.0`（commit `3784d0f`）· 语料指纹：26 篇 / 482 段 / 850 块（chunk_chars=500, overlap=50）
- **模型**：`trae/qwen-3.5` · temperature=0.0 · max_tokens=800 · reasoning_effort=none · max_steps=8
- **判据**：`app/core/llm/eval_tasks.py::TASKS`（判据先写跑后不改，W11 复用同一代码）

## 基线数字（r1/r2/r3 三轮，回归判的是「不低于 9/10」）

| 指标 | r1 | r2 | r3 | 回归底线 |
| --- | --- | --- | --- | --- |
| 判据成功率 | 9/10 | 9/10 | 9/10 | **≥ 9/10，且 × 只允许是 T10 已知判据误伤** |
| 平均步数 | 2.5 | 2.4 | 2.4 | 无硬线；突变 >0.3 要归因 |
| 平均 token | 7712 | 7357 | 7295 | 无硬线；+15% 以上要归因 |
| 平均延迟 | 14.4s | 13.0s | 12.9s | 无硬线 |
| 工具错误率 | 15% | 16% | 16% | 构成必须仍是 notify 环境 + not_found 预期，参数错必须为 0 |

## 错误信封的**真实**结构（W11 故障注入的断言依据）

- 失败：`{"ok": false, "tool": ..., "error_type": <枚举>, "error": {"message": ...}}`
  ——`error_type` 七类枚举定义在 `real_tools.py`（invalid_arg / invalid_json /
  not_found / timeout / internal / ...），side_effect_tools 用同构 `_error()`。
- **`hint` 不在错误信封里**：hint 只出现在**成功且空结果**的 `data` 里
  （如 not_found 的"0 行就是这个问题的答案"）。故障注入断言写
  `error_type`，别写 `hint`。
- 空结果不是错误：`ok=true + chunks=[] + data.hint`，模型该转行而不是报错。

## W7 小样本口径（filter_eval · gold=20）

W7 报告数字（532 块时点）与当前复跑（472 块）**不一致是已归因的语料漂移**：
W8（11-06）扩写 README/api-spec，语料合法增长；rag 层与 harness 自 11-01 零改动
（`git diff v1.1.0..HEAD -- app/rag/ scripts/filter_eval.py scripts/retrieval_harness.py`
为空）。当前漂移量：单文档 90%→85%、按章节 95%→90%、仅表格 100% 不变。
**规则：引用 W7 数字必须带 chunks 坐标；跨时点比较一律先 `corpus_fingerprint.py --update`。**

## 复跑命令（同一口径）

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\run_eval_agent_graph.py --model trae/qwen-3.5 --outdir qa_runs_w11_regression
# 对象：TASKS 全部 10 个（即回归子集本体）；产物 baseline.md 对比上表
```
