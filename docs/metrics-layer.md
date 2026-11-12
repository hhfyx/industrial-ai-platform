# W13 周二 · 指标层与成本核算（清单 + 对账 + 映射表）

> 实现：`app/core/metrics.py`（成本引擎 + 直录账本 + 小时聚合 + 对账），
> HTTP 视图 `/metrics/cost` 与 `/metrics/hourly`。周一的 trace 库
> （`trace_summary`）是本层的数据底座，本文件记录清单、对账与映射。

## 一、三层指标清单（定版）

| 层 | 指标 | 来源 | 口径注记 |
| --- | --- | --- | --- |
| 流量 | 请求总数 / QPS（按小时） | `metrics_hourly` kind=trace | 按接口拆行 |
| 流量 | 并发 / 队列深度 | W12 既有（压测 / GenerationQueue） | 进程内存态，不进本表 |
| 质量 | P50/P95/P99 | trace duration 分位 | **端到端**口径；服务端 elapsed 单独看 X-Elapsed-Ms |
| 质量 | 首 token（TTFT） | generation span `attrs.ttft_ms` | 流式真值；非流式记总时长 + streaming=false |
| 质量 | 错误率 | trace status=error 计数 | **429 与 5xx 分开**：429 是限流背压不进错误率（W12 判读约定） |
| 质量 | 降级率 | `degraded` 标记（extract/agent registry） | 降级 ≠ 失败，两者分开报 |
| 成本 | token/请求、小时/日成本 | generation span + usage_ledger | 唯一公式见 §二 |
| 成本 | 缓存命中率 → **省下的钱** | ResultCache.hit_rates + 中位数基线 | 见 §二"缓存省钱口径" |
| 成本 | 重试成本 | ledger `retry=1` 子集，**单列** | 对账三漏之首 |

## 二、成本核算

**唯一成本公式**（`metrics.compute_cost`，全项目只此一处）：

```
cost = prompt_tokens / 1e6 × price_in + completion_tokens / 1e6 × price_out
```

单价：输入 2 元 / 输出 8 元每百万（`docs/cost.md` W2 D1 示例锚点，
**不是任何供应商报价**；改 `COST_PRICE_IN/OUT` 环境变量换锚点）。

**缓存省钱口径**：命中 = 省一次等价调用。省多少按"同 endpoint 未命中
成本的中位数"滚动估计（窗口 50 笔，样本 <5 不下结论返 0）——不拍脑袋
写死，估计口径随流量形态自动更新。

**重试单列**：ledger 带 `retry=1`，聚合进 `retry_cost` 列。失败的尝试
同样计费（docs/cost.md 的 3.5 倍教训），不单列就会被"正常成本"吞掉。

## 三、对账机制（本周关键）

### 3.1 锚点的选择

本机网关（8326）**没有账单汇总端点**（实测 /v1/usage、/dashboard/billing/usage
等全部 404；/v1/models 正常，93 个模型）。所以对账锚点取
**逐笔 usage 直录**：五个 LLM 入口每次调用的 `response.usage`
（含 `prompt_tokens_details.cached_tokens`）原样落 `usage_ledger`
表（只 INSERT，永不 UPDATE/DELETE）。

对账 = `trace_summary`（观测聚合侧）vs `usage_ledger`（直录锚点侧），
逐 token + 逐成本比对，误差 >5% 必查。`/metrics/cost` 直接给出结论。

### 3.2 三个入口必须知道的坑（均已实测踩过）

**入口清点（漏一格 = 对账缺口）**：

| # | 入口 | 覆盖 | 挂点 |
| --- | --- | --- | --- |
| 1 | `llm.structured`（structured.py） | summarize / chat / 兜底 | `_request` |
| 2 | `llm.stream`（streaming.py） | SSE 流式 | 末 usage 块 |
| 3 | `llm.qa_answer`（generation/base.py） | RAG 问答（含越界重答） | `_ask` |
| 4 | `agent.model_step`（agent_graph.py） | Agent 模型步 | model_step |
| 5 | `llm.extract`（llm_extractor.py） | **extract 主路径** | `_call`（20 条验证时补） |
| 6 | `llm.plan_decompose`（guardrails.py） | judge/guard 类（任务分解） | decompose_task |

第 5 条是 20 条验证**实际抓到的缺口**：extract 走的是 LlmExtractor
自建的 `_request`，不走 structured.py——前四个入口都埋了，它没埋，
第一轮对账 `calls=10（只有 agent）`而 extract 的 10 笔调用凭空消失。
**这就是"对账的意义"的实证**：不对账，extract 的成本整体低估 45%
（0.0093 / 0.0447）。

### 3.3 三类漏记的自动 signature（>5% 时 /metrics/cost 给出）

| 漏记 | signature |
| --- | --- |
| 漏记重试 / judge | trace 侧 generation 总数 < 直录调用总数（多出来的调用没进观测面） |
| 漏记 judge/guard | 直录有调用、但没有任何一条 trace 带 generation |
| 流式 usage 未开 | 直录有调用但 token 全 0（钱烧了没记数） |

### 3.4 第二个被对账抓出来的问题：成本公式分叉

周一 `record_usage` 自带一套 `TRACE_COST_*`（默认 0），metrics 又有
一套 `COST_PRICE_*`——同一笔调用两个价。第二轮 20 条验证里
token 逐位一致但 `cost_diff=1.0`（trace 侧全 0）。修复：tracing.py
调用 `metrics.compute_cost`，公式回到唯一出处。

## 四、验证记录（任务书 19:00–20:30）

第三轮 20 条（10 extract + 4 search 含缓存命中 + 4 错误路径 + 2 Agent）：

```
对账：within_5pct=True  token_diff=0.0  cost_diff=0.0  signals=[]
ledger:   calls=18  pin=18642  pout=903  cost=0.044508 元
aggregate: 同上，逐位一致
```

手核 3 条（成本公式复算，与 ledger 记录比对）：

| # | endpoint | pin | pout | 记录成本 | 复算 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | llm.extract | 223 | 64 | 0.000958 元 | ✅ 一致 |
| 2 | llm.extract | 217 | 17 | 0.000570 元 | ✅ 一致 |
| 3 | llm.extract | 217 | 133 | 0.001498 元 | ✅ 一致 |

测试：`tests/test_metrics_w13.py` 11 条（公式/账本/聚合幂等/对账三
signature/HTTP 视图），全套 190+ 全绿。

## 五、W11 / W12 对齐映射表（同一口径，不新造）

| W11/W12 的指标 | 新指标层的出处 | 说明 |
| --- | --- | --- |
| W11 成功率（20 任务 70–100% 区间） | `trace_summary.status`（agent.task 根 span error 率） | 判据冻结不动；trace 只是同一事实的时序投影 |
| W11 降级率 ~10% | registry `degraded` 标记（agent_tasks / extract 契约字段） | 降级计入失败与否，沿用 W11 口径（计入） |
| W11 平均 token 5662→6248 | trace_summary.prompt+completion / generation_count | usage_log 与 generation span 同源（W11 轨迹字段） |
| W11 工具错误率 17.6%→11.1% | agent.tool span attrs.ok=false 比例 | attrs 直接复用 `_action_record` 键名 |
| W11 预算拐点 10k | budget_used（W11 冻结）→ 后续可投影进 trace attrs | 不重算 |
| W12 分段耗时表（鉴权/缓存/检索/序列化） | rag.embed / rag.vector span + X-Elapsed-Ms | 分段口径不变，观测化 |
| W12 拐点 = 限流配额（429 55.7%） | `/metrics` rate_limit_rejections（W12 既有） | 429 不进错误率的判读约定延续 |
| W12 队列深度 32 触顶 | GenerationQueue（W12 既有，进程内存态） | 不进 SQLite（重启即失真的数不落盘） |
| W12 缓存命中率 ~100% | ResultCache.hit_rates → /metrics + 省钱估计 | 口径同 W12 压测（热缓存） |

**纪律**：W11 的评测数字（20 任务成功率等）仍以 `scripts/verify_resume_numbers.py`
为唯一真值源；指标层只是把**服务运行时**的同名事实对齐到同一口径，
绝不反过来改评测数字。

## 六、复跑

```powershell
# 服务
$env:TRACE_DIR='eval/results/traces_tue3'; uvicorn app.main:app --port 8107
# 打 20 条（见 tests/test_metrics_w13.py 与本文件 §四的构成）
# 看对账与聚合
curl http://127.0.0.1:8107/metrics/cost
curl http://127.0.0.1:8107/metrics/hourly
# 测试
pytest tests/test_metrics_w13.py -q
```

环境变量：`COST_PRICE_IN`（默认 2）、`COST_PRICE_OUT`（默认 8）、
`TRACE_DIR`（默认 data/traces）。
