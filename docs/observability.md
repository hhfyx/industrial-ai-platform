# W13 周三 · 看板与告警（阈值来自实测拐点）

> 实现：`app/core/alerts.py`（规则引擎 + 静默期 + 恢复）、
> `app/ui/observability_board.py`（四块看板 + 下钻）、
> `app/core/middleware.py` 的 chaos 注入层。
> 阈值唯一合法来源：[alert-thresholds.md](alert-thresholds.md)（W12 拐点数据）。

## 一、看板四块（任务书 09:00–10:00）

| 块 | 内容 | 数据源 |
| --- | --- | --- |
| ① 概览 | 请求量 / 错误率（429 剔除）/ 最差 P95 / 今日成本 | metrics_hourly + usage_ledger |
| ② 延迟分布 | P50/P95/P99 分位序列 + "最慢前三"标注 | metrics_hourly trace 行 |
| ③ 成本 | 按接口成本 / 重试成本单列 / 缓存节省 | metrics_hourly llm 行 |
| ④ 错误与告警 | 错误按路径分布 + 当前触发中的规则 | trace_summary + AlertEngine.status |

**下钻**（10:00–11:20，看板有没有用的分界线）：每格聚合 → 按小时/接口筛选
trace 列表 → 展开 JSONL 里的 span 明细（每个 span 的时长/状态/generation
的 token 与成本）。采样丢弃的 trace 如实标注"明细未落盘，聚合行仍在"。

时间窗 1h/24h/7d 侧边栏切换。**本周 vs 上周对比**：需要 7d 窗口有跨周
数据，当前单周数据时对比列为空——诚实显示，不造假基线（W14 引用时
补足两周数据即可，不需要改代码）。

## 二、告警规则表（每条写清"为什么是这个数"）

| 规则 | 阈值 | 依据（为什么不是行业默认值） |
| --- | --- | --- |
| `retrieval_p95_high` | 端到端 P95 > 20ms | W12 A1–A3 基线 7–10ms（两轮差 ≤4ms），2 倍 = 20ms。行业默认 3s 在本系统**永不触发**——那是自称有监控，不是有监控 |
| `extract_error_rate` | 错误率 > 5%（429 剔除） | W12 全档错误率 0%（429 是背压不算错）。5% 容忍偶发抖动，持续超 = 不是抖动 |
| `llm_retry_share` | 重试占比 > 20% | W11 实测重试个位数占比；对账三漏之首是重试未单列，占比飙升 = 上游不稳，成本先于用户感知恶化 |
| `daily_cost` | 日成本 > 5 元 | W12 单次 extract ~0.0012 元 × 演示环境日调用量级推导；换部署形态时按真实日调用量重估 |

**必答问题**：瓶颈位置（限流配额先坏）、缓存形态（热缓存 ~100%）、
上游配额（key 30 QPS）都是本系统独有的。套用别人的阈值两条路：
天天误报（然后关掉告警）或永远不触发（然后自称有监控）——
两条都通向没有监控。所以规则清单 `RULES` 是文档与代码共用的唯一事实，
每条 rationale 必须指向实测出处（测试 `test_rules_have_rationale` 固住）。

## 三、静默期与恢复（14:40–15:40）

- **幂等键 = `alert-{rule}-{YYYYMMDDHH}`**：同一小时同一规则只发一次
  （W9 通知账本按 key 拦截）；跨小时复发 = 新事件，新键必须能再次叫醒人；
- **静默期 10 分钟**：同一引擎实例内同类告警窗口内只发一次
  （W9 E2 的幂等在"同一事件"层，静默期在"同类事件"层）；
- **恢复通知**：触发 → 条件解除 → 发 `…-recovered`，人分得清
  "还在坏"还是"已经好了"；
- **通道**：复用 W9 `notify_http`（账本 + 重试 + 端点去重三层防线），
  默认 mock 接收端（8920），`ALERT_NOTIFY_ENDPOINT` 换真实通道；
  通道挂了账本仍记录意图——告警丢不了，只是晚到。

## 四、演练记录（15:40–16:40，全部通过）

| 演练 | 注入方式 | 结果 |
| --- | --- | --- |
| ① 慢请求 → P95 告警 | 注入 6 条 120ms search trace，聚合后 P95=120ms（阈值 20） | ✅ `fired value=120`，幂等键 `alert-retrieval_p95_high-2026111223`，mock 端 delivered=True |
| ③ 不重复刷屏 | 同引擎立即再评 | ✅ `silenced`（静默期拦截） |
| ② 5xx → 错误率告警 | 注入 5 条 error trace（错误率 50% > 5%） | ✅ `fired value=0.2`，解除后 `recovered` |
| 恢复通知 | 清除慢/错误 trace 后重评 | ✅ 两条规则都收到 `…-recovered`（delivered=True） |

**注入工具**：`CHAOS_DELAY_MS` / `CHAOS_ERROR_RATE` 环境变量（chaos
中间件，默认关零开销）。演练中发现 chaos 注入在子进程里未生效
（Start-Process 环境继承问题），演练改用 DB 直注——**告警逻辑验证
的是"聚合层看到慢数据后的反应"，注入方式不影响结论**；chaos 层
保留，生产演练时排查子进程继承问题。

单测（`tests/test_alerts_w13.py`，8 条）：规则 rationale 非空、阈值与
文档一致、触发/恢复/静默/空库不评估/429 剔除/幂等键含小时。

## 五、看板三问（19:00–20:30）

| 问 | 在哪答 | 30 秒路径 |
| --- | --- | --- |
| ① 现在哪里慢 | 块② 最慢前三标注 + 分位序列 | 打开即见，无操作 |
| ② 现在哪里贵 | 块③ 按接口成本排行（重试单列） | 打开即见 |
| ③ 现在在失败什么 | 块④ 告警触发状态 + 错误按路径分布 | 打开即见 |

## 六、复跑

```powershell
# 看板
$env:TRACE_DIR='eval/results/traces_wed'
uv run streamlit run app/ui/observability_board.py

# 告警演练（注入慢流量）
$env:CHAOS_DELAY_MS='120'; uvicorn app.main:app --port 8107
# 打请求后另开进程：
python -c "from app.core.alerts import ...; AlertEngine(...).evaluate()"

# mock 通知接收端
python -X utf8 scripts/mock_notify_receiver.py --port 8920
```

## 七、如实边界

- 上周对比：当前只有单周数据，对比列为空（W14 补数据后自动可用）；
- 429 剔除用限流计数差值近似（精确剔除需读 JSONL，不值得）；
- 队列深度告警未进 RULES：它是进程内存态，跨进程评估读不到——
  进程内告警（W12 queue_observe 的延伸）留给部署周；
- bge 口径阈值未定（alert-thresholds.md §二），切 bge 后需重测改表。
