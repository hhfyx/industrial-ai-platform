# W13 周一 · trace 模型与全链路接入（字段表与方案定版）

> 实现位置：`app/core/tracing.py`（唯一模块，约 400 行，零第三方依赖）。
> 本文是字段表与设计决定的记录；埋点分布见文末清单。

## 一、为什么自建，不上 Langfuse

| 方案 | 需要的资源 | 本项目现实 |
| --- | --- | --- |
| Langfuse 自托管 | Postgres + ClickHouse + 两个容器（2GB+ 内存） | 本机同时跑双 compose + LLM 网关 + bge 缓存，再塞一套分析库 = 观测成了主要负载 |
| Langfuse Cloud | 外网依赖 + 数据出境 | 工业语料不出去；断网即失去观测 |
| **自建** | 1 个守护线程 + 1 个 SQLite 文件 | 单实例体量（实测 ~4.5KB/trace） |

Cloud 保留为**可选对照**：span 形状刻意贴近 OTel 语义（trace/span/parent、
status、attrs），将来导出时写 JSONL→OTLP 转换器即可，不用改埋点。

## 二、三级模型字段表

```
trace（一次请求）
 └ span（一段操作）
    └ generation（一次 LLM 调用）
```

### trace（trace_summary 表 / JSONL summary 行）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| trace_id | str | **= request_id**（W8 的日志链不另起编号体系） |
| root_name | str | 根 span 名（http.request / agent.task / generate.task / rag.recall_chain） |
| started_at_ms | int | 根 span 开始时间（epoch ms） |
| duration_ms | int | 根 span 墙钟时长 |
| status | str | ok / error（任一 span error 即 error） |
| span_count / generation_count | int | 规模概览 |
| prompt_tokens / completion_tokens | int | 全 trace 的 generation 累计 |
| cost | float | 按价格表累计（`TRACE_COST_*_PER_1K`，默认 0 = 本机网关不计费） |
| request_path | str | 入口路径 |
| created_at | str | 落盘时间（人读） |

### span（JSONL spans 数组，每条）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| trace_id / span_id / parent_id | str | 树结构；root 的 parent_id=null |
| name | str | 见下方命名表 |
| kind | str | `span` / `generation` |
| start_ms / duration_ms | int | epoch ms 开始 + 墙钟耗时 |
| status | str | ok / error / cancelled（取消也是数据） |
| attrs | object | 自由键值（命名表逐个列出） |

### generation（kind=generation 的 span，attrs 约定）

| 字段 | 说明 |
| --- | --- |
| model | 模型名（进 span，不进 trace 级） |
| prompt_tokens / completion_tokens | usage（网关不回 usage 时记 0 并如实缺省） |
| cost | 按 1k token 价格算 |
| ttft_ms | **流式**：首个非空 content 块到达（真 ttft）；**非流式**：记总时长 + `streaming=false` 如实标注 |
| streaming | true / false |

**为什么三级而不是两级**（必答问题）：span 描述"系统做了什么"，generation
描述"模型做了什么"——两者的成本结构（毫秒 vs 秒、0 元 vs 真金白银）、
延迟形态（恒定 vs 长尾）、失败语义（超时重试 vs 降级拒答）完全不同。
合成一级，就无法回答"这次请求贵在检索还是贵在生成"。

## 三、span 命名表（面板的图例来源）

| name | kind | 挂点 | 关键 attrs |
| --- | --- | --- | --- |
| `http.request` | span | middleware（每请求根） | method / path / status_code |
| `rag.embed` | span | QAEngine._retrieve_timed + cold 端点 | embedder |
| `rag.vector` | span | 同上 | top_k |
| `rag.cache_hit` | span | cold 端点缓存命中路径 | key_head（指纹前 16 位） |
| `rag.bm25` / `rag.fusion` / `rag.rerank` / `rag.compress` | span | 混合检索链路 / compress 唯一入口 | top_k / mode / chunks / keep_ratio |
| `agent.model_step` | generation | agent_graph.model_step | model / step / tokens |
| `agent.tool` | span | agent_graph.tools_node | tool / step / ok / error_type |
| `agent.tool_retry` | span | agent_graph.retry_node | 同上 + retried=true |
| `llm.structured` | generation | structured._request（extract/summarize/decompose 全走这） | model / tokens |
| `llm.stream` | generation | streaming.stream_chat | model / ttft_ms / tokens |
| `llm.qa_answer` | generation | generate_answer._ask（含越界重答，重答=第二个 generation） | model / tokens |
| `generate.task` | span | GenerationQueue._execute | task_id / defect_class / count |

**与 W11 轨迹的关系**：`agent.tool` 的 attrs（ok / error_type / retryable）
与 `_action_record` 用同一批键名——任务轨迹（任务态）与 trace（时序态）
两份数据能直接对上。

**如实说明**：bm25/fusion/rerank/compress 在线链路默认关闭（W6 定版），
这些 span 只在评测脚本（latency_profile.py 已接）或显式开启时出现；
`react_loop.py`（评测 harness，非服务路径）未埋点，需要时照
agent_graph 的两行样式补。

## 四、贯穿要点（三处容易踩的坑，都已处理）

1. **create_task 拷贝 collector 引用**：Agent 后台任务在请求 flush 之后
   继续写会丢 → `_run_agent_task_with_trace` 用 trace_scope 开**新
   collector**，后台任务独立成 trace（trace_id = task_id）。
2. **长驻 worker 不继承 contextvars**：生成队列 worker 在 submit 时把
   `get_trace_id()` 存进 `GenerationTask.trace_id`，执行时显式 set/reset。
3. **埋点绝不同步写盘**（今天的坑）：请求路径只做内存 list append +
   queue put（微秒级）；单守护线程落 JSONL + SQLite；写失败计数
   （`/metrics` 的 `tracing.write_errors`）绝不阻塞业务。

## 五、采样与体积

- 采样：错误/取消 trace **100% 保留**；正常按 `TRACE_SAMPLE_RATE`
  （默认 1.0——本阶段要测磁盘增长；生产建议 0.1–0.2）。
- 体积：大文本只落 `sha256_16 + len + head(200)`（clip_text）。
- 增长实测（2026-11-12 冒烟）：7 条 trace = 31.3KB ≈ **4.5KB/trace**
  （Agent 长轨迹 9 spans 占大头）。全采样 1 万请求 ≈ **44MB**，
  SQLite 聚合表单独 12KB——按 0.2 采样率长期跑是每天 MB 级，可控。

## 六、验证记录（任务书 19:00–20:30 三条）

| # | 验证 | 结果 |
| --- | --- | --- |
| ① | 每条 trace 完整：search（含缓存命中）、extract、agent-tasks 三类请求，一次请求一条 trace，root=http.request，响应头 X-Trace-Id=X-Request-ID | ✅ |
| ② | 错误请求也有 trace：422 落 status=error，100% 采样；后台任务异常同样 | ✅ |
| ③ | 断网时观测不阻塞业务：sink._write 必炸的注入测试 + 全套 pytest 176 条全绿 | ✅ |

冒烟实测（smoke2，uvicorn:8107）：

```
[ok] http.request: http.request(72ms) > rag.embed(0ms) > rag.vector(0ms)
[ok] http.request: http.request(1ms) > rag.cache_hit(0ms)
[ok] agent.task: 9 spans > 5×agent.model_step(generation, tok 10354+195) > agent.tool ×2
[error] http.request(422): status=error, 100% 采样落盘
```

## 七、复跑

```powershell
# 冒烟（三类请求 + trace 检查）
$env:TRACE_DIR='eval/results/traces_smoke'
uvicorn app.main:app --port 8107
# 打 search / extract(422) / agent-tasks 各一条，然后：
python -X utf8 -c "..."  # 逐行读 traces.jsonl 打印 span 树

# 测试
pytest tests/test_tracing_w13.py -q   # 12 条
pytest tests/ -q                       # 全量回归
```

环境变量：`TRACE_DIR`（默认 data/traces）、`TRACE_SAMPLE_RATE`（默认 1.0）、
`TRACE_COST_INPUT_PER_1K` / `TRACE_COST_OUTPUT_PER_1K`（默认 0）。
