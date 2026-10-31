# industrial-ai-platform

**一个从"调模型"长到"能查、能引、能拒答"的文档智能平台。**

四块能力，都是同一条流水线上的前后段：

| 能力 | 入口 | 一句话 |
| --- | --- | --- |
| **知识库问答**（主交付） | `app/ui/streamlit_app.py` | 上传文档 → 检索 → **答案带 `[n]` 引用** → 点开看原文高亮 |
| 结构化抽取 | `POST /api/v1/extract` | 抽不到就是 `null`，永不编造（有原文回指校验） |
| 超长文档摘要 | `POST /api/v1/summarize/stream` | map-reduce + 流式，50 页 ≈ 1 秒索引 |
| 流式对话 | `POST /api/v1/chat/stream` | SSE 逐字，TTFT 有口径 |

重点不在"调通几个接口"，而在四件能拿证据说话的事：

- **接口不变、实现可换**：同一份 `v1` 契约，底下从正则换成真实模型（再换成
  JSON Schema 强约束 + 校验回灌），`tests/test_api.py` **一行未改**。
- **测试能离线跑绿**：**567 个测试，拔掉网线也全绿** —— 这是"实现真的可替换"
  无法伪造的证据。
- **延迟拆开打点**：一次问答拆成 检索 / 编码 / 填上下文 / 首 token 四段。
  实测检索只占首 token 的 **0.36%~0.80%** → 优化检索对首字毫无帮助。
- **知道它哪里不好**：见 [v1 已知问题](#v1-已知问题)。**这一节比效果数字重要。**

| 项目 | 值 |
| --- | --- |
| 端点 | `extract` · `extract/stream` · `chat/stream` · `summarize/stream` |
| Web UI | Streamlit（`:8501`）· 免费层**自动降级**为字面检索（见 [deploy-notes](docs/deploy-notes.md)） |
| 核心契约 | `POST /api/v1/extract`，`v1.1`（冻结；新增可选字段 `degraded`） |
| 测试 | **567 个，全离线**（拔网线也全绿），覆盖率 80% |
| 检索 | bge-small-zh（512 维，CPU 单条 10ms）· ChromaDB 持久化 · cosine |
| 运行时 | Python 3.11 + uvicorn + Streamlit，非 root（uid 10001） |

> **当前状态（v0.4.0）**：RAG 全链路在本地跑通并有测试；
> **真模型问答与线上链接未验证**（本机网关 chat 端点 502 未配置凭据）。
> 这条写在 [已知问题](#v1-已知问题) 第 1 条，不藏着。

---

## 这个项目解决什么

### ① 知识库问答（主交付）

```powershell
uv run streamlit run app/ui/streamlit_app.py     # http://localhost:8501
```

上传 PDF/DOCX/Markdown → 点索引（三阶段进度条 + 四段耗时）→ 提问 →
逐字出字 → 展开「参考来源」看原文（**依据句高亮**）。

答案里的每个 `[n]` 都能追溯到 `文件名 + 页码`；引用了不存在的编号会**显式告警**
（"编号 [9] 不在检索结果里"），因为那是模型在编。

### ② 结构化抽取

把一段非结构化文本（合同、工单、简历）按调用方指定的字段抽成结构化数据：

```http
POST /api/v1/extract
{"text": "合同编号：HT-2026-001，甲方：张三，签订日期：2026-10-20", "fields": ["合同编号", "甲方"]}
```

```json
{"request_id": "…", "data": {"合同编号": "HT-2026-001", "甲方": "张三"}, "confidence": 1.0, "elapsed_ms": 101}
```

三个设计取向：

1. **抽不到就是 `null`，永不编造。** 不只写在 prompt 里，`LlmExtractor._ground_value`
   会做**原文回指校验**：模型给出的值必须在正文里能找到依据，否则丢弃为 `null`。
2. **所有非 2xx 是同一个 4 字段形状**（`code` / `message` / `request_id` / `details`）。
3. **实现可换，契约冻死。** `docs/api-spec.md` 是唯一不能动的东西。

---

## 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  Web UI（Streamlit，:8501）    app/ui/                               │
│    streamlit_app.py   上传 / 索引 / 流式 / 参考来源展开               │
│    evidence.py        LCS 定位依据句 → 原文高亮（不是整段糊出来）      │
└────────────────────────────┬────────────────────────────────────────┘
                             │ 只用四个装配点：get_embedder / get_store
                             │                Indexer / QAEngine
┌────────────────────────────▼────────────────────────────────────────┐
│  RAG 编排（无 fastapi、无 streamlit）        app/rag/                │
│                                                                     │
│   ① 索引段（离线，一次）                                             │
│       loaders/   PDF 逐页 → Document(page)                          │
│                  ├ pdfplumber 表格 → Markdown（不进文字流）          │
│                  ├ docx/markdown 标题层级进 metadata                 │
│                  └ clean.py 四类噪声四种判据（不许改写正文）          │
│       splitters/ 中文分隔符优先级（。！？；，），只合并完整原子片       │
│       embedding/ bge-small-zh 512 维（或 hashing 降级）              │
│                  vector.py 归一化 + 余弦（全链路唯一相似度入口）      │
│       stores/    Chroma 持久化 / 内存（同 Protocol）                 │
│                                                                     │
│   ② 检索段（每次）   query→向量→cosine→TopK → 阈值闸门                │
│   ③ 生成段（每次）   拼上下文[1][2]… → 流式调模型 → 解析[n] → 引用    │
│                     query.py  QATiming：embed/search/context/ttft    │
└────────────────────────────┬────────────────────────────────────────┘
                             │  ExtractorDep / get_llm_client
┌────────────────────────────▼────────────────────────────────────────┐
│  API 层（FastAPI，:8000）                   app/api/v1/ · app/core/   │
│    extract.py     结构化抽取（v1 契约，冻结）                        │
│    stream.py      抽取流式（SSE，event: end）                        │
│    chat.py        聊天流式（SSE，[DONE]）                            │
│    summarize.py   文档摘要流式（上传 + SSE）                          │
│                                                                     │
│    core/     deps.py（Extractor Protocol + 回退）· config.py          │
│              errors.py（统一错误体）· sse.py（断开即停）              │
│    core/llm/ client.py 进程级单例 · streaming.py TTFT 口径            │
│              retry.py 指数退避 · structured.py 校验回灌 ≤2 轮         │
│                                                                     │
│    services/ llm_extractor.py（默认）· extractor.py（规则兜底）        │
│    prompts/  {qa,summarize,classify,extract}/vN.jinja2 + 硬编码检查   │
└─────────────────────────────────────────────────────────────────────┘
```

**跨层规则是硬的**：

| 规则 | 为什么 |
| --- | --- |
| `app/rag/` 与 `services/` **不 import fastapi** | 依赖注入是路由层的事；服务层只接受构造参数 |
| `app/rag/` **不 import streamlit** | UI 只有展示职责。所以 567 条测试**不需要 Streamlit** 就能跑 |
| UI **不 import** rag 内部实现 | 只用四个装配点，改检索逻辑不影响界面 |
| 相似度**只有一个入口**（`vector.cosine`） | 没有"直接算内积"的旁路，防住"忘了归一化"这个静默 bug |

这三条不是洁癖 —— 它们各自对应一个**会静默出错**的场景。详见
[rag-architecture.md](docs/rag-architecture.md) 与 [streamlit-ui.md](docs/streamlit-ui.md)。

### 目录

```
app/
├── ui/              ★ Web UI（主交付）
│   ├── streamlit_app.py  上传 / 索引进度 / 流式 / 引用展开 / 四类健壮性
│   └── evidence.py       LCS 找依据句 → 原文高亮
├── rag/             RAG 三段骨架（W4）
│   ├── schema.py    ★ Document / Chunk（page 从第一步就在结构里）
│   ├── query.py     QAEngine + QATiming（分阶段打点）
│   ├── deploy_config.py  免费层降级决策（hashing vs bge）
│   ├── loaders/     ① 解析
│   │   ├── clean.py     四类噪声四种判据 + "清洗不改写正文"的红线
│   │   ├── pdf.py       pypdf 逐页 + pdfplumber 表格 → Markdown；跨页拼接记 span
│   │   ├── docx.py      段落 + 标题层级进 metadata；表格同 PDF 格式
│   │   └── markdown.py  按 # 分节；围栏内的 # 与 | 都不算结构
│   ├── splitters/   ② 切分
│   │   ├── base.py        ChunkSize（含 overlap<chunk 约束）+ Splitter Protocol
│   │   ├── delimiter.py   DelimiterSplitter：中文分隔符优先级 + 只合并完整原子片
│   │   └── recursive.py   兼容层（周三的 import 路径 → DelimiterSplitter）

│   ├── embedding/   ③ 向量化
│   │   ├── vector.py    归一化 + 余弦（全链路唯一的相似度入口）
│   │   ├── bge.py       bge-small-zh-v1.5（512 维 / 512 token；query 侧自动加前缀）
│   │   └── hashing.py   字符 n-gram 哈希（无语义；CI 与降级用）
│   ├── stores/      ④ 入库
│   │   ├── base.py      VectorStore Protocol（score 钉死为余弦相似度，越大越像）
│   │   ├── chroma.py    ChromaStore：PersistentClient + cosine + where/页码过滤
│   │   └── memory.py    InMemoryVectorStore（不持久化，测试用）
│   ├── generation/  base.py（⑥ 生成：带 [n] 引用的答案）
│   ├── indexing.py  Indexer + IndexReport（串 ①②③④）
│   ├── retrieval.py Retriever（⑤ query→TopK，只调 embed_query）
│   ├── pipeline.py  RagPipeline（唯一装配点：强制共用 embedder 与 store）
│   ├── summarize.py map-reduce + 流式（W4 前的代码，本次只把 Chunk 改名 SummarySegment）
│   └── replay.py    DEMO_REPLAY：录屏/联调用的同构回放
├── services/        extractor.py（规则+extract_rule_based）· llm_extractor.py（LLM+校验回灌）
├── schemas/         extract.py（含可选 degraded）· chat.py
└── cli/             summarize.py · ask.py（RAG 问答：python -m app.cli.ask）
app/ui/              streamlit_app.py（Web UI：上传 / 索引 / 流式 / 引用展开）
                     └ 启动：streamlit run app/ui/streamlit_app.py  → :8501
scripts/             smoke_llm · demo_tools · consume_sse · cost · cost_model
                     latency_breakdown · record_demo · measure_tokens · drill_*
                     rag_cost_compare（长上下文 vs RAG：成本/注意力/更新代价）
                     bge_prefix_ab（query 指令前缀 A/B，走真实链路）
                     chroma_acceptance（100 条入库 + 10 条已知答案）
                     rag_e2e_demo（完整链路：引用 / 越界编号 / 拒答）
                     embedding_score_probe（阈值可分性）· measure_memory（内存实测）
                     rag_bench（五个耗时数字 + 检索占首 token 多少）
                     regression_w4（跨层回归：API/prompts/summarize/CLI/AB）
static/index.html    前端单页（fetch + ReadableStream，含中断/重试）
tests/               567 个**全离线**测试
  ├── fake_llm.py    FakeLLMClient（可复用，支持固定回复/脚本/异常/流式）
  ├── conftest.py    autouse: 默认注入离线规则版（拔网线也全绿）
  ├── test_rag_skeleton.py  RAG 骨架 48 条：结构不变量 + 三段链路
  ├── test_loaders.py     统一解析 70 条：页码 / 表格 / 扫描件 / 清洗不变量
  ├── test_embedding.py   向量层 24 条：手写余弦 vs numpy / 归一化契约 / query 前缀
  ├── test_chroma_store.py 持久化 39 条：distances 方向 / 重启后数据在 / where 过滤
  ├── test_rag_query_chain.py 问答链路 38 条：中文边界 / metadata 继承 / 引用解析 / 拒答
  ├── test_rag_query_timing.py 编排与打点 21 条：事件顺序 / 分阶段耗时 / 拒答不调模型
  └── fixtures/      make_fixtures.py 脚本生成样本（review 可读）
app/prompts/         模板库：{qa,summarize,classify,extract}/vN.jinja2 + schema.py
  ├── loader.py      load_prompt(name, version) · 版本切换 · 硬编码业务词检查
  └── hardcoded_blocklist.txt  业务词黑名单（模板含它就拒绝渲染）
eval/prompt-sets/    prompt A/B 输入池（4 类 × 10 条，每条带判定标准）
docs/                api-spec.md（冻结契约 v1.1）· cost.md · sse.md · SETUP.md
                     prompt-patterns.md（四范式实测）· system-prompt.md（四要素 + A/B）
                     prompt-templates.md（模板库）· structured-output.md（structured_call）
                     ab-scaffold.md（输入池 + 脚手架）· prompt-ab.md（四类 A/B 报告）
                     rag-architecture.md（三段全景 + 每段输入输出）·
                     document-parsing.md（统一解析：页码为何错位 / 表格 / 扫描件 / 清洗红线）·
                     embedding.md（归一化为何必须 / bge 实测 / query 前缀 A/B / API vs 本地）·
                     chroma-store.md（distances 方向 / 持久化 / metadata 规范 / 100 条验收）·
                     rag-query-chain.md（overlap 必答 / 中文边界 vs LangChain / 引用三层 / 拒答阈值）·
                     streamlit-ui.md（检索占首 token 多少 / 流式事件顺序 / 引用展开 / 四类健壮性）·
                     deploy-notes.md（免费层内存实测 / hashing 降级 / 5 个部署坑）·
                     design-decisions.md（6 条决策 + 代价 + 长上下文必答）
                     why-i-use-langchain.md · container-networking.md
```

---

## 快速开始（三条命令）

```powershell
uv sync                                                      # ① 装依赖
Copy-Item .env.example .env                                  # ② 填 LLM_API_KEY / BASE_URL / MODEL
uv run streamlit run app/ui/streamlit_app.py                 # ③ 打开 http://localhost:8501
```

打开后：上传 `tests/fixtures/sample50.pdf` → 点「索引」→ 问「违约金怎么算」
→ 展开「参考来源」看原文高亮。

<details>
<summary>API 侧（可选）</summary>

```powershell
uv run uvicorn app.main:app --reload      # http://127.0.0.1:8000/docs
curl.exe http://127.0.0.1:8000/health     # {"status":"ok"}
```

</details>

> **首次用 `bge-small-zh` 会下载 92 MB 模型**（加载约 16 秒）。国内直连
> `huggingface.co` 会 DNS 失败，代码默认走 `hf-mirror.com`；也可自己设
> `HF_ENDPOINT` / `HF_HOME`（`HF_HOME` 建议指到本项目 `.cache/hf`，别落 C 盘）。
>
> **不想下载？** 默认就是 `hashing`（1 MB，纯 CPU，只有字面检索）。
> 语义检索才需要 bge。

`.env` 里 5 个必需变量：

| 变量 | 说明 |
| --- | --- |
| `LLM_API_KEY` | 必填；空字符串会被启动校验拦下（比留空跑到 401 好） |
| `LLM_BASE_URL` | OpenAI 兼容端点，含 `/v1`；换供应商只改这一行 |
| `LLM_MODEL` | 模型名（**必须支持 `role=tool` 消息**，见下文「踩过的坑」） |
| `LLM_TIMEOUT` | 单次 HTTP 往返超时（秒） |
| `LLM_MAX_RETRIES` | SDK 内部重试，`0–5`；整体最坏耗时 ≈ `timeout × (1+max_retries)` |

验证：

```powershell
curl.exe http://127.0.0.1:8000/health
# {"status":"ok"}

curl.exe -s -X POST http://127.0.0.1:8000/api/v1/extract `
  -H "Content-Type: application/json" `
  --data-binary "@body.json"
```

> Windows 上必须用 `curl.exe`（PowerShell 的 `curl` 是 `Invoke-WebRequest` 别名），
> 且 JSON 用 `--data-binary "@file"` 传，行内引号会被 PowerShell 吃掉。

---

## v1 已知问题

> **这一节是这份 README 里最该被读的部分。**
> 召回率不高不是问题（任何人第一次都这样）；**不知道它哪里不好**才是问题。
> 每条都写清"现状 / 为什么 / 怎么修"。

### 🔴 阻断类（会让线上不能用）

| # | 问题 | 现状 | 为什么 | 怎么修 |
| --- | --- | --- | --- | --- |
| 1 | **真模型问答未验证** | 本地网关 chat 端点返回 `502 未配置凭据`，所有真实问答路径**没跑过** | 环境限制，不是代码缺陷 | 换一个可用的 `LLM_API_KEY` 重跑 `scripts/rag_e2e_demo.py` 与线上六步清单 |
| 2 | **线上链接未创建** | Streamlit Cloud 需要 GitHub 账号授权，我这边只有本地 git | 没有账号级凭据 | 手动在 share.streamlit.io 新建 app，入口填 `app/ui/streamlit_app.py` |
| 3 | **免费层语义检索降级** | 线上默认 `hashing`（只有字面相似） | bge 实测占 **528 MB**，加框架开销撞 1 GB 上限 | ONNX 量化版（未验证）/ 付费层 / 换 API embedding |

### 🟡 能力类（能用但达不到"好"）

| # | 问题 | 现状 | 为什么 | 怎么修 |
| --- | --- | --- | --- | --- |
| 4 | **召回率没有基线** | 只有 10 条人造语料的 top3 人工判断（命中 8/10） | 没有业务文档、没有标注集 | W5 建评测集，跑 hit@k / MRR |
| 5 | **TopK 与 overlap 凭直觉** | `top_k=5`、`overlap=50` **都没有数据支撑** | 没做过消融 | W5 做 `overlap ∈ {0,50,100}` × `top_k ∈ {3,5,8}` 的矩阵 |
| 6 | **纯向量检索对关键词强查询失败** | "设备送到什么地址"命中失败（top3 全是带"设备"的条款） | 向量相似度对专有名词/地址不敏感 | 混合检索（BM25 + 向量），或加 query 改写 |
| 7 | **引用只是"存在"不是"正确"** | 校验了编号是否在给定集合内，**没校验引用的内容是否真支撑那句结论** | 后者需要模型判断或人工标注 | 用 evidence 的 LCS 命中率做弱校验（已有雏形），下一步做严格校验 |
| 8 | **多轮不解决指代** | 保留 3 轮历史，但"它/那个条款"无法关联到文档 | 需要 query 改写 + 实体链接 | W8 |
| 9 | **表格超 512 token 被截断** | bge `max_seq_length=512` ≈ 757 字；2000 字的大表截断 | 表不切分（切碎更糟）是刻意取舍 | 表格单独摘要后再索引 |

### 🟢 工程类（不影响 demo，影响交付质量）

| # | 问题 | 现状 | 为什么 | 怎么修 |
| --- | --- | --- | --- | --- |
| 10 | **多进程写 Chroma 会锁** | 单 worker 正常，多 worker 未测 | SQLite 后端写并发差 | 换 Milvus，或加写队列 |
| 11 | **并发会话内存未测** | `cache_resource` 是进程级单例 | 需要线上观测 | 压测 / 限流 |
| 12 | **docker compose 没接 UI** | 只有 API service | 时间分配给了 README | 加 `ui` service（端口 8501:8501） |
| 13 | **无增量更新** | 只有整份重建 | `delete_by_source` 有了，但没做"改一页只重算那一页" | 按 `store_id` 比对差异 |
| 14 | **无权限隔离** | 所有用户共享一个库 | demo 阶段不需要 | 按 `metadata` 过滤（结构已支持：扁平标量） |

### 如果我只有一天来继续改，我会按这个顺序

1. **解决凭据 → 跑真模型**（问题 1）。没有真模型，后面所有质量结论都是空的。
2. **建 50 条评测集**（问题 4）。有了它，问题 5/6/7 才从"凭感觉"变成"有依据"。
3. **混合检索**（问题 6）。它是最便宜的质量提升，且能证明"知道哪里不好"。

---

## LLM 接入：三个可验证的结论

这一块不是"能调通"，而是每条都有实测数据支撑。完整记录在
[`learn/llm/`](../learn/llm/)。

### 1. `temperature=0` 不等于「每次都一样」

同一 prompt、`temp=0` 跑 5 次，出了 **3 种**不同输出（差异只落在末句一个词上）。
`seed` 在本机网关上实测**不可复现**：`temp=0 + seed=42` 五次仍出 3 种，
且换 `seed` 有时完全相同、有时不同。

**所以确定性要自己实现**（缓存 / 校验 / 一致性投票），不能指望参数。
对我们的直接影响：**测试里不能断言"模型输出了某个具体字符串"**，
只能断言"字段合法、契约成立"。

### 2. `finish_reason` 才是截断的唯一判据

`max_tokens` 设小时，内容可能停在句号之后、肉眼看像完整句子，但
`finish_reason == "length"` 说明它是被硬砍的。只看长度会误判。

顺带一个坑：推理模型默认开启时，思维链会吃光 `max_tokens`，`content` 返回 `null`。
接入层必须把 `content is None` **当失败**处理，不能 `or ""` 静默成空串 ——
空串会被下游当成"模型说没有"，那是语义错误。

### 3. Function Calling 不是模型执行函数

模型只输出 `tool_calls`（工具名 + 参数 JSON），**执行 100% 在我们这侧**。
`TOOL_REGISTRY[name](**args)` 是我们自己的 Python 在跑。

这个边界决定了一切安全设计：模型会幻觉出不存在的工具名（实测 `send_email`），
会给出类型错的参数，所以 `execute_tool` 的铁律是**任何输入都不抛异常**，
失败也返回结构化数据回灌，让模型下一轮自愈。

工具层还包含：`get_weather`（假数据）、`calculate`（四则运算），
以及并行 `tool_calls` 的按 id 配对回灌、3 轮上限（防无限计费）。

### 成本

基于**真实 usage** 测算（脚本 `scripts/cost_report.py`，单价可配置）：

| 样本 | tokens/次 | 每 1000 次（示例单价） |
| --- | --- | --- |
| 短文本（22 字 / 2 字段） | 198 | 0.56 元 |
| 标准合同（109 字 / 4 字段） | 316 | 1.17 元 |
| 长合同（2,636 字 / 6 字段） | 1,996 | 4.46 元 |

> 单价取"输入 2 元 / 输出 8 元每百万 token"作为示例锚点，**不是任何供应商报价**。
> 关键结论是成本结构：**输出单价通常是输入的 3–5 倍**，
> 所以「少让模型说话」比「少喂正文」更省。
>
> **完整成本模型（含重试期望成本、优化敏感性）见 [docs/cost.md](docs/cost.md)。**
> 一句话结论：回灌重试不是"多一点时间"，是**多 3.5 倍钱**。

---

## 测试能离线跑绿 —— W2 唯一无法伪造的成果

```powershell
pytest -q          # 567 passed，拔掉网线也一样
```

这 行命令为什么值得单独一节：**它能证明实现真的可替换**。
如果测试依赖真实模型，那它测的就不是你的代码，而是供应商的可用性。
W4 加检索、W9 加工具、W12 加缓存，都不会破坏已有契约 —— 因为契约被**离线**测试锁住了。

### 这不是"顺便做到"，是踩出来的

**2026-10-27 的实测事故**：网关凭据过期返回 `502 未配置凭据`，
`LlmExtractor` 于是降级到规则版，响应体多出 `degraded` 键 ——
**契约测试 `test_api.py` 直接挂了**，而那个测试本身一行都没改、也没写错。

那一刻暴露了真问题：**任何依赖外部服务的测试都是定时炸弹**。

### 三层离线策略

| 层 | 手段 | 覆盖什么 |
| --- | --- | --- |
| **单测** | `tests/fake_llm.py` 的 `FakeLLMClient` | 校验回灌、降级、键序、编造拦截 |
| **HTTP 层** | `httpx2.MockTransport` 注入 | SDK 请求构造、`response_format` 是否真的发出 |
| **API 层** | `conftest.py` 的 `autouse` fixture | 默认注入规则版，`test_api.py` 一行不改仍全绿 |

### 为什么 HTTP 层不用 respx（实测做不到）

任务要求"集成测试用 respx 拦截 HTTP"。**实测失败**：

```
httpx  : 0.28.1
httpx2 : 2.16.0
httpx is httpx2 -> False      ← openai 3.37 用的是 httpx2（独立分叉）
respx 只 patch httpx          → 请求真实发到了网络
```

改用 SDK 官方支持的 `http_client=` 注入 `httpx2.MockTransport` ——
在**传输层**拦截，比 respx 的库级 monkey patch 更底层也更稳。
这个事实写成了测试（`test_respx_cannot_intercept_this_sdk`），
将来 SDK 若换回 `httpx`，那条测试会红，提醒我们可以简化。

### `FakeLLMClient` 的契约

它不是 mock，是 **stub** —— 不验证"你调了几次哪个参数"，
而是"给定这个输入，按剧本返回这个输出"。这样测试断的是**业务行为**，不是实现细节。

```python
# 固定回复
FakeLLMClient(replies=['{"甲方": "张三"}'])

# 脚本：第一轮坏、第二轮好（测回灌）
FakeLLMClient(replies=['{"甲方": {"nested": 1}}', '{"甲方": "张三"}'])

# 直接抛异常（测降级）
FakeLLMClient(exc=TimeoutError("boom"))

# 流式（测 SSE / streaming）
FakeLLMClient(stream_text="杭州很美")
```

---

## 换实现：三层约束 + 校验回灌 + 降级标记

### 三层约束，从"软"到"硬"

| 层 | 手段 | 拦的是什么 |
| --- | --- | --- |
| ① prompt | 明确要求只输出 JSON 对象 | 让模型**倾向**正确（不可靠） |
| ② JSON Schema | `response_format={"type":"json_schema", strict: true}` | 结构坏（漏键、多键） |
| ③ Pydantic | 本地校验 `{field: value}` | Schema 也漏掉的语义问题 |

**三层都要**：`response_format` 并非所有网关都真正支持（本机网关接受参数但未必强制），
只靠 prompt 时坏 JSON 相当常见。强约束被网关拒绝时，代码会自动关掉它再试一次 ——
**强约束是优化项，不是前提**。

### 校验失败 → 回灌重试（最多 2 轮）

模型给了不合规输出时，把 **Pydantic 的错误原样喂回去**：

```
你上一次的输出不符合要求，问题如下：
- 字段 合同编号：Input should be a valid string（你给的是 {'nested': 1}）

请只重新输出一个合法的 JSON 对象：键必须与我列出的字段完全一致……
```

比"重试同一个 prompt 碰运气"有效得多，因为错误信息直接告诉了它**哪里不对、错在哪个字段、收到了什么值**。

**为什么是 2 轮**：每轮都是一次真实计费调用 + 秒级延迟。实测坏输出大多第一轮就修好，
第三轮之后收益急剧下降，而用户已经等了 6 秒以上 —— 那时"给个规则版结果 + 标降级"更有价值。

### 一个关键的本地修复：标量归一

**"值的类型不合契约" 和 "值的内容不对"是两回事。**

| 情况 | 处理 | 为什么 |
| --- | --- | --- |
| 模型给 `128000`（数字）而不是 `"128000"` | **本地转字符串** | 常见、确定性可修、零成本 |
| 模型给 `{"nested": 1}`（嵌套对象） | 回灌让模型改 | 真的结构理解错误 |
| 模型给了正文里没有的值 | 丢弃为 `null` | 编造，不许放过 |

把能自己修的推给模型去修，是白花钱 —— 一次回灌要 160 额外输入 token（约 1.2 倍成本）。

### 降级：`degraded: true`

两轮仍失败 / 网络故障 → 规则版兜底并标记（`docs/api-spec.md §2.5`）：

```json
{
  "request_id": "…",
  "data": {"合同编号": "HT-2026-001", "甲方": null},
  "confidence": 0.375,
  "elapsed_ms": 102,
  "degraded": true
}
```

**只在 `true` 时出现这个键** —— 正常结果仍是 4 字段，老客户端零影响。
同时 `X-Extract-Model` 带 `+fallback(原因)` 后缀：头里是**原因**，body 里是**布尔事实**。

实测验证（网关凭据过期时）：

```
body  : {..., "degraded": true}
header: x-extract-model: buddy/glm-5.3+fallback(RetryExhaustedError)
```

换用可用模型后：4 字段、无 `degraded`、header 干净 `trae/qwen-3.5`。

---

## 契约冻住了吗？—— 这是最硬的证据

项目里最重要的一条测试在 `tests/test_api.py`：

```python
def test_service_is_swappable_without_touching_the_route(...):
    """换掉抽取实现，路由与响应形状完全不变。"""
```

W2 做完之后：

| 断言 | 状态 |
| --- | --- |
| `tests/test_api.py` 一行未改 | ✅ |
| 全套 567 个测试仍全绿（**断网也是**） | ✅ 567 passed |
| 响应体正常时仍是 4 字段 | ✅ 实测无 `degraded` 键 |
| 降级时可多 1 个可选字段 | ✅ `degraded: true`，v1.1 向后兼容 |
| 规则版仍可用 | ✅ `EXTRACTOR_IMPL=rule` 一键回退 |

---

## 流式文档摘要（本周主交付）

上传 PDF / DOCX / Markdown → 服务端 map-reduce 分段摘要 → SSE 逐字返回。
前端在 `static/index.html`（打开 `/ui`）。

![流式摘要演示](docs/demo/streaming_summary.gif)

> 15 秒演示（600px / 36 帧 / 646 KB）：选择 50 页 PDF → 进度条推进 → 摘要逐字出现
> → **点中断后立即停止**。截图见 [`docs/demo/ui_screenshot.png`](docs/demo/ui_screenshot.png)。

```powershell
uv run uvicorn app.main:app --port 8000
# 浏览器打开 http://127.0.0.1:8000/ui

# 或走 CLI
uv run python -m app.cli.summarize tests/fixtures/sample50.pdf --stream

# 重录演示（不需要 API key，也不花钱）
$env:DEMO_REPLAY="1"; uv run uvicorn app.main:app --port 8020
uv run python -X utf8 scripts/record_demo.py --port 8020
```

### 演示为什么是脚本录的，而不是人工录屏

`scripts/record_demo.py` 用 **Playwright 驱动系统 Chrome 逐帧截图 + Pillow 合成 GIF**，
不是 ScreenToGif 那类 GUI 录屏工具。三个理由：

1. **可复现**：别人想重录一遍，跑同一条命令即可，不用对着屏幕重新框选区域
2. **无需人工**：不需要有人守着点录制/停止
3. **画面稳定**：固定 viewport，没有通知弹窗、鼠标乱入、窗口遮挡

配合 `DEMO_REPLAY=1`（`app/rag/replay.py`），服务端走预置节奏而**前端代码路径完全真实**
（仍是 `fetch` + `ReadableStream` 解帧）。所以演示确定、快、不花 API 费用。

> 录 demo 时踩到一个坑：Playwright 的**同步 API 不能跨线程**。
> 第一版用后台线程持续截图，一帧都没抓到（异常被 try/except 吞了）。
> 改成 asyncio + 异步 API 才解决 —— 截图任务和页面动作在同一个事件循环里。

### 做了什么

| 层 | 内容 |
| --- | --- |
| `rag/loaders/` | pypdf 逐页抽（页码从 1 开始）· python-docx 抽段落**和表格** · Markdown 按标题切块 |
| 统一模型 | `{source, page, text}`，非分页格式 `page=None`（诚实，不假装有页） |
| `rag/summarize.py` | map-reduce：分段摘要 → 汇总；段数 > 10 走层级汇总 |
| `rag/replay.py` | 演示回放：同构事件序列，供录屏与无 key 联调 |
| `api/v1/summarize.py` | multipart 上传 + SSE（进度与正文同流） |
| `static/index.html` | fetch + ReadableStream 手工解帧（**EventSource 不能 POST**） |
| `CORS` | 放开 `*` 但不开 credentials（规范不允许 `*` + credentials 组合） |

### 首 token 慢，是模型慢还是链路慢？—— 打点回答

用 `scripts/latency_breakdown.py` 在每一层打点（5 次取 p50）：

| 层 | TTFT p50 | 相对上一层的开销 |
| --- | ---: | --- |
| ① 上游直连（裸 httpx，绕过一切我方代码） | **1301 ms** | — 这是**模型+网络的真实下限** |
| ② 经 SDK 封装（+重试层+StreamStats） | 1341 ms | **+40 ms** |
| ③ 端到端（FastAPI + HTTP + SSE 编码） | 1463 ms | **+122 ms** |

**结论：单次请求慢在模型，我方链路只占 162 ms（约 12%），可忽略。**
这个数字就是优化的靶子 —— 想压到 1.5s 以内，换更快的模型或更短的 prompt 才有意义，
优化 FastAPI 是白费力气。

**但长文档的"首字慢"是另一回事 —— 那是我方链路的问题：**

50 页 PDF 按 800 token 切 → 5–9 段 → **串行** map 调用。
用户要等 10899 ms 才看到第一个字，而单次调用只有约 2-3 秒。
**慢的不是模型，是我们串行调了 9 次。**

优化方向（按收益排序）：

1. **map 并发**（`summarize_streaming` 已实现，信号量限 4 路）—— 最大收益
2. **降低段数**：调大 `chunk_tokens`（段少 = 调用少 = 首字快，但单段更贵）
3. **把进度写进流**（已做）：前端能看到"正在总结第 3/9 段"，体感完全不同
4. 换更快的模型 —— 收益最低，因为单次本来就不慢

> 这个打点习惯是后续优化延迟的核心武器：**先分层测量，再动手**。
> 不分层的优化全凭猜测，而猜测通常是错的。

### 长文档压测（50 页 PDF）

| 指标 | 值 |
| --- | --- |
| 解析 + 切分 | 50 段 → 5 块（chunk_tokens=800） |
| 进度首帧 | +3592 ms（用户 3.6 秒就看到"正在总结"） |
| TTFT（首个摘要字） | **10899 ms** |
| 总耗时 | 13713 ms |
| 模型调用 | 6 次（5 map + 1 reduce） |

摘要质量：准确抓出"违约金自第 1 期 1‰ 逐期递增至第 50 期 50‰"这条规律，
并指出了"50 项条款结构完全一致"。

---

## 踩过的坑（都是实测，不是传闻）

| 坑 | 现象 | 结论 |
| --- | --- | --- |
| 某些模型不支持 tool 回灌 | 第二轮 502 `UNSUPPORTED_CONTENT` | 能返回 `tool_calls` **≠** 支持 Function Calling；能力是**双向**的 |
| `exc.body` 已被 SDK 剥壳 | `body["error"]["code"]` 恒为 `None` | 直接读 `body["code"]`，并兜底两种形态 |
| 推理模式吃光 `max_tokens` | `content=None`、`usage` 却正常 | 接入层必须把 `None` 当失败 |
| `request.is_disconnected()` 失效 | 生成器里恒为 `False`，客户端断了还在跑 | `StreamingResponse` 占用了 `receive` 通道；改监听 ASGI `http.disconnect` |
| **项目从未配置 logging** | 所有 `logger.info` 被静默丢弃 | root logger 未配置时 WARNING 以下蒸发；表现为"检测没生效"，实际是"日志没输出" |
| 测试数据每页内容相同 | 无法区分"正常去页眉"和"丢了正文" | 正文必须每页不同；否则测试通过与否都不说明问题 |
| `chunk_tokens` 有隐式下限 | 小预算被静默抬到 200 字/段 | "参数不生效"比报错更难查，下限要可推导 |
| Windows 控制台 GBK | 中文输出乱码/崩 | 脚本内 `reconfigure(encoding="utf-8")` + `-X utf8` |
| msys `sh` 读不懂 GBK 路径 | pre-commit hook 找不到 python | 重装 hook 时置 `PYTHONUTF8=1` |

> 最该记住的一条：**接入任何新模型，第一件事是跑一遍两轮闭环**，
> 而不是验"能不能返回 tool_calls"。后者的通过率远高于前者。

---

## Docker

三个阶段构建（builder → slim-base → scratch 拉入瘦身 rootfs）、非 root、
镜像内没有 uv / tests / `.git`。

```powershell
docker compose up -d --build
docker compose logs -f api
```

### 镜像大小：待重测（不要把旧数字当结论）

| 时间点 | 大小 | 状态 |
| --- | --- | --- |
| v0.1.0（纯 FastAPI，无 LLM/文档依赖） | **157 MB** | ✅ 当时实测 |
| v0.2.0（接入 LLM + 文档解析） | **待实测** | ⚠️ **当前无法构建，见下** |

**为什么不能沿用 157 MB**：v0.2.0 的运行时依赖新增了
`openai`(约 10 MB) + `lxml`(约 8.5 MB，python-docx 带) + `pypdf`(约 3.8 MB)
+ `tiktoken`(约 2.4 MB) + `python-docx`(约 2.3 MB) + `regex`(约 1.4 MB)
+ `markdown`(约 0.3 MB)，净增约 **28–30 MB**。
所以真实值大概在 **180 MB 上下** —— 但**这是估算，不是实测**，不能写进表格当结论。

> **这个数字要么实测、要么标明待测。** 实测命令：
> ```powershell
> docker build -t industrial-ai:sizecheck .
> docker image ls industrial-ai:sizecheck --format '{{.Size}}'
> ```
> CI 的 docker job 也会打出这个数（只记录不设阈值 ——
> 镜像大小随基础镜像更新波动，硬设阈值会变成"定期假失败"）。

**当前环境为何构建不了**：Docker Desktop 的代理设为 `127.0.0.1:7897`（manual 模式），
而该代理未运行；容器内无外网，`uv sync` 拉不到 `requests`/`pydantic-core` 等 wheel，
构建在 builder 阶段就失败。**这不是 Dockerfile 的问题** —— 代理起来后即可完成。

容器内的坑：

- **容器里的 `127.0.0.1` ≠ 宿主机**，所以 uvicorn 必须 `--host 0.0.0.0`；
  健康检查故意打容器内 `127.0.0.1` —— 它要判断"容器里的进程活没活"。
- 把 `.env` 的 `PORT=8000` 改成 `PORT=abc`，容器会进入 `Restarting` 循环，
  日志里是 `Invalid value for '--port'`。这个错误发生在**容器里**而不是 compose 解析期，是有意设计的。
- `.env` 同时被 `.gitignore` 和 `.dockerignore` 排除，不会烤进镜像层。

---

## 开发

```powershell
uv run ruff check .         # lint
uv run ruff format .        # format
uv run pytest --cov=app     # 95 tests + coverage
uv run pre-commit install   # 装 git hooks（clone 后跑一次）
```

CI 在 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)：
Python 3.11 → `uv sync` → `ruff check` → `ruff format --check` → `pytest`。
**不需要真实 API key** —— 所有测试都用假 client 离线跑，这正是"服务层不依赖
fastapi、抽取器接受注入 client"的直接收益。

---

## 文档

- [docs/api-spec.md](docs/api-spec.md) —— **v1 契约**（现 v1.1，新增可选 `degraded`），唯一不能动的文件
- [docs/cost.md](docs/cost.md) —— 成本模型：规则 vs LLM、回灌重试的真实代价、优化敏感性
- [docs/sse.md](docs/sse.md) —— SSE vs WebSocket vs 长轮询、双换行为什么是硬要求、
  裸 SSE 的 flush 时机、TTFT 口径、网关故障记录
- [docs/SETUP.md](docs/SETUP.md) —— 从零搭环境（Windows：uv / git / GitHub 连通性）
- [docs/prompt-patterns.md](docs/prompt-patterns.md) —— zero-shot / few-shot / CoT /
  Self-Consistency 四范式实测：哪个赢、赢多少、为什么推理模型不该再加 CoT 提示
- [docs/system-prompt.md](docs/system-prompt.md) —— System Prompt 四要素拆解 + 宽松 vs 严格
  的 A/B 对抗测试：拒绝率为什么会骗人、判定器返工记录
- [docs/prompt-templates.md](docs/prompt-templates.md) —— 四类模板库（qa/summarize/classify/
  extract）：目录规范、模板自述、版本机制、**硬编码业务词检查**、真实模型端到端验证
- [docs/structured-output.md](docs/structured-output.md) —— `structured_call`：本网关
  `response_format` 为何"接受但不强制"、Pydantic 报错回灌、`max_tokens` 被思维链打爆两次
- [docs/ab-scaffold.md](docs/ab-scaffold.md) —— 40 条输入池（采集纪律 + 数据泄漏检查）与
  A/B 脚手架（固定温度/seed、结论阈值、混淆矩阵、幻觉率）
- [docs/prompt-ab.md](docs/prompt-ab.md) —— 四类模板 A/B 报告：extract v2 因**契约修复**
  胜出 +63pp，qa/summarize 因"分步结构"在推理模型上反而更差
- [docs/why-i-use-langchain.md](docs/why-i-use-langchain.md) —— 为什么本项目的结构化输出链路
  不用 LangChain：它抽象掉的那层恰好是我必须自己控制的那层（含完整失败栈）
- [docs/container-networking.md](docs/container-networking.md) —— 容器化验收：为什么
  `docker compose` 里**必然**返回 `degraded:true`（`127.0.0.1` 的语义 + 网关鉴权两个独立原因）
- [docs/rag-architecture.md](docs/rag-architecture.md) —— **RAG 三段全景图**：索引（解析→切分→
  向量化→入库）/ 检索（query→相似度→TopK）/ 生成（拼上下文→调用→引用），
  每段标出输入输出类型，以及"为什么 page 从第一步就必须在结构里"
- [docs/document-parsing.md](docs/document-parsing.md) —— **统一文档解析**：必答题「页码为什么
  会错位」（因为你在切分之后才去猜）、三格式产出对齐表、表格为何单独成块且必须从文字流摘掉、
  扫描件的两档判据、以及"清洗不许改写正文"这条可被测试断言的红线
- [docs/embedding.md](docs/embedding.md) —— **Embedding 层**：必答题「为什么必须归一化」
  （归一化后内积 == 余弦，实测 1.7e-17；忘了会让排序被向量长度污染——余弦 0.60 的短向量
  输给 0.15 的长向量）、手写余弦 vs numpy 对照、bge 的 512 维与 512 token 是两件事、
  query 指令前缀 A/B（hard 集 hit@1 +7.1pp）、API vs 本地三列对比、本地模型四项代价
- [docs/chroma-store.md](docs/chroma-store.md) —— **ChromaDB 持久化**：必答题「`distances` 在
  cosine 下是什么」（= 1 − 相似度，**越小越像**，范围 [0,2] 不是 [0,1]）、
  PersistentClient 必须显式给 path、metadata 字段规范、两种 id 的分工（内容派生 vs 位置派生）、
  100 条入库 + 10 条已知答案验收，以及一条被测试抓出来的 `index_text` 静默覆盖 bug
- [docs/rag-query-chain.md](docs/rag-query-chain.md) —— **RAG 问答链路**：必答题「`overlap` 解决
  什么」（防关键信息被切在边界；代价是冗余 9.4% + top_k 重复）、中文切分边界**对比 LangChain
  实测**（它 4/5 块停在半个词上）、metadata 继承四字段、引用约束三层（prompt/解析/校验）、
  越界编号必须可见、以及"阈值必须挂在 embedder 上"（全局 0.35 曾把正确命中的 0.247 拒掉）
- [docs/streamlit-ui.md](docs/streamlit-ui.md) —— **Web UI 主交付**：必答题「1.5 秒里检索占多少」
  （**9ms / 1128~2526ms = 0.36%~0.80%，不到 1%** → 优化检索对首 token 毫无帮助，W8 该动模型侧）、
  五个耗时数字、`retrieved` 事件必须在 `delta` 之前、参考来源可展开原文、四类健壮性、
  以及"模拟 TTFT 不能做分母"这个我自己踩过的坑
- [docs/design-decisions.md](docs/design-decisions.md) —— **四条决策记录**（每条都写代价）：
  ① 长上下文为何替代不了 RAG（成本线性 / lost in the middle / 无法更新与溯源）
  ② 为何 RAG 而非微调 ③ 为何本地 embedding 而非 API ④ 为何先 ChromaDB 而非 Milvus
- [docs/weekly/w1.md](docs/weekly/w1.md) —— **W1 总览**：先冻契约再写实现、
  5 个 handler 焊死错误形状、镜像 242→211→157 MB 的瘦身账本、
  以及"装完 Docker Desktop ≠ 引擎起来了"等四条机制级认知
- [docs/weekly/w2.md](docs/weekly/w2.md) —— **W2 总览**：三个必答题的实测答案、
  四次自我推翻（成本口径/单样本/tiktoken/reasoning 参数）、
  「测试能离线跑绿」的完整论证
- [docs/weekly/w3.md](docs/weekly/w3.md) —— **W3 周复盘**：五次判定/统计口径错误、
  三道自测错题（FC 谁执行 / 流式错误 / 哪些文件不能动）、
  以及 W4 开工清单（摘要延迟必须换 reasoning 档位）
- [docs/weekly/w4.md](docs/weekly/w4.md) —— **W4 周复盘（RAG 周）**：六个必答题实测答案汇总、
  **内存测量错了两次差点把部署结论做反**（tasklist CSV 含逗号 + launcher PID 不是被测进程）、
  拔网测试卡 90 秒的根因、五条本周学到的、W5 的三个可验证目标
- [docs/deploy-notes.md](docs/deploy-notes.md) —— **部署踩坑**：免费层内存实测
  （模型栈 527 MB vs 1 GB 上限）、为什么选 hashing 而非 ONNX、降级必须可见、
  以及"我无法创建线上链接"的诚实说明
- `learn/llm/01_params.md` —— 采样参数实测（temperature / seed / max_tokens / stop）
- `learn/llm/02_function_calling.md` —— Function Calling 实测与网关踩坑

## License

MIT
