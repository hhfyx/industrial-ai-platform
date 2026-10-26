# industrial-ai-platform

**一个用 FastAPI + LLM 做结构化信息抽取的后端平台。**

重点不在"调通一个接口"，而在**接口不变、实现可换**：同一份 `v1` HTTP 契约，
底下从正则实现换成了真实的模型调用，`tests/test_api.py` 一行未改、95 个测试仍全绿。

| 项目 | 值 |
| --- | --- |
| 端点 | `POST /api/v1/extract` |
| 契约版本 | `v1`（响应头 `X-API-Contract: v1`，见 [docs/api-spec.md](docs/api-spec.md)） |
| 测试 | **95 passed**，`pytest --cov=app` |
| 代码量 | 31 个 py 文件 / 约 3,100 行（`app/` + `scripts/` + `tests/`） |
| 镜像 | 多阶段构建，**157 MB** |
| 运行时 | Python 3.11 + uvicorn，非 root（uid 10001） |

---

## 这个项目解决什么

把一段非结构化文本（合同、工单、简历）按调用方指定的字段抽成结构化数据：

```http
POST /api/v1/extract
{"text": "合同编号：HT-2026-001，甲方：张三，签订日期：2026-10-20", "fields": ["合同编号", "甲方"]}
```

```json
{"request_id": "…", "data": {"合同编号": "HT-2026-001", "甲方": "张三"}, "confidence": 1.0, "elapsed_ms": 101}
```

三个设计取向值得先说清楚：

1. **抽不到就是 `null`，永不编造。** 不猜、不补全。这条不只写在 prompt 里，
   `LlmExtractor._ground_value` 会做**原文回指校验**：模型给出的值必须在正文里能找到依据，
   否则丢弃为 `null`。prompt 拦不住幻觉，代码能。
2. **所有非 2xx 是同一个 4 字段形状**（`code` / `message` / `request_id` / `details`）。
   调用方写一次错误处理就能覆盖 400/422/500。
3. **实现可换，契约冻死。** `docs/api-spec.md` 是唯一不能动的东西，
   规则版 → LLM 版是一次真实演进，不是重写。

---

## 架构

```
                    ┌──────────────────────────────────────────┐
   HTTP 请求  ──────▶│  app/api/v1/extract.py      路由（只做三件事）│
                    │    解析入参 / 调服务 / 包统一响应             │
                    └───────────────┬──────────────────────────┘
                                    │  ExtractorDep（依赖注入，这里换实现）
                    ┌───────────────▼──────────────────────────┐
                    │  app/core/deps.py           装配开关       │
                    │    EXTRACTOR_IMPL=llm|rule  一键回退       │
                    └───────────────┬──────────────────────────┘
                    ┌───────────────▼──────────────────────────┐
                    │  app/services/              服务层（无 fastapi）│
                    │    llm_extractor.py  LlmExtractor  ◀ 默认 │
                    │    extractor.py      RuleExtractor  ◀ 兜底│
                    └───────────────┬──────────────────────────┘
                    ┌───────────────▼──────────────────────────┐
                    │  app/core/llm/            接入层           │
                    │    client.py    get_llm_client() 进程级单例│
                    │    tools.py     Function Calling 工具注册表│
                    │    tool_loop.py 单/多轮闭环                │
                    └──────────────────────────────────────────┘
```

跨层规则：**服务层不 import fastapi**。依赖注入是路由层的事，
`LlmExtractor` 只接受一个 `AsyncOpenAI` 实例，所以测试里能直接塞假 client。

---

## 快速开始

```powershell
uv sync                                   # 创建 .venv 并装依赖
Copy-Item .env.example .env               # 然后填 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
uv run uvicorn app.main:app --reload      # http://127.0.0.1:8000/docs
```

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
| 全套 95 个测试仍全绿 | ✅ 95 passed |
| 响应体仍是 4 字段 | ✅ 实测 `X-Extract-Model: buddy/glm-5.3` |
| 规则版仍可用 | ✅ `EXTRACTOR_IMPL=rule` 一键回退 |

回退开关为什么用环境变量而不是代码开关：**应急时它必须能被最快改掉**。
改代码要过评审、构建、发版，那段时间线上一直在 500。

---

## 踩过的坑（都是实测，不是传闻）

| 坑 | 现象 | 结论 |
| --- | --- | --- |
| 某些模型不支持 tool 回灌 | 第二轮 502 `UNSUPPORTED_CONTENT` | 能返回 `tool_calls` **≠** 支持 Function Calling；能力是**双向**的 |
| `exc.body` 已被 SDK 剥壳 | `body["error"]["code"]` 恒为 `None` | 直接读 `body["code"]`，并兜底两种形态 |
| 推理模式吃光 `max_tokens` | `content=None`、`usage` 却正常 | 接入层必须把 `None` 当失败 |
| Windows 控制台 GBK | 中文输出乱码/崩 | 脚本内 `reconfigure(encoding="utf-8")` + `-X utf8` |
| msys `sh` 读不懂 GBK 路径 | pre-commit hook 找不到 python | 重装 hook 时置 `PYTHONUTF8=1` |

> 最该记住的一条：**接入任何新模型，第一件事是跑一遍两轮闭环**，
> 而不是验"能不能返回 tool_calls"。后者的通过率远高于前者。

---

## Docker

三个阶段构建（builder → slim-base → scratch 拉入瘦身 rootfs），最终 **157 MB**、
非 root、镜像内没有 uv / tests / `.git`。

```powershell
docker compose up -d --build
docker compose logs -f api
```

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

- [docs/api-spec.md](docs/api-spec.md) —— **v1 契约**，唯一不能动的文件
- [docs/SETUP.md](docs/SETUP.md) —— 从零搭环境（Windows：uv / git / GitHub 连通性）
- `learn/llm/01_params.md` —— 采样参数实测（temperature / seed / max_tokens / stop）
- `learn/llm/02_function_calling.md` —— Function Calling 实测与网关踩坑

## License

MIT
