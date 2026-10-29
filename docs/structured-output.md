# 结构化输出：能力边界与 `structured_call`

> W2 周三。**本周最重要的代码产出** —— W4 拼 RAG prompt、W5 查询改写、W7 judge、
> W9 工具参数解析全都要调它。**周三之后这个签名不许再改。**
>
> | 产物 | 路径 |
> | --- | --- |
> | 核心模块 | [`app/core/llm/structured.py`](../app/core/llm/structured.py) |
> | 能力探针 | [`scripts/probe_response_format.py`](../scripts/probe_response_format.py) |
> | 探针数据 | [`scripts/probe_response_format.json`](../scripts/probe_response_format.json) |
> | 实验缓存 | [`scripts/experiment_cache.py`](../scripts/experiment_cache.py) |
> | 测试 | [`tests/test_structured.py`](../tests/test_structured.py)（11 条，全离线） |

---

## 结论速览

| 问题 | 答案 |
| --- | --- |
| 本网关支持 `json_schema` strict 吗？ | **参数接受，但不强制** —— 这是本次最重要的认知 |
| 支持 `json_object` 吗？ | ✅ 接受，能挡代码围栏，**挡不住结构错误** |
| 最终方案 | **`json_object` 兜底 + Pydantic 强校验 + 报错原文回灌** |
| 为什么不用 strict | 它在本网关上给不出承诺，把正确性押在它身上等于没押 |
| 单次调用可解析？ | ✅ 实测返回 `Ticket` 实例，三字段全对 |
| 能自修吗？ | ✅ 一次/两次修复都有测试覆盖 |
| 修复用尽？ | 抛 `StructuredOutputError`（继承 `ValueError`），或给 fallback 降级 |
| 测试 | **11 条全绿**，全离线；全量 296 条无回归 |

---

## 1. 摸能力边界：`json_schema` 是"接受"还是"执行"

任务里说「多数中文模型只支持 json_object，这个认知本身就是面试点」。
但**结论不能靠猜**，所以我写了探针 [`probe_response_format.py`](../scripts/probe_response_format.py)，
对同一份 schema 跑三种模式。

### 1.1 探针结果

| 模式 | 网关接受？ | 真的按 schema 出？ | 输出样本 |
| --- | --- | --- | --- |
| `json_schema` + `strict:true` | ✅ HTTP 200 | **❌ 否** | `{"category":"故障报修","priority":"紧急","summary":"..."}` |
| `json_object` | ✅ HTTP 200 | ❌ 否 | `{"category":"故障/技术问题","priority":"紧急",...,"tags":[...]}` |
| 不传 `response_format` | — | ❌ 否 | `{"category":"故障","priority":"紧急","summary":"..."}` |

给的是**完整 schema**：4 个 required 字段、3 组 enum、`additionalProperties:false`。
结果模型自创了 `category / priority / summary / tags / suggested_team`，
**四个 required 字段一个都没出**。

### 1.2 三条交叉验证，确保不是偶然

光看一次会得出错误结论（"网关不支持 schema"），所以我做了三次对照：

1. **最小案例**：schema 只要求 `{"a": "X"}` → 输出**正好**是 `{"a": "X"}`
2. **换 system prompt**：改成「只按给定 JSON Schema 输出，不要自创字段名」
   → 输出**仍然**自创 `category/impact/possible_causes`
3. **完整 schema 重跑** → 输出仍是自创字段

**这三条一起才能说明问题**：最小案例"成功"是因为 **prompt 里直接写明了键和值**，
不是因为 schema 生效。所以——

> **本网关的 `response_format` 是「形式被接受、语义不强制」。**
> 返回 200 不代表它按 schema 约束了输出。

### 1.3 一句话说清方案

> **`response_format` 在本网关上不可信，所以正确性只能押在本地 Pydantic 校验上；
> `json_object` 只用来挡代码围栏和废话，schema 用 prompt 渲染出来提高一次通过率，
> 失败就把 Pydantic 的报错原文回灌，最多修 2 轮。**

这也是为什么模块**不**依赖 `strict`：`strict` 给不出承诺，把正确性押在它身上，
等于押在一个返回 200 却什么都不管的东西上。

---

## 2. 三层防线

| 层 | 手段 | 拦的是什么 | 可靠性 |
| --- | --- | --- | --- |
| ① prompt | schema 说明**由 Pydantic 模型生成** | 让模型倾向正确 | 低（只是概率） |
| ② `response_format=json_object` | 拒绝非 JSON | 代码围栏、前后废话、半截输出 | 中 |
| ③ **Pydantic 校验** | `schema.model_validate` | 类型/枚举/范围/必填 | **高（唯一真源）** |

### 为什么 schema 说明要"生成"而不是"手写"

`schema_hint()` 从 `schema.model_fields` 渲染出字段清单、枚举值、必填性：

```python
class Ticket(BaseModel):
    问题类型: Literal["登录与账号", ..., "服务与支持"] = Field(description="核心诉求类型")
```

→ 自动渲染成：

```
- 问题类型（必须是以下之一：登录与账号 / 支付与账单 / ...，必填）：核心诉求类型
```

**手写会和模型漂移**：改了 Pydantic 模型忘了改 prompt，模型就按旧字段名输出，
然后每一轮都校验失败、烧满 3 次调用。生成则永远同步。

枚举必须渲染进去的实证：探针里模型在**看不到枚举**时自创了 `"category":"故障报修"`。
给出枚举清单后，实测单次调用直接命中 `"功能异常"`。

---

## 3. 修复重试：为什么回灌报错原文有效

「你错了，请重试」是废话 —— 模型不知道错在哪，下一轮大概率重复同一个错误。
Pydantic 的 `errors()` 给的是**结构化的**三要素：字段路径、错在哪、**收到的是什么值**。

回灌的实际长相（来自 `_build_repair_messages`）：

```
你上一次的输出不符合要求的 schema，具体问题如下：
- 字段 紧急程度：Input should be 'P0', 'P1', 'P2' or 'P3'（你给的是 "P9"）

【修复要求】
1. 只修改上面报错的字段，其余字段必须原样保留，不要重写整个 JSON。
2. 若某个字段的报错是「缺少」，请补上它并给出符合类型的值。
...
```

### 三条设计细节，每条都对应一个真实的坑

**① 必须带 `got`（模型实际给的值）**
只说「期望 P0/P1/P2/P3」不够，说「你给了 P9」它才知道错在哪。
`_errors_of()` 专门把 `err["input"]` 提出来。

**② 必须写「只修改报错的字段」**
这是任务提示里点名的那条坑。不写的话，模型看到报错会**重写整个 JSON**，
把原本正确的字段也顺手改掉 —— 重试变成"随机重来"，不收敛。
测试 `test_repair_prompt_preserves_correct_fields` 锁住了这条。

**③ 修复轮温度微升到 0.2**
第 1 轮是 `T=0`（要确定性）。修复轮如果还是 0，模型容易**卡在同一个错误上**；
抬到 0.2 能跳出死循环，但又不会高到"重新创作"。
（`REPAIR_TEMPERATURE`，见模块常量）

### 为什么是 2 轮

每轮都是真实计费 + 秒级延迟。W4 的 RAG、W7 的 judge 都在线上链路，
用户等不起 5 轮。2 轮与 `llm_extractor.py` 的既有结论一致
（那边实测坏输出大多第 1 轮就修好）。

---

## 4. 模块签名（定版，不再改）

```python
async def structured_call(
    prompt: str,
    schema: type[M],
    *,
    max_repair: int = 2,
    fallback: BaseModel | None = None,
) -> M
```

### 为什么是这个形状

| 设计 | 理由 |
| --- | --- |
| `schema: type[M]` 而不是 JSON Schema dict | 调用方拿回的是**类型安全对象**，有补全、有校验；dict 还要自己 `model_validate` |
| `fallback` 而不是 `raise_on_fail: bool` | 语义直白：「我接受降级」就传值，「我要求必成功」就不传。布尔开关要读注释才懂 |
| `max_repair` 是关键字参数 | 调用点读起来是 `max_repair=2` 而不是神秘的位置参数 `2` |
| 抛 `StructuredOutputError` **且继承 ValueError** | 既有降级路径按 `except ValueError` 写，不会漏接；又能被精确捕获 |
| `prompt` 里**不含** schema 说明 | 由本模块统一拼。否则每个调用点都会自己拼一份，早晚不一致 |

### 关于「不 new client」

任务要求「内部只依赖 `get_llm_client()`，不 new client」。实现时有一个细节：

`client.py` 的 `get_llm_client(settings)` 带 `Depends`，是 **FastAPI 依赖**。但本模块是
**服务层**，按本仓库既有约定（`llm_extractor.py` docstring）**不 import fastapi**。

所以这里写的是 `get_llm_client(get_settings())` —— 拿到进程级单例，
**既没 import fastapi，也没在模块级实例化**。

> ⚠️ 一个必须记住的点：`AsyncOpenAI` 的 `max_retries` 只管 HTTP 层。
> 本模块的 `max_repair` 是**语义层**重试。两层独立叠加，最坏总耗时是
> `timeout × (1+max_retries) × (1+max_repair)`。这个数字在 W4 排期时要算进去。

---

## 5. 测试：11 条，全离线

任务要求的 4 条主路径：

| # | 测试 | 断言 |
| --- | --- | --- |
| 1 | `test_success_first_try` | 返回 `Ticket` 实例，**只调 1 次** |
| 2 | `test_repair_once_then_success` | 调 2 次；修复提示含字段名、`P9`、"只修改上面报错的字段" |
| 3 | `test_repair_twice_then_success` | 调 3 次；第 2 轮报错指出缺失字段 |
| 4 | `test_exhausted_raises_identifiable_error` | `attempts==3`；异常带 `last_raw`/`last_errors`；是 `ValueError` 子类 |

另外 7 条踩坑回归：

| 测试 | 锁住什么 |
| --- | --- |
| `test_fenced_json_is_accepted` | ```json 围栏被容错，**不浪费修复轮** |
| `test_extra_fields_do_not_crash` | 多余字段不触发失败（默认 `extra=ignore`） |
| `test_repair_prompt_preserves_correct_fields` | 修复提示必须写"原样保留" |
| `test_fallback_returns_instead_of_raising` | 降级不抛异常 |
| `test_content_none_raises_not_silently_empty` | `content=None` 直接抛，**不烧后续修复轮** |
| `test_schema_hint_lists_enums` | 枚举值渲染进 prompt |
| `test_parse_json_handles_fence_and_noise` | 解析器四级容错 |

### 两个测试写法上的坑

**① 异步测试用 `@pytest.mark.anyio`，不是 `pytest-asyncio`**
本仓库不装 `pytest-asyncio`（拉包要过 PyPI，实测不通），`anyio` 是 starlette 的既有依赖。
见 `tests/conftest.py` 的 `anyio_backend` fixture。

**② patch 必须打在 `structured.get_llm_client` 上**
`from app.core.llm.client import get_llm_client` 执行后，
`structured.get_llm_client` 是**模块级属性**。patch `app.core.llm.client.get_llm_client`
对已 import 的引用无效 —— 这是这类测试最常见的假绿。

### `content=None` 的处理为什么是"直接抛"

推理模型吃光 `max_tokens` 时 `content` 是 `None`。这**不能** `or ""` 静默成空串
（空串会被下游当成"模型说什么都没有"，是错误信息）。

也**不该**走修复轮：截断是**确定性的** —— 同样的 `max_tokens` 重试仍然截断，
重试只是多花一次钱。让它直接抛，上层去调 `max_tokens` 或降级。

测试里专门断言了 `fake.call_count == 1`：第一次就截断，就不该再烧 2 轮。

**后来补的实测（2026-10-29，四类模板端到端）**：这个假设在真实模型上被验证了 ——
`max_tokens=2048` 跑"摘要"任务时确实截断（`finish_reason=length`、
`completion_tokens` 打满 2048、`reasoning_content` 4621 字符、`content=None`），
而 `max_tokens=4096` 正常返回（`completion_tokens=1685`）。

**所以 `DEFAULT_MAX_TOKENS` 从 2048 改成了 4096。** 教训是：
本网关默认 `reasoning=high`，**思维链先吃预算**，短 JSON 输出也不能按 JSON 长度估算
`max_tokens`。详见 [docs/prompt-templates.md §8.3](prompt-templates.md)。

---

## 6. 实验缓存

[`scripts/experiment_cache.py`](../scripts/experiment_cache.py) 包一层，命中时**完全不发请求**。

### 实测

| 调用 | 耗时 | 结果 |
| --- | --- | --- |
| 第 1 次（走网络） | **3.23 s** | `{"问题类型":"功能异常","紧急程度":"P1","涉及产品":"消息推送服务"}` |
| 第 2 次（命中缓存） | **0.0006 s** | 完全相同 |
| 加速比 | **5179×** | |

### 三个设计决定

- **缓存键必须含 schema**：`(model, prompt, schema_json, temperature, max_tokens)` 的 sha256。
  不算 schema 就会拿到结构不兼容的旧结果 —— 缓存最隐蔽的串味方式。
- **只缓存成功，不缓存失败**：失败的调用也花了钱，但缓存它会让**一次偶发失败**
  变成后续所有实验里的**确定性失败**，数据就变成"我第一次跑出来的运气"。
- **JSON 而非 pickle**：能人读、能 diff、不会因换 Python 版本反序列化崩。

缓存目录 `scripts/.cache/` 已加入 `.gitignore`（可重跑，不入库）。

---

## 7. 复跑

```powershell
cd D:\git\网页\industrial-ai-platform
$env:LLM_API_KEY  = "<key>"
$env:LLM_BASE_URL = "http://127.0.0.1:8326/v1"
$env:LLM_MODEL    = "deepseek-account/deepseek-flash"

# ① 能力边界（3 种 response_format）
.\.venv\Scripts\python.exe -X utf8 scripts\probe_response_format.py

# ② 测试（全离线，不需要 key）
.\.venv\Scripts\python.exe -X utf8 -m pytest tests\test_structured.py -q
.\.venv\Scripts\python.exe -X utf8 -m pytest -q          # 全量 296 条
```

---

## 8. 已知局限

- **能力边界只测了本网关 + 本模型**（`deepseek-account/deepseek-flash`）。
  换 OpenAI 官方或支持 strict 的供应商，`strict` 可能真的好使 ——
  那时应重跑探针，**不要**直接沿用"strict 不可信"这个结论。
- 探针用**一个 task**、**一种 schema 复杂度**。更简单的 schema 可能恰好被遵从，
  更复杂的可能更糟。结论是"本网关不承诺"，不是"一律不生效"。
- 缺一个**故意造错 → 能自修**的**真实联网**演示。当前"能自修"由离线测试覆盖
  （`FakeLLMClient` 按脚本回放）；真实模型是否会照着报错改对，只在单次成功路径上验过。
  **这是本次最该补的一项**，列入下周。
- `json_object` 能挡围栏，但**挡不住**"是合法 JSON 但结构不对" —— 那全靠 ③。
- 缓存只做本地文件级，没有 TTL、没有容量上限。实验期够用，上生产要换成带 TTL 的。
