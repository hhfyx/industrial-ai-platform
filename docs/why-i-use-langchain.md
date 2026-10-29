# 我为什么没有用 LangChain

> W3 周日。**唯一能在面试里直接讲三分钟的东西。**
>
> 结论先行：**我在本项目的结构化输出链路上不用 LangChain。
> 不是因为它是坏工具，而是因为它抽象掉的那一层，恰好是我必须自己控制的那一层。**
>
> 实测依据：[`scripts/lc_extract.py`](../scripts/lc_extract.py) 与
> [`lc_extract_result.json`](../scripts/lc_extract_result.json)，
> 对照自研 [`app/core/llm/structured.py`](../app/core/llm/structured.py)。

---

## 一句话版本

**LangChain 替我抽象了「prompt 模板 / 链式编排 / 输出解析」三层；
但在「结构化输出的可靠性」这一层，它把错误处理全藏在框架内部，
而我的网关恰好需要自己写回灌逻辑 —— 于是用它等于多一层包装、少全部可观测性。**

实测：同一批 10 条抽取任务，**LangChain 四种配置全部 0/10，自研 10/10（得分 0.7667）。**

---

## 1. 它抽象了哪三层

读文档 + 看 API 契约（`langchain 1.4.7`），三层是这样的：

### 第一层：PromptTemplate —— 模板即对象

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是严谨的信息抽取器..."),
        ("human", "字段清单：\n{fields}\n\n正文：\n{text}"),
    ]
)
```

**它解决了什么**：变量占位、多角色（system/human）组织、批量填充、模板复用与序列化。
**它替掉了我原来的什么**：我原来用 Jinja2（[`app/prompts/`](../app/prompts/)），
在这一点上两者能力相当，LangChain 并没有明显优势 —— 但它把模板和"链"绑定了。

### 第二层：Runnable —— 一切皆可组合

```python
chain = prompt | llm | parser
result = await chain.ainvoke({"fields": ..., "text": ...})
```

`Runnable` 的 docstring 写得很清楚：*"A unit of work that can be invoked, batched,
streamed, transformed and composed."* 于是 `invoke / batch / stream / ainvoke / astream`
对所有环节统一可用，`|` 就是组合。

**它解决了什么**：把"拼装流程"变成"表达式"。批量、流式、重试、回调都挂在统一接口上。
**这是它真正的价值** —— 一个接口覆盖同步/异步/批量/流式，自己实现要写四套胶水。

### 第三层：OutputParser / with_structured_output —— 解析即契约

```python
llm.with_structured_output(ExtractV2, method="json_schema", strict=True)
```

签名里有四个 `method` 选项和一个 `strict` 开关，这本身就是"结构化输出有多种实现路线"的
知识封装：

| method | 底层做法 |
| --- | --- |
| `json_schema` | `response_format={"type":"json_schema"}` |
| `function_calling` | `tools` + `tool_choice` 强制调用 |
| `json_mode` | `response_format={"type":"json_object"}` |

**它解决了什么**：把"模型输出 → Pydantic 对象"这一步标准化，并自动塞进 JSON Schema。
**它替掉了我原来的什么**：我自研的 `structured_call` 做的是同一件事，但**多了两样它没有的**：
报错原文回灌、可识别的降级异常。

---

## 2. 同任务重写：它替我做了什么，又拿走了什么

我用 LangChain 重写了**「信息抽取」这一个模板**（`app/prompts/extract/` 的对应实现），
跑同一批 10 条输入、同一模型、同一温度。

### 2.1 实测结果

| 配置 | method | strict | 成功 | 均分（全量） |
| --- | --- | --- | ---: | ---: |
| `lc_json_schema_strict` | `json_schema` | `True` | **0/10** | **0.0** |
| `lc_json_schema` | `json_schema` | 默认 | **0/10** | **0.0** |
| `lc_function_calling` | `function_calling` | 默认 | **0/10** | **0.0** |
| `lc_json_mode` | `json_mode` | 默认 | **0/10** | **0.0** |
| **自研 `structured_call`** | `json_object` + 回灌 | — | **10/10** | **0.7667** |

**四种配置全部归零。** 这不是我选错了配置 —— 四种都试过了。

### 2.2 每一路为什么失败（这是本文最值钱的部分）

**① `json_schema` / `json_schema + strict`：模型没按 schema 出，框架直接抛 ValidationError。**

```
ValidationError: 1 validation error for ExtractV2
置信度
  Field required
```

原因：**本网关接受 `response_format={"type":"json_schema","strict":true}`，但不强制执行。**
（这是我在周三 [`docs/structured-output.md`](structured-output.md) 里独立发现的，
今天用 LangChain 得到完全一致的交叉验证。）

模型输出了它自己认为合理的字段，`置信度` 根本没出现。
LangChain 拿到这个结果 → Pydantic 校验 → **抛异常 → 结束**。

**框架在这里做了什么？** 它诚实地报告了失败。
**框架在这里没做什么？** **它不会自我修复。** 没有第二次调用，没有把错误喂回去。

**② `function_calling`：网关直接 400 拒绝。**

```
OpenAIInvalidRequestError: Error code: 400 - {'error': {'message': 'on...'}}
```

这与周二 [`learn/llm/02_function_calling.md`](../../learn/llm/02_function_calling.md)
记录的 `role=tool` 回灌问题同源：**这个网关的 `deepseek-*` 适配器表达不了工具调用。**

**③ `json_mode`：模型输出对了内容，但结构不是嵌套的。**

```
OutputParserException: Failed to parse ExtractV2 from completion {"主体": "..."
```

10 条全部失败。模型给了 `{"主体": "uv", ...}`，而 schema 要的是
`{"fields": {"主体": "uv"}, ...}` —— **json_mode 只保证"是合法 JSON"，不保证"是我的形状"。**

### 2.3 我另外排除了一个混淆因素

我一度怀疑是"嵌套 dict 结构对 LangChain 太复杂"，于是用**扁平模型**再测一次：

```python
class Flat(BaseModel):
    平台: str
    正确命令: str
```

结果：

```
json_schema FAIL ValidationError
  Invalid JSON: expected value at line 1 column 1
  input_value='平台：Window...'
```

**模型返回的是纯文本 `平台：Windows`，连 JSON 都不是。**
所以问题不在嵌套结构 —— **是本网关根本不强制任何格式。** 扁平也照样失败。

> **这一步很重要**：如果我只跑了嵌套版本就下结论，
> 会得出"LangChain 处理不了复杂 schema"这个错误归因。
> 真正的结论是"**它依赖网关强制格式，而我的网关不强制**"。

---

## 3. 四维对比

| 维度 | LangChain | 自研 `structured_call` | 谁赢 |
| --- | --- | --- | --- |
| **代码量** | 约 **95 行**（含 llm 构造、prompt、链、异常处理） | 约 **300 行**（模块本体，含注释与类型） | LC 更少 |
| **可读性** | `prompt \| llm \| parser` 一眼看懂 | 需读 `judge_*` 与回灌逻辑 | LC 更好 |
| **结构化可靠性** | **0/10**（依赖网关强制格式） | **10/10**（自己校验 + 回灌） | **自研** |
| **调试难度** | 出错只拿到一行 `ValidationError`/`OutputParserException`，**看不到模型原始输出、看不到重试了几次** | 异常自带 `last_raw` / `last_errors` / `attempts` | **自研** |

### 3.1 代码量的诚实对比

LangChain 版**确实更短**。这是因为它的"短"建立在两个假设上：

1. 网关会按 schema 出（本网关不会）
2. 出错就抛给调用方（而我要自己修）

**当这两个假设不成立时，"更短"就变成了"更短且不可用"。**

反过来，如果我用 LangChain 补上回灌逻辑（`with_fallbacks` + 自定义 parser + 重试），
代码量会涨到接近自研版本 —— **那就失去了用它的一半理由。**

### 3.2 调试难度：这是决定性的

出问题时，两边能拿到的东西完全不同：

| | LangChain 抛的信息 | 自研抛的信息 |
| --- | --- | --- |
| 模型原始输出 | ❌ 有时被包在异常里，不保证 | ✅ `last_raw` 一定在 |
| 具体哪个字段错 | ✅（Pydantic 的功劳） | ✅ |
| 模型给的值是什么 | ⚠️ 部分 | ✅ `last_errors` 里带 `got` |
| 重试了几次 | ❌ 框架内部 | ✅ `attempts` |
| 能否重判历史 | ❌ | ✅ 存了 `raw`，`--rejudge` 不重花钱 |

> **这条是本周最真实的差异**：LangChain 把错误处理收进了框架内部，
> 换来简洁；代价是**我看不见它在干什么**。
> 而在"网关不守规矩"的环境里，看不见就等于没法修。

### 3.3 但我也必须承认它赢的地方

**如果我今天要做的是「多步骤 agent + 工具调用 + 流式 + 内存」**，
LangChain 的 `Runnable` 那一层是明显有优势的：

- 我自研流式要写 `stream`/`astream` 两套
- 批量要自己写并发控制
- 组合要手写 pipeline

**LangChain 在这些场景省下的代码量是实打实的。**
所以我的结论是**特定于"结构化输出"这一层**，不是"LangChain 没用"。

---

## 4. 它在哪些地方碍事（逐条实测）

### 4.1 结构化输出：**最致命**（见 §2）

它把「校验失败」当成终点，我把「校验失败」当成**下一次调用的起点**。

```python
# 它的思路
try:
    return schema.model_validate(parsed)
except ValidationError:
    raise          # 交给调用方

# 我的思路
for round in range(max_repair + 1):
    ...
    except ValidationError as exc:
        messages = _build_repair_messages(messages, raw, errors, schema)
        continue   # 带上错误原文再问一次
```

**关键差异不是"谁对"**：是我的场景里 `max_repair` 必须可控（线上延迟与成本预算），
而框架把它埋在内部。

### 4.2 流式：它有自己的抽象，和我的冲突

我已有 [`app/core/llm/streaming.py`](../app/core/llm/streaming.py)，
它直接操作 `chat.completions.create(stream=True)` 并自己数 TTFT。
LangChain 的 `astream` 返回它自己的 chunk 对象 ——
**要接进来就得做一层转换，而我的 TTFT 口径是按"第一个非空 content"定的**（
[`docs/sse.md`](sse.md)）。框架的 chunk 边界不保证与我的口径一致。

**结论：流式这一层我没用它的抽象，也不需要。**

### 4.3 自定义重排 / RAG 检索策略

W4 要做 RAG。LangChain 的 retriever 抽象会介入"召回 → 重排 → 拼 prompt"，
而**重排策略是我想自己控制的部分**（分数融合、去重、截断）。
引入它等于多一层要调试的中间层，且**它的默认重排逻辑不是我要的**。

### 4.4 依赖 footprint

装上 LangChain 后，`uv.lock` 多了一串依赖（`langsmith`、`orjson`、`ormsgpack`、
`zstandard`、`xxhash`、`tenacity`、`requests-toolbelt`...）。

**这一条不是决定性的，但值得记**：一个 300 行的自研模块 vs 一套带 7+ 传递依赖的框架。
在"我要的只是三个函数"的场景下，依赖面越小越好查问题。

---

## 5. 最终决定：不用它做结构化输出，但保留它做别的

### 5.1 决定

| 场景 | 用什么 | 理由 |
| --- | --- | --- |
| **结构化输出**（W4/W5/W7/W9 的共用底座） | **自研 `structured_call`** | 需要自己控制回灌轮数与降级异常 |
| prompt 模板 | **自研 Jinja2 库** | 已有版本机制 + 硬编码检查，LC 无对应能力 |
| 实验编排（批量跑） | 自研 `ab_test.py` | 要记录 token/成本/混淆矩阵，LC 不提供 |
| **若将来做多步骤 agent + 工具链** | **考虑 LangChain** | `Runnable` 的组合与流式确实省代码 |

**一句话：我只在"它比我自己写更省事、且不牺牲可观测性"的地方用它。
结构化输出恰好是相反的地方。**

### 5.2 为什么这不是"框架恐惧症"

我可以给出**可验证的替代条件**：

> **如果我的网关真的支持 `json_schema` strict**（实测输出严格符合 schema），
> 那么 `with_structured_output(method="json_schema", strict=True)` 就能一次拿到
> 类型安全对象，回灌逻辑变得罕见（只在"语义错误"时才需要）。
> **在那种环境下我会用 LangChain** —— 因为它的简洁性是真的，
> 而我今天不用它，是因为它依赖的那个前提在本网关上不成立。

这不是"我不喜欢框架"，是"**这个框架的可靠性和我网关的可靠性假设冲突**"。

---

## 6. 这次对比教会我什么（面试版，三分钟）

### 6.1 三层抽象

1. **PromptTemplate** —— 模板即对象（变量占位 + 多角色）
2. **Runnable** —— 一切可组合，统一 `invoke`/`batch`/`stream`
3. **OutputParser / with_structured_output** —— 解析即契约

### 6.2 它碍事的地方

- **结构化输出**：把校验失败当终点，我需要它当起点（回灌）
- **流式**：它的 chunk 边界与我的 TTFT 口径不一致
- **RAG 重排**：它的默认策略不是我要的

### 6.3 我的取舍

**用不用框架，取决于"框架的可靠性假设是否与你的运行环境一致"。**

LangChain 的 `with_structured_output` 隐含假设：*网关会按 schema 输出*。
我的网关不满足这个假设 —— 所以同一个函数的四种配置全部 0/10。

**当假设不成立时，框架的简洁性会变成"简洁地失败"。**

### 6.4 一个更普适的判断力

**判断一个框架该不该用，不要看它"能不能做"，要看它"出错时你还能不能控制"。**

- 能做：两边都能做（抽取任务谁都能写）
- 出错时：框架抛一行异常，我拿得到原始输出、错误明细、重试次数、可离线重判

---

## 7. 复跑

```powershell
cd D:\git\网页\industrial-ai-platform
$env:LLM_API_KEY  = "<key>"
$env:LLM_BASE_URL = "http://127.0.0.1:8326/v1"
$env:LLM_MODEL    = "deepseek-account/deepseek-flash"

# LangChain 四种配置（累积写入，不覆盖）
.\.venv\Scripts\python.exe -X utf8 scripts\lc_extract.py

# 对照：自研方案
.\.venv\Scripts\python.exe -X utf8 scripts\ab_test.py --set extract --a v1 --b v1.1 --fresh
```

---

## 8. 已知局限

- **只测了一个网关和一个模型**。换 OpenAI 官方（真支持 `json_schema` strict），
  结论很可能反转 —— 我在 §5.2 已写明这种情况下会怎么选。
- **只对比了"抽取"一个任务**。摘要/分类/问答的对比没做 ——
  但那三类在当前网关下会撞同一个前提（不强制格式），结论应当一致。
- **没测它的 `with_fallbacks` / `RetryOutputParser`**。
  理论上可以用这两个补出回灌，但代码量会涨到接近自研版（§3.1），
  且可观测性仍然在框架内部。**这一点我没实测，只做了成本估算。**
- **代码量统计是粗估**（按行数），不含注释外的空行换算。
- **LangChain 1.4.7**，不同版本的 `method` 默认值可能变化
  （`json_schema` 是当前默认）。
