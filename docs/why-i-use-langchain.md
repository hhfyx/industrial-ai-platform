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

实测（同一批 10 条抽取任务）：

| 方案 | 嵌套 schema | 扁平 schema |
| --- | ---: | ---: |
| LangChain `with_structured_output` | **0/10** | **8/10** |
| 自研 `structured_call` | **10/10**（0.7667 分） | 同左（回灌补齐） |

**我要的正是嵌套结构 —— 它恰好在这一档失效。** 这个区别很重要，
它决定了我的理由是"环境判断"而不是"框架恐惧症"（§2.3 有完整排除过程）。

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

> ⚠️ **但这张表不是全部事实。** 我后来补跑了一个关键对照（见 §2.3）：
> **同样的 `json_schema` 换成扁平 schema，10 条里成功了 8 条。**
> 所以准确的说法不是"LangChain 不行"，而是
> **"在这套网关 + 嵌套 schema 的组合下它不行"** —— 这个区别很重要，
> 它决定了我到底是"框架恐惧症"还是"环境判断"。§2.3 完整交代。

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

完整错误原文（这条最有说服力，因为它**不是配置问题**）：

```
OpenAIInvalidRequestError: Error code: 400 - {'error': {'message':
'only tool_choice auto and none are supported',
'type': 'invalid_request_error', 'code': 'invalid_request'}}
```

**关键在 `only tool_choice auto and none are supported`** —— 网关明确表示它
**只支持 `tool_choice=auto` 和 `none`**。而 `with_structured_output(method="function_calling")`
必须用 `tool_choice` 指向具体函数才能强制结构化输出。

**所以这条路在本网关上是死路，跟我怎么写 prompt 无关。**
这与周二 [`learn/llm/02_function_calling.md`](../../learn/llm/02_function_calling.md)
记录的 `role=tool` 回灌问题同源。

**③ `json_mode`：模型输出了正确的**内容**，但结构不是 schema 要的。**

```
OutputParserException: Failed to parse ExtractV2 from completion
  {"平台": "Windows", "正确命令": "curl.exe"}. Got: 1 validation error for ExtractV2
  置信度
    Field required [type=missing,
    input_value={'平台': 'Windows', '正确命令': 'curl.exe'}, input_type=dict]
```

**注意这个错误长什么样**：模型把**两个业务字段都答对了**
（`平台=Windows`、`正确命令=curl.exe`），它只是没套上 `{"fields": {...}}` 这层壳。

`json_mode` 只保证"是合法 JSON"，**不保证"是我的形状"**。10 条全部因此失败。

### 2.3 排除混淆因素：扁平 schema 能成功 8/10（**这条推翻了我最初的归因**）

我最初怀疑"是嵌套 dict 让 LangChain 处理不了"，于是用**一条**输入测扁平模型：

```python
class Flat(BaseModel):
    平台: str
    正确命令: str
```

那次**成功了**。但一条样本不能下结论，所以我把它做成了完整的 10 条对照
（`scripts/lc_flat_vs_nested.py`，为每条样本动态造一个扁平 schema，
保证唯一差别就是"结构层次"）：

| schema 形状 | method | 成功 |
| --- | --- | ---: |
| **嵌套** `{fields: dict, 缺失字段, 置信度}` | `json_schema` + strict | **0/10** |
| **扁平** `{字段1: str|null, 字段2: str|null, ...}` | `json_schema` + strict | **8/10** |

**这个对照推翻了我原来的话。** 我原先写的是"扁平也照样失败，
本网关根本不强制任何格式" —— 那是**基于 1 条样本的错误归因**。

修正后的真实结论：

> **本网关对扁平 schema 能强制执行，对嵌套 schema 不能。**

扁平那 2 条失败也不是格式没强制，而是**形状选错**：

```
ValidationError: Input should be an object [type=model_type,
  input_value=[{'主体': 'Python', ..., '路径': '.local/bin'}], input_type=list]
```

模型输出了**数组** `[{...}]` 而不是对象 —— 因为 ex-01、ex-09 这两条的原文里
**有多个实体**（多个软件版本、两种优化手段），模型认为该返回列表。
这是**语义判断问题**，不是"schema 没生效"。

### 2.4 修正后的完整归因（这才是能拿去面试的版本）

我一开始的结论是"LangChain 四种配置全 0，所以不能用"。**这个结论太粗。**

准确的归因是三层：

| 层 | 事实 | 责任在谁 |
| --- | --- | --- |
| ① `function_calling` | 网关只支持 `tool_choice=auto/none` | **网关能力缺失**，与框架无关 |
| ② 嵌套 schema + `json_schema` strict | 0/10，模型不套外层壳 | **嵌套结构不被强制** |
| ③ 扁平 schema + `json_schema` strict | 8/10 | schema 强制**基本有效** |

**所以真正该说的是**：

> **不是"LangChain 不行"，而是"我要的输出结构（嵌套）不在它的有效区间内"。**
> 扁平结构下它工作得很好 —— 如果我的抽取结果可以做成扁平（每个业务字段一个顶层键），
> `with_structured_output(method="json_schema", strict=True)` 是可用方案，
> 8/10 的成功率已经够用（加上回灌就能补到 10/10）。

**那我为什么最终还是不用它？** 因为我要的是**嵌套结构**（字段数不定，
无法预先定义成扁平 Pydantic 类）—— 而嵌套恰好是它在本网关上失效的那一档。

### 2.5 一个额外的教训：1 条样本足以让我写错一整个小节

原文 §2.3 那句"扁平也照样失败、本网关根本不强制任何格式"，
是基于**1 条**输入得出的。补跑到 10 条才发现方向全错。

> **样本量不够时，错误不会以"不确定"的形式出现，而会以"确定的错误结论"出现。**
> 1 条成功可以说"这条能过"，**不能**说"这一类都能过"。
> 这个坑和我周六在成本口径上踩的是同一类（见 [weekly/w3.md §3.1](weekly/w3.md)）。

---

## 3. 四维对比

| 维度 | LangChain | 自研 `structured_call` | 谁赢 |
| --- | --- | --- | --- |
| **代码量** | 约 **95 行**（含 llm 构造、prompt、链、异常处理） | 约 **300 行**（模块本体，含注释与类型） | LC 更少 |
| **可读性** | `prompt \| llm \| parser` 一眼看懂 | 需读 `judge_*` 与回灌逻辑 | LC 更好 |
| **结构化可靠性** | 嵌套 **0/10**（不强制）/ 扁平 **8/10** | **10/10**（自己校验 + 回灌） | **自研** |
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

> **注意这个差异与 schema 形状无关。** 即使换成扁平 schema（8/10 成功那档），
> 失败的 2 条**仍然是直接抛异常、不重试**。所以"它不自我修复"这一条
> 是我不用它的**独立理由**，不受 §2.3 那个修正的影响。

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

### 5.2 为什么这不是"框架恐惧症"（现在有实测支撑了）

我可以给出**可验证的替代条件**，而且这个条件有一半已经在今天被验证了：

> **① 如果我要的输出能做成扁平 schema**（每个业务字段一个顶层键），
> 那么 `with_structured_output(method="json_schema", strict=True)` **今天就能用** ——
> 实测 8/10 成功，失败的两条也是"模型返回数组"这种**语义**问题，
> 用一次回灌就能修好。
>
> **② 如果我的网关支持 `tool_choice` 指向具体函数**，
> 那 `method="function_calling"` 也能用。但本网关明确回了
> `only tool_choice auto and none are supported`，所以这条路**不是配置问题**。
>
> **③ 我真正需要的是嵌套结构**（字段数运行时才知道，无法预先定义成扁平类），
> 而嵌套恰好是它在本网关上失效的那一档。

**所以我的不用，不是"我不喜欢框架"，而是三件具体的事叠在一起：**

1. 我需要嵌套结构，而嵌套不被强制（0/10）
2. `function_calling` 这条路被网关的能力限制堵死（`tool_choice` 只支持 auto/none）
3. 即使前两条都解决，它**仍然不会自我修复**（这条与 schema 形状无关）

**如果是扁平任务 + 网关支持函数调用，我会用它。** 这不是妥协，是它的有效区间。

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

### 6.3 我最想说的一件事：我怎么知道"是它的错"而不是"我的错"

面试里被问"为什么不用 LangChain"，最容易被追问的是
**"是不是你不会用？"**

我的答案是**三层排除法**，每一层都有实测数据：

| 排除什么 | 怎么排 | 结果 |
| --- | --- | --- |
| ① 是不是我选错 method？ | **四种 method 全试** | 全部 0/10（扁平除外） |
| ② 是不是嵌套太复杂？ | **换成扁平 schema 跑 10 条** | **8/10 成功** → 证明问题在嵌套，不在我 |
| ③ 是不是网关根本不支持？ | **读 400 的完整错误原文** | `only tool_choice auto and none are supported` → 能力缺失，非配置 |

**关键在于第 ② 步**：它把我从"我的配置错了"摘出来，同时**也限制了我能宣称的范围** ——
我不能说"LangChain 在这个网关上没用"，只能说"它在我需要的嵌套结构上没用"。

> **能说清"我的结论边界在哪"，比结论本身更有说服力。**
> 大多数候选人只会说"我用 / 不用 LangChain"，说不清是在什么条件下成立。

### 6.3 我的取舍

**用不用框架，取决于"框架的可靠性假设是否与你的运行环境一致"。**

LangChain 的 `with_structured_output` 隐含假设：*网关会按 schema 输出*。

**我的网关是"部分满足"这个假设的** —— 扁平结构满足（8/10），嵌套结构不满足（0/10）。
而我要的恰好是嵌套。

> 这个"部分满足"才是最真实的情况，也是我花了两次才说对的：
> 第一次我说"完全不满足"（基于 1 条样本），第二次补跑 10 条才发现是"部分满足"。
>
> **面试里如果只能说一句**：不是"框架不行"或"网关不行"，
> 而是**"两者的有效区间不重叠"** —— 这句话需要数据支撑，而不是态度。

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

# 四种配置 × 10 条（累积写入，不覆盖）
.\.venv\Scripts\python.exe -X utf8 scripts\lc_extract.py

# 完整失败原文（每种配置 1 条，省调用量）
.\.venv\Scripts\python.exe -X utf8 scripts\lc_failure_stacks.py

# 关键对照：扁平 vs 嵌套 × 10 条
.\.venv\Scripts\python.exe -X utf8 scripts\lc_flat_vs_nested.py

# 对照组：自研方案
.\.venv\Scripts\python.exe -X utf8 scripts\ab_test.py --set extract --a v1 --b v1.1 --fresh
```

---

## 8. 已知局限

- **`reasoning=high` 全程开着**。本网关默认如此，而它对结构化输出的影响没测
  （理论上思维链变长会挤占输出预算，与 §摘要那类延迟问题同源）。

- **只测了一个网关和一个模型**。换 OpenAI 官方（真支持 `json_schema` strict），
  结论很可能反转 —— 我在 §5.2 已写明这种情况下会怎么选。

- **只对比了"抽取"一个任务**。摘要/分类/问答没做对照 ——
  但它们撞的是同一个前提（嵌套 schema 不被强制），§2.3 的结论应当可迁移。
  **"应当"不等于"已验证"**，这是本文最大的外推。

- **扁平 8/10 那两条失败的归因是推断**。我看到模型输出数组 `[{...}]`，
  判断是"原文有多个实体、模型认为该返回列表"。**没有再单独设计实验验证这个推断。**

- **没测它的 `with_fallbacks` / `RetryOutputParser`**。
  理论上能用这两个补出回灌，但代码量会涨到接近自研版（§3.1），
  且可观测性仍在框架内部。**这一点我没实测，只做了成本估算。**

- **代码量统计是粗估**（按行数），不含注释外的空行换算。

- **LangChain 1.4.7**，不同版本的 `method` 默认值可能变化
  （`json_schema` 是当前默认）。

- **§2.3 是本文修过一次的地方**。初版基于 1 条样本写了"扁平也失败"，
  补跑到 10 条才发现是 8/10 成功。如果你看到任何引用旧结论的副本，以本文为准。

---

## 附录 · 完整失败栈（可复核）

数据来源：`scripts/lc_failure_stacks.json`（每种配置 1 条输入）、
`scripts/lc_extract_result.json`（10 条 × 4 臂）。
复跑：`python -X utf8 scripts/lc_failure_stacks.py`。

### 附-1 四种配置的原始错误

| 配置 | 错误类型 | 完整原文 |
| --- | --- | --- |
| `json_schema` + `strict=True` | `ValidationError` | `置信度 / Field required [type=missing, input_value={'平台': 'Windows', '正确命令': 'curl.exe'}, input_type=dict]` |
| `json_schema`（默认） | `ValidationError` | 同上 |
| `function_calling` | `OpenAIInvalidRequestError` | `Error code: 400 - {'error': {'message': 'only tool_choice auto and none are supported', 'type': 'invalid_request_error', 'code': 'invalid_request'}}` |
| `json_mode` | `OutputParserException` | `Failed to parse ExtractV2 from completion {"平台": "Windows", "正确命令": "curl.exe"}. Got: 1 validation error for ExtractV2 / 置信度 / Field required` |

### 附-2 四条最值得记的细节

**① `function_calling` 的 400 不是配置错误**

原文明说 `only tool_choice auto and none are supported` ——
网关**只支持 `tool_choice=auto/none`**，而强制结构化输出必须指定具体函数。
**这条路在本网关上是死路，跟 prompt 怎么写无关。**

**② `json_schema` 的失败长这样（不是"输出坏了"）**

模型给的是 `{"平台": "Windows", "正确命令": "curl.exe"}` ——
**两个字段的值完全正确**，只是没套上 `{"fields": {...}}` 那层壳。
所以这不是"模型不会抽"，是"模型不照嵌套 schema 排版"。

**③ `json_mode` 的失败和 ② 是同一件事**

它拿到的是**内容正确、结构不对**的输出，然后 OutputParserException。
两者差别只在报错类型，根因相同。

**④ 扁平 schema 的 2 条失败是"形状选错"，不是"没强制"**

```
ValidationError: Input should be an object [type=model_type,
  input_value=[{'主体': 'Python', ..., '路径': '.local/bin'}], input_type=list]
```

模型返回了**数组**。它面对"原文有多个实体"（ex-01 多个软件版本、
ex-09 两种优化手段）时，认为该返回列表而不是对象。
**这是语义判断问题，不是 schema 没生效** —— 反过来说明扁平结构的
schema 强制是起作用的（否则它不会正好卡在"该不该是数组"这个点上）。

### 附-3 各组成功率

| 臂 | schema | method | 成功 | 文件 |
| --- | --- | --- | ---: | --- |
| `lc_json_schema_strict` | 嵌套 | `json_schema` + strict | 0/10 | `lc_extract_result.json` |
| `lc_json_schema` | 嵌套 | `json_schema` | 0/10 | 同上 |
| `lc_function_calling` | 嵌套 | `function_calling` | 0/10 | 同上 |
| `lc_json_mode` | 嵌套 | `json_mode` | 0/10 | 同上 |
| **`lc_flat_json_schema`** | **扁平** | `json_schema` + strict | **8/10** | `lc_flat_vs_nested.json` |
| 自研 `structured_call` | 嵌套 | `json_object` + 回灌 | **10/10** | `ab_results/extract_v1_vs_v1.1_full_fresh` |
