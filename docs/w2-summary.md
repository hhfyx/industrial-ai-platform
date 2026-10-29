# W2 总结：从"能调通"到"能测、能换、能算账"

> 周期：2026-10-26 ～ 10-29（4 天）
> 产物：**327 个测试全离线**、12 篇实测文档、`structured_call` 定版签名、
> 4 类 prompt 模板（每类 v1/v2）+ 40 条评测输入池 + A/B 脚手架
> 关键提交：`60592d0` → `6cb498b`（10/26 起共 18 个）

---

## 一句话总结

W2 做完了三件"看起来像同一件事、实际完全不同"的工作：

1. **能调通** —— 接上 LLM，`/api/v1/extract` 真的在调模型（Day 1）
2. **能测、能换** —— 327 个测试拔掉网线全绿，证明实现可替换（Day 2–3）
3. **能算账、能选** —— 用实测数据回答"用哪种 prompt、花多少钱、值不值"（Day 3–4）

**第二件是唯一无法伪造的成果**，第三件是这次真正超出预期的地方。

---

## 一、按时间线的实际交付

| 日期 | 主题 | 关键产物 | 状态 |
| --- | --- | --- | --- |
| 10/26 | LLM 接入地基 | `client.py` / `retry.py` / `streaming.py` / `tools.py` / `tool_loop.py` | ✅ |
| 10/26 | Streaming + SSE | `/extract/stream`、`/chat/stream`、`summarize/stream` | ✅ |
| 10/26 | 文档解析 + Web UI | `rag/loaders/`、`rag/summarize.py`、`static/index.html`、demo GIF | ✅ |
| 10/27 | 换实现 + 离线测试 + 成本表 | `structured.py`、`test_wiring.py`、`docs/cost.md` | ✅ |
| 10/28 | 结构化输出（能力边界） | `docs/structured-output.md`、三层防线 | ✅ |
| 10/28 | Prompt 四范式实测 | `docs/prompt-patterns.md`（8 臂 × 20 输入） | ✅ |
| 10/28 | System Prompt A/B | `docs/system-prompt.md`（四要素拆解） | ✅ |
| 10/28–29 | 模板库 + A/B 脚手架 | `app/prompts/`、`eval/prompt-sets/`、`ab_test.py` | ✅ |
| 10/29 | 四类 A/B 报告 + LangChain 取舍 | `docs/prompt-ab.md`、`docs/why-i-use-langchain.md` | ✅ |

**测试从 0 → 327**，全部离线（`LLM_BASE_URL` 指向不可达地址 + 代理置空仍全绿）。

---

## 二、三个"必答题"及其实测答案

这周的三个必答题，答案都不是"看文档得出的"，是跑出来的。

### 2.1 CoT 让准确率变好了还是变差了？—— **变差了**

同一任务（值域封闭的单步分类）、20 条输入、8 个对照臂：

| 臂 | 配置 | 全对 | Δ vs A | 字段级 | token | 延迟 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **A** zero-shot | 直接出答案 | 12/20 | — | 0.867 | 212 | 724 ms |
| **B** few-shot 原序 | 3 示例，顺序 X | 13/20 | +0.05 | 0.883 | 391 | 896 ms |
| **C** few-shot 逆序 | **同 3 示例，倒序** | **16/20** | **+0.20** | **0.933** | 391 | 878 ms |
| **D** few-shot 换分布 | 换示例集 Y | 15/20 | +0.15 | 0.917 | 390 | 880 ms |
| **E** CoT 提示 | 要求先写理由 | 12/20 | **+0.00** | 0.850 | 454 | 1476 ms |
| **F** 自带思维链 | `reasoning=high` | 13/20 | +0.05 | 0.867 | 394 | 1681 ms |
| **G** CoT×思维链 | 两者叠加 | 12/20 | **+0.00** | 0.850 | **1259** | **6209 ms** |
| **I** Self-Consistency | 采样 5 次投票 | 15/20 | +0.15 | 0.917 | 1954 | 3986 ms |

**三条结论，每条都有数据：**

1. **CoT 提示零收益、负代价**：12/20 → 12/20，但 token +114%、延迟 +104%。
2. **推理模型再加 CoT 提示会变差**：G 比 F 少 1 条（13→12）——
   因为思维链本来就在跑，再加提示是让同一个动作做两遍。
3. **赢的是 few-shot，而且赢在"排序"不在"数量"**：
   C（同 3 个示例倒序）比 B 多 3 条（+0.15）。
   **示例顺序是超参，不是排版细节。** 这是本次最反直觉的一条。

> ⚠️ **文档里明确写了适用边界**：这套结论只在"值域封闭的单步分类"上成立。
> 换任务必须先验"示例顺序"再验"要不要 few-shot"，不能直接套。

### 2.2 temperature=0 是否等于"每次都一样"？—— **不等于**

`docs/cost.md` 与 `learn/llm/01_params.md` 里记录：5 次跑出 3 种不同输出，
且 `seed` 在本网关**实测无效**（同 seed 5 次仍 3 种）。

工程结论：**确定性要自己实现**（缓存 / 校验 / 投票），不能靠参数。

### 2.3 首 token 慢是模型慢还是链路慢？—— **都有，但要分开**

分层打点（`scripts/latency_breakdown.py`，5 次 p50）：

| 层 | TTFT | 开销 |
| --- | ---: | ---: |
| 上游直连（裸 httpx） | **1301 ms** | 模型+网络下限 |
| 经 SDK 封装 | 1341 ms | **+40 ms** |
| 端到端 FastAPI+SSE | 1463 ms | **+122 ms** |

**单次请求慢在模型**（我方只占 162 ms / 12%）。
**但长文档的"首字慢"是链路问题**：50 页 → 9 段**串行** map，
用户等 10899 ms 而单次只要 2–3 秒。**慢的不是模型，是我们串行调了 9 次。**

---

## 三、这周真正学到的四个"机制级"认知

这些不是知识点，是踩到之后才明白的。

### 3.1 `request.is_disconnected()` 在流式响应里恒为 False

任务要求写的是这个 API。实测：

```
[gen] i=0 is_disconnected=False
...
[gen] i=9 is_disconnected=False   ← 客户端早已关闭，仍全是 False
```

**原因**：`StreamingResponse` 已占用 ASGI `receive` 通道，生成器再去读拿不到断开事件。

**解法**：起后台任务监听 `request.receive()` 拿 `http.disconnect`，主循环读一个
`asyncio.Event` 置位的非阻塞标志。实测生效：

```
INFO chat/stream 客户端断开，停止生成（已发送 39 块，270 字，4088 ms）
```

> 这段代码"看起来完全符合要求"，但它一帧都没停过。**符合 API 文档 ≠ 行为正确。**

### 3.2 项目从未配置 logging，`logger.info` 全被静默丢弃

排查断开检测时卡了很久，最后发现不是检测没生效，是**日志根本没输出**。
root logger 未配置时 WARNING 以下直接蒸发。

新增 `app/core/logging.py`，在 `main.py` import 时装配。
**这类问题会伪装成"功能没实现"**，比真正的 bug 更费时间。

### 3.3 `chunk_tokens` 有隐式下限，参数被静默抬高

原实现 `max(int(tokens/0.7), 200)`：传 `chunk_tokens=30` 实际得到 200 字/段。

**"参数不生效"比报错更难查** —— 调用方以为自己在用小预算，日志里一切正常。

### 3.4 测试依赖外部服务 = 定时炸弹（10/27 实证）

网关凭据过期（`502 未配置凭据`）→ `LlmExtractor` 降级 → 响应体多出 `degraded` 键
→ **契约测试 `test_api.py` 直接挂掉，而那个测试一行都没改、也没写错**。

这条因果链直接解释了为什么必须有 `dependency_overrides` + `FakeLLMClient`：

> **如果测试依赖真实模型，那它测的就不是你的代码，而是供应商的可用性。**

---

## 四、四次"自我推翻"——这周最有价值的部分

比"做对了什么"更能说明工程成熟度的，是**发现自己错了并改口径**。

### 4.1 成本口径错误：以为省 62%，实际贵 37%

原 `summary.json` 用**总量**算成本比值。但**失败的请求不产出 token**，
于是"总成本更低"只等于"有几次没跑起来"。

按每条成功样本归一后：

| 版本 | 原报 | 归一口径 | 真相 |
| --- | ---: | ---: | --- |
| 摘要 v2 | 省 62% | 0.03266 → 0.04471 元 | **v2 实际贵 37%** |

更隐蔽的是 `extract` —— 方向对我有利，一开始没发现：
总量口径两版一样贵，按成功归一 v2 省 58%，因为 v1 有一条失败样本
在修复轮里烧了巨额 completion token，**撑大了分母**。

> **结论：污染是双向的，不能靠"数字看着不对"来发现，只能把口径写死在代码里。**

### 4.2 单样本导致错误结论

`docs/why-i-use-langchain.md` 的提交信息里写着"修正一处基于单样本的错误结论"。
类似地，参数实验里也曾因单次测量得出与多次平均完全相反的结论。

### 4.3 tiktoken 估中文高估 49%

| 文本 | tiktoken 估算 | 真实 token | 偏差 |
| --- | ---: | ---: | ---: |
| 中文 176 字 | 177 | 119 | **1.487×** |
| 英文 395 字符 | 108 | 120 | 0.900× |
| 代码 367 字符 | 93 | 105 | 0.886× |

**精确成本只能用网关返回的真实 `usage`**；预估用密度系数（中文 0.68 tok/字）。

### 4.4 `reasoning_effort=none` 并没有真的关掉推理

5 次样本：关推理均值 242 tokens、默认 211 —— **方差比差异还大**。
诚实结论：**没能证明这个参数有效**，不能假设"关掉推理能省钱"。

---

## 五、`structured_call`：本周最重要的代码产出

`docs/structured-output.md` 的定位是"W4 拼 RAG prompt、W5 查询改写、
W7 judge、W9 工具参数解析全都要调它"。签名已定版：

```python
async def structured_call(
    prompt: str,
    schema: type[M],
    *,
    max_repair: int = 2,
    fallback: BaseModel | None = None,
) -> M:
```

### 三层防线（为什么不是一层）

| 层 | 手段 | 拦什么 | 可靠性 |
| --- | --- | --- | --- |
| ① prompt | 要求只输出 JSON | 让模型"倾向"正确 | ❌ 不可靠 |
| ② JSON Schema | `response_format=json_schema` | 结构坏 | ⚠️ 网关未必执行 |
| ③ Pydantic | 本地校验 | 语义问题 | ✅ 确定 |

**关键实测**：本网关**接受** `response_format` 参数但**不一定执行**。
所以强约束层不能当唯一保障 —— 被网关拒绝时自动关掉再试一次（**强约束是优化项，不是前提**）。

### 修复重试：为什么回灌报错原文有效

把 Pydantic 的错误**原样**喂回对话：

```
你上一次的输出不符合要求，问题如下：
- 字段 合同编号：Input should be a valid string（你给的是 {'nested': 1}）
```

比"重发同一个 prompt 碰运气"有效，因为错误信息直接告诉了模型**哪里不对、错在哪个字段、收到了什么值**。

**为什么是 2 轮**：每轮一次计费调用 + 秒级延迟，第三轮收益急剧下降而用户已等 6 秒+。

---

## 六、成本模型的硬结论

`docs/cost.md`（`scripts/cost_model.py` 生成，数据来自实测 token 数）：

| 场景 | 单次成本 | 每 1,000 次 |
| --- | ---: | ---: |
| 规则版（兜底，纯 CPU） | **0** | **0** |
| 标准合同 · 一次成功 | 0.001166 元 | 1.17 元 |
| 标准合同 · 2 次回灌后成功 | 0.004138 元 | 4.14 元 |

**① 回灌重试不是"多一点时间"，是"多 3.5 倍钱"。**
所以"提高首次成功率"（更明确的 prompt、标量归一）比"加大重试"划算得多。

**② prompt 精简 30% 只省 11.8%。**
因为标准合同的输入只占总价 39%：`30% × 39% ≈ 11.8%`。
想靠精简 prompt 省钱，得先让**输出**变短。

**③ 缓存是唯一"纯收益"的手段。**
省 50% 且无质量代价 —— 因为返回的是同一输入的同一份结果。
前提是"同一输入"真的可复现，所以要按**内容**做缓存键，不能靠"再跑一次期望相同"。

---

## 七、System Prompt：一个被普遍忽略的假象

`docs/system-prompt.md` 里最值钱的一节：「**拒绝率打平是假象**」。

只看"拒绝率 0.60 = 0.60"，结论会是"两版一样，白加了 219 token"。**这是错的。**

把"拒绝"拆成四档：

| 拒绝形态 | A 版 | B 版 | 下游能用吗 |
| --- | ---: | ---: | --- |
| 显式拒绝（合法 JSON + 说明） | **0** | **3** | ✅ 程序可识别 |
| 笼统搪塞（合法 JSON + "无法判断"） | **2** | **0** | ❌ 与"信息不足"无法区分 |
| 自然语言拒绝（非合法 JSON） | 1 | 0 | ❌ 解析崩 |
| 正常分类 | 2 | 2 | — |

**同一个 0.60，A 版全是"搪塞"，B 版全是"显式拒绝"。**
指标打平，可用性天差地别。

> **这是"只看聚合指标会骗人"最好的例子** —— 必须拆到"下游能不能用"这一层。

---

## 八、为什么不用 LangChain（`why-i-use-langchain.md`）

结论：**不是因为它坏，而是它抽象掉的那一层，恰好是我必须自己控制的那一层。**

实测失败栈记录在文档里（这是"能直接讲三分钟"的素材）：

- `json_schema` / `json_schema + strict`：模型没按 schema 出时，框架直接抛 `ValidationError`
- 拿不到原始 error body → 无法把 Pydantic 报错**原样回灌**做修复
- 而"回灌报错原文"正是修复成功率的核心手段（见 §5）

**这个判断的依据是实测，不是偏好。**

---

## 九、模板库与 A/B 脚手架的纪律

`app/prompts/` 的结构与规则（`docs/prompt-templates.md`）：

```
app/prompts/
├── loader.py              模板加载 + 版本解析
├── hardcoded_blocklist.txt 硬编码业务词黑名单
├── extract/  v1.jinja2  v1.1.jinja2  v2.jinja2  + schema.py
├── classify/ v1.jinja2  v2.jinja2     + schema.py
├── qa/       v1.jinja2  v2.jinja2     + schema.py
└── summarize/v1.jinja2  v2.jinja2     + schema.py
```

**两条纪律**：

1. **模板只做字符串拼装，不放业务逻辑；模板里绝不硬编码业务词**
   （用 `hardcoded_blocklist.txt` + 检查脚本强制）
2. **A/B 只动一个维度** —— 否则测出来的差异无法归因

`eval/prompt-sets/` 是 **40 条 prompt 实验集**，文档里明确写了：
**W7 的 200 条 RAG 评测集不能复用这 40 条** —— 两者目的不同，混用会让指标失去意义。

---

## 十、已知局限（诚实地写出来）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 镜像大小 | ⚠️ **待实测** | v0.1.0 曾 157 MB；v0.2.0 因新增依赖约 +28 MB，但**未实测**（构建需代理） |
| `reasoning_effort=none` | ⚠️ **未证有效** | 5 次样本无统计差异，不能假设能省钱 |
| 四范式结论 | 🟡 **有边界** | 只在"值域封闭单步分类"成立，换任务必须重验 |
| chunking / embedding | ⬜ 未做 | W4 范围 |
| RAG 评测集（200 条） | ⬜ 未做 | W7 范围 |

---

## 十一、这周回答"你怎么保证代码可测试"的完整答案

如果面试只问一个问题，答案在这里：

**① 依赖注入是前提** —— 服务层不 import fastapi，`LlmExtractor` 从构造参数拿 client，
内部绝不 `new`。所以测试能直接塞 `FakeLLMClient`。

**② 三层离线是手段**：

| 层 | 手段 | 覆盖 |
| --- | --- | --- |
| 单测 | `FakeLLMClient`（stub，不是 mock） | 校验回灌、降级、键序、编造拦截 |
| HTTP | `httpx2.MockTransport` 注入 | SDK 请求构造、`response_format` 真发了吗 |
| API | `conftest.py` autouse `dependency_overrides` | `test_api.py` 一行不改仍全绿 |

**③ 拔网线验证是证据** —— 不是"我觉得能离线"，是：

```
LLM_BASE_URL=http://127.0.0.1:9/v1   # 不可达
http_proxy=http://127.0.0.1:9        # 代理置空
pytest -q                            # 327 passed
```

**④ 装配层单独验** —— `test_wiring.py` 证明默认实现是 `LlmExtractor`、
缺 key 启动即失败、装配不发网络请求。这补的是 CI 唯一的盲区。

> 一句可复用的话：**"能离线测绿" ≈ "实现真的可替换"。
> 因为如果测试依赖真实模型，它测的就不是你的代码，而是供应商的可用性。"**

---

## 附：产物索引

| 类别 | 路径 |
| --- | --- |
| 接入层 | `app/core/llm/`：`client.py` `retry.py` `streaming.py` `structured.py` `tools.py` `tool_loop.py` |
| Prompt 库 | `app/prompts/`（4 类 × 2 版 + schema + 加载器 + 硬编码检查） |
| 评测集 | `eval/prompt-sets/`（40 条，4 类 jsonl） |
| 实验脚本 | `scripts/`：`lab_patterns.py` `lab_sysprompt.py` `ab_test.py` `e2e_prompts.py` `latency_breakdown.py` `cost_model.py` |
| 文档 | `docs/`（12 篇）：`structured-output.md` `prompt-patterns.md` `system-prompt.md` `prompt-templates.md` `ab-scaffold.md` `prompt-ab.md` `why-i-use-langchain.md` `cost.md` `sse.md` `api-spec.md` |
| 预研笔记 | `learn/llm/`：`01_params.md` `02_function_calling.md` `03_cost_and_reliability.md` |
| 测试 | `tests/`（327 个，全离线） |
