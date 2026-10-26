# SSE 与流式调用（W1 周三）

> 实测记录。代码：`app/core/llm/streaming.py`（流式封装）、`scripts/raw_sse.py`
> （裸 SSE 服务）、`scripts/bench_stream.py`（流式 vs 非流式）、
> `app/api/v1/stream.py`（SSE 端点）、`tests/test_streaming.py`。

---

## 必答问题：为什么 AI 产品的流式用 SSE 而不是 WebSocket？

表格就是答案，但先一句总纲：**AI 对话的本质是"服务端单向持续推送一段文本"，
WebSocket 的全部双向能力在这里用不上，还要为此付高昂的实现和运维成本。**

| 维度 | SSE | WebSocket |
| --- | --- | --- |
| **方向性** | 单向（服务端 → 客户端）。AI 对话恰好只需要这个 | 全双工（客户端、服务端都能随时发）—— 对话场景里**客户端其实没什么要实时发的** |
| **走 HTTP** | 就是 HTTP（`Content-Type: text/event-stream`）。天然的代理、防火墙、负载均衡友好 | 先发 HTTP `Upgrade`，之后切换成独立 TCP 协议。**很多内网代理/防火墙不认 `Upgrade`，直接掐断** —— 这是 WebSocket 部署时最常踩的坑 |
| **断线重连** | **浏览器原生自动重连**。`EventSource` 断线自动重连 + 可带 `Last-Event-ID` 续传 | 没有原生重连，要自己写 `onclose → 重连 → 对账` 的全套逻辑 |
| **协议复杂度** | 文本分帧（`\n\n`），肉眼可读、可 curl、可抓包 | 帧类型、掩码、心跳、子协议、扩展 —— 实现和处理都重得多 |
| **工具链** | 浏览器内建 `EventSource`，零依赖；curl 一行能测 | 只有 `WebSocket` API，要自己封装；测试要专门工具 |
| **适用** | 服务端 → 客户端的持续数据流（聊天、进度、通知） | 真需要双向低延迟（推塔游戏协同、协同编辑、实时白板） |

**结论**：对话生成是"我请求一次，服务端慢慢吐完"的模型，单向够用。
选 WebSocket 等于为了一个用不到的双向能力，付全部的实现和维护成本。
> 一句话：**能用 SSE 就不用 WebSocket —— WebSocket 只在你"随时需要把客户端的东西推给服务端"时才值。**
> 反向也成立：如果真有实时双向需求，硬上 SSE 会拧巴成短轮询。

---

## 1. 流式调用：三个必须处理对的细节

### 1.1 TTFT 的口径必须写死

**TTFT（Time To First Token）= 从发起请求到收到第一个"非空 content"的时间。**

首块通常会先到（常常只带 `role` 或空 `delta`），把它计入 TTFT 会得到一个
**偏小但没用**的数字。必须等到第一个 `delta.content` 非空才算。

实测（今天早些时候网关正常时）：

```
chunks=28  TTFT=1457ms  total=3195ms
```

TTFT 和 total 是**两个不同的数，别混**：
- TTFT = 体感（用户看到第一行的等待）
- total = 成本/吞吐（生成完的时间）

### 1.2 `usage` 默认不回来，必须显式要

流式下只要最后一块带 usage，且需要：

```python
stream_options = {"include_usage": True}
```

**不加这个参数，usage 永远不会来** —— 成本就算不出来。这是流式最主要的
"静默坑"：token 一样烧，只是你拿不到数字。

且**带 usage 的那块 `choices` 是空数组**：

```json
{"choices": [], "usage": {"prompt_tokens": 15, "completion_tokens": 20, "total_tokens": 35}}
```

所以代码里 `chunk.choices[0]` 会 `IndexError`。先判 `chunk.usage`，再判 `choices`。

### 1.3 `finish_reason` 在"空 delta"的那块上

流式里 `finish_reason` 出现在**已经没有任何内容**的那块：

```
{"delta": {}, "finish_reason": "length"}     ← 生成结束了
{"choices": [], "usage": {...}}               ← 独立 usage 块
data: [DONE]                                  ← OpenAI 自己的终止符
```

不要把"最后一个有内容的块之后还有块就继续循环"当成 bug —— 那是正常的两块收尾。

---

## 2. SSE 协议：字节就是一切

### 2.1 原始字节（`curl -N` 抓到）

```
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"# "},"finish_reason":null}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"三个字"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{},"finish_reason":"length"}]}

data: {"id":"chatcmpl-...","choices":[],"usage":{"prompt_tokens":15,...}}

data: [DONE]
```

### 2.2 为什么必须双换行

SSE 的分帧规则是：**一行 `字段: 值`，一个空行 = 一个事件结束。**

所以每一帧**必然以 `\n\n` 结尾**。为什么：

- 只有一条 `\n`：整个响应是一行连续文本，客户端无法判断事件边界，干脆不派发。
- `data:` 之后必须一个空格。`data:值` 虽也合法，但统一写 `data: ` 更稳。
- `data` 里的内容若含换行，**必须拆成多条 `data:`**，否则换行会被当成字段分隔 ——
  内容被截断或解析错，且**没有任何报错**。

```
# 错误：内容里的换行没拆
data: line1
line2\n\n      ← line2 会被当成一个未知字段 (忽略)

# 正确
data: line1
data: line2\n\n
```

**双换行缺失是 SSE 最难查的 bug**：服务端日志一切正常，浏览器就是 `onmessage`
不触发，而 curl 也可能因为缓冲看起来"到达了"。所以抽了独立的 `sse_frame()` 函数
并用测试锁住字节形状（`tests/test_streaming.py`）。

### 2.3 SSE 没有"标准终止符"

`data: [DONE]` **不是** SSE 规范，是 OpenAI 自己的约定。SSE 的结束方式是：
关闭连接（无 Content-Length，靠 `Connection: close`）、或发自定义 `event: end`。

我们自己的端点用 `event: end` 并带结构化信息（`finished: true`），
比 `[DONE]`（只能说"完了"）有信息量。**换上游供应商时要意识到 `[DONE]` 会变。**

### 2.4 心跳：`:` 开头的注释行

长时间无内容，代理/负载均衡会认为连接空闲而掐断。心跳用注释行：

```
: ping\n\n
```

客户端会忽略以 `:` 开头的行，但连接被保活了。这是合法的 SSE 帧，不是 hack。

---

## 3. 手写裸 SSE：理解 yield + flush 的时机

这一格**故意不套 FastAPI**，用 asyncio + 原始 socket，`scripts/raw_sse.py`。
目的不是再造一个 demo，而是看清三件事：

### 3.1 每次 `drain()` 才是一次真正的 flush

```python
writer.write(frame)
await writer.drain()  # ← 这一行才是"把缓冲里的数据送出去"
```

实测（`--delay 0.8 --ping 0.3`）：

```
15:58:54.941  id: 1
15:58:54.955  data: {"seq": 1, ...}
15:58:55.252  : ping            ← 0.3s 后
15:58:55.764  id: 2             ← 0.8s 后
```

每条**逐条到达**、间隔 0.8s —— 证明服务端每次 `drain()` 都真的 flush 了，
没有攒着。**不 drain 的话，数据留在缓冲区，客户端看到的是"一次性全到"**，
看起来像服务端没写逐条，其实是缓冲的锅。

### 3.2 `curl -N` 是第一道关：客户端也在缓冲

对照实验（同一服务端，服务端确实每 0.8s 发一条）：

```
A) curl -N   : 15:58:54.941 → 15:58:55.764 ...    逐条到达
B) curl 不带-N: 15:58:59.872 全部同一毫秒出现        ← 4 条一起吐出
```

**`-N`（`--no-buffer`）是必须的**。不加的话 curl 自带缓冲，等到攒够才吐，
你会误以为是服务端没 flush。**这是排查 SSE 的第一条**：先 `curl -N`，
确认服务端节奏对，再往上查浏览器/代理。

### 3.3 客户端中途断开是"正常路径"

用户点"停止生成"、关标签页 —— 在裸 socket 上就是 `ConnectionResetError` / `BrokenPipeError`。
要捕获并**正常收尾**，不要当错误打堆栈。流式服务里"被中断的请求"是常态，不是事故。

---

## 4. 服务端的 SSE 端点（`/api/v1/extract/stream`）

### 4.1 为什么是新增端点，而不是把 `/extract` 改成流式

`docs/api-spec.md` §4 明确写了契约冻结。把已有端点改成流式 = **破坏所有现有调用方**
（他们拿到的不再是 JSON）。所以流式是**新路径**，`/extract` 一字未动，
`tests/test_api.py` 仍全绿 —— 这是"契约冻住"的又一次实证。

### 4.2 流式接口最反直觉的约束：错误只能在流里说

一旦开始推事件，HTTP 状态码就是 200，**之后任何错误都不能再返回 500**。
错误只能作为一条 `event: error` 交给客户端。所以：

- 所有可能失败的准备（鉴权、参数、连接）必须在**第一个 yield 之前**做完；
- 流的中间出错，`except` 里 catch 并 yield 一个 `error` 事件，而不是 raise。

事件协议（端点自己定义的，见下）：
```
event: start   data: {"model": "...", "fields": [...]}
event: delta   data: {"text": "增量"}     # 很多条
event: stats   data: {"ttft_ms":..., "total_ms":..., "usage":{...}}
event: end     data: {"finished": true}
event: error   data: {"code":"...", "message":"..."}   # 流开始后才可能出现
```

### 4.3 心跳 + 关缓冲

- **心跳**：长时间无内容会被代理判定空闲掐断。出字正常时不需要，但模型卡住时要保活。
- **`X-Accel-Buffering: no`**：关掉 nginx 这类反代的缓冲（实测 curl 不带 `-N` 那样）。
  服务端和反代/浏览器/curl 的缓冲是**四件事**，每一层都可能把事件攒住。

---

## 5. 网关中断记录（本次实测遇到的上游故障）

基准测试进行到一半，本机网关**整个挂掉**：

```
v1/models          → JSONDecodeError（返回空响应）
trae/qwen-3.5      → 502
qodercn/gmodel     → 502
buddy/glm-5.3      → 502
非流式 buddy/glm-5.3 → 超时（LLM_TIMEOUT=30 对非流式偏短）
```

排查顺序（排除每个可能）：
1. 单个模型挂？→ 试了 3 个不同模型，全 502 → 不是模型单点
2. 流式特有的问题？→ 非流式也 502/超时 → 不是 stream 参数
3. 我的代码问题？→ `v1/models` 也失败 → **网关本身挂了，与代码无关**

**结论**：这是上游服务故障，不是代码 bug。约 10 分钟后网关自行恢复，
重跑基准拿到了完整数据（见下）。**留给文档的教训**：`docs/api-spec.md` §3 早就说
"模型故障映射 500"，真实写照 —— 现实中它是"网关整体 502"，而我们的 `LlmExtractor`
会 fallback 到规则版而不是裸奔（`test_llm_extractor.py` 已验证）。

> **运维提示**：LLM 类依赖必须有几道防线 —— SDK 连接池 + 业务层重试 +
> 上游超时 + 降级实现。只靠任何一道，都会在网关故障时全量失败。

---

## 6. 实测结论：流式 vs 非流式（6 轮样本）

网关恢复后 `scripts/bench_stream.py --runs 6` 的真实结果：

| 指标 | 非流式 | 流式 | 说明 |
| --- | --- | --- | --- |
| **TTFT** | **3209 ms** | **1567 ms** | **体感提升约 2.0×** |
| 总耗时 | 3209 ms | 3159 ms | 差 −50ms，**总时间没变** |
| 分块数 | 1 | 27–67 | 平均块间隔 35.7–88.5 ms |
| 首块样例 | — | `#` → ` ` → `杭` | 逐字符，打字机效果 |

**两个数字必须一起看**：

1. **流式不缩短总耗时**（3209 vs 3159ms，差异在噪声内）。
   它做的是**把等待摊平**：从"白屏 3.2 秒然后一次出现全部"变成
   "1.6 秒后开始逐字出现"。用户感知的等待减半，服务器成本完全一样。
2. **TTFT 才是流式的唯一 KPI**。如果有人问你"流式有没有用"，
   拿总耗时回答会得出"没用"的错误结论。

### 一个诚实的保留：样本量

首轮 3 次时方差极大（第 1 轮非流式 7638ms，第 2 轮 2223ms），
结论一度是"1.4×"。扩到 6 轮后收敛到 2.0×。
**这类测量必须报样本量**，单次测出来的数字没有意义 ——
网关的排队、冷启动、以及上游动态 batching 都会带来秒级抖动。

### 首 token 的位置也值得注意

实测首个 `delta` 常常是 `#`、` `（Markdown 标题符号），而不是正文第一个字。
所以**如果用"第一个非空 delta"做 TTFT，测到的是"模型开始输出格式标记"的时间**，
比"用户看到第一个有意义的字"要早一点点。这个差异很小，但口径要明确 ——
本文档统一用"第一个非空 delta"，因为那是客户端真正能渲染的时刻。

---

## 附：单次调用原始指标（网关正常时段）

| 指标 | 值 | 来源 |
| --- | --- | --- |
| TTFT（流式单次） | **1457ms** | `stream_chat` 实测 |
| 总耗时（流式单次） | 3195ms | 同上 |
| 分块数 | 28 | 同上 |
| `usage` | 仅 `include_usage` 时在末块 | 原始抓包确认 |
| 非流式 TTFT | = 总耗时 | 首字就是全部结束那刻 |

明细在 `scripts/stream_result.json`（`bench_stream.py` 产出）。
