# 结构化抽取 API 契约（docs/api-spec.md）

> **这份文档是本项目唯一不能动的东西。**
> 路径、请求体、响应体、错误码含义冻结在 `v1`；「怎么抽」是可替换的实现。
> W2 把 `app/services/extractor.py` 的正则换成 LLM 调用时，本文件**一个字都不用改**，
> 路由、schema、测试也一行不用动 —— 测试全绿才叫「API 稳定、实现可换」。
>
> 自检：本文件里出现 `正则`、`prompt`、`模型参数` 之类的实现细节 = 契约被污染了。

| 项 | 值 |
| --- | --- |
| 服务 | `industrial-ai-platform` |
| 契约版本 | `v1`（响应头 `X-API-Contract: v1`） |
| 应用版本 | `0.2.0` |
| 端点 | `POST /api/v1/extract` |
| 内容类型 | `application/json; charset=utf-8` |
| W1 实现 | `app/services/extractor.py` 规则版（正文里不可见） |
| W2 实现 | **同一签名**换成 LLM，契约不变 |

---

## 1. 请求

```http
POST /api/v1/extract HTTP/1.1
Content-Type: application/json
```

### 1.1 请求体（2 个字段，不可增删）

| 字段 | 类型 | 必填 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `text` | `string` | 是 | 去首尾空白后非空；长度 ≤ `20000` | 待抽取正文，纯文本 |
| `fields` | `string[]` | 是 | 1–50 项；每项去空白后 1–64 字符；去重 | 期望抽取的字段名，中英文都收 |

- **未知字段名不报错**：`fields` 里出现字典外的名字，返回 `null` 而不是 `422`。
  字段名由调用方定义，服务端字典永远不完备；报错等于把「不支持」的责任推给调用方，
  而正确做法是**如实回答「没抽到」**。
- **未知键报 `422`**：请求体多出任何顶层字段（如 `texts`），不静默忽略。
- **`fields` 重复报 `422`**：重复项会让 `confidence` 的分母失真。

### 1.2 字段名参考（不是契约，只是 W1 规则表覆盖范围）

匹配顺序：精确名 → 别名（大小写不敏感）。

| 语义 | 可用名字（任选其一） | 输出形态 |
| --- | --- | --- |
| 合同编号 | `合同编号` / `合同号` / `contract_no` / `contract_id` | `HT-2026-001`（原样） |
| 日期 | `日期` / `签订日期` / `签署日期` / `date` / `sign_date` | 统一 `YYYY-MM-DD` |
| 金额 | `金额` / `合同金额` / `总金额` / `amount` / `total_amount` | 字符串，保留 `,` 与小数 |
| 甲方 | `甲方` / `买方` / `委托方` / `party_a` / `buyer` | 标签后到句读符前的文本 |

表外字段名走兜底：`<字段名>：<值>` 形式能抽到，否则 `null`。

### 1.3 请求示例

```json
{
  "text": "合同编号：HT-2026-001，甲方：张三，签订日期：2026-10-20，合同金额：¥128,000.00",
  "fields": ["合同编号", "甲方"]
}
```

---

## 2. 成功响应（`200 OK`）

### 2.1 响应体（**4 个必选字段** + 1 个可选字段，顺序固定）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `request_id` | `string` | 本次请求 id，等于响应头 `X-Request-ID`；报障只报它 |
| `data` | `object` | 字段名 → 值；**键与请求 `fields` 完全一致、顺序一致**；抽不到为 `null` |
| `confidence` | `number` | 整体置信度 `0.0`–`1.0`：命中比例，必抽字段缺失按比例打折 |
| `elapsed_ms` | `integer` | 服务端处理耗时（毫秒，含内部延迟），不含网络往返 |
| `degraded` | `boolean`（**可选**，见 §2.5） | 本次结果是否由兜底实现产出。**只在 true 时出现**，正常时为缺省 |

`data` 的值恒为 `string | null`：金额也返回原始字符串（保留 `,`/小数），
需要数值请调用方自行 `Decimal` 化 —— 服务端不做隐式转换，避免精度/币种歧义。

### 2.5 `degraded` 字段（v1.1 新增，向后兼容）

**为什么需要它**：LLM 实现在两种情况下会退回规则版 —— 模型/网络失败，或
结构化输出连续两轮校验不通过。这两种情况**仍然返回 200 和完整结果**（这是刻意的：
用户要的是抽取结果，不是错误码），但调用方有权知道"这个结果不是模型给的"。

规则（严格遵守才不会破坏调用方）：

1. **只在 `true` 时出现这个键。** 正常结果里**没有** `degraded` 键，
   所以按 4 字段解析的老客户端不受任何影响 —— 这是它向后兼容的全部原因。
2. **不参与 `confidence` 计算。** 两者的关系是：`degraded=true` 说明**来源**变了，
   而 `confidence` 仍然只反映"抽到了多少"。低置信度 + 未降级、高置信度 + 已降级
   都是可能的，调用方需要分开判断。
3. **同时 `X-Extract-Model` 响应头会带 `+fallback(...)` 后缀**，与 `degraded` 同步出现。
   头里是**原因**，body 里是**布尔事实** —— 前者给人排查，后者给程序分支。

```json
{
  "request_id": "…",
  "data": {"合同编号": "HT-2026-001", "甲方": null},
  "confidence": 0.375,
  "elapsed_ms": 102,
  "degraded": true
}
```

### 2.2 全命中

```json
{
  "request_id": "0c7d3f2a-1f47-4a0e-9a5f-2b6f0f1f9c31",
  "data": {"合同编号": "HT-2026-001", "甲方": "张三"},
  "confidence": 1.0,
  "elapsed_ms": 101
}
```

### 2.3 部分命中（不报错，降置信度）

`甲方` 在正文里不存在：

```json
{
  "request_id": "…",
  "data": {"合同编号": "HT-2026-001", "甲方": null},
  "confidence": 0.375,
  "elapsed_ms": 100
}
```

### 2.4 响应头

| 头 | 出现时机 | 说明 |
| --- | --- | --- |
| `X-Request-ID` | 所有响应 | 回显客户端传入值，未传则服务端生成 |
| `X-API-Contract` | 所有响应 | 固定 `v1`，调用方据此判断兼容性 |
| `X-Extract-Model` | 仅 `200` | 产出该结果的实现标识（W1=`rule-v1`，W2=模型名）。**实现信息只走头，不进 body** |
| `X-Elapsed-Ms` | 仅 `200` | 与 body 的 `elapsed_ms` 一致，方便只读头的网关/日志 |

---

## 3. 错误响应

**所有非 2xx 都是同一个形状（4 个字段，不多不少）**：

```json
{
  "code": "INVALID_INPUT",
  "message": "请求体校验失败，详见 details",
  "request_id": "0c7d3f2a-…",
  "details": [
    {"field": "text", "reason": "text 去空白后不能为空"}
  ]
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | `string` | 机器可判的稳定错误码，见 §3.1；**只追加，不改旧值** |
| `message` | `string` | 给人看的一句话，不含堆栈、不含内部路径 |
| `request_id` | `string` | 与响应头一致 |
| `details` | `array` | 字段级明细；`422` 必有，`400`/`500` 为空数组 |
| `details[].field` | `string` | 请求体内路径，如 `text`、`fields.2`；跨字段为 `_model` |
| `details[].reason` | `string` | 原因 |

### 3.1 错误码表

| HTTP | `code` | 触发条件 | 调用方动作 |
| --- | --- | --- | --- |
| `400` | `INVALID_JSON` | 请求体无法解析为 JSON | 修请求体，不要重试 |
| `422` | `INVALID_INPUT` | 结构合法但校验失败：`text` 空/超长、`fields` 空/超 50 项/重复/单项超长、多传未知键、**顶层不是对象**（如 `[]`） | 按 `details[].field` 修，不要重试 |
| `500` | `INTERNAL_ERROR` | 服务端未预期异常（含抽取实现自身故障） | 带 `request_id` 报障；可退避重试 1 次 |
| `500` | `RESPONSE_CONTRACT_VIOLATION` | 出参不符合契约（服务端 bug，不是请求的错） | 带 `request_id` 报障，**不要重试** |

> **`400` 与 `422` 的界线：能不能解析成 JSON。**
> 解析不了 = `400`（连字段都看不到）；能解析但字段不合规 = `422`（能指出具体字段）。
> 顶层是数组/字符串这类「能解析但不是对象」算 `422`，`details[0].field` 为 `_model`。

**没有 `413`**：正文超长属于**校验失败**（`422` + `field: text`），
不是协议层的「实体过大」；网关自己的 `413` 不在本契约内。

### 3.2 错误示例

`422` 正文为空：

```json
{"code": "INVALID_INPUT", "message": "请求体校验失败，详见 details", "request_id": "…",
 "details": [{"field": "text", "reason": "text 去空白后不能为空"}]}
```

`422` 正文超长：

```json
{"code": "INVALID_INPUT", "message": "正文长度超过上限 20000", "request_id": "…",
 "details": [{"field": "text", "reason": "长度 25000 超过上限 20000"}]}
```

`400` 畸形 JSON：

```json
{"code": "INVALID_JSON", "message": "请求体不是合法的 JSON", "request_id": "…", "details": []}
```

---

## 4. 稳定性规则（W2 换实现时逐条核对）

| # | 规则 | W2 怎么做 |
| --- | --- | --- |
| 1 | 路径、方法、请求体 2 字段、约束不变 | 不改 |
| 2 | 响应体恒为 4 个必选字段；新增只能是**可选**字段，且先改本文件 | degraded 已按此规则加入（v1.1） |
| 3 | 错误形状恒为 4 字段，`code` 语义不变 | 不改；模型超时映射到 `500 INTERNAL_ERROR` |
| 4 | `confidence` 语义不变：0–1，越高越可信 | 采用模型分数或归一化 |
| 5 | 抽不到 = `null`，**永不编造** | prompt 强制「无依据输出 null」 |
| 6 | 规则版与 LLM 版跑**同一套** `tests/test_api.py` | 测试即契约的可执行形态 |
| 7 | 服务层入口签名不变：`async extract(text, fields)` → 统一结果对象 | 规则版/LLM 版共用签名，路由只依赖签名 |

> 规则 6 是本契约的落地方式：**测试一行不改而全绿**，才算实现真的可换。

---

## 5. 错误码一览（机器可读）

```
INVALID_JSON                 400
INVALID_INPUT                422
INTERNAL_ERROR               500
RESPONSE_CONTRACT_VIOLATION  500
```

## 6. 变更记录

| 日期 | 版本 | 变更 | 兼容性 |
| --- | --- | --- | --- |
| 2026-10-24 | v1 | 初版：POST /api/v1/extract + 规则版实现 | — |
| 2026-10-27 | v1.1 | 新增**可选**字段 degraded（只在降级时出现）；规则版与 LLM 版共存，LLM 失败自动兜底 | ✅ 向后兼容：老客户端按 4 字段解析不受影响 |

---

# 项目一：缺陷图像生成 API（`POST /api/v1/generate`）

> W8 周三新增的**独立接口线**。本节只追加，上文 extract 契约（§1–6）一个字不改。
> 复用平台级约定：统一错误体 4 字段、错误码"只追加不改旧值"、`request_id` /
> `X-Request-ID` 透传、`X-API-Contract` 头 —— 全部与 extract 同源，不新造。

| 项 | 值 |
| --- | --- |
| 端点 | `POST /api/v1/generate` |
| 模式 | **同步**（请求挂起直到出图；异步版 = 202 + task_id，后续按同一契约套壳） |
| 实现开关 | `GENERATOR_IMPL=fake`（默认，离线）/ `sd`（真机）——服务层签名不变 |
| 实现标识头 | `X-Generator-Model`（200 时出现） |

## 7. 请求

```http
POST /api/v1/generate HTTP/1.1
Content-Type: application/json
```

### 7.1 请求体（4 个字段，不可增删）

| 字段 | 类型 | 必填 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `defect_class` | `string` | 是 | 白名单内；去空白后 1–32 字符 | 缺陷类别，白名单见 §7.2 |
| `count` | `int` | 否 | 1–8，默认 1 | 生成张数。**上限是显存边界，不是产品策略** |
| `size` | `int` | 否 | `512` 或 `768`，默认 512 | 输出边长（正方形）。其它尺寸出分布外畸形图，契约层直接不收 |
| `seed` | `int` | 否 | ≥ 0 | 随机种子；传了则同参数可复现 |

- **多传未知键 → 422**（与 extract 同款 `extra="forbid"`）。
- **`count`/`size` 的上限进契约而不是进文档建议**：`count=1000` 的请求会在
  GPU 上挂到 OOM 并连带打死排队请求 —— 没有上限的生成接口等于自杀。

### 7.2 缺陷类别白名单

`划痕` · `凹坑` · `气泡` · `裂纹` · `锈蚀` · `污渍`

白名单 = LoRA 适配器目录名。**新类别 = 加权重 + 加白名单，同一个提交**。
表外类别**不静默用基座模型生成**（调用方以为在生成划痕，实际是随机噪声）——
返回 `422 UNKNOWN_DEFECT_CLASS`，与 extract"抽不到 = null 不编造"同一产品哲学。

## 8. 成功响应（`200 OK`）

```json
{
  "request_id": "…",
  "images": [
    {"path": "data/generated/划痕_512_42.png", "size": 512, "seed": 42}
  ],
  "elapsed_ms": 12400,
  "model": "sd15-lora-v0"
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `request_id` | `string` | 与响应头 `X-Request-ID` 一致（平台级约定） |
| `images[]` | `array` | 长度 = 请求 `count`；顺序即批次顺序 |
| `images[].path` | `string` | 相对路径，可直接 GET 静态服务取图。**不回 base64**（8 张 768px 的 JSON 会膨胀到 10MB+） |
| `images[].seed` | `int` | **实际使用**的种子（请求未传时为服务端生成值） |
| `elapsed_ms` | `int` | 含 GPU 推理与存盘，不含网络往返；响应头 `X-Elapsed-Ms` 同值 |
| `model` | `string` | 实现标识；`X-Generator-Model` 头同步 |

**没有 `confidence`**：生成的"置信度"不存在，硬造一个会污染两个契约的语义。

## 9. 错误码（本端点新增；形状与 §3 完全一致）

| HTTP | `code` | 触发条件 | 调用方动作 |
| --- | --- | --- | --- |
| `422` | `UNKNOWN_DEFECT_CLASS` | 类别不在白名单 | 改类别，不要重试 |
| `503` | `MODEL_NOT_LOADED` | 权重未加载 / 设备不可用 | 联系维护者；不重试 |
| `503` | `GPU_OOM` | 显存不足 | **带 `Retry-After` 头**，按它退避后可减参重试 |
| `504` | `GENERATION_TIMEOUT` | 单次生成超过 180s | 减小 count/size 后重试 |
| `422` | `INVALID_INPUT` | count/size/seed 越界、空类别、多传键 | 按 `details[].field` 修 |
| `500` | `INTERNAL_ERROR` | 未预期异常 | 带 `request_id` 报障 |

```
UNKNOWN_DEFECT_CLASS         422
MODEL_NOT_LOADED             503
GPU_OOM                      503
GENERATION_TIMEOUT           504
```

## 10. 失败处理的设计立场

- **OOM/超时/权重三类的 HTTP 语义刻意不同**：503（可等）与 504（任务太重）分开，
  调用方的重试策略才能自动化。
- **异常路径的额外头跟异常走**：业务异常抛出后路由自己的写头代码不执行，
  `Retry-After` 由 `AppError.headers` 携带、全局 handler 透传。
- **同步版先行**：异步版的复杂度在"状态机 + 查询 + 清理"，同步版先把契约、
  错误码、单测立住 —— 异步版只是给同一份契约套 202 + task_id。

| 日期 | 版本 | 变更 | 兼容性 |
| --- | --- | --- | --- |
| 2026-12-09 | 项目一 v0 | 新增 POST /api/v1/generate（同步版）+ 4 个新错误码 | ✅ 新端点，extract 契约零改动 |

---

# 平台 v2：鉴权 · 限流 · 冷热分离（`/api/v2/*`，W12 周一新增）

> **v1 契约（§1–10）一个字不改。** v2 是新前缀 + 必填鉴权 + 限流的新接口线：
> 给 v1 加必填鉴权 = 破坏所有现有调用方（§4 规则 1）。v2 的样板端点是 extract
> （请求/响应体与 v1 完全一致，复用同一 schema 类），其余端点按同一模式逐个迁移。

## v2.0 总览

| 项 | 值 |
| --- | --- |
| 前缀 | `/api/v2` |
| 契约版本 | `v2` |
| 鉴权 | 请求头 `X-API-Key`（必填；`/health` `/ready` `/metrics` `/docs` 除外） |
| key 配置 | 环境变量 `PLATFORM_API_KEYS`（逗号分隔多个；**未设置 = 鉴权关闭**，本地/CI 显式降级，生产必须设置） |
| 限流 | 双层令牌桶：全局 `PLATFORM_GLOBAL_QPS`（默认 10）+ 单 key `PLATFORM_KEY_QPS`（默认 3）；桶容量 = 速率 × 2s 突发 |
| 限流实现 | **单实例内存桶**。多实例部署时实际配额 = 单实例 × 实例数，精确限流需 Redis + Lua —— 部署第二个实例前必须重读本节 |

## v2.1 鉴权

| 失败场景 | HTTP | `code` | 说明 |
| --- | --- | --- | --- |
| 无 `X-API-Key` 头 | `401` | `UNAUTHORIZED` | 与"key 错误"**同一个 message** |
| key 不在配置列表 | `401` | `UNAUTHORIZED` | 401 永远不区分"缺"与"错" |

- **401 不做 oracle**：区分"key 缺失"与"key 错误"会让 401 变成
  "这把 key 存不存在"的枚举探测器。统一一句话："缺少或无效的 X-API-Key"。
- **日志脱敏**：key 在日志、metrics、错误响应里恒为掩码（前 3 位 + 长度），
  绝无明文。metrics 按 key 的 sha256 指纹（12 位）分桶，不看明文。

## v2.2 限流

| 超限场景 | HTTP | `code` | 响应头 |
| --- | --- | --- | --- |
| 单 key 配额超限 | `429` | `RATE_LIMITED` | `Retry-After`（按缺口速率动态计算，非固定值） |
| 全局 QPS 超限 | `429` | `RATE_LIMITED` | `Retry-After` |

- **两个桶都要有**：只做全局限流的坑是一个客户端挤掉所有人，
  而监控里"全局 QPS 正常"。`/metrics` 的 `rate_limit_rejections` 按 key 指纹
  分桶计数，"谁在挤谁"从 `/api/v2/rate-limit/status` 看。
- **检查顺序**：先单 key 桶后全局桶。单 key 超限是调用方自己的节奏问题，
  全局超限是全体的问题；且被全局拒绝的请求会把单 key 的令牌退回
  （两层桶互不污染）。
- 成功响应带 `X-RateLimit-Remaining`（当前 key 剩余令牌，整数）。

## v2.3 冷端点（检索-only，压测专用）

`POST /api/v2/search` —— 走完整检索链路（embed → search → TopK），
**零 LLM 调用**。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `query` | `string` | 1–500 字符 |
| `k` | `int` | 1–8，默认 3 |

响应（`200 OK`）：`{request_id, hits[], elapsed_ms}`；每条 hit 为
`{id, source, page, score, text(≤500字)}`。

- **用途**：压测服务层延迟上限（P99 不含模型侧）；标定热端点的模型开销
  （热 − 冷 = 模型代价）；检索质量回归探针。
- 无关查询返回低分 top-k（分数如实给出），不返回错误 —— 「没有」由调用方
  按分数阈值判断，与 `retrieve_spec_chunks` 的空结果语义同源。

## v2.4 liveness 与 readiness 分离（必答问题）

| 探针 | 端点 | 语义 | 就绪条件 |
| --- | --- | --- | --- |
| liveness | `GET /health`（v1 已有，不变） | 进程活着吗 | 事件循环能响应 |
| readiness | `GET /api/v2/ready` | 能接流量吗 | **模型单例已构造成功** |

- 模型加载要 30 秒。只有一个健康检查时，编排器会在加载窗口看到"不健康"→
  重启容器 → 永远起不来。两个问题、两个探针、两个端点。
- readiness 未就绪返回 `503 NOT_READY`（带 `Retry-After: 5`），
  就绪返回 `200 {status:"ready", checks:{generator,extractor}, elapsed_ms}`。
- readiness 只**观察**单例是否已构造，绝不触发构造 —— 探针请求把 30 秒
  加载提前拉起 = 观察者参与了被观察的系统。
- 三个探针端点全部绕过鉴权与限流：探针被 429 挡住 = 编排器在高负载时
  重启活着的容器（与死循环是同一课的镜像面）。

## v2.5 W11 五类错误 → HTTP 映射（W12 服务化端点的预备表）

Agent 服务路径（W12 本周要接 HTTP 的 interrupt/进度查询端点）抛错时按此表映射。
现状说明：五类目前存在于工具信封层（`error_contract.py`），此表是该层
未来 HTTP 化的映射契约，先于实现冻结 —— 与 W1"先契约后实现"同一顺序。

| category（信封） | HTTP | 业务码 | 调用方动作 |
| --- | --- | --- | --- |
| param | `422` | `AGENT_INVALID_ARG` | 改参数，不重试 |
| permission | `403` | `AGENT_NOT_PERMITTED` | 与 401 分开：401=没亮对 key，403=key 对但无权 |
| timeout | `504` | `AGENT_TIMEOUT` | 按 Retry-After 退避 |
| empty | `200` | （结果体表达） | "没查到"不是错误；映射 4xx 会教调用方无意义重试 |
| duplicate | `409` | `AGENT_CONFLICT` | 动作已发生（幂等拦截），不要重试 |
| internal | `500` | `AGENT_INTERNAL` | 退避重试 1 次，仍失败报障 |

## v2.6 背压（项目一队列，动态 Retry-After）

`POST /api/v1/generate/async` 的 `QUEUE_FULL`（429）升级为动态退避：

- 旧版：静态 `Retry-After: 60` —— 队列里排 30 个任务时按它退避**必然**再吃 429；
- 新版：`Retry-After = 排队任务数 × 近期平均任务时长`（滚动窗口 20 个样本，
  成功与失败都算 —— OOM 任务的耗时同样是后面排队者的等待）；
  message 带原因："生成队列已满（N 个任务在排队），预计等待 X 秒"。
- 无样本时用保守兜底（120s，SD 口径上沿）。**宁可高估不多估**：
  高估是客户端多等一会儿，低估是白跑一趟。

## v2.7 机器可读错误码（v2 新增，只追加）

```
UNAUTHORIZED                 401
RATE_LIMITED                 429
NOT_READY                    503
QUEUE_FULL                   429（动态 Retry-After，v1 端点沿用）
AGENT_INVALID_ARG            422（预备）
AGENT_NOT_PERMITTED          403（预备）
AGENT_TIMEOUT                504（预备）
AGENT_CONFLICT               409（预备）
AGENT_INTERNAL               500（预备）
```

## v2.8 变更记录（平台级）

| 日期 | 版本 | 变更 | 兼容性 |
| --- | --- | --- | --- |
| 2026-01-04 | v2.0 | 新增 `/api/v2` 接口线：X-API-Key 鉴权（401）、双层令牌桶限流（429 + 动态 Retry-After + X-RateLimit-Remaining）、`/api/v2/extract` 样板端点、`/api/v2/search` 冷端点、`/api/v2/ready` readiness、`/api/v2/rate-limit/status`；`/generate/async` 的 QUEUE_FULL 改动态 Retry-After | ✅ v1 全部端点零改动；v2 为纯新增 |
