# 错误分类与可恢复性（W11 周一 12/28）

> **一句话**：把"能不能重试"从模型手里拿回来，交给代码。
> 契约在 `app/core/llm/error_contract.py`，四个工具全部接到它上面，
> 分布表在 `scripts/error_stats.py`，20 条路径的单测在
> `tests/test_error_paths_w11.py`。

## 一、五类定义（20:00–20:25）

| 类 | error_type | 可恢复吗 | 靠什么恢复 | retryable | retry_after |
| --- | --- | --- | --- | --- | --- |
| 参数错 | `invalid_json` / `invalid_arg` / `unknown_tool` | 能 | **改参数**（按 hint 点名的字段改）| ✅ True | — |
| 权限错 | `not_permitted` | **不能** | 换工具 / 换范围 / 如实告知无权限 | ❌ False | — |
| 超时 | `timeout` | 能 | **退避重试**，或缩小查询范围 | ✅ True | 2.0s |
| 空结果 | `not_found` / `empty_result` | 部分 | 换关键词或换对象；同参重试必然还空 | ❌ False | — |
| 内部异常 | `internal` | 有条件 | 重试 1 次；仍失败如实告知，不要编造 | ✅ True | — |
| （既有）| `conflict` | 不能 | 效果已发生，直接向上报告 | ❌ False | — |

`conflict` 不是本周新增的类 —— 它是 W8/W9 的幂等拦截（重复投递）。
它不属于五类中的任何一类，硬塞会污染"可恢复"的判断，所以单列。

### 两个容易归错的边界

**1. 权限错 vs 参数错 —— 为什么必须分开。**

改造前，SQL 的表/列白名单拒绝、通知端点的 403，都被塞进 `invalid_arg`
（理由是"模型该做的是放弃这种写法"）。听起来合理，但恢复动作是错的：

- 参数错 → 模型该**改参数**，改完就能成功；
- 权限错 → 模型改任何一种参数都**永远失败**，它该换方案并如实告知用户。

混成一类，模型会对一个永远不可能成功的请求反复微调参数，把预算烧在
一个死路上。所以本周新增 `not_permitted` 独立成类，`retryable=False`。

**2. 空结果的两个落点 —— 读工具 vs 写工具。**

- **读工具**（检索 / SQL）：0 行是**事实**，`ok=true` + `outcome="empty_result"` + hint。
  W9 的 T10 实测过：把它当错误（ok=false），模型会重试；当事实，它才会如实
  转述"该批次没有这个指标"。所以这里必须留 `ok=true`。
- **写工具**（报告 / 通知）：没有内容可写/可发是**无法交付**，
  `ok=false` + `error_type="empty_result"`。

两者共享同一套恢复动作（换对象 / 先去取数据，不是改字段格式），
统计时按 `category == "empty"` 合并。

## 二、结构定版（20:25–20:50）

```
失败: {ok: false, tool, error_type, message, hint, retryable, retry_after[, details]}
成功: {ok: true,  tool, data, outcome: "success"|"empty_result"[, hint]}
```

三个字段、三种读者，这是本周最重要的一条设计判断：

| 字段 | 读者 | 内容 |
| --- | --- | --- |
| `message` | 人 | 发生了什么（日志、报错回溯）|
| `hint` | **模型** | 下一步该做什么（必须可执行）|
| `retryable` / `retry_after` | **代码** | 能不能再试、多久后试（必须可判定）|

### 必答问题：retryable 为什么不能靠模型判断？

模型手上有的是**文本**，不是**元信息**。同一个"没查到"可以是参数写错
（改一个字就成立），也可以是无权限（再试一万次还是拒绝）——但在模型眼里
都只是"失败了"，区别全靠我们喂进去的措辞。于是它只能**按语气猜**：
写得急切它重试，写得平淡它放弃。

而工具这侧知道确定的答案：

| 谁判的 | 它知道什么 | 对应类 |
| --- | --- | --- |
| 校验器 | 改哪个字段就成立 | 参数错 |
| 白名单 | 同一请求永远被拒 | 权限错 |
| 执行环境 | 第几次、多久之后可以再试 | 超时 |
| 查询结果 | 换对象才可能有结果 | 空结果 |
| 未预期异常 | 不知道，但可以试一次 | 内部异常 |

把这些编译成两个字段，循环就能用**代码**执行纪律：

1. `retryable=False` 的错误，observation 直接说"别试了，换策略"；
2. `retryable=True` 的按 `retry_after` 退避；
3. 同类可重试错误超过预算，`apply_retry_budget` **自动**把 retryable 翻成
   False —— "超时 1 次后不再重试"是代码强制，不是提示词建议。

前两条是"让模型知道"，第三条才是"让代码说了算"。**只有第三条是不可绕过的**，
前两条只是降低浪费。

### 一致性约束（写进代码，不靠自觉）

- 不可重试的错误一律 `retry_after=None`（"不能重试但可以 2 秒后再试"
  是自相矛盾的信号，模型会照着后者做）；
- `details` 为 None 时整个键不出现（模型侧少一个噪音字段）；
- 成功信封不放 `retryable`（成功不需要重试，放个 false 只会让
  "这个字段什么意思"变成新问题）。

## 三、埋点（20:50–21:15）

每条工具调用的轨迹记录新增四个字段（`react_loop._action_record`）：

```json
{"tool": "query_metrics_sql", "ok": false, "kind": "not_found",
 "ms": 12, "latency_ms": 12,
 "error_type": "not_found", "retryable": false, "executed": true}
```

- `latency_ms` 与 `ms` 是**同一个数**，两个都留：改老字段名会让 W10 的
  基线对比脚本**静默失配**，而静默失配正是 W10 复盘里最贵的坑。
- 成功调用也落 `error_type=null` / `retryable=null`：分布表要算
  "错误调用/全部调用"，分母必须是所有调用。

`scripts/error_stats.py` 出三张表 + 延迟分位，**递归**读轨迹目录
（产物在 `qa_runs_*/trajectories/*.json`，只扫一层会得到"0 次调用"——
而空表看起来像"零错误"，这是最坏的假阴性）。

### 用 W10 基线跑出来的结果（20 次调用 / 3 次失败）

```
样本量   : 20 次调用（失败 3 次）
错误率   : 15.0%
⚠ 其中 3 次失败来自**旧轨迹**（没有 error_type 字段，kind 只有 "error"）：
  它们的五类归属是**推测**，不能用于跨周对比。
```

**这张表最有价值的一行是那条警告。** W9/W10 的失败信封没有 `error_type` 键，
`wrap_observation` 把所有失败都写成 `kind="error"`，真实分类已经丢失。
硬按 `normalize_error_type("error")` 落成 internal，会把 W10 里那次
"查无批次"（真实是 `not_found`/空结果）误算成内部异常，跨周对比就会得出
"W11 内部异常变少了"这种假结论。所以脚本给这类记录打
`legacy_unclassified` 标记并显式告警 —— **宁可说不确定，不要给一个
看起来确定的错数。**

## 四、补全错误路径（21:15–21:40）：4 工具 × 5 类

| 工具 | 参数错 | 权限错 | 超时 | 空结果 | 内部异常 |
| --- | --- | --- | --- | --- | --- |
| `retrieve_spec_chunks` | JSON 坏 | 索引不可读（EACCES）| 超 `RETRIEVE_TIMEOUT_S`=5s | 0 命中（ok+empty）| 管道异常 |
| `query_metrics_sql` | limit 越界 / 列不在白名单 | **部署未授权该表**（`QA_METRICS_PERMITTED_TABLES`）| 超 `SQL_TIMEOUT_S`=3s | 查无批次 → not_found | 库文件损坏 |
| `generate_report` | 某节 heading 空 | 目录不可写（EACCES）| 超 `REPORT_WRITE_TIMEOUT_S`=5s | sections=[] → empty_result | 渲染崩 |
| `notify` | channel 不在白名单 | 端点 **403** | 端点超时/429 | 正文全空白 → empty_result | 端点不可达 |

四条与 W10 不同的判断，理由都写在代码注释里：

1. **端口不可达 = internal（可重试），403 = not_permitted（不可重试）。**
   归错类，模型就会对一个永远 403 的端点反复重试。
   `notify_http` 里 403 **不重试**（只打一次包），测试断言 `fake.calls == 1`。
2. **sections 空 = empty_result 而不是 invalid_arg。** 参数形状是对的
   （合法空数组），缺的是**内容**。Pydantic 的 `min_length=1` 保留作护栏，
   在**归类那一层**把"唯一的错误就是 sections 太短"翻译成 empty_result。
3. **超时常量放在工具内层（3–5s），而不是靠循环的 10s 兜底。**
   内层超时能被归类成 `timeout`（带退避秒数）；被外层兜底就只剩
   `internal`，丢掉"可以退避重试"的语义。
4. **HTTP 状态码归类抽成 `_http_status_kind()` 一个函数。**
   这段判断原先埋在 `_post_json` 的 except 里，于是只有"从该函数内部抛出"
   的错误才被正确归类；`urllib.error.HTTPError` 同时是 `URLError` 的子类，
   任何从别处冒上来的 403 都会掉进"端点不可达"分支被重试 2 次。
   这个 bug 是写测试时才暴露的 —— 第一版 `test_notify_permission_error`
   期望 `not_permitted`，实际拿到 `internal`。

## 五、单测（21:40–22:00）：20 条，全 mock

`tests/test_error_paths_w11.py`，4 工具 × 5 类 = 20 条 + 3 条横向纪律。

**为什么单独开文件**：`test_real_tools.py` / `test_side_effect_tools.py` 是
**功能**测试（工具能不能干对活），本文件是**契约**测试（干不成活时给出的
那张卡片对不对）。两者红的含义完全不同 —— 功能红是"坏了"，契约红是
"模型会被误导"。

每条都用一个比 `assert r["error_type"] == X` 严格得多的共用断言：

```python
def assert_error_shape(r, *, tool, error_type):
    assert r["ok"] is False
    assert r["tool"] == tool
    assert r["error_type"] == error_type
    for key in ("message", "hint", "retryable", "retry_after"):
        assert key in r  # ← 定版字段一个都不能少
    assert isinstance(r["retryable"], bool)
```

它守的是"所有失败路径长得一样"。只要有一处手搓 dict 漏了
hint/retryable，分布表和模型侧行为都会在那一处悄悄退化。

**mock 的纪律**：注入点在**工具边界之内**（底层函数 / 超时常量），
不是把整个工具替换成假函数 —— 后者会让测试变成"验证 mock 的行为"。

一个实测教训：Windows 上把目录 `chmod` 成只读，**写文件照样成功**
（见 `test_report_permission_error` 的注释）。所以权限错用
`Path.write_text` 抛 `PermissionError` 注入，而不是折腾文件系统
—— 真实环境的触发是只读挂载 / ACL / 杀软，抛的都是同一个 `OSError` 家族。

## 六、结果

- 全套 **1168 passed**（新增 26 条：20 条路径 + 3 条契约纪律 + 3 条重试预算）
- `ruff check` / `ruff format` 全通过
- 四个工具的错误信封**实测同形**（定版七字段），
  唯一的例外是成功信封（不带 retryable，这是刻意的）

重试预算的三条测试值得单说，因为它们守的是本周的**核心主张**：

| 测试 | 守什么 |
| --- | --- |
| `test_retry_budget_suppresses_after_limit` | 第 2 次同类失败时 retryable **被代码翻成 False** —— 它红了就说明"能否重试"又回到模型手里 |
| `test_retry_budget_is_per_tool_and_type` | 按 (tool, error_type) 计累计，拦得住"换措辞重试同一个坏工具" |
| `test_retry_budget_never_touches_non_retryable_or_success` | 不可重试的错误不计数、成功结果原样返回（无副作用）|

## 七、明天要接的（W11 周二：重试与降级）

今天只做了"分类 + 让代码能判定"，还没做"代码真的按它执行完一整条链路"：

1. `apply_retry_budget` 已接进循环（`react_loop`）与图（`agent_graph`），
   预算本身有测试，但**端到端**没测：还没有一条"模型连续两次超时，
   第二次收到的 observation 明确说不能重试"的完整循环测试；
2. 降级链还没建：检索失败该退到什么？报告写不成该给用户什么？
3. `retry_after` 目前只是**写进 observation 让模型自己等**，
   循环侧没有真正的 `await sleep`（同步阻塞会让 20 任务的总耗时造假）。
   要不要在循环层做退避，取决于明天的预算实验 —— 但无论如何，
   **分类与预算这两层今天是闭合的**，退避做不做都不影响它们的正确性。
