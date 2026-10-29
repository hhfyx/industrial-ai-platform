# A/B 脚手架与 40 条输入池

> W2 周五。**这 40 条是 prompt 实验集；W7 的 200 条是 RAG 评测集。
> 两者不能混用、不能复用同一批问题。**
>
> | 产物 | 路径 |
> | --- | --- |
> | 输入池（4 类 × 10 条） | [`eval/prompt-sets/`](../eval/prompt-sets/) |
> | 池校验器 | [`scripts/validate_prompt_sets.py`](../scripts/validate_prompt_sets.py) |
> | A/B 脚手架 | [`scripts/ab_test.py`](../scripts/ab_test.py) |
> | 结果目录 | `scripts/ab_results/`（CSV + summary json） |

---

## 结论速览

| 验收项 | 状态 |
| --- | --- |
| 40 条输入池 | ✅ qa / classify / extract / summarize 各 10 条 |
| 每条带判定标准 | ✅ 有测试锁定（`validate_prompt_sets.py`） |
| 采集纪律 | ✅ 全部从开发语料抽取、可追溯 source、**零数据泄漏** |
| 类别真实不均衡 | ✅ 故障 3 / 咨询 3 / 费用 2 / 投诉 2（未手工凑平衡） |
| 脚手架 | ✅ `--set qa --a v1 --b v2`，输出 CSV + 控制台摘要 |
| 固定温度与 seed | ✅ 写死在代码里，**故意不做成参数** |
| 判定器 | ✅ 分类比 label、抽取逐字段、问答/摘要规则判要点 |
| 缓存 + 干跑 | ✅ 干跑 2 条通过；缓存命延迟 1ms |

---

## 1. 采集纪律：这 40 条从哪来

### 1.1 来源

**全部从本仓库的开发语料抽取**，每条带 `source` 字段（精确到文件，有的到行号）：

| 语料 | 条数 | 用在哪 |
| --- | ---: | --- |
| `docs/sse.md` | 12 | qa / classify / extract / summarize |
| `docs/cost.md` | 9 | qa / classify / extract / summarize |
| `docs/SETUP.md` | 6 | qa / classify / extract / summarize |
| `README.md` | 7 | qa / classify / extract / summarize |
| `docs/structured-output.md` | 1 | summarize |
| `app/rag/summarize.py` | 2 | extract / summarize |
| `docs/api-spec.md` | 2 | qa / summarize |

### 1.2 为什么"从开发语料抽"而不是自己编

**自己编的输入会不知不觉地配合 prompt。** 你会本能地写出"模型容易答对"的问题 ——
那不是测试，是自证。开发语料里的内容是**先于 prompt 存在的**，
它不知道我在做什么实验，所以是干净的。

### 1.3 数据泄漏检查：机器可验，不靠自觉

`validate_prompt_sets.py` 的 `check_leakage()` 做三件事：

1. 扫描 `scripts/*_result.json` 与 `system_prompt_ab.json`，收集**我调 prompt 时
   真正发出去过的文本**（当前基线 **39 条**）
2. 检查输入池里的 `text` / `question` 是否与之完全相同
3. 再查**前缀 20 字重叠**（防同一段话被截断复用）

```
已用过的实验输入：39 条（泄漏检查基线）
[qa]       判定标准齐全: ✅   无数据泄漏: ✅
[classify] 判定标准齐全: ✅   无数据泄漏: ✅
[extract]  判定标准齐全: ✅   无数据泄漏: ✅
[summarize]判定标准齐全: ✅   无数据泄漏: ✅
```

> **为什么要写成检查**：任务原话是「绝不从你要汇报的结果里挑」。
> 这条纪律如果不自动化，它会在我第三次跑实验、急着要数据的时候失效。

### 1.4 类别分布必须真实不均衡

分类 10 条的实际分布：

| 类别 | 条数 |
| --- | ---: |
| 故障报修 | 3 |
| 功能咨询 | 3 |
| 费用争议 | 2 |
| 投诉建议 | 2 |

**这是语料自然分布的结果**，不是手工凑的。校验器专门有一条断言：
若 `len(dist) == len(cls)`（每个类别各 1 条 = 手工凑平衡），直接判失败。

> **为什么必须不均衡**：真实业务里投诉远少于故障。如果凑成均衡，
> 模型在"其他"类上的偏差会被稀释，测出来的准确率虚高。
> **不均衡的测试集才能暴露偏向多数类的失败。**

### 1.5 与 W7 的边界（重要）

| | 本池（prompt 实验集） | W7（RAG 评测集） |
| --- | --- | --- |
| 规模 | 40 条 | 200 条 |
| 用途 | 比 prompt 版本 | 比 RAG 链路 |
| **能否复用** | **不能** | **不能** |

**原因**：我调 prompt 时会反复看到这 40 条。如果 W7 用同一批问题，
它的评测结果会因为"这些题我在调 prompt 时见过并针对性优化过"而**虚高** ——
那就不是在测 RAG，是在测记忆。

这条边界由 `check_leakage()` 兜住：未来加 W7 数据时，
如果往本池或从本池复制问题，校验器会报。

---

## 2. 输入池格式

每类一个 `.jsonl`，每行一个 JSON。**每条必须带判定标准**：

### qa.jsonl
```json
{"id":"qa-01","source":"docs/sse.md#L10-L31","question":"...",
 "contexts":[{"id":1,"text":"..."}],
 "gold_points":["单向足够","走 HTTP 对代理友好",...],
 "min_points":4,"expect_uncertain":false,"valid_ids":[1,2,3,4,5]}
```

### classify.jsonl
```json
{"id":"cl-01","source":"docs/sse.md","text":"...",
 "gold_label":"故障报修","gold_reason":"..."}
```

### extract.jsonl
```json
{"id":"ex-01","source":"docs/SETUP.md#L35-L45","text":"...",
 "fields":[{"name":"主体","desc":"软件或工具名称","type":"string"}],
 "gold":{"主体":"uv","版本":"0.13.1","路径":"%USERPROFILE%\\..."}}
```

### summarize.jsonl
```json
{"id":"sm-01","source":"docs/sse.md#L10-L31","text":"...",
 "must_include":["单向","text/event-stream","Upgrade",...],
 "max_chars":120}
```

---

## 3. A/B 脚手架

### 3.1 用法

```powershell
# 干跑（只跑 2 条，确认管线通）
python -X utf8 scripts\ab_test.py --set classify --a v1 --b v1 --dry-run

# 正式对比
python -X utf8 scripts\ab_test.py --set extract --a v1 --b v2
```

输出：
- `scripts/ab_results/<set>_<a>_vs_<b>_full.csv` —— 逐条记录（可直接透视）
- `scripts/ab_results/<set>_..._.summary.json` —— 汇总

### 3.2 为什么 temperature 和 seed **不做成参数**

```python
FIXED_TEMPERATURE: float = 0.0
FIXED_SEED: int = 42
```

这两种值写死在代码里，不是懒。**允许调它，等于允许有人偷偷改来"调好看"。**

而且本仓库已有实测（`learn/llm/01_params.md`，位于上级目录 `D:\git\网页\learn\`）：

- `temperature=0` **不等于**「每次都一样」（5 次出 3 种输出）
- `seed` 在本网关**实测不可复现**

所以这两项只能"尽可能压噪声"，**真噪声控制靠样本量**。

### 3.3 结论阈值写死在代码里

```python
MIN_SIGNAL_DELTA = 0.20
```

10 条样本时，**1 条翻转 = 10pp**。所以至少要差 2 条（20pp）才认为有信号。
这个判断写在 `summarize()` 里，不靠人眼看数字猜。

输出会明确说"无显著差异"还是"B 更好"：

```
Δ平均得分 = +0.6186  （阈值 0.2）
结论：B 更好（差距达到信号阈值）
```

### 3.4 记录什么

每条都记：`score` / `judge_detail` / `prompt_tokens` / `completion_tokens` /
`latency_ms` / `cost_cny` / `raw_output`。

**成本必须记**：prompt 版本对比最容易忽略的就是代价。v2 准确率高但 token 翻倍，
在有 QPS 的场景下不是"更好"，是"更贵"。

### 3.5 用量怎么拿到（不动定版签名）

`structured_call` 的签名是周三定版的（`-> BaseModel`），**改了后面四周全部返工**。
但它拿不到 `usage`。所以加了一个**旁路钩子**：

```python
set_usage_sink(callable)  # 默认 None，生产路径零开销
```

`_request()` 在每次真实调用后把 `{prompt_tokens, completion_tokens}` 递给 sink。
**签名一字未动。**

缓存命中时没有新调用，usage 为 0 —— 这时退回从缓存文件反查当初那次的用量。

> 观测失败绝不影响主路径：sink 内部异常被吞掉并只记 `debug` 日志。

---

## 4. 判定器

| 类别 | 判定方式 | 能否全自动 |
| --- | --- | --- |
| **classify** | 比 `gold_label`，命中 1 / 不中 0 | ✅ 全自动 |
| **extract** | **逐字段比**，算 field-level 准确率 | ✅ 全自动 |
| **qa** | 规则判 `gold_points` 命中 | ⚠️ 部分（`needs_human=true` 标出） |
| **summarize** | 规则判 `must_include` **逐字**命中 + 长度 | ⚠️ 部分（同上） |

### 4.1 为什么抽取必须逐字段

一条 3 字段的样本"对了 2 个"和"全错"，在**整条口径**下都是 0 分。
但改进价值完全不同 —— 只看整条会让你误判"v2 没用"，其实 v2 已经修好了 60% 的字段。

### 4.2 问答/摘要为什么只能部分自动

- **qa** 的 `gold_points` 是短语，答案可能换个说法。判定器用**宽松匹配**
  （要点切成 ≥2 字片段，命中过半算覆盖）。宁可变宽、人工兜底，
  也不要因措辞不同判成漏答 —— **后者会让 A/B 的差异全是噪声。**
- **summarize** 相反，`must_include` 是**硬要求**（数字、专有名词必须逐字保留），
  所以要求**逐字出现**，不宽松。

两者口径相反，是刻意的：摘要错一个数字就是错，问答换个说法不一定错。

---

## 5. 干跑与真实 A/B

### 5.1 干跑（2 条，确认管线通）

```
[classify] A 臂真实调用 2178ms / B 臂缓存命中 1ms
✅ 管线通：token 拿到了、缓存命中不重花钱、CSV 落盘
```

### 5.2 第一次真实 A/B：extract v1 vs v2（10 条全量）

| 指标 | A (v1) | B (v2) |
| --- | ---: | ---: |
| 平均得分(0-1) | **0.1481** | **0.7667** |
| 整条全对率 | 0.0 | **0.6** |
| prompt token 合计 | 4306 | 4855 |
| completion token 合计 | **14533** | **7119** |
| 成本(元) | 0.124876 | **0.066662** |
| 平均延迟(ms) | 10121 | **4701** |

**Δ = +0.6186（远超阈值 0.2）→ B 更好。**

**v2 不但更准，还更便宜、更快** —— 因为 v1 的槽位对不上，模型在硬凑，
耗掉 14533 个 completion token（在修复轮里来回改）。

#### 这个结果要怎么读（重要）

**v1 输得这么惨，不是因为模型差，是契约不匹配。**

v1 的 schema 把槽位写死成 `主体` / `时间`（按工单场景设计），
而输入池来自技术文档，字段是「版本/路径/要求」—— 对不上。
模型不是不会抽，是**没有槽位可填**。

所以这次 A/B 真正验证的是：**模板的字段契约必须匹配输入形态**，
而不是"v2 模板写得好"。这条结论比那个 +61.9pp 更有用。

逐条看 v2 的失败也很有信息量：

| 条目 | v2 得分 | 问题 |
| --- | ---: | --- |
| ex-01 | 0.67 | 路径给 `.local/bin`，gold 是完整 `%USERPROFILE%\\.local\\bin\\uv.exe` —— **抽短了** |
| ex-03 | 0.00 | 问题描述被简化，少了"但 usage 正常"这半句 |
| ex-05 | 0.50 | 值带上了句末标点，与 gold 差一个"。" |

这三条都指向同一个规律：**v2 倾向于"提炼"而不是"照抄"**。
模板里已写"值保持原文原始写法"，但实测仍有压缩倾向 —— 列入待改。

---

## 6. 复跑

```powershell
cd D:\git\网页\industrial-ai-platform
$env:LLM_API_KEY = "<key>"
$env:LLM_BASE_URL = "http://127.0.0.1:8326/v1"
$env:LLM_MODEL = "deepseek-account/deepseek-flash"

.\.venv\Scripts\python.exe -X utf8 scripts\validate_prompt_sets.py   # 校验 40 条
.\.venv\Scripts\python.exe -X utf8 scripts\ab_test.py --set qa --a v1 --b v2 --dry-run
.\.venv\Scripts\python.exe -X utf8 scripts\ab_test.py --set extract --a v1 --b v2
```

---

## 7. 已知局限

- **只有 v1 和 extract 的 v2 是真模板**。qa / classify / summarize 的 v2 还没有，
  所以周五只能跑 extract 的真实 A/B。要三版对比得先写另外三个 v2。
- **单次运行**。除干跑外没做重复跑，**没测 A/B 的稳定性** ——
  temperature=0 也不完全可复现（见 `01_params.md`），
  同一个对比跑两次可能差 1 条。**恰好 1 条的差异不该当结论。**
- **qa / summarize 的判定是"部分自动"**。规则判要点命中，但"覆盖了却理解错了"
  判不出来。CSV 里标了 `needs_human=true`，**人工复核这步没做**。
- **classify 的类别体系是 v1 的 5 类**。输入池是按这个体系标的 gold；
  换类别体系要重标 gold，不能直接复用。
- **输入池 40 条里有 2 条判定可能有争议**：
  `cl-02`（问成本测算算"功能咨询"还是"其他"）实测模型判"其他"，
  我标的是"功能咨询" —— 这是我自己的标注口径问题，不是模型错。
  这类歧义条目不删除、保留在池里，因为**真实业务就有歧义**，
  但它会压低准确率，读 A/B 结果时要记得这 1 条。
- **cost_cny 用的是 `scripts/cost_model.json` 的示例单价**，不是真实账单。
  它适合做"A 比 B 贵几倍"的相对比较，不是一个能拿去做预算的绝对数。
