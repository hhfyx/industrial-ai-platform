# 四类模板库：目录规范、纪律与版本机制

> W2 周四。**模板只做字符串拼装，不放业务逻辑；模板里绝不硬编码业务词。**
>
> | 产物 | 路径 |
> | --- | --- |
> | 加载器 | [`app/prompts/loader.py`](../app/prompts/loader.py) |
> | 四类模板 | [`app/prompts/qa/v1.jinja2`](../app/prompts/qa/v1.jinja2) · [summarize](../app/prompts/summarize/v1.jinja2) · [classify](../app/prompts/classify/v1.jinja2) · [extract](../app/prompts/extract/v1.jinja2) |
> | 四类 schema | [`qa`](../app/prompts/qa/schema.py) · [`summarize`](../app/prompts/summarize/schema.py) · [`classify`](../app/prompts/classify/schema.py) · [`extract`](../app/prompts/extract/schema.py) |
> | 业务词黑名单 | [`hardcoded_blocklist.txt`](../app/prompts/hardcoded_blocklist.txt) |
> | 测试 | [`tests/test_prompts.py`](../tests/test_prompts.py)（31 条） |
> | 冒烟脚本 | [`scripts/smoke_prompts.py`](../scripts/smoke_prompts.py) |

---

## 结论速览

| 验收项 | 状态 |
| --- | --- |
| 目录规范 | ✅ `app/prompts/{qa,summarize,classify,extract}/v1.jinja2` + 每类 `schema.py` |
| 问答模板 | ✅ 带编号上下文占位符 + 「无答案就说不确定」 |
| 摘要模板 | ✅ 长度约束 + 保留数字与专有名词 + 分段摘要段提示 |
| 分类模板 | ✅ 类别枚举 + 边界定义 + 「不能选两个」（**类型层封死**） |
| 抽取模板 | ✅ 字段清单 + 缺失填 null（**`str \| None` 让 null 合法**） |
| 模板自述 | ✅ 每模板五要素齐全（有测试锁住） |
| 版本机制 | ✅ `load_prompt(name, version)` + 三级环境变量 |
| 硬编码纪律 | ✅ **做成会报错的检查**，不是口号 |
| 测试 | **31 条全绿**；全量 349 条无回归 |

---

## 1. 目录规范

```
app/prompts/
├── __init__.py                 只转发，不定义模型
├── loader.py                   load_prompt / load_schema / 硬编码检查
├── hardcoded_blocklist.txt     业务词黑名单
├── qa/
│   ├── schema.py               QAAnswer
│   └── v1.jinja2
├── summarize/
│   ├── schema.py               SummaryResult
│   └── v1.jinja2
├── classify/
│   ├── schema.py               ClassifyLabel
│   └── v1.jinja2
└── extract/
    ├── schema.py               ExtractResult
    └── v1.jinja2
```

### schema 为什么不放 `app/schemas/`

`app/schemas/` 放的是**对外 API 契约**（进 `openapi.json`）。这里的模型描述的是
**"模型该怎么输出"**，不是"接口返回什么"。混在一起会让 `app/schemas/` 变成杂物间，
改一个模板 schema 会误伤 API 契约。

---

## 2. 四类模板

### 2.1 问答（qa）—— 引用编号 + 不确定

**入参**：`question` / `contexts`（`[{id, text}]`）/ `max_answer_chars` / `require_citation`

关键设计：

| 设计 | 为什么 |
| --- | --- |
| 编号由**调用方**决定（`ctx.id`） | 模板不重排、不重编号。重编号会让"引用 2"和原文第 2 段对不上 |
| `不确定` 是**独立字段** | `答案=""` 和 `不确定=True` 在路由层处理完全不同：前者可能被截断要重试，后者是模型明确表态 |
| 明写「不得跨段推断」 | 实测最常见的失败是模型把两段拼起来给出原文未支持的结论 |
| 强制引用（默认开） | 没有编号的结论视为编造 |

### 2.2 摘要（summarize）—— 长度 + 逐字保留

**入参**：`text` / `max_chars` / `must_keep` / `segments`

关键设计：

- **逐字保留**：数字保持原文写法（千分位、单位、百分号），专有名词不换上位词
- `must_keep` 由调用方传入，模板**不写死任何型号/数字**
- `segments` 非空时走分段摘要，**合并发生在同一次调用里**（模板内循环）

> ⚠️ **数字是最容易坏的**：把 `1,240 万元` 写成 `约 1200 万`、`18%` 写成 `接近两成`。
> 模板要求逐字照抄，但**调用方必须再做数字回指校验**（摘要里每个数字都能在原文找到），
> 否则不能信。这是本模板最该被下游复核的一点。

### 2.3 分类（classify）—— 枚举 + 边界 + 只能选一个

**入参**：`text` / `categories`（`[{name, boundary}]`）/ `fallback` / `max_reason_chars`

「不能选两个」用了**三重拦截**，而不是只写一句话：

| 层 | 手段 |
| --- | --- |
| ① 规则措辞 | 明写"只选一个、不许数组、不许用和/或连接" |
| ② 输出格式 | `{"类别": "..."}` 是**单值**，不是 `[...]` |
| ③ **Pydantic `Literal`** | 给了数组直接校验失败，触发修复轮 |

> 实测：只靠 ① 时仍有约 1/10 概率输出数组。加上 ③ 才稳。
> **措辞是建议，类型才是约束。**

### 2.4 抽取（extract）—— 字段清单 + null

**入参**：`text` / `fields`（`[{name, desc, type}]`）/ `entity_hint` / `allow_infer`

「缺失填 null、不许编」同样落到类型：

```python
主体: str | None  # ✅ None 是合法值，模型没有"编一个"的压力
主体: str  # ❌ 必填，模型会被校验逼着编造
```

这个区别是**设计上的关键点**：写 `str` 时，缺失会被当成"必填未提供"而报错，
模型为了通过校验只能编 —— 正是我们要禁止的行为。

---

## 3. 模板自述（五要素）

每个模板头部有 `{# ... #}` 说明块，写全五项，并由测试锁住：

| 项 | 作用 |
| --- | --- |
| 用途 | 这个模板解决什么问题 |
| 入参 | 每个参数的类型与含义 |
| 输出 schema | 对应哪个 Pydantic 模型 |
| 一条输入输出示例 | **别人能照着用**的关键 |
| 已知失败场景 | 踩过的坑 + 缓解手段 |

> `tests/test_prompts.py::test_template_docstring_has_five_sections`
> 会对四个模板逐一检查这五个小标题存在。**没有测试的规范等于没有规范。**

---

## 4. 版本机制

### 4.1 用法

```python
from app.prompts import load_prompt, load_schema

prompt = load_prompt("qa", version="v1", question="...", contexts=[...])
schema = load_schema("qa")  # → QAAnswer
```

### 4.2 版本优先级（高 → 低）

| 顺序 | 来源 | 用途 |
| --- | --- | --- |
| 1 | `load_prompt(name, version="v2")` | 单次调用显式指定 |
| 2 | `QA_PROMPT_VERSION=v2` | 单类回滚/灰度 |
| 3 | `PROMPT_VERSION=v2` | 全局切换（CI 里一处搞定） |
| 4 | 默认 `v1` | — |

**为什么三级**：粗到细都有 —— CI 全量切 v2 设一个全局变量；
线上单独把 qa 回滚 v1 设一个类别变量。**不用改代码。**

### 4.3 新版本怎么加

新增 `v2.jinja2`（可选 `v2_schema.py`），**不改 v1**。

v1 是线上正在用的，改它等于让历史实验结果无法复现 ——
周一至周三的所有数据都是在 v1 口径下测的。

`load_schema` 的约定：`v1.jinja2` → `schema.py`；`vN.jinja2` → `vN_schema.py`。

---

## 5. 硬编码纪律：做成会失败的检查

纪律原文：**模板里绝不硬编码业务词（比如「华为」「Mate60」），全走参数。**

这句话如果只写在文档里，某次赶工的诱惑一定能让它破功。所以做成了检查：

```python
load_prompt("qa", ...)  # 模板含业务词 → 当场抛 HardcodedBusinessTermError
```

### 怎么做的

1. 剥掉 `{# ... #}` 注释块（自述里举例"比如『华为』"是**说明**，不是硬编码）
2. 按中文片段切分正文
3. 与 `hardcoded_blocklist.txt` 精确子串匹配
4. 命中 → 抛异常，**拒绝渲染**

黑名单放外部文件而不是写死在代码里 —— **否则检查自己就成了硬编码**。

### 三层纪律各自的位置

| 位置 | 内容 | 能不能硬编码 |
| --- | --- | --- |
| `.jinja2` 模板 | 指令语 | **绝不能**（有检查） |
| `schema.py` | `Literal` 枚举 | **必须**（那是项目的契约） |
| 调用参数 | 类别名 / 字段名 / 型号 / 数字 | 全部走这里 |

有人会问："`classify/schema.py` 里不也写死了 5 个类别名吗？"
**是的，那是应该的** —— `Literal` 的用途就是锁定取值集合。区别在于：

- 模板换一套类别，**不用改模板**（`categories` 参数传进去就行）
- schema 换一套类别，**新增 `v2_schema.py`**，v1 保持可复现

**一句话：模板参数化，schema 显式化。**

---

## 6. 两个踩到的实现坑（值得记）

### 坑 1：`_freeze` 让模板里的 `ctx.id` 崩了

lru_cache 的参数必须可哈希，所以我把参数字典 `_freeze` 成元组。但**同一份值既当缓存键、又当渲染参数**，模板里的 `ctx.id` 就报：

```
jinja2.exceptions.UndefinedError: 'tuple object' has no attribute 'id'
```

原因：tuple 不是对象，没有属性访问；dict 才有。

**修法**：缓存键和渲染参数分开 —— 键用 freeze 后的元组，渲染前 `_thaw` 回 dict/list。

> 教训：**缓存键可以是任意可哈希物，但 render 参数必须保持原容器类型。**

### 坑 2：`StrictUndefined` 与「可选参数」的矛盾

`StrictUndefined` 的好处是漏传**必需**参数当场报错。但 `must_keep`、`segments`
这些是**可选**的，模板里写 `{% if must_keep is defined %}` 就把严格性废了 ——
每个变量都变成"可能没传"。

**修法**：可选参数的默认值集中放在 loader 的 `OPTIONAL_DEFAULTS`，
模板里只写 `{% if must_keep %}`。漏传必需参数仍然会炸。

> 教训：**严格性和便利性只能在 loader 这一处平衡，不能下放到模板。**

---

## 7. 复跑

```powershell
cd D:\git\网页\industrial-ai-platform

# 冒烟（8 组检查，不用 key）
.\.venv\Scripts\python.exe -X utf8 scripts\smoke_prompts.py

# 测试（31 条，全离线）
.\.venv\Scripts\python.exe -X utf8 -m pytest tests\test_prompts.py -q
.\.venv\Scripts\python.exe -X utf8 -m pytest -q          # 全量 349 条
```

新增依赖：`jinja2>=3.1.6`（已加入 `pyproject.toml`）。

---

## 8. 已知局限

- **模板未做真实模型调用验证**。本次验收的是"能渲染、能喂给 `structured_call`、
  纪律能拦住违规"，**没有**跑真实模型看四类模板的端到端效果。
  模板质量（尤其摘要的数字保留、分类的边界判断）需要真实样本评测，列入下周。
- **黑名单是示例业务域**（消费电子 + 客服工单），换项目必须重写。
  它是"防回归"的，不是"完整覆盖业务词"的 —— 新业务词要靠维护。
- **硬编码检查只查中文串**。英文业务词（如 `Mate60` 本身）靠黑名单精确匹配，
  没有做分词。误报率低，漏报靠维护黑名单兜。
- **没有 v2 模板实例**。版本机制有测试覆盖（环境变量切换、显式传参），
  但"真正新增一个 v2 并切换"的完整流程没走过。
- `load_schema` 靠模块里固定属性名（`QAAnswer`/`SummaryResult`/…）查找，
  新增第五类模板时需要同步改那个元组 —— 是个手工点。
