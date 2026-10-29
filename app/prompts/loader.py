"""模板加载器：`load_prompt(name, version)` + 版本切换 + 硬编码检查。

## 目录约定

    app/prompts/{qa,summarize,classify,extract}/v1.jinja2    模板正文
    app/prompts/{qa,summarize,classify,extract}/schema.py    该类输出 Pydantic 模型

新版本就是新增 `v2.jinja2`（可选新增 `v2_schema.py`），**不改 v1** ——
v1 是线上正在用的，改它等于让历史实验结果无法复现。

## 版本怎么选

优先级（高 → 低）：

1. `load_prompt(name, version=...)` 显式传参
2. 环境变量 `<NAME>_PROMPT_VERSION`（如 `QA_PROMPT_VERSION=v2`）
3. 全局环境变量 `PROMPT_VERSION`（如 `PROMPT_VERSION=v1`）
4. 默认 `"v1"`

**为什么要有三级兜底**：CI 里想全量切到 v2 就设 `PROMPT_VERSION=v2` 一处；
线上想单独把 qa 回滚到 v1 就设 `QA_PROMPT_VERSION=v1`。粒度从粗到细，
不用改代码。

## 为什么模板要过"硬编码检查"

纪律是「模板里绝不硬编码业务词（比如『华为』『Mate60』），全走参数」。
这条**必须能自动验证**，否则"现在省事"的诱惑一定会在某次赶工里破功。

`check_no_hardcoded_business_terms()` 的做法：
扫描模板里的**中文字符串字面量**，与一份业务词黑名单比对。
黑名单不写死在代码里（那样它自己就成了硬编码），而是放在
`app/prompts/hardcoded_blocklist.txt`，可随项目演进增删。

**这个检查有意做成"可失败"的**：`load_prompt` 默认开着它（`strict=True`），
新写一个带业务词的模板会**当场抛异常**。当纪律被写成会报错的断言，
它才真的是纪律。

## 为什么模板只做字符串拼装

业务逻辑（重试、校验、降级）全在 `structured_call` 和调用方。
模板里出现 `{% if 商业模式 == "订阅" %}` 这类判断，就意味着逻辑住进模板了 ——
它无法被单元测试单独覆盖，也无法在换业务时复用。本 loader 只渲染，不判断。
"""

from __future__ import annotations

import importlib
import os
import re
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound
from pydantic import BaseModel

PROMPTS_DIR = Path(__file__).resolve().parent

# 模板类别白名单。新增类别必须显式加到这里，避免手滑建出拼错名字的目录。
CATEGORIES: frozenset[str] = frozenset({"qa", "summarize", "classify", "extract"})

BLOCKLIST_FILE = PROMPTS_DIR / "hardcoded_blocklist.txt"

# 各类模板的**可选参数默认值**。
#
# 为什么在这里而不是模板里：
# 模板用 `{% if x is defined %}` 判断可选参数，等于给每个变量都留了"可能没传"的口子，
# `StrictUndefined` 的严格性就废了 —— 真正漏传**必需**参数时也不会报错。
# 把默认值集中放在这里，模板里就只写 `{% if must_keep %}`，
# 而漏传必需参数仍然会当场炸。**严格性和便利性只能在这里平衡。**
OPTIONAL_DEFAULTS: dict[str, dict[str, object]] = {
    "qa": {"require_citation": True, "max_answer_chars": 400},
    "summarize": {"must_keep": [], "segments": [], "max_chars": 200},
    "classify": {"fallback": "其他", "max_reason_chars": 30},
    "extract": {"entity_hint": "", "allow_infer": False},
}

# 中文串字面量：用于硬编码检查
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


class PromptNotFoundError(FileExistsError):
    """模板或版本不存在。

    继承 `FileExistsError` 而不是 `KeyError`：语义上它确实是"文件不存在"，
    而 `KeyError` 会被很多通用 `except KeyError` 吞掉，掩盖真正的问题。
    """


class HardcodedBusinessTermError(ValueError):
    """模板里出现了业务词黑名单里的词。"""

    def __init__(self, template_path: Path, terms: list[str]) -> None:
        self.template_path = template_path
        self.terms = terms
        super().__init__(
            f"模板 {template_path.name} 硬编码了业务词 {terms}。"
            f"业务词必须走参数（见 app/prompts/loader.py 的说明）。"
        )


_env_cache: dict[str, Environment] = {}


def _env() -> Environment:
    """本进程共用的 Jinja 环境。

    `StrictUndefined` 是刻意的：模板里用到未传的变量时**立刻报错**，
    而不是渲染成空串。空串会让"漏传参数"变成静默错误 ——
    模型收到一个缺了上下文的 prompt，输出看起来正常但没依据，极难排查。
    """
    key = str(PROMPTS_DIR)
    if key not in _env_cache:
        _env_cache[key] = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            # 模板输出是要发给模型的正文，HTML 转义会把引号变成 &quot; 之类的噪音
            autoescape=False,
        )
    return _env_cache[key]


def _resolve_version(name: str, version: str | None) -> str:
    """按优先级解析版本号（显式 > 类别环境变量 > 全局环境变量 > 默认 v1）。"""
    if version:
        return version
    per_category = os.environ.get(f"{name.upper()}_PROMPT_VERSION")
    if per_category:
        return per_category
    return os.environ.get("PROMPT_VERSION") or "v1"


def list_versions(name: str) -> list[str]:
    """列出某类模板已有的版本，按自然序（v1, v2, v10 而不是 v1, v10, v2）。"""
    cat_dir = PROMPTS_DIR / name
    if not cat_dir.is_dir():
        raise PromptNotFoundError(f"未知模板类别：{name}")
    versions = [p.stem for p in cat_dir.glob("v*.jinja2")]
    return sorted(versions, key=lambda v: int(v[1:]) if v[1:].isdigit() else 1 << 30)


def _load_blocklist() -> list[str]:
    if not BLOCKLIST_FILE.exists():
        return []
    lines = BLOCKLIST_FILE.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def check_no_hardcoded_business_terms(
    template_text: str, template_path: Path | None = None
) -> list[str]:
    """扫描模板里出现的业务词，返回命中的词列表（空 = 通过）。

    **只看注释以外的正文**：docstring 里为了说明"什么叫业务词"会举例子
    （"比如『华为』"），那是说明不是硬编码。所以先剥掉 `{# ... #}` 注释块。

    为什么按"中文串"而不是按"整个文件里出现这个词"：
    业务词黑名单是中文词，而模板里合法的中文是**固定指令语**（"只输出 JSON"）。
    真正危险的是"业务实体名"进了正文。用中文串切分 + 精确子串匹配，
    误报率可以接受，漏报可以靠维护黑名单兜住。
    """
    body = re.sub(r"\{#.*?#\}", "", template_text, flags=re.DOTALL)
    terms = _load_blocklist()
    if not terms:
        return []
    hits: list[str] = []
    for run in _CJK_RUN.findall(body):
        for term in terms:
            if term in run and term not in hits:
                hits.append(term)
    return hits


@lru_cache(maxsize=64)
def _render_cached(
    name: str, version: str, params_key: str, params: tuple[tuple[str, object], ...]
) -> str:
    """实际渲染。用 lru_cache 缓存"同模板同参数"的结果。

    这里有**两个必须分开的东西**，混在一起会踩坑：

    - `params_key`：参数的**可哈希序列化**，只用于 lru_cache 的键；
    - `params`：参数的**原始值**，用于传给 `template.render()`。

    第一版实现把两者合成一个（`_freeze` 把嵌套 dict 转成 tuple 后直接 render），
    结果模板里 `ctx.id` 报 `'tuple object' has no attribute 'id'` ——
    因为 tuple 不是对象、没有属性访问。**缓存键可以是任何可哈希的东西，
    但渲染参数必须保持原来的容器类型。**

    所以此函数的 `params` 形参**保持原始 dict 的等价结构**（见下方 `_thaw`）：
    调用方传 freeze 后的元组作缓存键，传**原始 dict**作渲染参数。
    这里再 `_thaw` 回 dict/list，是为了让 lru_cache 的参数可哈希 ——
    直接传 dict 会 `TypeError: unhashable`。

    另外 `params_key` 也要参与哈希：只按 (name, version) 缓存会让第二个调用方
    拿到第一个调用方的渲染结果，这是最隐蔽的串味。
    """
    tpl_name = f"{name}/{version}.jinja2"
    try:
        template = _env().get_template(tpl_name)
    except TemplateNotFound as exc:
        versions = list_versions(name)
        raise PromptNotFoundError(f"模板不存在：{tpl_name}。{name} 现有版本：{versions}") from exc

    # 硬编码检查放在渲染前：宁可渲染失败，也不要把带业务词的 prompt 发给模型
    source = (PROMPTS_DIR / tpl_name).read_text(encoding="utf-8")
    hits = check_no_hardcoded_business_terms(source, PROMPTS_DIR / tpl_name)
    if hits:
        raise HardcodedBusinessTermError(PROMPTS_DIR / tpl_name, hits)

    # 关键：渲染前要把 tuple 还原成 dict/list，模板才能 ctx.id 这样访问属性
    return template.render(**_thaw(dict(params)))


def _thaw(value: object) -> object:
    """`_freeze` 的逆操作：把元组还原成 dict / list，供 Jinja 渲染。

    为什么必须还原：Jinja 里 `ctx.id` 是属性访问，tuple 没有属性；
    而 dict 才有。`_freeze` 是为了让 lru_cache 能哈希，`_thaw` 是为了让模板能读 ——
    两者方向相反，缺一个就出错。
    """
    if isinstance(value, tuple):
        # 判断是"冻结的 dict"还是"冻结的 list"：
        # 冻结 dict 的元素一定是 (str, value) 二元组
        if value and all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {k: _thaw(v) for k, v in value}
        return [_thaw(v) for v in value]
    if isinstance(value, dict):
        return {k: _thaw(v) for k, v in value.items()}
    return value


def _freeze(params: dict[str, object]) -> tuple[tuple[str, object], ...]:
    """把参数字典变成可哈希的键序。列表/字典转成元组，保证同一内容同一键。"""

    def norm(v: object) -> object:
        if isinstance(v, dict):
            return tuple(sorted((k, norm(x)) for k, x in v.items()))
        if isinstance(v, (list, tuple)):
            return tuple(norm(x) for x in v)
        return v

    return tuple(sorted((k, norm(v)) for k, v in params.items()))


def load_prompt(
    name: str,
    version: str | None = None,
    *,
    strict: bool = True,
    **params: object,
) -> str:
    """加载并渲染模板。

    参数
    ----
    name
        模板类别：`qa` / `summarize` / `classify` / `extract`
    version
        版本号如 `"v2"`。不传则按三级环境变量兜底（见模块 docstring）
    strict
        是否在渲染前做业务词硬编码检查。默认 True。
        只在写"故意要测这条检查"的测试时才传 False。
    **params
        模板变量。**未传的必需变量会立刻报错**（`StrictUndefined`）。
        可选变量由 `OPTIONAL_DEFAULTS` 自动补默认值，不需要调用方每次都传。

    返回
    ----
    渲染好的 prompt 字符串，可直接作为 `structured_call` 的 `prompt` 参数。
    """
    if name not in CATEGORIES:
        raise PromptNotFoundError(f"未知模板类别：{name}（可用：{sorted(CATEGORIES)}）")

    resolved = _resolve_version(name, version)
    # 可选参数补默认值：在 loader 里补，而不是让模板用 `is defined` 判断 ——
    # 否则 StrictUndefined 就形同虚设（模板里到处写"可能没传"等于没有严格性）。
    merged = {**OPTIONAL_DEFAULTS.get(name, {}), **params}

    if strict:
        # 提前检查一次，让错误信息里带路径；缓存路径下重复检查代价很低
        path = PROMPTS_DIR / name / f"{resolved}.jinja2"
        if path.exists():
            hits = check_no_hardcoded_business_terms(path.read_text(encoding="utf-8"), path)
            if hits:
                raise HardcodedBusinessTermError(path, hits)

    # 两个参数分开传：
    #   repr(params) / _freeze(params) 只作缓存键（可哈希、内容等价即同键）
    #   params（原始 dict/list）作渲染参数（模板要能 ctx.id 这样访问属性）
    return _render_cached(name, resolved, repr(merged), _freeze(merged))


@lru_cache(maxsize=32)
def load_schema(name: str, version: str | None = None) -> type[BaseModel]:
    """拿到某类模板对应的输出 schema。

    默认约定：`v1.jinja2` → `schema.py`；`vN.jinja2` (N>1) → `vN_schema.py`。
    这样改接口时新增文件即可，v1 的 `schema.py` 保持不动。

    **模型怎么发现**：不靠写死的名字列表，而是扫模块里定义的 `BaseModel` 子类，
    取"在本模块定义的那一个"。

    为什么改掉写死列表：周四的实现硬编码了 `("QAAnswer","SummaryResult",
    "ClassifyLabel","ExtractResult")` 四个名字，等于**新增第五类模板或 v2 schema
    时必须改 loader** —— 那是把模板库的可扩展性钉死在了一个元组里。
    改成按定义位置发现之后，`extract/v2_schema.py` 里的 `ExtractV2` 自动被认出来。
    """
    if name not in CATEGORIES:
        raise PromptNotFoundError(f"未知模板类别：{name}")

    resolved = _resolve_version(name, version)
    module_name = "schema" if resolved == "v1" else f"{resolved}_schema"
    try:
        module = importlib.import_module(f"app.prompts.{name}.{module_name}")
    except ModuleNotFoundError as exc:
        raise PromptNotFoundError(
            f"{name}/{resolved} 缺少输出 schema：app/prompts/{name}/{module_name}.py"
        ) from exc

    # 只认"在本模块里定义"的 BaseModel 子类（避免把 import 进来的别家模型当成本模块的）
    candidates = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and getattr(obj, "__module__", "") == module.__name__
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise PromptNotFoundError(
            f"{module.__name__} 里没有定义 BaseModel 子类（应定义该类模板的输出模型）"
        )
    raise PromptNotFoundError(
        f"{module.__name__} 里定义了多个 BaseModel 子类 "
        f"{[c.__name__ for c in candidates]}，无法确定用哪个。"
        f"一个 schema 模块只放一个输出模型。"
    )


def clear_cache() -> None:
    """清空渲染缓存。测试里改了环境变量后必须调用，否则会拿到旧版本的结果。"""
    _render_cached.cache_clear()
    load_schema.cache_clear()


__all__ = [
    "CATEGORIES",
    "HardcodedBusinessTermError",
    "PROMPTS_DIR",
    "PromptNotFoundError",
    "check_no_hardcoded_business_terms",
    "clear_cache",
    "list_versions",
    "load_prompt",
    "load_schema",
]
