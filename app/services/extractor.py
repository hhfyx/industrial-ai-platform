"""规则版抽取服务（W1 实现，W2 换成 LLM）。

**这个文件最容易被误解的一点：它返回 `str | None`，不做类型转换。**
契约（api-spec.md §2.1）规定 `data` 的值恒为字符串、抽不到为 `null`，
所以「转成 float/date」是调用方的事，抽取器乱转反而会让 W2 的 LLM 版行为不一致。

**为什么是 async 且故意 sleep 100ms**：
抢在 W2 之前把「抽取是 IO」这件事钉进签名里。规则版其实同步就够，
但路由层一旦按 async 写，换 LLM 时 `await` 的位置不用动，并发模型也不用改。
`asyncio.sleep` 让单元测试能真实覆盖「耗时被计入 elapsed_ms」的分支，
不是假 delay 的 mock —— mock 掉延迟就等于没测并发路径。

**留给 W2 的扩展点**：`RuleExtractor.extract()` 的签名
`async (text: str, fields: Sequence[str]) -> ExtractionOutput` 是契约的一部分（api-spec §4.7）。
换实现时新建 `LlmExtractor`，实现同一个方法，路由里的 `get_extractor` 换个装配即可。
**正则、prompt、模型名一律不许出现在本文件的对外接口里。**

分层要求：本文件**不 import fastapi / starlette** —— 抽错了层，W2 替换就无从谈起。
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Final

# 模拟的真实抽取延迟（毫秒）。W1 用它占住 async 的位置，W2 换成真实网络往返。
SIMULATED_LATENCY_MS: Final[int] = 100

MODEL_NAME: Final[str] = "rule-v1"


# ============================================================ 规则表
#
# 每个规则 = (别名集合, 抽取正则)。别名大小写不敏感，中文原样比对。
# 正则统一用 `第 1 组` 抓值，外层标签部分不参与返回 —— 这样下游拿到的都是干净的裸值。


@dataclass(frozen=True)
class _FieldRule:
    """一条抽取规则：认哪些名字（别名）、怎么抽（正则 + 可选规范化）。"""

    canonical: str
    aliases: tuple[str, ...]
    pattern: re.Pattern[str]
    # 值需要再做一次规范化（如日期统一 YYYY-MM-DD）时填这里；None 表示原样返回
    normalizer: str | None = None


# 合同编号：HT-2026-001 / HT2026001 / A-001 之类
_CONTRACT_NO = re.compile(
    r"(?:合同编号|合同号|合同代码|contract[_ ]?no|contract[_ ]?id)"
    r"\s*[:：=]?\s*"
    r"(?P<value>[A-Za-z]{0,4}[-_-]?[A-Za-z0-9]{2,}(?:[-_/][A-Za-z0-9]{1,})+|[A-Za-z0-9]{6,32})",
    re.IGNORECASE,
)

# 日期：2026-10-20 / 2026/10/20 / 2026年10月20日 / 2026.10.20
_DATE_VALUE = re.compile(
    r"(?P<value>\d{4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}\s*日?)",
)
_DATE = re.compile(
    r"(?:签订日期|签署日期|签发日期|填发日期|日期|date|sign[_ ]?date|signing[_ ]?date)"
    r"\s*[:：=]?\s*" + _DATE_VALUE.pattern + r"(?!\d)",
    re.IGNORECASE,
)

# 金额：¥128,000.00 / 人民币 128000 元 / 128,000.00 元 / RMB 1,234.5
# 前缀里必须同时收 `人民币`：只认 RMB/CNY 会漏掉最常见的写法（测试抓到的第一个 bug）
_AMOUNT = re.compile(
    r"(?:合同金额|总金额|金额|价款|amount|total[_ ]?amount|price)"
    r"\s*[:：=]?\s*"
    r"[¥￥]?\s*"
    r"(?:人民币|RMB|CNY)?"
    r"\s*"
    r"(?P<value>\d[\d,]*(?:\.\d{1,2})?)"
    r"\s*(?:元|万元)?",
    re.IGNORECASE,
)

# 甲方：标签后到句读符前的文本，允许中英文名/公司名。
# 值首字符**必须排除冒号等分隔符**，否则 `甲方：。` 会把冒号本身当成值（第二个 bug）
_PARTY_A = re.compile(
    r"(?:甲方|买方|发包方|委托方|party[_ ]?a|buyer|client)"
    r"\s*[:：=]?\s*"
    r"(?P<value>[^\s：:，,。;；、\n\r\t][^，,。;；、\n\r\t]*)",
    re.IGNORECASE,
)

FIELD_RULES: Final[tuple[_FieldRule, ...]] = (
    _FieldRule(
        canonical="合同编号",
        aliases=("合同编号", "合同号", "合同代码", "contract_no", "contract_id"),
        pattern=_CONTRACT_NO,
    ),
    _FieldRule(
        canonical="签订日期",
        aliases=("日期", "签订日期", "签署日期", "date", "sign_date", "signing_date"),
        pattern=_DATE,
        normalizer="date",
    ),
    _FieldRule(
        canonical="合同金额",
        aliases=("金额", "合同金额", "总金额", "价款", "amount", "total_amount", "price"),
        pattern=_AMOUNT,
    ),
    _FieldRule(
        canonical="甲方",
        aliases=("甲方", "买方", "发包方", "委托方", "party_a", "buyer", "client"),
        pattern=_PARTY_A,
    ),
)

# 别名 -> 规则，查询 O(1)；中文名直接命中，英文名先 lower 再查
_ALIAS_INDEX: Final[dict[str, _FieldRule]] = {
    alias.lower(): rule for rule in FIELD_RULES for alias in rule.aliases
}


def _normalize_date(raw: str) -> str | None:
    """把三种日期写法统一成 `YYYY-MM-DD`；非法日历日期返回 `None`。

    契约（api-spec.md §1.2）承诺日期是 `YYYY-MM-DD`，所以规范化**只能在服务层做**：
    W2 换 LLM 时模型输出什么格式都有可能，这个函数是保证契约不被实现细节污染的最后一道闸。

    为什么要用 `date(...)` 真跑一遍：`2026-13-45` 完全符合"四个数字-两个数字-两个数字"，
    但它不是真实日期。不做日历校验就会把脏值写进库里（测试抓到的第三个 bug）。
    """
    parts = re.split(r"[-/.年月日]+", raw.strip())
    try:
        year, month, day = (int(p) for p in parts[:3])
        return date(year, month, day).isoformat()
    except (ValueError, TypeError):
        return None


def extract_value(text: str, field_name: str) -> str | None:
    """抽单个字段。抽不到返回 `None`（契约要求，**不编造**）。

    匹配顺序：① 命中规则表 → 用对应正则；
    ② 未命中 → 走「标签兜底」`<字段名>：<值>`，让不在字典里的字段名也能用。
    """
    name = field_name.strip()
    rule = _ALIAS_INDEX.get(name.lower())
    if rule is not None:
        match = rule.pattern.search(text)
        if match is None:
            return None
        value = match.group("value").strip()
        if rule.normalizer == "date":
            # 形似日期但非法（如 2026-13-45）：宁可不给，也不返回脏值
            return _normalize_date(value)
        return value or None

    # 兜底：字段名不在字典里
    pattern = re.compile(
        r"(?:"
        + re.escape(name)
        + r")\s*[:：=]\s*(?P<value>[^\s，,。;；、\n\r\t][^，,。;；、\n\r\t]*)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        return None
    return match.group("value").strip() or None


def compute_confidence(fields: Sequence[str], data: dict[str, str | None]) -> float:
    """命中比例；有缺失再打 0.75 折。

    为什么不是直接 `0.75 ** 缺失数`：字段一多会指数级塌到 0，失去区分度。
    这里只做一档折扣，语义是"有缺的老师就得扣分"，足够 downstream 排序用了。
    """
    if not fields:
        return 0.0
    hits = sum(1 for name in fields if data.get(name) is not None)
    ratio = hits / len(fields)
    if hits < len(fields):
        ratio *= 0.75
    return round(ratio, 3)


@dataclass(frozen=True, slots=True)
class ExtractionOutput:
    """服务层内部结果（不是对外的 pydantic 模型）。

    services 层用自己的 dataclass、schemas 层用自己的 BaseModel：
    内层换实现时不会牵动外层的校验规则 —— 这也是 api-spec §4.7 敢承诺「签名不变」的底气。
    """

    data: dict[str, str | None]
    confidence: float
    model: str = MODEL_NAME
    elapsed_ms: int = 0
    missing: list[str] = field(default_factory=list)


class RuleExtractor:
    """规则版抽取器。

    只有 `extract()` 是对外接口，其余都是内部函数 ——
    W2 新建 `LlmExtractor` 时只要实现同名方法，路由层不需要知道底下是正则还是模型。
    """

    model_name: str = MODEL_NAME

    async def extract(self, *, text: str, fields: Sequence[str]) -> ExtractionOutput:
        """按 `fields` 顺序抽值，`data` 的键顺序与入参完全一致（契约 §2.1）。

        `data` 用 comprehension 构造而不是先建空 dict 再塞：
        插入顺序 = `fields` 的顺序，一次成型，不会因为后续补键打乱契约承诺的顺序。
        """
        start = time.perf_counter()

        # 模拟真实抽取的 IO 延迟：占住 async 语义，W2 换成真实模型调用时不改调用方
        await asyncio.sleep(SIMULATED_LATENCY_MS / 1000)

        data = {name: extract_value(text, name) for name in fields}
        missing = [name for name in fields if data[name] is None]
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return ExtractionOutput(
            data=data,
            confidence=compute_confidence(fields, data),
            model=self.model_name,
            elapsed_ms=elapsed_ms,
            missing=missing,
        )


__all__ = [
    "FIELD_RULES",
    "MODEL_NAME",
    "SIMULATED_LATENCY_MS",
    "ExtractionOutput",
    "RuleExtractor",
    "compute_confidence",
    "extract_value",
]
