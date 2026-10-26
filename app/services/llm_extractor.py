"""LLM 版抽取器：`RuleExtractor` 的可替换实现（W2 的兑现）。

**这个文件存在的唯一意义，就是证明"接口不变、实现可换"不是一句口号。**

它实现的方法签名与 `RuleExtractor.extract()` **一模一样**：

    async def extract(self, *, text: str, fields: Sequence[str]) -> ExtractionOutput

路由一行没改，`docs/api-spec.md` 一个字没改，`tests/test_api.py` 一行没改。
换的是组装那一行（`app/core/deps.py` 的 `get_extractor`）。

三层防御，对应三类真实会出的事故：

1. **模型输出不一定是合法 JSON**（尤其长文本、复杂字段时）。
   `_parse_payload` 容忍代码围栏和多层包装，仍失败则**整体降级**到规则版 ——
   对调用方来说这只是一次置信度略低的成功，不是 500。
2. **模型可能编造**。契约 §4.5 要求"抽不到 = null，永不编造"。
   仅靠 prompt 不可靠，所以 `_ground_value` 做**原文回指校验**：
   值里的关键片段必须在正文出现过，否则丢弃为 null。这条比 prompt 硬。
3. **模型可能返回未知字段或漏字段**。`data` 的键与顺序由**我们**按 `fields` 重建，
   绝不信模型给的键序 —— 顺序是契约（§2.1）。

**为什么降级而不是抛错**：契约 §3.1 规定模型超时/故障映射到 500。但"模型返回了
不合规的 JSON"是我方可以自愈的失败，用户要的是抽取结果，不是错误码。
只有**连规则版都算不出**时才该 500 —— 而规则版几乎不可能失败。
这个取舍写在 `docs/api-spec.md` §4 的"实现可换"里：契约管形状，实现管韧性。

**不 import fastapi**：本文件是服务层，依赖注入是路由层的事。
`AsyncOpenAI` 实例由 `deps.py` 传入，这里只依赖它的 `chat.completions.create`。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import Any, Final

from openai import AsyncOpenAI

from app.core.llm.retry import RetryPolicy, with_retry
from app.services.extractor import (
    ExtractionOutput,
    RuleExtractor,
    compute_confidence,
)

# 抽取是"照原文抄"，不要任何创造性。昨天实测过 temp=0 也不完全可复现，
# 但它是把漂移压到最小的选择（learn/llm/01_params.md §5）。
DEFAULT_TEMPERATURE: Final[float] = 0.0

# 单次抽取的输出上限。抽取结果是短 JSON，给太多只会让模型有空间啰嗦。
DEFAULT_MAX_TOKENS: Final[int] = 1024

SYSTEM_PROMPT: Final[str] = (
    "你是一个严谨的信息抽取器。规则：\n"
    "1. 只依据用户给出的正文取值，一个字都不要凭空补全。\n"
    "2. 正文里没有的字段，值必须是 null，严禁猜测或编造。\n"
    "3. 值保持正文里的原始写法（金额保留逗号和小数，不要换算或加单位）。\n"
    "4. 日期统一输出 YYYY-MM-DD；无法确定是合法日期就用 null。\n"
    "5. 只输出 JSON，不要解释、不要 Markdown 代码围栏。"
)

# 代码围栏（```json ... ```）。模型被要求"只输出 JSON"，但仍会有人加围栏。
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

# 回指校验时，值里被当作"关键片段"的最小长度。太短（如"张"）会在正文里到处命中，
# 校验就失去意义；3 个字符（中文一个词）是可用的下限。
_MIN_FRAGMENT = 3


class LlmExtractor:
    """LLM 抽取器。**无状态**，可安全地被 `get_extractor` 收成单例。

    为什么把 `client` 当构造参数而不是在内部 `get_llm_client()`：
    服务层不该知道依赖注入的存在。测试里可以直接塞一个假 client，
    不用碰环境变量、不用清 lru_cache。
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        fallback: RuleExtractor | None = None,
        extra_request: dict[str, Any] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # 降级目标：模型彻底不可用时仍能给出结果，而不是 500。
        self._fallback = fallback or RuleExtractor()
        # 网关特性（如本机网关的 reasoning_effort="none"）从这里注入，
        # 不写死在请求里 —— 换云端时这个 dict 应该是空的。
        self._extra_request = extra_request or {}
        # 重试策略。**默认关**（max_attempts=1），因为降级到规则版已经在
        # "失败"这条路上兜住了：再多试 2 次只会让用户多等几秒，而规则版几乎总能给出结果。
        # 需要"宁可多等也要模型结果"的场景（如 confidence 要求高）可以显式打开。
        self._retry_policy = retry_policy or RetryPolicy(max_attempts=1)

    @property
    def model_name(self) -> str:
        """写进响应头 `X-Extract-Model`（契约 §2.4：实现信息只走头，不进 body）。"""
        return self._model

    # ------------------------------------------------------------ 对外接口
    async def extract(self, *, text: str, fields: Sequence[str]) -> ExtractionOutput:
        """按 `fields` 顺序抽值。**签名与 `RuleExtractor.extract` 完全一致。**

        失败路径全部收敛到 `_fallback_output`，所以这个函数不会抛异常 ——
        与规则版行为对齐：抽取器的失败模式是"没抽到"，不是"崩了"。
        """
        start = time.perf_counter()

        try:
            # 走 `with_retry`：429/5xx/超时由它按指数退避+抖动重试，
            # 4xx 立刻抛出（凭证错了多试几次只是多烧钱）。
            # 重试次数用尽后抛 `RetryExhaustedError`，被下面的 except 接住并降级。
            response = await with_retry(
                lambda: self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": self._build_user_prompt(text, fields)},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    **self._extra_request,
                ),
                policy=self._retry_policy,
                label="extract",
            )
            payload = self._parse_payload(response.choices[0].message.content)
            raw_data = self._coerce_mapping(payload, fields)
        except Exception as exc:  # noqa: BLE001 - 意图明确：任何 LLM 侧失败都降级
            # 这里**故意**捕获宽泛异常：网络超时、401、429、坏 JSON、结构不符，
            # 对调用方而言都是"这次没抽好"，不该变成 500。
            # 降级原因写进日志，排查时看得到；对外只体现为 confidence 略低。
            _log_fallback(exc)
            return self._fallback_output(text, fields, start, reason=type(exc).__name__)

        # 键与顺序一律按 `fields` 重建，绝不信模型给的键序（契约 §2.1）
        data: dict[str, str | None] = {}
        for name in fields:
            value = raw_data.get(name)
            data[name] = self._ground_value(value, text)

        missing = [name for name in fields if data[name] is None]
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return ExtractionOutput(
            data=data,
            confidence=self._confidence(fields, data),
            model=self._model,
            elapsed_ms=elapsed_ms,
            missing=missing,
        )

    # ------------------------------------------------------------ 内部实现
    def _build_user_prompt(self, text: str, fields: Sequence[str]) -> str:
        """把字段名按序写进 prompt —— 顺序影响模型的输出顺序，但我们最终按 `fields` 重建。

        字段名用 JSON 数组原文写出而不是顿号拼接：字段名里可能有顿号、冒号，
        用结构化的形式模型才不会切错。
        """
        listed = json.dumps(list(fields), ensure_ascii=False)
        return (
            f"待抽取的字段（按此顺序返回全部字段）：{listed}\n\n"
            f'输出格式：{{"data": {{"字段名": "值或 null"}}}}\n\n'
            f"正文：\n{text}"
        )

    @staticmethod
    def _parse_payload(content: str | None) -> dict[str, Any]:
        """把模型输出解析成 dict。失败抛异常（由 `extract` 统一降级）。

        三级容忍，从最规范到最宽松：
        ① 纯 JSON；
        ② 包了代码围栏的 JSON（模型最常见的小动作）；
        ③ 文本里夹着 JSON（前后有解释性废话时，取第一个 { 到最后一个 }）。
        """
        if not content or not content.strip():
            raise ValueError("模型返回了空内容")  # 也可能是 reasoning 吃光 max_tokens

        raw = content.strip()

        # ② 代码围栏
        fenced = _FENCE.match(raw)
        if fenced:
            raw = fenced.group(1).strip()

        # ① 直接解析
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # ③ 捞第一个对象
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end <= start:
                raise ValueError(f"输出里找不到 JSON 对象：{raw[:120]!r}") from None
            try:
                parsed = json.loads(raw[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(f"输出不是合法 JSON：{exc.msg}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"输出 JSON 的顶层必须是对象，收到 {type(parsed).__name__}")
        return parsed

    @staticmethod
    def _coerce_mapping(payload: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
        """从 payload 里取出 `字段名 -> 值` 的映射。

        容忍两种常见包装：`{"data": {...}}`（我们要求的格式）和直接 `{...}`。
        字段名用 `fields` 里的**原样字符串**去取，不做大小写折叠 ——
        折叠会让 `Party_A` 误命中 `party_a`，而那可能是两个不同的字段。
        """
        candidate = payload.get("data", payload)
        if not isinstance(candidate, dict):
            raise ValueError("payload['data'] 必须是对象")

        # 模型偶尔会把缺失字段写成字符串 "null" / "None"，统一归一成真 None
        normalized: dict[str, Any] = {}
        for name in fields:
            value = candidate.get(name)
            if isinstance(value, str) and value.strip().lower() in {"null", "none", "n/a", ""}:
                value = None
            normalized[name] = value
        return normalized

    @staticmethod
    def _ground_value(value: Any, text: str) -> str | None:
        """回指校验：值必须能在正文里找到依据，否则丢弃为 `null`。

        契约 §4.5「抽不到 = null，**永不编造**」不能只靠 prompt ——
        模型编造一个不存在的甲方名字时，prompt 拦不住。这里做一道硬校验：

        - 值转字符串、去空白后为空 -> None
        - 值（或其关键片段）在正文里出现过 -> 保留
        - 一个片段都找不到 -> 判定为编造，返回 None

        为什么用"片段"而不是完整相等：正文里金额常是 `¥128,000.00`，
        而模型可能回 `128,000.00` —— 完整匹配会把正确结果误杀。
        反过来用"值是否包含于正文"又太严（值会被规范化）。
        按长度切片段、要求**至少一个**片段命中，是在"误杀正确值"和
        "放过编造值"之间的折中。
        """
        if value is None:
            return None
        if not isinstance(value, str):
            # 数字/布尔也接受：合同金额被模型写成 128000 是合理的
            value = str(value)
        stripped = value.strip()
        if not stripped:
            return None

        # 正文压缩掉空白，避免 `128, 000` 与 `128,000` 这种换行/空格差异误判
        haystack = re.sub(r"\s+", "", text)
        needle = re.sub(r"\s+", "", stripped)

        # 短值（如 "张三"、"12"）直接整体查；长值切片段查
        if len(needle) <= _MIN_FRAGMENT * 2:
            return stripped if needle in haystack else None

        fragments = [
            needle[i : i + _MIN_FRAGMENT]
            for i in range(0, len(needle) - _MIN_FRAGMENT + 1, _MIN_FRAGMENT)
        ]
        if any(frag in haystack for frag in fragments):
            return stripped
        return None

    @staticmethod
    def _confidence(fields: Sequence[str], data: dict[str, str | None]) -> float:
        """置信度沿用规则版的公式（契约 §4.4：语义不变）。

        为什么不让模型自己报个 0–1 的分数：模型自评分数没有校准，
        同一份正文它可以报 0.95 也可以报 0.7，下游排序完全没法用。
        用"命中比例 + 缺失打折"这个**可复现**的公式，跨实现、跨模型都可比。
        """
        return compute_confidence(fields, data)

    def _fallback_output(
        self,
        text: str,
        fields: Sequence[str],
        start: float,
        *,
        reason: str,
    ) -> ExtractionOutput:
        """降级到规则版，但**保留本次的真实耗时**与降级标记。

        `model` 字段带 `+fallback(原因)` 后缀：响应头 `X-Extract-Model` 会显示
        `buddy/glm-5.3+fallback(APITimeoutError)`。这是排查降级频率的唯一入口 ——
        否则"线上置信度莫名其妙偏低"永远查不到原因。
        """
        result = self._run_fallback_sync(text, fields)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return ExtractionOutput(
            data=result.data,
            confidence=result.confidence,
            model=f"{self._model}+fallback({reason})",
            elapsed_ms=elapsed_ms,
            missing=result.missing,
        )

    def _run_fallback_sync(self, text: str, fields: Sequence[str]) -> ExtractionOutput:
        """规则版是纯计算（除了那个 100ms 的模拟 sleep），但接口是 async。

        这里不能 `await`（调用点在 except 里，再 await 可能再抛），
        所以直接调用内部同步函数，复用同一套 `extract_value` 逻辑。
        """
        from app.services.extractor import extract_value  # 局部导入避免循环依赖

        data = {name: extract_value(text, name) for name in fields}
        missing = [name for name in fields if data[name] is None]
        return ExtractionOutput(
            data=data,
            confidence=compute_confidence(fields, data),
            model="rule-v1",
            elapsed_ms=0,
            missing=missing,
        )


def _log_fallback(exc: BaseException) -> None:
    """降级要留痕：没有日志的降级等于静默的数据质量下降。"""
    import logging

    logging.getLogger(__name__).warning(
        "LLM 抽取失败，降级到规则版：%s: %s", type(exc).__name__, exc
    )


__all__ = ["DEFAULT_MAX_TOKENS", "DEFAULT_TEMPERATURE", "SYSTEM_PROMPT", "LlmExtractor"]
