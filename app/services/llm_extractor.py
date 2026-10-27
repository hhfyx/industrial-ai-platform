"""LLM 版抽取器：`RuleExtractor` 的可替换实现（W2 主交付）。

**这个文件的核心命题：签名不变，行为更强。**

    async def extract(self, *, text: str, fields: Sequence[str]) -> ExtractionOutput

路由、`docs/api-spec.md`、`tests/test_api.py` 都不动，换的是 `deps.get_extractor`。

## 三层约束，从"软"到"硬"

| 层 | 手段 | 拦的是什么 |
| --- | --- | --- |
| ① prompt | 明确要求只输出 JSON | 让模型**倾向**正确（不可靠） |
| ② JSON Schema | `response_format` 强约束结构 | 结构坏（漏键、类型错） |
| ③ Pydantic | 本地校验 `{field: value}` | Schema 也漏掉的语义问题 |

**为什么三层都要**：`response_format` 并非所有网关都真正支持（本机网关接受参数
但未必强制），所以不能只靠它；而只靠 prompt 时坏 JSON 相当常见。
三层叠加后剩下的失败才交给降级。

## 校验失败 → 回灌重试（最多 2 轮）

这是今天最重要的机制。模型给了不合规输出时，**把 Pydantic 的错误原样喂回去**，
让它自己改 —— 比"重试同一个 prompt 碰运气"有效得多，因为错误信息直接告诉了它
哪里不对、错在哪个字段、收到了什么值。

两轮仍失败 → 降级规则版并标 `degraded: true`（api-spec §2.5）。
**为什么是 2 轮而不是 5 轮**：每轮都是一次真实计费调用 + 秒级延迟。
实测坏输出大多在第一轮就修好了，第三轮之后的收益急剧下降，
而用户已经等了 6 秒以上 —— 那时候"给个规则版结果 + 标记降级"比继续等更有价值。

## 不 import fastapi

服务层不该知道依赖注入的存在。`client` 从构造参数拿（今天的 DI 改造），
测试里直接塞 `FakeLLMClient` 就能跑，不需要碰环境变量或网络。
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from app.core.llm.retry import RetryPolicy, with_retry
from app.services.extractor import (
    ExtractionOutput,
    RuleExtractor,
    compute_confidence,
)

logger = logging.getLogger(__name__)

# 抽取是"照原文抄"，不要任何创造性。实测过 temp=0 也不完全可复现
# （learn/llm/01_params.md §5），但它是把漂移压到最小的选择。
DEFAULT_TEMPERATURE: Final[float] = 0.0

# 单次抽取的输出上限。抽取结果是短 JSON，给太多只会有空间啰嗦。
DEFAULT_MAX_TOKENS: Final[int] = 1024

# 校验失败后最多再问几轮。见模块 docstring：2 轮是"修得回来"与"别让用户干等"的平衡点。
MAX_CORRECTION_ROUNDS: Final[int] = 2

SYSTEM_PROMPT: Final[str] = (
    "你是一个严谨的信息抽取器。规则：\n"
    "1. 只依据用户给出的正文取值，一个字都不要凭空补全。\n"
    "2. 正文里没有的字段，值必须是 null，严禁猜测或编造。\n"
    "3. 值保持正文里的原始写法（金额保留逗号和小数，不要换算或加单位）。\n"
    "4. 日期统一输出 YYYY-MM-DD；无法确定是合法日期就用 null。\n"
    "5. 只输出一个 JSON 对象，键就是用户给的字段名，不要有 data 外层包装、"
    "不要解释、不要 Markdown 代码围栏。"
)

# 代码围栏（```json ... ```）。模型被要求"只输出 JSON"，但仍会有人加围栏。
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

# 回指校验时，值里被当作"关键片段"的最小长度。
_MIN_FRAGMENT: Final[int] = 3

# 这些字符串会被模型当成"没有"，统一归一成 None
_NULLISH: Final[frozenset[str]] = frozenset({"null", "none", "n/a", "na", "-", ""})


# ============================================================ 校验模型


@dataclass
class CorrectionTrace:
    """一次校验失败 + 回灌的记录。用于日志、测试断言与成本核算。"""

    round_index: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    raw_output: str = ""

    def summary(self) -> str:
        parts = [f"{e.get('field', '?')}: {e.get('reason', '?')}" for e in self.errors]
        return "; ".join(parts) or "（无字段级错误）"


def build_schema_model(fields: Sequence[str]) -> type[BaseModel]:
    """按请求的 `fields` **动态**构造 Pydantic 校验模型。

    为什么动态构造而不是用一个固定的 dict 模型：
    - `response_format` 里的 JSON Schema 需要**精确列出**请求的字段，
      模型才会按这几个键输出，而字段名是每个请求都不同的
    - 固定模型只能校验"键是不是字符串"，动态模型能校验"缺了哪个键"

    每个字段类型是 `str | None`：契约要求值恒为 `string | null`（§2.1）。
    允许 `None` 是刻意的 —— "抽不到"是正常结果，不是校验错误。
    """

    def _field() -> tuple[Any, Any]:
        # 用**字符串注解** "str | None"：create_model 支持字符串类型表达式，
        # 这样不必在运行时导入 typing.Optional，也躲开 ruff 的 UP045。
        #
        # **必须用 `...` 而不是 `None` 作默认值**：默认给 None 会让"缺键"静默通过
        # （pydantic 用默认值补齐），于是"模型漏了一个字段"永远发现不了。
        # 契约要求 `data` 的键与 `fields` 完全一致，缺键必须是校验错误 ——
        # 用 `...`（Required）才能让 required 真正生效。实测踩过这个坑。
        return ("str | None", ...)

    model: type[BaseModel] = create_model(  # type: ignore[call-overload]
        "ExtractPayload",
        __config__=ConfigDict(extra="forbid"),
        **{name: _field() for name in fields},
    )
    return model


def json_schema_for(fields: Sequence[str]) -> dict[str, Any]:
    """生成 `response_format` 用的 JSON Schema。

    **`additionalProperties: False` 是关键**：没有它，模型可以合法地多返回键，
    而契约要求 `data` 的键与 `fields` **完全一致**。让模型层就守住这条，
    比事后在代码里删多余键更省事，也更早暴露问题。

    `required` 列出**全部**字段：即便某字段抽不到，也应该出现且值为 `null` ——
    "缺键"和"值为 null"对下游是不同的东西，前者会让前端少一行展示位。
    """
    return {
        "type": "object",
        "properties": {name: {"type": ["string", "null"]} for name in fields},
        "required": list(fields),
        "additionalProperties": False,
    }


# ============================================================ 抽取器


class LlmExtractor:
    """LLM 抽取器。**无状态**，可安全地被 `get_extractor` 收成单例。

    `client` 从构造参数拿（DI 改造）：内部**绝不 new**。
    这样测试里塞 `FakeLLMClient`、生产里塞真的 `AsyncOpenAI`，
    两条路径走的是同一个类、同一份逻辑。
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
        use_response_format: bool = True,
        max_correction_rounds: int = MAX_CORRECTION_ROUNDS,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # 降级目标：模型彻底不可用时仍能给出结果，而不是 500。
        self._fallback = fallback or RuleExtractor()
        # 网关特性（本机网关的 reasoning_effort="none"）从这里注入，不写死。
        self._extra_request = extra_request or {}
        # 重试策略：默认**不开**（max_attempts=1）。
        # 降级到规则版已经在"失败"这条路上兜住了，再多试 2 次只会让用户多等几秒。
        self._retry_policy = retry_policy or RetryPolicy(max_attempts=1)
        # 有些网关不支持 response_format，用这个开关关掉（换回纯 prompt 约束）
        self._use_response_format = use_response_format
        self._max_correction_rounds = max_correction_rounds
        # 最近一次抽取的校验轨迹，供测试与成本核算读取（无状态承诺的例外，
        # 但它是"本次调用"的诊断数据，不跨调用共享语义）
        self.last_trace: list[CorrectionTrace] = []

    @property
    def model_name(self) -> str:
        """写进响应头 `X-Extract-Model`（契约 §2.4：实现信息只走头，不进 body）。"""
        return self._model

    # ------------------------------------------------------------ 对外接口
    async def extract(self, *, text: str, fields: Sequence[str]) -> ExtractionOutput:
        """按 `fields` 顺序抽值。**签名与 `RuleExtractor.extract` 完全一致。**

        三条出口，全部返回 `ExtractionOutput`，绝不抛异常：
        1. 模型一次给出合规输出 → 正常返回
        2. 校验失败 → 回灌错误重试（最多 `max_correction_rounds` 轮）→ 合规即返回
        3. 用尽仍未成功 / 网络或模型故障 → 降级规则版 + `degraded=True`
        """
        start = time.perf_counter()
        self.last_trace = []

        if not fields:
            # 空字段直接给空结果，不必花一次调用
            return ExtractionOutput(data={}, confidence=0.0, model=self._model, elapsed_ms=0)

        try:
            raw_data, corrections = await self._extract_with_correction(text, fields)
        except Exception as exc:  # noqa: BLE001 - 任何 LLM 侧失败都降级
            # **故意捕获宽泛异常**：网络超时、401、429、坏 JSON、结构不符，
            # 对调用方都是"这次没抽好"，不该变成 500。
            logger.warning("LLM 抽取失败，降级规则版：%s: %s", type(exc).__name__, exc)
            return self._degraded_output(text, fields, start, reason=type(exc).__name__)

        # 键与顺序一律按 `fields` 重建，绝不信模型给的键序（契约 §2.1）
        data: dict[str, str | None] = {}
        for name in fields:
            data[name] = self._ground_value(raw_data.get(name), text)

        missing = [name for name in fields if data[name] is None]
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return ExtractionOutput(
            data=data,
            confidence=compute_confidence(fields, data),
            model=self._model,
            elapsed_ms=elapsed_ms,
            missing=missing,
            degraded=False,
        )

    # ------------------------------------------------------------ 抽取 + 纠错
    async def _extract_with_correction(
        self, text: str, fields: Sequence[str]
    ) -> tuple[dict[str, Any], list[CorrectionTrace]]:
        """跑"请求 → 校验 → 失败则回灌"的循环。

        **回灌的做法**：把 assistant 的错误输出和一条 user 的纠正提示追加进
        messages，形成多轮对话。这样模型看得到**自己刚才写了什么**和**错在哪** ——
        比"重发同一个 prompt"有效得多。

        返回 `(合法数据, 轨迹列表)`。用尽轮数抛 `ValueError`（由调用方降级）。
        """
        schema_model = build_schema_model(fields)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(text, fields)},
        ]

        traces: list[CorrectionTrace] = []
        last_error: str = ""

        for round_index in range(self._max_correction_rounds + 1):
            content = await self._request(messages, fields)

            # ---- ① 解析成 dict（坏 JSON 也算一种"校验失败"，同样回灌）----
            try:
                payload = self._parse_payload(content, fields)
            except ValueError as exc:
                last_error = str(exc)
                trace = CorrectionTrace(
                    round_index=round_index,
                    errors=[{"field": "_json", "reason": last_error, "got": content[:160]}],
                    raw_output=content,
                )
                traces.append(trace)
                self.last_trace = traces
                if round_index >= self._max_correction_rounds:
                    raise ValueError(f"JSON 解析连续失败：{last_error}") from exc
                messages = self._build_correction_messages(messages, content, trace)
                continue

            # ---- ② Pydantic 校验结构 ----
            try:
                schema_model.model_validate(payload)  # 只为触发校验，值从原始 dict 取
            except ValidationError as exc:
                errors = [
                    {
                        "field": ".".join(str(p) for p in err["loc"]) or "_model",
                        "reason": err["msg"],
                        "got": err.get("input"),
                    }
                    for err in exc.errors()
                ]
                trace = CorrectionTrace(round_index=round_index, errors=errors, raw_output=content)
                traces.append(trace)
                self.last_trace = traces
                logger.info(
                    "第 %d 轮校验失败（%d 个字段问题），回灌重试", round_index + 1, len(errors)
                )
                if round_index >= self._max_correction_rounds:
                    raise ValueError(
                        f"校验连续失败 {self._max_correction_rounds + 1} 轮：{trace.summary()}"
                    ) from exc
                messages = self._build_correction_messages(messages, content, trace)
                continue

            # ---- ③ 通过：返回**已剥掉外层包装**的 dict ----
            # 这里直接用 payload，而不是重新 json.loads(content)：
            # 包装已在 _parse_payload 里剥过，重解析会把它又装回来。
            return payload, traces

        # 理论不可达（循环内都 raise 了），留作防御
        raise ValueError(f"抽取失败：{last_error}")

    async def _request(self, messages: list[dict[str, Any]], fields: Sequence[str]) -> str:
        """发一次请求，返回模型文本。带重试（429/5xx/超时）。

        `response_format` 用 `json_schema` 做**强约束**。不同网关支持度不一，
        所以失败时（400 一类）会关掉它重试一次 —— 兼容性比"死磕强约束"更重要。
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            **self._extra_request,
        }
        if self._use_response_format:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ExtractPayload",
                    "strict": True,
                    "schema": json_schema_for(fields),
                },
            }

        async def _call() -> Any:
            return await self._client.chat.completions.create(**payload)

        try:
            response = await with_retry(_call, policy=self._retry_policy, label="extract")
        except Exception as exc:  # noqa: BLE001
            # 网关不支持 response_format 时常常是 400/422。这时关掉它再试一次，
            # 而不是整体降级 —— 强约束是优化项，不是前提。
            if self._use_response_format and _looks_like_unsupported_format(exc):
                logger.warning("网关不接受 response_format，关闭强约束后重试：%s", exc)
                payload.pop("response_format", None)
                response = await with_retry(
                    _call, policy=self._retry_policy, label="extract-noschema"
                )
            else:
                raise

        content = response.choices[0].message.content
        if content is None:
            # 推理模式吃光 max_tokens 的典型症状（learn/llm/01_params.md §3.1）——
            # **必须当失败**，不能 `or ""` 静默成空串（那会被下游当成"模型说没有"）。
            raise ValueError("模型返回 content=None（可能被 max_tokens 截断或推理占满）")
        return content

    # ------------------------------------------------------------ prompt 构造
    def _build_user_prompt(self, text: str, fields: Sequence[str]) -> str:
        """字段名以 JSON 数组原文写出 —— 字段名里可能有顿号、冒号，结构化才不会切错。"""
        listed = json.dumps(list(fields), ensure_ascii=False)
        return (
            f"待抽取的字段（按此顺序返回全部字段）：{listed}\n\n"
            f"输出格式：一个 JSON 对象，键是上面列出的字段名，值为字符串或 null。\n"
            f'例如：{{"字段名A": "抽到的值", "字段名B": null}}\n\n'
            f"正文：\n{text}"
        )

    @staticmethod
    def _build_correction_messages(
        messages: list[dict[str, Any]], bad_output: str, trace: CorrectionTrace
    ) -> list[dict[str, Any]]:
        """把"上一轮的错误输出 + 具体错误"追加进对话，让模型自我修正。

        **错误信息要具体到字段级**：只说"格式不对"模型不知道改哪里；
        说清"字段 X 应该是 string 或 null，你给了 128000（number）"它才能改对。
        """
        detail_lines = [
            f"- 字段 {e.get('field')}：{e.get('reason')}"
            + (f"（你给的是 {e.get('got')!r}）" if e.get("got") is not None else "")
            for e in trace.errors
        ]
        correction = (
            "你上一次的输出不符合要求，问题如下：\n"
            + "\n".join(detail_lines)
            + "\n\n请只重新输出**一个合法的 JSON 对象**："
            "键必须与我列出的字段完全一致，值为字符串或 null，不要任何解释或代码围栏。"
        )
        return [
            *messages,
            {"role": "assistant", "content": bad_output},
            {"role": "user", "content": correction},
        ]

    # ------------------------------------------------------------ 解析与校验
    @staticmethod
    def _parse_payload(content: str | None, fields: Sequence[str] | None = None) -> dict[str, Any]:
        """把模型输出解析成 dict。失败抛 `ValueError`。

        三级容错，从最规范到最宽松：
        ① 纯 JSON；
        ② 包了代码围栏的 JSON（模型最常见的小动作）；
        ③ 文本里夹着 JSON（前后有解释性废话时，取第一个 { 到最后一个 }）。

        **解析成功后立刻剥掉 `{"data": {...}}` 外层包装**（如果 `fields` 给出）。
        为什么必须在**这里**剥、而不是校验之后：
        校验模型是 `additionalProperties: False` 的，`data` 这个键会被判成
        "多余字段"而整体失败。实测踩到 —— 测试里带 `{"data": ...}` 的样本
        一路降级，报的错是 `data: Extra inputs are not permitted`，
        看起来像"模型多给了键"，实际是"我们没先解包装"。

        只在**恰好有这个外层**时才剥：如果 `fields` 里真有一个叫 `data` 的字段，
        那 `payload` 的键就是字段名本身，`payload.get("data")` 不会是 dict，
        自然不会被误剥。
        """
        if not content or not content.strip():
            raise ValueError("模型返回了空内容")

        raw = content.strip()
        fenced = _FENCE.match(raw)
        if fenced:
            raw = fenced.group(1).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end <= start:
                raise ValueError(f"输出里找不到 JSON 对象：{raw[:120]!r}") from None
            try:
                parsed = json.loads(raw[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(f"输出不是合法 JSON：{exc.msg}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"输出 JSON 的顶层必须是对象，收到 {type(parsed).__name__}")

        parsed = LlmExtractor._unwrap_data_envelope(parsed, fields)
        return LlmExtractor._coerce_scalar_values(parsed)

    @staticmethod
    def _coerce_scalar_values(parsed: dict[str, Any]) -> dict[str, Any]:
        """把数字/布尔值**转成字符串**，而不是让校验失败。

        **为什么归一而不是拒绝**：契约（§2.1）要求值恒为 `string | null`，
        但"模型把金额写成 `128000` 而不是 `"128000"`"是**极其常见**的行为。
        严格拒绝它要浪费一整轮回灌（一次计费调用 + 几秒延迟），
        而转换是确定性的、零成本的，且转换后的结果**正是契约要的东西**。

        这边界值得写清楚：**"值的类型不合契约" 和 "值的内容不对"是两回事**。
        前者我们能在本地可靠地修，后者（编造、错了）才需要回灌或丢弃。
        把能自己修的推给模型去修，是白花钱。

        只处理标量（str/int/float/bool）。嵌套的 dict/list 不转 ——
        那说明模型结构理解错了，属于模型该修的问题。
        """
        coerced: dict[str, Any] = {}
        for key, value in parsed.items():
            if value is None or isinstance(value, str):
                coerced[key] = value
            elif isinstance(value, (int, float, bool)):
                # bool 必须在数字之前判断？不需要：str(True) == "True" 也可接受。
                # 但 float 的 128000.0 会变成 "128000.0"，这可能不是原文写法 ——
                # 所以整数型 float 去掉 .0，尽量贴近原文。
                if isinstance(value, float) and value.is_integer():
                    coerced[key] = str(int(value))
                else:
                    coerced[key] = str(value)
            else:
                coerced[key] = value  # 交给 Pydantic 报错，那才是真结构问题
        return coerced

    @staticmethod
    def _unwrap_data_envelope(
        parsed: dict[str, Any], fields: Sequence[str] | None
    ) -> dict[str, Any]:
        """容忍 `{"data": {字段: 值}}` 这种外层包装。

        判据要**同时满足三条**才剥，避免把真实字段名叫 `data` 的情况误伤：
        1. 顶层只有 `data` 一个键（多一个都说明这不是单纯的包装）
        2. `data` 的值是 dict
        3. 没有把 `data` 本身列为请求字段
        """
        if not fields or "data" in fields:
            return parsed
        inner = parsed.get("data")
        if len(parsed) == 1 and isinstance(inner, dict):
            return inner
        return parsed

    @staticmethod
    def _ground_value(value: Any, text: str) -> str | None:
        """回指校验：值必须能在正文里找到依据，否则丢弃为 `null`。

        契约 §4.5「抽不到 = null，**永不编造**」不能只靠 prompt ——
        模型编造一个不存在的甲方名字时，prompt 拦不住。

        为什么用"片段"而不是完整相等：正文里金额常是 `¥128,000.00`，
        而模型可能回 `128,000.00` —— 完整匹配会把正确结果误杀。
        """
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        stripped = value.strip()
        if stripped.lower() in _NULLISH:
            return None

        haystack = re.sub(r"\s+", "", text)
        needle = re.sub(r"\s+", "", stripped)

        if len(needle) <= _MIN_FRAGMENT * 2:
            return stripped if needle in haystack else None

        fragments = [
            needle[i : i + _MIN_FRAGMENT]
            for i in range(0, len(needle) - _MIN_FRAGMENT + 1, _MIN_FRAGMENT)
        ]
        return stripped if any(frag in haystack for frag in fragments) else None

    # ------------------------------------------------------------ 降级
    def _degraded_output(
        self, text: str, fields: Sequence[str], start: float, *, reason: str
    ) -> ExtractionOutput:
        """降级到规则版并**明确标记** `degraded=True`（api-spec §2.5）。

        `model` 带 `+fallback(原因)` 后缀：响应头会显示
        `buddy/glm-5.3+fallback(APITimeoutError)`。这是排查降级频率的唯一入口 ——
        否则"线上置信度莫名其妙偏低"永远查不到原因。
        """
        result = self._fallback.extract_sync(text, fields)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return ExtractionOutput(
            data=result.data,
            confidence=result.confidence,
            model=f"{self._model}+fallback({reason})",
            elapsed_ms=elapsed_ms,
            missing=result.missing,
            degraded=True,
        )


def _looks_like_unsupported_format(exc: BaseException) -> bool:
    """判断异常是否像"网关不认 response_format"。

    典型是 400/422 + 消息里提到 response_format / json_schema / unsupported。
    只用作**降级强约束**的依据，判错也只是多试一次，代价可控。
    """
    status = getattr(exc, "status_code", None)
    if status not in (400, 404, 422):
        return False
    message = str(getattr(exc, "message", "")) + str(exc)
    lowered = message.lower()
    return any(
        k in lowered for k in ("response_format", "json_schema", "unsupported", "not support")
    )


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "MAX_CORRECTION_ROUNDS",
    "SYSTEM_PROMPT",
    "CorrectionTrace",
    "LlmExtractor",
    "build_schema_model",
    "json_schema_for",
]
