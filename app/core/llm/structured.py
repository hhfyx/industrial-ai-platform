"""结构化输出：把"模型给的非结构化文本"变成"类型安全的 Pydantic 对象"。

**这是 W4/W5/W7/W9 的共用底座，签名一旦定版不再改。**

    async def structured_call(
        prompt: str,
        schema: type[BaseModel],
        *,
        max_repair: int = 2,
        fallback: BaseModel | None = None,
    ) -> BaseModel

## 本网关的能力边界（实测，不是文档抄的）

见 `scripts/probe_response_format.py` 与 `probe_response_format.json`：

| 模式 | 网关接受参数？ | 真的强制 schema？ |
| --- | --- | --- |
| `response_format={"type":"json_schema","strict":true}` | ✅ 200 | **❌ 不接受时也回 200，模型照样自创字段名** |
| `response_format={"type":"json_object"}` | ✅ 200 | ❌ 只保证"是合法 JSON"，字段结构全靠 prompt |
| 都不传 | — | ❌ 只有 prompt |

实测三条证据：

1. 传完整 `json_schema`（含 enum/required/additionalProperties）时，模型输出了
   `{"category":"故障报修","priority":"紧急","summary":"..."}` —— **schema 里的
   四个 required 字段一个都没出**，反而自创了三个。
2. 同一个 schema 换最小案例（只要求 `{"a":"X"}`）时，输出**正好**是 `{"a":"X"}`——
   说明它**在照着 prompt 说的做，而不是在照着 schema 说做**。
3. 把 system prompt 改成"只按给定 JSON Schema 输出"，模型输出**仍然**自创
   `category/impact/possible_causes`。

**一句话结论**：本网关的 `response_format` 是**形式被接受、语义不强制**。
所以本模块不依赖它，而是走 **`json_object` 兜底 + Pydantic 强校验 + 报错回灌**。

## 三层防线

| 层 | 手段 | 拦的是什么 |
| --- | --- | --- |
| ① prompt | 把 schema 的字段清单与约束**写进 prompt** | 让模型倾向正确（不可靠，但不能省） |
| ② `json_object` | 拒绝非 JSON 输出 | 代码围栏、前后废话、半截输出 |
| ③ Pydantic | 本地校验类型/枚举/范围 | schema 漏掉的一切语义问题 |

③ 才是唯一的**真源**。①② 只是提高③ 一次通过的概率，降低重试成本。

## 为什么重试要回灌 Pydantic 报错原文

「你错了，请重试」是废话：模型不知道自己错在哪，下一轮大概率**重复同一个错误**
或**改坏本来对的字段**。Pydantic 的 `errors()` 给的是结构化报错 ——
字段路径、错在哪、收到的是什么值，模型照着改三件事，改对率显著高于盲重试。

**关键：提示里必须写「只修报错的字段」**。否则模型看到报错会倾向于把整个 JSON
重写一遍，而重写会把原本正确的字段也改掉 —— 这是实测踩到的坑，见
`_build_repair_messages` 的注释。

## 为什么 max_repair=2

每轮都是一次真实计费调用 + 秒级延迟。W4 的 RAG、W7 的 judge 都在线上链路里，
用户等不起 5 轮。2 轮是「修得回来」与「别让用户干等」的平衡点
（与 `app/services/llm_extractor.py` 的既有结论一致，那里已实测坏输出大多第 1 轮就修好）。

## 不 new client

任务要求「内部只依赖 `get_llm_client()`」。这里注意一个细节：
`client.py` 里的 `get_llm_client(settings)` 带 `Depends`，是 FastAPI 依赖。
本模块是**服务层**，按本仓库既有约定（`llm_extractor.py` 的 docstring）**不 import fastapi**。
所以这里调用的是 `get_llm_client(get_settings())` —— 拿进程级单例，
但仍然不在模块级实例化，也不 import fastapi。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Final, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.core.llm.client import get_llm_client

logger = logging.getLogger(__name__)

# 模型参数的默认值。结构化抽取要的是确定性，不是创造性。
DEFAULT_TEMPERATURE: Final[float] = 0.0

# 单次调用的输出上限。
#
# **这个值被实测修正过两次，不是拍的**（2026-10-29）。
# 根因始终是同一个：本网关默认 `reasoning=high`，**思维链先吃 max_tokens**，
# 真正留给 JSON 的预算 = max_tokens - 思维链长度。
#
# 两次实测记录（都是同一台网关、同一个模型）：
#
#   第一次（周四，四类模板端到端）：
#     · max_tokens=2048 → finish=length，compl=2048（打满），reasoning 4621 字符，content=None
#     · max_tokens=4096 → finish=stop，compl=1685，reasoning 2938 字符，正常
#
#   第二次（周六，摘要 A/B 跑 10 条）：
#     · max_tokens=4096 → finish=length，compl=4096（打满），reasoning 9405 字符，content=None
#     · max_tokens=8192 → finish=stop，compl=4485，reasoning 12007 字符，正常
#
# **教训：预算要按"最重的那个任务"定，不是按典型任务。**
# 摘要任务输入长、思维链长，4096 在它上面直接打满；分类/抽取那种短输出
# 用 8192 只是多留了余地，不会被用掉（模型到 stop 就停）。
#
# 代价：max_tokens 是**上限**不是预留，调大不额外花钱 ——
# `usage.completion_tokens` 只算实际产出。所以宁可给足。
DEFAULT_MAX_TOKENS: Final[int] = 8192
# 重试时的温度：第 1 轮已经是 0，修复轮稍微抬一点点有助于跳出同一个死循环。
# 但不能抬太高 —— 修复轮的任务是"改正具体字段"，不是"重新创作"。
REPAIR_TEMPERATURE: Final[float] = 0.2

# 代码围栏：```json ... ``` 或 ``` ... ```
_FENCE = re.compile(r"^\s*```(?:json|jsonc)?\s*(.*?)\s*```\s*$", re.DOTALL)

M = TypeVar("M", bound=BaseModel)


class StructuredOutputError(ValueError):
    """修复轮数用尽仍未拿到合法结构化输出。

    继承 `ValueError` 而不是自定义基类：调用方（`llm_extractor.py` 的降级路径、
    W4/W7 的上层）已经在按 `ValueError` 捕获，不会因为引入新异常类型而漏接。
    同时它可被 `except StructuredOutputError` 精确捕获，区分"模型输不出结构"
    和"其它 ValueError"。
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_raw: str = "",
        last_errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_raw = last_raw
        self.last_errors = last_errors or []


@dataclass
class RepairAttempt:
    """一轮失败的记录。用于日志、测试断言与成本核算。"""

    round_index: int
    raw: str
    errors: list[dict[str, Any]] = field(default_factory=list)
    kind: str = "validation"  # validation | json | empty

    def summary(self) -> str:
        if not self.errors:
            return f"第 {self.round_index + 1} 轮失败（{self.kind}）"
        parts = [f"{e.get('field', '?')}: {e.get('reason', '?')}" for e in self.errors[:4]]
        more = "" if len(self.errors) <= 4 else f" 等 {len(self.errors)} 处"
        return f"第 {self.round_index + 1} 轮失败（{self.kind}）：" + "；".join(parts) + more


# ============================================================ schema → prompt


def schema_hint(schema: type[BaseModel]) -> str:
    """把 Pydantic 模型的字段清单渲染成模型能读的说明。

    **为什么用 Pydantic 当唯一真源**：字段名、类型、枚举、必填性都只在这一处定义。
    prompt 里的说明是从模型**生成**的，不是手写的 —— 手写会和模型漂移，
    生成则永远同步（改了模型，prompt 自动跟着变）。

    枚举值也渲染进去：实测模型在没看到枚举清单时会自创分类名
    （`probe_response_format.json` 里的 `"category":"故障报修"`）。
    """
    lines: list[str] = []
    for name, fld in schema.model_fields.items():
        desc = fld.description or ""
        ann = fld.annotation
        # 枚举：Pydantic v2 用 Literal，取 __args__
        allowed = getattr(ann, "__args__", None)
        if allowed and all(isinstance(a, str) for a in allowed):
            type_note = "必须是以下之一：" + " / ".join(allowed)
        else:
            type_note = _type_note(ann)
        required = "必填" if fld.is_required() else "可选，缺省用 null"
        one = f"- {name}（{type_note}，{required}）"
        if desc:
            one += f"：{desc}"
        lines.append(one)
    return "\n".join(lines)


def _type_note(ann: Any) -> str:
    import typing

    origin = typing.get_origin(ann)
    if origin is typing.Union:  # Optional[X] / X | None
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if args:
            return f"{getattr(args[0], '__name__', str(args[0]))} 或 null"
    name = getattr(ann, "__name__", None)
    if name:
        return name
    return str(ann).replace("typing.", "")


def build_prompt(prompt: str, schema: type[BaseModel]) -> str:
    """把用户 prompt 和 schema 说明拼成一条完整指令。"""
    return (
        f"{prompt}\n\n"
        "【输出要求】\n"
        "只输出一个 JSON 对象，不要任何解释、不要 Markdown 代码围栏、不要前后缀。\n"
        "键必须严格使用下面列出的字段名，不得自创字段、不得改名、不得省略必填字段。\n"
        f"{schema_hint(schema)}"
    )


# ============================================================ 解析


def parse_json(text: str) -> dict[str, Any]:
    """从模型输出里抠出 JSON 对象。

    四级容错，从最规范到最宽松：
    ① 纯 JSON
    ② 被 ```json 围栏包住（模型最常见的小动作）
    ③ 前后有解释性废话，取第一个 `{` 到最后一个 `}`
    ④ 都失败 → 抛 ValueError

    **为什么不做"自动修 JSON"**：漏个逗号就去帮它补，会让真正的结构问题
    被掩盖成"能解析"。宁可让它失败一轮、把错误原文回灌，也不要猜。
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("模型输出为空")

    m = _FENCE.match(stripped)
    if m:
        stripped = m.group(1).strip()

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        first, last = stripped.find("{"), stripped.rfind("}")
        if first != -1 and last > first:
            try:
                obj = json.loads(stripped[first : last + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(f"无法解析为 JSON：{exc}") from exc
        else:
            raise ValueError("输出中找不到 JSON 对象") from None

    if not isinstance(obj, dict):
        raise ValueError(f"期望 JSON 对象，实际是 {type(obj).__name__}")
    return obj


# ============================================================ 修复提示


def _errors_of(exc: ValidationError) -> list[dict[str, Any]]:
    """把 Pydantic 报错转成「字段 / 原因 / 你给的值」三元组。

    这是回灌的核心：**必须带 loc 和 input**。只带 msg 的话模型不知道改哪。
    """
    out: list[dict[str, Any]] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "_root"
        out.append(
            {
                "field": loc,
                "reason": err.get("msg", ""),
                "got": err.get("input"),
                "type": err.get("type", ""),
            }
        )
    return out


def _build_repair_messages(
    messages: list[dict[str, Any]],
    bad_output: str,
    errors: list[dict[str, Any]],
    schema: type[BaseModel],
) -> list[dict[str, Any]]:
    """构造修复轮的消息：把错误输出 + 结构化报错一起回灌。

    **提示里必须写「只修报错的字段」** —— 这是本模块最容易忽略的一处。
    实测：只写「你的输出不符合要求，请重新输出」时，模型会把整个 JSON 重写，
    常常**改坏原本已经正确的字段**（本来对不对它不管，它只管重来一遍）。
    写清「其余字段保持原样」之后，修复才有收敛性。

    另一处细节：把 `got`（模型实际给的值）也回灌。只说「期望 string」不够，
    说「期望 string，你给了 128000」它才知道错在哪。
    """
    detail: list[str] = []
    for e in errors:
        line = f"- 字段 {e.get('field')}：{e.get('reason')}"
        got = e.get("got")
        if got is not None:
            line += f"（你给的是 {json.dumps(got, ensure_ascii=False)}）"
        detail.append(line)

    repair = (
        "你上一次的输出不符合要求的 schema，具体问题如下：\n" + "\n".join(detail) + "\n\n"
        "【修复要求】\n"
        "1. **只修改上面报错的字段**，其余字段必须原样保留，不要重写整个 JSON。\n"
        "2. 若某个字段的报错是「缺少」，请补上它并给出符合类型的值。\n"
        "3. 严格使用 schema 给出的字段名与取值集合，不要新增、改名或删除字段。\n"
        "4. 只输出 JSON 对象本身，不要解释、不要代码围栏。\n\n"
        "字段说明：\n" + schema_hint(schema)
    )

    return [
        *messages,
        {"role": "assistant", "content": bad_output},
        {"role": "user", "content": repair},
    ]


# ============================================================ 主函数


async def structured_call(
    prompt: str,
    schema: type[M],
    *,
    max_repair: int = 2,
    fallback: BaseModel | None = None,
) -> M:
    """调模型拿结构化对象，失败按 `max_repair` 轮自我修复。

    参数
    ----
    prompt
        任务描述。**不要**在这里自己拼 schema 说明，本函数会拼。
    schema
        Pydantic 模型，字段的唯一真源。返回值的类型即为它。
    max_repair
        首次调用之外最多再修几轮。默认 2。
    fallback
        修复用尽后的兜底值。给了就返回它（**不再抛异常**），
        没给就抛 `StructuredOutputError`。

        为什么用 `fallback=None` 而不是"抛异常或一个 sentinel"：
        调用方语义更直白 —— 「我接受降级」就传个值，「我要求必成功」就不传。

    返回
    ----
    `schema` 的实例。类型安全，调用方 `result.field` 有补全、有校验。
    """
    if max_repair < 0:
        raise ValueError("max_repair 不能为负数")

    client = get_llm_client(get_settings())
    model = get_settings().llm.model

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": build_prompt(prompt, schema)},
    ]

    attempts: list[RepairAttempt] = []
    last_raw = ""

    for round_index in range(max_repair + 1):
        temperature = DEFAULT_TEMPERATURE if round_index == 0 else REPAIR_TEMPERATURE

        raw = await _request(
            client,
            model=model,
            messages=messages,
            temperature=temperature,
        )
        last_raw = raw

        # ---- ① 解析 JSON ----
        try:
            payload = parse_json(raw)
        except ValueError as exc:
            attempt = RepairAttempt(round_index=round_index, raw=raw, kind="json")
            attempt.errors = [
                {"field": "_json", "reason": str(exc), "got": raw[:200], "type": "json_invalid"}
            ]
            attempts.append(attempt)
            if round_index >= max_repair:
                return _exhausted(attempts, max_repair, last_raw, fallback)
            messages = _build_repair_messages(messages, raw, attempt.errors, schema)
            logger.info("第 %d 轮 JSON 解析失败，回灌修复", round_index + 1)
            continue

        # ---- ② Pydantic 校验 ----
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            errors = _errors_of(exc)
            attempt = RepairAttempt(
                round_index=round_index, raw=raw, errors=errors, kind="validation"
            )
            attempts.append(attempt)
            if round_index >= max_repair:
                return _exhausted(attempts, max_repair, last_raw, fallback)
            logger.info("第 %d 轮校验失败（%d 处），回灌修复", round_index + 1, len(errors))
            messages = _build_repair_messages(messages, raw, errors, schema)
            continue

    # 不可达：循环内每轮都 return 或 continue
    return _exhausted(attempts, max_repair, last_raw, fallback)


def _exhausted(
    attempts: list[RepairAttempt],
    max_repair: int,
    last_raw: str,
    fallback: BaseModel | None,
) -> Any:
    """修复用尽：有 fallback 就降级，没有就抛可识别异常。"""
    total = max_repair + 1
    last_errors = attempts[-1].errors if attempts else []
    detail = "；".join(a.summary() for a in attempts) or "无记录"

    if fallback is not None:
        logger.warning("结构化调用 %d 轮全部失败，使用 fallback 降级：%s", total, detail)
        return fallback

    raise StructuredOutputError(
        f"结构化调用失败：{total} 轮（1 次首调 + {max_repair} 次修复）均未通过校验。{detail}",
        attempts=total,
        last_raw=last_raw,
        last_errors=last_errors,
    )


async def _request(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
) -> str:
    """发一次请求，返回文本。

    这里**显式传 `response_format={"type":"json_object"}`**：本网关实测接受它，
    能挡掉代码围栏和前后废话（虽然挡不住结构错误 —— 那由 Pydantic 管）。

    失败时只做一次性兼容重试：有些网关不认 `response_format`，会直接 400。
    那种情况下关掉它再试，而不是整体失败 —— 强约束是优化项，不是前提。
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }

    try:
        response = await client.chat.completions.create(**payload)
    except Exception as exc:  # noqa: BLE001
        if _looks_like_format_rejected(exc):
            logger.warning("网关拒绝 response_format，去掉后重试：%s", exc)
            payload.pop("response_format", None)
            response = await client.chat.completions.create(**payload)
        else:
            raise

    content = response.choices[0].message.content

    # 用量旁路记录：**不改 `structured_call` 的签名**（周三已定版，改了后面四周全返工），
    # 而是把每次真实调用的 usage 写进一个可选的观测钩子。
    # 为什么要这个：A/B 脚手架要算 token/成本，而签名返回 `BaseModel` 拿不到 usage。
    # 让调用方自己去解析响应是不可能的事 —— 那就等于绕过本模块直连 HTTP。
    _record_usage(response)

    if content is None:
        # 推理模式吃光 max_tokens 的典型症状。**必须当失败**，不能 `or ""`
        # 静默成空串 —— 空串会被下游当成"模型说什么都没有"，那是错误信息。
        raise ValueError("模型返回 content=None（可能被 max_tokens 截断或推理占满）")
    return content


# ---- 用量观测钩子 -----------------------------------------------------------
# 默认 None（生产路径零开销）。实验脚手架把它设成一个可调用对象即可收集用量。

_USAGE_SINK: Any = None


def set_usage_sink(sink: Any) -> None:
    """注册用量收集器。传 `None` 关闭。

    sink 形如 `callable(usage: dict)`，会被传入
    `{"prompt_tokens": int, "completion_tokens": int}`。
    """
    global _USAGE_SINK
    _USAGE_SINK = sink


def _record_usage(response: Any) -> None:
    if _USAGE_SINK is None:
        return
    try:
        u = getattr(response, "usage", None)
        if u is None:
            return
        _USAGE_SINK(
            {
                "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
            }
        )
    except Exception:  # noqa: BLE001
        # 观测失败绝不能影响主路径
        logger.debug("usage sink 调用失败", exc_info=True)


def _looks_like_format_rejected(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        k in text
        for k in ("response_format", "json_object", "json_schema", "unsupported", "not support")
    )


__all__ = [
    "RepairAttempt",
    "StructuredOutputError",
    "build_prompt",
    "parse_json",
    "schema_hint",
    "set_usage_sink",
    "structured_call",
]
