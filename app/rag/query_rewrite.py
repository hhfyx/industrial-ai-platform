"""多轮改写层：`rewrite(question, history)` —— 指代消解，只改检索输入。

## 这一层的位置（架构红线）

`retrieve()` 是 W6 冻结的契约，W9 还要把它包成 Agent 工具。
本模块是 `pipeline.answer()` **调用方**的一层：

    用户问题 + 历史 → rewrite() → standalone_question → pipeline.answer(standalone_question)

为什么不能放进 retrieve() 内部：`retrieve(query, k)` 是"用 query 找块"，
改写是"怎么得到 query"——两件事混在一起，Agent 调工具时会被迫多传 history
（见 `retrieval.py` 模块 docstring："检索段不产生新信息"）。

## 三条边界（tests/test_query_rewrite.py 逐条锁住）

| 边界 | 行为 | 为什么 |
| --- | --- | --- |
| 首轮不改写 | `history` 为空 → 原样返回，**不调模型** | W7 的 200 条单轮题跑在"当前系统"上，首轮改写会让周日回归变红 |
| 改写失败降级 | structured_call 抛错/修复用尽 → fallback = 原问 | 检索不能因为改写层挂掉而挂掉；降级结果也落盘（W7 纪律） |
| 指代歧义反问 | `needs_clarification=true` → 交互路径反问；非交互路径用原问 + 记标记 | 硬猜会把检索带向错误方向，且无法归因 |

## 字符预算口径（字符近似，不引 tokenizer）

仓库现有预算全是字符口径（`pipeline.max_context_chars`）。本层的
`max_history_chars` 由调用方按「总输入预算 × 30% 折算字符数」传入；
超出预算时**从最旧截起**，保留最近的历史（指代消解最依赖近处的先行词）。

## 每次调用都落盘（W5/W7 纪律的延续）

改写层的"改了什么 / 为什么没改 / 是不是降级"必须可回放——
W7 的教训是归因要看文本不看标志位。`REWRITE_SINK` 与
`structured_call` 的 `_USAGE_SINK` 同构：默认 None（零开销），
实验脚本把它设成可调用对象即可收集。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Final

from app.core.llm.structured import StructuredOutputError, structured_call
from app.prompts.loader import load_prompt
from app.prompts.rewrite.schema import RewriteResult

logger = logging.getLogger(__name__)

# 进入改写模板的历史轮数上限。超过 3 轮的部分由模型压进 rolling_summary
# （"最近 3 轮原文 + 更早的摘要"的裁剪策略）。仅指**轮**（一问一答为一轮），
# 不是消息条数。
MAX_ROUNDS_IN_PROMPT: Final[int] = 3

# 历史文本的默认字符预算。**这不是拍的总预算**：调用方应按
# 「总输入预算 × 30%」折算传入；这个默认值只是不让没传的调用静默撑爆 prompt。
DEFAULT_MAX_HISTORY_CHARS: Final[int] = 2000

# 改写调用失败时的重试预算。改写是检索的前置层，每加一轮重试，
# 用户的等待就多一次完整调用 —— 2 轮与 structured_call 的默认一致，
# 但改写输出短（一句话 + 一个布尔），绝大多数坏输出第 1 轮就修好。
REWRITE_MAX_REPAIR: Final[int] = 2


# ---- 观测钩子 ---------------------------------------------------------------
# 默认 None（生产路径零开销）。multi_turn_eval 等实验脚手架注册可调用对象收集。

RewriteSink = Any  # callable(record: dict) -> None

REWRITE_SINK: RewriteSink | None = None


def set_rewrite_sink(sink: RewriteSink | None) -> None:
    """注册改写记录收集器。传 `None` 关闭。

    record 形如::

        {
            "question": str,              # 原问
            "standalone_question": str,   # 实际用于检索的问题
            "needs_clarification": bool,
            "missing": list[str],
            "rewrote": bool,              # 是否真的改了（降级/首轮/歧义都为 False）
            "degraded": bool,             # 是否走了降级 fallback
            "rounds": int,                # 进入改写时历史有几轮
            "elapsed_ms": int,
        }
    """
    global REWRITE_SINK
    REWRITE_SINK = sink


def _emit(record: dict[str, Any]) -> None:
    if REWRITE_SINK is None:
        return
    try:
        REWRITE_SINK(record)
    except Exception:  # noqa: BLE001 - 观测失败绝不能影响主路径
        logger.debug("rewrite sink 调用失败", exc_info=True)


# ---- 数据结构 ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RewriteOutcome:
    """改写层的完整产物：给检索的问题 + 可归因的元数据。"""

    standalone_question: str
    """实际用于检索的问题。**任何路径下它都不会是空串**——
    降级和歧义路径都回填原问，检索层永远有输入。"""

    needs_clarification: bool = False
    """指代歧义。交互路径（UI）应反问；非交互路径（CLI/eval）用原问并记标记。"""

    missing: list[str] = field(default_factory=list)
    """歧义时缺什么。仅在 needs_clarification=True 时非空。"""

    rewrote: bool = False
    """改写是否真的改变了问题。降级/首轮/歧义都为 False ——
    对照实验按它分母：rewrote=False 的样本不计入改写收益。"""

    degraded: bool = False
    """是否走了降级（模型调用失败、修复用尽）。"""

    rolling_summary: str = ""
    """更早轮次的滚动摘要（历史 > 3 轮时才有值）。空串 = 无摘要可存。"""


# ---- 内部工具 ---------------------------------------------------------------


def _normalize_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """把历史压成模板需要的最小结构：只留 role/content 纯文本。

    citations / timing / contexts 等展示字段不进 prompt——
    塞进去只会浪费 token 并干扰模型（与 `_history_for_model` 同一立场）。
    非法项（缺 role/content、role 不对）直接丢弃，不抛——
    历史是调用方拼的，坏一条不该让检索挂掉。
    """
    if not history:
        return []
    out: list[dict[str, str]] = []
    for m in history:
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        text = content.strip()
        if text:
            out.append({"role": role, "content": text})
    return out


def _truncate_by_chars(history: list[dict[str, str]], max_chars: int) -> list[dict[str, str]]:
    """按字符预算裁剪历史，**从最旧截起**（保留最近的）。

    指代消解最依赖近处的先行词："它"几乎总是指上一轮的对象。
    超预算时丢最旧的整条消息（不是截半条——半条历史比没有更误导）。
    """
    if max_chars <= 0:
        return history
    total = sum(len(m["content"]) for m in history)
    if total <= max_chars:
        return history
    kept: list[dict[str, str]] = []
    # 从最新往回收，装不下就停 —— 等价于"保留最近的、丢最旧的"
    for m in reversed(history):
        if total - sum(len(x["content"]) for x in kept) > max_chars and kept:
            break
        kept.append(m)
    kept.reverse()
    return kept


def _limit_rounds(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """只保留最近 `MAX_ROUNDS_IN_PROMPT` 轮（一轮 = 一问一答两条消息）。

    更早的轮次交给模型的 rolling_summary 去概括 —— 这就是
    "最近 3 轮原文 + 更早的摘要"裁剪策略的原文半边。
    摘要本身在模型返回的 rolling_summary 字段里，落盘与否由调用方决定。
    """
    max_messages = MAX_ROUNDS_IN_PROMPT * 2
    return history[-max_messages:]


# ---- 主函数 -----------------------------------------------------------------


async def rewrite(
    question: str,
    history: list[dict[str, str]] | None,
    *,
    max_history_chars: int = DEFAULT_MAX_HISTORY_CHARS,
) -> RewriteOutcome:
    """把多轮问题改写成独立问题。**首轮（空历史）不调模型，原样返回。**

    参数
    ----
    question
        用户当前问题原文。
    history
        对话历史（UI 的 message dict 或裁剪后的纯文本轮次均可，
        只读 role/content 两个字段）。
    max_history_chars
        历史字符预算（字符近似口径）。超出从最旧截起。

    返回
    ----
    `RewriteOutcome`。**永不抛异常、永不返回空 standalone_question**：
    模型调用失败时降级为原问（degraded=True），检索链路不能因改写层挂掉。
    """
    started = time.perf_counter()
    cleaned = _limit_rounds(_truncate_by_chars(_normalize_history(history), max_history_chars))

    # ---- 边界①：首轮不改写（不调模型）----
    # W7 的单轮回归题跑在"当前系统"上必须得到与昨天一致的结果。
    # 改写对首轮生效 = 单轮题被模型重写 = 回归红，且红得很隐蔽。
    if not cleaned:
        outcome = RewriteOutcome(standalone_question=question)
        _emit(_record(question, outcome, rounds=0, started=started))
        return outcome

    prompt = load_prompt("rewrite", question=question, history=cleaned)
    fallback = RewriteResult(
        standalone_question=question, needs_clarification=False, missing=[], rolling_summary=""
    )

    try:
        result = await structured_call(
            prompt,
            RewriteResult,
            max_repair=REWRITE_MAX_REPAIR,
            fallback=fallback,
        )
    except StructuredOutputError as exc:
        # ---- 边界②：降级用原问 ----
        # fallback 传了 structured_call 就不抛；这一支留给防御：
        # 万一上游把 fallback 参数拿掉，改写层也不能把检索带挂。
        logger.warning("改写调用失败，降级用原问：%s", exc)
        outcome = RewriteOutcome(standalone_question=question, degraded=True)
        _emit(_record(question, outcome, rounds=len(cleaned) // 2, started=started))
        return outcome

    # structured_call 的 fallback 降级是**静默**的（返回 fallback 对象本身）。
    # 检不出来，"降级"就只有 structured_call 日志里有，产物上完全不可见 ——
    # eval 归因时会把"改写没起作用"误读成"模型改写能力差"。
    # 用同一 fallback 对象（身份相等）判别降级，与 _exhausted 的返回语义对齐。
    outcome = (
        RewriteOutcome(standalone_question=question, degraded=True)
        if result is fallback
        else _to_outcome(question, result)
    )
    _emit(_record(question, outcome, rounds=len(cleaned) // 2, started=started))
    return outcome


def _to_outcome(question: str, result: RewriteResult) -> RewriteOutcome:
    """把模型输出归一成 RewriteOutcome，顺带执行两条校验层的规矩：

    1. `needs_clarification=false` 时 missing 强制清空 —— 模型爱顺手写
       "缺上下文"，不拦的话对照实验的分母就被污染。
    2. `standalone_question` 为空/纯空白时视作改写失败，降级用原问 ——
       空串进检索层是静默错误（检索会拿空 query 查全库）。
    """
    text = result.standalone_question.strip()
    if not text:
        logger.warning("改写输出 standalone_question 为空，降级用原问")
        return RewriteOutcome(standalone_question=question, degraded=True)

    if result.needs_clarification:
        # ---- 边界③：歧义。反问发生在交互路径；本层只标记并回填原问 ----
        return RewriteOutcome(
            standalone_question=question,
            needs_clarification=True,
            missing=list(result.missing),
            rewrote=False,
        )

    return RewriteOutcome(
        standalone_question=text,
        rewrote=text != question.strip(),
        rolling_summary=result.rolling_summary.strip(),
    )


def _record(
    question: str, outcome: RewriteOutcome, *, rounds: int, started: float
) -> dict[str, Any]:
    return {
        "question": question,
        "standalone_question": outcome.standalone_question,
        "needs_clarification": outcome.needs_clarification,
        "missing": list(outcome.missing),
        "rewrote": outcome.rewrote,
        "degraded": outcome.degraded,
        "rounds": rounds,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


__all__ = [
    "DEFAULT_MAX_HISTORY_CHARS",
    "MAX_ROUNDS_IN_PROMPT",
    "RewriteOutcome",
    "REWRITE_SINK",
    "rewrite",
    "set_rewrite_sink",
]
