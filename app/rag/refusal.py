"""拒答：**阈值挡不住的那批，靠"证据形态"也挡不住 —— 所以这里不做假动作**。

## 今天测出来的两个硬事实，决定了这个模块**不**做什么

`scripts/threshold_scan.py` 在 20 条有答案 + 5 条无答案上实测：

    有答案分数 0.5412 ~ 0.7036
    无答案分数 0.3945 ~ 0.6577
              ↑ rf-02 这一条比 17/20 条真答案还高

**两组重叠，区间 [0.5412, 0.6577]。** 阈值 0.40 只挡住 1/5 条无答案；
而把阈值升到 0.60 才开始拦得多，代价是误拒 5/20 条真答案。

于是我试了第二条路：**证据形态**。假设是"无答案的检索会更散/更平"。
`scripts/refusal_demo.py` 实测把它否掉了：

| 判据 | 应命中（20 条） | 应拒答（5 条） | 可分？ |
| --- | --- | --- | :---: |
| `score_margin`（top1-top2） | 0.0012 ~ 0.1208 | 0.0041 ~ 0.0311 | ❌ 完全嵌套 |
| `distinct_sources`（文档数） | 2 / 3 / 4 份都有 | 2 / 4 份都有 | ❌ 完全重叠 |

组合判据的实际后果：

    纯阈值（昨天）          误拒 0/20   误放 4/5
    阈值 + 形态（今天）     误拒 13/20  误放 0/5     ← 65% 的真答案被拒

**用 13 条真答案换 4 条误放，是明确的倒退。** 所以形态判据被撤掉了 ——
它留在代码里会让每个用户多遇到 65% 的"知识库里没有"，而那只解决 5 条样本里的 4 条。

## 那这个模块还剩什么

剩下的是**阈值闸门本身 + 把拒答原因说清楚**。这两件事都是有价值的：

1. **空结果** 与 **低分** 确实是两种不同情况，用户该做的事也不同
   （前者查导入，后者换问法）。
2. 拒答必须**带诊断数字**（实际分数、阈值），否则"为什么拒了"只能靠重跑。
3. 拒答必须**不调模型** —— 让模型对着不相关材料回答，是它在编。

这三条是确定成立的，与"能不能区分命中"无关。

## 一条必须写进文档的限制

**当前阈值 0.40 只能挡住明显无关的内容。** 它挡不住
"语料里碰巧有词面相近的段子"这种情况 —— rf-02（React 路由）在
一个 LLM 工程笔记库里拿到 0.6577，因为 `SETUP.md` 里确实讲了
"Streamlit 前端配置"。那是**真实的语义相近**，只是不能回答那个问题。

这类问题的正确解法不在检索层：要么知识库补齐，要么在生成层让模型
明确说"材料里没有这回事"（`generation.ANSWER_PROMPT` 的第 3 条已经在做）。
**把检索层的阈值当成这类问题的解，是拿用户的真答案去换。**
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from app.rag.generation.base import REFUSAL_TEXT
from app.rag.schema import Chunk

# 低分与空结果的分界由阈值本身决定，所以这里不需要额外常量。
# 之前有两组形态判据（MAX_SOURCE_SPREAD / MIN_SCORE_MARGIN），
# 实测无法区分后已删除 —— 保留它们只会诱导别人再打开。
# 详细数据见模块 docstring 的那张表与 `scripts/refusal_demo.json`。


class RefusalKind(StrEnum):
    """三种结果。用 StrEnum 是为了它进 JSON 就是字符串（UI/CLI 直接可用）。"""

    ANSWER = "answer"
    LOW_SCORE = "low_score"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class Refusal:
    """一次拒答的完整说明。**给的是"为什么"和"接下来做什么"。**"""

    kind: RefusalKind
    text: str
    """给用户看的完整文案（`REFUSAL_TEXT` + 针对性建议）。"""

    advice: tuple[str, ...]
    """建议动作，分条。UI 渲染成按钮，CLI 逐行打印。"""

    top_score: float | None = None
    threshold: float | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
    """判定的原始数据（分数、命中数…）。**排查时必须能拿到**，
    否则"为什么这条被拒了"只能靠重跑。"""

    @property
    def refused(self) -> bool:
        return self.kind is not RefusalKind.ANSWER

    @property
    def headline(self) -> str:
        """第一行，给 UI 的标题用。**不含建议**，建议单独渲染。"""
        return REFUSAL_TEXT.split("\n")[0]


def _format(advice: Sequence[str]) -> str:
    """拼文案。**必须以 `REFUSAL_TEXT` 开头**，见 `refusal.py` 最后一节。"""
    body = "\n".join(f"· {line}" for line in advice)
    return f"{REFUSAL_TEXT}\n\n{body}"


def decide(
    chunks: Sequence[Chunk],
    scores: Sequence[float],
    *,
    threshold: float,
    top_k: int = 5,
) -> Refusal:
    """判定这次检索能不能交给模型。**只做两件事：空 → 低分。**

    ## 为什么不是三件（形态判据被撤掉的记录）

    今天加过一个 `weak_evidence`（文档太散 / 分差太小）。实测它误拒 13/20
    条真答案、只换来 4/5 条误放的改善。数据在 `scripts/refusal_demo.json`。

    **保留一条已知无效的判据比没有它更糟**：它会给人"我们在防幻觉"的错觉，
    而实际代价是每三个用户里有一个拿到本该有的答案。
    """
    if not chunks or not scores:
        advice = (
            "确认相关文档已经导入知识库",
            "换一种问法，用文档里可能出现的词",
            "如果知道是哪份文档，指定它再问一次",
        )
        return Refusal(
            kind=RefusalKind.EMPTY,
            text=_format(advice),
            advice=advice,
            threshold=threshold,
            diagnostics={"hits": 0},
        )

    top_score = max(scores)
    if top_score < threshold:
        advice = (
            f"最相关的一条相似度只有 {top_score:.3f}，低于可信阈值 {threshold:.2f}",
            "换个问法，或补充更具体的限定词",
            "缩小范围：先指定文档或章节再问",
        )
        return Refusal(
            kind=RefusalKind.LOW_SCORE,
            text=_format(advice),
            advice=advice[1:],
            top_score=top_score,
            threshold=threshold,
            diagnostics={
                "top_score": round(top_score, 4),
                "threshold": threshold,
                "hits": len(chunks),
                "scores": [round(score, 4) for score in scores],
            },
        )

    return Refusal(
        kind=RefusalKind.ANSWER,
        text="",
        advice=(),
        top_score=top_score,
        threshold=threshold,
        diagnostics={
            "hits": len(chunks),
            "distinct_sources": len({chunk.source for chunk in chunks}),
            "scores": [round(score, 4) for score in scores],
        },
    )


def describe_kinds() -> dict[str, str]:
    """三种结果的说明。给 UI 的图例与文档用，避免各处各写一份。"""
    return {
        RefusalKind.ANSWER.value: "检索到了可信依据，交给模型作答",
        RefusalKind.LOW_SCORE.value: "最相关的一条不够像，拒答并建议换问法",
        RefusalKind.EMPTY.value: "一条都没命中，拒答并建议确认文档已导入",
    }


__all__ = [
    "Refusal",
    "RefusalKind",
    "decide",
    "describe_kinds",
]
