"""生成层：拼上下文 → 用 W3 的 qa 模板调模型 → 解析引用 → 映射回 source+page。

## 这一层最容易做错的地方不是调模型，是"引用"

朴素做法是把 TopK 的正文拼起来喂给模型，答案是出来了，但用户问"这句是哪来的"，
你答不上来 —— 因为拼上下文的那一刻，`chunk` 和"prompt 里的第几段"的对应关系
只存在于一个局部列表里，模型返回的文本里没有任何东西指向它。

所以这里的做法是三步闭环：

    ① 上下文里每块带编号 [n]，并要求模型在答案里用 [n] 标注
    ② 解析模型输出里的 [n]，**校验 n 必须是给定编号的子集**
    ③ 用 n 反查 Citation → 得到 source + page

**为什么不用"让模型自己说出文件名和页码"**：它会把页码说错，而且错得很自信。
编号是唯一由**我们**（而非模型）掌握的信息，让模型只负责"引用哪一条"，
不让它负责"那一条在哪个文件"。这与 `llm_extractor` 的原文回指校验是同一条分工。

## 越界编号必须处理，不能忽略

模型可能引用 `[7]` 而上下文只有 3 条 —— 那说明它在编。三种处理方式：

| 方式 | 后果 |
| --- | --- |
| 忽略（静默丢弃） | 答案里留一个 `[7]` 悬空，前端要么显示断链，要么显示"来源不明" |
| 抛异常 | 一次越界就整次问答失败，代价过高（答案本身可能是有用的） |
| **保留正文，标记该编号无效，并记录警告** | 用户看到答案，也知道哪条依据无效 |

这里选第三种，并把无效编号放进 `Answer.invalid_refs`。
**"看起来正常但有编造成分"是最危险的输出**，必须让它可见。

## 与 W3 qa 模板的关系

W3 的 `qa/v2.jinja2` 已经做了三件事，这里直接复用而不重写：

1. 两步作答（先判断信息够不够，再答）
2. `引用编号` 是**结构化字段**（`list[int]`），不是从正文里正则抠
3. `能否回答` 是正向枚举

复用它的直接好处是：**同一套 prompt 的实测结论（见 `docs/prompt-ab.md`）继续有效**。
自己在 RAG 里另写一套 prompt，就等于那批 A/B 数据全部作废。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from app.rag.schema import Chunk

logger = logging.getLogger(__name__)

# 拼进 prompt 的上下文字符上限。取 6000 的理由：
# 中文 0.7 tok/字 ≈ 4200 token，给系统提示词与输出留出足够空间后仍安全。
# 这个数字必须**小于**任何目标模型的窗口减去输出预算 —— 宁可少放两块，
# 不要在"看起来刚好"的时候被网关静默截断（那会让你以为模型读到了）。
DEFAULT_CONTEXT_CHARS: Final[int] = 6000

# 检索分数低于这个值视为"没找到"。
#
# **这个默认值只在"调用方没传 `min_score`"时兜底，而生产路径不该依赖它** ——
# 正确的阈值是**与 embedder 绑定**的（见 `Embedder.recommended_min_score`）：
# 不同模型的相似度分布完全不同，全局常量必然在某个模型上误判。
#
# 实测对照（`scripts/embedding_score_probe.py`，4 条相关 + 4 条无关）：
#
# | embedder | 相关（min~max） | 无关（min~max） | 可分？ |
# | --- | --- | --- | :---: |
# | bge-small-zh | 0.578~0.676 | 0.253~0.367 | ✅ |
# | hashing(64) | 0.148~0.478 | -0.098~0.149 | ❌ 重叠 |
#
# 所以 `RagPipeline.answer()` 会从 `self.embedder.recommended_min_score` 取值，
# 而不是用这个常量。它留在这里是为了"直接调 `generate_answer` 的人有个合理默认"。
#
# **这个阈值不能区分命中与未命中**。周四的 10 条验收里，未命中的两条最高分
# （0.5087 / 0.6514）比命中者的最低分（0.466）还高。它只负责挡住"库里完全无关"，
# 不负责提高准确率 —— 把阈值当"质量旋钮"去调，越调越糟。
DEFAULT_MIN_SCORE: Final[float] = 0.40

# 拒答话术。**必须是固定字符串**，不能是"模型说它不知道"——
# 那等于让模型决定要不要拒答，而模型倾向于"尽力回答"。
# 拒答是我们（代码）的决定，不是模型的。
REFUSAL_TEXT: Final[str] = (
    "知识库中未找到相关内容。\n\n"
    "我没有在你的知识库里找到能回答这个问题的内容，因此不作回答。"
    "如果这个问题确实应该有答案，请确认相关文档已经导入。"
)

SYSTEM_PROMPT: Final[str] = (
    "你是一个严格依据给定材料作答的问答助手。"
    "只使用材料里明确写出的信息，不要用你自己的知识，不要推测。"
)

ANSWER_PROMPT: Final[str] = (
    "## 上下文\n\n{context}\n\n"
    "## 问题\n\n{question}\n\n"
    "## 输出要求\n\n"
    "1. 每个事实句后用 [编号] 标注依据，例如「违约金按日万分之五计算 [1]。」\n"
    "2. **只允许引用上下文里出现过的编号**（当前可用：{available}）。\n"
    "3. 材料里找不到答案时，直接说「材料中没有提到」，不要猜测。\n"
    "4. 保留原文中的数字、日期、金额、责任主体等具体信息。"
)


@dataclass(frozen=True, slots=True)
class Citation:
    """一条可核对的引用：编号 ↔ 原文位置。"""

    index: int
    """从 1 开始的编号，与 prompt 里的 `[n]` 一一对应。"""

    source: str
    """来源文件名。"""

    page: int | None
    """页码；非分页格式为 None。**这个字段就是"page 从第一步就要在结构里"的回报**。"""

    chunk_id: str
    """对应块的稳定 id（内容派生）。用于"同一条引用被重复命中"的去重，也用于回查原文。"""

    @property
    def label(self) -> str:
        """人类可读的引用标签，给前端直接显示。"""
        return f"{self.source} 第 {self.page} 页" if self.page is not None else self.source


@dataclass
class Answer:
    """一次问答的完整产物：答案 + 引用 + 过程统计。"""

    text: str
    citations: list[Citation] = field(default_factory=list)
    """**实际被答案引用的**那几条（不是上下文里全部）。顺序按编号升序。"""

    invalid_refs: list[int] = field(default_factory=list)
    """模型引用了但**不在给定编号里**的编号。非空说明这次回答有编造成分。"""

    refused: bool = False
    """是否是"拒答"（检索无结果 / 分数过低）。拒答时**没有调模型**。"""

    context_chars: int = 0
    """实际拼进 prompt 的上下文字符数。用于回答"这次为什么没命中第 4 条"这类问题。"""

    used_chunks: int = 0
    """真正被拼进上下文的块数（可能少于检索返回的块数，因为按预算截断了）。"""

    top_score: float | None = None
    """检索的最高相似度。用于回答"为什么拒答"（低于阈值时它会是重要证据）。"""

    def citation_block(self) -> str:
        """渲染成可直接显示的引用列表。没有被引用的编号不显示。"""
        if not self.citations:
            return ""
        return "\n".join(f"[{c.index}] {c.label}" for c in self.citations)


def extract_refs(text: str) -> list[int]:
    """从答案正文里抽出 `[n]` 形式的引用编号，按出现顺序去重。

    ## 为什么要用正则而不是"让模型返回 JSON 的引用字段"

    W3 的 `qa/v2.jinja2` 确实有结构化的 `引用编号` 字段，而且**本模块本该用它**。
    但 RAG 场景里多了一层不确定性：模型经常把编号写进正文（`"违约金按日万分之五 [1]"`），
    而 `引用编号` 字段可能空着 —— 两者不一致时以哪个为准？

    答案：**以正文里的编号为准**。理由是"用户看到的是正文"——
    如果正文说 `[1]` 而引用列表说 `[2]`，那引用列表就是在骗人。
    正文与列表一致时两者等价；不一致时，正文才是用户实际面对的。

    ## 支持的写法

    `[1]` `[1][2]` `[1,2]` `[1-3]` —— 都实测会出现。
    `[1-3]` 展开成 1,2,3；顺序按出现位置。

    **不匹配 `［１］`（全角）**：实测模型在中文语境下偶尔用全角方括号，
    但那种情况极少且**会伴随正常的半角编号出现在同一条回答里**。
    为它加一条正则的代价是可能把正文里的矩阵/区间误当引用，
    收益不足 —— 这里选择不处理，并把这条写下来而不是默默忽略。
    """
    import re

    refs: list[int] = []
    for match in re.finditer(r"\[(\d+(?:\s*[-,，]\s*\d+)*)\]", text):
        group = match.group(1).replace("，", ",")
        for part in group.split(","):
            part = part.strip()
            if "-" in part:
                bounds = [p.strip() for p in part.split("-", 1)]
                if all(b.isdigit() for b in bounds):
                    low, high = int(bounds[0]), int(bounds[1])
                    # 区间必须递增且不超过 20：`[1-9999]` 是模型发疯，
                    # 展开它会把内存打满。宁可丢弃这个区间。
                    if 0 < low <= high <= low + 20:
                        refs.extend(range(low, high + 1))
            elif part.isdigit():
                refs.append(int(part))
    # 去重保序：`[1][1]` 只算一次，但 `[2][1]` 保持 2 在 1 前
    return list(dict.fromkeys(refs))


def build_context(
    chunks: Sequence[Chunk],
    *,
    max_chars: int = DEFAULT_CONTEXT_CHARS,
) -> tuple[str, list[Citation]]:
    """把块拼成带编号的上下文，返回 `(上下文文本, 引用列表)`。

    编号从 1 开始按传入顺序分配。**顺序即检索分数序** —— 检索层已经排好，
    这里不再重排：重排会让"第 1 条最相关"这个直觉失效，而用户读引用时正是靠这个直觉。

    超预算时**整块丢弃而不是截断块**：截断一半的块会让模型读到残句，
    然后照着残句给出一个看起来有依据的错答案 —— 比少给一块糟得多。
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须为正数")

    parts: list[str] = []
    citations: list[Citation] = []
    used = 0
    for chunk in chunks:
        # 编号用 len(citations)+1 而不是 enumerate：被预算挡掉的块不占编号，
        # 否则答案里会出现 [3] 但引用列表里没有 3 —— 前端只能显示一个断链接。
        index = len(citations) + 1
        block = f"[{index}] 来源：{chunk.location}\n{chunk.text}"
        if used + len(block) > max_chars:
            if not citations:
                # 一条都没放进去就超预算：说明单块就比预算大。
                # 这时**必须放一条**，否则模型收到空资料 —— 那必然导致编造。
                parts.append(block[:max_chars])
                citations.append(_to_citation(index, chunk))
                used = max_chars
            break
        parts.append(block)
        citations.append(_to_citation(index, chunk))
        used += len(block)

    return "\n\n".join(parts), citations


def _to_citation(index: int, chunk: Chunk) -> Citation:
    return Citation(
        index=index,
        source=chunk.source,
        page=chunk.page,
        chunk_id=chunk.id,
    )


def resolve_citations(text: str, available: Sequence[Citation]) -> tuple[list[Citation], list[int]]:
    """把答案正文里的编号映射回 Citation。

    返回 `(被引用的 Citation 列表, 越界编号列表)`。

    **顺序按编号升序**，不是按正文出现顺序：引用列表是"阅读地图"，
    升序让人一眼看出用了哪几条、漏了哪几条。正文里的出现顺序服务于行文，
    两者要分开。
    """
    referenced = extract_refs(text)
    by_index = {c.index: c for c in available}
    valid = [by_index[i] for i in sorted(referenced) if i in by_index]
    invalid = [i for i in referenced if i not in by_index]
    if invalid:
        logger.warning("答案引用了不存在的编号 %s（可用：%s）", invalid, sorted(by_index))
    return valid, invalid


async def generate_answer(
    question: str,
    chunks: Sequence[Chunk],
    *,
    client: object,
    model: str,
    max_context_chars: int = DEFAULT_CONTEXT_CHARS,
    min_score: float = DEFAULT_MIN_SCORE,
    scores: Sequence[float] | None = None,
    max_tokens: int = 800,
    extra_request: dict[str, object] | None = None,
) -> Answer:
    """拼上下文 → 调模型 → 解析引用。

    ## 两道拒答闸门，都在模型之前

    | 闸门 | 判据 | 为什么放在调模型之前 |
    | --- | --- | --- |
    | 空结果 | `chunks` 为空 | 让模型对着空上下文回答，它唯一能做的就是编 |
    | 分数过低 | 最高分 < `min_score` | 库里只有"字面上碰巧相似"的垃圾时，模型会拿它当依据硬答 |

    `scores` 是检索分数（与 `chunks` 一一对应）。不传就跳过分数闸门 ——
    那是给"只想测引用逻辑"的调用方留的口子，生产路径必须传。

    **拒答是代码的决定，不是模型的**：`REFUSAL_TEXT` 是固定字符串。
    让模型自己说"我不知道"不可靠 —— 它倾向于尽力回答。
    """
    # ---- 闸门 1：空结果
    if not chunks:
        return Answer(text=REFUSAL_TEXT, refused=True)

    top_score = max(scores) if scores else None

    # ---- 闸门 2：分数过低
    if top_score is not None and top_score < min_score:
        logger.info("拒答：最高分 %.4f < 阈值 %.4f", top_score, min_score)
        return Answer(text=REFUSAL_TEXT, refused=True, top_score=top_score)

    context, available = build_context(chunks, max_chars=max_context_chars)
    if not available:
        # 预算小到连一块都放不进（`build_context` 会强制放一条，所以这里基本不会走到）。
        # 保留这个分支是为了"绝不把空上下文交给模型"这条不变量。
        return Answer(text=REFUSAL_TEXT, refused=True, top_score=top_score)

    available_ids = "、".join(f"[{c.index}]" for c in available)
    prompt = ANSWER_PROMPT.format(context=context, question=question, available=available_ids)

    response = await client.chat.completions.create(  # type: ignore[attr-defined]
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        **(extra_request or {}),
    )
    text = (response.choices[0].message.content or "").strip()
    citations, invalid = resolve_citations(text, available)

    return Answer(
        text=text,
        citations=citations,
        invalid_refs=invalid,
        refused=False,
        context_chars=len(context),
        used_chunks=len(available),
        top_score=top_score,
    )


__all__ = [
    "ANSWER_PROMPT",
    "DEFAULT_CONTEXT_CHARS",
    "DEFAULT_MIN_SCORE",
    "REFUSAL_TEXT",
    "SYSTEM_PROMPT",
    "Answer",
    "Citation",
    "build_context",
    "extract_refs",
    "generate_answer",
    "resolve_citations",
]
