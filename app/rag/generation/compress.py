"""上下文压缩：**只改送进 LLM 的文本，不改块的可检索身份**。

## 先定口径（这是今天最重要的一条设计决定）

任务书原话：

> 压缩后 gold 锚点在压缩文本里可能找不到 → W5/W7 的"命中判定"要能处理这种情况。
> 最稳的做法：压缩不改变 chunk 的可检索身份，只改变送进 LLM 的文本
> （即 Recall 仍在压缩前的 chunk 上算）。**这个设计决定要在今天就定，
> 否则周六的消融表口径会乱。**

本模块照这个做，并且**在代码结构上强制它**：

| 层 | 用什么 | 压缩影响它吗 |
| --- | --- | --- |
| 切分 / 索引 | `Chunk.text`（原始块） | ❌ 不影响 |
| 检索 / 精排 / Recall@5 / MRR | `Chunk.text` | ❌ 不影响 |
| **只在这里**：拼上下文送模型 | `compress_for_prompt()` 的输出 | ✅ 唯一入口 |

**没有任何函数会改写 `Chunk.text`。** 压缩产物是独立的 `CompressedContext`，
它携带"原始 chunk_id"用于引用回查 —— 所以 UI 点引用仍然能定位到原文。

## 为什么"Recall 在压缩前算"不是偷懒，是唯一正确做法

如果让 Recall 在压缩文本上算，就会发生这件事：

    gold 块被检索到了 ✅
    → 压缩删掉了含答案的那句
    → 锚点在新文本里找不到
    → **Recall 掉下来，看起来像"检索变差了"**
    → 人去调检索参数，而真正的原因是压缩

那是把**压缩的风险**伪装成**检索的退化**。两者该修的地方完全不同。

所以评估必须双轨（任务书也说"不能只报省了 40% token 就完事"）：

| 轨道 | 指标 | 由谁量 |
| --- | --- | --- |
| **成本** | token / 字符节省 % | 本模块直接给 |
| **检索** | Recall@5 / MRR | **在压缩前的 chunk 上算**（`scripts/` 现有链路，不受影响） |
| **答案质量** | 答对率、编造率 | 必须**抽检**（LLM 生成后再判），不能从 token 数推 |

## 压缩算法：句子级 + 保序 + 兜底

1. **拆句**：按 `SENTENCE_END` 切（复用 `chunking.quality` 的常量），
   中文标点与英文标点都覆盖
2. **打分**：每句与 query 的相关性。**用 BM25 那套字面 IDF，不调 LLM** ——
   理由：压缩是每次请求都要做的，用 LLM 打分会让延迟翻倍，
   而"这句和问题有没有共同实词"字面就够判了（这也是 BM25 在 W6 周一证明过的强项）
3. **选句**：按分数排序取前 N，**再按原顺序重组**（保序）
4. **兜底**：至少保留 `min_ratio`（默认 30%）或 `min_sentences`（默认 2 句），
   两者取更宽松的。**兜底不是可选项** —— 压缩过头的典型症状是
   "答案句被删 → 模型开始编"，而那在指标上表现为"答案质量下降但召回不变"，
   极难归因。

## 位置加权：越靠前的句子权重越高

小块（中位 77 token）通常只有 2~4 句，其中第一句往往是块首句
（切分器的边界保证，见 `chunking-quality.md`）。给前几句一个轻微加成，
是因为**它们携带"这一块在讲什么"的信息**，而删掉它会让剩下的句子失去上下文。

这不是"块首一定含答案"的假设 —— W6 周三实测过，答案可能在块尾。
位置加权只是**微弱**的（见 `POSITION_BONUS`），不足以压过强相关句。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.rag.schema import Chunk

#: 句子结束符。**与 `chunking.quality.SENTENCE_END` 保持一致** ——
#: 两处各写一份会漂移，而"压缩用的句边界"和"质量度量用的句边界"
#: 不一致时，两边报的数字就不可比。
SENTENCE_END: frozenset[str] = frozenset("。！？；.!?;")

#: 默认最低保留比例。**兜底规则**，见模块 docstring。
DEFAULT_MIN_RATIO: float = 0.30

#: 默认最低保留句数。与 `DEFAULT_MIN_RATIO` **取更宽松的那个**：
#:
#: 一块只有 3 句时，30% = 0.9 句 → 向上取整 1 句，可能删掉答案；
#: 而有 2 句的下限，至少保住"上下文 + 答案"的搭配。
DEFAULT_MIN_SENTENCES: int = 2

#: 位置加成：第 1 句 ×1.15、第 2 句 ×1.08、之后 ×1.0。
#:
#: **刻意做得很小**：W6 周三实测"答案可能在块尾"（截断实验里
#: head 策略在答案居末尾时得分 0.65），所以不能真的偏好块首。
#: 这个加成只够在"两句分数接近"时起决胜作用。
POSITION_BONUS: tuple[float, ...] = (1.15, 1.08)


@dataclass(frozen=True, slots=True)
class Sentence:
    """压缩的最小单位。"""

    text: str
    index: int
    """在原块内的序号（0-based）。**保序重组时按它排。**"""

    score: float = 0.0
    """与 query 的相关性（归一化前）。"""

    kept: bool = False


@dataclass(frozen=True, slots=True)
class CompressedChunk:
    """一块压缩后的结果。

    **`chunk_id` 指向原始块** —— 这是"压缩不改可检索身份"的落地：
    引用回查、去重、UI 定位全都用它，而不是压缩后的文本。
    """

    chunk_id: str
    text: str
    """压缩后的正文。**只用于拼 prompt，不写回 `Chunk`。**"""

    original_chars: int
    kept_sentences: int
    total_sentences: int
    floor_applied: bool = False
    """是否触发了兜底（说明"按分数选"本来会删得更多）。"""

    @property
    def ratio(self) -> float:
        """正文保留比例（按字符）。"""
        return len(self.text) / self.original_chars if self.original_chars else 0.0


@dataclass
class CompressedContext:
    """一次压缩的完整产物：给模型的文本 + 每条的成本账。"""

    chunks: list[CompressedChunk] = field(default_factory=list)

    @property
    def original_chars(self) -> int:
        return sum(item.original_chars for item in self.chunks)

    @property
    def compressed_chars(self) -> int:
        return sum(len(item.text) for item in self.chunks)

    @property
    def saved_ratio(self) -> float:
        """字符节省比例。**这是成本指标，不是质量指标。**"""
        if not self.original_chars:
            return 0.0
        return 1.0 - self.compressed_chars / self.original_chars

    @property
    def floor_count(self) -> int:
        """触发兜底的块数。**它不为 0 就说明"只按分数选"会删过头。**"""
        return sum(1 for item in self.chunks if item.floor_applied)

    def as_dict(self) -> dict[str, object]:
        return {
            "chunks": len(self.chunks),
            "original_chars": self.original_chars,
            "compressed_chars": self.compressed_chars,
            "saved_ratio": round(self.saved_ratio, 4),
            "floor_count": self.floor_count,
            "per_chunk": [
                {
                    "chunk_id": item.chunk_id,
                    "original_chars": item.original_chars,
                    "compressed_chars": len(item.text),
                    "kept_sentences": item.kept_sentences,
                    "total_sentences": item.total_sentences,
                    "floor_applied": item.floor_applied,
                }
                for item in self.chunks
            ],
        }


def split_sentences(text: str) -> list[str]:
    """把一段文本切成句子，**保留标点**。

    ## 为什么保留标点

    删掉句号后，重组出来的文本会变成 "……口径**从**发起请求……" 这种
    连在一起的残句，模型读到的是语法断裂的上下文 ——
    那会同时降低答案质量**和**让人看不出"这是压缩过的"。

    ## 为什么不用 `chunking.quality` 里的逻辑

    那边只判断"块首块尾是不是句边界"，不做切分。这里需要真正的切分器。
    但**共用 `SENTENCE_END` 常量**，保证"什么算句末"两处一致。

    ## 空文本

    返回空列表，不返回 `[""]` —— 空串进下面的打分会被当成"0 词句"，
    然后可能被保留，产生一个空块。源头返回空更干净。
    """
    if not text or not text.strip():
        return []
    sentences: list[str] = []
    buffer: list[str] = []
    for char in text:
        buffer.append(char)
        if char in SENTENCE_END:
            piece = "".join(buffer).strip()
            if piece:
                sentences.append(piece)
            buffer = []
    tail = "".join(buffer).strip()
    if tail:
        sentences.append(tail)
    return sentences


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:[-.][A-Za-z0-9_]+)*|[0-9]+|[一-鿿]+")


def _tokens(text: str) -> tuple[str, ...]:
    """极简切词：中文按字、英文按词、数字整体。

    ## 为什么不复用 `app.rag.bm25.tokenize()`（jieba）

    两个理由：

    1. **依赖方向**：`bm25` 是检索侧的包，让 `generation` 依赖它会把
       "生成层可以不装 jieba"这条边界打破（部署精简版就是无 jieba 场景）。
    2. **目的不同**：压缩只需要"有没有共同实词"这个粗判，
       不需要 BM25 那种精切。中文按字反而更稳 —— 术语切碎与否不影响
       "这个词出现过"的判断。
    """
    return tuple(match.group(0).lower() for match in _WORD.finditer(text))


def _idf_weights(documents: Sequence[Sequence[str]]) -> dict[str, float]:
    """在一小组句子内部算 IDF。

    ## 为什么是"组内 IDF"而不是全库 IDF

    压缩要回答的是"**在这几个候选块里**，哪句最能区分"。全库 IDF 会让
    "错误码"这类词被压得很低（全库常见），但在当前这 5 块里它可能正是关键。

    **更实际的理由**：全库 IDF 需要持有全库 token 统计，而压缩不该依赖索引
    （部署精简版没有 BM25 索引也能压缩）。组内 IDF 是自足的。
    """
    total = len(documents) or 1
    df: dict[str, int] = {}
    for doc in documents:
        for token in set(doc):
            df[token] = df.get(token, 0) + 1
    # 平滑：未出现的词给最高权重，避免除零
    return {token: (total + 1) / (count + 0.5) for token, count in df.items()}


def _score_sentences(
    sentences: list[str], query_tokens: set[str], idf: dict[str, float]
) -> list[float]:
    """给每句打分：**共同实词的 IDF 之和 / 句子长度平方根**。

    除以 `sqrt(长度)` 而不是 `长度`：长句天然容易命中更多词，
    但它不因此更"聚焦"。开方是标准的长度归一折中 ——
    完全归一（除以长度）会把一句精准的短句压得和长句一样。
    """
    scores: list[float] = []
    for position, sentence in enumerate(sentences):
        tokens = _tokens(sentence)
        if not tokens:
            scores.append(0.0)
            continue
        matched = sum(idf.get(token, 0.0) for token in set(tokens) & query_tokens)
        base = matched / (len(tokens) ** 0.5)
        bonus = POSITION_BONUS[position] if position < len(POSITION_BONUS) else 1.0
        scores.append(base * bonus)
    return scores


def compress_chunk(
    chunk: Chunk,
    query: str,
    *,
    keep_ratio: float = 0.5,
    min_ratio: float = DEFAULT_MIN_RATIO,
    min_sentences: int = DEFAULT_MIN_SENTENCES,
    idf: dict[str, float] | None = None,
) -> CompressedChunk:
    """压缩一块。**返回新对象，绝不修改 `chunk.text`。**

    ## 参数

    `keep_ratio`：按分数保留多少句（0~1）。实际保留句数
    会被 `min_ratio` / `min_sentences` 兜底，**兜底取更宽松的那个**。

    ## 兜底为什么必须存在

    压缩过头的典型症状：答案句被删 → 模型开始编。
    而那个症状在指标上表现为"**召回不变、答案质量下降**" ——
    召回是压缩前算的，所以它当然不变。**这个组合极难归因**：
    人会怀疑模型、怀疑 prompt，而真正的原因是压缩删了一句。
    """
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio 必须在 (0, 1]，收到 {keep_ratio}")

    sentences = split_sentences(chunk.text)
    if len(sentences) <= 1:
        # 单句块（或空块）：压缩没有意义，原样返回。
        # **不能返回空** —— 空块会让模型收到"这一条什么也没有"，
        # 而它明明是被检索到的相关结果。
        return CompressedChunk(
            chunk_id=chunk.id,
            text=chunk.text,
            original_chars=len(chunk.text),
            kept_sentences=len(sentences),
            total_sentences=len(sentences),
        )

    query_tokens = set(_tokens(query))
    weights = idf if idf is not None else _idf_weights([_tokens(s) for s in sentences])
    scores = _score_sentences(sentences, query_tokens, weights)

    target = max(1, round(len(sentences) * keep_ratio))
    floor_ratio_count = max(1, int(len(sentences) * min_ratio + 0.999))
    floor = max(floor_ratio_count, min(min_sentences, len(sentences)))
    floor_applied = target < floor
    keep_count = max(target, floor)

    ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
    kept = set(ranked[:keep_count])
    # **保序重组**：句子顺序是原文的论证顺序，打乱后模型读到的是碎片。
    text = " ".join(sentences[i] for i in range(len(sentences)) if i in kept)

    return CompressedChunk(
        chunk_id=chunk.id,
        text=text,
        original_chars=len(chunk.text),
        kept_sentences=len(kept),
        total_sentences=len(sentences),
        floor_applied=floor_applied,
    )


def compress_for_prompt(
    chunks: Sequence[Chunk],
    query: str,
    *,
    keep_ratio: float = 0.5,
    min_ratio: float = DEFAULT_MIN_RATIO,
    min_sentences: int = DEFAULT_MIN_SENTENCES,
    keep_scores: bool = True,
) -> CompressedContext:
    """把一批块压缩成"送进 prompt 的文本"。**这是唯一入口。**

    把 `idf` 在整个候选集上算一次（`keep_scores=True` 时），
    而不是每块各算各的 —— 跨块比较"哪句更区分"需要同一个尺度，
    否则 A 块的第一名和 B 块的第一名不可比。

    `keep_scores=False` 时按块内 IDF 算，用于"我只想压单个块"的场景
    （测试里用得最多，也避免了跨块 IDF 在小样本上的不稳定）。
    """
    if not chunks:
        return CompressedContext()
    shared_idf: dict[str, float] | None = None
    if keep_scores:
        corpus = [_tokens(" ".join(split_sentences(chunk.text))) for chunk in chunks]
        corpus = [tokens for tokens in corpus if tokens]
        if corpus:
            shared_idf = _idf_weights(corpus)
    return CompressedContext(
        chunks=[
            compress_chunk(
                chunk,
                query,
                keep_ratio=keep_ratio,
                min_ratio=min_ratio,
                min_sentences=min_sentences,
                idf=shared_idf,
            )
            for chunk in chunks
        ]
    )


__all__ = [
    "DEFAULT_MIN_RATIO",
    "DEFAULT_MIN_SENTENCES",
    "POSITION_BONUS",
    "SENTENCE_END",
    "CompressedChunk",
    "CompressedContext",
    "Sentence",
    "compress_chunk",
    "compress_for_prompt",
    "split_sentences",
]
