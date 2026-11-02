"""上下文压缩的测试。**核心是锁住"不改变可检索身份"这条设计决定。**

## 这个文件锁什么

| 锁什么 | 防什么 |
| --- | --- |
| `compress_chunk` 不修改 `Chunk` | 压缩被写进索引 → Recall 跟着变 → 口径全乱 |
| 保序重组 | 句子被打乱 → 模型读到碎片 |
| 兜底规则生效 | 压缩过头删掉答案句 → 模型开始编 |
| 单句/空块不返回空 | 检索到的块被压成空串 → 模型收到"这条什么也没有" |
| `keep_ratio=1.0` 不丢内容 | 最基础的自检：不压缩时必须原样 |

## 为什么"不改 Chunk"值得单独一个测试

任务书的原文是：

> 压缩不改变 chunk 的可检索身份，只改变送进 LLM 的文本
> （即 Recall 仍在压缩前的 chunk 上算）。**这个设计决定要在今天就定，
> 否则周六的消融表口径会乱。**

在代码里，这条靠"`compress_chunk` 返回新对象、`Chunk` 是 frozen 的"来保证。
`test_compress_does_not_mutate_chunk` 就是把它钉住 ——
如果哪天有人为了省事改成 `chunk.text = 压缩后`，这条会立刻红。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from app.rag.generation.compress import (  # noqa: E402
    DEFAULT_MIN_RATIO,
    DEFAULT_MIN_SENTENCES,
    SENTENCE_END,
    CompressedContext,
    compress_chunk,
    compress_for_prompt,
    split_sentences,
)
from app.rag.schema import Chunk  # noqa: E402


def _chunk(text: str, *, index: int = 0, source: str = "api-spec.md") -> Chunk:
    return Chunk(text=text, page=None, source=source, index=index)


# --------------------------------------------------------------------------- 分句


def test_split_sentences_keeps_punctuation() -> None:
    """**标点必须留在句子里。**

    删掉句号后重组会得到"…口径**从**发起请求…"这种语法断裂的文本，
    那会同时降低答案质量、也让人看不出这是压缩过的。
    """
    text = "第一句。第二句！第三句？"
    assert split_sentences(text) == ["第一句。", "第二句！", "第三句？"]


def test_split_sentences_handles_english_punctuation() -> None:
    """英文标点也要覆盖 —— 块里混代码注释和英文时很常见。"""
    text = "First sentence. Second one! Third?"
    parts = split_sentences(text)
    assert len(parts) == 3


def test_split_sentences_keeps_tail_without_punctuation() -> None:
    """没有结尾标点的尾巴也要保留 —— 否则块尾内容会静默消失。"""
    text = "第一句。最后一句没有标点"
    assert split_sentences(text)[-1] == "最后一句没有标点"


def test_split_sentences_empty_returns_empty_list() -> None:
    """空文本返回 `[]` 而不是 `[""]` —— 空串会被当成"0 词句"并可能被保留。"""
    assert split_sentences("") == []
    assert split_sentences("   \n\t ") == []


def test_sentence_end_matches_quality_module() -> None:
    """**句边界常量必须与 `chunking.quality` 一致。**

    两处各写一份会漂移，而"压缩用的句边界"和"质量度量用的句边界"
    不一致时，两边报的数字就不可比。
    """
    from app.rag.chunking.quality import SENTENCE_END as QUALITY_END

    assert SENTENCE_END == QUALITY_END


# --------------------------------------------------------------------------- 核心：不改可检索身份


def test_compress_does_not_mutate_chunk() -> None:
    """**压缩不能修改 `Chunk`** —— 这是今天的设计决定，必须钉住。

    如果压缩写进了 `chunk.text`，会连环出事：

    1. `Chunk.id` 是 `sha256(source+page+正文)` 派生的 → id 变了
    2. 于是引用回查落空（UI 点 `[n]` 找不到原文）
    3. 而如果压缩发生在索引前，Recall 就会随压缩参数变化 → 消融表口径全乱
    """
    original = "正文超长返回 422 INVALID_INPUT。这条属于校验失败。今天天气很好。"
    chunk = _chunk(original)
    before_id = chunk.id
    before_text = chunk.text

    result = compress_chunk(chunk, "正文超长返回什么错误码？", keep_ratio=0.3)

    assert chunk.text == before_text, "Chunk.text 被改写了"
    assert chunk.id == before_id, "Chunk.id 变了 —— 引用回查会落空"
    assert result.chunk_id == before_id, "压缩结果必须指向原块 id"
    assert result.text != before_text, "压缩应当真的删了句子（否则测试无意义）"


def test_compressed_text_differs_but_identity_is_stable() -> None:
    """压缩文本可以变，但 `chunk_id` 不变 —— 这两件事必须解耦。"""
    chunk = _chunk("含答案的关键句。完全无关的填充句。另一句无关的。")
    result = compress_chunk(chunk, "含答案的关键句是什么", keep_ratio=0.34)
    assert result.chunk_id == chunk.id
    assert len(result.text) <= len(chunk.text)


# --------------------------------------------------------------------------- 保序


def test_sentences_keep_original_order() -> None:
    """**保序重组**：句子顺序是原文的论证顺序，打乱后模型读到的是碎片。

    这里用一个"分数顺序与原文顺序相反"的构造：最后一句含 query 关键词，
    它分数最高、会被保留，但**在输出里仍应排在后面**。
    """
    text = "第一句讲别的事。第二句也讲别的事。最后一句才是 422 错误码。"
    chunk = _chunk(text)
    result = compress_chunk(chunk, "422 错误码", keep_ratio=0.4)

    kept = [s for s in split_sentences(text) if s in result.text]
    assert kept, "至少要保留一句"
    assert kept == sorted(kept, key=text.index), "保留的句子必须保持原文顺序"


def test_high_score_sentence_is_kept() -> None:
    """与 query 强相关的句子必须被保留 —— 否则压缩就失去意义。"""
    text = "今天天气很好。正文超长返回 422 INVALID_INPUT。无关的一句话。"
    chunk = _chunk(text)
    result = compress_chunk(chunk, "正文超长返回什么错误码", keep_ratio=0.34)
    assert "422" in result.text or "INVALID_INPUT" in result.text


# --------------------------------------------------------------------------- 兜底规则


def test_min_sentences_floor_applies() -> None:
    """**兜底：至少保留 `min_sentences` 句。**

    压缩过头的典型症状是"答案句被删 → 模型开始编"，
    而它在指标上表现为「召回不变、答案质量下降」—— 极难归因。
    """
    text = "第一句。第二句。第三句。第四句。第五句。"
    chunk = _chunk(text)
    # keep_ratio=0.2 本意只留 1 句，但兜底要求至少 2 句
    result = compress_chunk(chunk, "完全无关的查询词", keep_ratio=0.2, min_sentences=2)
    assert result.kept_sentences >= 2
    assert result.floor_applied is True, "触发兜底时必须标记出来"


def test_min_ratio_floor_applies() -> None:
    """按比例兜底：至少保留 `min_ratio` 的句数（向上取整）。"""
    text = "。".join(f"第{i}句" for i in range(1, 11)) + "。"
    chunk = _chunk(text)
    result = compress_chunk(chunk, "无关", keep_ratio=0.1, min_ratio=0.3, min_sentences=1)
    assert result.kept_sentences >= 3, "10 句的 30% = 3 句"
    assert result.floor_applied is True


def test_floor_not_applied_when_not_needed() -> None:
    """keep_ratio 本身已经高于兜底时不标记 `floor_applied`。

    这样"兜底频率"才是个有意义的信号 —— 一直为 True 就说明兜底在频繁救场。
    """
    text = "。".join(f"第{i}句" for i in range(1, 11)) + "。"
    chunk = _chunk(text)
    result = compress_chunk(chunk, "第1句", keep_ratio=0.9)
    assert result.floor_applied is False
    assert result.kept_sentences >= 9


# --------------------------------------------------------------------------- 边界


def test_single_sentence_chunk_is_returned_as_is() -> None:
    """单句块压缩没有意义 —— **必须原样返回，不能返回空**。

    返回空的后果：模型收到"这一条什么也没有"，而它明明是被检索到的相关结果。
    """
    chunk = _chunk("只有一句话没有句号的块")
    result = compress_chunk(chunk, "任何查询", keep_ratio=0.1)
    assert result.text == chunk.text
    assert result.total_sentences <= 1


def test_empty_chunk_does_not_crash() -> None:
    """空块不崩 —— 走单句分支原样返回。"""
    chunk = _chunk("")
    result = compress_chunk(chunk, "查询", keep_ratio=0.5)
    assert result.text == ""


def test_keep_ratio_one_keeps_everything() -> None:
    """`keep_ratio=1.0`（不压缩）必须原样保留全部内容。

    这是最基础的自检 —— 如果它丢了内容，那"压缩"从一开始就是坏的。
    """
    text = "第一句。第二句。第三句。"
    chunk = _chunk(text)
    result = compress_chunk(chunk, "第二句", keep_ratio=1.0)
    for sentence in split_sentences(text):
        assert sentence in result.text, f"不压缩时丢了句子：{sentence}"


def test_invalid_keep_ratio_raises() -> None:
    """`keep_ratio` 越界必须抛，不能静默修正。"""
    chunk = _chunk("第一句。第二句。")
    with pytest.raises(ValueError, match="keep_ratio 必须在"):
        compress_chunk(chunk, "查询", keep_ratio=0.0)
    with pytest.raises(ValueError, match="keep_ratio 必须在"):
        compress_chunk(chunk, "查询", keep_ratio=1.5)


# --------------------------------------------------------------------------- 批量与统计


def test_compress_for_prompt_reports_saved_ratio() -> None:
    """成本指标必须能算出来 —— **但它是成本指标，不是质量指标**。"""
    chunks = [
        _chunk("正文超长返回 422。今天天气很好。无关句。", index=0),
        _chunk("TTFT 是首字延迟。无关的填充内容。", index=1),
    ]
    ctx = compress_for_prompt(chunks, "返回什么错误码", keep_ratio=0.5)
    assert isinstance(ctx, CompressedContext)
    assert ctx.original_chars > ctx.compressed_chars
    assert 0.0 < ctx.saved_ratio < 1.0
    assert len(ctx.chunks) == 2


def test_compress_for_prompt_empty_input() -> None:
    """空输入返回空上下文，不除零。"""
    ctx = compress_for_prompt([], "查询")
    assert ctx.chunks == []
    assert ctx.saved_ratio == 0.0


def test_floor_count_is_reported() -> None:
    """兜底次数必须可观测 —— 它是"压缩有没有过头"的直接信号。"""
    chunks = [_chunk("第一句。第二句。第三句。第四句。", index=0)]
    ctx = compress_for_prompt(chunks, "无关查询", keep_ratio=0.25, min_sentences=3)
    assert ctx.floor_count == 1


def test_default_floors_are_conservative() -> None:
    """默认兜底参数要"保守"：30% 且至少 2 句。

    这两个数不是拍出来的：任务书明确要求
    "设最低保留比例（如至少留 30% 或至少留 2 句）"。
    """
    assert DEFAULT_MIN_RATIO == 0.30
    assert DEFAULT_MIN_SENTENCES == 2


# --------------------------------------------------------------------------- 与 build_context 的集成


def test_build_context_keeps_citation_ids_with_compressed_text() -> None:
    """**注入压缩文本后，引用 `chunk_id` 必须不变。**

    这是"压缩不改可检索身份"在集成层的体现：
    引用回查走原块，所以"压缩后仍能点开看原文"自动成立。
    """
    from app.rag.generation.base import build_context

    chunks = [
        _chunk("正文超长返回 422。天气很好。无关句。", index=0),
        _chunk("TTFT 是首字延迟。填充。", index=1),
    ]
    ctx = compress_for_prompt(chunks, "错误码", keep_ratio=0.5)
    texts = {item.chunk_id: item.text for item in ctx.chunks}

    _raw_text, raw_citations = build_context(chunks)
    compressed_text, compressed_citations = build_context(chunks, texts=texts)

    assert [c.chunk_id for c in raw_citations] == [c.chunk_id for c in compressed_citations]
    assert len(compressed_text) <= len(_raw_text)
