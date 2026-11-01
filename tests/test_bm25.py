"""BM25 分词与索引的一致性测试。

## 这个文件防的是**本周最隐蔽的那个 bug**

    "query 与索引必须用同一分词器"

它的症状不是报错，而是**BM25 召回率比预期低一点**。
而"低一点"会被归因到"BM25 不适合中文"或"k1 没调好"——
两个都不是真正的原因，且都无法证伪。

所以这里锁的不是"指标好看"，而是四件结构性的事：

| 锁什么 | 防什么 |
| --- | --- |
| 派生词整体切出 | 词典漏词（`同轴光` → `同轴` + `光`） |
| 词典文件只有词、没有段落 | 上一版把 `>` 说明文字当成词灌进词典树 |
| 索引 token 流 == 当前分词器切出的流 | 词典改了而索引没重建（最隐蔽的那个） |
| 空文档有占位、不产 `nan` | 排序里"某几块永远不出现" |
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from app.rag.bm25 import index as bm25_index  # noqa: E402
from app.rag.bm25.tokenize import DICT_PATH, describe, dict_words, tokenize  # noqa: E402
from app.rag.schema import Chunk  # noqa: E402


def _chunk(text: str, *, index: int = 0) -> Chunk:
    return Chunk(text=text, page=None, source="probe.md", index=index)


# --------------------------------------------------------------------------- 分词


def test_domain_terms_are_not_split() -> None:
    """领域词必须整体切出。**这是任务点名的那条验收项。**

    实测 `jieba` 默认把 `同轴光` 切成 `同轴` / `光` —— 于是它同时与
    含「同轴」的块、含「光」的块产生匹配。在 BM25 里这两类匹配都有分数，
    不是零，所以切碎了不会失败，只会让排名悄悄变差。
    """
    expected = {
        "同轴光": ("同轴光",),
        "幂等性": ("幂等性",),
        "首字延迟": ("首字延迟",),
        "光通量": ("光通量",),
    }
    for word, want in expected.items():
        assert tokenize(word) == want, f"{word!r} 被切成了 {tokenize(word)}"


def test_identifiers_survive_as_single_tokens() -> None:
    """型号/错误码/版本号是一个整体 token。

    `INVALID_INPUT` 若被切成 `invalid`/`input`，这两个词极常见、IDF 很低；
    而它作为一个整体标识符，IDF 本应很高。型号匹配是 BM25 相对向量检索的
    **结构性优势**，切碎等于把优势自己拆掉 —— 这是实测出来的，不是预防性设计。
    """
    cases = {
        "INVALID_INPUT 是什么错误码": "invalid_input",
        "bge-small-zh 的维度": "bge-small-zh",
        "v0.4.1 已打 tag": "v0.4.1",
        "越界率 0.19%": "0.19",
    }
    for text, expected_token in cases.items():
        assert expected_token in tokenize(text), f"{text!r} → {tokenize(text)}"


def test_stopwords_are_dropped() -> None:
    """虚词不该进 token 流。留着会占 TF 与长度归一，把有区分度的词稀释掉。"""
    tokens = tokenize("的 了 是 在 和 模型 检索")
    assert "的" not in tokens and "了" not in tokens
    assert "模型" in tokens and "检索" in tokens


def test_tokenizer_is_deterministic() -> None:
    """同一个输入切两次必须一样。**这是索引/查询两侧可比的最低前提。**"""
    text = "同轴光镜头的景深与 INVALID_INPUT 错误码"
    assert tokenize(text) == tokenize(text)


def test_dict_file_contains_words_only() -> None:
    """词典文件里**只允许注释和词**，不允许说明段落。

    上一版在文件末尾追加了一段 `>` 引用块作为说明，结果：
    1. `load_userdict` 把整段话当成一个"词"灌进词典树
    2. 自检把整段话当成"必须整体切出的一个词"，于是报出一堆莫名其妙的失败

    说明写进注释（`#` 开头），词只写词。
    """
    assert DICT_PATH.exists(), "领域词典文件不见了"
    for line in DICT_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert len(stripped.split()) <= 3, f"这一行像说明文字而不是词：{stripped[:40]!r}"
        assert not stripped.startswith((">", "|", "-", "*")), (
            f"词典里出现了说明段落标记：{stripped[:40]!r}"
        )


def test_dict_words_all_come_out_whole() -> None:
    """词典里每一个词都必须整体切出。**加错词会立刻红。**"""
    broken = {word: tokenize(word) for word in dict_words() if tokenize(word) != (word.lower(),)}
    assert broken == {}, f"这些词被切碎了：{broken}"


def test_describe_reports_a_fingerprint() -> None:
    """`describe()` 是索引指纹的来源，缺项就无法判断"该不该重建"。"""
    fingerprint = describe()
    assert fingerprint["tokenizer"]
    assert fingerprint["dict_words"] >= 1
    assert fingerprint["stopwords"] >= 1


# --------------------------------------------------------------------------- 索引


def test_index_token_stream_matches_current_tokenizer() -> None:
    """**落盘索引的 token 流必须与当前分词器切出的一致。**

    这是"query 与索引同一分词器"的机器化检查。词典改了而索引没重建时，
    两侧的词边界不同 —— 分数系统性偏低，且没有任何报错。
    """
    chunks = [_chunk("同轴光的景深控制与光通量"), _chunk("幂等性靠位置派生 id", index=1)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.pkl"
        bm25_index.build(chunks, path=path, force=True)
        loaded = bm25_index.load(path)
    assert loaded is not None
    for chunk, stream in zip(chunks, loaded.tokens, strict=True):
        assert stream == tokenize(chunk.text), "索引里的 token 流与当前分词器不一致"


def test_empty_token_stream_gets_a_placeholder() -> None:
    """token 流为空的块必须有占位符。

    否则整个 BM25 matrix 会出现 `nan`，而 `nan` 在排序里既不在前也不在后 ——
    表现是"某几块永远不被检索到"，看起来像过滤把它们删了。
    """
    chunks = [_chunk("的 了 是"), _chunk("真有内容", index=1)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.pkl"
        bm25_index.build(chunks, path=path, force=True)
        loaded = bm25_index.load(path)
    assert loaded is not None
    assert loaded.tokens[0] == (bm25_index.EMPTY_TOKEN,)
    assert loaded.tokens[1] != (bm25_index.EMPTY_TOKEN,)


def test_rebuild_is_reused_when_fingerprint_matches() -> None:
    """指纹相同就复用。**复用判据是分词器 + 块集合 + 参数，不是"文件存在"。**"""
    chunks = [_chunk("alpha beta"), _chunk("gamma delta", index=1)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.pkl"
        first = bm25_index.build(chunks, path=path, force=True)
        second = bm25_index.build(chunks, path=path)
    assert first.reused is False
    assert second.reused is True


def test_changed_chunks_force_a_rebuild() -> None:
    """块集合变了必须重建 —— 否则旧 token 流会配到新库上。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.pkl"
        bm25_index.build([_chunk("old content")], path=path, force=True)
        report = bm25_index.build([_chunk("new content"), _chunk("more", index=1)], path=path)
    assert report.reused is False
    assert report.chunks == 2


def test_load_returns_none_on_corrupt_file() -> None:
    """损坏文件返回 `None`，不抛。

    调用方要能表达"现在没有 BM25，降级到纯向量"—— 那是正常状态，不是错误。
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.pkl"
        path.write_bytes(b"not a pickle")
        assert bm25_index.load(path) is None
        assert bm25_index.load(Path(tmp) / "missing.pkl") is None


def test_search_returns_empty_for_query_without_tokens() -> None:
    """全停用词的 query 返回空，而不是"分数全 0 的随机块"。

    与向量侧"空 query 返回空"是同一条纪律：让问题被看见。
    """
    chunks = [_chunk("同轴光的景深")]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.pkl"
        bm25_index.build(chunks, path=path, force=True)
        loaded = bm25_index.load(path)
    assert loaded is not None
    assert loaded.search("的了是", top_k=5) == []


@pytest.mark.parametrize("query,expect_first_source", [("同轴光", "probe.md")])
def test_search_ranks_relevant_chunk_first(query: str, expect_first_source: str) -> None:
    chunks = [_chunk("同轴光的景深控制"), _chunk("无关内容", index=1)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.pkl"
        bm25_index.build(chunks, path=path, force=True)
        loaded = bm25_index.load(path)
    assert loaded is not None
    ranked = loaded.search(query, top_k=2)
    assert ranked
    assert ranked[0][0] == chunks[0].id
