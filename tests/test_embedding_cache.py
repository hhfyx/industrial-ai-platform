"""`embedding/cache.py` 的契约测试。**全部离线**，不碰 bge / torch。

缓存的失败方式是**静默**的（命中错向量、位置错位、不开事务），
所以这里的每条测试都对着一个"看起来能跑但结果是错的"的场景。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.rag.embedding.cache import (
    CachedEmbedder,
    EmbeddingCache,
    cache_key,
    default_cache_path,
)
from app.rag.embedding.hashing import HashingEmbedder


class CountingEmbedder(HashingEmbedder):
    """记下每次真正被要求编码的文本。**用它区分"算了"和"查到了"**。

    `HashingEmbedder` 是纯函数且离线，所以它是最好的假实现：
    它能给出稳定向量，又能让我们数清"到底编码了几条"。
    """

    def __init__(self, dimension: int = 64) -> None:
        super().__init__(dimension=dimension)
        self.calls: list[list[str]] = []

    def embed(self, texts):  # noqa: ANN001, ANN201
        self.calls.append(list(texts))
        return super().embed(texts)


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "cache.sqlite"


# --------------------------------------------------------------------------- 键


def test_cache_key_depends_on_all_three_fields() -> None:
    """三个字段任一变化都必须换键。**少了任何一个，换模型后会静默命中旧向量。**"""
    base = cache_key(model="bge", dimension=512, text="付款")
    assert cache_key(model="bge-m3", dimension=512, text="付款") != base
    assert cache_key(model="bge", dimension=768, text="付款") != base
    assert cache_key(model="bge", dimension=512, text="验收") != base
    assert cache_key(model="bge", dimension=512, text="付款") == base


def test_cache_key_separator_prevents_boundary_collision() -> None:
    """`("ab","c")` 与 `("a","bc")` 不能撞同一个键。

    这是 `schema._digest` 踩过的坑：不带分隔符拼接时，
    文本里的字符会跨字段"借位"，两条不同的输入算出同一个摘要。
    """
    left = cache_key(model="ab", dimension=1, text="c")
    right = cache_key(model="a", dimension=1, text="bc")
    assert left != right


def test_cache_key_is_stable_across_calls() -> None:
    """同输入同键。带随机盐（`hash()`）会让命中率恒为 0 —— 那看起来像"缓存没生效"。"""
    assert cache_key(model="m", dimension=8, text="同一条") == cache_key(
        model="m", dimension=8, text="同一条"
    )


def test_default_cache_path_is_under_data_and_model_specific(tmp_path: Path) -> None:
    """落盘在 `data/embedding-cache/` 下，且按模型名分文件。

    放进 `~/.cache` 是 `ChromaStore` 明确拒绝过的做法（换机器就丢）。
    按模型分文件是为了"换模型不需要清缓存，回滚还能命中"。
    """
    path = default_cache_path("BAAI/bge-small-zh-v1.5")
    parts = path.parts
    assert "data" in parts
    assert "embedding-cache" in parts
    # `/` 必须被替换掉，否则会变成一个多层目录
    assert path.name.endswith(".sqlite")
    assert "/" not in path.name


# --------------------------------------------------------------------------- 读写


def test_get_many_keeps_positions_for_misses(cache_path: Path) -> None:
    """未命中位置必须是 `None` 占位，**不能压缩列表**。

    返回"只含命中项"的列表，调用方再 zip 回原文，向量就会与块**逐位错开** ——
    那种错不报错，只让检索结果诡异。
    """
    cache = EmbeddingCache(cache_path, dimension=3)
    cache.put_many([("k1", [1.0, 0.0, 0.0])])
    out = cache.get_many(["k1", "missing", "k1"])
    assert out[0] is not None
    assert out[1] is None
    assert out[2] is not None
    cache.close()


def test_dimension_mismatch_is_treated_as_miss(cache_path: Path) -> None:
    """维度对不上时按未命中处理，不硬解。

    硬解出来长度不对的向量会一路传到 `store.upsert`，在那里炸成"维度不匹配" ——
    报错点离真正的原因（换了模型）很远。
    """
    cache = EmbeddingCache(cache_path, dimension=3)
    cache.put_many([("k1", [1.0, 0.0, 0.0])])
    cache.close()

    other = EmbeddingCache(cache_path, dimension=4)
    assert other.get_many(["k1"]) == [None]
    other.close()


def test_put_many_skips_wrong_dimension(cache_path: Path) -> None:
    """维度错的向量不写。**宁可不缓存（下次重算），也不要存一条解不出来的。**"""
    cache = EmbeddingCache(cache_path, dimension=3)
    written = cache.put_many([("bad", [1.0, 2.0])])
    assert written == 0
    assert cache.get_many(["bad"]) == [None]
    cache.close()


def test_empty_inputs_are_noops(cache_path: Path) -> None:
    """空输入返回空，不抛（与 `Embedder.embed` 同口径）。"""
    cache = EmbeddingCache(cache_path, dimension=3)
    assert cache.get_many([]) == []
    assert cache.put_many([]) == 0
    cache.close()


def test_stats_counts_hits_and_misses(cache_path: Path) -> None:
    """命中统计必须有数字 ——"缓存有没有起作用"不能靠感觉。"""
    cache = EmbeddingCache(cache_path, dimension=3)
    cache.put_many([("k1", [1.0, 0.0, 0.0])])
    cache.get_many(["k1", "nope"])
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["entries"] == 1
    cache.close()


def test_survives_reopen(cache_path: Path) -> None:
    """换一个连接还能读到 —— 这是"隔天重跑仍然命中"的前提。"""
    first = EmbeddingCache(cache_path, dimension=3)
    first.put_many([("k1", [0.6, 0.8, 0.0])])
    first.close()

    second = EmbeddingCache(cache_path, dimension=3)
    values = second.get_many(["k1"])
    assert values[0] is not None
    assert values[0] == pytest.approx([0.6, 0.8, 0.0])
    second.close()


def test_corrupt_file_is_not_silently_accepted(tmp_path: Path) -> None:
    """库文件损坏时应炸（而不是返回垃圾）。

    这里刻意**不做"坏了就当没有"**：静默重建缓存会把磁盘错误伪装成
    "今天没命中"，而那会让人去查键设计 —— 查错方向。
    """
    bad = tmp_path / "bad.sqlite"
    bad.write_bytes(b"this is not a sqlite database")
    with pytest.raises(sqlite3.DatabaseError):
        EmbeddingCache(bad, dimension=3)


# --------------------------------------------------------------------------- CachedEmbedder


def test_cached_embedder_encodes_each_distinct_text_once(cache_path: Path) -> None:
    """同一批里的重复文本只编码一次。

    重叠切分很常见：相邻块共享 50 字，重复段落也会产生相同文本。
    不去重就白付编码成本。
    """
    inner = CountingEmbedder(dimension=8)
    with CachedEmbedder(inner, cache_path=cache_path) as cached:
        cached.embed(["甲", "乙", "甲"])
    assert inner.calls == [["甲", "乙"]]


def test_cached_embedder_second_call_skips_model(cache_path: Path) -> None:
    """第二次调用完全不碰内层 embedder。**这就是"提速"的机制。**"""
    inner = CountingEmbedder(dimension=8)
    with CachedEmbedder(inner, cache_path=cache_path) as cached:
        cached.embed(["甲", "乙"])
        assert len(inner.calls) == 1
        cached.embed(["甲", "乙"])
        assert len(inner.calls) == 1  # 没有新增调用
    assert cached.stats()["hits"] == 2


def test_cached_embedder_preserves_order_and_values(cache_path: Path) -> None:
    """返回顺序与入参严格一致，且与不缓存的实现逐位相同（1e-6 内）。"""
    inner = CountingEmbedder(dimension=8)
    texts = ["甲", "乙", "丙", "甲", "丁"]
    expected = inner.embed(texts)
    with CachedEmbedder(inner, cache_path=cache_path) as cached:
        actual = cached.embed(texts)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got == pytest.approx(want, abs=1e-6)


def test_cached_embedder_returns_normalized_vectors(cache_path: Path) -> None:
    """缓存回读的向量也必须归一化。

    缓存里可能存着旧版本写的向量；"未归一化"的后果是长向量天然更相近
    （见 `vector.py`），而这不会报错。
    """
    from app.rag.embedding.vector import l2_norm

    inner = CountingEmbedder(dimension=8)
    with CachedEmbedder(inner, cache_path=cache_path) as cached:
        cached.embed(["甲"])
        second = cached.embed(["甲"])[0]
    assert l2_norm(second) == pytest.approx(1.0, abs=1e-6)


def test_cached_embedder_exposes_inner_metadata(cache_path: Path) -> None:
    """`name` / `dimension` / `recommended_min_score` 必须透传。

    缓存若改了 `name`，索引元数据就会记成另一个模型 ——
    "这批向量是谁产的"从此失去意义。
    """
    inner = CountingEmbedder(dimension=8)
    with CachedEmbedder(inner, cache_path=cache_path) as cached:
        assert cached.name == inner.name
        assert cached.dimension == inner.dimension
        assert cached.recommended_min_score == inner.recommended_min_score


def test_cached_embedder_query_side_bypasses_cache(cache_path: Path) -> None:
    """查询侧直接透传，**不读不写缓存**。

    理由见 `cache.py` 模块 docstring：缓存查询向量就得复制一份前缀逻辑，
    而那份逻辑一旦与 `bge.py` 漂移，症状是"前缀有时加有时不加"，没有报错。
    """
    inner = CountingEmbedder(dimension=8)
    with CachedEmbedder(inner, cache_path=cache_path) as cached:
        first = cached.embed_query(["问题"])
        second = cached.embed_query(["问题"])
    assert len(inner.calls) == 2  # 每次都真的调了内层
    assert first == second
    assert cached.stats()["entries"] == 0


def test_cached_embedder_handle_empty_input(cache_path: Path) -> None:
    inner = CountingEmbedder(dimension=8)
    with CachedEmbedder(inner, cache_path=cache_path) as cached:
        assert cached.embed([]) == []
        assert cached.embed_query([]) == []
    assert inner.calls == []


def test_cached_embedder_is_transparent_to_embedder_protocol(cache_path: Path) -> None:
    """包装后仍满足 `Embedder` 协议 —— 上层不该需要知道有没有缓存。"""
    from app.rag.embedding import Embedder

    with CachedEmbedder(CountingEmbedder(dimension=8), cache_path=cache_path) as cached:
        assert isinstance(cached, Embedder)
