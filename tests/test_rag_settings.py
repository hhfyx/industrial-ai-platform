"""`app/rag/settings.py` 的契约测试：**参数只有一处出处**。

这组测试守的是"同一参数出现两个答案"这个具体故障：
今天之前 `chunk_size` 同时存在于 `ChunkSize` 与 `ChunkingConfig`，
实测同一批 322 段文档，两条链路切出 351 与 353 块且逐块文本不同。
"""

from __future__ import annotations

import pytest

from app.rag.settings import (
    DEFAULT_OVERLAP_CHARS,
    DEFAULT_STRATEGY,
    DEFAULT_TOP_K,
    RagConfig,
    get_rag_config,
    reset_rag_config,
    set_rag_config,
)


@pytest.fixture(autouse=True)
def _clean_config_cache():  # noqa: ANN202
    """每条测试都在干净的进程级缓存上跑。"""
    reset_rag_config()
    yield
    reset_rag_config()


# --------------------------------------------------------------------------- 默认值


def test_defaults_match_the_module_constants() -> None:
    """`RagConfig()` 的默认值必须等于模块常量 —— 否则"默认值"有两个答案。"""
    config = RagConfig()
    assert config.top_k == DEFAULT_TOP_K
    assert config.strategy == DEFAULT_STRATEGY
    assert config.overlap_chars == DEFAULT_OVERLAP_CHARS


def test_default_top_k_reaches_five_and_is_single_sourced() -> None:
    """`retrieval.DEFAULT_TOP_K` 必须是 `settings.DEFAULT_TOP_K` 的别名。

    今天就发生过一次：两个模块各写一个 5，改一处另一处不动，
    "默认取几条"于是有两个答案，而没有任何地方写得出来哪个是真的。
    """
    from app.rag.retrieval import DEFAULT_TOP_K as from_retrieval
    from app.rag.settings import DEFAULT_TOP_K as from_settings

    assert from_retrieval == from_settings == 5


def test_default_strategy_matches_chunking_factory() -> None:
    """配置里的默认策略必须与切分工厂的默认策略一致。

    不一致的表现是"配置说 heading、索引按 recursive 切"，
    而两边都不报错，只是检索效果对不上。
    """
    from app.rag.chunking.factory import DEFAULT_STRATEGY as factory_default

    assert RagConfig().strategy == factory_default


# --------------------------------------------------------------------------- 校验


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"top_k": 0}, "top_k"),
        ({"top_k": -1}, "top_k"),
        ({"score_threshold": 1.5}, "score_threshold"),
        ({"score_threshold": -2.0}, "score_threshold"),
        ({"chunk_chars": 0}, "chunk_chars"),
        ({"overlap_chars": -1}, "overlap_chars"),
        ({"overlap_chars": 500, "chunk_chars": 500}, "overlap_chars"),
        ({"max_context_chars": 0}, "max_context_chars"),
    ],
)
def test_invalid_config_raises(changes: dict[str, object], match: str) -> None:
    """非法参数在**构造时**就炸，而不是在检索时。

    尤其是 `overlap >= chunk_chars`：它会让切分不收敛，
    而失效形式是"看起来跑通了，实际全文挤进一块"。
    """
    with pytest.raises(ValueError, match=match):
        RagConfig(**changes)  # type: ignore[arg-type]


def test_score_threshold_direction_is_documented_as_similarity() -> None:
    """阈值的方向必须是"越大越像"。

    这个方向搞反的后果不是报错，是**拒答逻辑整体反转**：
    `score < threshold` 会变成"越不像越该答"。所以定义域校验用的是
    余弦的 [-1, 1]，而 `-1` 是合法的（表示"什么都答"）。
    """
    assert RagConfig(score_threshold=-1.0).score_threshold == -1.0
    assert RagConfig(score_threshold=1.0).score_threshold == 1.0


# --------------------------------------------------------------------------- 派生


def test_chunking_config_translates_all_chunking_fields() -> None:
    """`chunking_config()` 是**唯一**的翻译点，两个字段都要过去。"""
    config = RagConfig(chunk_chars=800, overlap_chars=100)
    chunking = config.chunking_config()
    assert chunking.chunk_chars == 800
    assert chunking.overlap_chars == 100


def test_with_overrides_is_immutable() -> None:
    """派生新配置不能改原对象 —— 它是 frozen 的，改它等于全进程串味。"""
    base = RagConfig()
    derived = base.with_overrides(top_k=8)
    assert base.top_k == 5
    assert derived.top_k == 8
    assert derived.chunk_chars == base.chunk_chars


def test_as_dict_has_every_documented_parameter() -> None:
    """`as_dict()` 要覆盖任务点名的五个参数 + 上下文预算。

    少一个的表现是 `/health` 与实验产物里看不到那个参数，
    于是"这次跑的是哪套配置"又变成要靠回忆。
    """
    payload = RagConfig().as_dict()
    for key in (
        "top_k",
        "score_threshold",
        "strategy",
        "chunk_chars",
        "overlap_chars",
        "max_context_chars",
    ):
        assert key in payload, f"{key} 必须出现在 as_dict 里"


# --------------------------------------------------------------------------- 环境变量


def test_from_env_reads_every_parameter() -> None:
    """五个参数都能被 `RAG_*` 覆盖。"""
    config = RagConfig.from_env(
        {
            "RAG_TOP_K": "8",
            "RAG_SCORE_THRESHOLD": "0.55",
            "RAG_STRATEGY": "heading",
            "RAG_CHUNK_CHARS": "400",
            "RAG_OVERLAP_CHARS": "20",
            "RAG_MAX_CONTEXT_CHARS": "3000",
        }
    )
    assert config.top_k == 8
    assert config.score_threshold == 0.55
    assert config.strategy == "heading"
    assert config.chunk_chars == 400
    assert config.overlap_chars == 20
    assert config.max_context_chars == 3000


def test_from_env_ignores_blank_values() -> None:
    """空字符串视为"没设"，回退默认值。

    `.env` 里写 `RAG_TOP_K=` 得到的是空串，直接 `int("")` 会炸 ——
    而那看起来像"配置系统坏了"，实际只是留了一行空键。
    """
    config = RagConfig.from_env({"RAG_TOP_K": "  ", "RAG_STRATEGY": ""})
    assert config.top_k == DEFAULT_TOP_K
    assert config.strategy == DEFAULT_STRATEGY


@pytest.mark.parametrize(
    ("environ", "match"),
    [
        ({"RAG_TOP_K": "eight"}, "RAG_TOP_K"),
        ({"RAG_SCORE_THRESHOLD": "high"}, "RAG_SCORE_THRESHOLD"),
        ({"RAG_STRATEGY": "magic"}, "RAG_STRATEGY"),
        ({"RAG_CHUNK_CHARS": "big"}, "RAG_CHUNK_CHARS"),
    ],
)
def test_from_env_rejects_malformed_values(environ: dict[str, str], match: str) -> None:
    """**格式错误必须抛，不能回退默认值**。

    静默回退的表现是"配置改了但没效果"，而那会让人去查环境变量有没有生效 ——
    查错方向，且可能查很久。
    """
    with pytest.raises(ValueError, match=match):
        RagConfig.from_env(environ)


def test_from_env_validates_after_parsing() -> None:
    """解析成功但语义非法的组合仍要被 `__post_init__` 拦住。"""
    with pytest.raises(ValueError, match="overlap_chars"):
        RagConfig.from_env({"RAG_CHUNK_CHARS": "100", "RAG_OVERLAP_CHARS": "100"})


# --------------------------------------------------------------------------- 进程级单例


def test_get_rag_config_is_cached_until_reset() -> None:
    """单例语义：同一进程里拿到的是同一个对象，`reset()` 后才重新解析。"""
    first = get_rag_config()
    assert get_rag_config() is first
    reset_rag_config()
    assert get_rag_config() is not first


def test_set_rag_config_overrides_singleton() -> None:
    """测试/实验脚本要能显式注入一套配置。"""
    custom = RagConfig(top_k=9)
    set_rag_config(custom)
    assert get_rag_config() is custom


# --------------------------------------------------------------------------- 与 Retriever 的联动


def test_retriever_reads_top_k_from_config() -> None:
    """**任务的字面要求**：`retrieve()` 从 config 读参数，签名不变。"""
    from app.rag.embedding import HashingEmbedder
    from app.rag.retrieval import Retriever
    from app.rag.stores.memory import InMemoryVectorStore

    set_rag_config(RagConfig(top_k=7))
    retriever = Retriever(embedder=HashingEmbedder(dimension=8), store=InMemoryVectorStore())
    assert retriever.top_k == 7


def test_retriever_explicit_config_wins_over_global() -> None:
    """构造时显式给 `config`，就不看全局单例。"""
    from app.rag.embedding import HashingEmbedder
    from app.rag.retrieval import Retriever
    from app.rag.stores.memory import InMemoryVectorStore

    set_rag_config(RagConfig(top_k=2))
    retriever = Retriever(
        embedder=HashingEmbedder(dimension=8),
        store=InMemoryVectorStore(),
        config=RagConfig(top_k=9),
    )
    assert retriever.top_k == 9


def test_retriever_rejects_conflicting_top_k_and_config() -> None:
    """`top_k` 与 `config.top_k` 冲突时抛 —— 与 `top_k=/k=` 同一条规矩。"""
    from app.rag.embedding import HashingEmbedder
    from app.rag.retrieval import Retriever
    from app.rag.stores.memory import InMemoryVectorStore

    with pytest.raises(ValueError, match="不一致"):
        Retriever(
            embedder=HashingEmbedder(dimension=8),
            store=InMemoryVectorStore(),
            top_k=3,
            config=RagConfig(top_k=8),
        )


def test_retrieve_signature_unchanged() -> None:
    """**签名不变**是任务要求。这里断言老调用方式仍然全部可用。"""
    import inspect

    from app.rag.retrieval import Retriever

    params = inspect.signature(Retriever.retrieve).parameters
    assert "query" in params
    assert params["top_k"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["k"].kind is inspect.Parameter.KEYWORD_ONLY
    # `where` 是新增的，但必须有默认值 —— 老调用 `retrieve(q)` 不能坏。
    assert params["where"].default is None
