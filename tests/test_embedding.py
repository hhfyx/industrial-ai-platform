"""Embedding 层测试：向量数学 + 归一化契约 + query 指令前缀。

## 这个文件最重要的两条

### 一、归一化是**契约**，不是建议

`test_every_embedder_returns_normalized_vectors` 对**每个**实现都断言模长 == 1。
忘了归一化的后果不是报错，是"命中率莫名其妙地差一点"——
然后你会去调 chunk_size、top_k、换模型、改 prompt，把时间花在所有别的变量上。

实测证据（`scripts/embedding_norm_mechanism.json`）：余弦 0.60 的短向量
输给余弦 0.15 的长向量（模长差 18 倍），排序被彻底翻掉。

### 二、`embed` 与 `embed_query` 必须是两条路

`test_query_instruction_only_applies_to_queries` 断言：同一段文字，
走 `embed()` 和走 `embed_query()` 得到**不同的向量**（在 bge 上）。
若两者相同，说明前缀没生效 —— 而那不会有任何报错，只是命中率低一点。
"""

from __future__ import annotations

import pytest

from app.rag.embedding import (
    Embedder,
    HashingEmbedder,
    cosine,
    cosine_batch,
    describe_implementations,
    dot,
    get_embedder,
    l2_norm,
    l2_normalize,
)
from app.rag.embedding.bge import DEFAULT_QUERY_INSTRUCTION
from app.rag.rag_availability import bge_available
from app.rag.retrieval import Retriever
from app.rag.stores import InMemoryVectorStore

# ============================================================ 1. 手写余弦 vs numpy


def test_cosine_matches_numpy() -> None:
    """手写余弦必须与 numpy 算得**一致**（误差量级 1e-16）。

    这条测试的意义不在"余弦对不对"（公式很简单），在**为后面所有断言立基准**：
    如果 `cosine()` 本身错了，那归一化、排序、检索的断言全都在验一个错的基准。
    """
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)

    worst = 0.0
    for dim in (3, 8, 128, 512):
        a = rng.normal(size=dim)
        b = rng.normal(size=dim)
        expected = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        actual = cosine(a.tolist(), b.tolist())
        worst = max(worst, abs(actual - expected))
        assert actual == pytest.approx(expected, abs=1e-12), f"dim={dim}"

    assert worst < 1e-12, f"最大误差 {worst:.2e} 超过阈值"


def test_normalized_dot_equals_cosine() -> None:
    """**归一化之后，内积 == 余弦** —— 这是"必须先归一化"的全部理由。

    实测误差 1.7e-17，完全是浮点数噪声。所以生产代码可以用 `dot`
    （快，走矩阵乘法），前提是**先归一化**。
    """
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1)
    a = rng.normal(size=256).tolist()
    b = rng.normal(size=256).tolist()

    an, bn = l2_normalize(a), l2_normalize(b)
    assert dot(an, bn) == pytest.approx(cosine(a, b), abs=1e-12)


def test_unnormalized_dot_flips_ranking() -> None:
    """**忘了归一化的后果**：排序被向量长度污染。

    构造两个向量：
    - A：方向很像查询（余弦 0.60），但**很短**（模长 0.5）
    - B：方向只是有点相关（余弦 0.15），但**很长**（模长 9.0）

    按内积 B 赢，按余弦 A 赢 —— **同一个候选集，结论相反**。
    B 的模长是 A 的 18 倍，把"相关度低"这件事完全盖过去了。
    """
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(11)
    dim = 128
    query = rng.normal(size=dim)
    query /= np.linalg.norm(query)

    ortho = rng.normal(size=dim)
    ortho -= ortho.dot(query) * query
    ortho /= np.linalg.norm(ortho)

    short_relevant = query * 0.6 + ortho * 0.8
    short_relevant = short_relevant / np.linalg.norm(short_relevant) * 0.5

    long_irrelevant = query * 0.15 + ortho * 0.99
    long_irrelevant = long_irrelevant / np.linalg.norm(long_irrelevant) * 9.0

    q = query.tolist()
    a = short_relevant.tolist()
    b = long_irrelevant.tolist()

    # 内积说 B 更像，余弦说 A 更像 —— 这就是"没归一化"造成的排序污染
    assert dot(q, b) > dot(q, a), "内积应当被长向量带偏"
    assert cosine(q, a) > cosine(q, b), "余弦应当认为方向更接近的 A 更相关"

    # 归一化之后，内积也给出正确结论（与余弦一致）
    assert dot(l2_normalize(q), l2_normalize(a)) > dot(l2_normalize(q), l2_normalize(b))


def test_cosine_rejects_dimension_mismatch() -> None:
    """维度不一致必须抛：那是"索引与查询用了不同 embedder"的结构性错误。

    报 0.0 会把它伪装成"没命中"，让人在错误的层面查半天。
    """
    with pytest.raises(ValueError, match="维度不一致"):
        cosine([1.0, 0.0], [1.0, 0.0, 0.0])


def test_cosine_zero_vector_returns_zero() -> None:
    """零向量（空文本块）与任何查询的余弦都是 0：排到最后，但不炸掉检索。"""
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine([1.0, 1.0], [0.0, 0.0]) == 0.0


def test_cosine_batch_keeps_order() -> None:
    """批量结果顺序必须与入参严格一致 ——
    调用方按位置把分数配回 `Chunk`，顺序一变就是静默错配。"""
    query = [1.0, 0.0]
    matrix = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    scores = cosine_batch(query, matrix)
    assert len(scores) == 3
    assert scores[0] > scores[2] > scores[1]
    assert scores[0] == pytest.approx(1.0)


def test_l2_normalize_is_idempotent() -> None:
    """归一化两次与一次结果相同（误差内）—— 双保险不会放大误差。"""
    once = l2_normalize([3.0, 4.0])
    twice = l2_normalize(once)
    assert once == pytest.approx(twice, abs=1e-15)
    assert l2_norm(once) == pytest.approx(1.0)


# ============================================================ 2. 归一化契约（每个实现都要过）


def _all_embedders() -> list[tuple[str, Embedder]]:
    """所有可用实现的清单。**新增实现要加到这里** —— 否则契约测试不会覆盖它。"""
    implementers: list[tuple[str, Embedder]] = [("hashing", HashingEmbedder(dimension=64))]
    if bge_available():
        implementers.append(("bge-small-zh", get_embedder("bge-small-zh")))
    return implementers


@pytest.mark.parametrize(("name", "embedder"), _all_embedders())
def test_every_embedder_returns_normalized_vectors(name: str, embedder: Embedder) -> None:
    """**契约核心**：每个实现返回的向量必须已 L2 归一化（模长 == 1）。

    这是"绝不允许忘记归一化"这条规则的执行点。新加实现如果忘了，
    这条测试立刻红 —— 这是它存在的唯一目的。
    """
    vectors = embedder.embed(["违约金按合同总额的千分之一计算。", "质保期 12 个月。"])
    assert len(vectors) == 2
    for index, vector in enumerate(vectors):
        assert len(vector) == embedder.dimension, f"{name} 第 {index} 条维度不对"
        assert l2_norm(vector) == pytest.approx(1.0, abs=1e-6), f"{name} 第 {index} 条没归一化"


@pytest.mark.parametrize(("name", "embedder"), _all_embedders())
def test_every_embedder_keeps_input_order(name: str, embedder: Embedder) -> None:
    """顺序必须与入参一致（与 `test_cosine_batch_keeps_order` 同一类风险）。

    **容差取 1e-6，不是 1e-12**：实测 bge 的批量编码与逐条编码有
    **1e-7 量级的浮点差异**（`scripts/embedding_batch_consistency.json`：
    最大分量差 8.93e-08，余弦 1.0000000000，检索排序完全相同）。
    原因是 batch 内 padding 改变了注意力计算的浮点累加顺序。

    把容差拧到 1e-12 会让这条测试变成"偶尔因为浮点抖动变红"——
    那种测试最后一定会被人加 `@pytest.mark.flaky` 或者直接删掉。
    **容差要卡在"能发现顺序错位"和"不会被浮点噪声触发"之间**：
    顺序真的错了的话，余弦会掉到 0.99 以下，1e-6 这个量级根本拦不住它。
    """
    texts = ["甲条款", "乙条款", "丙条款"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3
    for index, text in enumerate(texts):
        alone = embedder.embed([text])[0]
        assert vectors[index] == pytest.approx(alone, abs=1e-6), (
            f"{name} 第 {index} 条与单独编码不一致，可能存在顺序错位"
        )


@pytest.mark.parametrize(("name", "embedder"), _all_embedders())
def test_every_embedder_handles_empty_input(name: str, embedder: Embedder) -> None:
    """空输入返回空列表，不抛 —— 空文档/空查询是正常状态。"""
    assert embedder.embed([]) == []
    assert embedder.embed_query([]) == []


# ============================================================ 3. query 指令前缀


def test_bge_instruction_text_is_the_official_one() -> None:
    """前缀文案必须与 bge-zh v1.5 的官方约定**逐字一致**。

    少一个字、多一个空格都会让向量偏离训练时的分布 —— 而它不会报错，
    只会让效果变差。所以这条只能靠断死字符串来守。
    """
    assert DEFAULT_QUERY_INSTRUCTION == "为这个句子生成表示以用于检索相关文章："


@pytest.mark.skipif(not bge_available(), reason="需要 torch + sentence-transformers + 模型")
def test_query_instruction_only_applies_to_queries() -> None:
    """**核心断言**：同一段文字，`embed()` 与 `embed_query()` 结果必须不同。

    若相同，说明前缀没生效。而前缀没生效**不会有任何报错**，
    只会让命中率低一点 —— 那最容易被归因到切分/top_k/模型不行上。
    """
    embedder = get_embedder("bge-small-zh")
    passage_vector = embedder.embed(["违约金怎么算"])[0]
    query_vector = embedder.embed_query(["违约金怎么算"])[0]
    similarity = cosine(passage_vector, query_vector)
    assert similarity < 0.999, f"加前缀前后向量几乎相同（余弦 {similarity:.4f}），前缀可能没生效"


@pytest.mark.skipif(not bge_available(), reason="需要 torch + sentence-transformers + 模型")
def test_query_instruction_is_idempotent() -> None:
    """已经带前缀的输入不会被重复加前缀。

    重复加前缀的后果是 `前缀为这个句子生成表示…问题` —— 只有在翻日志时
    才会发现，而那时你已经在怀疑模型了。
    """
    embedder = get_embedder("bge-small-zh")
    once = embedder.embed_query(["违约金怎么算"])[0]
    twice = embedder.embed_query([DEFAULT_QUERY_INSTRUCTION + "违约金怎么算"])[0]
    assert cosine(once, twice) == pytest.approx(1.0, abs=1e-6)


def test_hashing_embedder_does_not_add_instruction() -> None:
    """哈希实现**不加前缀**：前缀对字符 n-gram 毫无意义，加了还会污染查询向量
    （那 20 个字会占掉大量 n-gram 权重）。"""
    embedder = HashingEmbedder(dimension=64)
    assert embedder.embed_query(["违约金怎么算"]) == embedder.embed(["违约金怎么算"])


def test_hashing_embedder_exposes_instruction_attribute() -> None:
    """两个实现都要有 `query_instruction` 属性 —— 脚本靠它做 A/B 开关。

    缺这个属性会让脚本用 `getattr(..., "")` 静默降级成"永远不加前缀"，
    于是 A/B 的两组数据完全相同，而你会以为"前缀没用"。
    """
    assert hasattr(HashingEmbedder(), "query_instruction")
    assert hasattr(get_embedder("bge-small-zh"), "query_instruction")


# ============================================================ 4. 装配


def test_get_embedder_by_name() -> None:
    assert isinstance(get_embedder("hashing"), Embedder)
    with pytest.raises(ValueError, match="未知的 embedder"):
        get_embedder("openai-text-embedding-3")


def test_unknown_embedder_error_lists_available() -> None:
    """错误信息要列出可用实现 —— "未知的 embedder" 帮不了任何人。"""
    with pytest.raises(ValueError, match="hashing"):
        get_embedder("nope")


def test_describe_implementations_covers_both() -> None:
    described = describe_implementations()
    assert "hashing" in described
    assert "bge-small-zh" in described


def test_bge_embedder_can_be_constructed_without_loading() -> None:
    """构造**不该**加载模型：`import` 这个模块不该花 16 秒，也不该触发下载。

    这条和 `app/core/config.py` 的"import 不解析配置"是同一个原则 ——
    "不该在不需要的时候炸"，也不该在不需要的时候慢慢来。
    """
    embedder = get_embedder("bge-small-zh")
    assert embedder.is_loaded is False
    assert embedder.dimension == 512
    assert embedder.max_seq_length == 512


# ============================================================ 5. 接进检索链路


@pytest.mark.skipif(not bge_available(), reason="需要 torch + sentence-transformers + 模型")
def test_retriever_uses_query_side_embedding() -> None:
    """`Retriever` 必须调 `embed_query`（带前缀），不是 `embed`。

    用"记录调用了哪个方法"的方式测，而不是看命中率 ——
    命中率会被语料和模型噪声干扰，而"调了哪个方法"是确定的。
    """
    calls: list[str] = []

    class SpyEmbedder:
        name = "spy"
        dimension = 4
        query_instruction = ""

        def embed(self, texts):
            calls.append("embed")
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, queries):
            calls.append("embed_query")
            return [[1.0, 0.0, 0.0, 0.0] for _ in queries]

    store = InMemoryVectorStore()
    retriever = Retriever(embedder=SpyEmbedder(), store=store)
    retriever.retrieve("问题")

    assert calls == ["embed_query"], f"Retriever 应当只调 embed_query，实际调了 {calls}"


@pytest.mark.skipif(not bge_available(), reason="需要 torch + sentence-transformers + 模型")
def test_bge_end_to_end_retrieval_finds_semantic_match() -> None:
    """真模型跑一次端到端：**换说法的查询**要能找到原文。

    这是 `HashingEmbedder` 做不到的事（它只有字面相似）——
    `"钱什么时候能到账"` 与 `"十五个工作日内付款"` 没有一个字相同。
    """
    from app.rag.pipeline import RagPipeline

    pipeline = RagPipeline(embedder=get_embedder("bge-small-zh"), top_k=1)
    pipeline.index_text("第五条 付款计划：验收合格后十五个工作日内付款。", source="合同")
    pipeline.index_text("第六条 违约责任：按日万分之五支付违约金。", source="合同")
    pipeline.index_text("第十三条 质保期：验收合格之日起 12 个月。", source="合同")

    hits = pipeline.retrieve("钱什么时候能到账").hits
    assert hits, "检索不到"
    assert "付款" in hits[0].chunk.text, f"语义检索失败，命中：{hits[0].chunk.text[:30]}"
