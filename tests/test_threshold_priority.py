"""阈值优先级测试：**精标结果必须真的生效**。

## 这组测试的由来

W5 周日实测发现一个 bug：精标把 `DEFAULT_SCORE_THRESHOLD` 从 0.40 改成 0.575，
但 `QAEngine(embedder=bge).min_score` 仍然是 **0.40** —— **精标完全没生效**。

根因是原来的判断用"配置值 ≠ 模块默认值"来推断"用户有没有设过"：

    精标后 config.score_threshold == 0.575 == DEFAULT_SCORE_THRESHOLD
    → 判定"没设过" → 回退 embedder.recommended_min_score(0.40)

**用值相等推断意图，在默认值被改动时必然失效。**

这组测试锁三件事：
1. 不传阈值 → 用配置值（而不是 embedder 推荐值）
2. 显式传阈值 → 用它（不被"跨 embedder 保护"覆盖）
3. 无可分性的 embedder → 保护生效，但只对"没显式传"的情况
"""

from __future__ import annotations

import pytest

from app.rag.embedding import HashingEmbedder, get_embedder
from app.rag.query import QAEngine
from app.rag.settings import DEFAULT_SCORE_THRESHOLD, RagConfig, get_rag_config, reset_rag_config
from app.rag.stores.memory import InMemoryVectorStore

DIM = 16


@pytest.fixture(autouse=True)
def _clean_config():  # noqa: ANN202
    reset_rag_config()
    yield
    reset_rag_config()


def test_engine_uses_config_threshold_not_embedder_recommendation() -> None:
    """**精标值必须生效** —— 这是那个 bug 的直接回归测试。

    不传 `min_score` 时，应该用 `RagConfig.score_threshold`，
    而不是 `embedder.recommended_min_score`。

    修之前的错误行为：bge 下拿到 0.40（推荐值），而配置写的是 0.575。
    """
    from app.rag.settings import set_rag_config

    set_rag_config(RagConfig(score_threshold=0.62))
    engine = QAEngine(embedder=HashingEmbedder(dimension=DIM), store=InMemoryVectorStore())
    # hashing 的推荐值是 0.0，但配置是 0.62 —— 保护会把它归零（见下一个测试）。
    # 所以这里用 bge 等价物验证：只要"不是回退到推荐值"就说明配置生效了。
    bge_like = get_embedder("bge-small-zh")
    engine_bge = QAEngine(embedder=bge_like, store=InMemoryVectorStore())
    assert engine_bge.min_score == 0.62, f"应当用配置里的 0.62，实为 {engine_bge.min_score}"
    # hashing 被保护归零是预期的
    assert engine.min_score == 0.0


def test_default_config_threshold_is_the_documented_value() -> None:
    """默认阈值与文档一致（0.575）。

    这条看着像废话，但它是那个 bug 的**触发条件**：
    默认值一旦被改，旧逻辑就会把新默认值误判成"用户没设过"。
    """
    assert DEFAULT_SCORE_THRESHOLD == 0.575
    assert RagConfig().score_threshold == 0.575


def test_bge_gets_config_threshold() -> None:
    """bge 的推荐值是 0.40，但配置是 0.575 —— 应当用 0.575。"""
    engine = QAEngine(embedder=get_embedder("bge-small-zh"), store=InMemoryVectorStore())
    assert engine.min_score == DEFAULT_SCORE_THRESHOLD


def test_explicit_min_score_wins_over_protection() -> None:
    """**显式传的阈值不被"跨 embedder 保护"覆盖。**

    这是修那个 bug 时引入的第二个问题：保护写成无条件覆盖后，
    `min_score=0.99` 被抹成 0.0，导致三条拒答测试失败 ——
    "保护"变成了"越权"。

    显式传参是调用方的明确决定（测试要构造拒答场景、CLI 有 `--min-score`），
    不该被静默改掉。
    """
    engine = QAEngine(
        embedder=HashingEmbedder(dimension=DIM),
        store=InMemoryVectorStore(),
        min_score=0.99,
    )
    assert engine.min_score == 0.99


def test_unseparable_embedder_gets_gate_disabled() -> None:
    """hashing 的分数无可分性 → 保护把配置阈值归零。

    不加这道保护，配置里的 0.575 会让 hashing **拒掉所有问题**
    （它的相关分数只有 0~0.48），而那是静默的：用户只看到"知识库里没有"。
    """
    engine = QAEngine(embedder=HashingEmbedder(dimension=DIM), store=InMemoryVectorStore())
    assert engine.min_score == 0.0


def test_protection_does_not_touch_explicit_zero() -> None:
    """显式传 0.0（明确关闸门）也要保留 —— 它是"我有意关掉"，不是"没设"。"""
    engine = QAEngine(
        embedder=HashingEmbedder(dimension=DIM),
        store=InMemoryVectorStore(),
        min_score=0.0,
    )
    assert engine.min_score == 0.0


def test_qa_engine_does_not_mutate_global_config() -> None:
    """构造 `QAEngine` 不该改进程级配置。

    它会把覆盖项存进**自己的** `config`；如果就地改全局单例，
    第二个引擎会继承第一个的参数 —— 那种串味极难查。
    """
    before = get_rag_config().score_threshold
    QAEngine(
        embedder=HashingEmbedder(dimension=DIM),
        store=InMemoryVectorStore(),
        min_score=0.9,
        top_k=9,
    )
    after = get_rag_config().score_threshold
    assert before == after
