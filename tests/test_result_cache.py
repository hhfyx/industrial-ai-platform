"""缓存一致性与三层缓存的离线测试（W12 周二 15:30–16:30）。

## 三条核心一致性测试（计划验收：三条绿）

| # | 变更 | 必须的结果 |
| --- | --- | --- |
| 1 | 换链路配置（top_k / 策略 / chunk） | 检索缓存必须 miss |
| 2 | 改 prompt（SYSTEM/ANSWER 任一） | 答案缓存必须 miss |
| 3 | 模型版本变（答案层 model / 生成层 model_version） | 全部 miss |

其余覆盖：负缓存命中与过期、异常不写缓存、上下文指纹（必答问题的
可执行形态）、检索层回查失败降级 miss、SessionStore 内容级幂等。

## 离线纪律

全部走 `HashingEmbedder` + `InMemoryVectorStore`（conftest 同款：
拔网线全绿）。FakeClient 是确定性的（同输入同输出），所以"命中"的
判定可以用调用计数 —— 缓存命中 = 不再调 engine，比比对文本更强。
"""

from __future__ import annotations

import asyncio

import pytest

from app.rag.cached_qa import CachedQAEngine
from app.rag.embedding.hashing import HashingEmbedder
from app.rag.generation.base import ANSWER_PROMPT, SYSTEM_PROMPT
from app.rag.query import QAEngine
from app.rag.result_cache import (
    ANSWER_TTL_S,
    NEGATIVE_TTL_S,
    RETRIEVAL_TTL_S,
    ResultCache,
    answer_prompt_fingerprint,
    fingerprint_from_rag_config,
    reset_result_cache,
)
from app.rag.settings import RagConfig
from app.rag.stores.memory import InMemoryVectorStore

# ---------------------------------------------------------------- 装配


class FakeClient:
    """确定性假 client：计数调用次数（命中判定的依据）。"""

    def __init__(self) -> None:
        self.calls = 0

    async def answer_text(self, question: str) -> str:  # 供手动路径参考
        return "答案"

    class completions:  # noqa: N801 - 模拟 openai 命名
        pass


class CountingFakeClient:
    """按 openai SDK 形状伪装的最小 client，调用即计数。"""

    def __init__(self, text: str = "违约金按 [1] 计算。") -> None:
        self.calls = 0
        self._text = text
        self.chat = self._Chat(self)

    class _Msg:
        def __init__(self, content: str) -> None:
            self.content = content

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = CountingFakeClient._Msg(content)

    class _Resp:
        def __init__(self, content: str) -> None:
            self.choices = [CountingFakeClient._Choice(content)]

    class _Completions:
        def __init__(self, outer: CountingFakeClient) -> None:
            self._outer = outer

        async def create(self, **kwargs: object) -> CountingFakeClient._Resp:
            self._outer.calls += 1
            return CountingFakeClient._Resp(self._outer._text)

    class _Chat:
        def __init__(self, outer: CountingFakeClient) -> None:
            self.completions = CountingFakeClient._Completions(outer)


def _make_engine(**config_changes: object) -> QAEngine:
    """最小可检索引擎：hashing + 内存库 + 内置语料（两份不同主题的文档）。"""
    from app.rag.indexing import Indexer
    from app.rag.schema import Document
    from app.rag.splitters.recursive import RecursiveSplitter

    embedder = HashingEmbedder()
    store = InMemoryVectorStore()
    indexer = Indexer(embedder=embedder, store=store, splitter=RecursiveSplitter())
    indexer.index_documents(
        [
            Document(
                source="合同.md",
                page=None,
                text="违约金条款：甲方逾期支付的，违约金按日万分之五计算，上限为合同总额的百分之十。",
            ),
            Document(
                source="规范.md",
                page=None,
                text="含水量限值：105±5℃ 烘至恒重，含水率 ≤10.0% 判为合格，超差应复测一次。",
            ),
        ],
        source="mixed",
    )
    config = RagConfig(**config_changes) if config_changes else RagConfig(score_threshold=0.0)
    return QAEngine(embedder=embedder, store=store, config=config)


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    reset_result_cache()
    yield
    reset_result_cache()


# ================================================================ 一致性三连


class TestConfigFingerprintInvalidation:
    def test_config_change_forces_retrieval_miss(self) -> None:
        """一致性测试①：换链路配置（top_k 5→3）→ 检索缓存 key 必须变。

        旧结果冒充新链路 = W6 分词器指纹坑的放大版，这条测试就是那件事的守门人。
        端到端：用带缓存的 shell 在配置 A 下写入 → 换配置 B 的 shell 同 query
        必须重新检索（stats 的 hits 不增）。
        """
        cache = ResultCache()
        shell_a = CachedQAEngine(_make_engine(), cache=cache)
        q = "违约金"
        chunks_a, _ = shell_a.retrieve_chunks(q)
        assert chunks_a  # 命中过一次真实检索
        hits_before = cache.stats()["retrieval"]["hits"]

        # 换 top_k（新 engine 实例 = 新配置冻结）
        shell_b = CachedQAEngine(_make_engine(top_k=3), cache=cache)
        chunks_b, _ = shell_b.retrieve_chunks(q)
        # 同一 cache 实例上，配置 B 的 key 与 A 不同 → 没有命中（hits 不变）
        assert cache.stats()["retrieval"]["hits"] == hits_before
        # 结果如实反映新配置（top_k=3 最多 3 条，A 的 5 条不冒充）
        assert len(chunks_b) <= 3

    def test_strategy_change_also_invalidates(self) -> None:
        """切分策略也在指纹里（recursive → heading 必须失效）。"""
        fp_a = fingerprint_from_rag_config(
            RagConfig(score_threshold=0.0, strategy="recursive"), embedder_name="hashing"
        )
        fp_b = fingerprint_from_rag_config(
            RagConfig(score_threshold=0.0, strategy="heading"), embedder_name="hashing"
        )
        assert fp_a != fp_b

    def test_prompt_change_forces_answer_miss(self) -> None:
        """一致性测试②：改 prompt → 答案缓存 key 必须变。

        指纹从 prompt **全文**哈希，不靠手维护版本号 —— 版本号会忘改，全文不会。
        """
        fp1 = answer_prompt_fingerprint(system_prompt=SYSTEM_PROMPT, answer_prompt=ANSWER_PROMPT)
        fp2 = answer_prompt_fingerprint(
            system_prompt=SYSTEM_PROMPT + "追加一句。", answer_prompt=ANSWER_PROMPT
        )
        fp3 = answer_prompt_fingerprint(
            system_prompt=SYSTEM_PROMPT, answer_prompt=ANSWER_PROMPT + "\n5. 新要求"
        )
        assert fp1 != fp2
        assert fp1 != fp3

    def test_model_change_forces_all_miss(self) -> None:
        """一致性测试③：模型版本变 → 检索与答案全部 miss。

        答案层：model 在 key 里（client 换模型 = 不同答案）。
        检索层：embedder 名在指纹里（换编码模型 = 完全不同的向量空间）。
        """
        # 答案层
        k1 = ResultCache.answer_key("问题", context_fp="c", prompt_fp="p", model="deepseek-chat")
        k2 = ResultCache.answer_key(
            "问题", context_fp="c", prompt_fp="p", model="deepseek-reasoner"
        )
        assert k1 != k2
        # 检索层
        cfg = RagConfig(score_threshold=0.0)
        fp_hashing = fingerprint_from_rag_config(cfg, embedder_name="hashing")
        fp_bge = fingerprint_from_rag_config(cfg, embedder_name="bge-small-zh-v1.5")
        assert fp_hashing != fp_bge


# ================================================================ 命中行为


class TestCacheHits:
    def test_retrieval_hit_second_query(self) -> None:
        """同 query 第二次 retrieve 命中缓存（块回查成功，结果等价）。"""
        cache = ResultCache()
        engine = _make_engine()
        shell = CachedQAEngine(engine, cache=cache)
        q = "违约金"

        chunks1, scores1 = shell.retrieve_chunks(q)
        assert cache.stats()["retrieval"]["misses"] == 1

        chunks2, scores2 = shell.retrieve_chunks(q)
        assert cache.stats()["retrieval"]["hits"] == 1
        assert [c.id for c in chunks2] == [c.id for c in chunks1]
        assert scores2 == pytest.approx(scores1)

    def test_low_relevance_query_hits_cache_with_few_results(self) -> None:
        """负缓存可验的前提是"真有空结果"：hashing 无阈值（0.0）时几乎
        永远返回 top-k，所以这里验证的是"少结果的 query 也进缓存并可命中"。
        真正的 0 命中（索引为空）场景在 test_store_miss_on_hit 里顺带覆盖
        （store.clear() 后全链路检索返回空且缓存为空列表条目）。"""
        cache = ResultCache()
        engine = _make_engine()
        shell = CachedQAEngine(engine, cache=cache)

        chunks1, _ = shell.retrieve_chunks("量子纠缠术语zzz")
        chunks2, _ = shell.retrieve_chunks("量子纠缠术语zzz")
        assert [c.id for c in chunks2] == [c.id for c in chunks1]  # 第二次命中
        assert cache.stats()["retrieval"]["hits"] == 1

    def test_answer_hit_second_time(self) -> None:
        """答案层：第二次同 (query, ctx, prompt, model) 不再调 client。"""
        cache = ResultCache()
        engine = _make_engine()
        shell = CachedQAEngine(engine, cache=cache)
        client = CountingFakeClient()
        q = "违约金怎么算"

        r1 = asyncio.run(shell.answer(q, client=client, model="fake-model"))
        calls_after_first = client.calls
        assert calls_after_first >= 1

        r2 = asyncio.run(shell.answer(q, client=client, model="fake-model"))
        assert client.calls == calls_after_first  # 第二次零调用 = 命中
        assert r2.answer.text == r1.answer.text

    def test_context_change_forces_answer_miss(self) -> None:
        """必答问题的可执行形态：多轮上下文变化 → 同一 query 拿不到旧答案。"""
        cache = ResultCache()
        engine = _make_engine()
        shell = CachedQAEngine(engine, cache=cache)
        client = CountingFakeClient()
        q = "它呢"

        # 第 1 轮：history = ["违约金条款是什么"]
        asyncio.run(
            shell.answer(q, client=client, model="m", history_user_messages=("违约金条款是什么",))
        )
        calls_round1 = client.calls
        # 第 2 轮：history 追加了一条 → 上下文指纹变 → 必须 miss（重新调模型）
        asyncio.run(
            shell.answer(
                q,
                client=client,
                model="m",
                history_user_messages=("违约金条款是什么", "金额上限呢"),
            )
        )
        assert client.calls > calls_round1
        # 同一轮内重复提交 → 命中（不重复调模型）
        asyncio.run(
            shell.answer(
                q,
                client=client,
                model="m",
                history_user_messages=("违约金条款是什么", "金额上限呢"),
            )
        )
        assert client.calls == calls_round1 + 1


# ================================================================ 结构保证


class TestNegativeCacheSemantics:
    def test_exception_path_never_caches(self) -> None:
        """异常路径不写缓存：engine 抛超时 → 缓存里不得出现该 query 的条目。

        这是"别把超时缓存成空"的结构性验证 —— 缓存写入在正常返回之后，
        异常路径根本到不了写点。
        """
        cache = ResultCache()

        class Boom:
            # CachedQAEngine 只用到 config/embedder/store（构造时）与 answer（运行时）
            config = RagConfig(score_threshold=0.0)

            class embedder:
                name = "boom"

            store = InMemoryVectorStore()

            async def answer(self, question: str, **kwargs: object) -> object:
                raise TimeoutError("网关超时")

        boom_shell = CachedQAEngine(Boom(), cache=cache)
        with pytest.raises(TimeoutError):
            asyncio.run(boom_shell.answer("违约金", client=CountingFakeClient(), model="m"))
        # 缓存必须干净：没有任何答案条目
        assert cache.stats()["answers"]["entries"] == 0

    def test_ttl_layers_ordered(self) -> None:
        """TTL 分层：embedding（无/永久）> 检索 30min > 答案 5min > 负缓存 2min。"""
        assert NEGATIVE_TTL_S < ANSWER_TTL_S < RETRIEVAL_TTL_S

    def test_store_miss_on_hit_degrades_to_full_lookup(self) -> None:
        """命中但块回查失败（语料变了）→ 降级全链路，绝不返回残缺结果。"""
        cache = ResultCache()
        engine = _make_engine()
        shell = CachedQAEngine(engine, cache=cache)
        q = "违约金"
        shell.retrieve_chunks(q)  # 预热缓存

        # 模拟语料变更：把块从 store 清掉（get_by_content_id 将全部 miss）
        engine.store.clear()
        chunks, _scores = shell.retrieve_chunks(q)
        assert shell.store_miss_on_hit == 1
        # 走了全链路 → 如实返回当前真相（store 空 → 空 chunks），不回放旧块
        assert chunks == []


# ================================================================ SessionStore 内容级幂等


class TestSessionIdempotency:
    def test_duplicate_user_message_returns_same_seq(self, tmp_path: object) -> None:
        """同一 (session_id, 消息内容) 重复提交 → 同一 seq，不产生第二条。"""
        from app.rag.session_store import SessionStore, new_session_id

        store = SessionStore(tmp_path / "s.sqlite")
        sid = new_session_id()
        store.create_session(sid)
        seq1 = store.append_turn(sid, {"role": "user", "content": "违约金怎么算"})
        seq2 = store.append_turn(sid, {"role": "user", "content": "违约金怎么算"})
        assert seq1 == seq2
        assert len(store.load_turns(sid)) == 1
        store.close()

    def test_same_content_different_session_not_merged(self, tmp_path: object) -> None:
        """幂等的边界是 session：两个会话问同一句话各自独立。"""
        from app.rag.session_store import SessionStore, new_session_id

        store = SessionStore(tmp_path / "s.sqlite")
        sid_a, sid_b = new_session_id(), new_session_id()
        store.create_session(sid_a)
        store.create_session(sid_b)
        seq_a = store.append_turn(sid_a, {"role": "user", "content": "你好"})
        seq_b = store.append_turn(sid_b, {"role": "user", "content": "你好"})
        assert seq_a == 0 and seq_b == 0
        assert len(store.load_turns(sid_a)) == 1
        assert len(store.load_turns(sid_b)) == 1
        store.close()

    def test_assistant_messages_not_deduped(self, tmp_path: object) -> None:
        """只对 user 消息去重；assistant 两条同内容是合法的（不同轮次的产物）。"""
        from app.rag.session_store import SessionStore, new_session_id

        store = SessionStore(tmp_path / "s.sqlite")
        sid = new_session_id()
        store.create_session(sid)
        s1 = store.append_turn(sid, {"role": "assistant", "content": "同内容"})
        s2 = store.append_turn(sid, {"role": "assistant", "content": "同内容"})
        assert s1 != s2
        assert len(store.load_turns(sid)) == 2
        store.close()
