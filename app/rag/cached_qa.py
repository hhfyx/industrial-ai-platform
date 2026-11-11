"""缓存装配层：把三层缓存接进 QAEngine（W12 周二 10:40–12:00）。

## 为什么单独一个文件

`result_cache.py` 是**机制**（key、TTL、LRU），这里是**接入点**——
把缓存装进 `QAEngine` 的检索/答案路径。机制与接入分文件的理由
与 mechanism/policy 分层相同：改 TTL 或换 Redis 只动机制文件，
改"哪一步查缓存"只动本文件，两边的提交不会互相搅。

## 接入点与 QAEngine 的真实形状

`QAEngine`（query.py）没有公开的 `retrieve()`，检索是内部
`_retrieve_timed(query, top_k)` → `(chunks, scores, timing)`；
公开入口是 `answer()` → `QAResult(answer, timing, contexts, ...)`。
所以：

- **检索层缓存**包 `_retrieve_timed`（chunks+scores 是稳定形状，
  拒答与否是它的下游决定，不该影响检索缓存）；
- **答案层缓存**包 `answer`（缓存 `QAResult.answer` + `contexts` 回放）。

## 读路径（cache-aside）

    检索：查缓存 → 命中回查 store（get_by_content_id）→ 拼 (chunks, scores)
    未命中 → 全链路 → 只有正常路径写缓存

命中回查用 `store.get_by_content_id`（既有接口，"引用双向可校验"的
第三跳）—— 不新造回查入口。任一块回查失败 → 降级全链路，
**绝不返回残缺结果**。

## 负缓存

检索 0 命中也写缓存条目（空 chunks 列表）—— 下一个同 query 直接拿到空，
不重跑 embed+search。这是检索层的负缓存：空是**检索的真相**，
 TTL 与正常条目相同（30min），因为它的失效条件与正常条目一致（配置/语料）。

答案层的拒答（refused）走负缓存 TTL（2min）：拒答由分数闸门决定，
阈值不在 key 指纹里，短 TTL 保证调阈值实验尽快绕开旧条目。
"别把超时缓存成空"由结构保证：本文件没有 try/except 包住 engine 调用，
异常根本到不了写点。
"""

from __future__ import annotations

import logging
from typing import Any

from app.rag.generation.base import ANSWER_PROMPT, SYSTEM_PROMPT
from app.rag.result_cache import (
    CachedAnswer,
    CachedRetrieval,
    ResultCache,
    answer_prompt_fingerprint,
    context_fingerprint,
    fingerprint_from_rag_config,
    get_result_cache,
)

logger = logging.getLogger(__name__)


class CachedQAEngine:
    """QAEngine 的缓存外壳。**组合而不是继承**：QAEngine 的接口不因缓存而变。

    - `retrieve_chunks(query, top_k)`：检索层缓存（包 `_retrieve_timed`）；
    - `answer(...)`：答案层缓存（包 `answer`），签名与 QAEngine.answer 相同，
      多一个 `history_user_messages` 关键字参数（上下文指纹的输入）。

    命中率与 store 回查失败次数在 `stats()` 里。
    """

    def __init__(self, engine: Any, *, cache: ResultCache | None = None) -> None:
        self.engine = engine
        self.cache = cache or get_result_cache()
        # 检索层指纹：engine 的 RagConfig + embedder 名。构造时算一次 ——
        # config 是 frozen dataclass 冻在 engine 里，指纹跟着冻结。
        self._config_fp = fingerprint_from_rag_config(
            engine.config, embedder_name=engine.embedder.name
        )
        self._prompt_fp = answer_prompt_fingerprint(
            system_prompt=SYSTEM_PROMPT, answer_prompt=ANSWER_PROMPT
        )
        self.store_miss_on_hit = 0

    # ------------------------------------------------------------------ 检索层

    def retrieve_chunks(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[list[Any], list[float]]:
        """检索层 cache-aside。返回 (chunks, scores)，与 `_retrieve_timed` 同形。

        空 query 直接透传（engine 对空 query 返回空，且不调 embedder ——
        缓存空串没有意义，反而占一个条目）。
        """
        engine = self.engine
        k = top_k if top_k is not None else engine.top_k
        if not (query or "").strip():
            return engine._retrieve_timed(query, top_k=top_k)

        key = ResultCache.retrieval_key(query, config_fingerprint=self._config_fp, k=k)
        cached = self.cache.get_retrieval(key)
        if cached is not None:
            chunks = [engine.store.get_by_content_id(cid) for cid in cached.chunk_ids]
            if chunks and all(c is not None for c in chunks):
                return chunks, list(cached.scores)
            # 任一块回查失败（语料重切/被删）→ 降级全链路并计数。
            # 这个计数是"语料变了但缓存没失效"的信号灯。
            self.store_miss_on_hit += 1
            logger.info("检索缓存命中但块回查失败，按 miss 处理: %s", query[:40])

        chunks, scores, _timing = engine._retrieve_timed(query, top_k=top_k)
        self.cache.put_retrieval(
            key,
            CachedRetrieval(
                chunk_ids=tuple(c.id for c in chunks),
                scores=tuple(scores),
                top_k=k,
                where=(),
            ),
        )
        return chunks, scores

    # ------------------------------------------------------------------ 答案层

    async def answer(
        self,
        question: str,
        *,
        client: object,
        model: str,
        top_k: int | None = None,
        history_user_messages: tuple[str, ...] | None = None,
    ) -> Any:
        """答案层 cache-aside。返回 `QAResult`（与 QAEngine.answer 同形）。

        key 含会话上下文指纹（必答问题的落点）：多轮对话里同一条 query
        在不同上下文中答案不同，指纹变了 = miss = 重新走全链路。
        """
        ctx_fp = context_fingerprint(history_user_messages or ())
        key = ResultCache.answer_key(
            question,
            context_fp=ctx_fp,
            prompt_fp=self._prompt_fp,
            model=model,
        )
        cached = self.cache.get_answer(key)
        if cached is not None:
            return self._replay_qa_result(cached)

        result = await self.engine.answer(question, client=client, model=model, top_k=top_k)
        self._store_answer(key, result)
        return result

    def _replay_qa_result(self, cached: CachedAnswer) -> Any:
        """缓存条目 → QAResult（形状与 engine.answer 的返回严格一致）。

        `timing` 回放成零值打点：缓存命中路径没有真实的检索/生成耗时，
        但 QAResult 要求 timing 必填 —— 全零是"这次没干活"最诚实的表达，
        且与 `_ms()` 的 `max(...,1)` 语义不冲突（那是真实测量的下界，
        这里是"没有测量"的占位）。"""
        from app.rag.generation.base import Answer, Citation
        from app.rag.query import QAResult, QATiming

        citations = [
            Citation(
                index=c["index"],
                source=c["source"],
                page=c["page"],
                chunk_id=c["chunk_id"],
                pages=tuple(c.get("pages") or ()),
            )
            for c in cached.citations
        ]
        answer = Answer(
            text=cached.text,
            citations=citations,
            invalid_refs=list(cached.invalid_refs),
            refused=cached.refused,
            top_score=cached.top_score,
        )
        return QAResult(
            answer=answer,
            timing=QATiming(),  # 全零：缓存命中没有真实耗时
            contexts=[],
            top_score=cached.top_score,
        )

    def _store_answer(self, key: str, result: Any) -> None:
        """答案写入。refused=True → 负缓存短 TTL；正常答案 → 标准 TTL。"""
        answer = result.answer
        citations = [
            {
                "index": c.index,
                "source": c.source,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "pages": list(c.pages),
            }
            for c in (answer.citations or [])
        ]
        negative = bool(answer.refused)
        self.cache.put_answer(
            key,
            CachedAnswer(
                text=answer.text,
                refused=negative,
                citations=tuple(citations),
                invalid_refs=tuple(answer.invalid_refs or ()),
                top_score=answer.top_score,
                negative=negative,
            ),
            negative=negative,
        )

    # ------------------------------------------------------------------ 指标

    def stats(self) -> dict[str, Any]:
        """缓存状态：机制层指标 + 本层的回查失败计数。"""
        return {
            **self.cache.stats(),
            "hit_rates": self.cache.hit_rates(),
            "store_miss_on_hit": self.store_miss_on_hit,
        }


__all__ = ["CachedQAEngine"]
