"""三层结果缓存：检索结果层 + 答案层 + 负缓存（W12 周二）。

## 三层 key 设计（对应周二 09:00 的"三层 key 表"）

| 层 | key 组成 | TTL | 理由 |
| --- | --- | --- | --- |
| embedding | 已有（`embedding/cache.py`，sqlite）| 永久 | 编码只依赖 (文本, 模型, 维度)，与链路配置无关 |
| **检索结果**（本文件）| `sha256(规范化query + 链路配置指纹 + embedder名)` | 中（30min） | 结果随配置变：换 top_k/策略/精排开关，命中列表就不同 |
| **答案**（本文件）| `sha256(规范化query + 上下文指纹 + prompt指纹 + 模型名)` | 短（5min） | 答案随 prompt、上下文、会话历史变 |
| **负缓存**（本文件）| 同检索层 key，值为 `EMPTY` 标记 | 短（2min） | "没找到"也要缓存，否则无答案 query 每次全链路重跑 |

## 为什么配置指纹是本周关键（周二 09:40）

检索结果缓存的 key 若不绑定链路配置：改 top_k=5→8 之后，缓存还在返回
k=5 的结果 —— **旧结果冒充新链路，无任何报错**。这是 BM25 分词器指纹
（W6 周一技术决定 ③）的放大版：那边是"索引与查询两侧分词器不一致"，
这边是"缓存与服务两侧配置不一致"。防线相同：**派生数据必须携带
产生它的配置指纹，取用时现算现比**。

指纹覆盖（全部来自 `RagConfig`，单一真相来源）：
strategy / chunk_chars / overlap_chars / top_k / score_threshold /
max_context_chars / embedder 名。索引库本身变了（换语料）不在指纹里 ——
store 的块 id 是内容派生的，语料一变检索结果自然变，但缓存 key 不含块 id，
所以**索引内容变化必须由调用方显式 `invalidate_index()`**
（Indexer 写路径上自动做，见 `bind_index`）。

## 为什么答案 key 要含"会话上下文指纹"（必答问题）

多轮对话里同一条 query 在不同上下文中答案不同。不加指纹，用户第 2 轮问
"它呢"会拿到第 1 轮缓存的答案 —— 最容易被用户发现、最难复现的 bug。
指纹取**最近 N 轮用户消息的哈希**：上下文变了 key 就变，上下文相同
（同一轮重复提问）才命中。

## 单实例边界

内存 LRU（与周一限流同一条纪律）：多实例各存各的，命中率稀释但**不出错**
（key 含配置指纹，错的是量不是值）。Redis 版留接口（`CacheBackend` 协议），
部署第二个实例那天再实现。

## 负缓存与"真没找到"的区分（周二 13:30）

只有**业务性空结果**（检索 0 命中、分数闸门拒答）才进负缓存；
**超时/异常/降级**一律不缓存 —— 把"服务暂时坏了"缓存成"没找到"，
2 分钟 TTL 内用户会拿到一个确定错误的答案，而那段时间服务可能已经恢复。
区分的机制是**只有正常返回路径才写缓存**，异常路径根本到不了写点
（try/except 在缓存层之外）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

logger = logging.getLogger(__name__)

_KEY_SEP: Final[str] = "\x1f"  # 与 embedding/cache.py 同一条纪律：分隔符不能与内容撞

# ---------------------------------------------------------------- TTL 分层
# embedding 层无 TTL（sqlite 永久），见 embedding/cache.py。
RETRIEVAL_TTL_S: Final[float] = 1800.0  # 30 min：索引不变期间结果稳定
ANSWER_TTL_S: Final[float] = 300.0  # 5 min：模型重排/降级波动，短 TTL 兜住
NEGATIVE_TTL_S: Final[float] = 120.0  # 2 min：空结果短缓存，索引更新后尽快放行

# LRU 容量（条）。检索结果一条 ~几 KB（top5 块），答案一条 ~1KB。
# 2000 条 ≈ 十几 MB 上界，与 EmbeddingCache 的"量级论"同一口径。
RETRIEVAL_CACHE_MAX: Final[int] = 2000
ANSWER_CACHE_MAX: Final[int] = 2000

# 答案层上下文指纹取最近 N 条用户消息（必答问题的实现参数）。
CONTEXT_WINDOW: Final[int] = 4


def normalize_query(query: str) -> str:
    """查询规范化：strip + 压缩空白 + 小写。

    与 embedding/cache.py **刻意不同**（那边坚持原文）：查询是**人打的**，
    前后空格、全半角、大小写的差异不是语义差异 —— "HT-2026-001 的金额" 和
    "ht-2026-001 的金额 " 该命中同一条缓存。块文本是机器切的，原文进 key
    才能保证向量与文本一一对应；查询没有这个不变量要守。
    """
    return re.sub(r"\s+", " ", query.strip().lower())


def _digest(*parts: object) -> str:
    """多字段拼 key（\x1f 分隔，杜绝 ("ab","c") 与 ("a","bc") 撞键）。"""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(_KEY_SEP.encode("utf-8"))
    return hasher.hexdigest()[:32]


# ---------------------------------------------------------------- 配置指纹


def retrieval_config_fingerprint(
    *,
    strategy: str,
    chunk_chars: int,
    overlap_chars: int,
    top_k: int,
    score_threshold: float,
    max_context_chars: int,
    embedder_name: str,
) -> str:
    """链路配置指纹（检索层 key 的第二段）。

    **显式传参而不是收 RagConfig 对象**：指纹的"覆盖哪些字段"必须在这里
    一眼可见 —— 收对象的话，RagConfig 加字段时这里不会自动跟上，
    "新参数影响检索结果却没进指纹"就是静默串味的起点。
    显式列出 = 加参数的人被迫路过这一行，看到它。
    """
    return _digest(
        strategy,
        chunk_chars,
        overlap_chars,
        top_k,
        score_threshold,
        max_context_chars,
        embedder_name,
    )


def fingerprint_from_rag_config(config: Any, *, embedder_name: str) -> str:
    """从 RagConfig 取指纹（便捷入口）。字段清单与上面的显式版本一致。"""
    return retrieval_config_fingerprint(
        strategy=config.strategy,
        chunk_chars=config.chunk_chars,
        overlap_chars=config.overlap_chars,
        top_k=config.top_k,
        score_threshold=config.score_threshold,
        max_context_chars=config.max_context_chars,
        embedder_name=embedder_name,
    )


def answer_prompt_fingerprint(*, system_prompt: str, answer_prompt: str) -> str:
    """prompt 指纹（答案层 key 的一段）。**prompt 一改，答案缓存必须全 miss** ——
    这是周二一致性测试的第二条。指纹从 prompt 全文算，不靠版本号手维护
    （版本号会忘改，全文哈希不会）。"""
    return _digest(system_prompt, answer_prompt)


def context_fingerprint(history_user_messages: Sequence[str]) -> str:
    """会话上下文指纹：最近 N 条用户消息的哈希。

    空 history = 单轮问答，指纹是常量 —— 与"无上下文"显式对应，
    而不是 None 传穿。
    """
    recent = list(history_user_messages[-CONTEXT_WINDOW:])
    return _digest(json.dumps(recent, ensure_ascii=False))


# ---------------------------------------------------------------- 后端协议


class CacheBackend(Protocol):
    """缓存后端协议。内存 LRU 是默认实现；Redis 是第二个实现（未来）。"""

    def get(self, key: str) -> tuple[Any, float] | None: ...

    def put(self, key: str, value: Any, ttl_s: float) -> None: ...

    def clear(self) -> None: ...

    def stats(self) -> dict[str, int]: ...


class LRUCache:
    """线程安全的内存 LRU + 惰性过期。

    读改写之间不 await（与周一限流桶同一条纪律），锁只保护 OrderedDict 结构。
    过期是**读时惰性判**：不设后台线程 —— 没有请求就没有清理的必要，
    过期条目被读到的瞬间才真正淘汰（与 GenerationQueue 的清理挂提交路径同一哲学）。
    """

    def __init__(self, max_entries: int) -> None:
        self._max = max_entries
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        # expires 与 store 同 key 存放（value 二元组的第二位），见 get/put
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired = 0

    def get(self, key: str) -> tuple[Any, float] | None:
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self.misses += 1
                return None
            value, expires = item
            if expires <= now:
                # 惰性淘汰：读到过期条目的瞬间删掉
                del self._store[key]
                self.expired += 1
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value, expires

    def put(self, key: str, value: Any, ttl_s: float) -> None:
        with self._lock:
            self._store[key] = (value, time.monotonic() + ttl_s)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._store),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "expired": self.expired,
            }


# ---------------------------------------------------------------- 缓存条目


@dataclass(frozen=True, slots=True)
class CachedRetrieval:
    """检索层缓存值。存**块 id + 分数**而不是整块 —— 块内容由 id 从 store 回查。

    为什么不直接存 Chunk 列表：语料更新后同一 id 的块内容可能变（重切），
    存 id + 回查让"语料变了但缓存 key 没变"的窗口内拿到的是**当前**块内容
    而不是过期文本；分数仍然缓存（分数由向量算出，重算要 embed）。
    代价是命中路径多一次 store 回查（内存库 <1ms，Chroma ~3ms，可接受）。
    """

    chunk_ids: tuple[str, ...]
    scores: tuple[float, ...]
    top_k: int
    where: tuple[tuple[str, str], ...]  # 过滤条件（可哈希形式）


@dataclass(frozen=True, slots=True)
class CachedAnswer:
    """答案层缓存值。与 `Answer` 的可缓存字段一一对应。"""

    text: str
    refused: bool
    citations: tuple[dict[str, Any], ...]
    invalid_refs: tuple[int, ...]
    top_score: float | None = None
    """检索最高分（拒答证据链的一部分，Answer.top_score 回放用）。"""
    negative: bool = False  # True = 负缓存条目（拒答/空结果）


class ResultCache:
    """三层结果缓存的门面：检索层 + 答案层 + 负缓存。

    ## 顶层接口刻意窄

    只暴露"按指纹取/放"，不让调用方自己拼 key —— key 组成是**缓存正确性的
    全部**，散给调用方等于把串味风险分发到每个调用点。
    """

    def __init__(
        self,
        *,
        retrieval_max: int = RETRIEVAL_CACHE_MAX,
        answer_max: int = ANSWER_CACHE_MAX,
    ) -> None:
        self._retrieval = LRUCache(retrieval_max)
        self._answers = LRUCache(answer_max)

    # ---------------- 检索层 ----------------

    @staticmethod
    def retrieval_key(
        query: str,
        *,
        config_fingerprint: str,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> str:
        return _digest(
            "retrieval",
            normalize_query(query),
            config_fingerprint,
            k,
            json.dumps(where or {}, sort_keys=True, ensure_ascii=False),
        )

    def get_retrieval(self, key: str) -> CachedRetrieval | None:
        hit = self._retrieval.get(key)
        return hit[0] if hit else None

    def put_retrieval(self, key: str, value: CachedRetrieval) -> None:
        self._retrieval.put(key, value, RETRIEVAL_TTL_S)

    # ---------------- 答案层（含负缓存）----------------

    @staticmethod
    def answer_key(
        query: str,
        *,
        context_fp: str,
        prompt_fp: str,
        model: str,
    ) -> str:
        return _digest("answer", normalize_query(query), context_fp, prompt_fp, model)

    def get_answer(self, key: str) -> CachedAnswer | None:
        hit = self._answers.get(key)
        return hit[0] if hit else None

    def put_answer(self, key: str, value: CachedAnswer, *, negative: bool = False) -> None:
        """写答案缓存。`negative=True` 走短 TTL（负缓存）。

        **负缓存只在正常返回路径调用**：拒答（refused）与空结果是正常业务
        结果，走这里；超时/异常根本到不了这一行（调用方的 try/except 在
        缓存层之外）——"别把超时缓存成空"靠结构保证，不靠自觉。
        """
        self._answers.put(key, value, NEGATIVE_TTL_S if negative else ANSWER_TTL_S)

    # ---------------- 失效与指标 ----------------

    def invalidate_retrieval_all(self) -> None:
        """索引内容变化时由 Indexer 写路径调用（换语料 → 检索层全失效）。

        答案层**不**跟着清：答案 key 含上下文与 prompt 指纹，不含索引状态；
        检索层失效后，重检的 chunks 变了，答案 key 的 context_fp 不变但
        命中的是旧答案 —— 所以索引变更时**答案层也必须清**。
        这个"必须"写在实现里而不是注释里（见下）。
        """
        self._retrieval.clear()
        # 索引变了 → 检索结果变 → 建立在旧检索上的答案全部过期。
        # 只清检索层是"半失效"，是缓存一致性最常见的缺口。
        self._answers.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "retrieval": self._retrieval.stats(),
            "answers": self._answers.stats(),
            "ttl": {
                "retrieval_s": RETRIEVAL_TTL_S,
                "answer_s": ANSWER_TTL_S,
                "negative_s": NEGATIVE_TTL_S,
            },
        }

    def hit_rates(self) -> dict[str, float]:
        """命中率（0–1）。没有请求时返回 0.0 —— 不是 None：
        调用方（/metrics）直接进数值聚合，None 会变成聚合里的坑。"""

        def _rate(layer: LRUCache) -> float:
            total = layer.hits + layer.misses
            return round(layer.hits / total, 4) if total else 0.0

        return {"retrieval": _rate(self._retrieval), "answers": _rate(self._answers)}


# 进程级单例（与 get_rate_limiter 同一条装配纪律）。
_cache: ResultCache | None = None


def get_result_cache() -> ResultCache:
    global _cache
    if _cache is None:
        _cache = ResultCache()
    return _cache


def reset_result_cache() -> None:
    """测试用：清单例。"""
    global _cache
    _cache = None


__all__ = [
    "ANSWER_TTL_S",
    "CONTEXT_WINDOW",
    "NEGATIVE_TTL_S",
    "RETRIEVAL_TTL_S",
    "CachedAnswer",
    "CachedRetrieval",
    "LRUCache",
    "ResultCache",
    "answer_prompt_fingerprint",
    "context_fingerprint",
    "fingerprint_from_rag_config",
    "get_result_cache",
    "normalize_query",
    "retrieval_config_fingerprint",
    "reset_result_cache",
]
