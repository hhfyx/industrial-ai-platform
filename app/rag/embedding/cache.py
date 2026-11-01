"""Embedding 结果缓存：**同一段文本只付一次编码成本**。

## 为什么需要它（不是"优化"，是"能不能做实验"）

W5 要跑的参数网格是 `chunk_chars ∈ {300,500,800,1000}` × `overlap ∈ {0,50,100}`。
每换一组参数就要**重建一次索引** —— 而重建的耗时几乎全在 `embed()` 里
（bge CPU 实测 ~10 ms/条，batch=64 时 ~2.4 ms/条）。

没有缓存时，同一条命令跑第二遍要重新编码全部块；有缓存时，第二次是纯查表。
**"第二次重建应该明显变快"就是缓存命中的可观测证据** —— 这条比"我实现了缓存"有用得多，
因为它是个能测出真假的时间差。

## 缓存键怎么算（这一节是设计里最容易错的地方）

    sha256(model_name \x1f dimension \x1f text)

三个字段都有理由，少一个都会静默串味：

| 字段 | 不写进去会怎样 |
| --- | --- |
| `model_name` | 换模型后仍命中旧向量 —— 而**维度可能恰好相同**（512 维的模型不止 bge 一个），于是"换了模型效果变差"永远查不出来 |
| `dimension` | 同上，且它会在入库时炸在 `ChromaStore`（维度不匹配），报错点在缓存的下游很远 |
| `text` | 不用说了 —— 但**要用原文而不是归一化后的文本**（见下） |

**为什么不对 `text` 做 strip/归一化**：`" 付款 "` 与 `"付款"` 是两个不同的输入，
模型给出的向量本来就不同（不是"应该相同"）。为了让缓存多命中几个而把它们并成一个键，
等于拿"少算几毫秒"换"向量与文本不再一一对应"——**这种不一致没有任何报错，
只会在某次检索里表现为"结果有点怪"**。

代价如实写在这里：前后空白不同的同一句话会各存一份。这个代价可以接受，
因为切分器产出的块文本本来就来自同一套 loader 清洗（见 `loaders/clean.py`）。

**键里不放时间戳/随机数/进程 id**：那些会让同一次输入每次都算出新键，
命中率恒为 0 —— 而那看起来和"缓存没生效"一模一样，最难查。

## 为什么只缓存 passage 侧（`embed`），不缓存 `embed_query`

两条理由：

1. **收益不对称**。索引侧是 1000+ 条，查询侧是每问一条。缓存查询向量省不到什么。
2. **风险不对称**。`embed_query` 要对文本**加检索指令前缀**（见 `bge.py` 第二节）。
   想在缓存层命中，就得在缓存里把前缀逻辑再实现一遍 —— 而那份逻辑一旦与
   `bge.py` 漂移，症状是"前缀有时加有时不加"，**没有任何报错**，只是命中率低一点。
   这正是 `bge.py` 用 `embed_query` 接口而不是靠注释提醒要防的那类故障。

所以：**查询侧不做缓存，也不复制前缀逻辑**。实验脚本要复用查询向量，
在脚本层自己算一次缓存住即可（那是脚本知道自己在干什么的地方）。

## 为什么用 sqlite 而不是一堆 .json

1000 块 × 512 维 fp32 = 2 MB，写成 1000 个 JSON 文件是 1000 次 open/close，
且目录里会有 1000 个文件（`ls` 都没法看）。sqlite 单文件、单事务批量写、
`PRIMARY KEY` 自带去重，正好是"按 key 查向量"这个形状。

**落盘位置在 `data/`**（`DEFAULT_CACHE_DIR`），与 `ChromaStore` 同一个理由：
`data/` 已在 `.gitignore` 里，且换机器 `cp -r` 能带走。

## 一个必须承认的精度损失：fp32 存盘

向量以 fp32（`struct` 格式 `f`）存进 BLOB，回读后与原始 fp64 结果有 **~1e-8**
的差异（实测：同一段文本首次编码 vs 缓存回读，最大分量差 8.3e-9）。

这不是 bug，也**不该"修"成 fp64**：512 维 × 8 字节是 4 KB/条，
而 bge 本身输出就是 fp32（`model.encode(convert_to_numpy=True)` 给的是 float32）。
用 fp32 存回去是**保持原精度**，用 fp64 反而是在存储一个不存在的精度。

影响面如实写清：1e-8 的扰动会改变余弦值的第 8 位小数。bge 的相关/无关分布间隔
是 0.367 → 0.578（见 `base.py`），差了近 0.2 —— **排在前面的几条不会因为
1e-8 换位**。所以 `docs/retrieval-tuning.md` 里所有命中率数字不受缓存影响，
这一点在文档里也写了一遍。
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import struct
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from app.rag.embedding.base import Embedder
from app.rag.embedding.vector import l2_normalize

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR: str = "data/embedding-cache"

# 键的分隔符。**不能用空串或 `|`**：`("ab","c")` 与 `("a","bc")` 会撞成同一串，
# 这在 `schema._digest` 里已经踩过一次，同一条纪律。
_KEY_SEP: str = "\x1f"


def _project_root() -> Path:
    """项目根 = `app/rag/embedding/cache.py` 往上三级（与 `stores/chroma.py` 同口径）。"""
    return Path(__file__).resolve().parents[3]


def cache_key(*, model: str, dimension: int, text: str) -> str:
    """算一条缓存键。**模型名 + 维度 + 原文**，理由见模块 docstring 第二节。"""
    hasher = hashlib.sha256()
    for part in (model, str(dimension), text):
        hasher.update(part.encode("utf-8"))
        hasher.update(_KEY_SEP.encode("utf-8"))
    return hasher.hexdigest()[:32]


def default_cache_path(model_name: str) -> Path:
    """默认库文件：`data/embedding-cache/<模型名>.sqlite`。

    按模型名分文件而不是共用一个库：换模型时**不需要清缓存也不会有串味风险**，
    旧模型的库原地留着，回滚时还能命中。
    """
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in model_name)
    return _project_root() / DEFAULT_CACHE_DIR / f"{safe}.sqlite"


class EmbeddingCache:
    """向量缓存（sqlite 后端）。**按 key 存、按 key 取，不做淘汰**。

    不做 LRU 淘汰的理由：这个库的规模上界由"知识库块数 × 换过几套参数"决定，
    量级是 MB 到几十 MB。加一套淘汰策略只会引入"为什么这条又没命中"的新问题，
    而收益是省几十 MB 磁盘。真要清，删文件即可（删文件等价于清空缓存）。
    """

    def __init__(self, path: str | Path, *, dimension: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        # `check_same_thread=False` + 自己加锁：批量编码可能被 `asyncio.to_thread`
        # 丢到线程池里（见 `embedding/__init__.py` 的说明）。sqlite 连接本身不是
        # 线程安全的，但"一个连接 + 一把锁"是安全的，且不需要每个线程开一个连接。
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "  key TEXT PRIMARY KEY,"
            "  dim INTEGER NOT NULL,"
            "  vec BLOB NOT NULL"
            ")"
        )
        self._conn.commit()
        self._vec_struct = struct.Struct(f"<{dimension}f")
        # 命中统计。**必须记，否则"缓存有没有起作用"只能靠感觉。**
        self.hits = 0
        self.misses = 0
        # 条目数快照：`stats()` 在 close 之后要用（见那里的说明）
        self._entries = 0

    # ------------------------------------------------------------------ 读写

    def get_many(self, keys: Sequence[str]) -> list[list[float] | None]:
        """批量取。未命中的位置是 `None`（而不是抛错、也不是跳过）——

        **位置必须严格对应入参**：返回一个"只含命中的列表"会让调用方对着两个
        不同长度的序列做 zip，然后向量与块错位。那种错不报错，只让检索结果诡异。
        """
        if not keys:
            return []
        out: list[list[float] | None] = []
        placeholders = ",".join("?" for _ in keys)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT key, dim, vec FROM vectors WHERE key IN ({placeholders})",
                list(keys),
            ).fetchall()
        found = {key: (dim, blob) for key, dim, blob in rows}
        for key in keys:
            hit = found.get(key)
            if hit is None:
                out.append(None)
                self.misses += 1
                continue
            dim, blob = hit
            # **维度对不上就当未命中，而不是硬解**：解出来长度不对的向量会一路
            # 传到 `store.upsert`，在那里炸成"维度不匹配"—— 报错点离真正的原因很远。
            if dim != self.dimension or len(blob) != self._vec_struct.size:
                logger.warning("缓存维度不一致（%d != %d），按未命中处理", dim, self.dimension)
                out.append(None)
                self.misses += 1
                continue
            out.append(list(self._vec_struct.unpack(blob)))
            self.hits += 1
        return out

    def put_many(self, items: Sequence[tuple[str, Sequence[float]]]) -> int:
        """批量写。返回写入条数。空输入不报错（与 `Embedder.embed` 同口径）。"""
        if not items:
            return 0
        rows: list[tuple[str, int, bytes]] = []
        for key, vector in items:
            values = list(vector)
            if len(values) != self.dimension:
                # 写进去就晚了：宁可这条不进缓存（下次重算），也不要存一条解不出来的。
                logger.warning(
                    "向量维度 %d 与缓存维度 %d 不一致，跳过写入", len(values), self.dimension
                )
                continue
            rows.append((key, self.dimension, self._vec_struct.pack(*values)))
        if not rows:
            return 0
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vectors (key, dim, vec) VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def stats(self) -> dict[str, object]:
        """缓存规模与命中情况。给实验脚本打表用。

        **close 之后仍可调用**：返回最后一次读到的快照。理由是调用方的自然顺序
        是"用完 → close → 打报告"，而让 `stats()` 在 close 后抛
        `Cannot operate on a closed database` 会把一个纯展示操作变成崩溃点
        （实测：测试里 `with` 块结束后读 `stats()` 就炸了）。
        """
        with self._lock:
            try:
                total = self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
                self._entries = int(total)
            except sqlite3.ProgrammingError:
                # 连接已关闭：用最后记录的值，而不是炸掉一个展示用调用。
                pass
        attempted = self.hits + self.misses
        return {
            "path": str(self.path),
            "entries": self._entries,
            "bytes": self.path.stat().st_size if self.path.exists() else 0,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / attempted, 4) if attempted else 0.0,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class CachedEmbedder:
    """给任意 `Embedder` 套一层缓存。**只缓存 `embed()`（passage 侧）**。

    契约完全透传：`name` / `dimension` / `recommended_min_score` 都取自被包裹的实现，
    所以上层（`Indexer` / `Retriever` / 测试）不需要知道有没有缓存。
    **这一点很重要**：缓存是透明的，如果它改变了 `name`，索引元数据就会
    记成另一个模型，而那会让"这批向量是谁产的"失去意义。
    """

    def __init__(self, inner: Embedder, *, cache_path: str | Path | None = None) -> None:
        self.inner = inner
        self.name = inner.name
        self.dimension = inner.dimension
        self.recommended_min_score = inner.recommended_min_score
        self.cache = EmbeddingCache(
            cache_path or default_cache_path(inner.name), dimension=inner.dimension
        )
        # 编码耗时（只统计真正算了向量的那部分），给"缓存到底省了多少"做证据。
        self.encode_ms = 0.0
        self.timings: list[dict[str, float | int]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """带缓存的批量向量化。**返回顺序与入参严格一致**（与 `Embedder` 契约同）。"""
        if not texts:
            return []
        started = time.perf_counter()
        keys = [
            cache_key(model=self.inner.name, dimension=self.dimension, text=text) for text in texts
        ]
        cached = self.cache.get_many(keys)

        # 只对未命中的位置调模型 —— **去重后再调**：同一批里出现两次相同文本
        # （重叠切分 + 重复段落里很常见）不应该编码两次。
        missing_texts: list[str] = []
        missing_positions: dict[str, list[int]] = {}
        for position, (text, hit) in enumerate(zip(texts, cached, strict=True)):
            if hit is not None:
                continue
            if text not in missing_positions:
                missing_positions[text] = []
                missing_texts.append(text)
            missing_positions[text].append(position)

        if missing_texts:
            encoded = self.inner.embed(missing_texts)
            self.cache.put_many(
                [
                    (cache_key(model=self.inner.name, dimension=self.dimension, text=text), vector)
                    for text, vector in zip(missing_texts, encoded, strict=True)
                ]
            )
            for text, vector in zip(missing_texts, encoded, strict=True):
                for position in missing_positions[text]:
                    cached[position] = list(vector)

        elapsed = (time.perf_counter() - started) * 1000
        self.encode_ms += elapsed
        self.timings.append(
            {
                "requested": len(texts),
                "unique_missing": len(missing_texts),
                "elapsed_ms": round(elapsed, 1),
            }
        )

        # 走到这里任何位置都不该是 None：未命中的都已填回。留一个显式检查而不是 `# type: ignore`，
        # 因为"某个位置漏填"会让向量与块错位 —— 那是最难查的一类错。
        result: list[list[float]] = []
        for position, vector in enumerate(cached):
            if vector is None:
                raise RuntimeError(f"缓存回填遗漏：第 {position} 条没有向量")
            # 再归一化一次是刻意的双保险：缓存里的向量可能来自旧版本实现，
            # 而"未归一化"的后果是"长向量天然更相近"（见 `vector.py` 的说明）。
            result.append(l2_normalize(vector))
        return result

    def embed_query(self, queries: Sequence[str]) -> list[list[float]]:
        """**不走缓存**，直接透传给被包裹的实现。理由见模块 docstring。"""
        if not queries:
            # 空输入早返回：与 `embed()` 和 `Embedder` 契约一致（空输入不抛、不调模型）。
            # 不早返回的话，内层实现会被要求编码一个空列表 —— 那在某些实现里
            # 会写进一条空键，成为一条永远命不中也永远不用的垃圾记录。
            return []
        return self.inner.embed_query(queries)

    def stats(self) -> dict[str, object]:
        return {**self.cache.stats(), "encode_ms_total": round(self.encode_ms, 1)}

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> CachedEmbedder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = [
    "DEFAULT_CACHE_DIR",
    "CachedEmbedder",
    "EmbeddingCache",
    "cache_key",
    "default_cache_path",
]
