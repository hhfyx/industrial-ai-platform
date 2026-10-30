"""索引段（Indexing）：解析 → 切分 → 向量化 → 入库。

**这一段是"一次性、离线、可以慢"的**，这是它和检索段最本质的区别：

| | 索引段 | 检索段 |
| --- | --- | --- |
| 触发 | 用户上传 / 定时同步 | 每次提问 |
| 耗时容忍 | 秒到分钟 | 毫秒到百毫秒 |
| 失败重试 | 可以整批重跑（幂等） | 必须当场给出结果或明确降级 |
| 成本 | 每份文档一次 | 每个问题一次 |

分清这两者的直接收益是：**索引慢不等于 RAG 慢**。把索引的耗时算进
"问答延迟"里，就会在错误的地方做优化（比如为了首字快而砍块数）。

## 四步各自的输入输出

    ① 解析   Path / bytes      → list[Document]     （已经有：app/rag/loaders/）
    ② 切分   list[Document]    → list[Chunk]        （splitter）
    ③ 向量化 list[Chunk].text  → list[list[float]]  （embedder）
    ④ 入库   Chunk + vector    → store.count()      （store.upsert）

**每一步都只认上一步的输出类型**，这就是这套骨架能换件的原因。
"解析"和"入库"完全不知道对方存在，中间靠 `Chunk` 这个类型接头。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.rag import Document
from app.rag.embedding import Embedder, get_embedder
from app.rag.loaders import load_document
from app.rag.schema import Chunk
from app.rag.splitters import ChunkSize, RecursiveSplitter, Splitter
from app.rag.stores import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    """一次索引的完整结果。**给的是可核对的数字，不是"成功了"。**

    每一项都对得上四步之一：documents→①、chunks→②、vectors→③、stored→④。
    出问题时第一件事就是看这份报告里哪一项对不上。
    """

    source: str
    documents: int = 0
    chunks: int = 0
    vectors: int = 0
    stored: int = 0
    store_total: int = 0
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        """四步计数必须自洽：切出多少块就该有多少向量、写进多少条。

        这条断言值得放在生产代码里而不是测试里：向量与块是**按位**对应的，
        数量一旦不一致，后面的检索就会把 A 的向量配上 B 的正文 ——
        那是最难定位的一类错误（结果不报错，只是答案诡异）。
        """
        return self.chunks == self.vectors == self.stored and self.chunks > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "documents": self.documents,
            "chunks": self.chunks,
            "vectors": self.vectors,
            "stored": self.stored,
            "store_total": self.store_total,
            "warnings": self.warnings,
            "elapsed_ms": self.elapsed_ms,
            "ok": self.ok,
        }


class Indexer:
    """装配好四个部件的索引器。

    依赖全部从构造函数注入：**这是能离线测的前提**。
    如果这里自己 `new` 一个 embedder 或 store，测试就必须联网或造真库。
    """

    def __init__(
        self,
        *,
        splitter: Splitter | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.splitter = splitter or RecursiveSplitter(ChunkSize())
        self.embedder = embedder or get_embedder()
        self.store = store  # 允许为 None：先 index_documents 再 assign_store

    def index_file(self, path: str | Path) -> IndexReport:
        """从磁盘文件索引。解析失败会抛 `ValueError`（那是真失败）。"""
        file_path = Path(path)
        result = load_document(file_path)
        return self.index_documents(
            result.documents, source=file_path.name, warnings=result.warnings
        )

    def index_documents(
        self,
        documents: Sequence[Document],
        *,
        source: str | None = None,
        warnings: Sequence[str] | None = None,
    ) -> IndexReport:
        """从已解析的 `Document` 索引（上传接口走这条路：内存里已有文本）。

        `warnings` 透传 loader 的告警：**"有 3 页没文本"这件事必须跟着报告一起走**，
        否则索引报告显示"全绿"，而用户永远不知道那 3 页查不到。
        """
        if self.store is None:
            raise RuntimeError("Indexer 没有装配 store；请传 store= 或先调用 assign_store()")
        started = time.perf_counter()
        report = IndexReport(
            source=source or (documents[0].source if documents else ""),
            documents=len(documents),
            warnings=list(warnings or []),
        )

        # ② 切分
        chunks = self.splitter.split(documents)
        report.chunks = len(chunks)
        if not chunks:
            # 空文档不是异常：报告里 chunks=0、ok=False，调用方按 ok 判断。
            # **这里刻意不抛**：用户上传一个纯扫描件 PDF 时，
            # 得到一个"0 块"的报告比得到一个 500 更容易理解。
            report.elapsed_ms = _elapsed_ms(started)
            report.store_total = self.store.count()
            return report

        # ③ 向量化（批量一次调用，不在循环里逐条调 —— 见 embedding/base.py）
        vectors = self.embedder.embed([c.text for c in chunks])
        report.vectors = len(vectors)

        # ④ 入库（upsert 语义：同一份文件重复索引不会翻倍）
        report.stored = self.store.upsert(chunks, vectors)
        report.store_total = self.store.count()
        report.elapsed_ms = _elapsed_ms(started)

        logger.info(
            "索引完成：%s（%d 文档 → %d 块 → %d 向量，库内共 %d）",
            report.source,
            report.documents,
            report.chunks,
            report.vectors,
            report.store_total,
        )
        return report

    def assign_store(self, store: VectorStore) -> None:
        """事后绑定 store。检索侧拿到同一个 store 实例时必须走这个口子。"""
        self.store = store


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def index_paths(
    paths: Sequence[str | Path],
    *,
    splitter: Splitter | None = None,
    embedder: Embedder | None = None,
    store: VectorStore,
) -> list[IndexReport]:
    """便捷函数：一次索引多个文件，共用同一个 store 与 embedder。

    **共用 embedder 不是优化，是正确性要求**：不同文档的向量必须在同一空间。
    这个函数把它变成默认行为，调用方就不需要记住这件事。
    """
    indexer = Indexer(splitter=splitter, embedder=embedder, store=store)
    reports: list[IndexReport] = []
    for path in paths:
        try:
            reports.append(indexer.index_file(path))
        except Exception as exc:  # noqa: BLE001 - 单文件失败不该毁掉整批
            # 与 loader 的"能读的读出来"同一个原则：一份坏文件不该让 20 份好文件白跑。
            logger.warning("索引 %s 失败：%s: %s", path, type(exc).__name__, exc)
            reports.append(
                IndexReport(
                    source=Path(path).name,
                    warnings=[f"索引失败：{type(exc).__name__}: {exc}"],
                )
            )
    return reports


__all__ = ["IndexReport", "Indexer", "Chunk", "index_paths"]
