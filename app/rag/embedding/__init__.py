"""向量化：`Chunk.text` → `list[float]`。索引链路与检索链路**共用**这一层。

**为什么两块要共用同一个 embedder**：查询向量和文档向量必须在同一个空间里，
否则余弦相似度算出来的是噪声。这不是"最好共用"，是"不共用就完全错"。
所以这里不提供"文档用 A、查询用 B"的任何接口 —— 接口上杜绝比文档里提醒有效。

**为什么不写成异步**：本地 embedder 是纯 CPU 计算，没有 IO 可等；
套一层 `async` 只会让调用方误以为"并发调它就有收益"，反而掩盖真正的性能位置。
真实模型（sentence-transformers）也是同步阻塞的，要并发应该在**批次**上做，
不是把单次调用改成协程。等真接了本地模型再按批量接口调整，现在不为不存在的需求付设计成本。
"""

from __future__ import annotations

from app.rag.embedding.base import Embedder
from app.rag.embedding.hashing import HashingEmbedder

__all__ = ["Embedder", "HashingEmbedder", "get_embedder"]


def get_embedder(name: str = "hashing") -> Embedder:
    """按名取 embedder（装配点）。

    现在只有一个实现。**保留这个函数而不是让调用方直接 new**：
    换本地模型时改这里一行，`indexing` / `retrieval` 都不用动 ——
    这是 `app/core/deps.py::get_extractor` 同一个模式。
    """
    if name == "hashing":
        return HashingEmbedder()
    raise ValueError(f"未知的 embedder：{name}（当前可用：hashing）")
