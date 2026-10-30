"""向量库：`Chunk + 向量` → 可检索。索引链路的最后一站，检索链路的第一站。

**这一层存在的唯一目的是"能被换掉"**。上层 `indexing` / `retrieval` 只认
[`base.VectorStore`](base.py) 这个 Protocol，所以从内存实现换到 ChromaDB、
再换到 Milvus，改的是装配点，不是业务代码。

## 当前只装配了内存实现，这是**有意的**

`docs/design-decisions.md` 第三条论证了"先 ChromaDB 而不是 Milvus"，
但"决定用 ChromaDB"和"今天就要 import chromadb"是两件事：

- 骨架阶段的目标是**三段链路能跑通、能被测试**，而 ChromaDB 的持久化价值
  要等到有第二份文档、第二次启动才体现
- 一旦 `chromadb` 进了 `pyproject.toml`，它就成了 CI 的硬依赖（还要拉 onnxruntime），
  为了一个还没被用到的能力付这份代价不值得
- 内存实现和 ChromaDB 是**同一个 Protocol**，所以换的时候业务代码一行不动 ——
  这正是现在就把接口定下来的意义

**所以这里不写 `ChromaStore` 的空壳**。写一个从未被 import、从未被跑过的适配器
不是"预留扩展点"，是**未被验证的代码**：它看起来"已经做完了"，实际第一次用
大概率要重写。真正的扩展点在 `get_store` 这一个分支上，加一条 `if` 就够了。
"""

from __future__ import annotations

from app.rag.stores.base import SearchHit, VectorStore
from app.rag.stores.memory import InMemoryVectorStore

__all__ = ["InMemoryVectorStore", "SearchHit", "VectorStore", "get_store"]


def get_store(backend: str = "memory") -> VectorStore:
    """按名取向量库（装配点）。

    换 ChromaDB 时在这里加一条分支，`indexing` / `retrieval` 都不需要动 ——
    与 `app/core/deps.py::get_extractor` 是同一个模式。
    """
    if backend == "memory":
        return InMemoryVectorStore()
    raise ValueError(f"未知的向量库：{backend}（当前可用：memory）")
