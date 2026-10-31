"""向量库：`Chunk + 向量` → 可检索。索引链路的最后一站，检索链路的第一站。

**这一层存在的唯一目的是"能被换掉"**。上层 `indexing` / `retrieval` 只认
[`base.VectorStore`](base.py) 这个 Protocol，所以从内存实现换到 ChromaDB、
再换到 Milvus，改的是装配点，不是业务代码。

## 两个实现

| 实现 | 持久化 | 用途 |
| --- | --- | --- |
| `memory`（`InMemoryVectorStore`） | ❌ 进程退出即丢 | 测试 / 快速实验 |
| `chroma`（`ChromaStore`） | ✅ `data/chroma/` | **生产 / 演示** |

内存实现**仍然保留**，不是历史包袱：CI 里 40 多条 store/检索测试要能秒级跑完，
而 Chroma 每次建 collection 都有固定开销（实测 0.4 秒）+ 落盘。
让绝大多数测试用内存实现、只留少量"Chroma 契约一致性"测试，
是"测试速度"与"真实覆盖"之间的取舍 —— **不是**"内存实现将被废弃"。

> **两个实现必须给出同一种 `score` 语义**（余弦相似度，越大越像）。
> `ChromaStore` 负责把 distance 换算成相似度（见 `chroma.py` 的必答题）；
> `InMemoryVectorStore` 本来就算余弦。这条由
> `tests/test_chroma_store.py::test_both_stores_agree_on_score_direction` 守着。

## 周三时这里写的"不写 ChromaStore 空壳"，周四怎样了

周三的判断是：**决定用 ChromaDB ≠ 今天就要 import chromadb**，
所以只留了 `get_store` 里的一个分支。

周四的需求变了（要持久化、要 where 过滤、要 100 条真数据入库），
于是 `ChromaStore` 被真正写出来并**跑过**：100 条入库、10 条已知答案的问题、
重启后数据还在、where 过滤生效。周三那条注释当时是对的 ——
它防的正是"写一个从未被 import 的适配器"。

现在删掉那句"所以这里不写 ChromaStore"，不是因为它错了，
而是因为它描述的状态已经不存在了。
"""

from __future__ import annotations

from app.rag.stores.base import SearchHit, VectorStore
from app.rag.stores.memory import InMemoryVectorStore

__all__ = ["ChromaStore", "InMemoryVectorStore", "SearchHit", "VectorStore", "get_store"]


def get_store(backend: str = "memory", **kwargs: object) -> VectorStore:
    """按名取向量库（装配点）。

    `chroma` 走**延迟 import**：没装 chromadb 时要报一句人话，
    而不是在包导入阶段就 `ModuleNotFoundError` 把整个应用带崩。
    与 `embedding.get_embedder` 是同一个原则 ——
    可选依赖的缺失应当是运行时一个清晰的错误，不是启动期一个神秘栈。

    换库时只改这一个名字，`indexing` / `retrieval` 一行不动。
    """
    if backend == "memory":
        return InMemoryVectorStore()
    if backend == "chroma":
        from app.rag.stores.chroma import ChromaStore

        return ChromaStore(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"未知的向量库：{backend}（当前可用：memory、chroma）")


def describe_implementations() -> dict[str, str]:
    """列出可用实现。给 CLI / 健康检查用，避免它们各写一份 if-chain。"""
    return {
        "memory": "进程内内存（不持久化，测试与快速实验用）",
        "chroma": "ChromaDB PersistentClient（持久化到 data/chroma）",
    }
