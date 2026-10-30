"""向量化：`Chunk.text` → `list[float]`。索引链路与检索链路**共用**这一层。

**为什么两块要共用同一个 embedder**：查询向量和文档向量必须在同一个空间里，
否则余弦相似度算出来的是噪声。这不是"最好共用"，是"不共用就完全错"。
所以这里不提供"文档用 A、查询用 B"的任何接口 —— 接口上杜绝比文档里提醒有效。

---

## 两个实现，用途不同

| 实现 | 语义能力 | 依赖 | 用途 |
| --- | --- | --- | --- |
| `hashing` | ❌ 只有字面相似 | 无 | 离线测试 / CI / 优雅降级 |
| `bge-small-zh` | ✅ 真正的语义 | torch + 92 MB 模型 | **生产** |

**默认是 `hashing`，这不是"还没做完"**：CI 要能在没有 torch、没有模型文件、
没有网络的机器上跑全绿（`tests/conftest.py` 的硬规矩）。`bge` 的测试
在缺依赖时 skip，而不是让整套测试红。

生产环境用 `EMBEDDER_IMPL=bge-small-zh` 或显式 `get_embedder("bge-small-zh")`。

## 为什么不写成异步

本地 embedder 是同步阻塞的 CPU 计算。套一层 `async` 只会让调用方误以为
"并发调它就有收益"，反而掩盖真正的性能位置（在**批量**上，不在并发上：
实测 batch=64 比 batch=1 快 4 倍）。等真要并发时，正确的做法是
`asyncio.to_thread` 把整批扔到线程池，而不是把单次调用改成协程。
"""

from __future__ import annotations

from app.rag.embedding.base import Embedder
from app.rag.embedding.hashing import HashingEmbedder
from app.rag.embedding.vector import cosine, cosine_batch, dot, l2_norm, l2_normalize

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "cosine",
    "cosine_batch",
    "dot",
    "get_embedder",
    "l2_norm",
    "l2_normalize",
]

# 实现名 → 说明。错误信息里列出来，比"未知的 embedder"有用得多。
_IMPLEMENTATIONS: dict[str, str] = {
    "hashing": "字符 n-gram 哈希（无语义，离线测试用）",
    "bge-small-zh": "bge-small-zh-v1.5（512 维，语义检索，需 torch）",
}


def get_embedder(name: str = "hashing", **kwargs: object) -> Embedder:
    """按名取 embedder（装配点）。

    `bge-small-zh` 走**延迟 import**：没装 torch 时要报一句人话，
    而不是在包导入阶段就 `ModuleNotFoundError` 把整个应用带崩。
    这与 `stores.get_store` 里 chroma 的处理是同一个原则 ——
    可选依赖的缺失应当是运行时一个清晰的错误，不是启动期一个神秘栈。

    调用方换实现只改这一个名字，`indexing` / `retrieval` 一行不动。
    """
    if name == "hashing":
        return HashingEmbedder(**kwargs)  # type: ignore[arg-type]
    if name == "bge-small-zh":
        from app.rag.embedding.bge import BGEEmbedder

        return BGEEmbedder(**kwargs)  # type: ignore[arg-type]
    available = "、".join(_IMPLEMENTATIONS)
    raise ValueError(f"未知的 embedder：{name}（当前可用：{available}）")


def describe_implementations() -> dict[str, str]:
    """列出可用实现。给 CLI / 健康检查用，避免它们各写一份 if-chain。"""
    return dict(_IMPLEMENTATIONS)
