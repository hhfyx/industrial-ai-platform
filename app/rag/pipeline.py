"""**唯一装配点**：把三段接成一条可调用的链路。

**为什么要有这个文件，而不是让调用方自己拼三个对象**：

    拼一次 = 记住"embedder 必须索引/检索共用"
           + 记住"store 必须共用同一个实例"
           + 记住"切分参数不能动"

这三条里任何一条被忘记，链路都不会报错，只会**答案变差**。
所以把它们固化成一份装配代码：正确的接法只有一种，且只有一处。

    索引段 →  Indexer  (loaders → splitter → embedder → store)
    检索段 →  Retriever(embedder → store → TopK)
    生成段 →  generate_answer(chunks → prompt → model → citations)

**`RagPipeline` 不是必需的抽象**：它只有 40 行、没有状态机、没有配置中心。
它存在的价值是把"三段共享同一个 embedder 和 store"这件事写死在一个地方。
如果将来各段真的需要独立部署（索引和服务分进程），这个类会被拆成两个进程
各自 `new` 自己那半 —— 那时删除它是一项明确的操作，而不是"发现它没用了"。
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.rag.embedding import Embedder, get_embedder
from app.rag.generation import DEFAULT_CONTEXT_CHARS, Answer, generate_answer
from app.rag.indexing import Indexer, IndexReport
from app.rag.retrieval import DEFAULT_TOP_K, RetrievalResult, Retriever
from app.rag.stores import VectorStore, get_store

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT: str = (
    "你是一个严谨的文档问答助手。只依据给定资料回答，并在结论后标注 [编号]。"
)


class RagPipeline:
    """三段式 RAG 的装配体。

    构造时**强制共享** embedder 与 store —— 这是唯一正确的接法，不给参数让它配错。
    """

    def __init__(
        self,
        *,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        top_k: int = DEFAULT_TOP_K,
        max_context_chars: int = DEFAULT_CONTEXT_CHARS,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.store = store or get_store()
        self.indexer = Indexer(embedder=self.embedder, store=self.store)
        self.retriever = Retriever(
            embedder=self.embedder,
            store=self.store,
            top_k=top_k,
        )
        self.max_context_chars = max_context_chars

    # ------------------------------------------------------------------ 索引段

    def index_file(self, path: str | Path) -> IndexReport:
        """索引一个磁盘文件。"""
        return self.indexer.index_file(path)

    def index_text(self, text: str, *, source: str) -> IndexReport:
        """索引一段内存文本（上传接口走这条路：文件已在内存里，不必先落盘）。

        **单页文档 `page=None`**：一段粘贴进来的文本没有页的概念，
        硬给 `page=1` 会让"第 1 页"变成谎言 —— 和 `Document.page` 的设计一致。
        """
        from app.rag.schema import Document

        return self.indexer.index_documents(
            [Document(source=source, page=None, text=text)],
            source=source,
        )

    # ------------------------------------------------------------------ 检索 + 生成

    def retrieve(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        """只做检索，不调模型。**联调时先看它** ——
        答案不对时，第一步要分清是"没检索到"还是"检索到了但模型没用好"。
        """
        return self.retriever.retrieve(query, top_k=top_k)

    async def answer(
        self,
        question: str,
        *,
        client: object,
        model: str,
        top_k: int | None = None,
        extra_request: dict[str, object] | None = None,
    ) -> Answer:
        """完整问答：检索 → 拼上下文 → 调模型 → 带引用返回。"""
        retrieved = self.retrieve(question, top_k=top_k)
        return await generate_answer(
            question,
            retrieved.chunks,
            client=client,
            model=model,
            max_context_chars=self.max_context_chars,
            extra_request=extra_request,
        )


def build_pipeline(
    *,
    backend: str = "memory",
    top_k: int = DEFAULT_TOP_K,
) -> RagPipeline:
    """按名装配（配置化入口）。

    目前 `backend` 只有 `memory`。加 ChromaDB 时在
    `stores/__init__.py::get_store` 里加一条分支，这里不用改。
    """
    return RagPipeline(store=get_store(backend), top_k=top_k)


__all__ = ["DEFAULT_SYSTEM_PROMPT", "RagPipeline", "build_pipeline"]
