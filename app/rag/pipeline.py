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

    def index_text(self, text: str, *, source: str, append: bool = False) -> IndexReport:
        """索引一段内存文本（上传接口走这条路：文件已在内存里，不必先落盘）。

        **单页文档 `page=None`**：一段粘贴进来的文本没有页的概念，
        硬给 `page=1` 会让"第 1 页"变成谎言 —— 和 `Document.page` 的设计一致。

        ## `append` 参数是为一个真实 bug 加的

        默认 `append=False` 表示"**这段文本就是 `source` 的全部内容**"：
        先按 source 删干净，再写。这与"索引一份文件"的语义一致 ——
        同一份文件重新上传，旧版本必须消失。

        原来没有 `append`，于是频繁调用 `index_text(text, source="合同")`
        （每段都是同一份"文档"的第 0 块）时，**后一次会静默覆盖前一次**：
        实测连续入 3 条不同文本，库里只剩 1 条，而每次 `IndexReport.ok` 都是 `True`。

        这个 bug 是 `tests/test_embedding.py` 的 bge 端到端测试抓出来的：
        它想入 3 条不同条款再检索，结果只有最后一条在库里，命中自然不对。
        **一次"看起来与 Chroma 无关"的测试，抓出了索引语义的缺陷** ——
        这也是"换 store 时要跑全套测试"的实际价值。

        `append=True` 的语义是"把这段加到已有内容后面"（同一 source 累积多个块）。
        显式选择比隐式默认安全：想追加的人会写出来，不想追加的人不会误伤。
        """
        from app.rag.schema import Document

        if append:
            # 追加：新块的序号接在已有块后面。
            # **必须算偏移量**：`index_documents()` 切出来的块 index 从 0 开始，
            # 而 store_id 是 `{doc_id}_{page}_{chunk_index}` —— 不偏移的话
            # 每批第一块都是 `_0`，后一批覆盖前一批（实测连续追加 3 条只剩 1 条）。
            offset = self.store.count_by_source(source)
        else:
            # 覆盖语义：先按 source 清掉旧版本，再写新的。
            # **必须在写入前删**，否则"旧文档 10 块、新文档 1 块"会留下 9 条孤儿。
            self.store.delete_by_source(source)
            offset = 0

        return self.indexer.index_documents(
            [Document(source=source, page=None, text=text)],
            source=source,
            index_offset=offset,
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
        min_score: float | None = None,
        extra_request: dict[str, object] | None = None,
    ) -> Answer:
        """完整问答：检索 → 拼上下文 → 调模型 → 带引用返回。

        **必须把检索分数传给生成层**：`generate_answer` 的分数闸门靠它拒答。
        不传的话（`scores=None`）闸门会被跳过，于是"库里只有字面碰巧相似的垃圾"
        时模型会拿它当依据硬答 —— 那正是要防的。

        ## 阈值从 embedder 取，不是全局常量

        `min_score` 不传时用 `self.embedder.recommended_min_score`。
        这条改动来自一个真实的测试失败：用 `HashingEmbedder` 时全局的 0.35
        把正确命中的答案（分数 0.247）误判成"未找到"并拒答。
        实测发现 hashing 的相关/无关分数**完全重叠**（0.148~0.387 vs 0.190），
        任何单一阈值都无意义 —— 所以它给 `0.0`（关掉闸门），而 bge 给 `0.35`。

        传 `0.0` 表示**显式关掉**闸门（与"忘了传"是两件事，后者更危险）。
        """
        retrieved = self.retrieve(question, top_k=top_k)
        threshold = self.embedder.recommended_min_score if min_score is None else min_score
        return await generate_answer(
            question,
            retrieved.chunks,
            client=client,
            model=model,
            max_context_chars=self.max_context_chars,
            min_score=threshold,
            scores=retrieved.scores(),
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
