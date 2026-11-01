"""BM25 检索：**与向量检索并列的一条路**，不替换它。

## 为什么是"并列"而不是"替换"

W5 的归因表里 15 条错例被判成「术语/同义词未命中」（50%），
但那个标签其实混了两件完全不同的事：

| 真实原因 | 该由谁解决 |
| --- | --- |
| 型号/错误码没对上（`INVALID_INPUT` 找不到） | **BM25**，字面精确匹配 |
| 说法不同（用户写"首字延迟"，文档写"TTFT"） | **向量**，语义近似 |

把它们都叫"术语未命中"，就会得出"要加一个更好的分词器"这种
**两边都不对症**的结论。BM25 上线后第一件事不是看召回率涨了多少，
而是**把这 15 条按上面的判据重新分**（见 `scripts/bm25_compare.py`）。

## 混合的三条路线，本项目先只落第一条

| 路线 | 做法 | 什么时候用 |
| --- | --- | --- |
| **并排对照**（本周） | 两路各跑各的，人工看谁把 gold 排上来 | 先搞清楚"BM25 到底在哪些题上强" |
| RRF 融合 | 按排名倒数加权合并，**不需要分数可比** | 有稳定基线后做 |
| 加权分数融合 | 两路分数归一化后加权 | 需要调权重，且分数分布稳定 |

**先做并排对照而不是直接上 RRF**，理由和 W5 三策略对比一致：
不做对照就不知道收益从哪来，而"融合后涨了 2pp"可能是 BM25 救了 3 条、
又带偏了 1 条 —— 那两个数必须分开看，否则后面所有调参都建在噪声上。

RRF 不做的另一个实际原因：`SearchHit.score` 的契约是余弦相似度，
BM25 分数无界（实测 0~30）。硬塞进同一个类型会让"分数"这个词在库里
有两个含义 —— 那是 W4 已经踩过一次的坑（Chroma 的 distance vs 相似度）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.rag.bm25 import index as bm25_index
from app.rag.bm25.tokenize import tokenize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.rag.schema import Chunk


@dataclass(frozen=True, slots=True)
class BM25Hit:
    """BM25 的一条命中。

    **不复用 `SearchHit`**：那个类型的 `score` 被接口钉死为余弦相似度
    （[-1, 1]，越大越像）。BM25 分数是无界的，混用会让调用方
    "写一个 0.5 的阈值"时在两套口径下得到完全不同的行为。
    """

    chunk: Chunk
    score: float
    rank: int
    """1-based 名次。**RRF 需要的是名次，不是分数** —— 先把它算好，
    以后要加融合时不用改检索层。"""


@dataclass
class BM25Retriever:
    """BM25 检索器。**建索引比查询贵得多，所以索引在构造时就要准备好**。

    与 `Retriever`（向量侧）刻意保持**同一套调用形状**：
    `retriever.retrieve(query, top_k=5)`。形状一致，对照实验与融合才好写，
    也不需要为两路各写一份评测脚本。
    """

    index: bm25_index.BM25Index
    chunks_by_id: dict[str, Chunk] = field(default_factory=dict)

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[Chunk],
        *,
        params: bm25_index.BM25Params | None = None,
        path: object | None = None,
        force: bool = False,
    ) -> tuple[BM25Retriever, bm25_index.BuildReport]:
        """对一批块建索引（或复用），返回检索器 + 建索引报告。

        **返回报告而不只是检索器**：耗时/体积是被任务点名要记录的代价列，
        藏在构造函数里会让调用方拿不到它。
        """
        from pathlib import Path

        target = Path(path) if path is not None else bm25_index.DEFAULT_INDEX_PATH
        report = bm25_index.build(chunks, params=params, path=target, force=force)
        loaded = bm25_index.load(target)
        if loaded is None:
            raise RuntimeError(f"BM25 索引写入 {target} 后读不回来")
        return (
            cls(index=loaded, chunks_by_id={chunk.id: chunk for chunk in chunks}),
            report,
        )

    def retrieve(self, query: str, *, top_k: int = 5) -> list[BM25Hit]:
        """按 BM25 分数取前 `top_k` 条。

        `top_k <= 0` 直接抛（与向量侧一致）：传 0 条是调用方写错了，
        返回空列表会让它以为"没检索到"。
        """
        if top_k <= 0:
            raise ValueError("top_k 必须为正数")
        text = (query or "").strip()
        if not text:
            return []
        ranked = self.index.search(text, top_k=top_k)
        hits: list[BM25Hit] = []
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk is None:
                # 索引里的 id 在当前块集合里找不到 —— 说明索引与块集合不同步。
                # **跳过这条而不是抛**：抛会把整个查询带崩，而"换一份索引"是
                # 运维动作；跳过并留痕，让体检脚本能报出来。
                continue
            hits.append(BM25Hit(chunk=chunk, score=score, rank=rank))
        return hits

    def tokens_of(self, text: str) -> tuple[str, ...]:
        """暴露分词，供对照脚本打印"两边到底切成了什么"。

        **这不是给业务用的**：它存在的唯一理由是"分词不一致"是本周最隐蔽的
        bug，而排查它的第一步永远是**把两侧的 token 流打出来看**。
        没有这个入口，人只能在 debugger 里翻 jieba 的内部状态。
        """
        return tokenize(text)

    def stats(self) -> dict[str, object]:
        return self.index.stats()


__all__ = ["BM25Hit", "BM25Retriever"]
