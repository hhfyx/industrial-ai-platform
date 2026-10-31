"""CLI `--json` 输出的契约测试。

## 为什么这个文件只测一个 dict

`--json` 是**给脚本和复核用的那一面**：W5 做矩阵实验、以及"拿一次问答的输出
去核对引用"的人，都只会看到这段 JSON，看不到 UI 里的块对象。

W4 v1 的缺口正是从这里漏过去的：`retrieved` 数组里**一个 id 都不带**，
`citations` 里虽有 `chunk_id`，但它指向的块在 JSON 侧无从对照 ——
于是"正文 [n] ↔ 引用列表 ↔ chunk id"这条链在 JSON 上是断的，
而断的地方不会报错，只会让人以为"引用没问题"。

这里把三个字段钉死：
1. `citations[].chunk_id` 必须等于这次检索里某一块的 `retrieved[].chunk_id`
2. 两边都带 `pages`（跨页范围，只给 `page` 就是半个真话）
3. `refused` / `invalid_refs` 这类"坏消息"必须原样输出，不能被"整理得更好看"
"""

from __future__ import annotations

from app.cli.ask import _json_payload
from app.rag.generation import Answer, Citation
from app.rag.retrieval import RetrievalResult
from app.rag.schema import Chunk
from app.rag.stores import SearchHit


def _chunk(text: str, *, page: int | None, source: str = "合同.pdf", pages: tuple = ()) -> Chunk:
    return Chunk(text=text, page=page, source=source, pages=pages)


def test_json_payload_links_citations_to_retrieved_by_chunk_id() -> None:
    """**双向校验的 JSON 面**：引用的 `chunk_id` 必须能在 `retrieved` 里找到。"""
    first = _chunk("违约金按合同总额的千分之一计算。", page=7, pages=(7,))
    second = _chunk("逾期超过三十日，甲方有权解除合同。", page=9, pages=(9,))
    retrieved = RetrievalResult(
        query="违约金",
        hits=(
            SearchHit(chunk=first, score=0.91),
            SearchHit(chunk=second, score=0.62),
        ),
        top_k=2,
        elapsed_ms=7,
    )
    answer = Answer(
        text="违约金按千分之一计算 [1]。",
        citations=[Citation(index=1, source="合同.pdf", page=7, chunk_id=first.id, pages=(7,))],
        top_score=0.91,
        used_chunks=2,
        context_chars=120,
    )

    payload = _json_payload("违约金", answer, retrieved)

    assert payload["question"] == "违约金"
    assert payload["answer"] == answer.text
    assert payload["refused"] is False
    assert payload["invalid_refs"] == []
    assert payload["top_score"] == 0.91

    citation = payload["citations"][0]
    retrieved_ids = {row["chunk_id"] for row in payload["retrieved"]}
    assert citation["chunk_id"] in retrieved_ids, (
        "引用的 chunk_id 在 retrieved 里找不到 —— JSON 侧的双向校验断了"
    )
    assert citation["pages"] == [7], "引用里必须带跨页范围"
    assert all("chunk_id" in row for row in payload["retrieved"]), (
        "retrieved 里必须逐条带 chunk_id，否则无法对照引用"
    )
    assert all("pages" in row for row in payload["retrieved"])


def test_json_payload_keeps_crosspage_and_no_page_sources() -> None:
    """跨页块与无页来源都不能在 JSON 里被拍平。

    - 跨页块 `pages=(6,7)`：`page` 只有 6，必须靠 `pages` 说清覆盖范围
    - Markdown/DOCX 的 `page=None`：不能变成 `1` 或 `0`
    """
    crosspage = _chunk("跨页的违约责任条款", page=6, pages=(6, 7))
    unpaged = _chunk("笔记里的一段话", page=None, source="notes.md")
    retrieved = RetrievalResult(
        query="q",
        hits=(SearchHit(chunk=crosspage, score=0.8), SearchHit(chunk=unpaged, score=0.4)),
        top_k=2,
    )

    payload = _json_payload("q", Answer(text="见 [1]"), retrieved)
    rows = {row["source"]: row for row in payload["retrieved"]}

    assert rows["合同.pdf"]["page"] == 6
    assert rows["合同.pdf"]["pages"] == [6, 7], "跨页范围被拍平成了单页"
    assert rows["notes.md"]["page"] is None, "无页码来源被伪造成了某个数字"
    assert rows["notes.md"]["pages"] == []


def test_json_payload_surfaces_bad_news() -> None:
    """`invalid_refs` 与 `refused` 必须原样出现 —— 它们是"这次回答有问题"的唯一信号。

    把它们"整理掉"（省略空数组、把拒答写成空 answer）会让复核者
    看不到编造痕迹 —— 而"看起来正常但有编造成分"是最危险的输出。
    """
    answer = Answer(
        text="事实 [7]。",
        citations=[],
        invalid_refs=[7],
        refused=False,
        top_score=0.1,
    )
    payload = _json_payload("q", answer, RetrievalResult(query="q", hits=(), top_k=0))

    assert payload["invalid_refs"] == [7], "越界编号被丢掉了"
    assert payload["retrieved"] == []
    assert payload["used_chunks"] == 0
