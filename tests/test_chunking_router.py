"""切分路由测试（缺口 2）：按文档实际结构选策略，规则即实验结论。

路由判据复用 `heading.py` 的 `_has_headings`（单一判据来源）；
这里的测试验证**路由决策本身**与**Indexer 的装配优先级**，
不重复验证 heading 策略的切分行为（`test_chunking_*` 已覆盖）。
"""

from __future__ import annotations

from pathlib import Path

from app.rag import Document
from app.rag.chunking.router import ROUTED_STRATEGY, route_strategy


def _doc(text: str, *, block: str = "text", heading_level: int | None = None) -> Document:
    metadata: dict[str, object] = {"block": block}
    if heading_level is not None:
        metadata["heading_level"] = heading_level
    return Document(source="test.md", page=None, text=text, metadata=metadata)  # type: ignore[arg-type]


def test_docs_with_real_headings_route_to_heading() -> None:
    """有标题行（loader 标 block=heading）→ heading。md/docx 的标题是真实结构。"""
    docs = [
        _doc("# W4 开工清单", block="heading", heading_level=1),
        _doc("优先级总览的内容段落", heading_level=2),
    ]
    assert route_strategy(docs) == ROUTED_STRATEGY


def test_docs_with_atx_first_line_route_to_heading() -> None:
    """ "标题 + 正文合一块"（首行 ATX）也算有标题——与 heading.py 同一判据。"""
    docs = [_doc("# 标题行\n\n正文段落")]
    assert route_strategy(docs) == ROUTED_STRATEGY


def test_docs_without_headings_route_to_recursive() -> None:
    """无标题（PDF 文本层弱 / OCR 后 / 纯流水文字）→ recursive。

    OCR 产出没有 heading 标记（OCR 不还原结构），**天然落 recursive** ——
    缺口 1 与缺口 2 的衔接不需要路由知道 OCR 的存在。
    """
    docs = [
        _doc("第一段流水文字，没有任何标题结构"),
        _doc("第二段，合同编号 HT-2026-001"),
    ]
    assert route_strategy(docs) == "recursive"


def test_empty_input_routes_to_default() -> None:
    """空输入没有结构可言 → recursive（默认兜底）。"""
    assert route_strategy([]) == "recursive"


def test_single_heading_with_large_body_is_conservative() -> None:
    """只有一条标题 + 大量正文 → 仍走 heading。

    这条**故意不断言保守侧**：一条真实标题在 50 页正文里确实撑不起
    "结构可信"，但 heading 策略自身的降级路径（`heading_fallback="page"`）
    会兜住它——路由只回答"有没有结构"，结构好不好由策略自己降级。
    写成"标题数 >= 2 才路由"是第二套判据，必然与 `_has_headings` 漂移。
    """
    docs = [_doc("# 唯一的标题", block="heading", heading_level=1)]
    docs += [_doc(f"第 {i} 段正文内容" * 10) for i in range(50)]
    assert route_strategy(docs) == ROUTED_STRATEGY


def test_table_only_docs_route_to_recursive() -> None:
    """纯表格文档：表格不是节边界（heading.py 的既有纪律），不算有结构。"""
    docs = [_doc("| a | b |", block="table"), _doc("| c | d |", block="table")]
    assert route_strategy(docs) == "recursive"


# ============================================================ Indexer 装配优先级


def test_indexer_explicit_splitter_wins_over_routing(tmp_path: Path) -> None:
    """显式 splitter（旧接口）优先于路由——实验脚本注入固定策略不能被偷换。"""
    from app.rag.indexing import Indexer
    from app.rag.splitters import RecursiveSplitter

    class RecordingSplitter(RecursiveSplitter):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        def split(self, documents):  # type: ignore[no-untyped-def]
            self.called = True
            return super().split(documents)

    splitter = RecordingSplitter()
    indexer = Indexer(splitter=splitter)
    docs = [_doc("# 有标题", block="heading", heading_level=1), _doc("正文")]
    indexer._split_with_routing(docs)
    assert splitter.called, "显式 splitter 没被使用——路由把它偷换了"


def test_indexer_routes_structured_docs_to_heading(tmp_path: Path) -> None:
    """默认装配（无 splitter 无 strategy）+ 有标题 → heading 切分。"""
    from app.rag.indexing import Indexer

    indexer = Indexer()
    docs = [
        _doc("# 第一章", block="heading", heading_level=1),
        _doc("第一章的正文，" * 30),
        _doc("# 第二章", block="heading", heading_level=1),
        _doc("第二章的正文，" * 30),
    ]
    chunks = indexer._split_with_routing(docs)
    assert chunks, "切分产出为空"
    assert all(c.metadata.get("routed_strategy") == "heading" for c in chunks)
    assert all(c.metadata.get("splitter") == "heading" for c in chunks)


def test_indexer_routes_plain_docs_to_recursive_and_stamps_metadata(tmp_path: Path) -> None:
    """默认装配 + 无标题 → recursive；路由来源写进块 metadata。"""
    from app.rag.indexing import Indexer

    indexer = Indexer()
    docs = [_doc("流水文字" * 100), _doc("第二段流水文字" * 100)]
    chunks = indexer._split_with_routing(docs)
    assert chunks
    assert all(c.metadata.get("routed_strategy") == "recursive" for c in chunks)


def test_ocr_docs_route_to_recursive_via_indexer() -> None:
    """OCR 兜底产出的 Document（无标题标记）经 Indexer 默认装配落 recursive。

    缺口 1 → 缺口 2 的衔接：metadata 带 source="ocr" 但路由不需要知道它。
    """
    from app.rag.indexing import Indexer

    docs = [
        Document(
            source="扫描件.pdf",
            page=1,
            text="OCR 识别的一段文字，" * 40,
            metadata={"block": "text", "source": "ocr", "ocr_confidence": 0.87},
            pages=(1,),
        )
    ]
    chunks = Indexer()._split_with_routing(docs)
    assert chunks
    assert all(c.metadata.get("routed_strategy") == "recursive" for c in chunks)
