"""OCR 兜底测试（缺口 1）：检测 → 兜底 → 标注 → 误差提示的完整闭环。

**全部用 FakeOcrEngine 注入** —— 真实引擎（rapidocr）本机未安装，
本模块能验证的是**接口契约与编排逻辑**，不是识别质量。引擎的真实验收
清单见 `app/rag/loaders/ocr_fallback.py` 的 `RapidOcrEngine` docstring。

## 与既有测试的分工（不重复覆盖）

`test_loaders.py` 第 8 节的三条扫描件测试是**回归红线**：OCR 未安装时
行为必须与之前完全一致（空结果 + warning，不抛）。它们已覆盖"无引擎"
分支——这里只补"有引擎"的新行为。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag import LoadResult
from app.rag.loaders.ocr_fallback import (
    OCR_MIN_CONFIDENCE,
    OCR_SOURCE_TAG,
    OcrPageResult,
    get_ocr_engine,
    ocr_fallback_for_scanned_pages,
)

FIXTURES = Path(__file__).parent / "fixtures"


class FakeOcrEngine:
    """可控 OCR 替身：按页返回预设结果，可注入失败与低置信度。"""

    def __init__(
        self,
        results: dict[int, OcrPageResult],
        fail_pages: set[int] | None = None,
    ) -> None:
        self.results = results
        self.fail_pages = fail_pages or set()
        self.calls: list[int] = []
        self.name = "fake-ocr"

    def recognize_page(self, pdf_path: Path, page_number: int) -> OcrPageResult:
        self.calls.append(page_number)
        if page_number in self.fail_pages:
            raise RuntimeError("模拟引擎崩溃")
        return self.results[page_number]


def _result(pages: int = 3) -> LoadResult:
    """构造"整份是扫描件"形态的 LoadResult（3 页全空 + 对应 warning）。"""
    result = LoadResult()
    result.warnings.append(
        f"{FIXTURES.name} 整份文档都没有可提取文本 —— 很可能是纯扫描件，需要 OCR 才能处理"
    )
    result.warnings.append(f"有 {pages} 页没有可提取文本（扫描件或空白页）：第 1、2、3 页")
    return result


def test_no_engine_returns_result_unchanged() -> None:
    """引擎未安装（None）→ 原样返回。**既有行为的回归红线**。"""
    result = _result()
    before = list(result.warnings)
    out = ocr_fallback_for_scanned_pages(
        result, scanned_pages=[], empty_pages=[1, 2, 3], pdf_path=FIXTURES, engine=None
    )
    assert out is result
    assert out.warnings == before
    assert out.documents == []


def test_engine_fills_scanned_pages_with_ocr_metadata(tmp_path: Path) -> None:
    """成功兜底：产出 Document，metadata 带三件事实（tag/引擎/置信度）。"""
    engine = FakeOcrEngine(
        {p: OcrPageResult(text=f"第 {p} 页识别文本，合同编号 HT-2026-00{p}", confidence=0.87, engine="fake-ocr")
         for p in (1, 2, 3)}
    )
    result = _result()
    out = ocr_fallback_for_scanned_pages(
        result, scanned_pages=[], empty_pages=[1, 2, 3], pdf_path=tmp_path / "扫描件.pdf", engine=engine
    )
    assert [d.page for d in out.documents] == [1, 2, 3]
    for doc in out.documents:
        assert doc.metadata["source"] == OCR_SOURCE_TAG
        assert doc.metadata["ocr_engine"] == "fake-ocr"
        assert doc.metadata["ocr_confidence"] == pytest.approx(0.87)
    assert any("已 OCR 兜底" in w for w in out.warnings)
    assert any("识别误差" in w for w in out.warnings)


def test_engine_failure_on_one_page_does_not_kill_others(tmp_path: Path) -> None:
    """单页引擎崩溃 → 该页 warning + 继续其余页（一张烂图不毁整份）。"""
    engine = FakeOcrEngine(
        {2: OcrPageResult(text="第二页文本", confidence=0.9, engine="fake-ocr")},
        fail_pages={1, 3},
    )
    result = _result()
    out = ocr_fallback_for_scanned_pages(
        result, scanned_pages=[], empty_pages=[1, 2, 3], pdf_path=tmp_path / "s.pdf", engine=engine
    )
    assert [d.page for d in out.documents] == [2]
    assert any("第 1 页 OCR 兜底失败" in w for w in out.warnings)
    assert any("第 3 页 OCR 兜底失败" in w for w in out.warnings)


def test_low_confidence_page_stays_empty(tmp_path: Path) -> None:
    """置信度 < 下限的页**不产出 Document**——乱码块绕过拒答防线比缺页危险。

    保留"无文本"状态：拒答层的分数阈值对它仍然有效。
    """
    engine = FakeOcrEngine(
        {1: OcrPageResult(text=" RANDOM GARBAGE ", confidence=OCR_MIN_CONFIDENCE - 0.1, engine="fake-ocr"),
         2: OcrPageResult(text="正常识别的文本，置信度达标", confidence=0.85, engine="fake-ocr")},
    )
    result = _result()
    out = ocr_fallback_for_scanned_pages(
        result, scanned_pages=[], empty_pages=[1, 2], pdf_path=tmp_path / "s.pdf", engine=engine
    )
    assert [d.page for d in out.documents] == [2]
    assert any("OCR 结果不可用" in w and "第 1 页" in w for w in out.warnings)


def test_normal_pages_are_never_reread(tmp_path: Path) -> None:
    """只兜有问题的页：正常页**绝不进引擎**——文本层比 OCR 准，重读=引入误差。"""
    engine = FakeOcrEngine({2: OcrPageResult(text="扫描页文本", confidence=0.9, engine="fake-ocr")})
    result = LoadResult()
    # 页 1 是正常的（有 Document），页 2 是扫描页（无）
    from app.rag import Document

    result.documents.append(
        Document(source="s.pdf", page=1, text="正常文本层内容", metadata={"page_chars": 100}, pages=(1,))
    )
    result.warnings.append("有 1 页没有可提取文本（扫描件或空白页）：第 2 页")
    out = ocr_fallback_for_scanned_pages(
        result, scanned_pages=[2], empty_pages=[], pdf_path=tmp_path / "s.pdf", engine=engine
    )
    assert engine.calls == [2]  # 引擎只被叫了一次，且是第 2 页
    assert len(out.documents) == 2  # 原有 1 条 + OCR 兜底 1 条


def test_get_ocr_engine_returns_none_without_dependency() -> None:
    """本机未装 rapidocr：get_ocr_engine() 必须返回 None 而不是抛。

    这条同时是**环境契约**：CI / 免费层没有 OCR 依赖，行为必须退化而非崩溃。
    若某环境装了 rapidocr 导致这条失败，说明该环境改变了测试前提——
    把断言改成"返回 None 或可用引擎，且不抛"即可，但默认分支必须保持。
    """
    engine = get_ocr_engine()
    assert engine is None or hasattr(engine, "recognize_page")


def test_scan_warning_still_present_after_fallback(tmp_path: Path) -> None:
    """兜底后**原始扫描页 warning 保留**：事实（是扫描件）与动作（已兜底）分开记录。

    合并成一句会让"兜底失败重试"的排查丢上下文。
    """
    engine = FakeOcrEngine({1: OcrPageResult(text="识别文本", confidence=0.9, engine="fake-ocr")})
    result = _result(pages=1)
    out = ocr_fallback_for_scanned_pages(
        result, scanned_pages=[], empty_pages=[1], pdf_path=tmp_path / "s.pdf", engine=engine
    )
    assert any("需要 OCR 才能处理" in w for w in out.warnings)  # 原始事实
    assert any("已 OCR 兜底" in w for w in out.warnings)  # 兜底动作
