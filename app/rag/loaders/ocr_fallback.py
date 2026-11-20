"""OCR 兜底：扫描页（无文本层）的自动文本抽取 + 误差标注。

## 解决什么问题（缺口 1，课程验收项）

`loaders/pdf.py` 的既有行为是**诚实但不完整**的：检测到扫描页
（`< SCAN_PAGE_CHARS` 字符）→ 打 warning → 告知用户"需要 OCR"。
这在"这份文档能不能用"的问题上是诚实的，但**知识库的建设停在
"承认缺陷"，而不是"补上缺陷"**。

本模块把行为升级为完整闭环：

    检测扫描页 → OCR 引擎抽文本 → metadata 标 source="ocr" + 置信度
    → 引用展示时 UI 标"（OCR 提取，可能有识别误差）"

## 设计：协议 + 延迟导入的引擎（依赖倒置）

**为什么不直接 `import paddleocr`**：

- OCR 引擎是重依赖（paddle 全家桶 GB 级；rapidocr-onnxruntime 轻一个量级
  但也不是零成本）。让它成为**硬依赖**等于把"扫描件兜底"的成本强加给
  每一个部署形态 —— 而本仓库的部署纪律是"Cloud 免费层 1GB 内存也能跑"。
- 所以引擎是**协议**（`OcrEngine`，`runtime_checkable`，与 `Embedder` /
  `VectorStore` / `ChunkingStrategy` 同一模式）：没有引擎时行为与现在
  **完全一致**（warning 照旧），装了引擎自动启用 —— 回归红线写在测试里。

**为什么默认引擎选 rapidocr 而不是 paddleocr**：

| | paddleocr | rapidocr-onnxruntime |
| --- | --- | --- |
| 依赖体积 | paddlepaddle GB 级 | onnxruntime + ~15MB 模型 |
| license | Apache（但 paddle 生态拖家带口） | Apache |
| CPU 推理 | 可跑但重 | 轻量，本仓已有 onnxruntime 依赖链 |
| 本机验证 | 未装 | 未装（接口按官方 API 写，**待真实引擎验收**） |

**诚实声明**：`RapidOcrEngine` 的实现按 rapidocr 官方 API 写，但本机没有装
它 —— 这条路径的**真实引擎验收**（真扫描件上出真文本）是"装了之后"的事。
测试全部用 `FakeOcrEngine` 注入：**接口契约与编排逻辑是可以离线验证的**，
引擎的识别质量不是本模块能证明的事。

## 误差为什么必须进 metadata（而不是只打 warning）

warning 只在**索引时**对**上传者**可见。但 OCR 的误差影响的是**每一个
后续提问者**：检索命中 OCR 块时，引用里必须带着"这段是机器识别的"这个
事实 —— 否则用户拿着一个识别错的数字去核对原件，发现对不上，
信任损失的是整个系统。

所以走 `metadata["source"]="ocr"`（`Chunk` 继承、`to_payload` 自动带到
Chroma，引用链路零改动），UI 侧按这个键显示误差提示。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.rag import Document, LoadResult

logger = logging.getLogger(__name__)

# OCR 抽出的文本 metadata 标记。**键名与取值定版**：
# UI（streamlit_app._render_sources）按 `source == "ocr"` 显示误差提示，
# 改键名必须同步改 UI 与测试。
OCR_SOURCE_TAG = "ocr"

OCR_WARNING_SUFFIX = "（OCR 提取，可能有识别误差）"
"""引用展示的统一后缀。UI 侧用它拼提示，改文案要同步。"""


@dataclass(frozen=True, slots=True)
class OcrPageResult:
    """单页 OCR 的产物。

    `confidence` 是逐行置信度的均值（0~1）。rapidocr 逐行给出分数，
    取均值是**页级诚实度**的最便宜代理 —— 不做加权（行长度加权是
    过度设计：引用误差提示只到页粒度）。
    """

    text: str
    confidence: float
    engine: str


@runtime_checkable
class OcrEngine(Protocol):
    """OCR 引擎的唯一接口。

    与 `Embedder` / `VectorStore` 同一模式：**测试注入替身，生产注入真引擎**。
    实现方只需保证：给一张"页面图像"，返回文本 + 逐行置信度均值。
    抛异常是允许的（单页失败不该毁整份，编排层会兜）。
    """

    name: str
    """引擎名，进 warning 与日志（排查"这段是谁识别的"）。"""

    def recognize_page(self, pdf_path: Path, page_number: int) -> OcrPageResult:
        """对 PDF 的第 `page_number` 页（1-based）做 OCR，返回文本与置信度。"""
        ...


class RapidOcrEngine:
    """rapidocr-onnxruntime 引擎适配器。**延迟导入**：未安装时构造即抛，
    编排层据此保持既有行为（warning 照旧）。

    **待真实引擎验收**：本机未安装 rapidocr，本类的正确性只到
    "接口形态正确"这一层。启用前必须：
    `uv add rapidocr-onnxruntime` → 跑一份真扫描件 → 核对文本与置信度。
    """

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415 - 延迟导入是设计

        self._engine = RapidOCR()
        self.name = "rapidocr"

    def recognize_page(self, pdf_path: Path, page_number: int) -> OcrPageResult:
        import fitz  # PyMuPDF：把 PDF 页渲染成图像给 OCR。延迟导入同上。

        doc = fitz.open(pdf_path)
        try:
            page = doc[page_number - 1]
            pix = page.get_pixmap(dpi=200)  # 200dpi：CPU OCR 的速度/精度平衡点
            lines: list[str] = []
            scores: list[float] = []
            result, _elapse = self._engine(pix.tobytes("png"))
            for item in result or []:
                text, score = item[1], float(item[2])
                lines.append(str(text))
                scores.append(score)
            text = "\n".join(lines).strip()
            confidence = sum(scores) / len(scores) if scores else 0.0
            return OcrPageResult(text=text, confidence=confidence, engine=self.name)
        finally:
            doc.close()


def get_ocr_engine() -> OcrEngine | None:
    """取可用引擎。**没有就返回 None，绝不抛** —— 调用方据此保持既有行为。

    返回 None 不是"OCR 失败"，是"OCR 能力未安装" —— 两者的 warning 文案不同。
    """
    try:
        return RapidOcrEngine()
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 - 引擎构造的任何失败都归为"不可用"
        logger.warning("OCR 引擎构造失败（%s: %s），按未安装处理", type(exc).__name__, exc)
        return None


def ocr_fallback_for_scanned_pages(
    result: LoadResult,
    *,
    scanned_pages: list[int],
    empty_pages: list[int],
    pdf_path: Path,
    engine: OcrEngine | None = None,
) -> LoadResult:
    """对扫描页做 OCR 兜底，成功则**替换**该页缺失的 Document。

    ## 编排规则（顺序敏感）

    1. `engine is None`（未安装）→ 原样返回。warning 保持"需要 OCR"的原文案
       —— 那句话在"能力未装"时仍然是对的。
    2. **只兜有问题的页**：`empty_pages`（完全无文本）+ `scanned_pages`
       （文本层残缺）。正常页**绝不重读** —— 文本层抽出的字比 OCR 准，
       重读等于主动引入误差。
    3. 单页失败 → 追加 warning，**继续下一页**（一张烂图不毁整份文档，
       与 pdf.py 的单页解析失败同一纪律）。
    4. 成功 → 产出 `Document`，metadata 带三件事实：
       `source="ocr"`（引用 UI 误差提示的键）、`ocr_engine`（谁识别的）、
       `ocr_confidence`（均值，页级诚实度）。warning 追加"已 OCR 兜底"。
    5. **置信度过低的页不产出 Document**（`< OCR_MIN_CONFIDENCE`）：
       低于阈值的识别结果是"看起来有内容实则乱码"，比"明确没有"更危险 ——
       拒答层对它的分数会正常放行，幻觉风险直接进答案。保持"无文本 +
       warning"的既有状态。

    ## 为什么追加而不是替换 warning

    "这页是扫描件"是**事实**（不管兜没兜底）；"已 OCR 兜底"是**动作**。
    合并成一句会让"兜底失败重试"的排查丢掉上下文。
    """
    if engine is None:
        return result

    from app.rag.loaders.pdf import SCAN_PAGE_CHARS  # noqa: PLC0415 - 避免循环导入

    # 阈值定版：页级均值 < 0.5 的识别结果不可信。取值依据写在常量旁边。
    min_confidence = OCR_MIN_CONFIDENCE

    for page in sorted({*scanned_pages, *empty_pages}):
        try:
            ocr = engine.recognize_page(pdf_path, page)
        except Exception as exc:  # noqa: BLE001 - 单页失败继续下一页
            logger.warning("第 %d 页 OCR 失败（%s: %s）", page, type(exc).__name__, exc)
            result.warnings.append(
                f"第 {page} 页 OCR 兜底失败（{type(exc).__name__}），该页保持无文本"
            )
            continue

        if not ocr.text or ocr.confidence < min_confidence:
            result.warnings.append(
                f"第 {page} 页 OCR 结果不可用"
                f"（置信度 {ocr.confidence:.2f} < {min_confidence}，{ocr.engine}），该页保持无文本"
            )
            continue

        result.documents.append(
            Document(
                source=pdf_path.name,
                page=page,
                text=ocr.text,
                metadata={
                    "block": "text",
                    "source": OCR_SOURCE_TAG,
                    "ocr_engine": ocr.engine,
                    "ocr_confidence": round(ocr.confidence, 4),
                    "page_chars": len(ocr.text),
                },
                pages=(page,),
            )
        )
        result.warnings.append(
            f"第 {page} 页已 OCR 兜底（{ocr.engine}，置信度 {ocr.confidence:.2f}；"
            f"原文提取 < {SCAN_PAGE_CHARS} 字符）。引用该页时可能有识别误差"
        )
        logger.info("第 %d 页 OCR 兜底成功（%s，置信度 %.2f）", page, ocr.engine, ocr.confidence)

    return result


OCR_MIN_CONFIDENCE = 0.5
"""页级 OCR 置信度下限。低于它的识别结果不产出 Document。

取 0.5 的理由：rapidocr 行级分数的分布重心在 0.8+（印刷体），正常页均值
不会低于 0.7；均值 < 0.5 通常意味着整页识别已经在猜。**宁可保持"该页无
文本"（拒答层有阈值可挡），不可让乱码块正常过检索进答案** —— 后者
绕过了拒答防线，是比"缺内容"更危险的失效形态。
"""


__all__ = [
    "OCR_MIN_CONFIDENCE",
    "OCR_SOURCE_TAG",
    "OCR_WARNING_SUFFIX",
    "OcrEngine",
    "OcrPageResult",
    "RapidOcrEngine",
    "get_ocr_engine",
    "ocr_fallback_for_scanned_pages",
]
