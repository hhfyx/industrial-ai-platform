"""生成测试文档的 fixture 工厂。

**为什么要脚本化生成，而不是把二进制文件塞进仓库**：
- 二进制 PDF 在 code review 里完全不可读，没人能确认里面到底是什么内容
- 生成脚本本身是"测试意图的文档"：读者一眼看到"50 页，每页 3 段正文 + 页眉"，
  就知道这个 fixture 在测什么
- 需要不同形态的样本（有页眉 / 无页眉 / 扫描件）时，改参数即可

**关键设计：每页正文必须"每页都不同"。**
我第一版犯了相反的错误 —— 每页放了三行**完全相同**的句子。结果
`_strip_repeated_lines` 正确地把它识别成了"页脚"并删掉，
测试却看起来像"loader 丢内容"。

这是一个值得记住的教训：**如果测试数据的每页内容完全一样，
你就无法区分"loader 正常去页眉"和"loader 丢数据"。**
真实文档的正文绝不重复，所以 fixture 也必须做到。
"""

from __future__ import annotations

from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent

# 每页真正重复的部分（页眉 + 页码），这是要被 loader 剔除的
_HEADER = "星辰科技内部资料 · 严格保密"


def _register_cjk_font() -> str | None:
    """注册一个支持中文的字体。找不到就返回 None（退回 Helvetica，中文会变方块）。

    **必须在生成时就解决**：用 `reportlab` 默认字体画中文，抽出来全是乱码/方块，
    那后续所有"页码对不对、内容全不全"的断言都会被乱码干扰。
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont("CJK", path))
                return "CJK"
            except Exception:  # noqa: BLE001, S112 - 换下一个候选
                continue
    return None


def make_pdf(path: Path, *, pages: int = 50, with_header: bool = True) -> Path:
    """生成一份多页 PDF，每页正文**各不相同**。

    结构（`with_header=True` 时）：
        [_HEADER]            ← 每页相同 → 应被删除
        [第 N 页 · 章节 N]   ← 每页不同 → **必须保留**
        [正文 3 行，含 N]    ← 每页不同 → **必须保留**
        [第 N 页 / 共 50 页] ← 页码行 → 应被删除
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    font = _register_cjk_font()
    pdf = canvas.Canvas(str(path), pagesize=A4)

    for index in range(1, pages + 1):
        y = 800
        if with_header:
            pdf.setFont(font or "Helvetica", 8) if font else pdf.setFont("Helvetica", 8)
            pdf.drawString(60, y, _HEADER)
            y -= 44

        pdf.setFont(font or "Helvetica", 15) if font else pdf.setFont("Helvetica", 13)
        pdf.drawString(60, y, f"第 {index} 章 · 第 {index} 页主题")
        y -= 28

        pdf.setFont(font or "Helvetica", 10) if font else pdf.setFont("Helvetica", 9)
        lines = [
            f"本节是第 {index} 页的正文，讨论第 {index} 项交付物的验收标准。",
            f"甲方应于第 {index} 次里程碑验收后十五个工作日内完成付款。",
            f"若第 {index} 期交付延期，乙方按合同总额的 {index}‰ 承担违约责任。",
        ]
        for line in lines:
            pdf.drawString(60, y, line)
            y -= 18

        if with_header:
            pdf.setFont(font or "Helvetica", 8) if font else pdf.setFont("Helvetica", 8)
            pdf.drawString(60, 60, f"第 {index} 页 / 共 {pages} 页")

        pdf.showPage()

    pdf.save()
    return path


def make_docx(path: Path) -> Path:
    """生成一份 DOCX：包含标题、段落、表格。"""
    from docx import Document as DocxDocument

    doc = DocxDocument()
    doc.add_heading("技术服务合同", level=1)
    doc.add_paragraph("甲方：杭州星辰科技有限公司")
    doc.add_paragraph("乙方：上海云脉信息技术有限公司")
    doc.add_paragraph("合同编号：HT-2026-001")

    doc.add_heading("第一条 服务范围", level=2)
    doc.add_paragraph("乙方为甲方提供工业设备预测性维护系统的开发与部署服务。")

    doc.add_heading("第二条 付款方式", level=2)
    doc.add_paragraph("甲方在验收后十五个工作日内支付合同金额的百分之九十。")

    table = doc.add_table(rows=3, cols=3)
    rows = [
        ("里程碑", "时间", "金额"),
        ("一期交付", "2026-11-30", "¥ 38,400.00"),
        ("二期交付", "2027-03-31", "¥ 89,600.00"),
    ]
    for row_index, values in enumerate(rows):
        for col_index, value in enumerate(values):
            table.cell(row_index, col_index).text = value

    doc.save(str(path))
    return path


def make_markdown(path: Path) -> Path:
    """生成一份 Markdown：多级标题 + 代码块（含 `#` 注释，测围栏识别）。"""
    content = """# 工业 AI 平台设计说明

本文档描述平台的总体架构与数据流。

## 一、整体架构

平台由接入层、调度层、模型层三部分组成。

### 1.1 接入层

接入层负责协议转换与限流，支持 HTTP 与 SSE。

```python
# 这段代码里的 # 不是标题，不该被切成新块
def handler(request):
    return {"ok": True}
```

## 二、数据流

原始文档经 loader 解析后，进入切分器，再按 token 预算分批送入模型。

## 三、部署

使用 Docker 多阶段构建，最终镜像约 157 MB。
"""
    path.write_text(content, encoding="utf-8")
    return path


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    make_pdf(FIXTURE_DIR / "sample50.pdf", pages=50, with_header=True)
    make_pdf(FIXTURE_DIR / "sample5_no_header.pdf", pages=5, with_header=False)
    make_docx(FIXTURE_DIR / "sample.docx")
    make_markdown(FIXTURE_DIR / "sample.md")

    for name in ("sample50.pdf", "sample5_no_header.pdf", "sample.docx", "sample.md"):
        p = FIXTURE_DIR / name
        print(f"  {name:24} {p.stat().st_size:>8} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
