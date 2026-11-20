"""切分策略路由：按**文档的实际结构**选策略（缺口 2，课程验收项）。

## 解决什么问题

三种策略都实测过（W4–W6，结论在 `docs/chunking-quality.md` /
`strategy-comparison.md` / `ablation-four.md`），但生产装配是**全局固定
recursive**——实测结论停在了"实验报告"，没有变成"生产机制"。

路由把这个缺口补上：**规则本身就是实验的产物**，每一条都有实测出处：

| 规则 | 依据 |
| --- | --- |
| 有真实标题行 → heading | heading 的核心优势"不该切的节一个字都不切"依赖真实标题；md/docx 的标题是**文件自带的结构**，可信 |
| 无标题（PDF 文本层弱 / OCR 后）→ recursive | W4 实测：无标题文档走 heading 会降级成按页切（`heading_fallback="page"`），分组边界来自页码而非语义——不如直接 recursive，省一次假的分组 |
| 语义策略**不自动路由** | W6 实测（40 条集）：语义切分前提不成立（70% 段短于 `semantic_min_chars`），Recall 37.5% vs 40.0% 不显著且索引慢 4.7 倍——它只在"长句连续、段落密集"的语料上才有发挥，这种判断需要看内容分布，不适合一行规则拍板 |

## 为什么路由键是 `Document` 列表而不是文件后缀

**后缀说的是"文件的出生地"，标题说的是"文件里有什么"。**

- 同是 `.md`：有标题结构的该走 heading，没有标题的（纯流水文字）走
  heading 会把整份切成一坨——按后缀路由会在这一类文档上静默变差；
- PDF 的标题是**猜出来的**（`pdf_headings.py` 字号启发式）：猜得可信时
  loader 才填 `block="heading"`，不可信时一行都不填——所以
  `Document.metadata` 里的标题标记**已经包含了"这次解析里结构是否可信"**
  这个信息，比后缀准确一层。

因此路由函数收 `Sequence[Document]`，判据复用 `heading.py` 的
`_has_headings`（同一份代码，不是两套判断——两套必然漂移）。

## 为什么语义永远不自动路由

路由的每条规则都对应一个**可机械判定的前提**（有标题/无标题）。
"语义切分合不合适"的前提是**内容分布**（句长、段密度），它需要跑一遍
embedding 才知道——为一个可能被拒绝的策略先付 embedding 的钱，
这个顺序本身就是错的。语义策略保持显式配置（`RagConfig.strategy="semantic"`），
这是"实验结论 says 默认关闭"的延续，不是偷懒。
"""

from __future__ import annotations

from collections.abc import Sequence

from app.rag.chunking.factory import DEFAULT_STRATEGY
from app.rag.schema import Document

ROUTED_STRATEGY = "heading"
"""结构可信时路由到的策略名。与 `factory.STRATEGIES` 中的名字一致。"""

# OCR 兜底产出的文本没有结构（OCR 不还原标题层级），必然落 recursive ——
# 这是缺口 1 与缺口 2 的自动衔接：路由函数不需要知道 OCR 的存在，
# metadata 里没有标题标记就是"无结构"，判定天然成立。


def route_strategy(docs: Sequence[Document]) -> str:
    """按这批 `Document` 的实际结构返回策略名（"recursive" | "heading"）。

    ## 判据与默认方向

    - **有标题行**（`block="heading"` 或首行 ATX）→ `"heading"`；
    - **无标题** → `"recursive"`（`factory.DEFAULT_STRATEGY`）。

    **宁可路由到保守侧**：recursive 在任何文本上都能跑（工厂 docstring 的
    "默认值必须是最不容易静默变差的那个"），heading 在无结构文档上会
    降级成按页切——两者都不崩，但 recursive 的行为可预测。判据存疑时
    （例如标题只有一行、正文 50 页）落 recursive：一条标题撑不起
    "结构可信"的结论。

    空输入返回 recursive：空列表没有结构可言，交给默认兜底。
    """
    if not docs:
        return DEFAULT_STRATEGY

    from app.rag.chunking.heading import _has_headings  # noqa: PLC0415 - 单一判据来源

    return ROUTED_STRATEGY if _has_headings(list(docs)) else DEFAULT_STRATEGY


__all__ = ["ROUTED_STRATEGY", "route_strategy"]
