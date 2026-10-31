"""切分策略的统一契约：`ChunkingConfig` + `ChunkingStrategy`。

## 为什么不在 `splitters/base.py` 上加个参数，而要开一个新包

W4 的 `splitters/` 里只有一个实现（`DelimiterSplitter`），它接受 `Sequence[Document]`
直接返回 `list[Chunk]`。要加第二、第三种策略时，有两条路：

| 路 | 代价 |
| --- | --- |
| 给 `DelimiterSplitter.split()` 加 `strategy=` 参数 | 一个类里塞三种互不相干的分段逻辑，**每一种的配置项都会污染另外两种**（`heading_max_level` 对递归字符毫无意义） |
| **按策略分文件，契约抽出来** | 多一个包和一层工厂，但"读哪个文件就能理解哪个策略" |

选后者。这不是分层洁癖：三种策略**依赖的文档特性完全不同**
（见 `docs/chunking-predictions.md` 的表），把它们放进同一个类，
将来"语义切分要个 embedder 参数"这件事就会变成"所有策略的签名都要动"。

## 统一签名

```python
chunk(docs: list[Document], cfg: ChunkingConfig) -> list[Chunk]
```

`docs` 是 **list 而不是 Sequence**：策略内部会按块重排/合并，
传 `tuple` 只会在某个策略里被 `list()` 一次，而签名先声明清楚就不会有人误以为"只读"。

## 三个必须由契约保证的性质

**① `page` / `pages` 继承，不重新推断。**
三种策略的实现方式完全不同（递归字符不知道自己跨页、标题策略按块合并、
语义策略按句重组），但"一块覆盖哪几页"这件事**只能从输入继承** ——
任何"事后按字数折算页码"的尝试都是错的（见 `design-decisions.md` §5）。
所以配置里不放页码相关的选项：**它不是可配置项，是契约**。

**② 表格整张不切。**
`doc.is_table` 为真的 `Document` 必须原样成为一个 `Chunk`。
这条在三种策略里都要显式处理，而不是指望"表格刚好短，不会被切开" ——
一张 20 行的合同金额表被切成两半，行列关系就永久丢失了。

**③ 空块不产出。**
`text` 为空或只有空白的块会污染索引，且"为什么这批少了一条"比
"为什么多了一条空的"难查一个量级。所以空白块在**产出前**被过滤，
而不是留给下游 `if not chunk.text: continue`（下游会忘）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.rag.schema import Chunk, Document

# --------------------------------------------------------------------------- 配置


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """三种策略共用的配置。**每种策略只读它认识的那几个字段。**

    ## 为什么是一个共用配置对象，而不是"每种策略一个配置类"

    任务要求的统一签名是 `chunk(docs, cfg) -> list[Chunk]`。如果每种策略
    自带配置类，工厂就得在运行期判断"传进来的 cfg 是不是这种策略的" ——
    那种判断在 Python 里只能靠 `isinstance` 或干脆不查，
    两者都会让"配置写错了策略"变成一个**静默失效**（字段被忽略，跑出来还是能跑）。

    共用对象 + 每种策略读自己的字段，代价是"cfg 里有一半字段与当前策略无关"。
    这个代价是可接受的：**无关字段被忽略是预期的，而非法的 cfg 类型会被拒绝**。

    ## 字段为什么是字符而不是 token

    整条链路（`ChunkSize` / `DEFAULT_CONTEXT_CHARS` / 成本表）都是字符口径。
    这里换成 token 会引入第二套单位，而两套单位之间的换算（中文 0.676 tok/字）
    是个估计值 —— 用估计值做**切分决策**，误差会直接落到块边界上。
    token 的事留给「拼上下文时的预算裁切」，那里本来就是估算。
    """

    chunk_chars: int = 500
    """目标块长（字符）。三种策略的上限都是它，但"怎么逼近上限"各不相同：

    - 递归字符：贪心合并原子片，装不下就开新块（块的**下限**可能远低于它）
    - 按标题：一节 <= 它就不动；超了才在节内二次切（**结构优先**）
    - 语义：在语义断点处断开，再把相邻小段合并到它以内（**语义优先**）
    """

    overlap_chars: int = 50
    """相邻块的重叠字符数。**只有递归字符与"按标题的节内二次切"会用它** ——
    语义策略在断点处断开时不加重叠：断点就是断点，往前拖 50 字会把
    上一个话题的结尾拖进新话题的开头，反而制造伪依据。这是有意的差异。"""

    # ---- 按标题策略 ----

    heading_max_level: int = 3
    """按标题切分时，最深按到几级标题。

    `3` 意味着 `#` / `##` / `###` 都会开新块，`####` 及更深只在 `###` 节内切。
    **为什么不无限深**：一份文档常有 6 级标题，最深两级往往是"小节内的分点"
    而不是独立主题，按它们切会产出 50 字的块 —— 结构边界不等于检索粒度。

    对 PDF（没有标题元数据）这条**不生效**，见 `heading.py` 的降级说明。
    """

    heading_keep_path: bool = True
    """是否把祖先标题路径写进块正文首行（`A > B > C`）。

    **这不是格式美化**：块是检索与生成的唯一输入。丢了路径，块里的
    `"接入层负责协议转换与限流"` 就不知道自己在讲哪一节 ——
    而路径恰好是最便宜的"主题上下文"，且能让引用标签比页码更好用。
    """

    # ---- 语义策略 ----

    semantic_breakpoint_percentile: float = 25.0
    """语义断点的取法：相邻句相似度**最低的 N% 位置**作为候选断点。

    **为什么不设绝对阈值**（如 `similarity < 0.5`）：相似度的绝对数值是
    embedder 相关的（hashing 与 bge 的分布完全不同，见 `embedding/base.py`
    那张表）。用固定阈值会让"换个模型就切不出块"——而那种失效不报错，
    只是块变长，最容易归因到"模型不行"。取**分位数**则对分布平移免疫。

    取 25 意味着"每 4 个句间位置里有 1 个被视为断点" —— 比真实段落边界密，
    所以还需要合并（见 `semantic_min_chars`），否则会切得过碎。
    """

    semantic_min_chars: int = 200
    """低于这个长度的语义段会被**并回上一段**。

    这是对"并列条款"那类文档的对策（预判里明确写了：语义策略在短句堆叠的
    合同上会切得过碎）。一旦每句都是断点，不合并就会产出 20 字的块群 ——
    检索时命中一堆碎片，上下文预算被切碎的内容占满。
    """

    semantic_max_chars: int = 800
    """超过这个长度且中途没有断点（单主题技术文档），**强制在句子边界处断开**。

    没有这条，语义策略在"一整节都在讲同一个概念"的文档上会一路合并到上限，
    退化成一个代价更高的字数切分器 —— 这比直接输掉更糟：
    付了 embedding 的钱，拿到递归字符的效果。
    """


# --------------------------------------------------------------------------- 契约


@runtime_checkable
class ChunkingStrategy(Protocol):
    """切分策略的唯一接口。

    `runtime_checkable` 让测试可以 `isinstance(strategy, ChunkingStrategy)`
    直接验证工厂装配结果 —— 与 `Embedder` / `VectorStore` 的处理保持一致。
    """

    name: str
    """实现名。写进 `chunk.metadata["splitter"]`，用于排查"这批向量是哪版策略切的"。"""

    def chunk(self, docs: list[Document], cfg: ChunkingConfig) -> list[Chunk]:
        """把一批 `Document` 切成 `Chunk`。空输入返回空列表，不抛异常。"""
        ...


# --------------------------------------------------------------------------- 共用工具


def inherited_pages(doc: Document) -> tuple[int, ...]:
    """`Document` 的覆盖页范围。**三种策略都必须用它，不许各自写一遍。**

    `Document.pages` 为空时回退到 `(page,)`；`page` 也是 `None`（Markdown/DOCX）
    时返回 `()`（而不是伪造 `(1,)`）。

    实测过的坑：某次直接在策略里写 `pages=doc.pages`，跨页段落的
    `pages=(6,7)` 是对的，但**单页段落**（`pages=()`、`page=7`）切出来的块
    `pages=()` —— 于是引用显示 `p7` 而"覆盖范围"是空的。
    `Chunk.__post_init__` 确实会补 `(page,)`，但**只补单页且 pages 为空的情况**；
    跨页 `(6,7)` 一旦在策略里被写成 `()`，就再也补不回来了。
    """
    if doc.pages:
        return doc.pages
    return (doc.page,) if doc.page is not None else ()


def make_chunk(
    *,
    text: str,
    doc: Document,
    index: int,
    strategy: str,
    pages: tuple[int, ...],
    extra_metadata: dict[str, object] | None = None,
) -> Chunk:
    """构造一个 `Chunk`，**统一 metadata 继承链**。

    这条是三种策略唯一允许构造 `Chunk` 的地方。理由：metadata 的继承规则
    （`source` / `page` / `pages` / loader 元数据 / `splitter` 标记）
    一旦在三个文件里各写一遍，其中一处漏掉 `heading_level` 或 `block`，
    表现就是"某策略切出来的表格块在检索后变回普通文本"——
    而那种差别只在下游"引用要不要标（表格）"时才暴露。

    `pages` 由调用方显式传入（用 `inherited_pages(doc)` 算出来的那个），
    而不是在这里重算 —— 因为语义策略会**跨 Document 合并句子**，
    合并后的块覆盖页范围是多个 Document 的并集，不是单个 `doc` 的。
    """
    metadata: dict[str, object] = {
        **doc.metadata,
        "splitter": strategy,
        # `chunk_chars` 记进元数据：改切分参数后能一眼看出"这批向量是哪套参数切的"。
        # 位置派生 id（`store_id`）在换参数时指向不同内容，这条是排查它的唯一线索。
        **(extra_metadata or {}),
    }
    return Chunk(
        text=text,
        page=doc.page,
        source=doc.source,
        metadata=metadata,  # type: ignore[arg-type]
        index=index,
        pages=pages,
    )


__all__ = [
    "ChunkingConfig",
    "ChunkingStrategy",
    "inherited_pages",
    "make_chunk",
]
