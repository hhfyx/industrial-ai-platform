r"""RAG 的两个**结构级**契约：`Document` 与 `Chunk`。

**为什么这两个类要单独抽一个文件，而不是留在 `__init__.py` / `summarize.py` 里**

W4 之前 `Document` 住在 [`app/rag/__init__.py`](__init__.py)，`Chunk` 住在
[`app/rag/summarize.py`](summarize.py) —— 那时它们只是"摘要链路内部的中间物"，
放哪都行。但从这一步开始，**它们同时是索引侧（写入）和检索侧（读出）的公共货币**：

    切分器输出 Chunk → embedding 输入 Chunk.text → store 存 Chunk 的副本
                     → 检索命中后按 Chunk.id 取回 → 生成层拼上下文带上 source/page

只要有一个环节是"我自己理解的 Chunk"，整套溯源就断了。所以把两个结构
放进同一个模块，让"它们必须一起改"这件事在 import 关系上就能看出来。

`summarize.py` 从本模块 import `Document` 再 re-export，历史调用点
（`from app.rag import Document`）一个都不用动。

---

## 三个必须现在就定死的字段

### 一、`page` 从这一步开始就在结构里

这是本次骨架最容易被省掉、也最不能省的东西。

**省掉的后果是"事后补不回来"**：文本一旦被切分成块、向量化入库，
页码信息就只存在于你切分那一刻的局部变量里。等到上线后发现"用户想点开原文第几页核对"，
你面对的是几千条向量 —— 唯一的选择是**全量重新解析 + 重新切分 + 重新向量化**。
50 页 PDF 重跑一次要几分钟和一次完整计费；几万页知识库重跑就是一次事故。

所以规则是：**`Document.page` 允许为 `None`（Markdown/DOCX 没有物理页），
但 `Chunk` 必须能回答"你来自哪一页/哪几页"**。`Document.page` 为 `None` 时，
切分器把 `None` 一路带下去，而不是伪造一个 `page=1` 让数据看起来完整。

`Chunk` 上同时留了两个字段，各自分工明确：

| 字段 | 类型 | 谁用 | 为什么不能合并 |
| --- | --- | --- | --- |
| `page` | `int \| None` | 引用展示（"p.7"） | 单页引用只需要一个数字，给前端最简单 |
| `pages` | `tuple[int, ...]` | 切分诊断、命中率统计 | 一个块跨了 p.7 到 p.9，只记 `page=7` 会丢掉覆盖范围的真相 |

### 二、`Chunk.id` 是**内容派生**的，不是自增的

`id` 用 `sha256(source + 页码 + 正文)` 的前 16 位。理由：

- **幂等**：同一份文件重复入库不会产生重复向量（增量更新全靠这一点）
- **可跨进程复现**：重建索引得到的 id 与旧索引一致，做 diff 才有意义
- **不含时间戳/随机数**：那些会让同一次输入每次跑出不同 id，等于把幂等性作废

代价也写清楚：**正文改一个字，id 就变** —— 也就是说"编辑过的文档"在库里表现为
"一条新记录 + 一条孤儿记录"，清理孤儿要靠比对 `source`。这是有意的取舍：
在"能精确判断哪些块变了"和"id 稳定但内容漂移"之间，前者对增量更新重要得多。

### 三、`metadata` 只放**可过滤的扁平值**

`metadata` 的类型收窄为 `dict[str, str | int | float | bool]`，不接受嵌套。

这不是洁癖：ChromaDB / Milvus / pgvector 的元数据过滤语义不一致，
嵌套结构有的支持有的不支持。收窄到扁平标量后，`where={"source": "合同.pdf"}`
在三个后端上都能正常工作 —— **结构上先保证可移植，而不是等迁移时才发现过滤写不了**。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, TypeAlias

# 元数据值只允许扁平标量。见模块 docstring 第三节。
MetadataValue: TypeAlias = str | int | float | bool
Metadata: TypeAlias = dict[str, MetadataValue]


def _digest(*parts: str) -> str:
    """稳定摘要：同样的输入永远得到同样的 16 位十六进制串。

    用 sha256 而不是内置 `hash()`：`hash()` 带进程级随机盐（PYTHONHASHSEED），
    **同一份文档在两个进程里会算出不同 id** —— 那正好把"跨进程可复现"这条废掉。
    """
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x1f")  # 分隔符：避免 ("ab","c") 与 ("a","bc") 撞同一串
    return hasher.hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Document:
    """一段来自某个源的文本（loader 的唯一产物）。

    **W4 周二的改动**：新增 `pages` 与 `metadata["block"]`。
    `page` 仍然是"起始页"，`pages` 是"这段文字实际覆盖的所有页" ——
    跨页断句被拼接之后，一段文字可能横跨 p6 与 p7，只记 `page=6` 会让
    "这段到底在第几页"变成半个真话。切分器随后把 `pages` 继承到 `Chunk.pages`。
    """

    source: str
    """来源标识：文件名（不含目录），便于在日志/前端显示。"""

    page: int | None
    """起始页码，从 1 开始；非分页格式为 None。"""

    text: str
    """该页/该段的正文。已做**最小**清洗（见 `loaders.clean`）。"""

    metadata: Metadata = field(default_factory=dict)
    """来源侧的附加信息。**只放事实**：block / heading_level / title_level 等。

    约定的键（loader 负责填，切分与生成只读）：
    - `block`：`"text"` | `"table"` | `"heading"` —— 表格要单独成块，不许混进文字流
    - `heading_level`：Markdown/DOCX 标题级别（1-6）
    - `page_chars`：本页可提取字符数（扫描件判定用）
    """

    pages: tuple[int, ...] = ()
    """覆盖的全部页码。分页格式是 `(7,)` 或 `(6, 7)`；非分页格式是 `()`。"""

    def __post_init__(self) -> None:
        if self.page is not None and self.page < 1:
            raise ValueError(f"page 必须 >= 1 或为 None，收到 {self.page}")

    @property
    def location(self) -> str:
        """人类可读的位置描述，用于日志与报错。"""
        return f"{self.source}:p{self.page}" if self.page is not None else self.source

    @property
    def block(self) -> str:
        """块类型：`text` / `table` / `heading`。缺省 `text`。"""
        return str(self.metadata.get("block", "text"))

    @property
    def is_table(self) -> bool:
        """表格必须**单独成块**（见 `splitters/recursive.py`）。

        表格被切碎或与正文混合后再也拼不回行列关系，
        而"这张表的第三列是什么"恰恰是合同问答里最常见的问题。
        """
        return self.block == "table"


@dataclass(frozen=True, slots=True)
class Chunk:
    """切分产物，也是**索引与检索之间的唯一交换单位**。

    与 `summarize.SummarySegment` 的区别见那边 docstring：那个是"待摘要的一段"，
    带的是 `pages` 元组；这里是"待入库/待检索的一块"，带的是**稳定 id** 与元数据。
    两者都是"一段文字 + 出处"，但生命周期不同（一个是单次请求内，一个要落库），
    混用一个类会让"想给摘要加个字段"变成"动到索引格式"。
    """

    text: str
    """块正文。embedding 的输入就是它，**不要再对入库文本做裁剪**。"""

    page: int | None
    """起始页（从 1 开始）；非分页格式为 None。用于引用展示。"""

    source: str
    """来源文件名，与 `Document.source` 同源。"""

    metadata: Metadata = field(default_factory=dict)
    """可过滤的扁平元数据。切分器会补 `chunk_index` / `pages` 等。"""

    index: int = 0
    """在本文档内的序号（从 0 开始）。排序、去重、调试时用它，不要用它当主键。"""

    pages: tuple[int, ...] = ()
    """覆盖的全部页码。单页块是 `(7,)`，跨页块是 `(7, 8, 9)`，无页格式是 `()`。"""

    id: str = ""
    """内容派生的稳定主键。**留空时自动计算**，见 `__post_init__`。"""

    def __post_init__(self) -> None:
        if not self.id:
            # frozen=True 下改字段要用 object.__setattr__。
            # 这里必须有：让"忘记填 id"从静默的重复入库变成不可能发生。
            object.__setattr__(self, "id", self.content_id())

    def content_id(self) -> str:
        """按 (source, 起始页, 正文) 算出的稳定 id。

        **为什么不把 `index` 也算进去**：块顺序由切分策略决定，
        改一次 `chunk_tokens` 就会让所有 `index` 平移。若 `index` 参与 id 计算，
        "只改了切分参数"会表现成"整库 id 全部失效"—— 幂等性会被参数调优打穿。
        反过来 (source, page, text) 只随**内容**变，正是我们要的性质。

        同一页里恰好有两块内容完全一样（例如重复的表格行）：它们会得到同一个 id。
        这是**有意**的 —— 完全相同的文本入库两次没有检索价值，去重才是正确行为。
        """
        return _digest(self.source, str(self.page), self.text)

    @property
    def block(self) -> str:
        """块类型：`text` / `table` / `heading`（从 `Document` 继承而来）。"""
        return str(self.metadata.get("block", "text"))

    @property
    def is_table(self) -> bool:
        """这一块是不是表格。生成层可以用它决定引用里要不要标"（表格）"。"""
        return self.block == "table"

    @property
    def location(self) -> str:
        """人类可读位置：`合同.pdf:p7` / `合同.pdf:p7-9` / `notes.md`。"""
        if not self.pages:
            return self.source if self.page is None else f"{self.source}:p{self.page}"
        if len(self.pages) == 1:
            return f"{self.source}:p{self.pages[0]}"
        return f"{self.source}:p{self.pages[0]}-{self.pages[-1]}"

    def to_payload(self) -> dict[str, Any]:
        """给存储层的扁平记录。

        **正文同时也进 metadata**：向量库返回的是 `document` 字段，
        但不同后端在"是否默认回传正文"上行为不一致。把正文再存一份到 metadata
        会让存储翻倍 —— 所以这里**不这么做**，而是在 store 接口里把
        `text` 作为一等字段显式传出（见 `stores.base.SearchHit`）。
        """
        return {
            "id": self.id,
            "text": self.text,
            "metadata": {
                **self.metadata,
                "source": self.source,
                "page": -1 if self.page is None else self.page,
                "pages": ",".join(str(p) for p in self.pages),
                "chunk_index": self.index,
            },
        }


def restore_location(metadata: Metadata) -> tuple[int | None, tuple[int, ...]]:
    """把 `to_payload()` 扁平化过的元数据还原成 `(page, pages)`。

    **为什么要走"扁平化再还原"这一趟**：Flatten 是存储层的硬约束（见模块
    docstring 第三节），但业务侧要的是结构化页码。这个函数是唯一允许知道
    "page 用 -1 表示 None"编码的地方 —— 编码只出现一次，解码也只出现一次，
    两边不会各自漂移。
    """
    raw_page = metadata.get("page", -1)
    page = None if raw_page is None or int(raw_page) < 0 else int(raw_page)
    raw_pages = str(metadata.get("pages", "") or "")
    pages = tuple(int(p) for p in raw_pages.split(",") if p.strip().isdigit())
    return page, pages


__all__ = [
    "Chunk",
    "Document",
    "Metadata",
    "MetadataValue",
    "restore_location",
]
