"""切分：`Document` → `Chunk`。索引链路的第一道工序。

**这一层只做一件事：把长文本切成"检索时能被单独命中"的块。**

不在这里做的事（各有归属，混进来就是将来换不掉的原因）：
- 不做 embedding（那是 `embedding/`，换模型不该动切分逻辑）
- 不做入库（那是 `stores/`，换库不该动切分逻辑）
- 不做清洗（那是 `loaders/`，原始文本的边界一旦模糊就无法排查"是模型读错还是我们切错"）

**为什么强调"切成能被单独命中"**：块太大，一个块里塞了三种主题，命中后答案不准；
块太小，一句话被切成两半，两半各自都不完整。所以 `ChunkSize` 的默认值
取 500 字符 / 50 字符重叠 —— 这个量级是"一段完整论述"的常见长度，
不是拍脑袋：一份 A4 合同条款段通常 200–600 字，取 500 让大多数条款**整段落在一个块里**。

**为什么必须有重叠**：跨块的信息断裂只能靠重叠缓解（完整论证见 `delimiter.py`）。
代价是索引冗余约 9.4%（实测）、chunk 数变多、检索时更容易命中重复内容。

## 两个名字，一个实现

`DelimiterSplitter` 是正式名（核心是**中文分隔符优先级表**，不是"递归"这个技巧）。
`RecursiveSplitter` 是别名，保留因为 W4 周三的文档与测试引用的是它 ——
**改名不该顺手把历史文档的链接打死**。

`recursive.py` 现在是 `DelimiterSplitter` 的兼容导出层，
它存在的唯一理由是"`from app.rag.splitters.recursive import RecursiveSplitter`
这条 import 还能用"。新代码请从 `app.rag.splitters` 直接拿。
"""

from __future__ import annotations

from app.rag.splitters.base import ChunkSize, Splitter
from app.rag.splitters.delimiter import DEFAULT_SEPARATORS, DelimiterSplitter

# 别名：周三的代码/文档/测试用的是这个名字。两者是**同一个类**，
# 用赋值而不是 `class RecursiveSplitter(DelimiterSplitter)` ——
# 子类会让 `isinstance` 与 `name` 出现两种答案，而它们本该是同一件事。
RecursiveSplitter = DelimiterSplitter

__all__ = [
    "DEFAULT_SEPARATORS",
    "ChunkSize",
    "DelimiterSplitter",
    "RecursiveSplitter",
    "Splitter",
]
