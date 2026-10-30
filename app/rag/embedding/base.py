"""Embedder 接口。

**接口只有两个成员，是刻意的**：

    dimension        向量维度（建集合时要用，事后改等于重建索引）
    embed(texts)     批量向量化

没有 `embed_one`：单条调用在接口层出现，调用方就会一条一条地调，
本地模型的批处理收益直接归零。想 embed 一条就传一个元素的列表 ——
让"批量"成为唯一的路径。

没有 `async`：理由见 `embedding/__init__.py` 的 docstring。

---

## `embed()` 的返回值**必须已 L2 归一化**

这是接口契约里最容易被当成"注释"而忽略的一条，所以单独写清楚：

调用方拿到向量后会直接用内积算相似度（因为归一化后内积 == 余弦，快得多）。
**如果某个实现在这里返回未归一化的向量，检索不会报错，只会静默变差** ——
"长向量天然更相近"会把不相关但很长的块顶到前面。

实测证据在 `scripts/embedding_norm_mechanism.json`：余弦 0.60 的短向量
输给余弦 0.15 的长向量（模长差 18 倍），排序被彻底翻掉。

所以这条不是"建议"，是契约：
- 每个实现都在 `embed()` 里调用 `vector.l2_normalize`
- `tests/test_embedding.py` 对**每个**实现都断言模长 == 1
- 加新实现时如果忘了，那条测试会立刻红 —— 这是它存在的唯一目的
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """向量化器的唯一接口。"""

    name: str
    """实现名。会写进索引元数据 —— 换模型后要能一眼看出"这批向量是谁产的"。"""

    dimension: int
    """向量维度。**入库前必须核对**：维度不匹配的库在检索时会静默返回错结果。"""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化。

        契约三条：
        1. **返回顺序与入参严格一致**，长度相同（错位会让向量配错块）
        2. **每个向量已 L2 归一化**（模长 == 1；零向量除外，见 `vector.l2_normalize`）
        3. 空输入返回空列表，不抛异常
        """
        ...

    def embed_query(self, queries: Sequence[str]) -> list[list[float]]:
        """**query 侧**批量向量化。

        与 `embed()` 的唯一区别是"要不要加检索指令前缀"——那是实现内部的知识：
        - `BGEEmbedder`：加前缀（bge-zh 的训练约定，实测能改善难查询的召回）
        - `HashingEmbedder`：不加（前缀对字符 n-gram 毫无意义，反而污染查询）

        **`Retriever` 只调这个方法**。把前缀处理放进接口，是为了让"忘了加前缀"
        在结构上不可能发生 —— 而忘了加前缀的唯一症状是命中率低一点，
        最容易被归因到切分 / top_k / 模型不行上。

        契约与 `embed()` 相同（顺序、归一化、空输入）。
        """
        ...
