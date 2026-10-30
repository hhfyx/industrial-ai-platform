"""向量数学：归一化与余弦相似度。**整个 RAG 链路只允许从这里取相似度。**

## 为什么单独抽一个模块

"忘了归一化"是这一类系统里最常见的静默 bug。它的表现形式不是报错，
而是**命中率莫名其妙地差一点** —— 于是你去调 chunk_size、调 top_k、
换模型、改 prompt，把时间花在所有别的变量上。

所以这里做三件事，都是为了让"忘了"变成不可能：

1. **归一化是 `Embedder` 契约的一部分**，不是调用方的可选步骤（见 `base.py`）
2. **相似度只有一个入口**（`cosine` / `cosine_batch`），没有"内积"这个旁路
3. **`cosine` 内部再除一次模长**，即使输入已经归一化 —— 幂等，且防住"上游偷偷改坏了"

> 第 3 条看起来多余，但它是我给自己留的保险。`HashingEmbedder` 归一化了、
> `BGEEmbedder` 也归一化了，但第三个 embedder 未必。除一次模长的成本是 O(d)，
> 而 bug 的成本是几天。

## 归一化之后，内积 == 余弦（实测 1.7e-17）

    余弦 = (a·b) / (|a| × |b|)
    归一化后 |a| = |b| = 1  →  余弦 = a·b

这不是"差不多"，是**同一个数**。所以生产代码用 `dot`（快，矩阵乘法一条指令），
而**前提必须是先归一化**。反过来，只要有任何一条路径没归一化，
`dot` 就会把"向量长"当成"相关性高"（见 `scripts/embedding_norm_mechanism.json`：
余弦 0.60 的短向量输给余弦 0.15 的长向量）。

## 不依赖 numpy

`loaders` / `splitters` / `stores` 都能在没有 numpy 的环境下跑。
只有 `BGEEmbedder` 需要 torch，它自己做延迟 import（见 `bge.py`）。
**把一个可选的重依赖扩散到全包，是"装了 torch 才能 import 整个 app"的开始。**
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def l2_norm(vector: Sequence[float]) -> float:
    """L2 模长。零向量返回 0.0（不抛）。"""
    return math.sqrt(sum(v * v for v in vector))


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """L2 归一化：把向量缩放到模长 1。

    **零向量原样返回**：它出现在"整页扫描件"这类空文本块上（见
    `HashingEmbedder` 的说明）。归一化零向量会除零，而让它原样通过
    意味着"它与任何查询的余弦都是 0"—— 正好是我们想要的语义：
    永远排在最后，但也不炸掉一次检索。
    """
    norm = l2_norm(vector)
    if norm == 0.0:
        return [float(v) for v in vector]
    return [v / norm for v in vector]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """内积。**只在已归一化的向量上才等于余弦** —— 见模块 docstring。"""
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致：{len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b, strict=True))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度。**这是全链路唯一的相似度入口。**

    对已归一化的向量，结果与 `dot` 完全一致（只多两次开方，实测误差 1.7e-17）；
    对未归一化的向量，它是**正确答案**。所以宁可慢一点点也不用 `dot`，
    除非能证明输入一定归一化 —— 而"能证明"这件事，只在测试里成立，不在运行时成立。

    维度不一致必须抛：那是"索引与查询用了不同 embedder"的结构性错误，
    报 0.0 会把它伪装成"没命中"。
    """
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致：{len(a)} vs {len(b)}")
    na, nb = l2_norm(a), l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def cosine_batch(query: Sequence[float], matrix: Sequence[Sequence[float]]) -> list[float]:
    """一个查询对一批向量算余弦。返回顺序与 `matrix` 一致。

    **顺序绝对不能变**：调用方是按位置把分数配回 `Chunk` 的，
    排序或去重会让分数静默错配到别的块上 —— 那是最难查的一类 bug
    （检索结果看起来正常，只是答案诡异）。
    """
    return [cosine(query, row) for row in matrix]


__all__ = ["cosine", "cosine_batch", "dot", "l2_norm", "l2_normalize"]
