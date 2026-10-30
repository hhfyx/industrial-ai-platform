"""Hashing embedder：**能跑通全链路、但不具备语义能力**的本地实现。

## 这个实现的定位（先说清，避免误用）

它是**占位实现**，不是产品方案。用途只有两个：

1. **让"索引 → 检索 → 生成"三段能在拔掉网线的机器上跑通并被测试**。
   一条链路的正确性不该依赖"今天我能不能下载 400MB 模型"。
2. **给出一个稳定的基线**：换成真的本地模型后，可用同一个接口跑同一套测试，
   差异只来自向量质量 —— 这样"换模型到底带来了什么"是可比较的。

## 它到底算的是什么

字符 n-gram（n=1..3）的**符号哈希 + 桶内累加 + L2 归一化**，即 feature hashing。

- 相同的字面 → 相同向量（这是它唯一的保证）
- 字面相近 → 向量相近（局部成立，但**不是语义相似**）
- 「付款」和「支付」：无关。"同义不同词"它完全无能为力，而真 embedding 的核心价值正是这个。

**所以本实现下，检索的召回率没有参考价值。** 任何"RAG 效果好不好"的结论
都必须换成真模型之后才能下 —— 这条写在 `docs/design-decisions.md` 里，
不是为了免责，是因为拿这个跑出来的数字去调参等于在优化噪声。

## 为什么仍然值得现在就写

`stores/` 和 `generation/` 都需要一个"维度已知、结果确定、纯 CPU"的向量源。
没有它，这两层只能靠 mock 测 —— 而 mock 测不出"存进去的向量和查出来的向量
是不是同一套编码"。这个实现能把那条边界也测掉。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

# 中文按字、英文按词切。**不引入分词库**：这个实现的目标是确定性，不是语言学正确性。
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


class HashingEmbedder:
    """字符 n-gram 哈希向量化（确定性、离线、无语义）。"""

    name = "hashing"

    def __init__(self, dimension: int = 256) -> None:
        """`dimension` 取 256 的理由：够小（测试快、可读），够大（冲突不致命）。

        **换这个值等于换索引格式** —— 已入库的向量维度变了就没法再比对。
        所以它不是"随便调调"的参数，改它必须同时重建索引。
        """
        if dimension <= 0:
            raise ValueError("dimension 必须为正数")
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化。顺序与入参一致，长度一致。"""
        # **顺序必须严格保持**：调用方是按位置把向量和 Chunk 配对的，
        # 这里一旦排序或去重，向量就会静默错配到别的块上 —— 那是最难查的一类 bug。
        return [self._embed_one(text) for text in texts]

    # ------------------------------------------------------------------ 内部

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            # 空文本给零向量。**不抛异常**：一整页扫描件（无文本）是正常状态，
            # 让它炸掉整批索引不划算；调用方可以按零向量过滤。
            return vector

        # 三种长度的 n-gram 一起上：单字给"字面命中"，双字/三字给一点局部顺序感。
        # 权重递减是因为越长越稀疏、越容易撞桶。
        for token in tokens:
            self._accumulate(vector, token, weight=1.0)
        for i in range(len(tokens) - 1):
            self._accumulate(vector, tokens[i] + tokens[i + 1], weight=0.7)
        for i in range(len(tokens) - 2):
            self._accumulate(vector, tokens[i] + tokens[i + 1] + tokens[i + 2], weight=0.4)

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        # L2 归一化：让余弦相似度可以直接用点积算，且短文本不会因为"字少"天然吃亏。
        return [v / norm for v in vector]

    def _accumulate(self, vector: list[float], key: str, *, weight: float) -> None:
        """把 `key` 分到某个桶上并累加（带符号，减少碰撞偏置）。

        **用 blake2b 而不是内置 hash()**：后者带进程级随机盐，
        同一份文档在两个进程里会得到不同向量 —— 那会让"入库的向量"
        和"查询时算的向量"根本不在一个空间里，且没有任何报错。
        """
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % self.dimension
        # 第 5 个字节决定符号：所有 n-gram 都加正号会让"撞桶"变成系统性的正向偏置，
        # 任意两个文本的相似度都被抬高。
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign * weight


__all__ = ["HashingEmbedder"]
