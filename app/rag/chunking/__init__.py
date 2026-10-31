"""切分策略：`Document` → `Chunk` 的**三种实现 + 一个工厂**。

## 为什么与 `splitters/` 并存，而不是替换它

`splitters/` 里是 W4 那个**唯一且已验证**的实现（`DelimiterSplitter`）：
块尾落句末标点、冗余 9.4%、中文分隔符优先级表 —— 40 多条测试锁着它。

W5 要回答的是"**该用哪种切分策略**"，那是另一个问题：
不是在"实现得好不好"上比较，而是在"依赖哪种文档特性"上比较
（见 `docs/chunking-predictions.md` 那张表）。

所以这里的关系是：

    splitters/    = 实现层（已经验证过的切分技法）
    chunking/     = 策略层（用哪种技法，以及另外两种新技法）

`recursive` 策略**包装** `DelimiterSplitter` 而不是重写它 ——
重写只会丢掉那些已经被测试证明过的性质，换来一个"看起来更整齐"的目录结构。

## 统一签名

```python
strategy.chunk(docs, cfg) -> list[Chunk]
```

三者都满足，所以工厂能按配置换策略而调用方不动一行。

## 三种策略各吃什么信号（一句话版）

| 策略 | 信号 | 缺这个信号时会怎样 |
| --- | --- | --- |
| `recursive` | 分隔符分布 | 无标点 → 硬切（不会崩，只是块难看） |
| `heading` | `heading_level` 元数据 | PDF 无标题 → **降级为按页切**（并显式标注） |
| `semantic` | 句向量相似度 | 无 embedder → **拒绝构造**；hashing 下 → 结论无参考价值 |

第三列是**最该被读的一列**：三种策略的失效方式完全不同 ——
一种是"退化但仍能跑"，一种是"降级但会标注"，一种是"必须先满足前提条件"。
把它们的失效方式混在一起统计，得到的平均值没有任何意义。
"""

from __future__ import annotations

from app.rag.chunking.base import ChunkingConfig, ChunkingStrategy, inherited_pages, make_chunk
from app.rag.chunking.factory import (
    DEFAULT_STRATEGY,
    STRATEGIES,
    build_strategy,
    describe_strategies,
    from_config,
    validate_config,
)
from app.rag.chunking.heading import HeadingChunking
from app.rag.chunking.recursive import RecursiveChunking
from app.rag.chunking.semantic import SemanticChunking, split_sentences

__all__ = [
    "DEFAULT_STRATEGY",
    "STRATEGIES",
    "ChunkingConfig",
    "ChunkingStrategy",
    "HeadingChunking",
    "RecursiveChunking",
    "SemanticChunking",
    "build_strategy",
    "describe_strategies",
    "from_config",
    "inherited_pages",
    "make_chunk",
    "split_sentences",
    "validate_config",
]
