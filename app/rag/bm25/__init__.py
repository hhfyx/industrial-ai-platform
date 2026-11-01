"""BM25 字面检索：**补向量检索的那一半**。

## 一句话

向量检索把整句话压成一个稠密向量，所有词糊在一起；
BM25 数词、按 IDF 加权，所以它在**型号、编号、错误码、专有术语**
这类"必须字面精确对上"的地方强，而向量在"同义、换述"的地方强。

两者的强弱是结构性的，不是程度差异。所以这一层**不是替换**，
是给检索链多一条路（见 `retriever.BM25Retriever` 的 docstring）。

## 必答问题：各自在什么类型的问题上强？

| 题型 | 谁强 | 为什么 |
| --- | --- | --- |
| 型号 / 编号 / 错误码（`INVALID_INPUT`、`v0.4.1`） | **BM25** | 稀有 token，IDF 高；向量里它几乎没有存在感 |
| 专有术语（`幂等性`、`同轴光`） | **BM25** | 字面唯一，且切词后是整体 token |
| 同义 / 换述（"首字延迟" vs "TTFT"） | **向量** | 字面完全不同，BM25 得 0 分 |
| 模糊描述（"怎么省钱"） | **向量** | 没有任何字面锚点 |
| 多跳 | **两者都弱** | W5 实测 BM25 与向量都只有 30% 左右，那是跨块问题 |

**预期**：本项目语料是技术文档（型号、错误码、函数名密集），
所以 BM25 的收益应当比通用语料更大。**这个预期要在周六验证，不在这里下结论** ——
周一的任务只到"能跑通并做对照"。

## 为什么这一层是独立包而不是塞进 `stores/`

`stores.VectorStore` 的 `score` 契约是**余弦相似度**（[-1, 1]）。
BM25 分数无界，塞进去会让 `SearchHit.score` 有两个含义 ——
W4 已经因为"distance vs 相似度"踩过一次。所以：
**BM25 有自己独立的命中类型**（`BM25Hit`），混排时由融合层负责归一化。
"""

from __future__ import annotations

__all__ = ["BM25Hit", "BM25Index", "BM25Params", "BM25Retriever", "tokenize"]


def __getattr__(name: str) -> object:
    """延迟导出：**没装 jieba 时 `import app.rag.bm25` 不该炸**。

    与 `embedding.get_embedder` / `stores.get_store` 同一原则：
    可选依赖的缺失应当是"用一个清晰错误拒绝调用"，而不是
    "在包导入阶段把整个应用带崩"。`/health` 要能在缺依赖时正常回答。
    """
    if name in ("BM25Hit", "BM25Retriever"):
        from app.rag.bm25 import retriever

        return getattr(retriever, name)
    if name in ("BM25Index", "BM25Params"):
        from app.rag.bm25 import index

        return getattr(index, name)
    if name == "tokenize":
        from app.rag.bm25 import tokenize as tokenizer

        return tokenizer.tokenize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
