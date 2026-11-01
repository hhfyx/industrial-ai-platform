"""BM25 索引：**字面检索**，与向量检索互补的那一半。

## BM25 是什么（三行讲完）

    score(D, Q) = Σ_{q ∈ Q} IDF(q) · [ tf(q,D) · (k1+1) ] / [ tf(q,D) + k1 · (1 - b + b·|D|/avgdl) ]

三个成分各自解决一个具体的失效模式：

| 成分 | 解决什么 | 直觉 |
| --- | --- | --- |
| **TF 饱和**（`k1`） | 一个词出现 100 次不代表相关 100 倍 | 出现 1→3 次涨得快，3→10 次几乎不涨 |
| **IDF 加权** | "所有块都有的词"不该有票 | 「公差」若每块都有 → IDF≈0；「同轴光」稀有 → IDF 高 |
| **长度归一**（`b`） | 长块天然更容易撞词 | 500 字的块与 100 字的块比 TF 不公平 |

**IDF 正是它补向量的地方。** 向量检索把一句话压成一个 512 维的稠密向量，
所有词被糊在一起 —— 「INVALID_INPUT」这种低频标识符在向量里几乎没有存在感，
而它旁边的「错误码」才是主导项。BM25 相反：`INVALID_INPUT` 在全库 1592 块里
只出现在 7 块，`log(1 + (N - df + 0.5)/(df + 0.5))` 给出很大的权重，
这 7 块会直接顶上来。

所以两者的强弱是**结构性的**，不是程度差异：

- BM25 强在：型号、编号、错误码、专有术语 —— **字面必须精确对上**
- 向量强在：同义、换述、跨语言、模糊描述 —— **字面完全不同也能对上**

## 为什么要用 `rank_bm25` 而不是自己写

自己写 BM25 只有三十行，但三个参数（`k1`/`b`）与 IDF 的变体（
Robertson / Lucene / ATIRE 三种都叫 BM25）会让"和标准实现对比"变得没有基准。
`rank_bm25.BM25Okapi` 是 Lucene 那一支的直译，与 Elasticsearch 默认接近 ——
用它，是为了让"BM25 的效果"这个说法有个可对齐的参照物。

## 持久化什么

`pickle` 存四样东西，**缺一样都不能重建**：

| 存什么 | 为什么必须存 |
| --- | --- |
| token 流（每个块的 token 元组） | 重新分词要 4 秒；而且**词典改了就会切出不同的流** |
| 块 id 列表 | 检索结果要能回查 `Chunk` |
| `k1` / `b` 与 avgdl | 参数变了分数就变了，不能从别的默认值猜 |
| 分词器指纹（`describe()`） | 词典一改，旧 token 流就与查询侧不一致 —— 这是最隐蔽的 bug |

第 4 项是**本周最重要的一行元数据**：没有它，词典改过而索引没重建时，
两侧分词结果不一致，而错误表现只是"BM25 效果不如预期"。

## 为什么不 pickle `BM25Okapi` 对象本身

它是可 pickle 的，但它内部缓存了 `idf` 字典与 `doc_freqs` ——
那是**从 corpus 派生的中间状态**。派生状态存进文件等于把缓存当源码：
版本升级、参数微调时，旧文件里的派生值会悄悄生效。
所以只存**原始输入**（token 流 + 参数），加载时用当前代码重新 build。
build 的成本实测约 0.4 秒，比"读到一个语义不明的旧对象"划算得多。
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.bm25.tokenize import describe, tokenize
from app.rag.schema import Chunk

DEFAULT_K1: float = 1.5
"""TF 饱和曲线的陡度。1.2~2.0 是 Lucene 的常规区间，1.5 是中位。

取 1.5 而不是 1.2 的理由：本项目块长 500 字，同一个术语在一个块里
出现 2~4 次是常态（术语表、参数表）。`k1` 越小饱和越快，
会让"出现 4 次"与"出现 2 次"几乎没差别 —— 在术语密集的表格块上会丢掉信号。
"""

DEFAULT_B: float = 0.75
"""长度归一强度。1.0 = 完全归一，0.0 = 不归一。

0.75 是 Lucene 默认，也是这几天的实测基准。**表头块/表格块的问题在这里**：
本项目 15.3% 的块以半句结尾，表格块动辄超长 —— `b` 调大能压住长块，
但也会压住"确实更长、确实更全"的块。W5 归因里 6 条表格块错例想说明的
正是这件事，它属于周六要验证的范围，不在这里拍。
"""

INDEX_VERSION: int = 1
"""文件格式版本。**改 pickload 结构就 +1**，让旧文件明确失效而不是悄悄错读。"""

EMPTY_TOKEN: str = "__empty__"
"""token 流为空时的占位。

BM25 对空文档会除零或给出 `nan`。**必须有占位**：一批块里只要有一个
token 流为空（纯符号块、纯数字表格块），整个 matrix 就会出现 `nan`，
而 `nan` 在排序里既不在前也不在后 —— 表现是"某几块永远不出现"，
查起来会以为是过滤把它们删了。
"""


@dataclass(frozen=True, slots=True)
class BM25Params:
    """可持久化的参数。**与 `BM25Okapi` 的构造参数一一对应**。"""

    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    def as_dict(self) -> dict[str, float]:
        return {"k1": self.k1, "b": self.b}


@dataclass
class BM25Index:
    """一份建好的 BM25 索引。**内存里长这样，落盘时也长这样**。"""

    chunk_ids: list[str]
    """每个块的内容派生 id（`Chunk.id`），顺序与 token 流一一对应。"""

    tokens: list[tuple[str, ...]]
    """每个块的 token 流。**这是唯一的重活，必须落盘**。"""

    params: BM25Params
    built_at: str
    build_ms: int
    tokenizer: dict[str, object] = field(default_factory=dict)
    """`tokenize.describe()` 的结果。词典一改，它就是"该重建了"的信号。"""

    _bm25: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.tokenizer:
            self.tokenizer = describe()

    def machine(self) -> object:
        """懒建 `BM25Okapi`。**派生状态不进文件**，见模块 docstring。"""
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            corpus = [list(stream) for stream in self.tokens]
            self._bm25 = BM25Okapi(corpus, k1=self.params.k1, b=self.params.b)
        return self._bm25

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """返回 (chunk_id, score) 降序。**分数不是余弦，不可与向量分数直接比**。

        这一点是整个混合检索最容易出错的地方：BM25 的分数是无界的正数
        （本项目实测 0~30），余弦相似度在 [-1, 1]。把两者丢进同一个排序
        等于让 BM25 单方面决定结果 —— 所以 `SearchHit.score` 的契约
        （"必须是余弦相似度"）**不能**被 BM25 复用，混排必须做归一化或 RRF。
        """
        if not self.chunk_ids:
            return []
        query_tokens = list(tokenize(query))
        if not query_tokens:
            # 无可匹配 token（全停用词/纯标点）：返回空，而不是"分数全 0 的随机块"。
            # 与向量侧的"空 query 返回空"是同一条纪律。
            return []
        scores = self.machine().get_scores(query_tokens)  # type: ignore[attr-defined]
        ranked = sorted(
            ((cid, float(score)) for cid, score in zip(self.chunk_ids, scores, strict=True)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]

    def stats(self) -> dict[str, object]:
        """给索引报告 / 体检用。**不含 token 流本身**（几万个字符，日志装不下）。"""
        total_tokens = sum(len(stream) for stream in self.tokens)
        return {
            "chunks": len(self.chunk_ids),
            "tokens": total_tokens,
            "avgdl": round(total_tokens / len(self.tokens), 2) if self.tokens else 0.0,
            "avg_tokens_per_chunk": round(total_tokens / len(self.chunk_ids), 2)
            if self.chunk_ids
            else 0.0,
            "params": self.params.as_dict(),
            "tokenizer": self.tokenizer,
            "build_ms": self.build_ms,
            "built_at": self.built_at,
            "version": INDEX_VERSION,
        }


@dataclass(frozen=True, slots=True)
class BuildReport:
    """一次建索引的结果。**耗时必须被记录下来**，任务点名要。"""

    path: Path
    chunks: int
    tokens: int
    build_ms: int
    size_bytes: int
    tokenizer: dict[str, object]
    reused: bool = False
    """True 表示"指纹没变，直接复用旧文件"—— 走的是快路径不是重建。"""

    def summary(self) -> dict[str, object]:
        return {
            "path": str(self.path.name),
            "chunks": self.chunks,
            "tokens": self.tokens,
            "build_ms": self.build_ms,
            "size_bytes": self.size_bytes,
            "size_kb": self.size_kb,
            "tokenizer": self.tokenizer,
            "reused": self.reused,
        }

    @property
    def size_kb(self) -> float:
        """落盘体积。**KB 保留一位**：BM25 索引是 400+ KB 量级，
        写成整数字节没人看得出它和"词典多 100 个词"有什么关系。"""
        return round(self.size_bytes / 1024, 1)


DEFAULT_INDEX_PATH: Path = Path("data") / "bm25" / "index.pkl"


def build(
    chunks: Sequence[Chunk],
    *,
    params: BM25Params | None = None,
    path: Path | None = None,
    force: bool = False,
) -> BuildReport:
    """对一批块建 BM25 索引并落盘。

    ## 为什么默认会"复用"

    `make build-bm25` 会被反复跑（改完文档想验证、CI、换机器）。
    每次都重切 1592 块要 4 秒，其中 90% 是分词。

    **复用的判据是分词器指纹 + 块 id 集合**，不是"文件存在"：
    词典改过（`dict_words` 变了）或块集合变了（切分参数变了），
    都必须重切 —— 旧 token 流配新查询分词器，就是那个最隐蔽的 bug。

    `force=True` 无视指纹强制重建。
    """
    started = time.perf_counter()
    target = path or DEFAULT_INDEX_PATH
    chunk_ids = [chunk.id for chunk in chunks]
    fingerprint = describe()

    if not force and target.exists():
        cached = load(target)
        if (
            cached is not None
            and cached.tokenizer == fingerprint
            and cached.chunk_ids == chunk_ids
            and (params is None or cached.params == params)
        ):
            size = target.stat().st_size
            return BuildReport(
                path=target,
                chunks=len(cached.chunk_ids),
                tokens=sum(len(s) for s in cached.tokens),
                build_ms=0,
                size_bytes=size,
                tokenizer=cached.tokenizer,
                reused=True,
            )

    streams: list[tuple[str, ...]] = []
    for chunk in chunks:
        stream = tokenize(chunk.text)
        streams.append(stream if stream else (EMPTY_TOKEN,))

    index = BM25Index(
        chunk_ids=chunk_ids,
        tokens=streams,
        params=params or BM25Params(),
        built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        build_ms=int((time.perf_counter() - started) * 1000),
    )
    # **build_ms 要在 pickle 之前定稿**：否则存进去的耗时不包含写盘，
    # 而人看到的"重建耗时"应该包含"能用"为止的全部时间。
    index.build_ms = int((time.perf_counter() - started) * 1000)

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(
            {
                "version": INDEX_VERSION,
                "chunk_ids": index.chunk_ids,
                "tokens": index.tokens,
                "params": index.params.as_dict(),
                "built_at": index.built_at,
                "build_ms": index.build_ms,
                "tokenizer": index.tokenizer,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    size = target.stat().st_size
    return BuildReport(
        path=target,
        chunks=len(chunk_ids),
        tokens=sum(len(s) for s in streams),
        build_ms=int((time.perf_counter() - started) * 1000),
        size_bytes=size,
        tokenizer=fingerprint,
    )


def load(path: Path | None = None) -> BM25Index | None:
    """读回索引。**文件不存在 / 版本不符 / 结构损坏都返回 `None`**，不抛。

    为什么用 `None` 而不是抛异常：调用方（检索链）要能表达
    "现在没有 BM25 可用，降级到纯向量" —— 那是一个正常状态，
    不是错误。而 `None` 逼着调用方显式处理它。
    """
    target = path or DEFAULT_INDEX_PATH
    if not target.exists():
        return None
    try:
        with target.open("rb") as handle:
            payload = pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, OSError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != INDEX_VERSION:
        return None
    params = payload.get("params") or {}
    try:
        return BM25Index(
            chunk_ids=list(payload["chunk_ids"]),
            tokens=list(payload["tokens"]),
            params=BM25Params(k1=float(params["k1"]), b=float(params["b"])),
            built_at=str(payload.get("built_at", "")),
            build_ms=int(payload.get("build_ms", 0)),
            tokenizer=dict(payload.get("tokenizer") or {}),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_or_build(
    chunks: Sequence[Chunk],
    *,
    params: BM25Params | None = None,
    path: Path | None = None,
) -> BM25Index:
    """要一份可用的索引：有就load，没有就build。**不会复用旧 token 流**。"""
    cached = load(path)
    if cached is not None:
        return cached
    build(chunks, params=params, path=path)
    loaded = load(path)
    if loaded is None:
        raise RuntimeError("BM25 索引刚写入却读不回来 —— 落盘路径或格式有问题")
    return loaded


__all__ = [
    "BM25Index",
    "BM25Params",
    "BuildReport",
    "DEFAULT_B",
    "DEFAULT_INDEX_PATH",
    "DEFAULT_K1",
    "EMPTY_TOKEN",
    "INDEX_VERSION",
    "build",
    "load",
    "load_or_build",
]
