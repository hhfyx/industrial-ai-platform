"""元数据过滤：**把"过滤后没结果"和"过滤条件写错了"分开**。

## 这个模块解决的具体故障

`VectorStore.search(vector, where={...})` 的语义是"精确匹配的元数据过滤"。
问题在于它对**写错的条件**和**真的没有匹配**返回同一个东西：空列表。

实测（`chunk_chars=500`，532 块，hashing embedder）：

| where | 返回条数 | 真实含义 |
| --- | ---: | --- |
| `{}` | 5 | 正常 |
| `{"source": "sse.md"}` | 5 | 正常 |
| `{"source": "不存在的文档.md"}` | **0** | 文档名打错了 / 该文档确实不在库里 |
| `{"section_path": "x"}`（recursive 策略下） | **0** | **该字段根本不存在** —— 条件永远不成立 |
| `{"blockk": "table"}` | **0** | **键名拼错了** |
| `{"block": "table"}` | 5 | 正常 |

后三行是灾难：用户以为"这份文档里没有相关内容"，实际是"你查询的字段不存在"。
更糟的是 **`section_path` 这一条不是笔误，是真实的接口陷阱** ——
它只由 `heading` 策略写入（见 `chunking/heading.py`），
而默认策略是 `recursive`（`chunking/factory.DEFAULT_STRATEGY`）。
所以在默认配置下，**任何按章节过滤的调用都会静默返回空**。

## 解决办法：先校验，再检索

`guard_where()` 在检索**之前**做三件事：

1. **键必须在库里出现过**。键名拼错 → 立刻抛 `UnknownFilterKey`，附上可用键列表。
2. **值必须在库里该键的取值集合里**。文档名打错 → 抛 `UnknownFilterValue`，
   附上接近的候选（若只有一个候选就直接说是哪一个）。
3. **字段存在但当前索引里全为空**（`section_path` 在 recursive 下）→
   抛 `FilterKeyNotPopulated`，说清"这个字段需要 X 策略才有"。

这不违反"检索段不产生新信息"的原则：校验读的是**索引时已经写下的元数据**，
不重写 query、不做二次摘要。它只是把"沉默的失败"变成"响亮的失败"。

## 为什么校验要付一次扫描的代价

`guard_where` 需要知道"库里有那些键/值"。两个来源：

- `InMemoryVectorStore`：内存遍历，几毫秒。
- `ChromaStore`：`collection.get(include=["metadatas"])` 全量拉元数据。

532 块时实测约 20 ms，**比一次 embed_query 还便宜**。
但这是 O(n) 的，不能每次提问都做。

所以它有两个入口：

| 入口 | 何时用 | 代价 |
| --- | --- | --- |
| `filter_schema(store)` | 建索引后 / 启动时一次，缓存起来 | O(n) 一次 |
| `guard_where(schema, where)` | 每次检索，纯内存比对 | O(#keys) |

生产路径用后者（schema 跟着 store 缓存），实验脚本用前者图省事。

## `section_path` 的正确打开方式

它不是"某个策略的附属品"，而是**只有 heading 策略才能提供的定位维度**。
`docs/retrieval-tuning-final.md` 里那条 `heading` 策略的价值，正落在这里：

    recursive → 能按 source / block 过滤，**不能按章节**
    heading   → 额外有 section_path / heading_title，**能按章节**

如果线上要用章节过滤，索引就必须用 `heading`。这不是可选的优化，
是"要不要这个能力"的决定 —— 而 `guard_where` 的任务是让这个决定无法被忽略。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from app.rag.schema import Chunk, MetadataValue
from app.rag.stores import VectorStore

# 这几个键由 `Chunk.to_payload()` 统一写入，任何策略都会有。
# 其余键（`block` / `heading_level` / `section_path` …）取决于 loader 与切分策略。
CORE_KEYS: frozenset[str] = frozenset({"source", "page", "pages", "chunk_index", "content_id"})

# 只有 heading 策略会写的键。**写在这里是为了报错时能说清"缺什么策略"**。
HEADING_ONLY_KEYS: frozenset[str] = frozenset(
    {"section_path", "heading_title", "section_chars", "section_docs", "section_part"}
)


class FilterError(ValueError):
    """过滤条件的基类。**继承 ValueError**：这些都是"调用方传错了参数"。"""


class UnknownFilterKey(FilterError):
    """键名在索引的元数据里从未出现过 —— 通常是拼写错误或字段不适用。"""


class UnknownFilterValue(FilterError):
    """键存在，但没有任何块取这个值 —— 通常是文档名/章节名写错了。"""


class FilterKeyNotPopulated(FilterError):
    """键在 schema 里登记过，但**当前索引里所有块都是空值**。

    与 `UnknownFilterValue` 分开，因为它的成因完全不同、修法也完全不同：
    前者是"你写错了"，这个是"你的索引根本没有这个维度"。
    """


@dataclass(frozen=True, slots=True)
class FilterSchema:
    """索引里实际存在的过滤维度。**建索引后算一次，之后只读。**"""

    values: Mapping[str, frozenset[str]]
    """键 → 该键在库里出现过的全部取值（已转成字符串便于比较）。"""

    null_ratio: Mapping[str, float]
    """键 → 空值占比。用于识别"字段存在但全为空"（`section_path` 在 recursive 下）。"""

    total_chunks: int

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self.values)

    def describe(self) -> str:
        """一行摘要，给日志和 CLI `--debug` 用。"""
        parts = [
            f"{key}({len(values)} 值)"
            for key, values in sorted(self.values.items(), key=lambda kv: -len(kv[1]))
        ]
        return f"{self.total_chunks} 块的可过滤维度：" + "、".join(parts)


def _stringify(value: object) -> str:
    """把元数据值统一成字符串比较。

    **必须统一**：`where={"page": 3}`（int）与库里存的 `3`（int）能匹配，
    但 `renderer` 传来的可能是 `"3"`。统一成 str 后两种写法都能过校验 ——
    而**向量库那边仍是原始类型的精确匹配**，所以校验通过不等于检索一定命中。
    这个边界写在 `guard_where` 的返回说明里。
    """
    return str(value)


def build_schema(chunks: Sequence[Chunk]) -> FilterSchema:
    """从块列表算 schema。**这是唯一知道"库里有什么维度"的地方。**"""
    values: dict[str, set[str]] = {}
    nulls: dict[str, int] = {}
    total = len(chunks)
    for chunk in chunks:
        metadata = chunk.to_payload()["metadata"]
        for key, value in metadata.items():
            # `pages` 是逗号拼接的字符串，空串表示"不知道覆盖范围"——算空值。
            text = _stringify(value)
            values.setdefault(key, set())
            nulls.setdefault(key, 0)
            if text == "" or value is None:
                nulls[key] += 1
            else:
                values[key].add(text)
    return FilterSchema(
        values={key: frozenset(items) for key, items in values.items()},
        null_ratio={key: (nulls[key] / total if total else 0.0) for key in values},
        total_chunks=total,
    )


def schema_from_store(store: VectorStore) -> FilterSchema:
    """从 store 直接算 schema（O(n) 一次）。

    两条路径：
    - `InMemoryVectorStore` 有 `_rows`（测试与本地实验走这条，微秒级）
    - 其余（Chroma）走 `collection.get(include=["metadatas"])`

    **不假设每个 store 都能高效全量拉**：这是"建索引后一次"的操作，
    它的代价远小于一次 embed，所以不需要为它设计增量维护。
    """
    rows = getattr(store, "_rows", None)
    if rows is not None:
        # 内存库：直接读，避免为了拿元数据再走一遍 Chroma 风格的接口。
        chunks = [chunk for chunk, _vector in rows.values()]
        return build_schema(chunks)
    describe = getattr(store, "describe_metadata", None)
    if describe is not None:
        return describe()
    raise TypeError(
        f"{type(store).__name__} 不支持提取过滤 schema；"
        "内存库直接可用，Chroma 需要实现 describe_metadata()"
    )


def _suggest(value: str, candidates: Iterable[str], *, limit: int = 3) -> list[str]:
    """给"写错了"的情况提候选。**不引入 difflib**：这里候选集很小，
    而且真正有用的是"最接近的几个"，不是编辑距离的精确排序。"""
    lowered = value.lower()
    scored: list[tuple[int, str]] = []
    for candidate in candidates:
        cand = candidate.lower()
        if lowered in cand or cand in lowered:
            scored.append((0, candidate))
        else:
            # 共同字符数当粗略相似度：中文没有词边界，编辑距离对小串噪声太大。
            overlap = len(set(lowered) & set(cand))
            if overlap:
                scored.append((1, candidate))
    scored.sort(key=lambda item: (item[0], -len(item[1])))
    return [candidate for _, candidate in scored[:limit]]


def guard_where(
    schema: FilterSchema,
    where: Mapping[str, MetadataValue] | None,
) -> dict[str, MetadataValue]:
    """校验过滤条件，**不合法就抛**，而不是让它变成空结果。

    返回可直接传给 `store.search(where=...)` 的 dict（原样返回，不改类型）。

    三种失败各自的错误信息都必须能直接指导下一步：

    | 抛什么 | 什么时候 | 错误信息里有什么 |
    | --- | --- | --- |
    | `UnknownFilterKey` | 键不在 schema 里 | 可用键列表 + 拼写候选 |
    | `FilterKeyNotPopulated` | 键在但全部为空 | **需要哪种切分策略** |
    | `UnknownFilterValue` | 值不在该键的取值里 | 接近的候选值 |
    """
    if not where:
        return {}

    checked: dict[str, MetadataValue] = {}
    for key, value in where.items():
        if key not in schema.values:
            available = sorted(schema.values)
            raise UnknownFilterKey(
                f"过滤键 {key!r} 不在当前索引里。可用键：{'、'.join(available)}。"
                + (
                    f" 你是不是想用 {_suggest(key, available)}？"
                    if _suggest(key, available)
                    else ""
                )
            )

        allowed = schema.values[key]
        if not allowed:
            # 键存在但没有任何非空取值 —— `section_path` 在 recursive 策略下就是这种。
            hint = (
                "该键只由 heading 切分策略写入，当前索引用的是别的策略"
                if key in HEADING_ONLY_KEYS
                else "该键当前全为空值"
            )
            raise FilterKeyNotPopulated(
                f"过滤键 {key!r} 存在但当前索引里 {schema.null_ratio.get(key, 0):.0%} 的块都是空值：{hint}。"
                f" 按它过滤永远返回空，且**不会报错** —— 所以这里拦住。"
            )

        text = _stringify(value)
        if text not in allowed:
            suggestions = _suggest(text, allowed)
            raise UnknownFilterValue(
                f"过滤键 {key!r} 的取值 {value!r} 在库里不存在。"
                + (f" 接近的候选：{'、'.join(suggestions)}。" if suggestions else "")
                + f"（该键共 {len(allowed)} 个取值）"
            )

        checked[key] = value
    return checked


def resolve_section_filter(
    schema: FilterSchema,
    *,
    section: str,
    strategy: str,
) -> dict[str, MetadataValue]:
    """按章节名过滤的**唯一正确入口**。

    `section_path` 是层级路径（`A > B > C`），而人说的"某一章"通常只给一层。
    这里做的是**前缀匹配到最具体的哪一节**，而不是要求调用方写全路径。

    为什么需要这个函数：直接 `where={"section_path": "架构"}` 会失败，
    因为库里存的是 `环境搭建与开发流程 > 架构 > 目录` 这种完整路径。
    让人先查到全路径再传进来，等于把"查 schema"的负担丢给 UI。
    """
    if strategy != "heading":
        raise FilterKeyNotPopulated(
            f"按章节过滤需要 heading 切分策略，当前是 {strategy!r}。"
            "section_path 只有 heading 策略会写（见 chunking/heading.py），"
            "recursive 索引里不存在这个维度 —— 按它过滤会静默返回空。"
        )
    allowed = schema.values.get("section_path", frozenset())
    if not allowed:
        raise FilterKeyNotPopulated("索引里没有 section_path（所有块都为空）")

    matches = [path for path in allowed if section in path]
    if not matches:
        raise UnknownFilterValue(
            f"没有哪个 section_path 包含 {section!r}。"
            f"共 {len(allowed)} 个章节，例如：{'、'.join(sorted(allowed)[:3])}"
        )
    # 多个匹配时取**最短的那条**：它是最靠近用户所说"某一章"的层级，
    # 而更长的路径是它的子节。取最长会把范围缩到一个小节，反而漏掉同章内容。
    best = min(matches, key=len)
    return {"section_path": best}


__all__ = [
    "CORE_KEYS",
    "HEADING_ONLY_KEYS",
    "FilterError",
    "FilterKeyNotPopulated",
    "FilterSchema",
    "UnknownFilterKey",
    "UnknownFilterValue",
    "build_schema",
    "guard_where",
    "resolve_section_filter",
    "schema_from_store",
]
