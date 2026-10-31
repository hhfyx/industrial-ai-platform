"""ChromaDB 实现：**持久化**的向量库。

## 必答题：`distances` 在 cosine 空间下是什么

**cosine 距离 = 1 - 余弦相似度。越小越像。方向与相似度相反。**

这题搞反，**阈值逻辑就全错** —— 而且不会报错，只会"筛出来的是最不像的"。

实测（三个构造向量：同向 / 正交 / 反向，查询向量为 `[1,0,0]`）：

| 候选 | 与查询的余弦 | Chroma `distance` | `1 - distance` |
| --- | ---: | ---: | ---: |
| A 同向 | **1.0** | **0.0** | 1.0 |
| B 正交 | 0.0 | 1.0 | 0.0 |
| C 反向 | **-1.0** | **2.0** | -1.0 |

三条结论：

1. `distance = 1 - cosine`，**误差 0.00e+00**（不是近似式，是定义式）
2. **距离范围是 [0, 2]**，不是 [0, 1] —— 因为余弦可以是负数
3. 返回顺序是**距离升序**（A → B → C），即**越像越靠前**

> ⚠️ 最常见的误用：写 `if distance < 0.8: 认为命中`。
> 这句话在 cosine 空间里的意思是"**相似度 > 0.2 就算命中**"，
> 比你以为的宽松得多。反过来写 `if distance > 0.8` 筛出来的则是最不像的一批。

**所以本实现把口径统一成相似度再返回**（`1 - distance`）：
上层 `SearchHit.score` 永远是"越大越像"，不需要知道底下是 Chroma 还是别的。
换库时也不用重新推导一遍方向。

## 持久化：必须显式给 path

```python
chromadb.Client()                        # ❌ 内存 / 临时，进程退出就没了
chromadb.PersistentClient(path="data/chroma")   # ✅
```

**这是 W4 最高频的"数据怎么丢了"**。实测：写完 2 条 → `del` 掉 client →
新建 `PersistentClient` 打开同一个 path → `count()` 仍是 2，metadata 也还在。

默认路径由本模块给成 `data/chroma`（相对项目根），**不放在 C 盘用户目录**：
Chroma 默认会写到 `~/.cache/chroma` 之类的位置，那是"换台机器数据就没了"
以及"备份脚本没想到要带它"的典型来源。

## 为什么要传一个"永不被调用"的 embedding function

`create_collection()` 不传 `embedding_function` 时，Chroma 会**自己**加载一个
默认模型（ONNX MiniLM）。实测：那样会在建集合时**卡住下载**（本次探测直接跑到超时，
只能 kill）。两个后果：

1. 离线环境建不了库
2. 更糟的是，如果有人忘了传 `embeddings=`，Chroma 会**静默**用那个模型算向量 ——
   于是库里的向量一半来自 bge、一半来自 MiniLM，**检索结果变得无法解释**

所以传一个 `__call__` 直接 `raise` 的守卫函数：我们永远显式传向量，
那个函数不该被调用；一旦被调用就是要报错。实测守卫生效：
`query_texts=`（不传向量的写法）会立刻抛 AssertionError。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from app.rag.schema import Chunk, MetadataValue, restore_location
from app.rag.stores.base import SearchHit

logger = logging.getLogger(__name__)

# 默认持久化目录（相对项目根）。**必须显式给**，见模块 docstring。
# 放在 `data/` 下而不是 `~/.cache`：备份脚本会带上，换机器 `cp -r` 就行。
DEFAULT_PERSIST_PATH: str = "data/chroma"

# 默认 collection 名。Chroma 要求 3-512 字符、`[a-zA-Z0-9._-]`，
# 且**首尾必须是字母数字** —— `"t"` 太短会直接 ValueError（实测过）。
DEFAULT_COLLECTION: str = "industrial_rag"

# HNSW 距离度量。**必须是 cosine**：默认是 l2，而 l2 与余弦的排序
# 在归一化向量上虽然等价，但 distance 的数值含义完全不同（范围 [0,4]），
# 拿它当"相似度"用会得到一堆无法解释的数字。
DISTANCE_SPACE: str = "cosine"


def _make_guard_embedding_function() -> object:
    """构造守卫 EF。**延迟构造**是因为 chromadb 是可选依赖。

    继承 `chromadb.EmbeddingFunction` 而不是写个裸类：裸类能跑，但每次建 collection
    都会刷一条 `DeprecationWarning: legacy embedding function config:
    '_NoAutoEmbedding' object has no attribute 'is_legacy'`（实测 28 条/测试会话）。

    **"每次跑测试都刷一条废警告"的结果是所有人开始无视警告** ——
    那会让真正的警告也一起被忽略。所以宁可多继承一层。
    """
    from chromadb import EmbeddingFunction

    class _NoAutoEmbedding(EmbeddingFunction):
        """守卫：Chroma 想自己算向量时立刻炸。

        **必须实现三个方法，缺一个就会在特定调用路径上抛 `AttributeError`**（实测）：
        - `__call__`：写入时的路径
        - `embed_query`：`query_texts=` 时的路径（Chroma 1.5.9 会调它）
        - `name()`：`get_or_create_collection` 比对 EF 名时要用

        缺 `embed_query` 的症状是 `AttributeError` —— 那**也是炸**（守卫的目的达到了），
        但错误信息指向 Chroma 内部，看不出"我们禁止它自己算向量"这个意图。
        守卫要炸得让人看懂。

        **`name()` 必须稳定**：Chroma 会把 EF 名字记进 collection 配置，改名等于换 EF，
        重开旧库时会因为"EF 不一致"报错。这个名字是**已经落盘的契约**。
        """

        _MESSAGE = (
            "Chroma 试图自己算 embedding。本项目所有向量必须来自同一个 BGEEmbedder；"
            "混用两个模型会让检索结果无法解释（见 ChromaStore 的模块 docstring）。"
        )

        def __init__(self) -> None:
            # **必须重写 `__init__` 且不能只调 super()**：Chroma 的基类
            # `__init__` 里就是那句 `warnings.warn("does not implement __init__")` ——
            # 它靠"子类是否覆写"来发警告，而不是靠真正调用。
            # 所以这里什么也不做（而不是 super().__init__()），警告才消失。
            pass

        def __call__(self, input):  # noqa: ANN001, ANN204 - Chroma 的鸭子类型接口
            raise RuntimeError(self._MESSAGE)

        def embed_query(self, input):  # noqa: ANN001, ANN204
            """`query_texts=` 走这里。同样炸 —— 理由同上。"""
            raise RuntimeError(self._MESSAGE)

        @staticmethod
        def name() -> str:
            """EF 名。**改它等于让所有已落盘的库失效**，见类 docstring。"""
            return "never_auto_embedding"

        @staticmethod
        def build_from_config(config: dict) -> _NoAutoEmbedding:  # noqa: ANN401
            """Chroma 反序列化 EF 时用。守卫没有状态，所以忽略 config。"""
            return _NoAutoEmbedding()

        def get_config(self) -> dict:
            """序列化配置。空 dict 就够 —— 守卫没有任何参数。"""
            return {}

    return _NoAutoEmbedding()


def _project_root() -> Path:
    """项目根 = `app/rag/stores/chroma.py` 往上三级。

    **不用 `Path.cwd()`**：cwd 取决于你在哪启动 uvicorn（项目根 / tests 目录 /
    Docker 的 `/app`），用 cwd 拼相对路径会让"数据在哪个目录"随启动方式变化 ——
    那正是"数据怎么丢了"的第二种成因。
    """
    return Path(__file__).resolve().parents[3]


class ChromaStore:
    """ChromaDB 持久化向量库。

    **每次 `upsert` 都批量写**（一次 `collection.upsert(ids=[...], ...)`）：
    逐条 `add` 在 100 条时就能看出差距，在 15,000 条（5000 页）时是
    "几十秒 vs 几分钟"的差别 —— 而且中间失败会留下半份索引。
    """

    name = "chroma"

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        import chromadb

        if path is None:
            path = _project_root() / DEFAULT_PERSIST_PATH
        self.path = Path(path)
        # `exist_ok=True`：目录能创建就创建，已存在也不报错。
        # 不创建的话 PersistentClient 会自己建 —— 但那是在**别的 cwd** 下建的，
        # 于是"数据在哪"又变成玄学。
        self.path.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": DISTANCE_SPACE},
            embedding_function=_make_guard_embedding_function(),  # type: ignore[arg-type]
        )
        logger.info(
            "Chroma 就绪：%s（collection=%s，现有 %d 条）",
            self.path,
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------ 写

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        """批量写入。按 `store_id`（位置派生）覆盖。"""
        if len(chunks) != len(vectors):
            # **必须炸，不能截断**：静默截断会让"漏了一条"表现成
            # "索引成功，但检索就是差一点"。与 `InMemoryVectorStore` 一致。
            raise ValueError(
                f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致，"
                "向量与块的按位对应关系已失效"
            )
        if not chunks:
            return 0

        ids = [c.store_id for c in chunks]
        # id 重复时 Chroma 会报 "Duplicate ids"。这是**真实的错误**：
        # 说明同一份文档的同一页同一个 chunk_index 出现了两次，
        # 那是切分器给了重复 index —— 保留第一个之外的任何一个都是猜。
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(
                f"store_id 重复：{duplicates[:3]}。同一 (doc,page,chunk_index) 出现了两次，"
                "检查切分器的 index 分配"
            )

        payloads = [c.to_payload() for c in chunks]
        self._collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[p["metadata"] for p in payloads],
        )
        return len(ids)

    def delete_by_source(self, source: str) -> int:
        """按 source 删除。先查 ids 再 delete —— 这样能回报真实删除条数。"""
        existing = self._collection.get(where={"source": source}, include=[])
        ids = existing.get("ids") or []
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        logger.info("按 source 删除 %d 条：%s", len(ids), source)
        return len(ids)

    def count_by_source(self, source: str) -> int:
        """该 source 已有多少块。追加时用来算 `index_offset`。"""
        existing = self._collection.get(where={"source": source}, include=[])
        return len(existing.get("ids") or [])

    def clear(self) -> None:
        """清空。

        **用 `delete(where=...)` 而不是删 collection 再重建**：
        删 collection 会连它的 metadata（`hnsw:space=cosine`）一起丢掉，
        重建时若忘了再传就会被悄悄改成默认的 l2 —— 那是静默的口径变更。
        """
        existing = self._collection.get(include=[])
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        logger.info("Chroma 已清空（删除 %d 条）", len(ids))

    # ------------------------------------------------------------------ 读

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        *,
        where: dict[str, MetadataValue] | None = None,
    ) -> list[SearchHit]:
        """精确匹配过滤的检索。"""
        if top_k <= 0:
            raise ValueError("top_k 必须为正数")
        if self._collection.count() == 0:
            return []
        return self._query(vector, top_k=top_k, where=_build_where(where))

    def search_in_pages(
        self,
        vector: list[float],
        *,
        page_from: int,
        page_to: int,
        top_k: int = 5,
        where: dict[str, MetadataValue] | None = None,
    ) -> list[SearchHit]:
        """按页码区间检索（含两端）。

        **范围过滤必须写成 `$and` + 两个独立的比较**：
        Chroma 实测 `{"page": {"$gte": 1, "$lte": 3}}` 会抛
        `ValueError: Expected operator expression to have exactly one operator`。
        这不是文档里显眼的地方，是本格踩到的真实坑。

        参数顺序有意图：`page_from > page_to` 直接抛，而不是静默返回空 ——
        空结果是"这段没命中"，参数颠倒是"我写错了"，两者必须区分。
        """
        if page_from > page_to:
            raise ValueError(f"page_from({page_from}) 不能大于 page_to({page_to})")
        if top_k <= 0:
            raise ValueError("top_k 必须为正数")
        if self._collection.count() == 0:
            return []

        clauses: list[dict] = [{"page": {"$gte": page_from}}, {"page": {"$lte": page_to}}]
        if where:
            clauses.extend({key: value} for key, value in where.items())
        return self._query(vector, top_k=top_k, where={"$and": clauses})

    def count(self) -> int:
        return self._collection.count()

    def ids(self) -> list[str]:
        """导出全部 id（测试与排查用）。顺序不保证。"""
        return list(self._collection.get(include=[]).get("ids") or [])

    # ------------------------------------------------------------------ 内部

    def _query(
        self,
        vector: list[float],
        *,
        top_k: int,
        where: dict | None,
    ) -> list[SearchHit]:
        """统一的查询入口：把 Chroma 的 distance 换算成相似度。

        `include=["documents", "metadatas", "distances"]` 是**显式**的：
        不传 include 时 Chroma 也会返回它们（实测默认就含 `documents`），
        但那是实现细节 —— 显式写出来，将来默认值变了不会静默影响我们。
        """
        raw = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for store_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            meta = dict(metadata or {})
            page, pages = restore_location(meta)
            # 我们自己写进 metadata 的规范字段（见 `schema.Chunk.to_payload`）。
            # 其余键是 loader/切分器留下的业务元数据（`block`、`heading_level`…），
            # 要原样还给上层 —— **过滤掉它们会让"表格块"在检索后变回普通文本**，
            # 而那正是生成层决定"引用要不要标（表格）"的依据。
            reserved = {"source", "page", "pages", "chunk_index", "doc_id", "content_id"}
            hits.append(
                SearchHit(
                    chunk=Chunk(
                        text=text or "",
                        page=page,
                        source=str(meta.get("source", "")),
                        # `content_id` 是内容派生的（见 schema.to_payload）。
                        # 从库里读回来时直接用它，**不要重新算** —— 重新算要拿
                        # (source,page,text) 拼一遍，而 text 可能是被截断过的。
                        # 兜底用 `store_id`：老版本写进去的记录没有 content_id，
                        # 那时至少要有个可用的标识，而不是空字符串。
                        id=str(meta.get("content_id", store_id)),
                        metadata={k: v for k, v in meta.items() if k not in reserved},
                        index=int(meta.get("chunk_index", 0)),
                        pages=pages,
                    ),
                    # **这一行就是必答题的落地**：distance 越小越像，
                    # 换成相似度（越大越像）再交出去。
                    score=1.0 - float(distance),
                )
            )
        return hits


def _build_where(where: dict[str, MetadataValue] | None) -> dict | None:
    """把精确匹配的 dict 原样交给 Chroma。

    这层包装存在的意义是**留一个地方做校验**：`where` 的值必须是扁平标量
    （Chroma 不接受嵌套），在这里拦住比让 Chroma 抛内部错误好定位。
    """
    if not where:
        return None
    for key, value in where.items():
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(
                f"where[{key!r}] 必须是 str/int/float/bool，收到 {type(value).__name__}；"
                "Chroma 的 where 不支持嵌套结构"
            )
    return dict(where)


__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_PERSIST_PATH",
    "DISTANCE_SPACE",
    "ChromaStore",
]
