"""检索段（Retrieval）：query 向量化 → 相似度 → TopK。

**这一段是"每次都跑、必须快"的**。它和索引段的区别写在 `indexing.py` 顶部；
这里只强调一件事：**检索段不产生新信息，它只负责"从已有的块里挑"**。
任何"检索时顺手做点加工"的想法（重写 query、二次摘要、在线聚类）都属于
"加了一层看不见的延迟"，要加就得单独一轮、单独一轮的指标。

## 三步各自的输入输出

    ① query 向量化  str                → list[float]
    ② 相似度        vector + store     → list[SearchHit]（带分数）
    ③ TopK          list[SearchHit]    → list[Chunk]（可再按预算裁）

**① 必须用和索引完全相同的 embedder**，这不是约定而是硬约束：
查询向量和文档向量不在同一个空间时，余弦相似度算出来的是噪声，
而且**不会报错**。`Retriever` 因此在构造时要求显式传入 embedder，
不给"忘了传就默认一个"的机会 —— 默认值在这是有害的。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.rag.embedding import Embedder
from app.rag.filters import FilterSchema, guard_where, schema_from_store
from app.rag.schema import Chunk, MetadataValue
from app.rag.settings import DEFAULT_TOP_K as DEFAULT_TOP_K_FROM_CONFIG
from app.rag.settings import RagConfig, get_rag_config
from app.rag.stores import SearchHit, VectorStore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = DEFAULT_TOP_K_FROM_CONFIG
"""默认取几条。**这是一个 re-export，不是独立定义。**

W5 周三之前这里有个自己的 `DEFAULT_TOP_K = 5`，而 `ChunkSize` / UI 滑块 /
CLI 各有自己的一份。今天全部收敛到 `app/rag/settings.RagConfig`：

    app/rag/settings.py  ← 唯一来源（可从 RAG_TOP_K 环境变量覆盖）
        ↓ re-export
    app/rag/retrieval.py ← 这个常量（保持老 import 可用）

**为什么保留这个别名而不是让所有人改 import**：`from app.rag.retrieval import
DEFAULT_TOP_K` 已经出现在测试、CLI 与文档里。删掉它等于一次纯机械的破坏性改动，
而收益只是"少一个名字"。保留别名、把真值指向配置，是零成本的做法。

取 5 的理由（边际收益拐点）见 `settings.DEFAULT_TOP_K` 的 docstring——
那里有 topk_scan 的实测表。
"""


def default_top_k() -> int:
    """默认 top_k 的**取值出口**。

    存在的意义不是多一层函数，而是让"谁在用默认值"这件事可被搜索到：
    调用方写 `default_top_k()` 时，W5 改参数只改一处；
    写死 `top_k=5` 的地方则在评审时一眼能看见。`tests/test_rag_skeleton.py`
    有一条断言守住"CLI / UI / QAEngine / Retriever 四处一致"。
    """
    return DEFAULT_TOP_K


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """检索段的完整产物。"""

    query: str
    hits: tuple[SearchHit, ...]
    top_k: int
    elapsed_ms: int = 0
    where: dict[str, MetadataValue] = field(default_factory=dict)
    """**实际生效**的过滤条件（已过校验）。空 dict 表示没过滤。

    为什么要把生效条件带回来：调用方传的可能是个拼出来的 dict，
    而"检索结果为空"时第一件要确认的就是"过滤条件是不是我想要的"。
    不给它，界面上只能显示"没找到"，而用户无法判断是文档没导入还是条件写错了。
    """

    @property
    def chunks(self) -> list[Chunk]:
        """只要块（丢掉分数）。分数跨库不可比，见 `stores/base.SearchHit`。"""
        return [hit.chunk for hit in self.hits]

    @property
    def is_empty(self) -> bool:
        return not self.hits

    def scores(self) -> list[float]:
        return [hit.score for hit in self.hits]


class Retriever:
    """检索器：query → TopK 命中块。

    ## 参数从哪来（W5 周三改）

    构造时可以显式传 `top_k` / `config`，都不传就读 `get_rag_config()`。
    **`retrieve()` 的签名没有变** —— 这是任务里"签名不变"那条的字面要求，
    也是它对调用方的全部承诺：`retrieve(query)` / `retrieve(query, top_k=8)`
    的行为和昨天一样，只是"不传时取几"现在有正式出处。
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        top_k: int | None = None,
        config: RagConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        # **`config` 与 `top_k` 同时给且不一致时抛**：与 `retrieve(top_k=, k=)`
        # 的规矩一致 —— 两个来源给了不同答案，猜一个会得到事后无法解释的配置。
        self.config = config or get_rag_config()
        if top_k is not None:
            if config is not None and top_k != config.top_k:
                raise ValueError(
                    f"top_k({top_k}) 与 config.top_k({config.top_k}) 不一致，无法判断取几条"
                )
            self.config = self.config.with_overrides(top_k=top_k)
        if self.config.top_k <= 0:
            raise ValueError("top_k 必须为正数")
        self.top_k = self.config.top_k
        # 过滤 schema 的缓存。**懒算**：绝大多数调用不带 where，
        # 为它们付一次 O(n) 全量扫描是白花。
        self._filter_schema: FilterSchema | None = None

    @property
    def filter_schema(self) -> FilterSchema:
        """当前索引的过滤维度（首次访问时算一次并缓存）。

        **缓存而不是每次重算**：Chroma 上它是 O(n) 全量拉元数据（实测 532 块 ~20ms）。
        索引变了要手动 `refresh_filter_schema()` —— 这个类不管索引生命周期
        （`Indexer` 管），假装它管会是"缓存静默过期"的来源。
        """
        if self._filter_schema is None:
            self._filter_schema = schema_from_store(self.store)
        return self._filter_schema

    def refresh_filter_schema(self) -> FilterSchema:
        """索引变更后重算过滤维度。**不调用会让新文档的 source 无法被过滤到。**"""
        self._filter_schema = None
        return self.filter_schema

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        k: int | None = None,
        where: dict[str, MetadataValue] | None = None,
    ) -> RetrievalResult:
        """检索。

        ## 参数名契约：`top_k` 是规范名，`k` 是它的别名

        W4 的开工清单里写的是 `retrieve(query, k=5)`，而实现里叫 `top_k` ——
        两者都留，是因为两边都已经有真实调用方：项目内部一路用 `top_k`
        （`QAEngine`、CLI、UI 都是 `top_k=`），而外部/文档口径写 `k`。
        只留一个，另一方就要改调用点；而这类"改个参数名"的改动最容易
        在某个没被测试覆盖的分支上变成 `TypeError`。

        **两者同时传且不相等时直接抛** —— 不静默取其一。同时传两个不同的
        条数说明调用方自己也没想清要几条，猜一个会得到"检索 5 条、上下文按 3 条拼"
        这种事后无法解释的结果。

        **空 query 直接返回空结果，不调 embedder**：空字符串的向量是零向量，
        零向量与任何向量的余弦都是 0，于是会返回一堆"分数全是 0"的块 ——
        看起来检索成功，实际是随便挑的。让它在源头就返回空，问题才能被看见。

        ## `where`：过滤条件在检索前校验（W5 周三加）

        传入的 `where` 会先过 `guard_where()`。写错键名/文档名/用了当前策略
        不产生的字段 → **抛异常**，而不是返回空列表。

        为什么必须这样：没有校验时，`where={"source": "打错的文档名.md"}`
        与 `where={"section_path": "x"}`（recursive 策略下）都返回 0 条，
        与"这份文档里确实没有相关内容"**完全无法区分**。
        实测在 532 块的库上，四种写错的方式全都静默返回空。
        """
        started = time.perf_counter()
        if k is not None and top_k is not None and k != top_k:
            raise ValueError(f"top_k({top_k}) 与 k({k}) 同时给出且不相等，无法判断取几条")
        requested = top_k if top_k is not None else k
        k_value = self.top_k if requested is None else requested
        if k_value <= 0:
            raise ValueError("top_k 必须为正数")

        # 过滤条件先校验。**放在空 query 检查之前**：参数写错应该在
        # "有没有查"之前暴露 —— 否则空 query 会掩盖一个写错的 where。
        checked_where = guard_where(self.filter_schema, where) if where else {}

        text = (query or "").strip()
        if not text:
            return RetrievalResult(query=query, hits=(), top_k=k_value, elapsed_ms=_ms(started))

        # ① query 向量化：与索引共用 embedder，见模块 docstring。
        #
        # **调 embed_query 而不是 embed**：bge 这类模型要求 query 侧加检索指令前缀
        # （"为这个句子生成表示以用于检索相关文章："），passage 侧不加。
        # 忘加不会有任何报错，只会让命中率低一点 —— 而"低一点"最容易被
        # 归因到切分/top_k/模型不行。所以把前缀这件事放进接口，调用方碰不到。
        query_vector = self.embedder.embed_query([text])[0]

        # ②③ 相似度 + TopK 由 store 一次完成（store 才知道怎么做相似度）
        hits = self.store.search(query_vector, top_k=k_value, where=checked_where or None)

        result = RetrievalResult(
            query=query,
            hits=tuple(hits),
            top_k=k_value,
            elapsed_ms=_ms(started),
            where=checked_where,
        )
        logger.debug(
            "检索完成：%r → %d 条（%d ms，最高分 %.4f，过滤 %s）",
            text[:40],
            len(hits),
            result.elapsed_ms,
            hits[0].score if hits else 0.0,
            checked_where or "无",
        )
        return result


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


__all__ = ["DEFAULT_TOP_K", "RetrievalResult", "Retriever", "default_top_k"]
