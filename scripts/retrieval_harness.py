"""参数网格搜索的共用底座：**语料怎么造、索引怎么建、命中率怎么算**。

这个文件的存在理由是 `scripts/token_audit.py` 与 `scripts/grid_search.py`
必须**用完全同一个语料、同一种切分、同一套命中判定**。三处任何一处漂移，
两边报出来的"越界占比"与"命中率"就无法互相解释 —— 而那种不一致最难查，
因为两个脚本各自都"跑通了"。

## 一、语料为什么是"每份文档合并成一段正文"

`scripts/chunking_samples.py` 用的是 loader 原样输出（1530 个 `Document`），
那适合**人工读切分质量**：能看见每个小节被切成什么样。

但网格搜索要测的是"**chunk_chars 从 300 改到 1000 时切出来的块有什么变化**"，
而 loader 把 markdown 按标题切成了 1530 段（平均 172 字）。在这种输入上，
chunk_chars=1000 的切分结果与 500 **几乎一样**（实测：1563 vs 1606 块，差 2.7%），
因为每段本来就短于预算，切分器根本没机会合并。

也就是说：**直接用 loader 粒度跑网格，测不出网格的效果**。
所以这里先把每份文档的**正文段**按原顺序合并成一个 `Document`，
再交给三种策略。这模拟的是"用户上传了一份完整文档"这个真实场景。

代价必须写清：合并丢掉了 loader 给的标题边界，`heading` 策略因此退化
（它靠 `heading_level` 元数据找节边界，合并后没有元数据可依）。
所以**网格搜索只用 `recursive` 策略** —— 它的输入要求最简单（只要分隔符分布），
合并不会让它静默变差。跨策略比较是 `chunking_samples.py` 的活，不是这个网格的活。

表格与代码块**不参与合并**：它们在 `recursive` 策略下整块不切（契约要求），
合并进正文会让一张 1300 字的表被当成正文切开 —— 那正好违反被验证的性质。
所以它们单独作为 `Document` 传入，原地成块。

## 二、为什么命中判定看"锚点片段是否落在返回的块里"

`eval/retrieval-gold.jsonl` 每条的 `anchor` 是一段**逐字原文**。
判定一条查询"命中"的方式是：

    返回的 top_k 块里，有没有哪一块的正文包含这个 anchor

这是**可复现、不依赖模型判断**的口径。它测的不是"答案对不对"，
而是"原文有没有被检出来" —— 调 chunk/overlap 时后者才是被影响的变量，
前者还叠着生成质量，混进来就分不清是谁的问题了。

另外两条口径一起记下来，否则数字不可比：

- **top_k=5**，取全项目唯一默认值（`retrieval.DEFAULT_TOP_K`），不在这里另定一个。
- 同样的算子在 `chroma_acceptance.py` 里叫 `position`（top1/top2/top3/未命中）。
  这里报两个数：`hit@5`（有没有命中）与 `mrr`（命中排第几的倒数，
  区分"排第一"和"勉强挤进第五"）。

## 三、"重复内容率"怎么算（细调 overlap 的主指标）

任务里写的是"TopK 里来自相邻 chunk 的比例"。这里按**可判定的方式**实现：

对每条查询的 top_k 命中列表，统计有多少**无序对** (i, j) 满足
`同一份文档 && abs(chunk_index 之差) == 1`。分母是总对数。

`chunk_index` 用 `Chunk.index`（同一 `Document` 内部的序号），
所以"相邻"是**切分意义上的相邻**，不是"在库里 id 相邻"。

**为什么这个数越高越糟**：overlap>0 时相邻两块共享一段尾巴，
两块的向量会高度相似，于是一起挤进 top_k —— 上下文预算被同一段内容占两次
（`splitters/delimiter.py` 的 docstring 把这条列为重叠的第三个代价）。
它是 overlap 唯一**直接产生**的坏处，所以必须单独测。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import Document  # noqa: E402
from app.rag.chunking import ChunkingConfig, build_strategy  # noqa: E402
from app.rag.loaders import load_document  # noqa: E402
from app.rag.schema import Chunk  # noqa: E402

GOLD_PATH: Path = PROJECT_ROOT / "eval" / "retrieval-gold.jsonl"

# 索引策略固定为 recursive。理由见模块 docstring 第一节（合并会破坏 heading 的元数据依赖）。
INDEX_STRATEGY: str = "recursive"

# 参与检索评测的源文件白名单。**与 `eval/retrieval-gold.jsonl` 的 source 必须一致** ——
# 语料里缺了某份文档，那条 gold 必然"未命中"，而那会被误读成"参数不好"。
#
# ## ⚠️ W5 周六发现：这个白名单会让"召回率"不可比
#
# 它只有 13 份文档，而 `strategy_compare.load_raw_corpus` 扫的是**全部** docs。
# 于是同一套 50 条题在两个脚本里跑出完全不同的召回率：
#
#     白名单 13 份 → 717 段 → 764 块 → Recall@5 40.0%（但误拒 9）
#     全部 docs    → 1820 段 → 1835 块 → Recall@5 40.0%
#
# **库越小越容易命中**（竞争者少），所以"用小子集调出来的参数"
# 在完整库上不成立。这是评测集最容易犯的一类错：**测的不是线上那个库**。
#
# 现在 `load_raw_corpus` 默认扫全部 docs（与 `strategy_compare` 一致）；
# 白名单只在**需要复现历史数字**时显式传入。
CORPUS_SOURCES: tuple[str, ...] = (
    "README.md",
    "docs/sse.md",
    "docs/cost.md",
    "docs/chunking-quality.md",
    "docs/chroma-store.md",
    "docs/design-decisions.md",
    "docs/embedding.md",
    "docs/rag-architecture.md",
    "docs/rag-query-chain.md",
    "docs/prompt-patterns.md",
    "docs/SETUP.md",
    "docs/document-parsing.md",
    "docs/api-spec.md",
)


ANALYSIS_DOCS: frozenset[str] = frozenset(
    {
        # 本周（W5）新增的分析报告 —— 它们在复述结论，是最强的语义竞争者
        "retrieval-tuning-final.md",
        "retrieval-tuning-token-grid.md",
        "retrieval-params.md",
        "eval-50.md",
        "strategy-comparison.md",
        # 历史周报与清单：同样是"对项目的描述"，不是被检索的业务文档
        "w4-checklist.md",
    }
)
"""**不进检索语料**的文档：分析报告、周报、清单。

它们描述"这个项目做了什么"，而不是"用户要查的知识"。
混进去的后果见 `full_corpus_sources` 的说明（17% 的 top5 被污染）。

## `retrieval-tuning-token-grid.md` 是 W6 补进来的（漏排了）

W5 周日只排除了 `retrieval-tuning-final.md`，漏了同一性质的
`retrieval-tuning-token-grid.md`（84 段 / 86 块）。它是**周三的 token 审计 +
chunk/overlap 网格报告**，和 final 一样在复述结论、复述项目做了什么 ——
判据（"分析报告"）从来没区分过这两份，只是当时只想到一份。

漏排的后果不是"数字偏大"这么简单，而是**同一份 harness 两次跑出两个库**：
- 现行 `full_corpus_sources()`（漏排）→ 1597 段 / 1678 块
- 按判据排干净 → 1513 段 / 1592 块

两个口径下 Recall@5 都是 40.0%（16/40）、MRR 都是 0.2402 —— **主指标不变**，
变的只是块数/耗时这类元数据列。所以这是一次**零结论风险的纯口径统一**：
补上它不会推翻任何 W5 结论，只会让"块数"这一列第一次可复现。

历史文件名 `retrieval-tuning.md`（周三版本，后改名为 `-token-grid.md`）
如果哪天被恢复，同样要排 —— 判据是"是不是分析报告"，不是名单里有没有。
"""


def full_corpus_sources() -> tuple[str, ...]:
    """**全部** md 文档（docs/ 递归 + 根 README），**排除分析文档**。

    这是"线上的库"实际包含的东西 —— 用户上传什么就在库里，
    不会因为我们只关心 13 份就少放。用它做评测才和线上同构。

    ## ⚠️ 为什么必须排除本项目的分析文档

    W5 周日实测发现：`docs/retrieval-tuning-final.md`、`retrieval-params.md`、
    `eval-50.md`、`strategy-comparison.md` 这些**我自己写的分析文档**
    也在 `docs/` 下，于是它们进了检索库。

    后果（实测）：34 条错例的 top5 共 170 个槽位里，**29 个（17%）**
    来自这些分析文档。也就是说"检索失败"里有一批是**被我自己的总结文字干扰**的
    —— 那些文档在复述同样的结论，语义上和所有问题都近，是最强的竞争者。

    这是一个非常隐蔽的评测污染：不报错、让召回率**随我写文档的多少而下降**、
    越努力写文档评测越差 —— 而人会去调参数。

    判据用**文件名白名单**而不是"git 是否跟踪"：后者会把新加的文档漏进来，
    而新文档同样会造成污染。

    **按名字排序**：保证不同机器、不同文件系统顺序下切出来的块序一致
    （块序影响 `store_id`，进而影响同分时的排序）。
    """
    docs = sorted(
        f"docs/{p.relative_to(PROJECT_ROOT / 'docs').as_posix()}"
        for p in (PROJECT_ROOT / "docs").rglob("*.md")
        if p.name not in ANALYSIS_DOCS
    )
    return ("README.md", *docs)


@dataclass
class Corpus:
    """一次加载的语料。**prose 与 fixed 分开存**，因为两者对切分参数的反应完全不同。"""

    prose: list[Document] = field(default_factory=list)
    """每份文档的正文段合并成一段后的 Document。受 chunk_chars / overlap 影响。"""

    fixed: list[Document] = field(default_factory=list)
    """表格块与代码块。`recursive` 策略下整块不切，与参数无关。"""

    source_chars: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def all_docs(self) -> list[Document]:
        return [*self.prose, *self.fixed]

    @property
    def total_chars(self) -> int:
        return sum(len(doc.text) for doc in self.all_docs)


def load_corpus(sources: Sequence[str] = CORPUS_SOURCES) -> Corpus:
    """按白名单加载并**合并正文**。文件缺失时**记 warning 而不是静默跳过**。

    ## ⚠️ 这个粒度**不等于线上**

    线上走 `RagPipeline.index_file(path)` → `load_document` → 切分器，
    拿到的是 loader **原样输出**（1748 段，平均 169 字）。
    而这里把每份文档的正文段合并成一段（359 段）。

    差异的后果（W5 周六实测）：
    - 合并后 `chunk_chars` 才"有感觉"（300→800 块数差 34%）
    - 但**分数分布也变了**：合并语料可答 min=0.480，原样语料是 0.5157；
      拒答 max 也从 0.6288 变成 0.7958
    - 于是**阈值不可跨粒度移植** —— 用合并语料定出的 0.575，
      在原样语料上会误拒 16 条

    所以：
    - **调 `chunk_chars` / `overlap` 的网格**用这个（参数敏感性高）
    - **任何要报告"线上会怎样"的评测**（策略对比、阈值、答案质量）
      必须用 `load_raw_corpus`
    """
    corpus = Corpus()
    for relative in sources:
        path = PROJECT_ROOT / relative
        if not path.exists():
            corpus.warnings.append(f"语料缺失：{relative}")
            continue
        result = load_document(path)
        corpus.warnings.extend(result.warnings)
        prose_parts: list[str] = []
        for doc in result.documents:
            kind = doc.metadata.get("block", "text")
            if kind == "text":
                prose_parts.append(doc.text)
            else:
                corpus.fixed.append(doc)
        if prose_parts:
            merged = "\n\n".join(prose_parts)
            corpus.prose.append(
                Document(
                    source=path.name,
                    page=None,
                    text=merged,
                    metadata={"block": "text"},
                )
            )
            corpus.source_chars[path.name] = len(merged)
    return corpus


def chunk_corpus(corpus: Corpus, cfg: ChunkingConfig) -> list[Chunk]:
    """用固定策略切分语料。**同一个 cfg 同时作用于 prose 与 fixed**。"""
    strategy = build_strategy(INDEX_STRATEGY, cfg=cfg)
    return strategy.chunk(corpus.all_docs, cfg)


def load_raw_corpus(sources: Sequence[str] | None = None) -> list[Document]:
    """**loader 原样输出，不合并** —— 这是线上 `index_file` 的实际输入。

    见 `load_corpus` 的警示：任何要报告"线上会怎样"的评测都用这个。

    `sources=None` 用**全部** md（`full_corpus_sources()`），
    与 `strategy_compare.load_raw_corpus` 一致 —— 两者必须同库，否则召回率不可比。
    """
    targets = full_corpus_sources() if sources is None else sources
    docs: list[Document] = []
    for relative in targets:
        path = PROJECT_ROOT / relative
        if not path.exists():
            continue
        docs.extend(load_document(path).documents)
    return (
        docs  # --------------------------------------------------------------------------- 金标集
    )


@dataclass(frozen=True, slots=True)
class GoldItem:
    """一条检索金标。`anchor` 是逐字原文片段，命中判定靠它。"""

    id: str
    question: str
    anchor: str
    source: str


def load_gold(path: Path = GOLD_PATH) -> list[GoldItem]:
    """读金标集。**逐条校验 anchor 在源文件里确实存在**。

    这条校验是抄 `chroma_acceptance.py::assert_gold_exists` 的：
    金标写错时，"未命中"看起来和"检索失败"一模一样，于是你会去调 chunk_size、
    换模型 —— 时间全花在一个不存在的目标上。**测试数据本身出错是最贵的错。**
    """
    if not path.exists():
        raise FileNotFoundError(f"金标集不存在：{path}（先跑 subagent 生成或手写）")
    items: list[GoldItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        items.append(
            GoldItem(
                id=raw["id"],
                question=raw["question"],
                anchor=raw["anchor"],
                source=raw["source"],
            )
        )
    errors = validate_gold(items)
    if errors:
        raise ValueError("金标集校验失败：\n" + "\n".join(f"  ✗ {e}" for e in errors))
    return items


def validate_gold(items: Sequence[GoldItem]) -> list[str]:
    """返回问题列表（空 = 通过）。**不抛异常**，让调用方决定怎么报。"""
    errors: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            errors.append(f"{item.id}：id 重复")
        seen.add(item.id)
        path = PROJECT_ROOT / item.source
        if not path.exists():
            errors.append(f"{item.id}：源文件不存在 {item.source}")
            continue
        text = path.read_text(encoding="utf-8")
        occurrences = text.count(item.anchor)
        if occurrences != 1:
            errors.append(f"{item.id}：anchor 在 {item.source} 出现 {occurrences} 次（应为 1）")
        if not item.question.strip():
            errors.append(f"{item.id}：question 为空")
    return errors


# --------------------------------------------------------------------------- 命中判定


@dataclass
class QueryOutcome:
    """一条查询在某个索引上的结果。"""

    id: str
    hit_rank: int | None
    """命中的排名（1 起）；未命中为 None。"""

    top_scores: list[float] = field(default_factory=list)
    adjacent_pairs: int = 0
    """top_k 里来自相邻 chunk 的无序对数（见模块 docstring 第三节）。"""

    total_pairs: int = 0

    @property
    def hit(self) -> bool:
        return self.hit_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.hit_rank if self.hit_rank else 0.0


def evaluate_query(hits: Sequence[Any], item: GoldItem) -> QueryOutcome:
    """判定一条查询。`hits` 是 `SearchHit` 序列（带 `.chunk` 与 `.score`）。"""
    rank: int | None = None
    texts = [hit.chunk.text for hit in hits]
    for position, text in enumerate(texts, start=1):
        if item.anchor in text:
            rank = position
            break

    adjacent = 0
    for i in range(len(hits)):
        for j in range(i + 1, len(hits)):
            left, right = hits[i].chunk, hits[j].chunk
            if left.source != right.source:
                continue
            if abs(left.index - right.index) == 1:
                adjacent += 1
    total = len(hits) * (len(hits) - 1) // 2 if len(hits) > 1 else 0
    return QueryOutcome(
        id=item.id,
        hit_rank=rank,
        top_scores=[round(hit.score, 4) for hit in hits],
        adjacent_pairs=adjacent,
        total_pairs=total,
    )


def summarize_outcomes(outcomes: Sequence[QueryOutcome]) -> dict[str, Any]:
    """把逐条结果压成一行指标。**分母永远是全部 gold 条数**，不是"命中的那些"。"""
    n = len(outcomes)
    if n == 0:
        return {
            "queries": 0,
            "hits": 0,
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "adjacent_pair_rate": 0.0,
            "per_query": [],
        }
    hits = sum(1 for outcome in outcomes if outcome.hit)
    adjacent = sum(outcome.adjacent_pairs for outcome in outcomes)
    total_pairs = sum(outcome.total_pairs for outcome in outcomes)
    return {
        "queries": n,
        "hits": hits,
        "hit_at_k": round(hits / n, 4),
        "mrr": round(sum(outcome.reciprocal_rank for outcome in outcomes) / n, 4),
        # 未命中时这条查询贡献 0/0，**不进分母** —— 否则"没检出东西"会被算成
        # "没有重复内容"，把一个坏结果美化成一个好结果。
        "adjacent_pair_rate": round(adjacent / total_pairs, 4) if total_pairs else 0.0,
        "per_query": [
            {
                "id": outcome.id,
                "hit_rank": outcome.hit_rank,
                "adjacent_pairs": outcome.adjacent_pairs,
                "total_pairs": outcome.total_pairs,
            }
            for outcome in outcomes
        ],
    }


__all__ = [
    "CORPUS_SOURCES",
    "GOLD_PATH",
    "INDEX_STRATEGY",
    "Chunk",
    "Corpus",
    "GoldItem",
    "PROJECT_ROOT",
    "QueryOutcome",
    "chunk_corpus",
    "evaluate_query",
    "load_corpus",
    "load_gold",
    "summarize_outcomes",
    "validate_gold",
]
