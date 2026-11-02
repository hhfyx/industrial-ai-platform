"""检索语料口径的测试：**哪些文档不进检索库，以及这个名单会不会漏**。

## 为什么这个文件只有几行，却比一堆指标测试重要

W5 周日发现了一件很隐蔽的事：我自己写的分析报告（`retrieval-tuning-final.md`、
`strategy-comparison.md` 等）也在 `docs/` 下，于是它们进了检索库。
那些文档在**复述**同样的结论，语义上和所有问题都近 —— 是最强的竞争者。

后果不是报错，而是：**越努力写文档，评测越差**。而人会去调参数。

当时的修法是加 `ANALYSIS_DOCS` 白名单。但**名单是人手写的，就一定会漏**：
W5 只想到 `retrieval-tuning-final.md`，漏了同一性质的
`retrieval-tuning-token-grid.md`（84 段 / 86 块）—— 于是同一份 harness
在两个提交上跑出 1597 段 / 1678 块 与 1513 段 / 1592 块 两个库，
而主指标（40.0% / 0.2402 / 4-10）完全相同，所以**没有任何数字会报警**。

这正是"静默污染"的典型形态：错的不是指标，是元数据（块数、耗时、错例条数），
而所有引用这些元数据的地方（文档主表、错例表抬头、消融对比）会各自抄一份。

## 所以这里锁两件事

1. **判据是"文件名像不像分析报告"，不是"名单里有没有"** —— 用前缀规则扫一遍，
   发现任何 `retrieval-tuning-*` / `retrieval-params` / `eval-50` /
   `strategy-comparison` / `w4-checklist` 还留在语料里，就失败。
   这样下次新增一份 `retrieval-tuning-xxx.md` 会立刻红，而不是等数字对不上才发现。
2. **语料必须非空且不含 README 之外的根目录文件** —— 防止 glob 写错导致
   语料悄悄缩小（"库越小越容易命中"，缩小库会让召回虚高）。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import retrieval_harness as rh  # noqa: E402


def test_no_analysis_doc_leaks_into_the_corpus() -> None:
    """**按名字模式判，不按名单判** —— 名单是事后补的，新文档是事前写的。

    这个坑被咬了**三次**，每次都是同一句话：我在写分析文档，然后用它污染评测。

    | 轮次 | 漏了什么 | 后果 |
    | --- | --- | --- |
    | W5 周日 | `retrieval-tuning-token-grid.md` | 1678 块 vs 1592 块 |
    | W6 周一 | `bm25.md` + `weekly/w6.md`（当天新写） | 1650 块，w6.md 里写着「Recall@5 40.0%」 |
    | W6 周三 | `hybrid-retrieval.md`（昨天新写） | 1404 块 vs 1363 块 |

    三次主指标都几乎不变，所以**没有任何数字会报警**。只能靠规则守。
    """
    leaked = [src for src in rh.full_corpus_sources() if rh._is_analysis_name(Path(src).name)]
    assert leaked == [], (
        "这些文档在复述结论、不该进检索库，却漏在语料里：\n  "
        + "\n  ".join(leaked)
        + "\n判据是 ANALYSIS_NAME_PATTERNS（名字模式），不是 ANALYSIS_DOCS（名单）"
    )


def test_new_analysis_doc_is_excluded_without_touching_any_list() -> None:
    """**新写一份分析文档，不改任何名单也该被排除。**

    这条直接针对三轮里"漏的总是当天新写那份"的模式：它用一个**还没存在**的
    文件名去问规则，而不是检查仓库里现有的文件。

    如果这条红了，说明判据退化成了名单 —— 那下周又会漏一次。
    """
    hypothetical = [
        "reranker.md",
        "hybrid-rerank.md",
        "retrieval-tuning-v2.md",
        "eval-200.md",
        "strategy-compare-2.md",
        "w7-checklist.md",
    ]
    not_excluded = [name for name in hypothetical if not rh._is_analysis_name(name)]
    assert not_excluded == [], (
        f"这些'将来会写'的分析文档不会被排除：{not_excluded}\n"
        "把它们的前缀补进 ANALYSIS_NAME_PATTERNS"
    )


def test_prose_docs_are_not_falsely_excluded() -> None:
    """**反向检查**：真正的知识文档不该被模式误伤。

    判据收得太紧会把正经文档也排掉（语料变小 → 库越小越容易命中 → 召回虚高）。
    这里点名几份必须留在语料里的文档。
    """
    must_keep = [
        "sse.md",
        "api-spec.md",
        "chroma-store.md",
        "rag-architecture.md",
        "document-parsing.md",
        "design-decisions.md",
        "SETUP.md",
        "cost.md",
    ]
    excluded = [name for name in must_keep if rh._is_analysis_name(name)]
    assert excluded == [], f"这些正经知识文档被误排了：{excluded}"


def test_weekly_reports_never_enter_the_corpus() -> None:
    """**整个 `docs/weekly/` 都不进检索库** —— 这是泄题，不只是竞争。

    W6 周一实测：`weekly/w6.md` 进了库，而它正文里写着
    「向量 16 / BM25 14 / 差异 10 条」。检索"BM25 表现如何"时那块会直接命中 ——
    **那不是检索能力，是答案就在库里。**

    周报每周新增一份、每份都在复述当周结论，所以逐个加名单必然漏，
    必须按目录排除。
    """
    leaked = [src for src in rh.full_corpus_sources() if "/weekly/" in src]
    assert leaked == [], (
        f"周报进了检索库（它们复述当周结论，等于泄题）：{leaked}\n应当在 ANALYSIS_DIRS 里按目录排除"
    )


def test_analysis_dirs_actually_take_effect() -> None:
    """`ANALYSIS_DIRS` 不是写给自己看的 —— 被它指到的目录必须真的从语料里消失。"""
    kept = rh.full_corpus_sources()
    for prefix in rh.ANALYSIS_DIRS:
        on_disk = [
            f"{prefix}{p.relative_to(PROJECT_ROOT / prefix).as_posix()}"
            for p in (PROJECT_ROOT / prefix).rglob("*.md")
            if (PROJECT_ROOT / prefix).exists()
        ]
        if not (PROJECT_ROOT / prefix).exists():
            continue
        assert on_disk, f"{prefix} 下没有任何 md —— 这条断言失去意义"
        leaked = [src for src in kept if src.startswith(prefix)]
        assert leaked == [], f"{prefix} 在 ANALYSIS_DIRS 里，但仍出现在语料：{leaked}"


def test_legacy_doc_list_is_covered_by_the_name_rule() -> None:
    """历史名单 `ANALYSIS_DOCS` 里每一条都必须**同时**被名字规则命中。

    这条不是检查名单本身有用（它已经不是判据了），而是查一个回归：
    如果有人往里加了一份名字不符合模式的文件，说明**有人还在按感觉手工加名单**
    —— 那正是四轮都漏的原因。
    """
    unexplained = sorted(name for name in rh.ANALYSIS_DOCS if not rh._is_analysis_name(name))
    assert unexplained == [], (
        "ANALYSIS_DOCS 里有名字不符合任何名字模式的条目 ——"
        f"说明有人还在手工加名单（那就一定会再漏）：{unexplained}"
    )


# ------------------------------------------------- 白名单判据（W6 周六反转）


def test_corpus_is_derived_from_allowlist_not_blacklist() -> None:
    """**语料必须来自 `KNOWLEDGE_DOCS` 白名单，不是"全扫再排除"。**

    这是 W6 周六的判据反转。黑名单被同一个坑咬了四次：

    | 轮次 | 漏了什么 |
    | --- | --- |
    | W5 周日 | `retrieval-tuning-token-grid.md` |
    | W6 周一 | `bm25.md` + `weekly/w6.md`（当天新写，正文里写着 Recall@5 40.0%） |
    | W6 周三 | `hybrid-retrieval.md`（昨天新写） |
    | W6 周六 | `context-compression.md`（昨天新写） |

    判定方式：语料里的每一项都必须**在 `KNOWLEDGE_DOCS` 里**（README 除外）。
    如果哪天有人把判据改回黑名单，新写的分析文档会重新出现在语料里，这条就红。
    """
    sources = set(rh.full_corpus_sources())
    allowlist = set(rh.KNOWLEDGE_DOCS) | {"README.md"}
    leaked = sorted(sources - allowlist)
    assert leaked == [], (
        "语料里出现了白名单之外的文件 —— 判据可能被改回黑名单了：\n  "
        + "\n  ".join(leaked)
        + "\n应当只收 KNOWLEDGE_DOCS 里明确列出的知识文档"
    )


def test_every_allowlisted_doc_actually_exists() -> None:
    """白名单里不能有失效路径 —— 否则"写了但没进语料"是无声的失真。"""
    missing = [rel for rel in rh.KNOWLEDGE_DOCS if not (rh.PROJECT_ROOT / rel).exists()]
    assert missing == [], f"白名单里有 {len(missing)} 份文件不存在：{missing}"


def test_corpus_size_matches_the_historical_baseline() -> None:
    """**语料规模必须与 W5/W6 的历史基线一致（26 份）。**

    为什么锁死这个数：W5、W6 的全部历史数字（Recall@5=40.0%、天花板 50%…）
    都在这个库上算的（1288 段 / 1364 块）。换了集合，消融表与历史就不可比 ——
    而"数字对不上"会被归因到算法改动，查起来极费时间。

    如果这条红了，先问：**是不是新增/删除了知识文档？**
    如果是，那所有历史数字都要重新标注口径，而不只是改这个断言。
    """
    sources = rh.full_corpus_sources()
    assert len(sources) == 26, f"语料份数从 26 变成了 {len(sources)}"
    assert "README.md" in sources


def test_newly_written_doc_does_not_enter_corpus_automatically() -> None:
    """**任何新写的 md 默认都不进语料** —— 除非显式加进白名单。

    这是反转判据的核心收益。用**还不存在**的文件名验证：

        ablation-four.md        ← 今天写的（消融脚本配的文档）
        answer-quality-w6.md   ← 明天可能写的
        w7-plan.md             ← 下周一定写的
        cost-analysis.md
        notes.md

    **黑名单判据对这 5 个全部漏判**（`_is_analysis_name` 都返回 False，
    也就是说它们会进语料）。白名单判据全部正确排除 —— 这才是这条测试的价值：
    它测的是"将来会写的那份"，不是"仓库里现有的那份"。
    """
    hypothetical = [
        "ablation-four.md",
        "answer-quality-w6.md",
        "w7-plan.md",
        "cost-analysis.md",
        "notes.md",
    ]
    # 白名单判据：不在名单里 = 不进语料
    allowlist = set(rh.KNOWLEDGE_DOCS)
    wrongly_allowed = [name for name in hypothetical if f"docs/{name}" in allowlist]
    assert wrongly_allowed == [], f"这些新文档名在白名单里（不该）：{wrongly_allowed}"
    # 反向证据：黑名单判据对它们全部失效（这正是反转的理由）
    blacklist_catches = [name for name in hypothetical if rh._is_analysis_name(name)]
    assert len(blacklist_catches) < len(hypothetical), (
        "如果黑名单能抓住全部假设文档，那反转判据就没有必要了 ——请重新评估这次改动"
    )


def test_corpus_is_not_built_by_scanning_all_docs() -> None:
    """**语料不能"全扫 docs 再排除"** —— 判据必须是白名单。

    判定方式：磁盘上的 md 数量明显多于语料份数（说明确实在筛选），
    且语料里每一项都在白名单内。如果哪天有人改回 `rglob("*.md")`，
    新写的分析文档会重新混进来，而这条断言的前半部分不会变
    （磁盘数一直比语料多），所以还要靠
    `test_newly_written_doc_does_not_enter_corpus_automatically`
    去测"将来的文件"。
    """
    on_disk = list((rh.PROJECT_ROOT / "docs").rglob("*.md"))
    sources = rh.full_corpus_sources()
    assert len(sources) < len(on_disk) + 1, (
        f"语料 {len(sources)} 份 vs 磁盘 {len(on_disk)} 份 md —— 筛选没生效？"
    )
    assert set(sources) <= (set(rh.KNOWLEDGE_DOCS) | {"README.md"})


def test_corpus_covers_docs_and_readme_only() -> None:
    """语料 = 根 README + `docs/` 下的 md，不许从别处捞。

    glob 写错（比如 `docs/*.md` 少一层 `rglob`）会让语料悄悄变小。
    **库越小越容易命中**（竞争者少），所以"语料变小"会让召回虚高 ——
    那是比"变低"更危险的一类错：它看起来像改进。
    """
    sources = rh.full_corpus_sources()
    assert sources, "语料为空"
    assert len(sources) == len(set(sources)), "同一份文档在语料里出现两次"
    assert sources[0] == "README.md", "README 必须排在第一位（块序稳定）"

    bad = [
        src
        for src in sources
        if not (src == "README.md" or (src.startswith("docs/") and src.endswith(".md")))
    ]
    assert bad == [], f"语料里出现了不该被扫进来的路径：{bad}"

    # 顺序稳定：块序影响 store_id，进而影响同分时的排序。
    docs_part = list(sources[1:])
    assert docs_part == sorted(docs_part), "docs/ 部分必须按名字排序 —— 否则不同机器切出的块序不同"


def test_docs_dir_would_be_larger_without_the_exclusion() -> None:
    """排除是**真的在起作用**：不加排除时语料应该更大。

    这条防的是"名单写了但过滤条件没生效"（比如 `if p.name not in ...`
    写成了 `if p.name in ...`，语料反而只剩分析文档）。
    """
    on_disk = sorted(
        f"docs/{p.relative_to(PROJECT_ROOT / 'docs').as_posix()}"
        for p in (PROJECT_ROOT / "docs").rglob("*.md")
    )
    kept = [src for src in rh.full_corpus_sources() if src != "README.md"]
    assert len(kept) < len(on_disk), (
        f"排除没生效：语料 {len(kept)} 份 == docs 下的 {len(on_disk)} 份"
    )
    excluded_on_disk = [src for src in on_disk if src not in kept]
    assert excluded_on_disk, "排除没生效：一份文档都没被排除"
