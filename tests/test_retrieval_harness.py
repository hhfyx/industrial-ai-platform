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

#: 名字长这样就算"分析/复盘报告"，**不该进检索库**。
#: 判据是"它在复述项目做了什么"，不是"这份文件我记不记得排除"。
ANALYSIS_NAME_PATTERNS: tuple[str, ...] = (
    "retrieval-tuning",
    "retrieval-params",
    "eval-50",
    "strategy-comparison",
    "bm25",
    "w4-checklist",
)


def _looks_like_analysis(name: str) -> bool:
    return any(
        name.startswith(pattern) or name == f"{pattern}.md" for pattern in ANALYSIS_NAME_PATTERNS
    )


def test_no_analysis_doc_leaks_into_the_corpus() -> None:
    """**按名字判，不按名单判** —— 名单是人手写的，一定会漏。

    这条已经被同一个坑咬了两次：

    - W5 漏了 `retrieval-tuning-token-grid.md`（与 final 同性质）→ 1678 块 vs 1592 块
    - W6 周一漏了当天新写的 `bm25.md` → 1650 块 vs 1592 块

    两次主指标都几乎不变，所以**没有任何数字会报警**。只能靠规则守。
    """
    leaked = [src for src in rh.full_corpus_sources() if _looks_like_analysis(Path(src).name)]
    assert leaked == [], (
        "这些文档在复述结论、不该进检索库，却漏在语料里：\n  "
        + "\n  ".join(leaked)
        + "\n把它们加进 ANALYSIS_DOCS（判据是『是不是分析报告』，不是『名单里有没有』）"
    )


def test_weekly_reports_never_enter_the_corpus() -> None:
    """**整个 `docs/weekly/` 都不进检索库** —— 这是泄题，不只是竞争。

    W6 周一实测：`weekly/w6.md` 进了库，而它正文里写着
    「向量 16 / BM25 14 / 差异 10 条」。检索"BM25 表现如何"时那块会直接命中 ——
    **那不是检索能力，是答案就在库里。**

    周报每周新增一份、每份都在复述当周结论，所以逐个加 `ANALYSIS_DOCS` 必然漏，
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


def test_analysis_docs_stay_in_sync_with_the_name_patterns() -> None:
    """`ANALYSIS_DOCS` 里每一条都必须能被上面的前缀规则解释。

    反过来不检查（规则是"应该排除"的意图，名单是"已经排除"的事实）——
    但名单里出现一个规则解释不了的怪名字，说明有人在按感觉加白名单，
    下次就会漏。发现这种就把它写进 `ANALYSIS_NAME_PATTERNS`。
    """
    unexplained = sorted(name for name in rh.ANALYSIS_DOCS if not _looks_like_analysis(name))
    assert unexplained == [], (
        "ANALYSIS_DOCS 里有名字不符合任何分析报告规则，"
        f"要么它是别的类别、要么该补进 ANALYSIS_NAME_PATTERNS：{unexplained}"
    )


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
