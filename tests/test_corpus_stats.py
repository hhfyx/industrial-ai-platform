"""**文档 ↔ 实测一致性**：兑现 README 里那句"不一致就红"。

## 这个文件为什么必须存在（2026-11-05）

`README.md` 第 350 行、`eval/corpus_stats.py` 第 25/29 行都写着：

    tests/test_corpus_stats.py 守着文档与实测一致 —— 不一致就红

**而这个文件当时并不存在。**

这是本轮一直在修的那类问题的**原样重演**：一个没有实现的自述。
更讽刺的是，它出现在专门为了"消除文档数字打架"而写的模块里。

教训：**"我会有一个测试守着它"这句话本身就是需要被测试的断言。**
所以在写这个文件的同时，另加了 `test_claimed_guards_exist`（见文件末尾）
——它扫描文档里所有"XXX 守着 / XXX 会红"的说法，检查那个文件真的存在。

## 这里的断言尺度（重要）

**不**断言"块数等于某个固定值"—— 语料是 `docs/` 本身，每轮工作都在改，
钉死数字会让这个测试变成每次提交都要手动改的噪音源。

**要**断言的是：
1. 文档里写的**数字**与实测**一致**（这才是"不一致就红"的本意）
2. 文档里写的数字带**坐标**（篇数 / chunk_chars / overlap），否则无法核对
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from eval.corpus_stats import EVAL_CHUNK_CHARS, EVAL_OVERLAP_CHARS, measure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"


@pytest.fixture(scope="module")
def stats():
    """实测语料规模。**module 级**：只切一遍语料，别每条测试都切。"""
    return measure(rebuild=True)


def test_measure_does_not_dirty_worktree() -> None:
    """**`measure()` 默认不写盘**（2026-11-05 修正）。

    ## 这条测的是"副作用"，不是数据

    第一版 `measure()` 无条件写 `eval/results/corpus-stats.json`，于是：

        $ python scripts/corpus_stats.py
        $ git status
         M eval/results/corpus-stats.json

    **跑一次"只读"的统计会弄脏工作区** —— 而且那个文件与权威评测结果
    放在同一目录，让人分不清"这是缓存还是结果"。

    更糟：缓存里存旧值、重算给新值，于是 `--cached` 与重算**两个答案**。
    谁先跑谁得旧数。

    现在默认 `write=False`，缓存挪到 `data/cache/`（已被 gitignore）。
    """
    import subprocess

    def dirty_files() -> set[str]:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        return {line[3:].strip() for line in result.stdout.splitlines() if line.strip()}

    before = dirty_files()
    measure(rebuild=True)  # 默认 write=False
    after = dirty_files()

    new_dirty = after - before
    assert not new_dirty, (
        f"`measure()` 弄脏了工作区：{sorted(new_dirty)}\n"
        "统计函数不该有写副作用 —— 要落盘请显式传 write=True。"
    )


def test_cached_and_fresh_agree(stats) -> None:
    """缓存与实时重算必须给**同一个答案**。

    ## 这条防的是"双答案"

    缓存存在的意义是加速，不是提供另一种事实。
    如果 `--cached` 和重算不一致，那么"块数是多少"就取决于
    上一个人什么时候跑的 —— 这正是原来四个矛盾数字的成因。

    现在 `write=False` 是默认，所以这条主要防止有人用
    `write=True` 写了过期缓存后又用 `rebuild=False` 读它。
    """
    fresh = stats
    cached = measure(rebuild=False)

    # 缓存可能不存在（write=False 默认）—— 那时 `cached` 会回退到重算
    assert cached.chunks == fresh.chunks, (
        f"缓存与实测不一致：缓存 {cached.chunks} 块 / 实测 {fresh.chunks} 块。\n"
        "缓存里存的是旧值 —— 这就是'谁先跑谁得旧数'的来源。"
    )
    assert cached.raw_chunks == fresh.raw_chunks, (
        f"raw 口径也不一致：缓存 {cached.raw_chunks} / 实测 {fresh.raw_chunks}"
    )


def test_readme_numbers_match_measurement(stats) -> None:
    """**核心断言**：README 里引用的块数必须与实测**同量级**（容差 1 块）。

    ## ⚠ 为什么容差不等于 0（2026-11-05 实测确认的结构性缺陷）

    **README 自己在语料里**（26 篇 = README + 25 篇 docs），于是：

        往 README 里写块数 -> 改变 README 大小 -> 改变语料 -> 改变块数

    这是一个**自指循环**。实测验证过：

    | 对 README 做的事 | 块数 |
    | --- | ---: |
    | 原样 | 818 |
    | 把数字改成 818（实测值） | 819 |
    | 再加一行说明 | 820 |

    所以严格相等会让这个测试变成"每次改 README 都要手动同步数字"，
    而**噪音源最终会被人 skip 掉** —— 那时护栏彻底失效。

    ## 判据：差 ≤ 1 块

    - 真正要防的是**代际差异**（README 写 1365、实测 818，差 67%）-> 立刻红
    - 自指造成的 ±1 波动是**测量方式的固有误差**，不是文档错误 -> 放过

    实测验证：把 818 改成 999 -> 测试红（差 181 块）。护栏有效。

    ## 失败时怎么办

    1. 跑 `python -X utf8 scripts/corpus_stats.py` 拿新数
    2. 更新 README 里那个数（以及坐标）
    3. 提交时说明"语料变了，块数从 X 到 Y"
    """
    content = README.read_text(encoding="utf-8")

    # README 里引用当前规模的句式：`26 篇 / 464 段 / 818 块`
    #
    # ⚠ **必须容忍 Markdown 粗体标记**：实际写法是
    # `**26 篇 / 463 段 / 814 块**`，中间会夹满 `*`。
    # 第一版正则写 `(\d+) 篇 / (\d+) 段 / (\d+) 块`，
    # 遇到粗体时**完全不匹配** -> `matches` 为空 -> 走 `pytest.skip` ->
    # **测试静默通过，护栏等于没生效**。
    #
    # 这正是它要防的那类错误（"看起来有护栏、实际没生效"），
    # 所以先把 `*` `_` ``` 去掉再匹配。
    plain = re.sub(r"[*_`]", "", content)
    pattern = re.compile(r"(\d+)\s*篇\s*/\s*(\d+)\s*段\s*/\s*(\d+)\s*块")
    matches = pattern.findall(plain)

    if not matches:
        pytest.skip("README 里没有『N 篇 / N 段 / N 块』格式的引用（可能改写过了）")

    # 只核对**当前**那处（历史快照会明确标注"当时/历史"）
    #
    # ⚠ 判据不能用"上下文 200 字内出现『历史』" ——
    # 现在值紧跟在"上面这个 1365 块是历史快照"这句话**后面**，
    # 于是它被误判成历史值跳过了（第一版就这样，测试形同虚设）。
    #
    # 正确判据：**看这句引用自己那一行**，而不是周围 200 字。
    # "当时语料 1288 段 / 1365 块" 在同一行里带"当时"，
    # 而"26 篇 / 464 段 / 818 块" 那一行没有。
    #: 自指循环允许的误差（块）。理由见本函数 docstring。
    self_reference_tolerance = 1

    stale: list[str] = []
    for docs, segs, chunks in matches:
        # 定位到具体那一行
        needle = f"{docs} 篇"
        pos = plain.find(needle)
        if pos < 0:
            continue
        line_start = plain.rfind("\n", 0, pos) + 1
        line_end = plain.find("\n", pos)
        line = plain[line_start : line_end if line_end > 0 else len(plain)]

        # 这一行自称是历史值的，跳过
        if any(word in line for word in ("历史", "当时", "快照", "旧版", "旧值")):
            continue

        drift = abs(int(chunks) - stats.chunks)
        if drift > self_reference_tolerance:
            stale.append(
                f"README 写 {docs} 篇/{segs} 段/{chunks} 块，"
                f"实测 {stats.documents} 篇/{stats.segments} 段/{stats.chunks} 块"
                f"（差 {drift} 块，超过自指容差 {self_reference_tolerance}）"
            )

    assert not stale, (
        "README 的语料数字与实测不一致：\n  "
        + "\n  ".join(stale)
        + "\n\n跑 python -X utf8 scripts/corpus_stats.py 拿新数并更新 README。"
    )


def test_readme_numbers_carry_coordinates(stats) -> None:
    """README 里写块数时必须带**坐标**（篇数/chunk_chars/overlap）。

    ## 为什么这是硬要求

    本轮核实出的四说（1365/1364/1160/1592）之所以"看起来都对"，
    正是因为它们**都没有坐标** —— 读者无法判断它对应哪版语料、哪组参数。

    带坐标后，一个数字要么能被核对（"26 篇 + chunk_chars=500"），
    要么一眼就能看出是历史值（"当时 1288 段"）。
    """
    content = README.read_text(encoding="utf-8")

    # 只要有当前规模引用，就必须同时出现分块参数
    if not re.search(r"\d+\s*篇\s*/\s*\d+\s*段", content):
        pytest.skip("README 没有引用当前语料规模")

    assert str(EVAL_CHUNK_CHARS) in content, (
        f"README 引用了语料规模但没写 chunk_chars={EVAL_CHUNK_CHARS} —— 没有参数坐标的块数无法核对"
    )
    assert str(EVAL_OVERLAP_CHARS) in content, f"README 没写 overlap={EVAL_OVERLAP_CHARS}"


def test_both_chunk_count_flavors_are_documented(stats) -> None:
    """两个块数口径（检索 / 延迟）都要在文档里说清楚。

    `all_docs` 口径（检索评测用）与 `load_raw_corpus` 口径（延迟打点用）
    数值不同且**都正确**。只记一个，另一个就会在文档里变成"数据打架"。
    """
    assert stats.chunks != stats.raw_chunks, (
        "两个口径给出相同数字 —— 要么是巧合，要么代码变了（检查 measure 的实现）"
    )

    notes = (PROJECT_ROOT / "docs" / "latency.md").read_text(encoding="utf-8")
    assert str(stats.raw_chunks) in notes or "load_raw_corpus" in notes, (
        "docs/latency.md 应当说明它用的块数是 raw 口径"
    )


def test_claimed_guards_exist() -> None:
    """**文档里声称"XXX 守着"的，那个 XXX 必须真的存在。**

    ## 这条规则是从自己的错误里长出来的

    2026-11-05 核实发现：README 和 `eval/corpus_stats.py` 三处写着
    "tests/test_corpus_stats.py 守着文档与实测一致 —— 不一致就红"，
    而**那个文件从来没被写过**。

    一个没有实现的自述，比没有自述更糟：它让人以为有防护，
    于是不再人工核对。

    这条测试扫描文档里所有"`xxx.py` 守着" / "`xxx.py` 会红" 的句式，
    检查被点名的文件是否存在于仓库里。
    """
    # 只扫会做承诺的文档（README + docs/ + eval 模块的 docstring）
    targets = [README, *(PROJECT_ROOT / "docs").glob("*.md")]
    targets += list((PROJECT_ROOT / "eval").glob("*.py"))

    pattern = re.compile(r"`([\w/\\.-]+\.py)`\s*(?:守着|会红|会失败|会报警)")
    missing: list[str] = []

    for path in targets:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for claimed in pattern.findall(text):
            # 归一化路径分隔符
            candidate = claimed.replace("\\", "/")
            # 支持只写文件名（tests/xxx.py）或在别处写全路径
            found = any(
                (PROJECT_ROOT / p).exists()
                for p in (candidate, f"tests/{candidate}", f"scripts/{candidate}")
            )
            if not found:
                missing.append(f"{path.name} 声称 `{claimed}` 存在，但找不到")

    assert not missing, "文档声称存在但实际没有的文件（空头承诺）：\n  " + "\n  ".join(
        sorted(set(missing))
    )
