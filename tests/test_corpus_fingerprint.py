"""**语料指纹**测试：语料被改写时，历史评测数字失效这件事必须可见。

## 为什么需要（2026-11-04 核实出的问题）

语料就是 `docs/` 本身（`KNOWLEDGE_DOCS` 白名单 26 篇）。而本项目
**每轮工作都在写文档** —— 本轮 `docs/deploy-notes.md` 就涨了 +69 段。

于是：历史 Recall@5 = 40%（1365 块语料）-> 文档反复改写 ->
现在 463 段 / 814 块 -> **但没有任何测试会失败**。

"文档变好了"是好事，但"语料基线悄悄变了"必须被知道。

## 边界

**不阻止改文档**（边界见 `eval/corpus_fingerprint.py`）。
本测试只保证：改了会被看见，并指出**哪几篇**。
"""

from __future__ import annotations

import pytest
from eval.corpus_fingerprint import (
    FINGERPRINT_PATH,
    collect,
    compute,
    diff,
    load_baseline,
)


def test_fingerprint_file_is_committed() -> None:
    """指纹文件必须入库（跨提交对比才有意义）。"""
    assert FINGERPRINT_PATH.exists(), (
        "缺语料指纹文件。首次建立：python -X utf8 scripts/corpus_fingerprint.py --update"
    )


def test_corpus_fingerprint_matches_baseline() -> None:
    """**核心断言**：语料内容与入库指纹一致。

    ## 失败时怎么办（这是这份测试最重要的部分）

    看到这条红了，先问一个问题：**这次改动会影响评测吗？**

    | 情况 | 做法 |
    | --- | --- |
    | 改的是措辞/排版，题目锚点没动 | 重跑评测确认数字没变，然后 `--update` |
    | 改了某道题的锚点原文 | **该题必须重新出**（锚点对不上原文了） |
    | 新增了文档进语料 | 重跑，新块可能挤占 Top-K |
    | 删了文档 | 引用它的题直接失效，必须处理 |

    **最坏的做法**：直接 `--update` 了事。那等于把"文档与代码脱节"
    盖章确认 —— 正是本轮发现的问题（README 里四个互相矛盾的块数）。
    """
    baseline = load_baseline()
    if baseline is None:
        pytest.skip("还没有语料指纹基线")

    changes = diff(baseline, compute())
    has_changes = any(changes.values())
    if not has_changes:
        return

    detail: list[str] = []
    for kind, label in (
        ("changed", "内容被改写"),
        ("added", "新增进语料"),
        ("removed", "从语料消失"),
    ):
        items = changes.get(kind) or []
        if items:
            detail.append(f"  {label}（{len(items)} 篇）：")
            detail.extend(f"    - {name}" for name in items)

    assert not has_changes, (
        "语料内容与入库指纹不一致：\n"
        + "\n".join(detail)
        + "\n\n历史评测数字（如 README 的 Recall@5 40%）绑定在旧语料上。"
        "\n处理方式见本测试 docstring —— 不要直接 --update。"
    )


def test_all_corpus_documents_exist() -> None:
    """白名单里的每篇文档都必须真实存在。

    `KNOWLEDGE_DOCS` 里写了一个不存在的文件名时，
    `full_corpus_sources()` 会静默跳过它（设计如此 —— 允许文档未创建）。
    但指纹能抓到：那篇会从 `files` 里消失，表现为 `removed`。

    这条单独测一次，是为了给出**更清楚的错误信息**：
    "缺文件"和"文件被删了"是两件事。
    """
    digests = collect()
    unreadable = [name for name, digest in digests.items() if digest == "<UNREADABLE>"]
    assert not unreadable, f"以下语料文档读不出来：{unreadable}"


def test_fingerprint_has_per_file_hashes() -> None:
    """指纹必须**逐篇**记录，不能只有总哈希。

    只给总哈希时，报错只能说"语料变了"，说不出**哪篇变了** ——
    而"哪篇"直接决定"该重跑哪几道题"。
    """
    current = compute()
    files = current.get("files")
    assert isinstance(files, dict) and files, "指纹里没有逐篇哈希"
    assert len(files) == current["documents"], "篇数与逐篇哈希数量不符"
    assert current.get("digest"), "没有总哈希"


def test_diff_separates_change_kinds() -> None:
    """`diff()` 要把"改写/新增/删除"分开报（三者的处理方式完全不同）。"""
    baseline = {"files": {"a.md": "h1", "b.md": "h2"}}
    current = {"files": {"a.md": "CHANGED", "c.md": "h3"}}

    changes = diff(baseline, current)
    assert changes["changed"] == ["a.md"], changes
    assert changes["added"] == ["c.md"], changes
    assert changes["removed"] == ["b.md"], changes


def test_fingerprint_uses_relative_paths() -> None:
    """指纹用**相对路径**，不写绝对路径。

    绝对路径会让指纹在换机器/换盘符后全部变成 `added` + `removed`，
    制造满屏假阳性 —— 那样这个机制会被人关掉。
    """
    current = compute()
    for name in current.get("files") or {}:
        assert not name.startswith("/"), f"出现了绝对路径：{name}"
        assert ":" not in name or name.count(":") == 0, f"出现了盘符路径：{name}"
