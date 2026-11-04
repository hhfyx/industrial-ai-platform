"""**语料指纹**：语料被改写时，让历史评测数字失效这件事**可见**。

## 这个问题是怎么暴露的（2026-11-04）

`docs/deploy-notes.md` 在本轮被改动了 **+69 段**（从 32 段涨到 101 段），
`README.md` 也涨了 137 行。而**语料就是 `docs/` 本身**
（`KNOWLEDGE_DOCS` 白名单里 26 篇，README + 25 篇 docs）。

于是发生了一件很微妙的事：

- 历史上测过 Recall@5 = 40%（在 1365 块的语料上）
- 后来文档被反复改写（每轮工作都在写文档）
- 现在语料是 **463 段 / 814 块**
- **但没有任何测试会失败** —— 因为"文档改好了"本身是好事

问题不在文档变长了（那是正常的），而在于：
**语料变了 = 评测基线变了，而这件事完全静默。**

## 这个模块做什么

计算语料的**内容指纹**（每篇文档的哈希 + 整体哈希），入库。
语料变了，`tests/test_corpus_fingerprint.py` 会红，并列出
**哪几篇文档变了**。

## 它**不**阻止改文档（明确边界）

文档必须能改 —— 这个项目每轮工作都在写文档。
`docs/` 既是产物**又是语料**，这个双重身份无法回避。

指纹的作用是让人**知道**"改动会影响评测"，然后做一个选择：

| 选择 | 做法 |
| --- | --- |
| A. 接受基线变化 | 重跑评测，更新数字，`--update` 指纹 |
| B. 认为某篇不该进语料 | 从 `KNOWLEDGE_DOCS` 移除（这是设计决策，要写理由） |

**最坏的是 C：既不重跑也不移除，就这么放着** —— 于是文档里
"Recall@5 40%" 变成一个无法核对的孤儿数字。这正是本轮核实出的问题。

## 为什么用"每篇哈希"而不只用一个总哈希

只给总哈希时，报错只能说"语料变了"，说不出**哪篇变了**。
而"哪篇变了"直接决定影响范围：

- 改 `docs/chunking-boundaries.md`（评测题 f-xx 的锚点在里面）-> 该题可能失效
- 改 `README.md` -> 影响面广（多道题的锚点指向它）
- 改一篇没被任何题引用的文档 -> 只影响块数，不影响单题

每篇哈希让报错能直接指向"去重跑哪几道题"。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 指纹文件位置。**入库** —— 它的价值就是跨提交对比。
FINGERPRINT_PATH: Path = PROJECT_ROOT / "eval" / "results" / "corpus-fingerprint.json"


def _file_digest(path: Path) -> str:
    """单篇文档的内容哈希。

    **按内容哈希，不按 mtime**：git 操作会改 mtime，
    用它做判据会制造大量假阳性（每次 checkout 都说"变了"）。
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "<UNREADABLE>"


def collect() -> dict[str, str]:
    """返回 `{文档相对路径: 哈希}`，覆盖白名单里所有**存在的**文档。

    用 `full_corpus_sources()` 而不是自己扫 `docs/` ——
    必须与评测实际使用的语料**完全一致**，否则指纹管的是另一批文件。
    """
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from scripts.retrieval_harness import full_corpus_sources

    digests: dict[str, str] = {}
    for source in full_corpus_sources():
        path = Path(source)
        # 相对路径入库，避免不同机器上的绝对路径差异污染指纹
        try:
            key = path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            key = path.name
        digests[key] = _file_digest(path if path.is_absolute() else PROJECT_ROOT / path)
    return digests


def compute() -> dict[str, object]:
    """当前语料指纹（逐篇哈希 + 总哈希 + 篇数）。"""
    digests = collect()
    total = hashlib.sha256(
        json.dumps(digests, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"digest": total, "documents": len(digests), "files": digests}


def load_baseline() -> dict[str, object] | None:
    if not FINGERPRINT_PATH.exists():
        return None
    try:
        return json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def diff(baseline: dict[str, object] | None, current: dict[str, object]) -> dict[str, list[str]]:
    """返回 `{changed: [...], added: [...], removed: [...]}`。

    **分三类而不是一个列表**：三类的影响完全不同。
    - `changed` —— 内容被改写，引用它的题可能要重跑
    - `added` —— 新文档进语料，可能改变排序（新块竞争 Top-K）
    - `removed` —— 文档消失，引用它的题直接失效
    """
    if baseline is None:
        return {"changed": [], "added": [], "removed": []}
    before = baseline.get("files") or {}
    after = current.get("files") or {}
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {"changed": [], "added": [], "removed": []}

    return {
        "changed": sorted(k for k in before if k in after and before[k] != after[k]),
        "added": sorted(k for k in after if k not in before),
        "removed": sorted(k for k in before if k not in after),
    }


def write_baseline(current: dict[str, object] | None = None) -> None:
    payload = current if current is not None else compute()
    FINGERPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "FINGERPRINT_PATH",
    "collect",
    "compute",
    "diff",
    "load_baseline",
    "write_baseline",
]
