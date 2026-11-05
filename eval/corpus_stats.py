"""**语料与评测集的权威数字** —— 唯一事实来源。

## 为什么需要这个文件

2026-11-04 核实发现：同一件事在文档里有**四种说法**：

| 来源 | 声称的块数 |
| --- | --- |
| README L341 | 1365 块 |
| README L376 / L415 | 1364 块 |
| README L433（能力类表格） | 1160 块 |
| `eval/results/retrieval-baseline.json` | 1592 块 |

而**实测**（`chunk_chars=500, overlap=50`，当前语料 26 篇）是 **814 块**。

四个数字全都"看起来合理"，因为它们分别对应**四个不同时间点**的语料
与分块参数。问题不在某个数错了，而在**没有一个地方能回答"现在是多少"**。

## 这个模块怎么解决

把数字**算出来**，不手写：

```bash
python -X utf8 scripts/corpus_stats.py        # 打印给人看
python -X utf8 scripts/corpus_stats.py --json # 给 CI / 测试用
```

文档里的数字由 CI（或人工）从这个脚本的输出复制，且
`tests/test_corpus_stats.py` 会校验文档与实测一致 —— **不一致就红**。

## 为什么数字会变（这一点必须写在文档里）

块数变化**不是 bug**，是语料和参数在演进：

- 语料从 25 篇扩到 26 篇（新增/删除了某篇文档）
- 分析类文档曾反复污染语料（`context-compression.md` 是第 5 次泄漏），
  后来改成**白名单** `KNOWLEDGE_DOCS`
- `MIN_CHUNK_CHARS` / `TOC_ANCHOR_RATIO` 这类切分规则的改动
  （2026-11-04 的短块合并让 Markdown 报告的块数从 105 降到 60）

**所以正确的做法不是"改数字"，而是"标注这个数字是哪一版的"。**
任何写进文档的块数都必须带 (语料篇数, chunk_chars, overlap) 三个坐标，
否则它就是一个无法核对的孤立数字。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 评测用的分块配置（与 `docs/ablation-four.md` 的四配置消融一致）。
#:
#: **改这里之前**先想清楚：所有历史评测数字都绑在这组参数上，
#: 换了参数，旧数字全部作废（不是"微调"，是"换了一把尺子"）。
EVAL_CHUNK_CHARS = 500
EVAL_OVERLAP_CHARS = 50

#: 主要分块档位的块数对照（供文档引用）。
#:
#: 三档的含义见 `docs/chunking.md`：500 是默认，300 用于对比"更细的块"，
#: 1000 用于对比"更粗的块"。这不是"三选一"，是消融实验的三个刻度。
REFERENCE_CONFIGS: tuple[tuple[int, int], ...] = (
    (300, 30),
    (500, 50),
    (1000, 100),
)


@dataclass(frozen=True)
class CorpusStats:
    """一份语料的实测规模。**所有字段都是算出来的，不含手写值。**"""

    #: 白名单文档数（`KNOWLEDGE_DOCS` 里真实存在的那些）
    documents: int
    #: 段数：loader 产出的 `Document` 条数（切块前）
    segments: int
    #: 评测配置（`EVAL_CHUNK_CHARS` / `EVAL_OVERLAP_CHARS`）下的块数
    chunks: int
    #: 分块参数，方便文档标注
    chunk_chars: int
    overlap_chars: int
    #: 三档块数对照 `{chunk_chars: chunks}`（检索口径）
    by_chunk_chars: dict[int, int]
    #: **`load_raw_corpus()` 口径**的块数（loader 原样输出、不合并）
    #:
    #: ## 为什么会有两个块数（这不是 bug）
    #:
    #: 两条路径的输入不同，且**都正确**：
    #:
    #: | 口径 | 输入 | 谁在用 |
    #: | --- | --- | --- |
    #: | `chunks`（`all_docs`，含合并） | 合并后的文档 | **检索评测**（`run_retrieval_eval.py`） |
    #: | `raw_chunks`（`load_raw_corpus`） | loader 原样输出 | **延迟打点**（`latency_profile.py`） |
    #:
    #: `scripts/retrieval_harness.py` 的注释早就警告过：
    #: "任何要报告'线上会怎样'的评测都用 `load_raw_corpus`"。
    #:
    #: 2026-11-04 实测：同一份 26 篇语料，`all_docs` 切出 **814 块**，
    #: `raw` 切出 **1163 块**。相差 349 块 —— 合并把跨页/跨段碎片先拼起来了。
    #:
    #: **两个数都必须记录。** 只记一个，另一个就会在文档里变成"数据打架"
    #: —— 这正是本轮核实出的老问题（README 里 1365/1364/1160 三种说法）。
    raw_chunks: int

    def as_badge(self) -> str:
        """一行给人看的摘要，可直接粘进文档。"""
        return (
            f"{self.documents} 篇 / {self.segments} 段 / "
            f"**{self.chunks} 块**（检索口径）/ **{self.raw_chunks} 块**（延迟口径）"
            f"（chunk_chars={self.chunk_chars}, overlap={self.overlap_chars}）"
        )


def measure(rebuild: bool = True, write: bool = False) -> CorpusStats:
    """实测当前语料的规模。

    ## 参数

    - `rebuild=True`（默认）：真的切一遍全语料（准，但要几秒）
    - `rebuild=False`：只读缓存（快，但可能过时）
    - `write=False`（**默认**）：**不写盘**

    ## ⚠ 为什么 `write` 默认是 False（2026-11-05 修正）

    第一版 `measure()` **无条件写盘**。后果：

        每次跑 scripts/corpus_stats.py
          -> 改写 eval/results/corpus-stats.json
          -> git status 里出现 "M eval/results/corpus-stats.json"

    "跑一次统计"居然会弄脏工作区 —— 而那个文件躺在版本库里，
    与**权威评测结果**放在同一目录，很难分辨"这是缓存还是结果"。

    更糟的是它制造了**双答案**：缓存里存着旧值，重算给新值。
    谁先跑谁得旧数，取决于上一个人什么时候跑的。

    ## 缓存文件该不该入库

    **不该**。它是派生数据（能随时从语料重算），不是结果。
    入库只会造成两个后果：① 每次重算都产生无意义的 diff
    ② 让人误以为它是权威数字。

    真正的权威来源是**实时重算**；缓存只在同一进程内加速。
    """
    cache = PROJECT_ROOT / "data" / "cache" / "corpus-stats.json"
    if not rebuild and cache.exists():
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            return CorpusStats(
                documents=payload["documents"],
                segments=payload["segments"],
                chunks=payload["chunks"],
                chunk_chars=payload["chunk_chars"],
                overlap_chars=payload["overlap_chars"],
                by_chunk_chars={int(k): v for k, v in payload["by_chunk_chars"].items()},
                raw_chunks=payload.get("raw_chunks", 0),
            )
        except (KeyError, ValueError, TypeError):
            pass  # 缓存坏了就重算，不要因此失败

    # 延迟导入：这个模块要能被文档/测试轻量导入，不该拖起整个 RAG 栈
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from scripts.retrieval_harness import (
        chunk_corpus,
        full_corpus_sources,
        load_corpus,
        load_raw_corpus,
    )

    from app.rag.chunking import ChunkingConfig, build_strategy

    sources = full_corpus_sources()
    corpus = load_corpus(sources)

    # `Corpus` 的字段（见 `scripts/retrieval_harness.py`）：
    #   prose  —— 按文档组织的正文，每个元素是一篇 `Document`（= 段数来源）
    #   fixed  —— 已经是固定块的文档（如表格、代码），不再走分块
    # **用 getattr 兜住字段改名**：这个模块要给文档/测试轻量导入，
    # 不该因为 harness 内部重构就崩。
    prose = getattr(corpus, "prose", None) or []
    fixed = getattr(corpus, "fixed", None) or []
    segments = len(prose) + len(fixed)

    def count_at(chunk_chars: int, overlap: int) -> int:
        cfg = ChunkingConfig(chunk_chars=chunk_chars, overlap_chars=overlap)
        return len(chunk_corpus(corpus, cfg))

    by_cfg = {cc: count_at(cc, ov) for cc, ov in REFERENCE_CONFIGS}

    # 第二个口径：`latency_profile.py` 用的是 `load_raw_corpus()`（不合并）。
    # 两个数都要报，理由见 `CorpusStats.raw_chunks` 的 docstring。
    raw_docs = load_raw_corpus(sources)
    raw_cfg = ChunkingConfig(chunk_chars=EVAL_CHUNK_CHARS, overlap_chars=EVAL_OVERLAP_CHARS)
    raw_chunks = len(build_strategy("recursive", cfg=raw_cfg).chunk(raw_docs, raw_cfg))

    stats = CorpusStats(
        documents=len(sources),
        segments=segments,
        chunks=by_cfg.get(EVAL_CHUNK_CHARS, count_at(EVAL_CHUNK_CHARS, EVAL_OVERLAP_CHARS)),
        chunk_chars=EVAL_CHUNK_CHARS,
        overlap_chars=EVAL_OVERLAP_CHARS,
        by_chunk_chars=by_cfg,
        raw_chunks=raw_chunks,
    )

    # 落盘**只在显式要求时**（默认 write=False，见 `measure` 的 docstring）。
    #
    # 缓存路径在 `eval/results/` 下但**不进版本库**（.gitignore 里挡掉），
    # 理由：它是派生数据，不是评测结果。混在一起会让人以为它是权威数字。
    if write:
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(stats)
        payload["by_chunk_chars"] = {str(k): v for k, v in stats.by_chunk_chars.items()}
        cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


__all__ = [
    "EVAL_CHUNK_CHARS",
    "EVAL_OVERLAP_CHARS",
    "REFERENCE_CONFIGS",
    "CorpusStats",
    "measure",
]
