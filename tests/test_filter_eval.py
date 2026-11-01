"""`scripts/filter_eval.py` 的口径测试。

评测脚本本身也需要测试，因为它的**判定口径**一旦错，产出的
"过滤提升 25pp"就是假的 —— 而那种错误看起来完全正常。

这里只测纯函数与数据契约（不跑 bge）：
- gold 的 `source` 与块的 `source` 的映射对不对
- "命中位置"的判定有没有把未命中当成命中
- 表格金标确实落在表格块上（否则"仅表格"场景测的是别的东西）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import filter_eval as fe  # noqa: E402

from app.rag import Document  # noqa: E402
from app.rag.chunking import ChunkingConfig, build_strategy  # noqa: E402
from app.rag.stores.base import SearchHit  # noqa: E402

# --------------------------------------------------------------------------- source 映射


@pytest.mark.parametrize(
    ("gold_source", "expected"),
    [
        ("docs/sse.md", "sse.md"),
        ("README.md", "README.md"),
        ("docs/weekly/w1.md", "w1.md"),
    ],
)
def test_gold_source_maps_to_chunk_source(gold_source: str, expected: str) -> None:
    """gold 写相对路径，`Chunk.source` 只有文件名。

    这个映射错了的后果很具体：`where={"source": "docs/sse.md"}` 会
    抛 `UnknownFilterValue`（上一轮的校验拦住了），或者在没有校验的实现上
    静默返回 0 条 —— 而后者会被读成"过滤让命中率归零"。
    """
    assert fe.gold_source_to_chunk_source(gold_source) == expected


def test_every_gold_source_maps_to_an_existing_chunk_source() -> None:
    """每一条 gold 的 source 都必须能在语料里找到对应块。

    找不到的话，那条 gold 在"单文档"场景里必然"未命中" ——
    而那会被误读成"过滤器不好用"。
    """
    from retrieval_harness import load_corpus

    corpus = load_corpus()
    chunk_sources = {doc.source for doc in corpus.all_docs}
    gold = fe.load_gold()
    missing = [
        (item.id, fe.gold_source_to_chunk_source(item.source))
        for item in gold
        if fe.gold_source_to_chunk_source(item.source) not in chunk_sources
    ]
    assert not missing, f"这些 gold 的 source 在语料里找不到：{missing}"


# --------------------------------------------------------------------------- 命中判定


def _hit(text: str, source: str = "sse.md", index: int = 0) -> SearchHit:
    from app.rag.schema import Chunk

    return SearchHit(
        chunk=Chunk(text=text, page=None, source=source, index=index),
        score=0.5,
    )


def test_rank_of_finds_anchor_at_position() -> None:
    """命中位置从 1 开始，且按列表顺序。"""
    hits = [_hit("无关一"), _hit("无关二"), _hit("这里含锚点ABC")]
    assert fe.rank_of(hits=hits, anchor="锚点ABC") == 3


def test_rank_of_returns_none_when_absent() -> None:
    """未命中必须是 `None`，不能是 0 或 -1。

    用 0 当"未命中"的哨兵会让 `if rank:` 判断为假 —— 于是**top0 也算未命中**，
    而 top0 这个位置并不存在，属于把两种状态混成一个。
    """
    assert fe.rank_of(hits=[_hit("无关")], anchor="不存在") is None


def test_rank_of_returns_first_occurrence() -> None:
    """同一锚点出现在多条里时取**最先**的那条（即排名最靠前的）。"""
    hits = [_hit("锚点X"), _hit("锚点X")]
    assert fe.rank_of(hits=hits, anchor="锚点X") == 1


# --------------------------------------------------------------------------- 表格金标


def test_table_gold_anchors_land_in_exactly_one_table_chunk() -> None:
    """**表格金标的每条 anchor 必须唯一落在一个 `block="table"` 块里。**

    这是"仅表格"场景有效的前提。不满足时：
    - 落在 0 个 → 那条 gold 永远未命中，把命中率压低
    - 落在多个 → 过滤前后都可能命中，测不出过滤的效果
    """
    items = fe.load_table_gold()
    assert items, "表格金标不能为空"

    corpus_sources = fe.PROJECT_ROOT / "docs"
    docs: list[Document] = []
    from app.rag.loaders import load_document

    for path in sorted(corpus_sources.rglob("*.md")):
        docs.extend(load_document(path).documents)
    docs.extend(load_document(fe.PROJECT_ROOT / "README.md").documents)

    cfg = ChunkingConfig(chunk_chars=500, overlap_chars=0)
    chunks = build_strategy("recursive", cfg=cfg).chunk(docs, cfg)

    problems: list[str] = []
    for item in items:
        table_hits = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("block") == "table" and item.anchor in chunk.text
        ]
        if len(table_hits) != 1:
            problems.append(f"{item.id}: 落在 {len(table_hits)} 个表格块")
    assert not problems, "表格金标定位不准：\n" + "\n".join(problems)


def test_main_gold_does_not_overlap_table_gold() -> None:
    """两套金标不该有同一条 anchor —— 否则汇总时同一条被算两次。"""
    table_ids = {item.id for item in fe.load_table_gold()}
    main_ids = {item.id for item in fe.load_gold()}
    assert not (table_ids & main_ids)


def test_gold_ids_are_unique_within_each_set() -> None:
    for loader in (fe.load_gold, fe.load_table_gold):
        items = loader()
        ids = [item.id for item in items]
        assert len(ids) == len(set(ids)), f"{loader.__name__} 里有重复 id"


# --------------------------------------------------------------------------- 产物契约


def test_misconfiguration_probe_covers_the_four_misspellings() -> None:
    """探针必须覆盖四类写错方式 —— 少一类就有一条静默失败没被记录。

    这些是实测出来"原样查库返回 0 条"的情况（见 `filter_eval.json`）。
    其中"文档名带目录"最容易发生：gold 里就写的是 `docs/sse.md`。
    """
    import inspect

    source = inspect.getsource(fe.probe_misconfigurations)
    for expected in ("键名拼错", "文档名拼错", "文档名带目录", "section_path"):
        assert expected in source, f"探针漏了「{expected}」这一类"


def test_filter_eval_artifact_shape_if_present() -> None:
    """如果产物已生成，检查它有三条场景且每条都记了救回/丢失。

    产物不入库（`data/` 之外但在 `scripts/` 下），所以用 skip 而不是 assert 存在 ——
    但**一旦存在就必须完整**，避免半份数据被当成结论。
    """
    path = fe.DEFAULT_OUT
    if not path.exists():
        pytest.skip("filter_eval.json 尚未生成")
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = {row["scenario"] for row in payload["scenarios"]}
    assert names == {"单文档", "仅表格", "按章节"}
    for row in payload["scenarios"]:
        assert "gained" in row and "lost" in row
        assert row["queries"] > 0, f"{row['scenario']} 没有样本，不该出现在产物里"
