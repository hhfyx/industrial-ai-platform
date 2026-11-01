"""`scripts/run_retrieval_eval.py` 的口径与失败处理测试。

**重点不是"指标算得对"，而是"数据出错时会不会被静默吞掉"。**
这套脚手架最容易犯的错是：锚点解析不到就跳过该条，
于是它被算成"未命中"，召回率凭空下降 —— 而人会去调参数，查错方向。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_retrieval_eval as rre  # noqa: E402

from app.rag.schema import Chunk  # noqa: E402
from app.rag.stores.base import SearchHit  # noqa: E402


def _chunk(text: str, *, source: str = "sse.md", index: int = 0, page: int | None = None) -> Chunk:
    return Chunk(text=text, page=page, source=source, index=index)


def _hit(text: str, *, source: str = "sse.md", index: int = 0, score: float = 0.6) -> SearchHit:
    return SearchHit(chunk=_chunk(text, source=source, index=index), score=score)


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _anchor(quote: str, source: str = "sse.md") -> dict:
    return {"source": source, "page": None, "head12": quote[:12], "quote": quote}


# --------------------------------------------------------------------------- 锚点反查


def test_resolve_anchor_finds_the_single_chunk() -> None:
    chunks = [_chunk("第一段无关内容。", index=0), _chunk("这里含锚点ABC并且唯一。", index=1)]
    resolved = rre.resolve_anchor(_anchor("锚点ABC"), chunks)
    assert resolved.index == 1


def test_resolve_anchor_reports_zero_match_with_reason() -> None:
    """0 块匹配必须抛，且提示两种可能原因（写错 / 被切碎）。

    静默跳过会让这条题算作"未命中"，召回率凭空下降 —— 而那看起来
    完全像"检索效果变差"，人会去调参数。
    """
    chunks = [_chunk("完全无关的内容。")]
    with pytest.raises(rre.AnchorResolutionError) as exc_info:
        rre.resolve_anchor(_anchor("不存在的锚点"), chunks)
    message = str(exc_info.value)
    assert "找不到" in message
    assert "切碎" in message  # 提示第二种成因


def test_resolve_anchor_reports_ambiguous_match() -> None:
    """>1 块匹配也要抛 —— 这时"命中哪一块"不确定，判定不可复现。"""
    chunks = [_chunk("重复片段XYZ在这里。", index=0), _chunk("又一次重复片段XYZ。", index=1)]
    with pytest.raises(rre.AnchorResolutionError, match="2 个块"):
        rre.resolve_anchor(_anchor("重复片段XYZ"), chunks)


def test_resolve_anchor_rejects_docs_prefix_source() -> None:
    """`docs/sse.md` 这种写法要给明确的提示。

    `Chunk.source` 只有文件名（loader 用 `file_path.name`）。
    上一轮在过滤评测里踩过这个坑，所以这里必须有专门的错误信息。
    """
    chunks = [_chunk("内容", source="sse.md")]
    with pytest.raises(rre.AnchorResolutionError) as exc_info:
        rre.resolve_anchor(_anchor("内容", source="docs/sse.md"), chunks)
    assert "docs/" in str(exc_info.value)


def test_resolve_anchor_requires_source_and_quote() -> None:
    with pytest.raises(rre.AnchorResolutionError, match="缺 source 或 quote"):
        rre.resolve_anchor({"source": "", "quote": "x"}, [_chunk("x")])


def test_resolve_all_collects_every_error_at_once() -> None:
    """**一次报所有错**，不是遇到第一个就停。

    50 条里错 5 条时逐个报告要跑五遍，而每遍都要十几秒建索引。
    """
    items = [
        rre.GoldItem(id="f-01", type="fact", question="q1", anchors=[_anchor("不存在的A")]),
        rre.GoldItem(id="f-02", type="fact", question="q2", anchors=[_anchor("不存在的B")]),
    ]
    with pytest.raises(rre.AnchorResolutionError) as exc_info:
        rre.resolve_all(items, [_chunk("无关")])
    message = str(exc_info.value)
    assert "f-01" in message
    assert "f-02" in message


def test_resolve_all_fills_ids_for_refusal_without_anchors() -> None:
    items = [rre.GoldItem(id="r-01", type="refusal", question="不该答的", anchors=[])]
    rre.resolve_all(items, [_chunk("任何内容")])
    assert items[0].gold_chunk_ids == []


def test_multi_hop_resolves_two_distinct_chunks() -> None:
    chunks = [_chunk("第一处证据AAA。", index=0), _chunk("第二处证据BBB。", index=1)]
    item = rre.GoldItem(
        id="m-01",
        type="multi_hop",
        question="合并两处才能答",
        anchors=[_anchor("证据AAA"), _anchor("证据BBB")],
    )
    rre.resolve_all([item], chunks)
    assert len(item.gold_chunk_ids) == 2
    assert item.gold_chunk_ids[0] != item.gold_chunk_ids[1]


# --------------------------------------------------------------------------- 命中判定


def _item_with(chunks: list[Chunk]) -> rre.GoldItem:
    return rre.GoldItem(
        id="f-01",
        type="fact",
        question="q",
        anchors=[],
        gold_chunk_ids=[chunk.id for chunk in chunks],
    )


def test_fact_hit_rank_counts_from_one() -> None:
    target = _chunk("目标块", index=7)
    hits = [_hit("别的", index=0), _hit("目标块", index=7)]
    item = _item_with([target])
    outcome = rre.evaluate_query(item=item, hits=hits, k=5)
    assert outcome.hit_rank == 2
    assert outcome.hit is True


def test_fact_miss_is_none_not_zero() -> None:
    """未命中必须是 `None`。

    用 0 当哨兵会让 `if rank:` 为假，而 `rank <= k` 却为真 ——
    于是"未命中"被算成"命中在第 0 位"，召回率虚高。
    """
    item = _item_with([_chunk("找不到的块", index=9)])
    outcome = rre.evaluate_query(item=item, hits=[_hit("别的")], k=5)
    assert outcome.hit_rank is None
    assert outcome.hit is False


def test_multi_hop_requires_all_anchors() -> None:
    """**多跳必须两个锚点都命中**，只中一个算未命中。

    只给一处模型答不出完整答案，算成命中会让数字虚高。
    """
    first = _chunk("证据一", index=0)
    second = _chunk("证据二", index=1)
    item = rre.GoldItem(
        id="m-01",
        type="multi_hop",
        question="q",
        anchors=[],
        gold_chunk_ids=[first.id, second.id],
    )
    hits_partial = [_hit("证据一", index=0)]
    assert rre.evaluate_query(item=item, hits=hits_partial, k=5).hit is False

    hits_full = [_hit("证据一", index=0), _hit("证据二", index=1)]
    outcome = rre.evaluate_query(item=item, hits=hits_full, k=5)
    assert outcome.hit is True
    # 排名取**较晚**的那一处：要看到两处，起决定作用的是后面的
    assert outcome.hit_rank == 2


def test_multi_hop_anchor_ranks_expose_which_one_failed() -> None:
    """`anchor_ranks` 要能看出"差在哪一处"，否则排查时只知道"没命中"。"""
    first = _chunk("证据一", index=0)
    item = rre.GoldItem(
        id="m-01",
        type="multi_hop",
        question="q",
        anchors=[],
        gold_chunk_ids=[first.id, "never-appears"],
    )
    outcome = rre.evaluate_query(item=item, hits=[_hit("证据一", index=0)], k=5)
    assert outcome.anchor_ranks[0] == 1
    assert outcome.anchor_ranks[1] is None


def test_refusal_query_has_no_hit_rank() -> None:
    """拒答题没有"命中"，它由阈值单独判 —— 不能混进召回率。"""
    item = rre.GoldItem(id="r-01", type="refusal", question="q", anchors=[])
    outcome = rre.evaluate_query(item=item, hits=[_hit("碰巧像的")], k=5)
    assert outcome.hit_rank is None
    assert outcome.top_score == pytest.approx(0.6)


# --------------------------------------------------------------------------- 汇总


def _outcome(qid: str, kind: str, rank: int | None, score: float = 0.6) -> rre.QueryOutcome:
    return rre.QueryOutcome(id=qid, type=kind, hit_rank=rank, anchor_ranks=(rank,), top_score=score)


def test_recall_and_mrr_are_computed_over_all_answerable() -> None:
    """分母是**全部可答题**，不是"命中的那些"。"""
    items = [
        rre.GoldItem(id="f-01", type="fact", question="", anchors=[]),
        rre.GoldItem(id="f-02", type="fact", question="", anchors=[]),
    ]
    outcomes = [_outcome("f-01", "fact", 1), _outcome("f-02", "fact", None)]
    metrics = rre.summarize(outcomes=outcomes, items=items, ks=(1, 5), threshold=0.4)
    assert metrics["recall"]["recall@1"]["rate"] == pytest.approx(0.5)
    assert metrics["recall"]["recall@5"]["rate"] == pytest.approx(0.5)
    # MRR = (1/1 + 0) / 2
    assert metrics["mrr"] == pytest.approx(0.5)


def test_false_refusal_counts_only_low_score_not_recall_miss() -> None:
    """**「误拒」与「召回未命中」必须分开。**

    这是第一版的真实 bug：`false_refusal` 写成"所有未命中的可答题"，
    实测 40 条里 19 条被算成"误拒" —— 而那 19 条的 top1 分数大都在 0.6 以上，
    **它们根本没被阈值挡，只是没检索准**。

    混淆后的报告会同时出现「误拒 19 条」与「正确拒答 0/10」，
    读起来像"拒答机制挡掉了 19 条真答案"，而实际挡掉 0 条。
    """
    items = [
        rre.GoldItem(id="f-01", type="fact", question="", anchors=[]),  # 未命中但分数高
        rre.GoldItem(id="f-02", type="fact", question="", anchors=[]),  # 未命中且分数低
        rre.GoldItem(id="r-01", type="refusal", question="", anchors=[]),
    ]
    outcomes = [
        # 检索没命中，但 top1 分数 0.62 高于阈值 0.4 → 不是误拒，是召回问题
        _outcome("f-01", "fact", None, score=0.62),
        # 检索没命中，且 top1 0.20 低于阈值 → 这才是误拒
        _outcome("f-02", "fact", None, score=0.20),
        _outcome("r-01", "refusal", None, score=0.90),  # 该拒却高分 → 误放
    ]
    metrics = rre.summarize(outcomes=outcomes, items=items, ks=(5,), threshold=0.4)
    refusal = metrics["refusal"]
    assert refusal["false_refusal"] == ["f-02"], "只有低分那条算误拒"
    assert refusal["recall_miss"] == ["f-01", "f-02"], "两条都没召回"
    assert refusal["false_accept"] == ["r-01"]


def test_overall_accuracy_requires_both_hit_and_above_threshold() -> None:
    """端到端"答对" = 检索命中 **且** 没被阈值拒。两个条件缺一不可。"""
    items = [
        rre.GoldItem(id="f-01", type="fact", question="", anchors=[]),  # 命中且高分 → 对
        rre.GoldItem(id="f-02", type="fact", question="", anchors=[]),  # 命中但低分 → 错
    ]
    outcomes = [
        _outcome("f-01", "fact", 1, score=0.8),
        _outcome("f-02", "fact", 2, score=0.2),
    ]
    metrics = rre.summarize(outcomes=outcomes, items=items, ks=(5,), threshold=0.4)
    assert metrics["refusal"]["overall_accuracy"] == pytest.approx(0.5)


def test_refusal_accuracy_separates_false_refusal_and_false_accept() -> None:
    """**三个错误分开报**：误拒 / 误放 / 召回未命中。

    合成一个"合计错"会把权衡藏起来 —— `docs/retrieval-params.md` §3.3
    已经踩过：单看合计错会选出牺牲大量真答案的阈值。
    """
    items = [
        rre.GoldItem(id="f-01", type="fact", question="", anchors=[]),
        rre.GoldItem(id="r-01", type="refusal", question="", anchors=[]),
        rre.GoldItem(id="r-02", type="refusal", question="", anchors=[]),
    ]
    outcomes = [
        _outcome("f-01", "fact", None, score=0.20),  # 该答没答且低分 → 误拒
        _outcome("r-01", "refusal", None, score=0.20),  # 该拒且拒了 ✓
        _outcome("r-02", "refusal", None, score=0.90),  # 该拒却高分 → 误放
    ]
    metrics = rre.summarize(outcomes=outcomes, items=items, ks=(5,), threshold=0.4)
    refusal = metrics["refusal"]
    assert refusal["refused_correctly"] == 1
    assert refusal["refusal_accuracy"] == pytest.approx(0.5)
    assert refusal["false_refusal"] == ["f-01"]
    assert refusal["false_accept"] == ["r-02"]


def test_by_type_splits_fact_and_multi_hop() -> None:
    """多跳与事实型分开报 —— 混看会稀释事实型的真实水位。"""
    items = [
        rre.GoldItem(id="f-01", type="fact", question="", anchors=[]),
        rre.GoldItem(id="m-01", type="multi_hop", question="", anchors=[]),
    ]
    outcomes = [_outcome("f-01", "fact", 1), _outcome("m-01", "multi_hop", None)]
    metrics = rre.summarize(outcomes=outcomes, items=items, ks=(5,), threshold=0.4)
    assert metrics["by_type"]["fact"]["recall@5"] == 1.0
    assert metrics["by_type"]["multi_hop"]["recall@5"] == 0.0


# ------------------------------------------------- 检索窗口 max(KS) vs 线上 top_k


def test_recall_miss_honours_top_k_not_the_search_window() -> None:
    """**`recall_miss` 必须按 top_k 判，不能按"在检索窗口里出现过"。**

    这是 W6 修的真实 bug：一次运行拉 top10（`max(KS)`）读多档 recall，
    而 `recall_miss` 写成 `not o.hit`（= 不在 top10 里）。
    结果 40 条可答里排第 7 的题不算 miss：

        报告写「召回未命中 23 条」，而同一页的 Recall@5 显示未命中 24 条

    差的那条实测是 m-04（排第 7）。它还被 `export_error_cases` 一并跳过，
    于是"错例表 34 条 / 指标说 30 条"，没人看得出为什么。
    """
    items = [
        rre.GoldItem(id="f-01", type="fact", question="", anchors=[]),  # top1，在 top5 内
        rre.GoldItem(id="f-02", type="fact", question="", anchors=[]),  # top7，只在 top10 内
        rre.GoldItem(id="f-03", type="fact", question="", anchors=[]),  # 彻底没进窗口
    ]
    outcomes = [
        _outcome("f-01", "fact", 1),
        _outcome("f-02", "fact", 7),
        _outcome("f-03", "fact", None),
    ]
    metrics = rre.summarize(outcomes=outcomes, items=items, ks=(5, 10), threshold=0.4, top_k=5)
    # top7 那条在 top10 里算命中……但 recall_miss 的口径是 top5
    assert metrics["recall"]["recall@10"]["hits"] == 2
    assert metrics["recall"]["recall@5"]["hits"] == 1
    assert metrics["refusal"]["recall_miss"] == ["f-02", "f-03"], (
        "top7 在 top5 口径下就是没召回 —— 不能因为它在 top10 里就漏掉"
    )


def test_recall_miss_count_matches_recall_at_top_k() -> None:
    """恒等式：`len(recall_miss) == answerable - recall@top_k.hits`。

    这条是把上面那个 bug 锁死在数字层面 —— 只要两者对不上，
    报告就自相矛盾，无论措辞怎么写。
    """
    items = [
        rre.GoldItem(id=f"f-{i:02d}", type="fact", question="", anchors=[]) for i in range(1, 6)
    ]
    outcomes = [
        _outcome("f-01", "fact", 1),
        _outcome("f-02", "fact", 3),
        _outcome("f-03", "fact", 6),
        _outcome("f-04", "fact", 11),
        _outcome("f-05", "fact", None),
    ]
    for top_k in (1, 3, 5, 8):
        metrics = rre.summarize(
            outcomes=outcomes, items=items, ks=(1, 3, 5, 8, 10), threshold=0.4, top_k=top_k
        )
        expected = metrics["answerable"] - metrics["recall"][f"recall@{top_k}"]["hits"]
        assert len(metrics["refusal"]["recall_miss"]) == expected, (
            f"top_k={top_k} 时召回未命中条数与 Recall@{top_k} 的缺口对不上"
        )


def test_overall_accuracy_uses_the_same_top_k_as_recall() -> None:
    """端到端答对的分子必须是 **top_k 内命中**，不能是"窗口内命中"。

    第一版分子用 `o.hit`（top10 命中）配阈值，而分母旁边的 Recall@5 用 top5：
    分子 top10、分母 top5，两个数根本不可比。
    """
    items = [
        rre.GoldItem(id="f-01", type="fact", question="", anchors=[]),  # top1 高分 → 对
        rre.GoldItem(id="f-02", type="fact", question="", anchors=[]),  # top7 高分 → 不算对
    ]
    outcomes = [
        _outcome("f-01", "fact", 1, score=0.8),
        _outcome("f-02", "fact", 7, score=0.8),
    ]
    metrics = rre.summarize(outcomes=outcomes, items=items, ks=(5,), threshold=0.4, top_k=5)
    assert metrics["refusal"]["overall_accuracy"] == pytest.approx(0.5), (
        "top7 不算端到端答对 —— 它没进 top5"
    )


def test_evaluate_query_truncates_to_the_search_window() -> None:
    """`k` 必须真的截断候选。第一版传了 k 却完全没读，传 k=5 实际按 10 条判。"""
    gold = _chunk("gold")
    item = rre.GoldItem(id="f-01", type="fact", question="", anchors=[], gold_chunk_ids=[gold.id])
    hits = [_hit(f"noise{i}", index=i + 1) for i in range(9)] + [_hit("gold")]

    narrow = rre.evaluate_query(item=item, hits=hits, k=5)
    wide = rre.evaluate_query(item=item, hits=hits, k=10)
    assert narrow.hit_rank is None, "gold 排第 10，k=5 时应判为未进窗口"
    assert wide.hit_rank == 10


# --------------------------------------------------------------------------- 数据校验


def test_load_questions_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {"id": "f-01", "type": "fact", "question": "a", "anchors": [_anchor("x1")]},
            {"id": "f-01", "type": "fact", "question": "b", "anchors": [_anchor("x2")]},
        ],
    )
    with pytest.raises(ValueError, match="id 重复"):
        rre.load_questions(path)


@pytest.mark.parametrize(
    ("kind", "anchors", "expected"),
    [("fact", 2, "1 个锚点"), ("multi_hop", 1, "2 个锚点"), ("refusal", 1, "0 个锚点")],
)
def test_load_questions_enforces_anchor_count(
    tmp_path: Path, kind: str, anchors: int, expected: str
) -> None:
    """锚点数量必须与 type 匹配。**少了会让多跳退化成单跳而不报警。**"""
    path = _write(
        tmp_path,
        [
            {
                "id": "x-01",
                "type": kind,
                "question": "q",
                "anchors": [_anchor(f"quote{i}") for i in range(anchors)],
            }
        ],
    )
    with pytest.raises(ValueError, match=expected):
        rre.load_questions(path)


def test_load_questions_rejects_unknown_type(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"id": "x-01", "type": "guess", "question": "q", "anchors": []}])
    with pytest.raises(ValueError, match="type 非法"):
        rre.load_questions(path)


def test_leak_check_flags_question_containing_quote_head() -> None:
    """问题照抄了 quote 里的中文实句 → 报出来。

    这不是检索问题，是数据质量问题：照抄时纯字面匹配（hashing）也能满分，
    于是这张表分不出"换 embedder"的影响。

    判据是**最长公共子串 ≥5 字符且有 ≥4 个汉字**，不是"前 10 字是否出现"。
    """
    items = [
        rre.GoldItem(
            id="f-01",
            type="fact",
            question="支付合同价程是怎么被切开的",
            anchors=[_anchor("LangChain 切出了 支付合同价程 这样的痕迹")],
        )
    ]
    problems = rre.check_no_leak(items)
    assert problems
    assert "f-01" in problems[0]


def test_leak_check_ignores_shared_identifier() -> None:
    """共享一个产品名/术语**不算**泄题。

    实测的唯一一条误报：问题「LangChain 的切分留下了什么痕迹？」
    而 quote 以 `LangChain` 开头。问这件事必须提到产品名 ——
    要求"不提产品名也能问"，等于要求问题写得不像人话。

    判据因此要求公共子串**含 ≥4 个汉字**：`LangChain` 是纯 ASCII，不算。
    """
    items = [
        rre.GoldItem(
            id="f-01",
            type="fact",
            question="LangChain 这款工具的切分表现如何",
            anchors=[_anchor("LangChain 切出了 支付合同价程 这样的痕迹")],
        )
    ]
    assert rre.check_no_leak(items) == []


def test_leak_check_passes_on_paraphrased_question() -> None:
    items = [
        rre.GoldItem(
            id="f-01",
            type="fact",
            question="首字延迟是怎么定义的",
            anchors=[_anchor("TTFT（Time To First Token）= 从发起请求到收")],
        )
    ]
    assert rre.check_no_leak(items) == []


# --------------------------------------------------------------------------- 报告


def test_markdown_report_contains_the_disclaimer() -> None:
    """报告必须带"这是调参集"的免责声明。

    少了它，50 条的数字迟早会被摘出来写成"准确率 X%"。
    """
    payload = {
        "meta": {
            "generated_at": "2026-11-20 21:00:00",
            "chunk_chars": 500,
            "overlap": 50,
            "strategy": "recursive",
            "embedder": "bge",
        },
        "metrics": {
            "total_questions": 50,
            "answerable": 40,
            "refusals": 10,
            "recall": {"recall@5": {"hits": 30, "total": 40, "rate": 0.75}},
            "mrr": 0.6,
            "by_type": {},
            "threshold": 0.4,
            "refusal": {
                "refused_correctly": 4,
                "refusal_total": 10,
                "refusal_accuracy": 0.4,
                "false_refusal": [],
                "false_accept": ["r-01"],
                "recall_miss": [],
                "overall_accuracy": 0.68,
                "note": "合成值仅作参考",
            },
        },
        "per_question": [],
    }
    markdown = rre.render_markdown(payload)
    assert "调参集" in markdown
    assert "不能用来报" in markdown
    assert "Recall@k" in markdown
    # 报告必须区分"误拒"与"召回未命中"，否则读者会把召回问题当成拒答问题
    assert "召回未命中" in markdown
