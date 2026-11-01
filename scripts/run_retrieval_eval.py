"""检索评测脚手架：**自算 Recall@k / MRR / 拒答准确率，不调 LLM**。

## 一条命令

    python -X utf8 scripts/run_retrieval_eval.py

产出 `eval/results/retrieval-<时间戳>.json` 与同名 `.md`。
想固定名字加 `--out`。

## 为什么纯距离就够

这套指标全部只依赖"哪一块被排在第几位"，不需要模型参与：

| 指标 | 定义 | 依赖 |
| --- | --- | --- |
| `recall@k` | 该命中的题里，top_k 命中的比例 | 排序 |
| `mrr` | 命中排名的倒数平均 | 排序 |
| `refusal_accuracy` | 该拒的拒了 + 该答的答了 / 全部 | 排序 + 阈值 |

**不调 LLM 是刻意的**：一旦引入生成，指标里就混进了模型当天的状态
（温度、网关波动、我改过的 prompt），而那时"召回降了"到底是检索的问题
还是模型的问题就分不出来了。

## 锚点反查：这套脚手架的核心（也是唯一有难度的部分）

`eval/questions-50.jsonl` 里**没有**硬编码 `chunk.id`，只有原文锚点：

```json
{"source": "sse.md", "page": null, "head12": "TTFT（Time To", "quote": "TTFT（Time To First Token）= …"}
```

原因见 `eval/README.md` §一：`Chunk.id` 是 `sha256(source + page + 正文)` 的派生值，
**改切分参数就变**（实测 chunk_chars 300/500/800 各不同）。

所以运行时要做三件事：

1. 按 `source` 找当前索引里该文档的块
2. 找**包含 `quote`** 的那一块 → 它的 `chunk.id` 就是 gold
3. **找不到或找到多块都要报错**，不能跳过

### 第 3 条是重点：三种失败必须分开报

| 情况 | 含义 | 该怎么做 |
| --- | --- | --- |
| 0 块匹配 | 锚点写错了 / 这句被切碎到没有一个块包含它 | 改数据或改切分参数 |
| >1 块匹配 | quote 不够唯一（同一段文字出现在两处） | 改 quote |
| 恰好 1 块 | 正常 | —— |

**静默跳过是最坏的处理**：一条题解析不到 gold，它会被算成"未命中"，
于是召回率凭空下降，而人会去调参数 —— 查错方向。
这与 W4 的 `assert_gold_exists()` 是同一条纪律。

## 多跳的评分口径

`hit@k` = **两个 anchor 对应的块都在 top_k 里**。只命中一个算未命中。

理由：多跳题的价值就是"模型要同时看到两处"。只给一处它答不出完整答案，
算成命中会让数字虚高 —— 而虚高正是这套集子最该防的事。

## 拒答准确率怎么算

拒答题没有 anchor，判据是 **top1 分数 < 阈值** → 正确地拒了。

    拒答准确率 = (拒答题里正确拒的 + 事实/多跳题里正确答的) / 全部 50 条

**这个指标会把"该答却拒"和"该拒却答"混在一起**，所以报告里同时给
两个分开的数（`false_refusal` / `false_accept`）。合成一个数会把权衡藏起来
（`docs/retrieval-params.md` §3.3 已经踩过这个坑：单看"合计错"会选出
牺牲大量真答案的阈值）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from retrieval_harness import load_corpus, load_raw_corpus  # noqa: E402

EVAL_FILE: Path = PROJECT_ROOT / "eval" / "questions-50.jsonl"
RESULTS_DIR: Path = PROJECT_ROOT / "eval" / "results"

# 报告的 k 值。**与 tasks 里 top_k 的选择对齐**（k 从 1 到 10，
# 这样任何一档的 recall 都能从同一份运行里读出来，不必跑四遍）。
KS: tuple[int, ...] = (1, 3, 5, 8, 10)


@dataclass
class GoldItem:
    """一条评测题。`anchors` 是唯一真值，`gold_chunk_ids` 是运行时派生的。"""

    id: str
    type: str
    question: str
    anchors: list[dict[str, Any]]
    gold_answer: str = ""
    notes: str = ""
    gold_chunk_ids: list[str] = field(default_factory=list)

    @property
    def is_refusal(self) -> bool:
        return self.type == "refusal"


class AnchorResolutionError(RuntimeError):
    """锚点解析失败。**三种失败各有独立的错误信息**（见模块 docstring）。"""


def load_questions(path: Path = EVAL_FILE) -> list[GoldItem]:
    """读评测集。**格式错误立刻抛**，不跳过。"""
    if not path.exists():
        raise FileNotFoundError(f"评测集不存在：{path}（先按 eval/README.md 标注）")

    items: list[GoldItem] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{lineno} 不是合法 JSON：{exc}") from exc

        for key in ("id", "type", "question", "anchors"):
            if key not in raw:
                raise ValueError(f"{path.name}:{lineno} 缺字段 {key!r}")

        if raw["id"] in seen:
            raise ValueError(f"{path.name}:{lineno} id 重复：{raw['id']}")
        seen.add(raw["id"])

        kind = raw["type"]
        if kind not in ("fact", "multi_hop", "refusal"):
            raise ValueError(f"{path.name}:{lineno} type 非法：{kind!r}")

        expected_anchors = {"fact": 1, "multi_hop": 2, "refusal": 0}[kind]
        if len(raw["anchors"]) != expected_anchors:
            raise ValueError(
                f"{path.name}:{lineno} {kind} 应有 {expected_anchors} 个锚点，"
                f"实为 {len(raw['anchors'])}"
            )

        items.append(
            GoldItem(
                id=raw["id"],
                type=kind,
                question=raw["question"],
                anchors=list(raw["anchors"]),
                gold_answer=raw.get("gold_answer", ""),
                notes=raw.get("notes", ""),
                # 文件里必须是空的；在这里填说明有人在手填，那是错的
                gold_chunk_ids=list(raw.get("gold_chunk_ids") or []),
            )
        )
    return items


def resolve_anchor(
    anchor: dict[str, Any],
    chunks: list[Any],
) -> Any:  # noqa: ANN401
    """把一个锚点解析成当前的 chunk。**三种失败各自的错误信息不同。**"""
    source = str(anchor.get("source", ""))
    quote = str(anchor.get("quote", ""))
    if not source or not quote:
        raise AnchorResolutionError(f"锚点缺 source 或 quote：{anchor!r}")

    same_doc = [chunk for chunk in chunks if chunk.source == source]
    if not same_doc:
        available = sorted({chunk.source for chunk in chunks})
        raise AnchorResolutionError(
            f"锚点的 source {source!r} 在当前索引里不存在。"
            f"可用：{'、'.join(available)}。"
            "（常见原因：写成了 'docs/sse.md'，而 Chunk.source 只有文件名）"
        )

    matched = [chunk for chunk in same_doc if quote in chunk.text]
    if not matched:
        raise AnchorResolutionError(
            f"在 {source} 的 {len(same_doc)} 个块里找不到包含该 quote 的块。"
            f"quote 前 20 字：{quote[:20]!r}\n"
            "  可能原因：① quote 写错（不是逐字原文）② 该句被切碎，"
            "没有任何一块完整包含它 —— 换一个更短、更居中的片段"
        )
    if len(matched) > 1:
        raise AnchorResolutionError(
            f"在 {source} 里有 {len(matched)} 个块包含该 quote，无法确定 gold。"
            f"quote 前 20 字：{quote[:20]!r}\n"
            "  原因：这段文字在文档里出现了多次。换一个唯一的片段"
        )
    return matched[0]


def resolve_all(items: list[GoldItem], chunks: list[Any]) -> None:
    """把所有锚点解析成 chunk id，**就地写回 `gold_chunk_ids`**。"""
    errors: list[str] = []
    for item in items:
        if item.is_refusal:
            item.gold_chunk_ids = []
            continue
        resolved: list[str] = []
        for anchor in item.anchors:
            try:
                chunk = resolve_anchor(anchor, chunks)
            except AnchorResolutionError as exc:
                errors.append(f"  ✗ {item.id}: {exc}")
                continue
            resolved.append(chunk.id)
        item.gold_chunk_ids = resolved

    if errors:
        # **一次把所有错报出来**，而不是遇到第一个就停 —— 50 条里错 5 条时，
        # 逐个报告要跑五遍，而每遍都要 12 秒建索引。
        raise AnchorResolutionError(
            f"{len(errors)} 个锚点解析失败（共 {len(items)} 条）：\n" + "\n".join(errors)
        )


def check_no_leak(items: list[GoldItem]) -> list[str]:
    """检查"问题有没有照抄原文"。

    这条是**数据质量**检查，不是检索检查：照抄原文会让纯字面匹配
    （`HashingEmbedder`）也满分，于是这张表分不出"换 embedder"的影响。

    ## 判据：问题与 quote 的**最长公共子串**，而不是"前 10 字是否出现"

    第一版写成 `quote[:10] in question`，抓出来的唯一一条是：

        f-22  问题「同一份中文合同上，LangChain 的切分留下了什么痕迹？」
              quote「LangChain 切出了 …支付合同价程…」

    重叠的正是产品名 `LangChain` —— 问这件事**必须**提到它。
    要求"不提产品名也能问"，等于要求问题写得不像人话。

    所以判据换成两条同时满足：

    1. 最长公共子串 **≥ 5 字符**
    2. 该子串含 **≥ 4 个汉字**

    第 2 条把"共享一个标识符"排除掉：`LangChain`（9 个 ASCII 字符）
    长度够了但不含汉字，不算泄题；而"支付合同价程"（6 个汉字）才是照抄。

    **阈值为什么是 5 而不是 8**：实测"支付合同价程是怎么被切开的"
    与原文的重叠只有 6 个字 —— 口语化改写后的中文句子，与原文的重叠
    通常就是 4~6 字的短语。设 8 会让这条检查在真实数据上**从不触发**，
    等于关掉它。这个数是实测调出来的，不是拍的。

    > 第一版还有一个更严重的 bug：它按标点截断取首段，而中文句子开头
    > 常是"第五条 付款计划：…"，截到第一个空格只剩 3 字、低于阈值，
    > 于是**所有条目都返回空串、检查形同关闭**。最长公共子串没有这个问题：
    > 它不假设标点位置，只回答"这两串文字重叠了多少"。
    """
    problems: list[str] = []
    for item in items:
        for anchor in item.anchors:
            quote = str(anchor.get("quote", ""))
            overlap = _longest_common_substring(item.question, quote)
            if len(overlap) >= 5 and _cjk_count(overlap) >= 4:
                problems.append(f"  ✗ {item.id}: 问题照抄了 quote 的 {len(overlap)} 字 {overlap!r}")
    return problems


def _longest_common_substring(left: str, right: str) -> str:
    """最长公共子串。用 `difflib` 的成熟实现，不手写 DP。"""
    from difflib import SequenceMatcher

    match = SequenceMatcher(None, left, right, autojunk=False).find_longest_match(
        0, len(left), 0, len(right)
    )
    return left[match.a : match.a + match.size]


def _cjk_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


@dataclass
class QueryOutcome:
    """一条题在某份索引上的结果。"""

    id: str
    type: str
    hit_rank: int | None
    """单锚点：命中的排名。多锚点：**全部命中**的排名（最靠后的那个）。"""

    anchor_ranks: tuple[int | None, ...]
    """每个锚点各自的排名。多跳题用它看"差在哪一处"。"""

    top_score: float | None
    top_k_ids: list[str] = field(default_factory=list)

    @property
    def hit(self) -> bool:
        return self.hit_rank is not None

    @property
    def all_anchors_hit(self) -> bool:
        return bool(self.anchor_ranks) and all(rank is not None for rank in self.anchor_ranks)


def evaluate_query(*, item: GoldItem, hits: list[Any], k: int) -> QueryOutcome:
    """判定一条题。

    ## 多跳的口径：**全部锚点都命中才算命中**

    排名取"最靠后的那个锚点"的排名 —— 因为要看到两处，
    起决定作用的是较晚出现的那一处。只命中一个 → 记 `None`（未命中）。
    """
    positions = {hit.chunk.id: index for index, hit in enumerate(hits, start=1)}
    ranks = tuple(positions.get(chunk_id) for chunk_id in item.gold_chunk_ids)

    if item.is_refusal:
        rank: int | None = None  # 拒答题没有"命中"，由阈值单独判
    elif all(r is not None for r in ranks):
        rank = max(r for r in ranks if r is not None)
    else:
        rank = None

    return QueryOutcome(
        id=item.id,
        type=item.type,
        hit_rank=rank,
        anchor_ranks=ranks,
        top_score=hits[0].score if hits else None,
        top_k_ids=[hit.chunk.id for hit in hits],
    )


def summarize(
    *,
    outcomes: list[QueryOutcome],
    items: list[GoldItem],
    ks: tuple[int, ...],
    threshold: float,
) -> dict[str, Any]:
    """把逐条结果压成指标。**每一项的分母都写在报告里。**"""
    by_id = {item.id: item for item in items}
    answerable = [o for o in outcomes if not by_id[o.id].is_refusal]
    refusals = [o for o in outcomes if by_id[o.id].is_refusal]

    recall: dict[str, Any] = {}
    for k in ks:
        hit = sum(1 for o in answerable if o.hit and o.hit_rank <= k)
        recall[f"recall@{k}"] = {
            "hits": hit,
            "total": len(answerable),
            "rate": round(hit / len(answerable), 4) if answerable else 0.0,
        }

    # MRR 用 top_k 里能看到的范围算；这里用最大的 k（KS 的末位）。
    max_k = max(ks)
    mrr_hits = [o for o in answerable if o.hit and o.hit_rank <= max_k]
    mrr = round(sum(1 / o.hit_rank for o in mrr_hits) / len(answerable), 4) if answerable else 0.0

    # 按题型分开报：多跳的召回天然低于事实型，混在一起会让事实型的真实水平被稀释
    by_type: dict[str, Any] = {}
    for kind in ("fact", "multi_hop"):
        group = [o for o in answerable if o.type == kind]
        if not group:
            continue
        hit5 = sum(1 for o in group if o.hit and o.hit_rank <= 5)
        by_type[kind] = {
            "total": len(group),
            "recall@5": round(hit5 / len(group), 4),
            "mrr": round(sum(1 / o.hit_rank for o in group if o.hit) / len(group), 4),
        }

    # 拒答：top1 < threshold 就是正确地拒了
    refused_correctly = [o for o in refusals if o.top_score is not None and o.top_score < threshold]

    # ---- 「误拒」与「召回未命中」是两件不同的事，**不能混成一个数**
    #
    # 第一版写成 `false_refusal = [o for o in answerable if not o.hit]`，
    # 结果 40 条可答题里有 19 条"未进入 top5"，全被算成"误拒" ——
    # 但那 19 条的 top1 分数大都 0.6 以上，**它们根本没被拒，只是没检索准**。
    #
    # 混淆的后果：报告里会同时出现「误拒 19 条」与「正确拒答 0/10」，
    # 读者会以为"拒答机制把 19 条真答案挡掉了"，而实际挡掉的是 0 条。
    #
    # 正确口径：**只有 top1 低于阈值才是"被拒"**。检索未命中是召回问题，
    # 由 Recall@k 单独度量。
    false_refusal = [
        o.id for o in answerable if o.top_score is not None and o.top_score < threshold
    ]
    # 召回失败（进了候选但排不进 top5）：**单独一类**，不属于拒答
    recall_miss = [o.id for o in answerable if not o.hit]
    false_accept = [o.id for o in refusals if o not in refused_correctly]

    # 端到端"答对了"：检索命中 **且** 没被阈值拒
    answered_correctly = [
        o for o in answerable if o.hit and o.top_score is not None and o.top_score >= threshold
    ]

    total = len(outcomes)
    correct = len(refused_correctly) + len(answered_correctly)
    return {
        "total_questions": total,
        "answerable": len(answerable),
        "refusals": len(refusals),
        "recall": recall,
        "mrr": mrr,
        "by_type": by_type,
        "threshold": threshold,
        "refusal": {
            "refused_correctly": len(refused_correctly),
            "refusal_total": len(refusals),
            "refusal_accuracy": round(len(refused_correctly) / len(refusals), 4)
            if refusals
            else 0.0,
            # 三个数分开：拒错了 / 该拒没拒 / 检索没召回
            "false_refusal": false_refusal,
            "false_accept": false_accept,
            "recall_miss": recall_miss,
            "overall_accuracy": round(correct / total, 4) if total else 0.0,
            "note": (
                "false_refusal 只统计「top1 低于阈值」的条数；"
                "recall_miss 是「没进 top_k」的条数，属于召回问题、不是拒答问题。"
                "overall_accuracy 把三者混在一起，仅作粗略参考"
            ),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Markdown 报告。**先给结论表，再给逐条明细。**"""
    meta = payload["meta"]
    metrics = payload["metrics"]
    lines: list[str] = []

    lines.append("# 检索评测结果")
    lines.append("")
    lines.append(
        f"> 生成时间 {meta['generated_at']} ｜ chunk_chars={meta['chunk_chars']} "
        f"overlap={meta['overlap']} strategy={meta['strategy']} "
        f"embedder={meta['embedder']}"
    )
    lines.append(
        f"> 题目 {metrics['total_questions']} 条"
        f"（可答 {metrics['answerable']} / 拒答 {metrics['refusals']}），"
        f"阈值 {metrics['threshold']}"
    )
    lines.append("")
    lines.append("## 召回")
    lines.append("")
    lines.append("| k | Recall@k | 命中 / 总数 |")
    lines.append("| ---: | ---: | ---: |")
    for key, row in metrics["recall"].items():
        lines.append(
            f"| {key.split('@')[1]} | {row['rate'] * 100:.1f}% | {row['hits']}/{row['total']} |"
        )
    lines.append("")
    lines.append(f"**MRR：{metrics['mrr']:.4f}**")
    lines.append("")

    if metrics["by_type"]:
        lines.append("按题型（多跳天然低于事实型，混看会稀释事实型的水位）")
        lines.append("")
        lines.append("| 题型 | 条数 | Recall@5 | MRR |")
        lines.append("| --- | ---: | ---: | ---: |")
        for kind, row in metrics["by_type"].items():
            lines.append(
                f"| {kind} | {row['total']} | {row['recall@5'] * 100:.1f}% | {row['mrr']:.4f} |"
            )
        lines.append("")

    refusal = metrics["refusal"]
    lines.append("## 拒答")
    lines.append("")
    lines.append(
        f"- 正确拒答：**{refusal['refused_correctly']}/{refusal['refusal_total']}**"
        f"（{refusal['refusal_accuracy'] * 100:.1f}%）"
    )
    lines.append(
        f"- 误拒（top1 低于阈值，**被阈值挡掉的真答案**）：{len(refusal['false_refusal'])} 条"
    )
    lines.append(f"- 误放（该拒却答）：{len(refusal['false_accept'])} 条 {refusal['false_accept']}")
    lines.append(
        f"- 召回未命中（进了候选但排不进 top_k）：**{len(refusal['recall_miss'])} 条** "
        f"—— 这**不是**拒答问题，由上面的 Recall@k 度量"
    )
    lines.append(f"- 合成准确率：{refusal['overall_accuracy'] * 100:.1f}%（{refusal['note']}）")
    lines.append("")

    lines.append("## 逐条")
    lines.append("")
    lines.append("| id | 类型 | 命中位置 | 各锚点位置 | top1 分数 |")
    lines.append("| --- | --- | --- | --- | ---: |")
    for row in payload["per_question"]:
        rank = f"top{row['hit_rank']}" if row["hit_rank"] else "未命中"
        anchors = " / ".join(f"top{r}" if r else "未" for r in row["anchor_ranks"]) or "—"
        score = f"{row['top_score']:.4f}" if row["top_score"] is not None else "—"
        lines.append(f"| {row['id']} | {row['type']} | {rank} | {anchors} | {score} |")
    lines.append("")
    lines.append(
        "> **这套 50 条是调参集**：只能用来选参数，不能用来报"
        "「我们的准确率是 X%」。50 条里 1 条 = 2pp，"
        "而且题面在调参时被反复见过。见 `eval/README.md` §〇。"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索评测：Recall@k / MRR / 拒答准确率")
    parser.add_argument("--questions", type=Path, default=EVAL_FILE)
    parser.add_argument("--chunk-chars", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--embedder", choices=["bge", "hashing"], default="bge")
    parser.add_argument("--out", type=Path, default=None, help="不给就按时间戳命名")
    parser.add_argument(
        "--merged-corpus",
        action="store_true",
        help="用合并正文语料（只适合调 chunk/overlap；**分数分布不等于线上**）",
    )
    parser.add_argument("--no-write", action="store_true", help="只打印，不落盘")
    args = parser.parse_args(argv)

    from app.rag.chunking import ChunkingConfig, build_strategy
    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.settings import get_rag_config

    cfg = get_rag_config()
    chunk_chars = args.chunk_chars if args.chunk_chars is not None else cfg.chunk_chars
    overlap = args.overlap if args.overlap is not None else cfg.overlap_chars
    strategy = args.strategy or cfg.strategy
    threshold = args.threshold if args.threshold is not None else cfg.score_threshold

    items = load_questions(args.questions)
    leak = check_no_leak(items)
    if leak:
        print("题目质量问题（问题照抄了原文，会让纯字面匹配也满分）：", file=sys.stderr)
        print("\n".join(leak), file=sys.stderr)
        return 2

    if args.embedder == "hashing":
        print("⚠ hashing 下召回率无参考价值，仅验证流程", file=sys.stderr)
        embedder: Any = get_embedder("hashing")
    else:
        embedder = CachedEmbedder(get_embedder("bge-small-zh"))

    # **默认用 loader 原样输出**（与线上 `index_file` 的输入一致）。
    # 合并语料（`load_corpus`）只适合调 chunk/overlap 的网格 ——
    # 它的分数分布与线上不同（可答 min 0.480 vs 0.5157，拒答 max 0.6288 vs 0.7958），
    # 拿它定的阈值会在线上误拒。见 `retrieval_harness.load_corpus` 的警示。
    if args.merged_corpus:
        corpus = load_corpus()
        docs_to_chunk = corpus.all_docs
        corpus_desc = f"合并正文（{len(docs_to_chunk)} 段，**不等于线上**）"
    else:
        docs_to_chunk = load_raw_corpus()
        corpus_desc = f"loader 原样（{len(docs_to_chunk)} 段，与线上一致）"

    chunk_cfg = ChunkingConfig(chunk_chars=chunk_chars, overlap_chars=overlap)
    chunks = build_strategy(strategy, cfg=chunk_cfg).chunk(docs_to_chunk, chunk_cfg)
    print(
        f"语料：{corpus_desc}；索引 {len(chunks)} 块"
        f"（chunk={chunk_chars} overlap={overlap} strategy={strategy}），"
        f"题目 {len(items)} 条",
        file=sys.stderr,
    )

    # 锚点反查 —— **失败就整体退出**，不跳过任何一条
    try:
        resolve_all(items, chunks)
    except AnchorResolutionError as exc:
        print(f"锚点解析失败：\n{exc}", file=sys.stderr)
        return 2

    from app.rag.stores.memory import InMemoryVectorStore

    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))

    max_k = max(KS)
    outcomes: list[QueryOutcome] = []
    for item in items:
        vector = embedder.embed_query([item.question])[0]
        hits = store.search(vector, top_k=max_k)
        outcomes.append(evaluate_query(item=item, hits=hits, k=max_k))

    metrics = summarize(outcomes=outcomes, items=items, ks=KS, threshold=threshold)

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "questions_file": str(args.questions.relative_to(PROJECT_ROOT)),
            "chunk_chars": chunk_chars,
            "overlap": overlap,
            "strategy": strategy,
            "embedder": args.embedder,
            "chunks": len(chunks),
            "corpus_docs": len(docs_to_chunk),
            "corpus_kind": "merged" if args.merged_corpus else "raw",
            "ks": list(KS),
        },
        "metrics": metrics,
        "per_question": [
            {
                "id": o.id,
                "type": o.type,
                "hit_rank": o.hit_rank,
                "anchor_ranks": list(o.anchor_ranks),
                "top_score": round(o.top_score, 4) if o.top_score is not None else None,
            }
            for o in outcomes
        ],
    }

    # 控制台摘要（人先看这个）
    print()
    print(
        f"Recall@5 {metrics['recall']['recall@5']['rate'] * 100:.1f}%  "
        f"MRR {metrics['mrr']:.4f}  "
        f"拒答 {metrics['refusal']['refused_correctly']}/{metrics['refusal']['refusal_total']}  "
        f"误拒 {len(metrics['refusal']['false_refusal'])}  "
        f"误放 {len(metrics['refusal']['false_accept'])}"
    )
    for kind, row in metrics["by_type"].items():
        print(
            f"  {kind:<10} n={row['total']:<3} Recall@5 {row['recall@5'] * 100:.1f}%  MRR {row['mrr']:.4f}"
        )

    if args.no_write:
        return 0

    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out.stem if args.out else f"retrieval-{time.strftime('%Y%m%d-%H%M%S')}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"\n落盘：{json_path.relative_to(PROJECT_ROOT)}")
    print(f"      {md_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
