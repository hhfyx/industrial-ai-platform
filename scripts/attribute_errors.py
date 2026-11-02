"""逐条分析错例，按四类归因自动给出**候选判断**，人再复核。

## ⚠ 这个脚本的判据在 W6 周六被重写了（原有的 ② 类标签是错的）

**旧判据的错**：`classify()` 里凡是"不是多跳、不是表格、不是明确断句"的，
一律落到 `② 术语/同义词未命中` —— 那**本质上是默认分支**。

结果是 30 条错例里 15 条（50%）被标成"术语未命中"，而这个标签
**在 W6 周一被数据推翻**：

| 检查项 | 结果 |
| --- | --- |
| 含稀有标识符（`INVALID_INPUT`、`bge-small-zh` 这类） | **0 条** |
| 只共享「模型/输入/上限/文档」这类高频普通词 | 12 条 |
| **连一个公共 token 都没有** | 3 条 |

"术语未命中"这个说法暗示"加个更好的分词器/词典就能解决"，
而 **0 条含稀有标识符**说明那条路对这批题无效 —— 方向从一开始就错了。

W6 周六的实测更直接：把混合检索（BM25 + RRF）接上去，
**这 15 条救回 0 条**。W5 §5.2 预写的判据是"只救回 1~2 条 → 假设错了，
要改归因表"，实测 0 条。

## 新判据：按"字面锚点的稀有度"分类，而不是按"是不是术语"

| 新类别 | 判据（可机判） | 该谁修 |
| --- | --- | --- |
| **① 切分断语义** | gold 块以半句结尾（保留原判据） | 切分器 |
| **③ 多跳需跨块** | 多锚点且有缺失（保留原判据） | 跨块合并 / 两轮检索 |
| **⑤ 长块被挤掉** | gold 块显著长于候选均值，且共享普通词 | 长块降权 / 表格路由 |
| **⑥ 字面零交集** | query 与 gold 块**零公共 token** | 查询改写 / 向量侧 |
| **⑦ 语义区分度不足** | 有公共 token 但都是高频词，排位靠后 | 更强 embedder / 精排 |
| **② 术语未命中** | **仅当含稀有标识符且仍排在后面** | 分词器 / 词典 |

**② 从"默认分支"变成了"稀有情形"** —— 这才符合实际分布。
④（问题模糊）仍然机器判不了，留给人工。

## 为什么这个改动值得

旧标签让整周的改进方向偏了两天：W6 周一做 BM25 是为了救 ② 类，
而 ② 类实际上不存在。**一个占 50% 的错误标签，比一个占 5% 的错误标签贵十倍。**
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import retrieval_harness as rh  # noqa: E402
import run_retrieval_eval as rre  # noqa: E402

DEFAULT_OUT: Path = PROJECT_ROOT / "scripts" / "error_attribution.json"

# 半句结尾的信号：块末不是这些标点，且长度接近预算
SENTENCE_END = "。！？；：…”』）)】` \n"

#: 稀有标识符的判定：含 `_` `-` `.` 的 ASCII 串，或长度 ≥6 的纯英文/数字。
#:
#: **这就是 W6 周一推翻旧标签用的那条判据**，现在固化进脚本。
#: 含这类 token 的 query 才是"BM25 该强"的题；不含的，加词典也没用。
_RARE_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[-._][A-Za-z0-9]+)+$|^[A-Za-z]{6,}$")

#: 极简切词：英文按词、数字整体、**中文按单字**。
#:
#: ## ⚠ 中文那一支不能加 `+`
#:
#: 第一版写的是 `[一-鿿]+`，于是**整句中文被吃成一个 token** ——
#: `"模型的输入长度上限是多少"` 会变成**一个** token，
#: 与任何块都不可能有交集。实测：query 与 gold 块的公共 token 恒为 0。
#:
#: 后果：判据里的"零交集（⑥）"会吞掉所有中文错例，
#: 而"术语未命中（②）"永远不可能触发 —— **判据整体失效**。
#:
#: 中文按单字切是**有意的粗糙**：本项目不需要分词精度（那要 jieba，
#: 而部署精简版没有），只需要"这两个说法有没有共同的字"。
#: 单字匹配对"同义改写"是合理的近似 —— 「首字延迟」vs「TTFT」
#: 共享 0 字（判 ⑥ 零交集，正确）；「模型」vs「模型的」共享「模/型」（判 ⑦，正确）。
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:[-.][A-Za-z0-9_]+)*|[0-9]+|[一-鿿]")

#: 判"是否有字面交集"时要忽略的虚词/单字。
#:
#: 中文按字切之后，几乎任意两句都会共享「的/是/在/个」这类字。
#: 不剔掉它们，"零交集（⑥）"判不出来 —— 实测「首字延迟这个指标的口径」
#: 与「TTFT 指从发起请求……的时间」共享 `{个, 指, 的}`，
#: 而**这三字没有任何区分信息**。
#:
#: 这与 `bm25.tokenize.STOPWORDS` 是同一个道理，但**那份带 jieba 依赖**，
#: 这里要能在无 jieba 环境跑，所以单列一份小的。
_STOPCHARS: frozenset[str] = frozenset(
    "的了是在和与或而及等也都就还把被让给对从到为以于之其这那个些有没不很会能要该着过得地吗呢啊吧"
    "上中下前后里外时个种条次"
)
"""**单字**停用表（只用于本模块的交集判定）。"""


def _tokens(text: str) -> set[str]:
    """切出用于"有没有字面交集"判断的 token。

    规则：英文按词、数字整体、中文按字，**并剔除虚词单字**。
    """
    return {
        token
        for token in (match.group(0).lower() for match in _WORD.finditer(text))
        if token not in _STOPCHARS
    }


def _is_rare(token: str) -> bool:
    return bool(_RARE_TOKEN.match(token))


#: 块尾的 Markdown 装饰性内容。**判"断句"之前必须先剥掉它们。**
#:
#: 实测（W6 周日重写归因时发现）：f-08 / f-10 的 gold 块尾是 `\n\n---`
#: （Markdown 分隔线），f-27 是 `\n\n---`。第一版把这些判成"① 切分断语义"，
#: 于是 ① 类从 2 条虚增到 4 条 —— **其中 2 条是误判**。
#:
#: 分隔线不是句子的一部分，块尾落在分隔线上**恰恰说明切分器工作正常**
#: （它在段落边界切开了）。
_MD_TAIL_DECORATION = re.compile(r"(?:\s*(?:---+|\*\*\*+|___+)\s*)+$")


def looks_truncated(text: str) -> bool:
    """块尾是不是落在半句/半个词上。

    这是"切分断语义"的**直接证据**：如果 gold 块的结尾没有句末标点，
    那它多半是从一句话中间切出来的。

    ## 先剥 Markdown 装饰，再判

    块尾常见 `\\n\\n---` 这种分隔线。它不是句子内容，
    落在分隔线上说明**切在段落边界上**，那是正确行为。
    不剥掉会把"切得好"判成"切断了"。
    """
    stripped = _MD_TAIL_DECORATION.sub("", text.rstrip()).rstrip()
    if not stripped:
        # 整块只有分隔线与空白 —— 这种块没有语义内容，不算"断句"
        return False
    return stripped[-1] not in SENTENCE_END


def classify(
    *,
    item: Any,  # noqa: ANN401
    outcome: Any,  # noqa: ANN401
    gold_chunks: list[Any],
    top50_rank: int | None,
    top1_score: float | None,
    threshold: float,
) -> dict[str, Any]:
    """给一条错例打**候选中**的归因，并附判断依据。

    ## 判据顺序（**顺序本身就是设计**）

    1. 拒答误放 —— 单独一类，与检索无关
    2. ③ 多跳缺锚点 —— 有硬信号（`anchor_ranks` 里有 `None`）
    3. ① 真断句 —— 块尾/块首落在半句上（**表格块先排除**，它不是句子）
    4. **字面交集分析** —— 决定 ⑤/⑥/⑦/② 中的哪一类

    第 4 步是 W6 周六新增的核心。旧版在这里直接给 `② 术语未命中`，
    而那是个**默认分支**，不是判断。
    """
    signals: list[str] = []

    # ---- 拒答题误放：单独一类
    if item.is_refusal:
        return {
            "candidate": "refusal_false_accept",
            "signals": [
                f"该拒却答：top1={top1_score} >= 阈值 {threshold}",
                "拒答判据是分数，而分数不含「能不能回答」这个信息",
            ],
            "auto_confidence": "high",
        }

    # ---- ③ 多跳：锚点有 None
    if item.type == "multi_hop":
        missing = [i for i, rank in enumerate(outcome.anchor_ranks) if rank is None]
        if missing:
            signals.append(f"多跳的 {len(missing)} 个锚点未命中（锚点位置 {outcome.anchor_ranks}）")
            if len(missing) == 1:
                signals.append("只缺一处 → 需要「跨块合并」或两轮检索才能补齐")
            else:
                signals.append("两处都缺 → 不只是跨块问题，可能整体没定位到")
            return {
                "candidate": "③ 多跳需跨块",
                "signals": signals,
                "auto_confidence": "high",
            }

    # ---- **表格块要单独看**：它整块不切是契约，不是"断语义"
    #
    # 第一版把所有"以非句末标点结尾"的 gold 块都判成 ①，结果 7 条里 5 条
    # 是表格（结尾是 `|`）。**表格不以句号结尾是正常的** —— 它不是句子。
    table_chunks = [chunk for chunk in gold_chunks if chunk.metadata.get("block") == "table"]

    # ---- ④ 真断句（表格已排除）
    if not table_chunks:
        truncated = [looks_truncated(chunk.text) for chunk in gold_chunks]
        if any(truncated):
            signals.append(f"gold 块以非句末标点结尾（{truncated}）→ 强信号：切分断了语义")
            return {
                "candidate": "① 切分断语义",
                "signals": signals,
                "auto_confidence": "medium",
            }

    # ---- 以下是 W6 周六新增的**字面交集分析**
    query_tokens = _tokens(item.question)
    gold_tokens: set[str] = set()
    for chunk in gold_chunks:
        gold_tokens |= _tokens(chunk.text)
    shared = query_tokens & gold_tokens
    rare_shared = {token for token in shared if _is_rare(token)}

    signals.append(f"query 有 {len(query_tokens)} 个 token，gold 块有 {len(gold_tokens)} 个")
    signals.append(f"公共 token {len(shared)} 个：{sorted(shared)[:8]}")
    if rare_shared:
        signals.append(f"其中**稀有标识符** {sorted(rare_shared)} → BM25/分词器该强")

    # ⑤ 长块被挤掉：表格块，或显著长于中位
    lengths = sorted(len(chunk.text) for chunk in gold_chunks)
    if table_chunks:
        signals.append(
            f"gold 块是**表格**（{len(table_chunks)} 个，长度 {lengths}）—— 整块不切是契约"
        )
        signals.append("表格块通常很长，会被更短更聚焦的正文块挤掉")
        # **表格块常常同时"零交集"**（实测 8 条表格里有 6 条公共 token = 0）。
        # 两类标签重叠时必须选一个，这里选 ⑤ 而不是 ⑥，理由是**修法不同且更具体**：
        #
        #   ⑤ → 长块降权 / 表格单独路由（针对块结构）
        #   ⑥ → 查询改写 / 换 embedder（针对表达差异）
        #
        # 表格块的根因是"它是表格且长"，不是"用户表述不同" ——
        # 同一条 query 换成正文块就能命中。所以 ⑤ 更接近真因。
        #
        # 但如果表格块**不是**最长的那些候选，那"长"这个解释就不成立，
        # 该看表达差异（⑥）。这里把两种可能都写进 signals，让人复核。
        signals.append("修法不是改切分，而是「长块降权」或「表格单独路由」")
        if top50_rank is None:
            signals.append("⚠ 另：该 query 与表格块公共 token 极少（可能同时是 ⑥ 字面零交集）")
        return {
            "candidate": "⑤ 长块被挤掉",
            "signals": signals,
            "auto_confidence": "high" if top50_rank is None else "medium",
        }
    if max(lengths) >= 400 and top50_rank is None:
        signals.append(f"gold 块很长（最长 {max(lengths)} 字）且不在 top50 → 被短块挤掉")
        return {
            "candidate": "⑤ 长块被挤掉",
            "signals": signals,
            "auto_confidence": "medium",
        }

    # ⑥ 字面零交集：query 与 gold 一个字面 token 都不共享
    if not shared:
        signals.append("**零公共 token** → 字面完全没对上，BM25/词典再强也没用")
        signals.append("这是查询表述与文档表述不一致（同义/换述）")
        return {
            "candidate": "⑥ 字面零交集",
            "signals": signals,
            "auto_confidence": "high",
        }

    # ② 术语未命中：**只有真的含稀有标识符才算**
    if rare_shared:
        signals.append(f"含稀有标识符 {sorted(rare_shared)} 却仍排不上 → 分词/词典或权重问题")
        return {
            "candidate": "② 术语未命中",
            "signals": signals,
            "auto_confidence": "high",
        }

    # ⑦ 语义区分度不足：有公共 token，但全是高频普通词
    signals.append("公共 token 都是高频普通词（无稀有标识符）→ 字面匹配无区分度")
    signals.append("BM25/词典无从发挥：这些词在库里到处都是，IDF 低")
    signals.append("该修的是 embedding 表达力或加精排，不是分词器")
    return {
        "candidate": "⑦ 语义区分度不足",
        "signals": signals,
        "auto_confidence": "medium",
    }


def main() -> int:
    from app.rag.chunking import ChunkingConfig, build_strategy
    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.settings import get_rag_config
    from app.rag.stores.memory import InMemoryVectorStore

    cfg = get_rag_config()
    embedder = CachedEmbedder(get_embedder("bge-small-zh"))
    docs = rh.load_raw_corpus()
    chunk_cfg = ChunkingConfig(chunk_chars=cfg.chunk_chars, overlap_chars=cfg.overlap_chars)
    chunks = build_strategy(cfg.strategy, cfg=chunk_cfg).chunk(docs, chunk_cfg)
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))

    by_id = {chunk.id: chunk for chunk in chunks}
    items = rre.load_questions()
    probe = [
        rre.GoldItem(
            id=item.id,
            type=item.type,
            question=item.question,
            anchors=item.anchors,
            gold_answer=item.gold_answer,
            notes=item.notes,
        )
        for item in items
    ]
    rre.resolve_all(probe, chunks)

    rows: list[dict[str, Any]] = []
    for item in probe:
        vector = embedder.embed_query([item.question])[0]
        hits = store.search(vector, top_k=5)
        outcome = rre.evaluate_query(item=item, hits=hits, k=5)

        if item.is_refusal:
            if outcome.top_score is not None and outcome.top_score < cfg.score_threshold:
                continue
        elif outcome.hit:
            continue

        # gold 在 top50 里的最好位置
        deeper = store.search(vector, top_k=50)
        rank50: int | None = None
        for rank, hit in enumerate(deeper, start=1):
            if hit.chunk.id in item.gold_chunk_ids:
                rank50 = rank
                break

        gold_chunks = [by_id[cid] for cid in item.gold_chunk_ids if cid in by_id]
        verdict = classify(
            item=item,
            outcome=outcome,
            gold_chunks=gold_chunks,
            top50_rank=rank50,
            top1_score=outcome.top_score,
            threshold=cfg.score_threshold,
        )
        rows.append(
            {
                "id": item.id,
                "type": item.type,
                "question": item.question,
                "gold_answer": item.gold_answer,
                "anchor_ranks": list(outcome.anchor_ranks),
                "top1_score": round(outcome.top_score, 4) if outcome.top_score else None,
                "gold_rank_in_top50": rank50,
                "gold_chunk_lengths": [len(chunk.text) for chunk in gold_chunks],
                "gold_chunk_tails": [chunk.text.rstrip()[-12:] for chunk in gold_chunks],
                "top5": [
                    {
                        "score": round(hit.score, 4),
                        "source": hit.chunk.source,
                        "head": hit.chunk.text[:60].replace("\n", " "),
                    }
                    for hit in hits
                ],
                **verdict,
            }
        )

    from collections import Counter

    print(f"错例 {len(rows)} 条")
    print("自动候选分布：", dict(Counter(row["candidate"] for row in rows)))
    print()
    for row in rows:
        print(
            f"── {row['id']}（{row['type']}） 得分 {row['top1_score']}  "
            f"gold 在 top50 第 {row['gold_rank_in_top50']}"
        )
        print(f"   候选：{row['candidate']}（置信 {row['auto_confidence']}）")
        for signal in row["signals"]:
            print(f"   · {signal}")

    DEFAULT_OUT.write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n落盘：{DEFAULT_OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
