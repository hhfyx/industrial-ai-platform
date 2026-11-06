"""上下文去重（W8 周二）：chunk_id 精确去重 + overlap 近重复合并，**带计数**。

## 为什么在生成层去重，而不是检索层

`retrieve()` 是 W6 冻结的契约（W9 还要包成 Agent 工具）。
"喂给模型的上下文里不要有重复"是**生成层**的职责：检索层返回的 top_k
是"库里有这些块"的事实，展示与组装是另一件事。
这正好也是周一改写层的同一立场：检索段的输出契约不动，加工放调用侧。

## 两级去重

| 级 | 判据 | 典型来源 | 保留谁 |
| --- | --- | --- | --- |
| ① 精确 | `chunk_id` 相同 | 同一文件重复索引、调用方拼列表手滑 | **分数高者** |
| ② 近重复 | 文本相似度 ≥ 阈值 | **overlap**：相邻块共享尾巴，`adjacent_pair_rate` 实测 5.5%~11% | **分数高者** |

两级都保留**分数高**的那条：分数是检索层对"和 query 像不像"的度量，
重复出现时，低分条只是高分条的一个 poorer 副本 —— 留它只会占上下文预算。

## 为什么近重复用 difflib 而不是 embedding

去重发生在 prompt 组装时刻，调用方手里只有 `chunks + scores`；
重新编码等于多一次 embed 调用（bge 下 5 块 ≈ 数百 ms）。
`SequenceMatcher.ratio()` 是字符级最长匹配比例，对"overlap 造成的
头部/尾部共享"足够敏感（同文档相邻块 ratio 实测 0.3~0.6，
纯 overlap 重复段 ratio ≥ 0.7），且纯 CPU、确定可复现。
**它不判语义**——两个语义相同措辞不同的块它判不出，那是 embedding 的事，
但本层的敌人是 overlap 的**字面**重复，不是语义重复。

## 计数为什么是产物的一部分

"去重了几条"是本周验收的第一条。没有计数，去重就只是一个
"看起来上下文干净了"的感觉 —— 计数让 `adjacent_pair_rate`（检索层指标）
与"prompt 里实际重复率"可以对账。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.rag.schema import Chunk

logger = logging.getLogger(__name__)

# 近重复判定阈值。0.75 的依据：
# - overlap=50 / chunk=500 → 相邻块共享约 10% 正文，ratio 实测 0.10~0.15 —— 不会误伤
# - 真正的近重复（半块以上共享）ratio ≥ 0.7
# - 表格/代码块的重复行也会推高 ratio，误伤时丢的也只是低分副本
# 调参要跑 `scripts/multi_turn_eval.py` 或 slice20 对照，不许拍脑袋改。
NEAR_DUP_THRESHOLD: float = 0.75


@dataclass(frozen=True, slots=True)
class DedupeReport:
    """去重结果 + 计数。**计数是产物，不是日志**——验收要的是数字。"""

    chunks: tuple[Chunk, ...]
    """去重后的块，保持传入顺序（分数序）。"""

    scores: tuple[float, ...]
    """与 `chunks` 一一对应的分数（保留条目的原分数）。"""

    exact_removed: int = 0
    """① 级删除条数（chunk_id 重复）。"""

    near_removed: int = 0
    """② 级删除条数（近重复文本）。"""

    @property
    def removed(self) -> int:
        return self.exact_removed + self.near_removed

    def as_dict(self) -> dict[str, int]:
        return {
            "input": len(self.chunks) + self.removed,
            "exact_removed": self.exact_removed,
            "near_removed": self.near_removed,
            "kept": len(self.chunks),
        }


def _text_ratio(a: str, b: str) -> float:
    """字符级相似度。空串与任何东西都是 0（两个空块不构成"重复"）。

    **`autojunk=False` 是必须的，不是偏好**：默认的 autojunk 启发式会把
    "出现频率 >1% 的字符"当垃圾（序列 ≥200 字符时触发）。中文块动辄
    数百字且高频字（的、是、按）远超 1% —— 实测两段 195 字共享的文本
    被判 ratio=0.0，近重复检测整个失效。关掉后回到 0.9+。
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def dedupe_chunks(
    chunks: Sequence[Chunk],
    scores: Sequence[float] | None = None,
    *,
    threshold: float = NEAR_DUP_THRESHOLD,
) -> DedupeReport:
    """对检索结果做两级去重。**保持分数序**（传入顺序），分数相同保先出现的。

    `scores` 为 None 时所有块视为同分（保先出现）—— 给"只测去重逻辑"的
    调用方留的口子；生产路径必须传（与 `generate_answer` 的 scores 口径一致）。

    ## 为什么 O(n²) 可以接受

    top_k ≤ 10（`DEFAULT_TOP_K=5`），两两比较最多 45 对，
    每对一次 `SequenceMatcher`（微秒级）。任何"更聪明"的数据结构
    （minhash / simhash）在这个规模上都是过度设计。
    """
    if scores is not None and len(scores) != len(chunks):
        raise ValueError(f"chunks({len(chunks)}) 与 scores({len(scores)}) 数量不一致")
    if scores is None:
        scores = [0.0] * len(chunks)

    kept: list[Chunk] = []
    kept_scores: list[float] = []
    exact_removed = 0
    near_removed = 0

    for chunk, score in zip(chunks, scores, strict=True):
        # ---- ① 精确去重：chunk_id ----
        # 内容派生 id 相同 = 同一块正文（可能同文件重复索引，也可能调用方拼重）。
        existing_at = next((i for i, c in enumerate(kept) if c.id == chunk.id), None)
        if existing_at is not None:
            exact_removed += 1
            if score > kept_scores[existing_at]:
                # 分数更高 → 用新条目替换旧条目（保持位置不变，序仍成立）。
                kept[existing_at] = chunk
                kept_scores[existing_at] = score
            continue

        # ---- ② 近重复：与任一已保留块文本相似 ----
        # 只和**已保留**的比较（被删掉的低分副本不用再比）。
        near_hit = any(_text_ratio(chunk.text, c.text) >= threshold for c in kept)
        if near_hit:
            near_removed += 1
            # ② 级不做"高分替换"：近重复条目 id 不同（id 是内容哈希），
            # 但文本几乎一样 —— 换入不会更准（同一段文字），只添乱。
            continue

        kept.append(chunk)
        kept_scores.append(score)

    if exact_removed or near_removed:
        logger.info(
            "上下文去重：精确 %d 条、近重复 %d 条（%d → %d）",
            exact_removed,
            near_removed,
            len(chunks),
            len(kept),
        )
    return DedupeReport(
        chunks=tuple(kept),
        scores=tuple(kept_scores),
        exact_removed=exact_removed,
        near_removed=near_removed,
    )


__all__ = ["DedupeReport", "NEAR_DUP_THRESHOLD", "dedupe_chunks"]
