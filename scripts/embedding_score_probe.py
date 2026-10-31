"""分数分布探针：证明"拒答阈值必须挂在 embedder 上"。

**这个脚本只做一件事**：取几组"真正相关"和"确定无关"的问答对，
用不同 embedder 算余弦，看两组分数有没有**可分界线**。

为什么要它：W4 周五实现拒答时，我先写了一个全局 `DEFAULT_MIN_SCORE = 0.35`。
结果 `tests/test_rag_skeleton.py` 里两条用 `HashingEmbedder` 的端到端测试挂了 ——
正确命中的答案（分数 0.247）被误判成"未找到"并拒答。

那次失败不是测试写错了，是**阈值不该是全局常量**。这个脚本把理由量化出来。

## 读法

- 如果 `min(相关) > max(无关)` → **可分**，阈值取两者之间
- 如果 `min(相关) < max(无关)` → **分布重叠**，任何单一阈值都会同时误杀和漏放

运行：

    uv run python -X utf8 scripts/embedding_score_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# "真正相关"的问答对：问题与原文是**同一件事**，用词可以不同。
RELEVANT: list[tuple[str, str]] = [
    ("违约金怎么算", "第六条 违约责任：乙方逾期交付的，按日万分之五支付违约金。"),
    ("什么时候付钱", "第五条 付款计划：验收合格后十五个工作日内付款。"),
    ("质保多久", "第十三条 质保期：系统验收合格之日起 12 个月。"),
    ("软件著作权归谁", "第八条 知识产权：本项目产生的软件著作权归甲方所有。"),
]

# "确定无关"的问答对：问题与原文**没有任何语义关系**，
# 但刻意保留"字面上可能有零星共同字"的情况（更难、更接近真实）。
IRRELEVANT: list[tuple[str, str]] = [
    ("违约金怎么算", "附件二 培训计划：为运维人员提供 3 天培训，含上机实操。"),
    ("违约金怎么算", "第十二条 交付地点：杭州市余杭区文一西路 969 号。"),
    ("什么时候付钱", "第九条 保密义务：保密期三年。"),
    ("质保多久", "第一条 服务范围：工业设备预测性维护系统的开发与部署。"),
]


def score_all(embedder) -> dict[str, list[float]]:
    """算两组分数。**同一 embedder 内可比，跨 embedder 不可比** —— 这正是结论。"""
    from app.rag.embedding.vector import cosine

    def one(question: str, passage: str) -> float:
        query_vector = embedder.embed_query([question])[0]
        passage_vector = embedder.embed([passage])[0]
        return cosine(query_vector, passage_vector)

    return {
        "relevant": [one(q, p) for q, p in RELEVANT],
        "irrelevant": [one(q, p) for q, p in IRRELEVANT],
    }


def analyze(scores: dict[str, list[float]]) -> dict:
    rel = scores["relevant"]
    irr = scores["irrelevant"]
    separable = min(rel) > max(irr)
    return {
        "relevant_min": round(min(rel), 4),
        "relevant_max": round(max(rel), 4),
        "irrelevant_min": round(min(irr), 4),
        "irrelevant_max": round(max(irr), 4),
        "separable": separable,
        # 可分时给出中点作为建议阈值；不可分时没有建议
        "suggested_threshold": (round((min(rel) + max(irr)) / 2, 4) if separable else None),
    }


def main() -> int:
    try:
        from app.rag.embedding import HashingEmbedder, get_embedder
    except ImportError as exc:  # pragma: no cover
        print(f"缺少依赖：{exc}", file=sys.stderr)
        return 1

    candidates = [
        ("hashing-64", HashingEmbedder(dimension=64)),
        ("hashing-256", HashingEmbedder(dimension=256)),
    ]
    # bge 需要 torch + 模型；缺失时跳过而不是让整个脚本失败
    try:
        candidates.append(("bge-small-zh", get_embedder("bge-small-zh")))
    except Exception as exc:  # noqa: BLE001
        print(f"（跳过 bge：{type(exc).__name__}: {exc}）", file=sys.stderr)

    payload: dict[str, dict] = {}
    print(
        f"{'embedder':16} {'相关(min~max)':>20} {'无关(min~max)':>20} {'可分?':>6} {'建议阈值':>10}"
    )
    for name, embedder in candidates:
        try:
            scores = score_all(embedder)
        except Exception as exc:  # noqa: BLE001 - 模型加载失败时跳过
            print(f"{name:16} 跳过（{type(exc).__name__}: {exc}）", file=sys.stderr)
            continue
        info = analyze(scores)
        info["scores"] = {k: [round(x, 4) for x in v] for k, v in scores.items()}
        info["declared_min_score"] = embedder.recommended_min_score
        payload[name] = info

        mark = "✅" if info["separable"] else "❌ 重叠"
        threshold = (
            f"{info['suggested_threshold']:.3f}"
            if info["suggested_threshold"] is not None
            else "无（分布重叠）"
        )
        print(
            f"{name:16} "
            f"{info['relevant_min']:>8.3f}~{info['relevant_max']:<8.3f} "
            f"{info['irrelevant_min']:>8.3f}~{info['irrelevant_max']:<8.3f} "
            f"{mark:>6} {threshold:>10}"
        )

    print("\n结论：")
    for name, info in payload.items():
        if info["separable"]:
            print(
                f"  {name}: 相关最低 {info['relevant_min']} > 无关最高 "
                f"{info['irrelevant_max']}，可分。声明值 {info['declared_min_score']}"
            )
        else:
            print(
                f"  {name}: 相关最低 {info['relevant_min']} ≤ 无关最高 "
                f"{info['irrelevant_max']}，**分布重叠** → 任何单一阈值都会误判。"
                f"声明值 {info['declared_min_score']}（0 表示关掉闸门）"
            )

    print("\n注意：样本只有 4+4 条，这里的阈值只是**量级参考**，不是调好的参数。")
    print("      正确的做法是按 embedder 跑标注集、在 ROC 上取工作点（W5 有评测集后做）。")

    out = PROJECT_ROOT / "scripts" / "embedding_score_probe.json"
    out.write_text(
        json.dumps(
            {
                "meta": {
                    "relevant_pairs": len(RELEVANT),
                    "irrelevant_pairs": len(IRRELEVANT),
                    "note": "样本小，阈值只作量级参考；跨 embedder 的分数不可比",
                },
                "embedders": payload,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n已写入 {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
