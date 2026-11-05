"""分层抽样：从 120 条抽出 20 条全链路小样本集（周四调管线用）。

分层维度（每层配额写死，总计 20）：
- fact 命中预期高/低不能由我们预知（还没跑），所以按**题型 × 文档规模**分层：
  - fact 大文档（prompt-ab/deploy-notes/why-langchain/prompt-templates）4 条
  - fact 小文档（streamlit-ui/structured-output/container 等）4 条
  - multi_hop 6 条（多跳是召回最难的一类，必须占三成）
  - refusal 6 条（含最高分近邻 3 条——它们最容易暴露阈值问题）

固定种子保证任何人重跑抽到同一批（answer_quality.py 同款纪律）。
产出 eval/questions-slice20.jsonl + 抽样清单落盘。
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "eval" / "questions-120.jsonl"
DST = PROJECT_ROOT / "eval" / "questions-slice20.jsonl"
LIST_MD = PROJECT_ROOT / "eval" / "results" / "slice20-roster.md"

SEED = 20261202

BIG_DOCS = {"prompt-ab.md", "deploy-notes.md", "why-i-use-langchain.md", "prompt-templates.md"}
# 拒答题里向量 top1 最高的 3 条（最容易触发误放，调管线时最先看到问题）
HIGH_SCORE_REFUSALS = {"r-01", "r-16", "r-18"}


def main() -> None:
    items = [
        json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rng = random.Random(SEED)

    facts_big = [i for i in items if i["type"] == "fact" and i["anchors"][0]["source"] in BIG_DOCS]
    facts_small = [
        i for i in items if i["type"] == "fact" and i["anchors"][0]["source"] not in BIG_DOCS
    ]
    hops = [i for i in items if i["type"] == "multi_hop"]
    refs = [i for i in items if i["type"] == "refusal"]

    picks: list[dict] = []
    picks += rng.sample(facts_big, 4)
    picks += rng.sample(facts_small, 4)
    picks += rng.sample(hops, 6)
    # 拒答：3 条最高分近邻 + 3 条随机（覆盖"阈值边缘"与"轻松拒绝"两种）
    refs_high = [i for i in refs if i["id"] in HIGH_SCORE_REFUSALS]
    refs_rest = [i for i in refs if i["id"] not in HIGH_SCORE_REFUSALS]
    picks += refs_high + rng.sample(refs_rest, 3)

    assert len(picks) == 20, f"抽了 {len(picks)} 条"
    assert len({p["id"] for p in picks}) == 20

    picks.sort(key=lambda i: i["id"])
    DST.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in picks), encoding="utf-8"
    )

    kinds = Counter(p["type"] for p in picks)
    docs = sorted({p["anchors"][0]["source"] for p in picks if p["type"] != "refusal"})
    lines = [
        "# 分层抽样 20 条（周四全链路调管线用）",
        "",
        f"> 种子 {SEED}，重跑必得同批。来源 questions-120.jsonl（120 条）。",
        "",
        "| 层 | 配额 | 实际 |",
        "| --- | ---: | ---: |",
        "| fact 大文档 | 4 | "
        + str(
            sum(1 for p in picks if p["type"] == "fact" and p["anchors"][0]["source"] in BIG_DOCS)
        )
        + " |",
        "| fact 小文档 | 4 | "
        + str(
            sum(
                1
                for p in picks
                if p["type"] == "fact" and p["anchors"][0]["source"] not in BIG_DOCS
            )
        )
        + " |",
        "| multi_hop | 6 | " + str(kinds.get("multi_hop", 0)) + " |",
        "| refusal（含 3 条最高分近邻） | 6 | " + str(kinds.get("refusal", 0)) + " |",
        "",
        "事实题覆盖文档：" + "、".join(docs),
        "",
        "## 拒答特抽（阈值边缘，最先暴露误放）",
        "",
    ]
    lines += [f"- {p['id']}: {p['question']}" for p in picks if p["type"] == "refusal"]
    lines += ["", "## 完整清单", ""]
    lines += [f"- {p['id']}（{p['type']}）{p['question']}" for p in picks]
    LIST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"✓ 20 条 → {DST.relative_to(PROJECT_ROOT)}")
    print(f"  清单 → {LIST_MD.relative_to(PROJECT_ROOT)}")
    print(f"  题型分布：{dict(kinds)}")


if __name__ == "__main__":
    main()
