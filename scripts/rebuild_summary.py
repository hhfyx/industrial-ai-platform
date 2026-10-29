"""从已有 CSV 重算 summary（不重新调 API）。

为什么需要这个：改了 `summarize()` 的口径（加分"全量"口径，失败按 0 分），
已有 CSV 里已经存了逐条结果，重算即可 —— **不该为了改统计口径重花一次 API 的钱**。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ab_test as AB  # noqa: E402

RES = Path(__file__).resolve().parent / "ab_results"


def rebuild(csv_path: Path) -> None:
    rows = []
    with csv_path.open(encoding="utf-8-sig") as f:
        for d in csv.DictReader(f):
            rows.append(
                AB.Row(
                    set_name=d["set_name"],
                    arm=d["arm"],
                    item_id=d["item_id"],
                    ok=d["ok"] == "True",
                    score=float(d["score"] or 0),
                    judge_detail=d["judge_detail"],
                    prompt_tokens=int(d["prompt_tokens"] or 0),
                    completion_tokens=int(d["completion_tokens"] or 0),
                    latency_ms=int(d["latency_ms"] or 0),
                    cost_cny=float(d["cost_cny"] or 0),
                    raw_output=d["raw_output"],
                    error=d["error"],
                )
            )
    set_name = rows[0].set_name
    arm_a = csv_path.stem.split("_vs_")[0].split("_", 1)[1]
    arm_b = csv_path.stem.split("_vs_")[1].split("_")[0]

    s = AB.summarize(rows, set_name, arm_a, arm_b)
    out = csv_path.with_suffix(".summary.json")
    out.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    d = s["delta"]
    print(
        f"{csv_path.name:<48} A={s['arms']['A']['mean_score_all']:.4f} "
        f"B={s['arms']['B']['mean_score_all']:.4f} Δ={d['mean_score']:+.4f} "
        f"err(A/B)={s['arms']['A']['errors']}/{s['arms']['B']['errors']}"
    )


def main() -> None:
    for p in sorted(RES.glob("*_full*.csv")):
        rebuild(p)


if __name__ == "__main__":
    main()
