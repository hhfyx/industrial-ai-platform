"""错误分布表（W11 周一 20:50–21:15 的验收：能出分布）。

## 读什么

两种输入，同一个表格口径：

1. **轨迹 JSON**（`qa_runs_*/trajectories/*.json` / `--traj`）——
   W9/W10/W11 的评测产物，`steps[].actions[]` 里每条调用都有
   `{tool, ok, kind, ms}`；W11 起还有 `{error_type, retryable, latency_ms}`。
2. **调用流水**（`--jsonl`）—— 线上/联调时按行落的调用记录。

两者字段不同（老轨迹没有 error_type），所以这里做**归一化**：
缺 `error_type` 的按 `kind` 推、缺 `retryable` 的按分类表补。
不这么做的话，跨周对比会得出"W11 错误率下降"这种假结论 —— 实际只是
老轨迹没有那个字段。

## 出什么

- **五类分布**：param / permission / timeout / empty / internal（+ duplicate）。
  按 category 而不是按 error_type 数 —— category 是"该做什么"，
  它才是可靠性设计的对象；error_type 太细，看表格时会被细节淹没。
- **可重试率**：失败调用里 retryable=True 占多少。这个数字直接回答
  "我们有多少失败是模型改一下就能自愈的"。
- **按工具**：哪个工具在漏。4 个工具 × 5 类，空着的格子也是信息
  （说明那类的路径还没被测到）。
- **延迟分位**：失败调用的 p50/p95 延迟 —— 超时类错误该落在这里。

## 怎么用

    python -X utf8 scripts/error_stats.py --traj qa_runs_w11
    python -X utf8 scripts/error_stats.py --jsonl reports/tool_calls.jsonl
    python -X utf8 scripts/error_stats.py --traj qa_runs_w10 --json   # 机器可读

**口径纪律**（W10 的教训）：表头必须打出数据来源与样本量。
没有样本量的百分比是"看起来像结论的数字"。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.llm.error_contract import (  # noqa: E402
    ERROR_SPECS,
    category_of,
    is_retryable,
    normalize_error_type,
)

#: 五类 + duplicate 的固定展示顺序。空类也打出来（0 行本身是信息）。
CATEGORY_ORDER: Final[tuple[str, ...]] = (
    "param",
    "permission",
    "timeout",
    "empty",
    "internal",
    "duplicate",
)

CATEGORY_LABELS: Final[dict[str, str]] = {
    "param": "参数错",
    "permission": "权限错",
    "timeout": "超时",
    "empty": "空结果",
    "internal": "内部异常",
    "duplicate": "重复拦截",
}


# ============================================================ 读入 + 归一化


def normalize_call(raw: dict[str, Any]) -> dict[str, Any]:
    """把一条调用记录归一化成统计口径。

    兼容三代形状：
    - W11：`{tool, ok, error_type, retryable, latency_ms}`
    - W9/W10：`{tool, ok, kind, ms}`
    - 线上流水：可能只有 `{tool, ok, error}`（无 kind）

    ## 老轨迹的诚实标注（这条很重要）

    W9/W10 的 action 里 `kind` 记的是 `wrap_observation` 的包装结果，
    而当时的失败信封没有 `error_type` 键 —— 于是**所有失败都被写成
    `kind="error"`**，真实分类已经丢失。硬按 `normalize_error_type("error")`
    落成 internal，会把 W10 里那次"查无批次"（真实是 not_found/空结果）
    误算成内部异常，跨周对比就会得出"W11 内部异常变少了"这种假结论。

    所以这里给这类记录打 `legacy_unclassified=True`，统计时单独计数，
    并在报告里显式提示"这些失败的五类归属不可信"。**宁可说不确定，
    不要给一个看起来确定的错数。**
    """
    ok = bool(raw.get("ok", False))
    latency = raw.get("latency_ms", raw.get("ms", raw.get("latency")))
    latency_ms = int(latency) if isinstance(latency, (int, float)) else None
    tool = raw.get("tool") or "?"
    if ok:
        return {
            "tool": tool,
            "ok": True,
            "error_type": None,
            "category": None,
            "retryable": None,
            "latency_ms": latency_ms,
            "legacy_unclassified": False,
        }
    raw_etype = raw.get("error_type")
    raw_kind = raw.get("kind")
    # "error" 是老信封的占位值，不带分类信息
    legacy = raw_etype is None and (raw_kind is None or raw_kind == "error")
    error_type = raw_etype or normalize_error_type(raw_kind)
    retryable = raw.get("retryable")
    if retryable is None:
        retryable = is_retryable(error_type)
    return {
        "tool": tool,
        "ok": False,
        "error_type": error_type,
        "category": category_of(error_type),
        "retryable": bool(retryable),
        "latency_ms": latency_ms,
        "legacy_unclassified": legacy,
    }


def iter_trajectory_calls(path: Path) -> list[dict[str, Any]]:
    """从轨迹文件（或目录下全部 *.json）里抠出所有工具调用。

    **目录要递归**：产物布局是 `qa_runs_w10/trajectories/*.json`，
    只扫一层会得到"0 次调用"——而空表看起来像"零错误"，这是最坏的假阴性。
    所以这里用 rglob，并在样本量为 0 时显式告警（见 render）。

    只统计 `executed=True` 的 —— 重复熔断拦下的调用没真正跑，
    把它算进"错误率"会让工具显得比实际更糟。（缺 executed 字段的
    老轨迹按已执行算：W9/W10 没这个字段，但那时也没有未执行的记录。）
    """
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    calls: list[dict[str, Any]] = []
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ⚠ 跳过 {file.name}：{exc}", file=sys.stderr)
            continue
        if not isinstance(data, dict) or "steps" not in data:
            continue  # 不是轨迹（比如 baseline 之外的 json 配置），静默跳过
        for step in data.get("steps", []):
            for action in step.get("actions", []):
                if action.get("executed") is False:
                    continue
                calls.append({**action, "_source": file.stem})
    return calls


def iter_jsonl_calls(path: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            calls.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return calls


# ============================================================ 统计


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "—"


def _percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def summarize(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """核心统计。返回可 JSON 化的 dict（`--json` 直接 dump）。"""
    normalized = [normalize_call(c) for c in calls]
    total = len(normalized)
    failures = [c for c in normalized if not c["ok"]]

    by_category: dict[str, int] = dict.fromkeys(CATEGORY_ORDER, 0)
    by_error_type: Counter[str] = Counter()
    by_tool: dict[str, Counter[str]] = defaultdict(Counter)
    legacy_count = 0
    for c in normalized:
        by_tool[c["tool"]]["total"] += 1
        if c["ok"]:
            continue
        by_category[c["category"]] = by_category.get(c["category"], 0) + 1
        by_error_type[c["error_type"]] += 1
        by_tool[c["tool"]][c["category"]] += 1
        by_tool[c["tool"]]["failed"] += 1
        if c["legacy_unclassified"]:
            legacy_count += 1
            by_tool[c["tool"]]["legacy_unclassified"] += 1

    retryable_failures = [c for c in failures if c["retryable"]]
    latencies = [c["latency_ms"] for c in normalized if c["latency_ms"] is not None]
    fail_latencies = [c["latency_ms"] for c in failures if c["latency_ms"] is not None]

    return {
        "total_calls": total,
        "failed_calls": len(failures),
        "error_rate": (len(failures) / total) if total else None,
        "legacy_unclassified_failures": legacy_count,
        "retryable_failures": len(retryable_failures),
        "retryable_rate_of_failures": (
            len(retryable_failures) / len(failures) if failures else None
        ),
        "by_category": by_category,
        "by_error_type": dict(by_error_type.most_common()),
        "by_tool": {tool: dict(counter) for tool, counter in sorted(by_tool.items())},
        "latency": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
        },
        "fail_latency": {
            "p50": _percentile(fail_latencies, 0.50),
            "p95": _percentile(fail_latencies, 0.95),
        },
    }


# ============================================================ 渲染


def render(summary: dict[str, Any], *, source: str) -> str:
    lines: list[str] = []
    total = summary["total_calls"]
    failed = summary["failed_calls"]
    lines.append("=" * 78)
    lines.append("工具错误分布（W11 周一 · 五类口径）")
    lines.append("=" * 78)
    lines.append(f"  数据来源 : {source}")
    lines.append(f"  样本量   : {total} 次调用（失败 {failed} 次）")
    if total == 0:
        lines.append("  ⚠ 没有任何调用记录 —— 空表不是「零错误」，是「没测」。")
        return "\n".join(lines)
    lines.append(f"  错误率   : {_pct(failed, total)}（失败/全部调用）")
    if failed:
        lines.append(
            f"  可重试率 : {_pct(summary['retryable_failures'], failed)}"
            f"（{summary['retryable_failures']}/{failed} 次失败可由模型改策略自愈）"
        )
    else:
        lines.append("  可重试率 : —（没有失败调用）")
    legacy = summary.get("legacy_unclassified_failures", 0)
    if legacy:
        lines.append(
            f"  ⚠ 其中 {legacy} 次失败来自**旧轨迹**（没有 error_type 字段，kind 只有"
            ' "error"）：它们的五类归属是**推测**，不能用于跨周对比。'
        )
    lines.append("")

    lines.append("一、五类分布（按「该做什么」分，不按 error_type 分）")
    lines.append(f"  {'类':<10}{'含义':<10}{'次数':>6}{'占失败':>10}   恢复动作")
    lines.append("  " + "-" * 74)
    for cat in CATEGORY_ORDER:
        count = summary["by_category"].get(cat, 0)
        if cat == "duplicate":
            recovery = "停止并报告（效果已发生）"
        else:
            spec = next((s for s in ERROR_SPECS.values() if s.category == cat), None)
            recovery = spec.recovery if spec else "—"
        lines.append(
            f"  {cat:<10}{CATEGORY_LABELS[cat]:<10}{count:>6}{_pct(count, failed):>10}   {recovery}"
        )
    lines.append("")

    if summary["by_error_type"]:
        lines.append("二、error_type 明细")
        for etype, count in summary["by_error_type"].items():
            retryable = "可重试" if is_retryable(etype) else "不可重试"
            lines.append(f"  {etype:<16}{count:>4} 次   {retryable}")
        lines.append("")

    lines.append("三、按工具（哪个工具在漏；空格子 = 那类路径还没被踩到）")
    header = f"  {'工具':<24}{'调用':>6}{'失败':>6}" + "".join(
        f"{CATEGORY_LABELS[c][:4]:>8}" for c in CATEGORY_ORDER
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for tool, counter in summary["by_tool"].items():
        row = f"  {tool:<24}{counter.get('total', 0):>6}{counter.get('failed', 0):>6}"
        row += "".join(f"{counter.get(c, 0):>8}" for c in CATEGORY_ORDER)
        lines.append(row)
    lines.append("")

    lat = summary["latency"]
    fail_lat = summary["fail_latency"]
    lines.append("四、延迟（毫秒）")
    lines.append(f"  全部调用 : p50={lat['p50']}  p95={lat['p95']}  mean={lat['mean']}")
    lines.append(f"  失败调用 : p50={fail_lat['p50']}  p95={fail_lat['p95']}")
    lines.append("")
    lines.append("  读表提示：timeout 类的延迟应当明显高于 param 类 —— 如果两者")
    lines.append("  一样，说明超时没有在工具层拦住，而是被外层兜底了。")
    return "\n".join(lines)


# ============================================================ CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="工具错误分布表（五类口径）")
    parser.add_argument("--traj", type=Path, help="轨迹目录（qa_runs_*）或单个轨迹 JSON")
    parser.add_argument("--jsonl", type=Path, help="调用流水（每行一个 JSON）")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args(argv)

    if not args.traj and not args.jsonl:
        parser.error("至少给一个输入：--traj 或 --jsonl")

    calls: list[dict[str, Any]] = []
    sources: list[str] = []
    if args.traj:
        found = iter_trajectory_calls(args.traj)
        calls.extend(found)
        sources.append(f"traj:{_rel(args.traj)}")
    if args.jsonl:
        calls.extend(iter_jsonl_calls(args.jsonl))
        sources.append(f"jsonl:{_rel(args.jsonl)}")

    summary = summarize(calls)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render(summary, source=" + ".join(sources)))
    return 0


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
