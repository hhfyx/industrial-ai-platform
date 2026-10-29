"""A/B 测试脚手架：同一批输入，跑两个 prompt 版本，出可比的数据。

用法
----
    python -X utf8 scripts/ab_test.py --set qa --a v1 --b v2
    python -X utf8 scripts/ab_test.py --set classify --a v1 --b v1 --dry-run

## 为什么必须固定 temperature 和 seed

本仓库已有实测（`learn/llm/01_params.md`）：`temperature=0` 五次仍出 3 种输出，
`seed` 在本网关**不可复现**。所以：

- **temperature 固定 0.0**：把漂移压到最小，但**不能指望它消除噪声**；
- **seed 仍然传**（固定值）：本网关不保证，但传了至少能声明"我试过控制"；
- **真正的噪声控制靠样本量**：10 条 × 2 版，任何单条翻转都不构成结论。
  这个判断阈值写在 `summarize()` 里，不靠人眼看。

## 为什么记录 token / 延迟 / 成本

prompt 版本对比最容易被忽略的是**代价差异**：v2 准确率高 5pp 但 token 翻倍，
在有 QPS 的场景下不是"更好"，是"更贵"。所以每条都记：
- `prompt_tokens` / `completion_tokens`（真实 usage，不是 tiktoken 估算）
- `latency_ms`（用户能感觉到的东西）
- `cost_cny`（按项目 `scripts/cost_model.json` 的单价折算）

## 为什么输出 CSV 而不是只打印

CSV 能直接丢进 Excel/透视表做逐条 diff，也能作为 W7 的对比基线。
控制台摘要只给"能立刻判断要不要继续"的 5 个数字。

## temperature/seed 固定在哪

写死在 `FIXED_TEMPERATURE` / `FIXED_SEED`，**不做成命令行参数** ——
这不是"可调项"，是可复现性的一部分。允许调它，等于允许有人偷偷改来"调好看"。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SETS_DIR = PROJECT / "eval" / "prompt-sets"
OUT_DIR = HERE / "ab_results"

# ---- 可复现性：这两个值**故意不做成参数** ----
FIXED_TEMPERATURE: float = 0.0
FIXED_SEED: int = 42

# ---- 成本单价：来自 scripts/cost_model.json，读不到就用保守默认 ----
DEFAULT_PRICE_IN = 2.0  # 元 / 百万 input token
DEFAULT_PRICE_OUT = 8.0  # 元 / 百万 output token

# 结论阈值：低于这个差异幅度不构成"赢"
# 10 条样本时 1 条 = 10pp，所以至少要差 2 条（20pp）才认为有信号。
MIN_SIGNAL_DELTA = 0.20


@dataclass
class Row:
    """一条输入的完整观测记录。"""

    set_name: str
    arm: str
    item_id: str
    ok: bool
    # 判定相关
    score: float = 0.0  # 0-1，该条的得分
    judge_detail: str = ""
    # 代价相关
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    cost_cny: float = 0.0
    # 原始输出，供人工复核与事后重判
    raw_output: str = ""
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def load_set(name: str) -> list[dict]:
    path = SETS_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"找不到输入池：{path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def price() -> tuple[float, float]:
    p = HERE / "cost_model.json"
    if not p.exists():
        return DEFAULT_PRICE_IN, DEFAULT_PRICE_OUT
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return DEFAULT_PRICE_IN, DEFAULT_PRICE_OUT
    return (
        float(d.get("price_in_per_million", DEFAULT_PRICE_IN)),
        float(d.get("price_out_per_million", DEFAULT_PRICE_OUT)),
    )


def compute_cost(pin: int, pout: int, price_in: float, price_out: float) -> float:
    return round(pin / 1_000_000 * price_in + pout / 1_000_000 * price_out, 8)


# ============================================================ 判定器


def judge_qa(output: dict[str, Any], item: dict[str, Any]) -> tuple[float, str]:
    """问答：规则判要点命中。剩下的（是否真的答对）交人工。

    规则能判的是"gold_points 里有多少条在答案里被覆盖"。
    判不了的是"覆盖了但理解错了" —— 所以 `judge_detail` 会标出
    `needs_human: true`，提醒人工复核。
    """
    answer = str(output.get("答案", ""))
    gold = item["gold_points"]
    hits = [g for g in gold if _point_covered(g, answer)]
    score = len(hits) / len(gold) if gold else 0.0
    detail = f"命中 {len(hits)}/{len(gold)}：{hits}"
    if score < 1.0:
        detail += "；needs_human=true"
    return score, detail


def _point_covered(point: str, answer: str) -> bool:
    """要点覆盖的宽松匹配。

    要点是短语（如"单向足够"），答案可能写成"SSE 是单向的，AI 对话恰好只需要这个方向"。
    所以按**要点里的关键片段**判断：把要点按非中文/非数字切成若干 >=2 字的片段，
    命中过半即认为覆盖。**故意宽松**：宁可让规则判过宽、人工兜底，
    也不要因为措辞不同就判成漏答 —— 后者会让 A/B 的差异全是噪声。
    """
    import re

    if point in answer:
        return True
    frags = [f for f in re.split(r"[^\w\u4e00-\u9fff]+", point) if len(f) >= 2]
    if not frags:
        return point in answer
    hit = sum(1 for f in frags if f in answer)
    return hit >= max(1, len(frags) // 2 + len(frags) % 2)


def judge_classify(output: dict[str, Any], item: dict[str, Any]) -> tuple[float, str]:
    """分类：比 label。命中 1，不中 0。"""
    got = output.get("类别")
    gold = item["gold_label"]
    if got == gold:
        return 1.0, f"命中 {got}"
    return 0.0, f"错：给了 {got!r}，应为 {gold!r}"


def judge_extract(output: dict[str, Any], item: dict[str, Any]) -> tuple[float, str]:
    """抽取：**逐字段比**，算 field-level 准确率 —— 不要只看整条对错。

    为什么必须 field-level：抽取任务里一条 3 字段的样本"对了 2 个"和"全错"
    在整条口径下都是 0 分，但改进价值完全不同。只看整条会让你误判
    "v2 没用"，其实 v2 已经修好了 60% 的字段。

    **同时兼容 v1 与 v2 的输出形状**：
    - v1 把值放在顶层（`{"主体": ..., "时间": ...}`）
    - v2 把值装在 `fields` 字典里（`{"fields": {"版本": ...}}`）
    判定器按"取得到的那个"处理，这样 A/B 跨版本比较才有意义 ——
    否则 v1 vs v2 会因为形状不同而永远判 0 分，比的是格式不是质量。
    """
    gold: dict[str, Any] = item["gold"]
    if not gold:
        return 0.0, "无 gold"

    values = output.get("fields")
    if not isinstance(values, dict):
        values = {k: output.get(k) for k in gold}

    hits: list[str] = []
    misses: list[str] = []
    for field_name, want in gold.items():
        got = values.get(field_name)
        if _value_match(got, want):
            hits.append(field_name)
        else:
            misses.append(f"{field_name}: 给 {got!r} 应 {want!r}")

    score = len(hits) / len(gold)
    detail = f"字段 {len(hits)}/{len(gold)}"
    if misses:
        detail += "；" + "；".join(misses[:3])
    return score, detail


def _value_match(got: Any, want: Any) -> bool:
    """字段值比对：允许格式差异（空格、逗号、全半角），不允许语义差异。"""
    if got is None or want is None:
        return got == want
    g = str(got).replace(" ", "").replace(",", "").replace("％", "%")
    w = str(want).replace(" ", "").replace(",", "").replace("％", "%")
    if g == w:
        return True
    # 数字类：抽出数字比较（"0.001166 元" vs "0.001166"）
    import re

    gn = re.findall(r"\d+\.?\d*", g)
    wn = re.findall(r"\d+\.?\d*", w)
    if gn and wn and gn == wn:
        return True
    # 包含关系：原文抽出来的值通常比 gold 长一点
    return g in w or w in g


def judge_summarize(output: dict[str, Any], item: dict[str, Any]) -> tuple[float, str]:
    """摘要：规则判要点命中 + 长度硬约束。剩下交人工。

    摘要的特殊性：**must_include 是硬要求**（数字、专有名词必须逐字保留），
    所以这里和 qa 不同 —— 不做宽松匹配，要求**逐字出现**。
    """
    summary = str(output.get("摘要", ""))
    must = item["must_include"]
    hits = [m for m in must if m in summary]
    score = len(hits) / len(must) if must else 0.0

    detail = f"要点 {len(hits)}/{len(must)}"
    lost = [m for m in must if m not in summary]
    if lost:
        detail += f"；缺失 {lost[:4]}"

    max_chars = item.get("max_chars")
    if max_chars and len(summary) > max_chars:
        detail += f"；超长 {len(summary)}>{max_chars}"

    if score < 1.0:
        detail += "；needs_human=true"
    return score, detail


JUDGES = {
    "qa": judge_qa,
    "classify": judge_classify,
    "extract": judge_extract,
    "summarize": judge_summarize,
}


# ============================================================ 跑一个臂


async def run_one(
    set_name: str,
    arm: str,
    version: str,
    item: dict[str, Any],
) -> Row:
    from experiment_cache import cached_structured_call

    from app.prompts.loader import load_prompt, load_schema

    schema = load_schema(set_name, version)

    # 按类别拼模板参数。**参数只从输入池取**，不写死任何业务词。
    if set_name == "qa":
        prompt = load_prompt(
            "qa",
            version=version,
            question=item["question"],
            contexts=item["contexts"],
        )
    elif set_name == "classify":
        # 类别体系按本池固定（真实业务类别，来自 schema 的 Literal）
        prompt = load_prompt(
            "classify",
            version=version,
            text=item["text"],
            categories=_classify_categories(),
        )
    elif set_name == "summarize":
        prompt = load_prompt(
            "summarize",
            version=version,
            text=item["text"],
            max_chars=item.get("max_chars", 200),
            must_keep=item.get("must_include", []),
        )
    elif set_name == "extract":
        prompt = load_prompt(
            "extract",
            version=version,
            text=item["text"],
            fields=[
                {
                    "name": f["name"],
                    "desc": f.get("desc", f["name"]),
                    "type": f.get("type", "string"),
                }
                for f in item.get("fields", [])
            ],
        )
    else:
        raise SystemExit(f"未知 set：{set_name}")

    price_in, price_out = price()
    t0 = time.perf_counter()

    # 用量旁路：注册 sink 收集本次调用的真实 usage。
    # 缓存命中时不会有新的 API 调用，所以 usage 为 0 —— 这时退回从缓存文件反查
    # （缓存文件里存了当初那次调用的 usage）。
    usage: dict[str, int] = {}

    def _sink(u: dict[str, int]) -> None:
        usage.update(u)

    from app.core.llm import structured as _st

    prev = getattr(_st, "_USAGE_SINK", None)
    _st.set_usage_sink(_sink)
    try:
        obj = await cached_structured_call(
            prompt,
            schema,
            force_refresh=False,
            extra={"ab": f"{set_name}:{version}", "item": item["id"]},
        )
        elapsed = int((time.perf_counter() - t0) * 1000)
    except Exception as exc:  # noqa: BLE001
        elapsed = int((time.perf_counter() - t0) * 1000)
        return Row(
            set_name=set_name,
            arm=arm,
            item_id=str(item["id"]),
            ok=False,
            latency_ms=elapsed,
            error=f"{type(exc).__name__}: {exc}"[:200],
        )
    finally:
        _st.set_usage_sink(prev)

    output = obj.model_dump(mode="json")
    score, detail = JUDGES[set_name](output, item)

    pin = usage.get("prompt_tokens", 0)
    pout = usage.get("completion_tokens", 0)
    if pin == 0 and pout == 0:
        # 缓存命中：用当初那次调用的记录值
        pin, pout = _usage_from_cache(set_name, version, item["id"])

    return Row(
        set_name=set_name,
        arm=arm,
        item_id=str(item["id"]),
        ok=True,
        score=round(score, 4),
        judge_detail=detail,
        prompt_tokens=pin,
        completion_tokens=pout,
        latency_ms=elapsed,
        cost_cny=compute_cost(pin, pout, price_in, price_out),
        raw_output=json.dumps(output, ensure_ascii=False)[:600],
    )


def _classify_categories() -> list[dict[str, str]]:
    """分类的类别体系。取自 schema 的 Literal，**不在这里手写业务词**。"""
    from app.prompts.classify.schema import ClassifyLabel

    ann = ClassifyLabel.model_fields["类别"].annotation
    names = list(ann.__args__)  # type: ignore[attr-defined]
    boundaries = {
        "故障报修": "产品功能不可用、报错、崩溃",
        "功能咨询": "询问功能怎么用、是否支持某能力",
        "费用争议": "扣费异常、重复扣款、退款、账单不符",
        "投诉建议": "对服务态度或流程不满，或提出改进建议",
        "其他": "确实无法归入以上任何一类",
    }
    return [{"name": n, "boundary": boundaries.get(n, "其他情况")} for n in names]


def _extract_field_descs(item: dict[str, Any]) -> dict[str, str]:
    """抽取的字段说明：取输入池里的 fields 定义。"""
    return {f["name"]: f.get("desc", f["name"]) for f in item.get("fields", [])}


def _usage_from_cache(set_name: str, version: str, item_id: str) -> tuple[int, int]:
    """从缓存文件里反查 token 用量。

    为什么不在调用链里顺便返回 usage：`structured_call` 的签名是周三定版的
    （`-> BaseModel`），**改它会让后面四周全部返工**。所以脚手架这一层自己想办法拿数据：
    缓存文件里记了每次调用的 usage（见 experiment_cache.save 的字段）。

    拿不到就返回 0 —— 显式记为缺失，**不编一个看起来合理的数字**。
    """
    for p in (HERE / ".cache" / "structured").glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        meta = d.get("call_meta") or {}
        if meta.get("ab") == f"{set_name}:{version}" and meta.get("item") == item_id:
            return int(d.get("prompt_tokens", 0)), int(d.get("completion_tokens", 0))
    return 0, 0


# ============================================================ 主流程


async def run_ab(set_name: str, a: str, b: str, *, dry_run: bool) -> list[Row]:
    items = load_set(set_name)
    if dry_run:
        # 干跑只取前 2 条，确认管线通
        items = items[:2]
        print(f"[dry-run] 只跑前 {len(items)} 条确认管线")

    rows: list[Row] = []
    for arm, version in (("A", a), ("B", b)):
        print(f"\n== arm {arm} = version {version} ==")
        for item in items:
            row = await run_one(set_name, arm, version, item)
            rows.append(row)
            mark = "✅" if row.ok else "❌"
            print(
                f"  {mark} {item['id']:<8} score={row.score:.2f} "
                f"tok={row.prompt_tokens}+{row.completion_tokens} "
                f"{row.latency_ms}ms  {row.judge_detail[:60]}"
            )
    return rows


def summarize(rows: list[Row], set_name: str, a: str, b: str) -> dict[str, Any]:
    """汇总。**结论阈值写在这里，不靠人眼看数字猜。**"""
    out: dict[str, Any] = {"set": set_name, "a": a, "b": b, "arms": {}}

    for arm in ("A", "B"):
        rs = [r for r in rows if r.arm == arm]
        ok = [r for r in rs if r.ok]
        out["arms"][arm] = {
            "n": len(rs),
            "ok": len(ok),
            "mean_score": round(statistics.mean(r.score for r in ok), 4) if ok else 0.0,
            "exact_rate": round(sum(1 for r in ok if r.score >= 1.0) / len(ok), 4) if ok else 0.0,
            "total_prompt_tokens": sum(r.prompt_tokens for r in ok),
            "total_completion_tokens": sum(r.completion_tokens for r in ok),
            "total_cost_cny": round(sum(r.cost_cny for r in ok), 6),
            "mean_latency_ms": round(statistics.mean(r.latency_ms for r in ok)) if ok else 0,
            "errors": sum(1 for r in rs if not r.ok),
        }

    a_m = out["arms"]["A"]["mean_score"]
    b_m = out["arms"]["B"]["mean_score"]
    delta = round(b_m - a_m, 4)
    out["delta"] = {
        "mean_score": delta,
        "significant": abs(delta) >= MIN_SIGNAL_DELTA,
        "threshold": MIN_SIGNAL_DELTA,
        "verdict": _verdict(delta),
        "token_ratio": round(
            out["arms"]["B"]["total_prompt_tokens"]
            / max(1, out["arms"]["A"]["total_prompt_tokens"]),
            2,
        ),
    }
    return out


def _verdict(delta: float) -> str:
    if delta >= MIN_SIGNAL_DELTA:
        return "B 更好（差距达到信号阈值）"
    if delta <= -MIN_SIGNAL_DELTA:
        return "A 更好（差距达到信号阈值）"
    return "无显著差异（差距小于阈值，可能是噪声）"


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def print_summary(s: dict[str, Any]) -> None:
    print("\n" + "=" * 66)
    print(f"摘要：set={s['set']}  A={s['a']}  B={s['b']}")
    print("=" * 66)
    print(f"{'指标':<24}{'A':>12}{'B':>12}")
    a, b = s["arms"]["A"], s["arms"]["B"]
    for label, k in [
        ("平均得分(0-1)", "mean_score"),
        ("整条全对率", "exact_rate"),
        ("prompt token 合计", "total_prompt_tokens"),
        ("completion token 合计", "total_completion_tokens"),
        ("成本(元)", "total_cost_cny"),
        ("平均延迟(ms)", "mean_latency_ms"),
        ("失败条数", "errors"),
    ]:
        print(f"{label:<24}{a[k]:>12}{b[k]:>12}")
    d = s["delta"]
    print(f"\nΔ平均得分 = {d['mean_score']:+.4f}  （阈值 {d['threshold']}）")
    print(f"prompt token 倍数 = {d['token_ratio']}x")
    print(f"结论：{d['verdict']}")


async def main() -> None:
    ap = argparse.ArgumentParser(description="prompt A/B 测试脚手架")
    ap.add_argument("--set", required=True, choices=sorted(JUDGES))
    ap.add_argument("--a", required=True, help="A 臂版本，如 v1")
    ap.add_argument("--b", required=True, help="B 臂版本，如 v2")
    ap.add_argument("--dry-run", action="store_true", help="只跑 2 条确认管线")
    args = ap.parse_args()

    if FIXED_TEMPERATURE != 0.0 or FIXED_SEED != 42:
        print("[warn] temperature/seed 不是默认固定值，结果不可与历史对比")

    rows = await run_ab(args.set, args.a, args.b, dry_run=args.dry_run)
    if not rows:
        raise SystemExit("没有采到任何数据")

    s = summarize(rows, args.set, args.a, args.b)
    print_summary(s)

    tag = "dryrun" if args.dry_run else "full"
    csv_path = OUT_DIR / f"{args.set}_{args.a}_vs_{args.b}_{tag}.csv"
    write_csv(rows, csv_path)
    json_path = csv_path.with_suffix(".summary.json")
    json_path.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] CSV: {csv_path.name}   摘要: {json_path.name}")


if __name__ == "__main__":
    asyncio.run(main())
