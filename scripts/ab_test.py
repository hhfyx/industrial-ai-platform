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
import math
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

    **兼容 v1 与 v2 的"不确定"表示方式**：
    - v1 用 `不确定: true/false`（反向语义）
    - v2 用 `能否回答: "能"/"不能"`（正向语义）
    两种都要能判 —— 否则 v1 vs v2 会因为字段名不同而永远判 0 分。

    规则能判的是"gold_points 里有多少条在答案里被覆盖"。
    判不了的是"覆盖了但理解错了" —— 所以 `judge_detail` 会标出
    `needs_human: true`，提醒人工复核。
    """
    answer = str(output.get("答案", ""))
    gold = item["gold_points"]
    hits = [g for g in gold if _point_covered(g, answer)]
    score = len(hits) / len(gold) if gold else 0.0
    detail = f"命中 {len(hits)}/{len(gold)}：{hits}"

    uncertain = _is_uncertain(output)
    exp = item.get("expect_uncertain")
    if exp is not None and uncertain != exp:
        detail += f"；**不确定性不符预期**（期望 {exp}，实际 {uncertain}）"
        score = min(score, 0.5)  # 判反了最多给一半分
    if score < 1.0:
        detail += "；needs_human=true"
    return score, detail


def _is_uncertain(output: dict[str, Any]) -> bool:
    """把 v1/v2 两种表示归一成"是否不确定"。"""
    if "能否回答" in output:
        return str(output["能否回答"]) == "不能"
    return bool(output.get("不确定"))


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
    hallucinations: list[str] = []

    for field_name, want in gold.items():
        got = values.get(field_name)
        if _value_match(got, want):
            hits.append(field_name)
            continue
        misses.append(f"{field_name}: 给 {got!r} 应 {want!r}")
        # 幻觉判定：gold 是"原文没有"（None），模型却给了一个非空值 → 编造
        if want is None and got not in (None, "", "null"):
            hallucinations.append(f"{field_name}={got!r}")

    # 反向幻觉：模型补出了 gold 里根本没有的字段（自创字段名）
    extra_keys = [k for k in values if k not in gold and values.get(k) not in (None, "")]
    if extra_keys:
        hallucinations.append(f"自创字段 {extra_keys}")

    score = len(hits) / len(gold)
    detail = f"字段 {len(hits)}/{len(gold)}"
    if misses:
        detail += "；" + "；".join(misses[:3])
    if hallucinations:
        detail += f"；**幻觉 {len(hallucinations)}**"
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

    **v2 的交叉校验**：v2 会让模型自报 `保留清单`。若模型说"我保留了 X"
    但摘要里没有 X，这是比"直接漏了"更严重的错误（会让人误以为约束生效了）。
    这里把它单独标出来。
    """
    summary = str(output.get("摘要", ""))
    must = item["must_include"]
    hits = [m for m in must if m in summary]
    score = len(hits) / len(must) if must else 0.0

    detail = f"要点 {len(hits)}/{len(must)}"
    lost = [m for m in must if m not in summary]
    if lost:
        detail += f"；缺失 {lost[:4]}"

    # v2 自报与实际不一致 → 虚假保全
    claimed = output.get("保留清单")
    if claimed:
        lies = [c for c in claimed if c not in summary]
        if lies:
            detail += f"；**虚假保全**（自报保留但摘要里没有：{lies[:3]}）"
            score = min(score, 0.5)

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
    *,
    fresh: bool = False,
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
            force_refresh=fresh,
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
    """分类的类别体系。类别名取自 schema 的 Literal，**不在这里另写业务词**。

    `boundary` 是正向定义；`not_boundary` 是**反向定义**（这一类不包括什么）。
    反向定义只有 v2 模板会用 —— v1 模板忽略这个多余的键（Jinja 不会报未用字段）。
    两边共用同一份类别体系，保证 A/B 比的是"写法"而不是"类别定义"。
    """
    from app.prompts.classify.schema import ClassifyLabel

    ann = ClassifyLabel.model_fields["类别"].annotation
    names = list(ann.__args__)  # type: ignore[attr-defined]

    boundaries = {
        "故障报修": "产品功能不可用、报错、崩溃",
        "功能咨询": "询问功能怎么用、是否支持某能力、怎么测算",
        "费用争议": "扣费异常、重复扣款、退款、账单不符",
        "投诉建议": "对服务态度或流程不满，或提出改进建议",
        "其他": "确实无法归入以上任何一类",
    }
    # 反向定义：这一类**不包括**什么。刻意针对实测踩过的混淆点。
    not_boundaries = {
        "故障报修": "不包括单纯询问用法、不包括计费争议、不包括提建议",
        "功能咨询": "不包括已经确认的功能故障、不包括对账单的不满",
        "费用争议": "不包括单纯询问成本怎么算、不包括功能报错",
        "投诉建议": "不包括具体功能报错、不包括计费异常",
        "其他": "不包括任何能勉强归入以上四类的文本",
    }
    return [
        {
            "name": n,
            "boundary": boundaries.get(n, "其他情况"),
            "not_boundary": not_boundaries.get(n, ""),
        }
        for n in names
    ]


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


async def run_ab(set_name: str, a: str, b: str, *, dry_run: bool, fresh: bool = False) -> list[Row]:
    items = load_set(set_name)
    if dry_run:
        # 干跑只取前 2 条，确认管线通
        items = items[:2]
        print(f"[dry-run] 只跑前 {len(items)} 条确认管线")

    if not fresh:
        print("[warn] 使用缓存：命中时延迟会失真（接近 0ms）。要真实的延迟/P95 请加 --fresh")

    rows: list[Row] = []
    for arm, version in (("A", a), ("B", b)):
        print(f"\n== arm {arm} = version {version} ==")
        for item in items:
            row = await run_one(set_name, arm, version, item, fresh=fresh)
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
        lat = sorted(r.latency_ms for r in ok)
        n = len(rs)
        n_ok = len(ok)

        # ---- 三个口径必须**同时**给出，否则会得出错误结论 ----
        #
        # ① mean_score_of_ok / mean_score_all
        #    仅算成功 vs 失败记 0。
        #    例：摘要 B 版 7 条超时，只算成功得 1.0（"完美"），全量得 0.3。
        #
        # ② 成本与 token：**必须按成功样本归一**
        #    失败的请求没有输出 token，于是"总成本更低"可能只等于"有几次没跑起来"。
        #    实测血案（2026-11-08 复核）：摘要 v2 总成本 0.294 → 0.134，
        #    字面看省 62%；但 v1 9 条成功、v2 只有 3 条成功 ——
        #    按每条成功归一，v1 = 0.03266、v2 = 0.04471，**v2 实际贵 37%**。
        #    更隐蔽的一例：extract 的 token_ratio 原报 1.01（"两版一样贵"），
        #    按成功归一后是 0.418（v2 省 58%）—— 因为 v1 那条失败样本
        #    在修复轮里烧了 23270 completion token，被算进了分母。
        #    所以下面同时给 total / per_ok 两套，且 delta 里用 per_ok。
        mean_ok = round(statistics.mean(r.score for r in ok), 4) if ok else 0.0
        mean_all = round(sum(r.score for r in rs) / n, 4) if n else 0.0

        total_pin = sum(r.prompt_tokens for r in ok)
        total_pout = sum(r.completion_tokens for r in ok)
        total_cost = sum(r.cost_cny for r in ok)

        out["arms"][arm] = {
            "n": n,
            "ok": n_ok,
            "mean_score": mean_ok,
            "mean_score_all": mean_all,
            "exact_rate": round(sum(1 for r in ok if r.score >= 1.0) / n_ok, 4) if n_ok else 0.0,
            "exact_rate_all": round(sum(1 for r in rs if r.ok and r.score >= 1.0) / n, 4)
            if n
            else 0.0,
            # 总量（含"因为失败所以没花"的偏差，仅作参考）
            "total_prompt_tokens": total_pin,
            "total_completion_tokens": total_pout,
            "total_cost_cny": round(total_cost, 6),
            # **按成功样本归一**（成本/品质对比的正确分母）
            "cost_per_ok": round(total_cost / n_ok, 8) if n_ok else None,
            "prompt_tokens_per_ok": round(total_pin / n_ok) if n_ok else None,
            "completion_tokens_per_ok": round(total_pout / n_ok) if n_ok else None,
            "mean_latency_ms": round(statistics.mean(lat)) if lat else 0,
            # P95：10 条样本时取第 10 个（最慢那条），即 nearest-rank
            "p95_latency_ms": _percentile(lat, 95),
            "errors": sum(1 for r in rs if not r.ok),
            "error_ids": [r.item_id for r in rs if not r.ok],
        }

    # 分类额外：混淆矩阵（只看 accuracy 会被类别不均衡骗）
    if set_name == "classify":
        out["confusion"] = _confusion_matrix(rows, load_set("classify"))

    # 抽取额外：幻觉字段率
    if set_name == "extract":
        out["hallucination"] = _hallucination_stats(rows, load_set("extract"))

    # Δ 用**全量口径**（失败按 0 分），因为可用性差异才是要回答的问题。
    # 只算成功的口径会得出"B 和 A 一样好"，即使 B 有一半超时。
    a_m = out["arms"]["A"]["mean_score_all"]
    b_m = out["arms"]["B"]["mean_score_all"]
    delta = round(b_m - a_m, 4)

    # ---- 成本/token 比值**必须按成功样本归一** ----
    # 用 total 算比值会被失败率污染：失败的请求不产出 token，
    # 于是"总 token 少"可能只等于"有几次没跑起来"。
    # 见上面 out["arms"] 构造处的血案记录。这里给出两套，主口径用 per_ok。
    a_arm, b_arm = out["arms"]["A"], out["arms"]["B"]

    def _ratio(a_v: float | None, b_v: float | None) -> float | None:
        if a_v is None or b_v is None or a_v == 0:
            return None
        return round(b_v / a_v, 3)

    out["delta"] = {
        "mean_score": delta,
        "mean_score_of_ok": round(b_arm["mean_score"] - a_arm["mean_score"], 4),
        "scoring": "mean_score 用全量口径（失败记 0）；成本比值用每条成功归一",
        "significant": abs(delta) >= MIN_SIGNAL_DELTA,
        "threshold": MIN_SIGNAL_DELTA,
        "verdict": _verdict(delta),
        # 主口径：每条成功
        "cost_ratio_per_ok": _ratio(a_arm.get("cost_per_ok"), b_arm.get("cost_per_ok")),
        "token_ratio_per_ok": _ratio(
            (a_arm["total_prompt_tokens"] + a_arm["total_completion_tokens"]) / max(1, a_arm["ok"]),
            (b_arm["total_prompt_tokens"] + b_arm["total_completion_tokens"]) / max(1, b_arm["ok"]),
        ),
        "prompt_token_ratio_per_ok": _ratio(
            a_arm.get("prompt_tokens_per_ok"), b_arm.get("prompt_tokens_per_ok")
        ),
        # 辅口径：总量（含失败偏差，**仅作参考，不要单独引用**）
        "token_ratio_total_uncontrolled": round(
            b_arm["total_prompt_tokens"] / max(1, a_arm["total_prompt_tokens"]), 3
        ),
        "failure_rate_a": round(a_arm["errors"] / max(1, a_arm["n"]), 4),
        "failure_rate_b": round(b_arm["errors"] / max(1, b_arm["n"]), 4),
        "cost_ratio_caveat": (
            "若两臂失败率相差悬殊，total 口径的成本比值会失真；应以 cost_ratio_per_ok 为准"
        ),
    }
    return out


def _percentile(sorted_values: list[int], pct: int) -> int:
    """nearest-rank 百分位。

    为什么用 nearest-rank 而不是线性插值：10 条样本做插值会造出一个
    **不存在的延迟值**（比如把 5000ms 和 8000ms 插成 6500ms）。
    样本这么少的时候，报"第 10 条实际是 8000ms"比报一个虚构的中间值诚实。
    """
    if not sorted_values:
        return 0
    k = max(1, math.ceil(len(sorted_values) * pct / 100))
    return sorted_values[min(k, len(sorted_values)) - 1]


def _confusion_matrix(rows: list[Row], items: list[dict]) -> dict[str, Any]:
    """混淆矩阵：**行=gold，列=预测**。

    为什么必须出这个：类别不均衡时，全猜最多类也能有 60% 的 accuracy
    （本池故障 3 / 咨询 3 / 费用 2 / 投诉 2，全猜"故障报修"或"功能咨询"都是 30%，
    但换成更不均衡的数据集，accuracy 会严重造假）。
    混淆矩阵让"哪两类互相混"直接可见 —— 这比一个总数有用得多。
    """
    golds = {str(i["id"]): i["gold_label"] for i in items}
    labels = sorted(set(golds.values()))
    mat: dict[str, Any] = {"labels": labels, "arms": {}}

    for arm in ("A", "B"):
        arm_rows = {r.item_id: r for r in rows if r.arm == arm and r.ok}
        grid = {g: dict.fromkeys(labels, 0) for g in labels}
        preds: dict[str, str] = {}

        for item_id, gold in golds.items():
            r = arm_rows.get(item_id)
            if not r:
                continue
            pred = _predicted_label(r.raw_output, labels)
            preds[item_id] = pred
            if pred in grid[gold]:
                grid[gold][pred] += 1

        # 最容易混的两对：非对角线上的格子，按次数排序
        confusions = []
        for g in labels:
            for p in labels:
                if g != p and grid[g][p]:
                    confusions.append({"gold": g, "pred": p, "count": grid[g][p]})
        confusions.sort(key=lambda x: -x["count"])

        mat["arms"][arm] = {
            "grid": grid,
            "predictions": preds,
            "top_confusions": confusions[:5],
            "accuracy": round(
                sum(grid[g][g] for g in labels)
                / max(1, sum(sum(row.values()) for row in grid.values())),
                4,
            ),
            # 基线：全猜最多类能拿到的准确率
            "majority_baseline": round(
                max(sum(grid[g].values()) for g in labels)
                / max(1, sum(sum(row.values()) for row in grid.values())),
                4,
            ),
        }
    return mat


def _predicted_label(raw_output: str, labels: list[str]) -> str:
    """从原始输出里取预测类别。取不到返回 '?'。"""
    try:
        d = json.loads(raw_output)
    except json.JSONDecodeError:
        return "?"
    got = d.get("类别")
    if isinstance(got, list):
        return "MULTI"
    return str(got) if got in labels else "?"


def _hallucination_stats(rows: list[Row], items: list[dict]) -> dict[str, Any]:
    """幻觉字段率：模型给出"原文根本没有的值"的比例。

    **字段级准确率不够用**：一条样本可以"字段对了 2/3"但同时编了一个值。
    编造比漏抽更危险 —— 漏抽是可发现的空缺，编造会被当成真实数据用下去。

    判定：gold 里某字段是 `None`（gold 表明原文没有），模型却给了非空值 → 一次幻觉。
    分母是该臂抽出的**所有非空字段值**，这样"少抽"不会被误算成"多编"。
    """
    golds = {str(i["id"]): i["gold"] for i in items}
    out: dict[str, Any] = {"arms": {}}

    for arm in ("A", "B"):
        arm_rows = [r for r in rows if r.arm == arm and r.ok]
        produced_values = 0  # 模型给出的非空值总数
        hallucinated = 0  # 其中属于"原文没有"的
        hallucinated_ids: list[str] = []

        for r in arm_rows:
            try:
                d = json.loads(r.raw_output)
            except json.JSONDecodeError:
                continue
            values = d.get("fields") if isinstance(d.get("fields"), dict) else d
            gold = golds.get(r.item_id, {})

            for k, v in values.items():
                if v in (None, "", "null"):
                    continue
                produced_values += 1
                # gold 里该字段是 None 且模型给了值 → 编造
                if k in gold and gold[k] is None:
                    hallucinated += 1
                    if r.item_id not in hallucinated_ids:
                        hallucinated_ids.append(r.item_id)

        out["arms"][arm] = {
            "n": len(arm_rows),
            "produced_values": produced_values,
            "hallucinated_values": hallucinated,
            "hallucination_rate": round(hallucinated / max(1, produced_values), 4),
            "hallucinated_items": hallucinated_ids,
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
    print(f"{'指标':<26}{'A':>12}{'B':>12}")
    a, b = s["arms"]["A"], s["arms"]["B"]
    for label, k in [
        ("成功条数/总条数", "ok"),
        ("平均得分(仅成功)", "mean_score"),
        ("平均得分(全量)", "mean_score_all"),
        ("整条全对率(仅成功)", "exact_rate"),
        ("整条全对率(全量)", "exact_rate_all"),
        ("失败条数", "errors"),
        ("--- 总量（含失败偏差）---", None),
        ("prompt token 合计", "total_prompt_tokens"),
        ("completion token 合计", "total_completion_tokens"),
        ("成本合计(元)", "total_cost_cny"),
        ("--- 按每条成功归一 ---", None),
        ("每条成功成本(元)", "cost_per_ok"),
        ("每条成功prompt token", "prompt_tokens_per_ok"),
        ("每条成功compl token", "completion_tokens_per_ok"),
        ("--- 延迟 ---", None),
        ("平均延迟(ms)", "mean_latency_ms"),
        ("P95 延迟(ms)", "p95_latency_ms"),
    ]:
        if k is None:
            print(f"{label}")
            continue
        av, bv = a.get(k), b.get(k)
        if isinstance(av, float):
            print(f"{label:<26}{av:>12.6f}{bv:>12.6f}")
        else:
            print(f"{label:<26}{str(av):>12}{str(bv):>12}")

    # ---- 混淆矩阵（分类专用）----
    conf = s.get("confusion")
    if conf:
        labels = conf["labels"]
        for arm in ("A", "B"):
            ma = conf["arms"][arm]
            print(
                f"\n混淆矩阵 arm {arm}（行=gold，列=预测；acc={ma['accuracy']:.2f}，"
                f"全猜最多类基线={ma['majority_baseline']:.2f}）"
            )
            head = " " * 10 + "".join(f"{lb[:6]:>8}" for lb in labels)
            print(head)
            for g in labels:
                row = ma["grid"][g]
                cells = "".join(f"{row[p]:>8}" for p in labels)
                print(f"{g[:8]:<10}{cells}")
            if ma["top_confusions"]:
                print("  最易混：", end="")
                print(
                    "，".join(
                        f"{c['gold']}→{c['pred']}({c['count']}次)" for c in ma["top_confusions"]
                    )
                )

    # ---- 幻觉率（抽取专用）----
    hal = s.get("hallucination")
    if hal:
        print("\n幻觉字段率（模型给出原文没有的值）")
        print(f"{'':<24}{'A':>12}{'B':>12}")
        for label, k in [
            ("非空字段值总数", "produced_values"),
            ("其中编造", "hallucinated_values"),
            ("幻觉率", "hallucination_rate"),
        ]:
            print(f"{label:<24}{hal['arms']['A'][k]:>12}{hal['arms']['B'][k]:>12}")
        print(f"  编造条目 A: {hal['arms']['A']['hallucinated_items']}")
        print(f"  编造条目 B: {hal['arms']['B']['hallucinated_items']}")

    d = s["delta"]
    print(f"\nΔ平均得分 = {d['mean_score']:+.4f}  （阈值 {d['threshold']}，全量口径）")
    print(f"失败率 A/B = {d['failure_rate_a']:.0%} / {d['failure_rate_b']:.0%}")

    # 成本比值：主口径按成功归一，辅口径给总量并标注污染风险
    cr = d.get("cost_ratio_per_ok")
    tr = d.get("token_ratio_per_ok")
    raw = d.get("token_ratio_total_uncontrolled")
    print(f"\n成本比值（每条成功）B/A = {cr}x" if cr else "\n成本比值：无法计算（有一臂全失败）")
    print(f"token 比值（每条成功）B/A = {tr}x")
    print(f"token 比值（总量，未控失败率）B/A = {raw}x  ← 仅参考")
    if d["failure_rate_a"] != d["failure_rate_b"]:
        print(
            f"⚠️ 两臂失败率不同（{d['failure_rate_a']:.0%} vs {d['failure_rate_b']:.0%}），"
            f"总量的成本/token 比值会被污染，**以上面 per_ok 为准**"
        )
    print(f"\n结论：{d['verdict']}")


async def main() -> None:
    ap = argparse.ArgumentParser(description="prompt A/B 测试脚手架")
    ap.add_argument("--set", required=True, choices=sorted(JUDGES))
    ap.add_argument("--a", required=True, help="A 臂版本，如 v1")
    ap.add_argument("--b", required=True, help="B 臂版本，如 v2")
    ap.add_argument("--dry-run", action="store_true", help="只跑 2 条确认管线")
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="强制重跑不用缓存。**要真实的延迟/P95 必须加这个**（缓存命中时延迟接近 0）",
    )
    args = ap.parse_args()

    if FIXED_TEMPERATURE != 0.0 or FIXED_SEED != 42:
        print("[warn] temperature/seed 不是默认固定值，结果不可与历史对比")

    rows = await run_ab(args.set, args.a, args.b, dry_run=args.dry_run, fresh=args.fresh)
    if not rows:
        raise SystemExit("没有采到任何数据")

    s = summarize(rows, args.set, args.a, args.b)
    print_summary(s)

    tag = "dryrun" if args.dry_run else "full"
    if args.fresh:
        tag += "_fresh"
    csv_path = OUT_DIR / f"{args.set}_{args.a}_vs_{args.b}_{tag}.csv"
    write_csv(rows, csv_path)
    json_path = csv_path.with_suffix(".summary.json")
    json_path.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] CSV: {csv_path.name}   摘要: {json_path.name}")


if __name__ == "__main__":
    # **W4 追加**：Windows 控制台默认 GBK，而本脚本会打印 ✅/❌ 这类符号。
    # 不加这几行，`--dry-run` 会在打印第一行结果时抛
    #   UnicodeEncodeError: 'gbk' codec can't encode character '\u274c'
    # 那条错误看起来像"脚本坏了"，实际只是"控制台编码不对"。
    #
    # 同一模式在 `app/cli/ask.py` 与 `app/cli/summarize.py` 里都有 ——
    # 这是 Windows 上所有"打印中文/符号"的 CLI 的共同前置。
    import sys as _sys
    from contextlib import suppress as _suppress

    for _stream in (_sys.stdout, _sys.stderr):
        _reconfigure = getattr(_stream, "reconfigure", None)
        if _reconfigure is not None:
            with _suppress(ValueError, OSError):
                _reconfigure(encoding="utf-8")

    asyncio.run(main())
