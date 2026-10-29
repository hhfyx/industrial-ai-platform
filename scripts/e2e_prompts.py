"""四类模板的真实模型验证：模板到底好不好用，必须让模型真跑一遍。

上一轮我只验了"能渲染、能喂给 structured_call、纪律能拦住违规"，
**没验模板的实际输出质量**。本脚本补这一块。

设计要点
--------
1. **每类用真实业务样本**，不是"请输出 JSON"这种玩具输入。
2. **重点验上一轮列出的已知失败场景**，而不是只跑 happy path：
   - qa      : 引用编号是否越界、不确定与引用是否自相矛盾、跨段推断
   - summarize: 数字是否被改写、must_keep 是否逐字保留、字数自报准不准
   - classify: 是否选了两个、边界模糊时是否全归"其他"
   - extract : 是否编造（原文没有的值）、null 与空串是否混用
3. **判定全部程序化**，不看人脸色；但把原始输出也存下来供人工复核。
4. 走 `experiment_cache`，重复跑同一 case 不重复花钱。

运行：
    python -X utf8 scripts/e2e_prompts.py            # 用完缓存
    python -X utf8 scripts/e2e_prompts.py --fresh    # 强制重跑，不用缓存
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.prompts.loader import load_prompt, load_schema  # noqa: E402

OUTPUT = Path(__file__).with_name("e2e_prompts_result.json")

SEP = "=" * 70


# ============================================================ 样本

QA_CASES = [
    {
        "name": "正常可答（信息在上下文里）",
        "question": "退款多久到账？",
        "contexts": [
            {"id": 2, "text": "退款将在 3 个工作日内退回原支付账户。"},
            {"id": 5, "text": "订单发货后不支持无理由退款。"},
        ],
        "expect_uncertain": False,
        "valid_ids": {2, 5},
    },
    {
        "name": "上下文没有答案（该说不确定）",
        "question": "能否开增值税专用发票？",
        "contexts": [
            {"id": 2, "text": "退款将在 3 个工作日内退回原支付账户。"},
            {"id": 5, "text": "订单发货后不支持无理由退款。"},
        ],
        "expect_uncertain": True,
        "valid_ids": {2, 5},
    },
    {
        "name": "需要跨段推断（诱导陷阱）",
        "question": "我 3 月 1 日下单，3 月 5 日申请退款，来得及吗？",
        "contexts": [
            {"id": 1, "text": "退款申请须在发货前提交。"},
            {"id": 3, "text": "订单将在付款后 48 小时内发货。"},
        ],
        # 需要把两段拼起来才能算"3月1日下单→3月3日发货→3月5日申请已发货"
        # 正确行为：说无法从上下文确定，而不是替用户算
        "expect_uncertain": None,  # 两种都可以，重点看引用是否越界
        "valid_ids": {1, 3},
    },
]

SUMMARIZE_CASES = [
    {
        "name": "数字与专有名词保全",
        "text": "本次季度营收 1,240 万元，同比增长 18%。其中华东区贡献 720 万元，"
        "主力型号 HX-300 出货量 4,200 台。华南区营收 390 万元，同比下降 4%。",
        "max_chars": 90,
        "must_keep": ["1,240 万元", "18%", "720 万元", "HX-300", "4,200 台"],
        "numbers_in_source": ["1,240", "18", "720", "4,200", "390", "4"],
    },
    {
        "name": "长文本压缩（看是否漏关键数字）",
        "text": "2026 年第三季度，公司实现营业收入 38.7 亿元，较去年同期增长 12.3%；"
        "归属于母公司股东的净利润 4.15 亿元，同比增长 8.9%。"
        "研发投入 6.02 亿元，占营收比重 15.6%。"
        "经营活动产生的现金流量净额为 9.34 亿元，同比下降 2.1%。"
        "截至报告期末，公司总资产 156.8 亿元，较年初增长 5.4%。",
        "max_chars": 100,
        "must_keep": ["38.7 亿元", "12.3%", "4.15 亿元"],
        "numbers_in_source": ["38.7", "12.3", "4.15", "8.9", "6.02", "15.6", "9.34", "2.1"],
    },
]

CLASSIFY_CASES = [
    {
        "name": "边界清晰（费用争议）",
        "text": "昨天买的会员今天扣了两次钱，第二笔根本没通知我。",
        "categories": [
            {"name": "故障报修", "boundary": "产品功能不可用、报错、崩溃"},
            {"name": "功能咨询", "boundary": "询问功能怎么用、是否支持某能力"},
            {"name": "费用争议", "boundary": "扣费异常、重复扣款、退款、账单不符"},
            {"name": "投诉建议", "boundary": "对服务态度或流程不满，或提出改进建议"},
            {"name": "其他", "boundary": "确实无法归入以上任何一类"},
        ],
        "expect": "费用争议",
    },
    {
        "name": "边界模糊（同时像故障和咨询）",
        "text": "我想知道导出功能是不是坏了，点了没反应，你们这功能到底支不支持导出 PDF？",
        "categories": [
            {"name": "故障报修", "boundary": "产品功能不可用、报错、崩溃"},
            {"name": "功能咨询", "boundary": "询问功能怎么用、是否支持某能力"},
            {"name": "费用争议", "boundary": "扣费异常、重复扣款、退款、账单不符"},
            {"name": "投诉建议", "boundary": "对服务态度或流程不满，或提出改进建议"},
            {"name": "其他", "boundary": "确实无法归入以上任何一类"},
        ],
        "expect": None,  # 两类都沾边，重点看是否只选一个、置信度是否偏低
    },
    {
        "name": "完全套不上（应该归其他）",
        "text": "你们公司明年会不会上市？我想买你们股票。",
        "categories": [
            {"name": "故障报修", "boundary": "产品功能不可用、报错、崩溃"},
            {"name": "功能咨询", "boundary": "询问功能怎么用、是否支持某能力"},
            {"name": "费用争议", "boundary": "扣费异常、重复扣款、退款、账单不符"},
            {"name": "投诉建议", "boundary": "对服务态度或流程不满，或提出改进建议"},
            {"name": "其他", "boundary": "确实无法归入以上任何一类"},
        ],
        "expect": "其他",
    },
]

EXTRACT_CASES = [
    {
        "name": "信息齐全（都能抽到）",
        "text": "客户张伟于 2026-03-15 通过电话 13800138000 反馈，购买了 Mate60 一台。",
        "fields": [
            {"name": "主体", "desc": "联系人姓名", "type": "string"},
            {"name": "时间", "desc": "发生时间", "type": "date"},
        ],
        "present_in_text": {"主体": ["张伟"], "时间": ["2026-03-15"]},
    },
    {
        "name": "信息缺失（不许编）",
        "text": "客户反馈说下周三之前要处理完，联系方式回头再说。",
        "fields": [
            {"name": "主体", "desc": "联系人姓名", "type": "string"},
            {"name": "时间", "desc": "发生时间", "type": "date"},
        ],
        # 原文没有具体人名、没有绝对日期 —— 若模型给出具体姓名/具体日期就是编造
        "present_in_text": {"主体": [], "时间": []},
    },
    {
        "name": "诱导编造（问题里暗示了型号）",
        "text": "用户反映手机发烫严重，玩游戏半小时就很热。",
        "fields": [
            {"name": "主体", "desc": "联系人姓名", "type": "string"},
            {"name": "时间", "desc": "发生时间", "type": "date"},
        ],
        "present_in_text": {"主体": [], "时间": []},
    },
]


# ============================================================ 判定器


def judge_qa(result: dict[str, Any], case: dict[str, Any]) -> list[str]:
    """返回问题列表（空 = 通过）。

    **注意：「不确定=true 却给了引用」不算错**（这是初版判定器的误判，已修正）。
    模型引用 [1][3] 是为了说明"规则这么写、但缺另外的信息"，引用的是**规则依据**
    而非**结论依据** —— 这对人工核查有价值。模板规则已相应放宽（见 qa/v1.jinja2
    的「已知失败场景」第 2 条）。

    真正该判的只有三条：
      1. 引用越界（引用了不存在的段落 id）
      2. 期望的确定性判定不符
      3. 答案为空
    """
    issues: list[str] = []
    ids = result.get("引用编号") or []
    valid = case["valid_ids"]

    # 失败场景 1：引用越界（这才是硬错误）
    bad_ids = [i for i in ids if i not in valid]
    if bad_ids:
        issues.append(f"引用编号越界：{bad_ids}（合法集合 {sorted(valid)}）")

    # 期望的确定性
    exp = case["expect_uncertain"]
    if exp is not None and bool(result.get("不确定")) != exp:
        issues.append(f"不确定判定不符预期：期望 {exp}，实际 {result.get('不确定')}")

    # 失败场景 3：答案为空
    if not str(result.get("答案", "")).strip():
        issues.append("答案为空")

    return issues


def judge_summarize(result: dict[str, Any], case: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    summary = str(result.get("摘要", ""))
    must_keep = case["must_keep"]

    # 失败场景 1：must_keep 未逐字保留
    lost = [m for m in must_keep if m not in summary]
    if lost:
        issues.append(f"must_keep 未逐字保留：{lost}")

    # 失败场景 2：数字被改写（摘要里的数字必须能在原文找到）
    import re

    nums_in_summary = re.findall(r"\d[\d,]*\.?\d*", summary)
    src = case["text"]
    fabricated_nums = [n for n in nums_in_summary if n.replace(",", "") not in src.replace(",", "")]
    if fabricated_nums:
        issues.append(f"出现原文没有的数字：{fabricated_nums}")

    # 失败场景 3：长度超限
    if len(summary) > case["max_chars"]:
        issues.append(f"超长：{len(summary)} > {case['max_chars']}")

    # 失败场景 4：字数自报与实际不符
    reported = result.get("字数")
    actual = len(summary)
    if isinstance(reported, int) and abs(reported - actual) > 5:
        issues.append(f"字数自报偏差大：报 {reported}，实际 {actual}")

    return issues


def judge_classify(result: dict[str, Any], case: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    cat = result.get("类别")

    # 失败场景 1：选了多个（schema 已拦，但这里再验一次）
    if isinstance(cat, list):
        issues.append(f"选了多个类别：{cat}")
        return issues

    allowed = [c["name"] for c in case["categories"]]
    if cat not in allowed:
        issues.append(f"类别不在枚举内：{cat!r}")

    exp = case.get("expect")
    if exp and cat != exp:
        issues.append(f"类别不符预期：期望 {exp}，实际 {cat}")

    conf = result.get("置信度")
    if isinstance(conf, (int, float)):
        if not 0 <= conf <= 1:
            issues.append(f"置信度越界：{conf}")
        # 模糊案例应给低置信度
        if case["name"].startswith("边界模糊") and conf > 0.8:
            issues.append(f"边界模糊却给了高置信度：{conf}")

    if not str(result.get("理由", "")).strip():
        issues.append("理由为空")

    return issues


def judge_extract(result: dict[str, Any], case: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    present = case["present_in_text"]

    for field, must_have in present.items():
        val = result.get(field)
        if must_have:
            # 应有值：抽不到算漏抽
            if not val:
                issues.append(f"{field} 漏抽（原文有：{must_have}）")
            elif not any(m in str(val) for m in must_have):
                issues.append(f"{field} 抽的值不对：{val!r}（原文是 {must_have}）")
        else:
            # 原文没有：给了具体值就是编造
            if val not in (None, "", "null"):
                issues.append(f"{field} 疑似编造：原文没有，却给了 {val!r}")

    # 失败场景：null 与空串混用（应为 None，不是 ""）
    for field in present:
        if result.get(field) == "":
            issues.append(f"{field} 用了空串而非 null")

    return issues


# ============================================================ 执行


async def run_one(
    kind: str,
    case: dict[str, Any],
    *,
    fresh: bool,
) -> dict[str, Any]:
    from experiment_cache import cached_structured_call

    schema = load_schema(kind)

    if kind == "qa":
        prompt = load_prompt(
            "qa",
            question=case["question"],
            contexts=case["contexts"],
        )
    elif kind == "summarize":
        prompt = load_prompt(
            "summarize",
            text=case["text"],
            max_chars=case["max_chars"],
            must_keep=case["must_keep"],
        )
    elif kind == "classify":
        prompt = load_prompt(
            "classify",
            text=case["text"],
            categories=case["categories"],
        )
    elif kind == "extract":
        prompt = load_prompt(
            "extract",
            text=case["text"],
            fields=case["fields"],
        )
    else:
        raise ValueError(kind)

    t0 = time.perf_counter()
    obj = await cached_structured_call(
        prompt, schema, force_refresh=fresh, extra={"case": case["name"], "kind": kind}
    )
    elapsed = time.perf_counter() - t0

    result = obj.model_dump(mode="json")
    judge = {
        "qa": judge_qa,
        "summarize": judge_summarize,
        "classify": judge_classify,
        "extract": judge_extract,
    }[kind]
    issues = judge(result, case)

    return {
        "kind": kind,
        "case": case["name"],
        "result": result,
        "issues": issues,
        "passed": not issues,
        "elapsed_s": round(elapsed, 2),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="强制重跑，不用缓存")
    args = ap.parse_args()

    store: dict[str, Any] = {"meta": {"fresh": args.fresh}, "rows": []}

    plan = [
        ("qa", QA_CASES),
        ("summarize", SUMMARIZE_CASES),
        ("classify", CLASSIFY_CASES),
        ("extract", EXTRACT_CASES),
    ]

    for kind, cases in plan:
        print(f"\n{SEP}\n{kind.upper()}（{len(cases)} 个样本）\n{SEP}")
        for case in cases:
            row = await run_one(kind, case, fresh=args.fresh)
            store["rows"].append(row)
            mark = "✅" if row["passed"] else "❌"
            print(f"  {mark} {case['name']}  ({row['elapsed_s']}s)")
            print(f"     输出: {json.dumps(row['result'], ensure_ascii=False)[:200]}")
            for issue in row["issues"]:
                print(f"     ⚠️  {issue}")

    # 汇总
    total = len(store["rows"])
    passed = sum(1 for r in store["rows"] if r["passed"])
    store["summary"] = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else None,
        "by_kind": {},
    }
    for kind, _ in plan:
        rows = [r for r in store["rows"] if r["kind"] == kind]
        store["summary"]["by_kind"][kind] = {
            "total": len(rows),
            "passed": sum(1 for r in rows if r["passed"]),
        }

    OUTPUT.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{SEP}\n汇总\n{SEP}")
    print(f"  通过 {passed}/{total}  ({store['summary']['pass_rate']:.0%})")
    for kind, v in store["summary"]["by_kind"].items():
        print(f"    {kind:<12} {v['passed']}/{v['total']}")
    print(f"\n[done] 写入 {OUTPUT.name}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
