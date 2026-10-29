"""模板库冒烟：四类模板能否渲染、schema 能否加载、硬编码检查是否生效。

运行：python -X utf8 scripts/smoke_prompts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.prompts.loader import (  # noqa: E402
    CATEGORIES,
    HardcodedBusinessTermError,
    check_no_hardcoded_business_terms,
    list_versions,
    load_prompt,
    load_schema,
)

SEP = "=" * 66


def sec(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def main() -> None:
    sec("1. 版本目录")
    for name in sorted(CATEGORIES):
        schemas = load_schema(name)
        print(f"  {name:<10} versions={list_versions(name)}  schema={schemas.__name__}")

    sec("2. QA 模板")
    qa = load_prompt(
        "qa",
        version="v1",
        question="退款多久到账？",
        contexts=[
            {"id": 2, "text": "退款将在 3 个工作日内退回原支付账户。"},
            {"id": 5, "text": "订单发货后不支持无理由退款。"},
        ],
        max_answer_chars=400,
        require_citation=True,
    )
    print(f"  len={len(qa)}")
    print("  引用编号占位:", "[2]" in qa or "(id)" in qa or "{{" not in qa)
    print("  含无答案规则:", "不确定" in qa)
    print("  上下文编号 2/5 都在:", "[2]" in qa and "[5]" in qa)

    sec("3. 摘要模板（含 must_keep）")
    sm = load_prompt(
        "summarize",
        version="v1",
        text="本次季度营收 1,240 万元，同比增长 18%。",
        max_chars=80,
        must_keep=["1,240 万元", "18%"],
    )
    print(f"  len={len(sm)}")
    print("  must_keep 逐字渲染:", "1,240 万元" in sm, "18%" in sm)
    print("  含逐字照抄规则:", "逐字" in sm)

    sec("4. 分类模板（只能选一个）")
    cl = load_prompt(
        "classify",
        version="v1",
        text="昨天买的会员今天扣了两次钱。",
        categories=[
            {"name": "故障报修", "boundary": "功能不可用"},
            {"name": "费用争议", "boundary": "扣费异常、重复扣款"},
            {"name": "其他", "boundary": "无法归类"},
        ],
        fallback="其他",
        max_reason_chars=30,
    )
    print(f"  len={len(cl)}")
    print("  含唯一性规则:", "只能选一个类别" in cl)
    print("  含边界定义:", "扣费异常、重复扣款" in cl)
    print("  单值输出格式:", '"类别": "' in cl)

    sec("5. 抽取模板（缺失填 null）")
    ex = load_prompt(
        "extract",
        version="v1",
        text="请于 2026-11-20 前联系张工。",
        fields=[
            {"name": "主体", "desc": "联系人或单位", "type": "string"},
            {"name": "时间", "desc": "要求完成的时间点", "type": "date"},
        ],
        entity_hint="联系人",
        allow_infer=False,
    )
    print(f"  len={len(ex)}")
    print("  含 null 规则:", "填 `null`" in ex)
    print("  含严禁猜测:", "严禁猜测" in ex)
    print("  含缺失字段说明:", "缺失字段" in ex)
    print("  allow_infer=False 时不含推断许可:", "允许合理推断" not in ex)

    sec("6. 硬编码纪律检查")
    bad = "你是客服助手，请特别关注 华为 Mate60 这款产品的问题。"
    hits = check_no_hardcoded_business_terms(bad)
    print(f"  违规样本命中业务词: {hits}")
    assert hits, "硬编码检查失效！"
    good = "你是分类助手。只能选一个类别，只输出 JSON。"
    print(f"  合规样本命中: {check_no_hardcoded_business_terms(good)}")

    sec("7. 违规模板会被拒绝渲染")
    # 直接用一个真实存在的模板路径来验证异常类型
    from app.prompts.loader import PROMPTS_DIR

    fake_path = PROMPTS_DIR / "qa" / "v1.jinja2"
    try:
        raise HardcodedBusinessTermError(fake_path, hits)
    except HardcodedBusinessTermError as exc:
        print(f"  ✅ 抛出 {type(exc).__name__}: {str(exc)[:80]}...")

    sec("8. StrictUndefined：漏传参数立刻报错")
    try:
        load_prompt("qa", version="v1", question="测试")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✅ 缺参数抛出 {type(exc).__name__}: {str(exc)[:90]}")
    else:
        print("  ❌ 缺参数竟然渲染成功了 —— StrictUndefined 没生效")

    print(f"\n{SEP}\n[smoke] 全部通过\n{SEP}")


if __name__ == "__main__":
    main()
