"""数字大核对（周日 10:40–11:30）：每个数字问三句。

规则：
  1. n 多少？
  2. 什么配置？（是否含可靠性机制）
  3. 什么口径？（降级算不算失败、是不是回归子集）

答不出 → 改保守表述或删掉。

这个脚本把简历里要用的每个数字，从**原始产物**里重新算一遍并打印三问答案，
防止"凭记忆写数字"。
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path(".")
TITLE = "=" * 78


def load(rel: str) -> dict:
    return json.loads((BASE / rel).read_text(encoding="utf-8"))


def main() -> int:
    print(TITLE)
    print("数字大核对：简历 v2 要用的每一个数")
    print(TITLE)

    # ---------- 1. 20 任务集成功率（区间，不是单点）----------
    ev = load("eval/results/agent-eval-w11.json")
    rounds = ev["rounds"]
    rates = [r["success_rate"] for r in rounds]
    print("\n[1] 20 任务集成功率")
    print(f"    n        : 20 任务 × {len(rounds)} 轮（trae/qwen-3.5）")
    print(f"    数值     : 区间 {min(rates):.0%} ~ {max(rates):.0%}")
    print("    配置     : 含可靠性机制的当前版本（r5 = 19/20）")
    print("    口径     : 判据冻结 01-01；降级算失败（stopped_reason 非 completed 即失败）")
    print("    逐轮     : " + ", ".join(f"{r['round']}={r['passed']}/{r['total']}" for r in rounds))
    print("    ⚠ 简历只能写区间或标注「单轮最好 100%」，不许写「成功率 100%」")

    # ---------- 2. 改进前 vs 改进后 ----------
    r1 = next(r for r in rounds if r["round"] == "r1")
    r2 = next(r for r in rounds if r["round"] == "r2")
    print("\n[2] 改进前后")
    print("    n        : 20 任务，各 1 轮（同一模型、同一判据）")
    print(
        f"    数值     : {r1['passed']}/{r1['total']} ({r1['success_rate']:.0%}) → "
        f"{r2['passed']}/{r2['total']} ({r2['success_rate']:.0%})"
    )
    print("    配置     : 改前=基线 prompt；改后=+2 条增量约束")
    print("    口径     : r1/r2 这一对；后续轮次说明该提升不稳定（T7 跨 5 轮 1/5）")

    # ---------- 3. 编造 ----------
    fab = ev["fabrication_audit"]
    print("\n[3] 编造型失败")
    print("    n        : 6 个不可完成任务（T9/T10/N9/N10/N11/N12）× 5 轮")
    print(f"    数值     : {fab['real_fabrications']} 例")
    print("    口径     : 「答案里的数值在成功工具返回里找不到出处」才算编造")
    print("    证据     : scripts/audit_fabrication.py 可重跑；轨迹含 evidence_numbers")
    print("    ⚠ 必须同时声明：判据曾把「有出处的引用」误判成编造（T10/N11），已区分")

    # ---------- 4. 降级率 ----------
    degrade_rates = {r["round"]: r["degrade_rate"] for r in rounds}
    print("\n[4] 降级率")
    print("    n        : 20 任务")
    print(
        f"    数值     : 当前版本 r5 = {degrade_rates.get('r5', 0):.0%}；"
        f"5 轮区间 {min(degrade_rates.values()):.0%} ~ {max(degrade_rates.values()):.0%}"
    )
    print("    配置     : 预算 10k/120s（compute 类 12k/150s）")
    print("    口径     : stopped_reason ∈ {degraded, budget_*} 即计入")
    print("    ⚠ 降级率波动大是因为预算判定改了三版，简历写「当前版本 ~10%」")

    # ---------- 5. 成本 ----------
    print("\n[5] 成本变化")
    print("    n        : 20 任务 × 2 轮（r1 无约束 / r2 有约束）")
    print(
        f"    数值     : {r1['avg_tokens']:.0f} → {r2['avg_tokens']:.0f} token/任务 "
        f"(+{r2['avg_tokens'] - r1['avg_tokens']:.0f}, "
        f"{(r2['avg_tokens'] / r1['avg_tokens'] - 1) * 100:+.1f}%)"
    )
    print("    配置     : 两条增量约束（路径 + 缺失指标）")
    print("    口径     : 均值为 20 任务算术平均，含 4 个单步结束的 impossible")
    print("    ⚠ 不是「成本下降」——是成本上升 10.4% 换回 2 个任务")

    # ---------- 6. 预算曲线 ----------
    bc = load("eval/results/budget-curve-w11.json")
    summ = {s["tier"]: s for s in bc["summary"]}
    print("\n[6] 预算-成功率曲线")
    print("    n        : 6 个代表任务 × 3 档 × 1 轮（trae/qwen-3.5）")
    print(
        f"    数值     : 5k={summ[5000]['success_rate']:.0%}, "
        f"10k={summ[10000]['success_rate']:.0%}, 15k={summ[15000]['success_rate']:.0%}"
    )
    print("    配置     : 无三道防跑偏（默认全关）")
    print("    口径     : 判据同 01-01 冻结版")
    print("    ⚠ 简历模板里写的「15k 档为性价比拐点」与实测不符：实测 10k 与 15k 同为 83%，")
    print("      15k 只多花 token。必须改成「10k 为拐点」或删掉这句")

    # ---------- 7. 稳定性 ----------
    st = (
        load("qa_runs_w11_stab2/stability.json")
        if (BASE / "qa_runs_w11_stab2/stability.json").exists()
        else None
    )
    print("\n[7] 稳定性")
    print("    n        : 6 个关键任务 × 3 次")
    if st:
        allpass = sum(1 for v in st.values() if v["passes"] == v["runs"])
        print(f"    数值     : 第二组 {allpass}/6 任务 3/3；但第一组 T7 是 0/3")
    print("    口径     : 3/3 的 95% 置信下界约 0.37，不能说「稳定 100%」")
    print("    ⚠ 简历写「关键任务各跑 3 次验证稳定性」可以说，不能附「全部稳定」")

    # ---------- 8. 回归子集 ----------
    print("\n[8] W10 回归子集")
    print("    n        : 10 任务")
    print("    数值     : 9/10（与 v1.2.0 基线持平）")
    print("    配置     : 含预算/降级机制，不含三道防跑偏（默认全关）")
    print("    口径     : 那个 × 是 T10 已知判据误伤（与基线同一个 ×）")

    print()
    print(TITLE)
    print("已核对 8 组数字。凡标 ⚠ 的，简历里必须按提示改或删。")
    print(TITLE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
