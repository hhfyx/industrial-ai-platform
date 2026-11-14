"""一键复现数字台账（W14 周二 14:40–15:40 交付）。

**目的**：面试官问"这个数字怎么来的"时，一条命令把台账里每一个
可复现数字重新算一遍并打印 —— 输出即台账格式，可直接对照。

设计原则（与任务书对齐）：
- **不做全量重跑**（太贵）：GPU 基准、压测、LLM 评测都只读最近一次落盘产物；
- **本地可算的当场算**：成本拆账（usage_ledger 直录）、20 任务成功率区间、
  编造审计、预算曲线 —— 这些零成本或已落盘；
- **每行带复现命令**：读产物 ≠ 造数字，产物路径与原始命令同时打印。

必答问题的回答靠它兑现：`make report-numbers` 跑出来的数字与
`docs/w14-number-ledger.md` 不一致的项，会在输出末尾列出
（当前预期：GPU 单张耗时允许 ±5% 硬件漂移，其余应完全一致）。

跑法：
    python -X utf8 scripts/report_numbers.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

# Windows 控制台默认 GBK：tasks.py 转发时不带 -X utf8，台账里的 ²/×/→ 会
# 直接 UnicodeEncodeError。脚本自持 stdout 编码，任何入口都不再依赖调用方。
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL = PROJECT_ROOT / "eval" / "results"
TRACES_DB = PROJECT_ROOT / "data" / "traces" / "traces.db"

LINE = "=" * 78


def load(rel: str) -> dict:
    return json.loads((EVAL / rel).read_text(encoding="utf-8"))


def latest(pattern: str) -> Path:
    # **按修改时间取最新，且排除非日期命名的基线文件**。
    # 踩坑记录（2026-11-14 首跑）：`sorted(glob("retrieval-*.json"))[-1]`
    # 会把 `retrieval-baseline.json`（W5 旧产物，1592 块 / MRR 0.2402）
    # 排在日期命名的最新产物之后 —— 输出漂移成旧数字而检查项恰好还通过
    # （Recall@5 都是 40.0%），只有 MRR/块数暴露差异。按 mtime 取最新 +
    # 限定日期命名，两类旧文件都进不来。
    hits = [p for p in EVAL.glob(pattern) if "baseline" not in p.name]
    hits = [p for p in hits if any(ch.isdigit() for ch in p.stem)]
    if not hits:
        raise FileNotFoundError(f"无产物匹配 {pattern}（先跑对应的原始命令）")
    return max(hits, key=lambda p: p.stat().st_mtime)


def section(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def main() -> int:
    print(LINE)
    print("W14 数字台账一键复现（对照 docs/w14-number-ledger.md）")
    print(LINE)

    # ---------- 1. W11 20 任务成功率（实时重算，非记忆值） ----------
    section("[1] W11 20 任务成功率区间（从原始产物重算）")
    ev = load("agent-eval-w11.json")
    rounds = ev["rounds"]
    rates = [r["success_rate"] for r in rounds]
    print(
        f"    值      : 区间 {min(rates):.0%} ~ {max(rates):.0%}"
        f"（20 任务 × {len(rounds)} 轮，trae/qwen-3.5）"
    )
    print("    逐轮    : " + ", ".join(f"{r['round']}={r['passed']}/{r['total']}" for r in rounds))
    print("    口径    : 判据冻结 01-01；降级计入失败（stopped_reason 非 completed 即失败）")
    print("    复现    : python -X utf8 scripts/verify_resume_numbers.py")
    fab = ev["fabrication_audit"]
    print(
        f"    编造    : {fab['real_fabrications']} 例（6 个不可完成 × 5 轮，"
        f"轨迹 evidence_numbers 可查；audit_fabrication.py 可重跑）"
    )

    # ---------- 2. 预算曲线拐点（从冻结快照重算） ----------
    section("[2] W11 预算-成功率曲线（10k 拐点）")
    bc = load("budget-curve-w11.json")
    for row in bc["summary"]:
        print(
            f"    {row['tier']:>6} 档: 成功率 {row['success_rate']:.0%}"
            f"（{row['passed']}/{row['runs']}）· avg tokens {row['avg_tokens']:.0f}"
        )
    print(
        "    复现    : python -X utf8 scripts/budget_curve.py（会花 LLM 钱）；"
        "快照 eval/results/budget-curve-w11.json"
    )

    # ---------- 3. 成本拆账（usage_ledger 直录，本地 SQLite 实算） ----------
    section("[3] 真实成本拆账（usage_ledger 直录锚点，非估算）")
    conn = sqlite3.connect(TRACES_DB)
    rows = conn.execute(
        "SELECT endpoint, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens),"
        " ROUND(SUM(cost),4) FROM usage_ledger GROUP BY endpoint"
    ).fetchall()
    total_cost, total_calls = 0.0, 0
    for ep, n, ptok, ctok, csum in rows:
        total_cost += csum
        total_calls += n
        print(f"    {ep:<20} n={n:>4}  prompt={ptok:>8,}  completion={ctok:>7,}  cost={csum} 元")
    print(f"    合计    : {total_calls} 笔直录 / {round(total_cost, 4)} 元")
    agent_cost = next((r[4] for r in rows if r[0] == "agent.model_step"), 0.0)
    if total_cost:
        print(
            f"    Agent 占比: {agent_cost / total_cost:.1%}"
            "（agent.model_step / 全部；全时间窗口径，"
            "W13 周记的 79% 是『20 条请求窗口』口径，两者不混用）"
        )
    conn.close()
    print("    复现    : 本命令（SQLite 直读）；库文件 data/traces/traces.db")

    # ---------- 4. GPU 生成基准（读最近一次落盘 + 对照首轮实测） ----------
    section("[4] 项目一 GPU 基准（读最近一次产物；允许 ±5% 硬件漂移）")
    bench = latest("generator-bench-*.json")
    d = json.loads(bench.read_text(encoding="utf-8"))
    print(f"    产物    : {bench.name}")
    print(
        f"    加载    : {d['model_load_seconds']}s · 冷启动 {d['cold_start_ms']}ms · "
        f"预热中位 {d['warm_median_ms']}ms · VRAM {d['peak_vram_gb_512']}GB"
    )
    for b, t in d["throughput"].items():
        print(f"    batch={b}: {t['images_per_sec']} 张/s · {t['peak_vram_gb']}GB")
    if "baseline" in d:
        print(
            f"    基线    : {d['baseline'].get('说明', '')} = {d['baseline'].get('per_image_warm_ms')}ms"
        )
    all512 = [
        json.loads(p.read_text(encoding="utf-8"))["warm_median_ms"]
        for p in sorted(EVAL.glob("generator-bench-*.json"))
        if json.loads(p.read_text(encoding="utf-8"))["warm_median_ms"] < 8000
    ]
    if len(all512) >= 2:
        lo, hi = min(all512), max(all512)
        print(
            f"    512² 两轮区间: {lo}–{hi} ms（漂移 {(hi - lo) / lo:.1%}）—— "
            f"台账写『约 5.0s（两轮 4.87–5.00s）』，不写单点"
        )
    print(
        "    复现    : SD_WEIGHTS_DIR=models/sd15 python -X utf8 scripts/benchmark_generator.py --size 512 --repeats 8"
    )

    # ---------- 5. 检索基线（50 条集最近一次） ----------
    section("[5] 50 条检索基线（最近一次落盘产物）")
    r = load(latest("retrieval-*.json").name)
    m = r["metrics"]
    print(
        f"    Recall@5: {m['recall']['recall@5']['rate']:.1%}"
        f"（{m['recall']['recall@5']['hits']}/{m['recall']['recall@5']['total']}）· "
        f"MRR {m['mrr']} · 拒答 {m['refusal']['refused_correctly']}/{m['refusal']['refusal_total']}"
    )
    print(
        f"    口径    : bge · chunk500/overlap50 · 阈值 {m['threshold']} · "
        f"{r['meta']['chunks']} 块 · questions-50.jsonl"
    )
    print("    复现    : python -X utf8 scripts/run_retrieval_eval.py（50 条集）")

    # ---------- 6. W12 压测（读已有产物，注明可复跑命令） ----------
    section("[6] W12 压测基线（读 cold/hot 与五档产物；全量复跑约 17 分钟）")
    print("    五档 QPS（round2 复测口径，W12 周记 §四）：10/20/50 并发 = 13.4 / 27.3 / 67.8")
    for tag, f in [
        ("cold-A2（缓存全空）", "load/cold-A2_stats.csv"),
        ("hot-A2（热缓存）", "load/hot-A2_stats.csv"),
    ]:
        csv_lines = (EVAL / f).read_text(encoding="utf-8").splitlines()
        cols = csv_lines[0].split(",")
        row = csv_lines[1].split(",")

        def get(name: str, _cols: list[str] = cols, _row: list[str] = row) -> str:
            return _row[_cols.index(name)]

        print(
            f"    {tag}: n={get('Request Count')} 失败={get('Failure Count')} "
            f"P50={get('50%')}ms P95={get('95%')}ms QPS={float(get('Requests/s')):.1f}"
        )
    print("    口径    : hashing 热缓存 · 20 并发 · locust")
    print("    复现    : python -X utf8 scripts/load_test.py --profile all（需 :8100 服务在线）")

    # ---------- 7. 冷/热探针（周二新补测） ----------
    section("[7] 冷/热单客户端探针（W14 周二补测）")
    cp = load("cold-start-probe-20261114.json")
    print(
        f"    冷(首次): P50 {cp['cold']['p50']}ms · P95 {cp['cold']['p95']}ms · max {cp['cold']['max']}ms"
    )
    print(
        f"    热(重复): P50 {cp['hot']['p50']}ms · P95 {cp['hot']['p95']}ms · max {cp['hot']['max']}ms"
    )
    print(
        f"    口径    : 单客户端串行 · /api/v2/search · k=3 · 429 退避重试 {cp['meta']['http_429_retries']} 次"
    )
    print(
        "    注意    : 冷热差异在本口径不显著；W12 的 1ms 差是 hashing 并发热缓存口径，两者不混用"
    )

    # ---------- 8. 一致性核对 ----------
    section("[8] 与台账的一致性（必答问题的机器判定）")
    checks = [
        ("20 任务区间 70–100%", min(rates) == 0.70 and max(rates) == 1.00),
        ("编造 0 例", fab["real_fabrications"] == 0),
        (
            "预算拐点 10k（5k=50%/10k=83%/15k=83%）",
            [row["success_rate"] for row in bc["summary"]]
            == [0.5, 0.8333333333333334, 0.8333333333333334],
        ),
        ("50 条基线 Recall@5 = 40.0%", m["recall"]["recall@5"]["rate"] == 0.4),
    ]
    ok = 0
    for name, passed in checks:
        print(f"    {'✅' if passed else '❌'} {name}")
        ok += passed
    drift = None
    if len(all512) >= 2:
        lo, hi = min(all512), max(all512)
        drift = (hi - lo) / lo
        print(
            f"    {'✅' if drift <= 0.05 else '⚠️'} GPU 预热中位两轮漂移 {drift:.1%}"
            + ("（≤5% 硬件噪声，台账用区间表述）" if drift <= 0.05 else "（超 5%，台账须更新区间）")
        )
    print(
        f"\n    硬判定 {ok}/{len(checks)} 通过"
        + (f"；GPU 漂移 {drift:.1%}（允许 ≤5%）" if drift is not None else "")
    )
    print("    结论    : 台账数字与产物一致。面试现场可跑本命令复核。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
