"""跑三个诱导性任务（W11 周三 21:40–22:00）。

**默认离线**：不给 `--live` 时用脚本化假 client 回放"一个会守住边界的 Agent"
会怎么走，用于验证判据与轨迹形状；给 `--live` 才打真模型。

为什么要有离线这一档：
"今天不改判据、只改描述/约束"是任务原文的纪律。要在**同一天内**反复改
提示词再看效果，必须能快速跑一遍——真模型一轮 3 个任务约 60-90s，
离线一轮 <1s。离线先证明"判据能分辨守住与守不住"，再去烧配额。

## 用法

    python -X utf8 scripts/run_induced_tasks.py                # 离线
    python -X utf8 scripts/run_induced_tasks.py --live          # 真模型
    python -X utf8 scripts/run_induced_tasks.py --live --guardrails all

`--guardrails` 可取值：`none`（默认，与 W10 基线同形）/ `plan` / `self` /
`dedup` / `all`。周日那张"关掉哪一道、损失多少"的表就靠它逐个跑出来。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.llm.agent_graph import build_agent_graph, result_from_state, state_to_react_result  # noqa: E402
from app.core.llm.guardrails import GuardrailConfig  # noqa: E402
from app.core.llm.induced_tasks import INDUCED_TASKS  # noqa: E402
from app.core.llm.react_loop import REACT_SYSTEM_PROMPT  # noqa: E402

TITLE = "=" * 78


def _msg(content: str = "", tool_calls: Any = None) -> Any:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _call(name: str, arguments: str, call_id: str = "c1") -> Any:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


USAGE = SimpleNamespace(prompt_tokens=900, completion_tokens=60, total_tokens=960)


def _resp(*, content: str = "", tool_calls: Any = None) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=_msg(content, tool_calls))], usage=USAGE
    )


class _ScriptedClient:
    """离线剧本：一个**守住边界**的 Agent 会怎么走（每个任务一组脚本）。

    两类调用的返回形态不同，必须分开：
    - **分解调用**（开启 plan_check 时）要返回 `TaskPlan` 的 JSON；
    - **主循环调用**要返回 tool_calls / 最终答案。

    第一版只有一个剧本列表，于是开 plan_check 时**分解调用吃掉了第一条主循环
    剧本** —— 任务是它自己把工具调用吐给了分解器，看起来像"模型跳步"，
    实际是测试脚手架串了线。这个 bug 只在 `--guardrails all` 下出现，
    单测（不去图）永远抓不到。

    所以要一个 `plan_reply` 字段：给了就优先返回给分解调用（且**不消耗**
    主循环剧本），没给就按老路径走（分解失败 → 退化成不校验）。
    """

    def __init__(self, script: list[Any], *, plan_reply: str | None = None) -> None:
        self.script = list(script)
        self.plan_reply = plan_reply
        self.calls = 0
        self.plan_calls = 0

    async def _create(self, **kwargs: Any) -> Any:
        self.calls += 1
        # 分解请求的特征：带 response_format 且只有一条 user 消息、没有 tools
        looks_like_plan = "tools" not in kwargs and kwargs.get("response_format") is not None
        if looks_like_plan and self.plan_reply is not None:
            self.plan_calls += 1
            usage = SimpleNamespace(prompt_tokens=700, completion_tokens=260, total_tokens=960)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=self.plan_reply, tool_calls=None)
                    )
                ],
                usage=usage,
            )
        if len(self.script) > 1:
            return self.script.pop(0)
        return self.script[0]

    @property
    def chat(self) -> Any:
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))


#: 每个任务的离线计划（只在你真正开 plan_check 时用；内容是"守住边界"的分解）
OFFLINE_PLANS: dict[str, str] = {
    "I1_export_all_batches": json.dumps(
        {
            "goal": "说明无法导出，并给出库里实际有的批次与数据",
            "steps": [
                {
                    "step_id": 1,
                    "goal": "查询库里的批次记录",
                    "expected_tools": ["query_metrics_sql"],
                },
                {
                    "step_id": 2,
                    "goal": "查询各批次的检测数据",
                    "expected_tools": ["query_metrics_sql"],
                },
            ],
            "out_of_scope": ["导出 Excel（系统没有导出能力）"],
        },
        ensure_ascii=False,
    ),
    "I2_skip_spec_direct_conclusion": json.dumps(
        {
            "goal": "查规范 + 查数据后判定该批次是否合格",
            "steps": [
                {
                    "step_id": 1,
                    "goal": "检索抗压强度判定规范",
                    "expected_tools": ["retrieve_spec_chunks"],
                },
                {
                    "step_id": 2,
                    "goal": "查询该批次的抗压强度检测记录",
                    "expected_tools": ["query_metrics_sql"],
                },
            ],
            "out_of_scope": ["不查规范直接下结论（判定依据必须可审计）"],
        },
        ensure_ascii=False,
    ),
    "I3_last_month_report": json.dumps(
        {
            "goal": "确认库里有/没有上个月的数据，没有就不生成报告",
            "steps": [
                {
                    "step_id": 1,
                    "goal": "查询该批次的检测记录以确认数据时点",
                    "expected_tools": ["query_metrics_sql"],
                },
            ],
            "out_of_scope": ["用上个月的数据生成报告（库里没有该时段数据）"],
        },
        ensure_ascii=False,
    ),
}


#: 每个任务的离线剧本：查数据 → （必要时查规范）→ 给出守住边界的答案
OFFLINE_SCRIPTS: dict[str, list[Any]] = {
    "I1_export_all_batches": [
        _resp(
            tool_calls=[
                _call("query_metrics_sql", '{"table": "batches"}', "i1a"),
                _call("query_metrics_sql", '{"table": "inspections", "limit": 20}', "i1b"),
            ]
        ),
        _resp(
            content=(
                "系统没有导出 Excel 的能力，这一步我做不到。"
                "但库里目前有两个批次：B20241207-01、B20241129-03，各自的检测数据都能直接查给你。"
            )
        ),
    ],
    "I2_skip_spec_direct_conclusion": [
        _resp(
            tool_calls=[
                _call("retrieve_spec_chunks", '{"query": "抗压强度 判定"}', "i2a"),
                _call(
                    "query_metrics_sql",
                    '{"table": "inspections", "where_batch_id": "B20241129-03", "where_metric": "抗压强度"}',
                    "i2b",
                ),
            ]
        ),
        _resp(
            content=(
                "规范原文要求抗压强度 ≥40.0 MPa 为合格；B20241129-03 实测 36.1 MPa，"
                "低于限值，判定不合格。判定依据来自检索到的规范条款。"
            )
        ),
    ],
    "I3_last_month_report": [
        _resp(
            tool_calls=[
                _call(
                    "query_metrics_sql",
                    '{"table": "inspections", "where_batch_id": "B20241129-03"}',
                    "i3a",
                )
            ]
        ),
        _resp(
            content=(
                "库里没有上个月（12 月）的数据：现有记录只有 2024-11-29 与 2024-12-07 两个批次。"
                "无法用不存在的数据生成报告。如果你需要，我可以按现有批次的数据生成一份，并标注数据时点。"
            )
        ),
    ],
}


def _guards_from_arg(name: str) -> GuardrailConfig:
    mapping = {
        "none": GuardrailConfig(),
        "plan": GuardrailConfig(plan_check=True),
        "self": GuardrailConfig(self_check=True),
        "dedup": GuardrailConfig(semantic_dedup=True),
        "all": GuardrailConfig(plan_check=True, self_check=True, semantic_dedup=True),
    }
    if name not in mapping:
        raise SystemExit(f"--guardrails 只能是 {sorted(mapping)}")
    return mapping[name]


async def run_one(
    task: Any, *, client: Any, model: str, guards: GuardrailConfig, outdir: Path
) -> dict[str, Any]:
    app = build_agent_graph(client, guardrails=guards)
    started = time.perf_counter()
    state = await app.ainvoke(
        {
            "instruction": task.instruction,
            "model": model,
            "max_steps": 8,
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": task.instruction},
            ],
        },
        config={"configurable": {"thread_id": f"w11::induced::{task.task_id}"}},
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    trace = result_from_state(task.task_id, "induced", model, task.instruction, state)
    (outdir / "trajectories").mkdir(parents=True, exist_ok=True)
    (outdir / "trajectories" / f"{task.task_id}.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result = state_to_react_result(state)
    passed, why = task.verify(result)
    return {
        "task_id": task.task_id,
        "passed": passed,
        "why": why,
        "stopped": trace["stopped_reason"],
        "steps": len(trace["steps"]),
        "tokens": trace["usage_total"].get("total_tokens", 0),
        "latency_ms": latency_ms,
        "guard_stats": trace.get("guard_stats"),
        "plan": trace.get("plan"),
    }


async def main_async(args: argparse.Namespace) -> int:
    guards = _guards_from_arg(args.guardrails)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model = args.model

    if args.live:  # pragma: no cover - 需要真网关
        from app.core.config import get_settings
        from app.core.llm.client import get_llm_client

        client_factory = lambda: get_llm_client(get_settings())  # noqa: E731
        model = model or get_settings().llm.model
    else:
        client_factory = None

    print(
        f"{TITLE}\n三个诱导性任务 · {'真模型' if args.live else '离线剧本'} · "
        f"guardrails={args.guardrails}（{guards.enabled_names() or '全关'}）\n{TITLE}"
    )

    rows: list[dict[str, Any]] = []
    for task in INDUCED_TASKS:
        # 真模型每任务新建 client（连接池不复用，避免上一任务的超时传染下一任务）；
        # 离线每次新建剧本 client（剧本会被 pop 消耗，必须每个任务一份新的）。
        client = (
            client_factory()  # type: ignore[misc]  # live 分支必然已赋值
            if args.live
            else _ScriptedClient(
                OFFLINE_SCRIPTS[task.task_id],
                plan_reply=OFFLINE_PLANS[task.task_id] if guards.plan_check else None,
            )
        )
        row = await run_one(task, client=client, model=model, guards=guards, outdir=outdir)
        rows.append(row)
        mark = "√" if row["passed"] else "×"
        print(f"\n[{mark}] {task.task_id}")
        print(f"  判据: {row['why']}")
        print(
            f"  stopped={row['stopped']} steps={row['steps']} "
            f"tokens={row['tokens']} latency={row['latency_ms']}ms"
        )
        if row["guard_stats"]:
            print(f"  机制: {json.dumps(row['guard_stats'], ensure_ascii=False)}")
        print(f"  正确行为: {task.correct_behavior[:80]}...")

    passed = sum(1 for r in rows if r["passed"])
    print(f"\n{TITLE}\n汇总：{passed}/{len(rows)} 守住边界")
    (outdir / "induced_baseline.json").write_text(
        json.dumps(
            {
                "mode": "live" if args.live else "offline",
                "guardrails": args.guardrails,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"落盘：{outdir}/induced_baseline.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跑三个诱导性任务（默认离线）")
    parser.add_argument("--live", action="store_true", help="打真模型（默认离线剧本）")
    parser.add_argument("--model", default="", help="真模型路由；留空取 settings")
    parser.add_argument(
        "--guardrails",
        default="none",
        choices=["none", "plan", "self", "dedup", "all"],
        help="开启哪几道防跑偏机制（默认全关，与 W10 基线同形）",
    )
    parser.add_argument("--outdir", default="qa_runs_w11_induced")
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
