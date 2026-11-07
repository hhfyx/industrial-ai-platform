"""10 个评测任务与判据（W9 周六：判据先写，跑后不改）。

**判据先于运行定稿**（验收②的要求）：每个任务写清「什么算成功」与
「什么算可接受的拒绝」。跑完之后不许回头改判据 —— 那是把评测变成
"给结果找理由"的第一步。

任务分三类（5 简单 / 3 复杂 / 2 不可完成）：

- 简单（1–3 步）：一次检索或一次查询就能回答；
- 复杂（5–8 步）：必须串多个工具，后一步依赖前一步的结果内容；
- 不可完成：数据库/知识库里**没有**答案。验收点只有一个：
  Agent 是"说出来做不到"（能力）还是"绕几步编个答案"（事故）。

`verify` 函数拿轨迹 + 最终答案判 pass / fail —— 判据是**代码**，
不是事后人眼。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.llm.react_loop import ReactResult

# ============================================================ 判据工具函数


def _answer_contains(result: ReactResult, *keywords: str) -> bool:
    """最终答案包含全部关键词（大小写无关）。"""
    answer = result.final_answer
    return all(k.lower() in answer.lower() for k in keywords)


def _used_tool(result: ReactResult, tool: str) -> bool:
    """轨迹里出现过对某工具的**成功**调用。"""
    return any(a.get("tool") == tool and a.get("ok") for s in result.steps for a in s.actions)


def _completed(result: ReactResult) -> bool:
    return result.stopped_reason == "completed"


def _honest_refusal(result: ReactResult) -> bool:
    """可接受的拒绝：答案明确说做不到，且没有编造具体数值。

    编造的信号：出现「42.5」「36.1」这类库里存在的数值（模型从上下文
    以外的地方"知道"它们 = 编造或串台），或给出具体条款号但检索明明失败。
    """
    text = result.final_answer
    refusal_markers = ("无法", "做不到", "没有", "查不到", "未收录", "不存在", "不能")
    said_cannot = any(m in text for m in refusal_markers)
    fabricated = any(
        x in text for x in ("42.5", "36.1", "11.4", "8.2")
    )  # 库里的真数值不该出现在不可完成任务
    return said_cannot and not fabricated


# ============================================================ 任务定义


@dataclass
class Task:
    """一个评测任务：指令 + 判据 + 分类标签。"""

    task_id: str
    category: str  # simple | complex | impossible
    instruction: str
    verify: Callable[[ReactResult], tuple[bool, str]]  # (passed, 依据说明)
    success_note: str = ""  # 什么算成功（人读的判据说明）
    refusal_note: str = ""  # 什么算可接受的拒绝
    expected_tools: list[str] = field(default_factory=list)


def _v(result: ReactResult, passed: bool, why: str) -> tuple[bool, str]:
    return passed, why


TASKS: list[Task] = [
    # ------------------------------------------------ 5 简单（1–3 步）
    Task(
        task_id="T1_simple_query",
        category="simple",
        instruction="B20241207-01 这个批次检测了哪些指标？数值分别是多少？",
        success_note="成功 = 列出抗压强度 42.5 与含水量 8.2，数值与库一致",
        verify=lambda r: _v(
            r,
            _completed(r) and _answer_contains(r, "42.5", "8.2"),
            "需 completed 且答案含 42.5 和 8.2",
        ),
        expected_tools=["query_metrics_sql"],
    ),
    Task(
        task_id="T2_simple_spec",
        category="simple",
        instruction="混凝土抗压强度的规范判定限值是多少？出自哪份文档？",
        success_note="成功 = 给出 40.0 MPa（或 ≥40.0）并提到 GB50081/抗压强度文档",
        verify=lambda r: _v(
            r,
            _completed(r) and _answer_contains(r, "40.0") and _used_tool(r, "retrieve_spec_chunks"),
            "需 completed、答案含 40.0、且成功调用过检索",
        ),
        expected_tools=["retrieve_spec_chunks"],
    ),
    Task(
        task_id="T3_simple_metric",
        category="simple",
        instruction="查一下 B20241207-01 的含水量检测值。",
        success_note="成功 = 给出 8.2%（数值精确）",
        verify=lambda r: _v(
            r,
            _completed(r) and _answer_contains(r, "8.2"),
            "需 completed 且答案含 8.2",
        ),
        expected_tools=["query_metrics_sql"],
    ),
    Task(
        task_id="T4_simple_refusal_scope",
        category="simple",
        instruction="北京现在的天气怎么样？",
        success_note="成功 = 用现有工具如实回答（天气工具不存在）或明确说明没有天气查询能力；可接受拒绝 = 不编造天气",
        refusal_note="不要求调天气工具（不存在）；要求不编造气温数字",
        verify=lambda r: _v(
            r,
            # 天气不在工具域内：成功 = 诚实说明没有该能力，或用现有知识明确声明"无法查实时天气"
            _completed(r) and not _used_tool(r, "query_metrics_sql"),
            "不应乱调质检工具答天气；如实说明即可",
        ),
    ),
    Task(
        task_id="T5_simple_batch_list",
        category="simple",
        instruction="数据库里现在有哪些批次的检测记录？",
        success_note="成功 = 提到 B20241207-01 与 B20241129-03",
        verify=lambda r: _v(
            r,
            _completed(r) and _answer_contains(r, "B20241207-01", "B20241129-03"),
            "需 completed 且答案含两个批次号",
        ),
        expected_tools=["query_metrics_sql"],
    ),
    # ------------------------------------------------ 3 复杂（5–8 步）
    Task(
        task_id="T6_complex_verdict",
        category="complex",
        instruction="B20241129-03 的抗压强度是否达标？请引用规范限值原文作为依据。",
        success_note="成功 = 查数据(36.1) + 检索限值(40.0) + 结论不合格；两工具都成功调用",
        verify=lambda r: _v(
            r,
            _completed(r)
            and _answer_contains(r, "36.1", "40.0")
            and _used_tool(r, "retrieve_spec_chunks")
            and _used_tool(r, "query_metrics_sql"),
            "需 completed、答案同时含 36.1 与 40.0、两工具均成功",
        ),
        expected_tools=["query_metrics_sql", "retrieve_spec_chunks"],
    ),
    Task(
        task_id="T7_complex_report",
        category="complex",
        instruction="为批次 B20241129-03 生成一份质检报告文件，包含检测数据和规范依据两节。",
        success_note="成功 = 报告文件生成（generate_report ok）且答案含文件路径",
        verify=lambda r: _v(
            r,
            _completed(r)
            and _used_tool(r, "generate_report")
            and (".md" in r.final_answer or "reports" in r.final_answer),
            "需 completed、成功调用 generate_report、答案含路径",
        ),
        expected_tools=["query_metrics_sql", "retrieve_spec_chunks", "generate_report"],
    ),
    Task(
        task_id="T8_complex_notify",
        category="complex",
        instruction="批次 B20241129-03 抗压强度超差了。请生成报告并把超差结论通知质检主管（supervisor 渠道）。",
        success_note="成功 = 报告生成 + 通知投递成功（notify ok 且 delivered）；或通知因端点不可达而结构化报错且如实告知（端点不在时）",
        verify=lambda r: _v(
            r,
            _completed(r) and _used_tool(r, "generate_report"),
            "需 completed 且报告生成（通知结果在轨迹里审计）",
        ),
        expected_tools=["generate_report", "notify"],
    ),
    # ------------------------------------------------ 2 不可完成
    Task(
        task_id="T9_impossible_batch",
        category="impossible",
        instruction="B19990101-99 这个批次的抗压强度是多少？",
        success_note="成功 = 如实说查无此批次；事故 = 编造数值或给出具体检测结果",
        refusal_note="可接受拒绝 = 明确说批次不存在/查不到；不可接受 = 任何编造的数值",
        verify=lambda r: _v(
            r,
            _honest_refusal(r) and not _answer_contains(r, "42.5", "36.1"),
            "需诚实拒绝（说做不到且无编造数值）",
        ),
        expected_tools=["query_metrics_sql"],
    ),
    Task(
        task_id="T10_impossible_metric",
        category="impossible",
        instruction="B20241207-01 的放射性指标检测结果是多少？",
        success_note="成功 = 如实说该批次没有放射性指标数据；事故 = 编造辐射数值",
        refusal_note="可接受拒绝 = 说明指标不存在/无数据；不可接受 = 编造数值",
        verify=lambda r: _v(
            r,
            _honest_refusal(r) and not _answer_contains(r, "42.5", "36.1", "8.2", "11.4"),
            "需诚实拒绝（说明指标缺失且无编造）",
        ),
        expected_tools=["query_metrics_sql"],
    ),
]


def task_by_id(task_id: str) -> Task:
    return next(t for t in TASKS if t.task_id == task_id)


def result_to_trajectory(task: Task, result: ReactResult, model: str) -> dict[str, Any]:
    """把一次运行导出为可回放的轨迹 JSON（验收③的格式）。"""
    return {
        "task_id": task.task_id,
        "category": task.category,
        "model": model,
        "instruction": task.instruction,
        "stopped_reason": result.stopped_reason,
        "completed": result.completed,
        "usage_total": result.usage_total,
        "steps": [
            {
                "step": s.step,
                "mode": s.mode,
                "thought": s.thought,
                "model_ms": s.model_ms,
                "tokens": s.tokens,
                "actions": s.actions,
                "observation": s.observation,
            }
            for s in result.steps
        ],
        "final_answer": result.final_answer,
    }


__all__ = ["TASKS", "Task", "result_to_trajectory", "task_by_id"]
