"""质检 Agent 的统一注册表（W9 周六：主循环联调的接线层）。

**这一层存在的唯一理由：让 tools schema 与 executor 同源。**

周四的 P0（schema 四件套 vs 注册表旧两件）证明：凡是「给模型看的东西」
和「实际执行的东西」必须同一个地方出。本模块把周六要用的四个工具的
schema、执行函数、描述绑成一个常量，主循环只从这里拿，不允许散装拼。

工具阵容（周三定版的质检域 + 周四/周五的真实现）：

| 工具 | 实现 | 副作用 |
| --- | --- | --- |
| `retrieve_spec_chunks` | real_tools（RAG 真管线） | 无（只读） |
| `query_metrics_sql` | real_tools（SQLite mode=ro） | 无（只读） |
| `generate_report` | side_effect_tools（落盘） | **有**（写文件） |
| `notify` | side_effect_tools（HTTP + 幂等） | **有**（网络投递） |

`async` 执行器统一签名为 `(raw_args: str) -> dict`（返回统一信封），
直接对接 `run_react_loop(executor=...)`。
"""

from __future__ import annotations

import json
from typing import Any, Final

from pydantic import BaseModel, Field, ValidationError

from app.core.llm.real_tools import (
    QUERY_METRICS_SQL_DESCRIPTION,
    RETRIEVE_DESCRIPTION,
    execute_retrieve_tool,
    execute_sql_tool,
)
from app.core.llm.side_effect_tools import (
    GENERATE_REPORT_FILE_DESCRIPTION,
    NOTIFY_HTTP_DESCRIPTION,
    execute_notify_http_tool,
    execute_report_tool,
)
from app.core.llm.tools import ToolSpec

# ============================================================ schema（给模型看）


class RetrieveArgs(BaseModel):
    """`retrieve_spec_chunks` 的入参（复用周四定义的字段约束）。"""

    query: str = Field(
        min_length=2, max_length=120, description="检索词，用指标名或关键词，例如「抗压强度 判定」"
    )
    k: int = Field(default=3, ge=1, le=8, description="返回最相关的 k 条条款，默认 3，最多 8")


class QueryMetricsArgs(BaseModel):
    """`query_metrics_sql` 的入参（白名单在类型层）。"""

    table: str = Field(description="要查询的表（白名单）：inspections=检测记录；batches=批次台账")
    columns: list[str] = Field(
        default_factory=list,
        description="要返回的列（可选）。inspections：batch_id, metric, value, unit, measured_at；batches：batch_id, model, produced_at",
    )
    where_batch_id: str = Field(
        default="",
        max_length=32,
        description="按批次号过滤，如「B20241129-03」。留空返回全表（受 LIMIT 限制）",
    )
    where_metric: str = Field(
        default="", max_length=32, description="按指标名过滤（仅 inspections 有效），如「抗压强度」"
    )
    limit: int = Field(default=20, ge=1, le=100, description="返回行数上限，默认 20，最大 100")


class GenerateReportArgs(BaseModel):
    """`generate_report` 的入参（sections 的 content 是 Markdown 正文）。"""

    task_id: str = Field(
        min_length=4, max_length=64, description="任务标识，报告文件名主键，如「qa-B20241129-03」"
    )
    title: str = Field(min_length=1, max_length=80, description="报告标题")
    sections: list[dict[str, str]] = Field(
        min_length=1,
        max_length=12,
        description="报告小节 [{heading, content}]；**数值必须来自前面工具的返回，不要编造数据**",
    )


class NotifyArgs(BaseModel):
    """`notify` 的入参（channel 白名单 + 幂等键）。"""

    channel: str = Field(
        description="通知渠道（白名单）：team=生产班组；supervisor=质检主管；archive=仅归档"
    )
    message: str = Field(
        min_length=1, max_length=2000, description="通知正文，说清结论和接收方要做什么"
    )
    idempotency_key: str = Field(
        min_length=6,
        max_length=64,
        description="幂等键，同一事件复用同一键（如 batch-B20241129-03-fail）；重试不要换新键",
    )


def _spec(name: str, description: str, args_model: type[BaseModel]) -> ToolSpec:
    """借用 W2 的 ToolSpec 生成 OpenAI schema（args_model 必须有 model_json_schema）。

    W2 的 ToolSpec.func 字段我们不用（执行走统一信封异步函数），
    传 None 占位 —— schema 生成只依赖 name/description/args_model。
    """
    return ToolSpec(name=name, description=description, args_model=args_model, func=None)  # type: ignore[arg-type]


AGENT_TOOL_SPECS: Final[tuple[ToolSpec, ...]] = (
    _spec("retrieve_spec_chunks", RETRIEVE_DESCRIPTION, RetrieveArgs),
    _spec("query_metrics_sql", QUERY_METRICS_SQL_DESCRIPTION, QueryMetricsArgs),
    _spec("generate_report", GENERATE_REPORT_FILE_DESCRIPTION, GenerateReportArgs),
    _spec("notify", NOTIFY_HTTP_DESCRIPTION, NotifyArgs),
)

#: 主循环直接用的 tools schema。
AGENT_TOOLS_SCHEMA: Final[list[dict[str, Any]]] = [
    spec.to_openai_schema() for spec in AGENT_TOOL_SPECS
]

AGENT_TOOL_NAMES: Final[tuple[str, ...]] = tuple(spec.name for spec in AGENT_TOOL_SPECS)

# ============================================================ executor（实际执行）


async def execute_agent_tool(name: str, raw_arguments: str) -> dict[str, Any]:
    """统一执行入口：按名字分发到四个工具的统一信封实现。

    未知工具返回 unknown_tool（与 W2 行为一致），错误是数据不是异常。
    **这是主循环 executor 参数的唯一合法取值** —— schema 和它出自同一模块，
    从根上杜绝周四那种「模型看到的工具」与「实际执行的工具」漂移。
    """
    if name not in AGENT_TOOL_NAMES:
        return {
            "ok": False,
            "tool": name,
            "error_type": "unknown_tool",
            "error": {
                "message": f"没有名为 {name!r} 的工具",
                "available_tools": list(AGENT_TOOL_NAMES),
            },
        }
    if name == "retrieve_spec_chunks":
        return await execute_retrieve_tool(raw_arguments)
    if name == "query_metrics_sql":
        return await execute_sql_tool(raw_arguments)
    if name == "generate_report":
        return await execute_report_tool(raw_arguments)
    return await execute_notify_http_tool(raw_arguments)


__all__ = [
    "AGENT_TOOL_NAMES",
    "AGENT_TOOL_SPECS",
    "AGENT_TOOLS_SCHEMA",
    "GenerateReportArgs",
    "NotifyArgs",
    "QueryMetricsArgs",
    "RetrieveArgs",
    "execute_agent_tool",
]


def _unused_json_guard() -> None:  # pragma: no cover
    """保留 json/ValidationError 的引用说明：信封实现里各自做解析。
    （真实解析在 real_tools / side_effect_tools 的信封函数里。）"""
    _ = json, ValidationError
