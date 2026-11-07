"""质检域的 4 个工具（W9 周三：工具设计 —— 本周最重要的软技能）。

任务域沿用周一 [`docs/agent-design.md`](../../../../docs/agent-design.md) 定版的
质检报告助手：查规范 → 查数据 → 生成报告 → 通知，每个环节一个工具。

**本模块的核心是"四要素描述模板"**（每个工具的 description 必须四项全有）：

1. **做什么** —— 一句话说清功能边界；
2. **什么时候用、什么时候不用** —— 不写"什么时候不用"，模型就会乱用；
   这一句对成功率的贡献大于 system prompt 里任何一句"你要谨慎"；
3. **参数含义 + 示例** —— 模型照着示例生成参数，比照着类型定义猜可靠；
4. **返回什么形状** —— 模型要知道拿到的东西长什么样才能继续推理。

**统一错误结构**（今晚 21:30–21:50 的验收项）：

    成功: {"ok": True,  "tool": name, "data": {...}}
    失败: {"ok": False, "tool": name,
           "error_type": invalid_arg | invalid_json | unknown_tool
                       | not_found | conflict | timeout | internal,
           "error": {"message": ..., ...}}

error_type 是**封闭枚举**而不是自由文本：模型对 "not_found" 的反应是
可预测的（换参数或如实告知），对 "啊这里出错了" 的反应是不可预测的。
七类各自对应模型的一种标准修正动作 —— 这就是错误结构统一的理由：
**错误分类学就是模型的"修正动作菜单"**。

| error_type | 含义 | 模型该做什么 |
| --- | --- | --- |
| `invalid_json` | 参数不是合法 JSON | 重新生成合法 JSON |
| `invalid_arg` | 参数缺字段/类型错/模板不在白名单 | 按错误里的 fields 提示改参数 |
| `unknown_tool` | 工具名不存在 | 换工具或直接回答 |
| `not_found` | 查询对象不存在（批次/条款） | 换关键词，或如实告知用户"查无此物" |
| `conflict` | 幂等冲突（重复通知） | 不要重试，任务实际已完成 |
| `timeout` | 执行超时被强制中断 | 决定重试还是放弃并如实告知 |
| `internal` | 未预期异常 | 如实告知用户"工具内部错误"，不要编造结果 |
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final, Literal

from pydantic import BaseModel, Field, ValidationError

from app.core.llm.tools import ToolExecutionError

# ============================================================ ① search_knowledge_base


class SearchKnowledgeBaseArgs(BaseModel):
    """`search_knowledge_base` 的入参。

    `k` 有上限（≤8）：不带上限的话模型可能要 top-50，
    返回塞进上下文就是observation 截断兜底 —— 在源头限住更便宜。
    """

    query: str = Field(
        min_length=2,
        max_length=120,
        description="检索词，用指标名或关键词，例如「抗压强度 判定」「含水量 限值」",
    )
    k: int = Field(default=3, ge=1, le=8, description="返回最相关的 k 条条款，默认 3")


SEARCH_KB_DESCRIPTION: Final[str] = (
    "在产品规范库里检索**质量规范条款原文**（判定标准、限值、检测方法）。"
    "什么时候用：当需要某项指标的判定依据、限值数字、规范编号时，"
    "例如「抗压强度超标了，规范里限值是多少」。"
    "什么时候不用：不要用它查某一批次的检测数据（那是 query_metrics 的事）；"
    "不要在不知道要查什么指标时盲搜。"
    "参数：query 是指标名/关键词组合（如「抗压强度 判定」），k 是返回条数（默认 3，最多 8）。"
    "返回：{clause_id, title, text, metric} 列表，按相关度排序；"
    "查不到时返回 not_found 错误而不是空列表。"
)


# 假规范库：条款 id -> 内容。真实系统这里是向量库/全文索引。
_SPEC_LIBRARY: Final[dict[str, dict[str, str]]] = {
    "GB/T 50081-2019#6.5": {
        "title": "混凝土物理力学性能试验方法——抗压强度",
        "metric": "抗压强度",
        "text": "试件尺寸 100mm×100mm×100mm，强度等级 C40 的单方判定限值 ≥40.0 MPa；"
        "三个试件的算术平均值作为该组强度值，最小值不低于限值的 85%。",
    },
    "GB/T 50082-2009#4.2": {
        "title": "普通混凝土长期性能和耐久性能试验方法——含水量",
        "metric": "含水量",
        "text": "烘干法测定：试件在 105±5℃ 烘至恒重，含水率 ≤10.0% 判为合格；"
        "含水率超过限值时应复测一次，复测仍超差判为不合格。",
    },
    "QC-VIS-01": {
        "title": "外观质量检验规程",
        "metric": "外观",
        "text": "目测检查表面蜂窝、麻面、裂缝；允许个别气孔，"
        "出现贯穿性裂缝直接判不合格，无需复测。",
    },
}

_METRIC_KEYWORDS: Final[dict[str, str]] = {
    "抗压强度": "抗压强度",
    "强度": "抗压强度",
    "含水量": "含水量",
    "含水率": "含水量",
    "外观": "外观",
    "裂缝": "外观",
}


def search_knowledge_base(query: str, k: int = 3) -> dict[str, Any]:
    """在假规范库里按指标关键词检索，返回最相关的条款。查不到抛 not_found。

    打分只看 **query 里的关键词**映射到哪个指标 —— 第一版把
    `keyword in doc["text"]` 也算命中，结果每篇文档都能在自己文本里
    找到自己的词，全库永远非空，not_found 永远不触发。
    """
    scored: list[tuple[int, str]] = []
    for clause_id, doc in _SPEC_LIBRARY.items():
        score = 0
        if doc["metric"] in query:
            score += 2  # query 直接点名了指标
        for keyword, metric in _METRIC_KEYWORDS.items():
            if metric == doc["metric"] and keyword in query:
                score += 1  # query 里的同义/相关词命中该指标
        if score > 0:
            scored.append((score, clause_id))
    scored.sort(reverse=True)
    if not scored:
        available = "、".join(sorted({doc["metric"] for doc in _SPEC_LIBRARY.values()}))
        raise ToolExecutionError(
            f"规范库中检索不到与「{query}」相关的条款；当前收录的指标：{available}",
            kind="not_found",
        )
    hits = [{"clause_id": cid, **_SPEC_LIBRARY[cid]} for _, cid in scored[: max(1, min(k, 8))]]
    return {"query": query, "hits": hits, "total": len(hits)}


# ============================================================ ② query_metrics（只读）


class QueryMetricsArgs(BaseModel):
    """`query_metrics` 的入参。

    `sql_template` 是 **Literal 白名单**：模型永远不能写自由 SQL，
    只能选预置模板。这是"只读"承诺的落地方式 —— 注入面在类型层面被封死。
    """

    sql_template: Literal["inspections_by_batch", "metric_by_batch"] = Field(
        description="预置查询模板（白名单）："
        "inspections_by_batch=按批次号查全部检测项；"
        "metric_by_batch=按批次号+指标名查单项数值"
    )
    params: dict[str, str] = Field(
        default_factory=dict,
        description="模板参数。inspections_by_batch 需要 {batch_id}；"
        "metric_by_batch 需要 {batch_id, metric}。"
        '示例：{"batch_id": "B20241129-03", "metric": "抗压强度"}',
    )


QUERY_METRICS_DESCRIPTION: Final[str] = (
    "查询**质检检测数据表**（某批次的实测数值），只读。"
    "什么时候用：当问题涉及具体批次号的检测数值、是否超差时，"
    "例如「B20241129-03 的抗压强度是多少」。"
    "什么时候不用：不要用它查规范条文和限值（那是 search_knowledge_base 的事）；"
    "没有具体批次号时不要调用（先向用户要批次号）。"
    "参数：sql_template 只能选两个预置模板（inspections_by_batch / metric_by_batch），"
    "params 里传 batch_id（格式如 B20241129-03）和可选的 metric。"
    "返回：rows 列表，每行含 metric、value、unit、limit、verdict（超差行才有 verdict）；"
    "批次不存在时返回 not_found；模板名错误或参数缺失返回 invalid_arg。"
)


# 假检测数据表。verdict 出现即超差（数据侧预判定，只描述事实，不做结论）。
_BATCH_DATA: Final[dict[str, list[dict[str, Any]]]] = {
    "B20241207-01": [
        {"metric": "抗压强度", "value": 42.5, "unit": "MPa", "limit": "≥40.0"},
        {"metric": "含水量", "value": 8.2, "unit": "%", "limit": "≤10.0"},
    ],
    "B20241129-03": [
        {"metric": "抗压强度", "value": 36.1, "unit": "MPa", "limit": "≥40.0", "verdict": "超差"},
        {"metric": "含水量", "value": 11.4, "unit": "%", "limit": "≤10.0", "verdict": "超差"},
    ],
}

#: 每个模板必填的参数键 —— 白名单的第二道闸（模板对 ≠ 参数全）。
_TEMPLATE_REQUIRED_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "inspections_by_batch": ("batch_id",),
    "metric_by_batch": ("batch_id", "metric"),
}


def query_metrics(sql_template: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    """按白名单模板查假数据表。只读：没有任何写路径。"""
    params = params or {}
    required = _TEMPLATE_REQUIRED_KEYS.get(sql_template)
    if required is None:
        raise ToolExecutionError(
            f"未知查询模板 {sql_template!r}；可用模板：{sorted(_TEMPLATE_REQUIRED_KEYS)}",
            kind="invalid_arg",
        )
    missing = [key for key in required if not params.get(key)]
    if missing:
        raise ToolExecutionError(
            f"模板 {sql_template} 缺少必填参数：{missing}",
            kind="invalid_arg",
        )
    batch_id = params["batch_id"]
    rows = _BATCH_DATA.get(batch_id)
    if rows is None:
        raise ToolExecutionError(
            f"查无批次 {batch_id!r}；数据表中已有的批次：{sorted(_BATCH_DATA)}",
            kind="not_found",
        )
    if sql_template == "metric_by_batch":
        metric = params["metric"]
        matched = [row for row in rows if row["metric"] == metric]
        if not matched:
            available = "、".join(row["metric"] for row in rows)
            raise ToolExecutionError(
                f"批次 {batch_id} 没有指标 {metric!r} 的检测数据；该批次已有指标：{available}",
                kind="not_found",
            )
        rows = matched
    return {"batch_id": batch_id, "rows": rows, "count": len(rows)}


# ============================================================ ③ generate_report


class ReportSection(BaseModel):
    """报告的一节：标题 + 正文。正文最小长度防模型生成空节。"""

    heading: str = Field(min_length=1, max_length=60, description="小节标题，如「检测结果比对」")
    body: str = Field(min_length=1, max_length=4000, description="正文，Markdown 文本")


class GenerateReportArgs(BaseModel):
    """`generate_report` 的入参。sections 至少一节。"""

    title: str = Field(
        min_length=1, max_length=80, description="报告标题，如「批次 B20241129-03 质检报告」"
    )
    sections: list[ReportSection] = Field(
        min_length=1,
        max_length=12,
        description="报告小节列表，每节 {heading, body}；"
        '示例：[{"heading": "结论", "body": "抗压强度超差，判定不合格"}]',
    )


GENERATE_REPORT_DESCRIPTION: Final[str] = (
    "把已整理好的小节渲染成结构化质检报告（Markdown）。"
    "什么时候用：当判定结论已经形成、各小节的正文内容已经明确时。"
    "什么时候不用：不要用它替你做判断（判定逻辑在你的推理里，不在工具里）；"
    "结论未定时不要先渲染报告。"
    "参数：title 是报告标题；sections 是 [{heading, body}] 列表，body 为 Markdown 正文。"
    "返回：title、markdown（完整报告文本）、sections（节数）；"
    "sections 为空或正文为空返回 invalid_arg。"
)


def generate_report(title: str, sections: list[dict[str, str]]) -> dict[str, Any]:
    """渲染 Markdown 报告。只做排版，不做判断 —— 判定留在 Agent 的推理层。"""
    if not sections:
        raise ToolExecutionError("sections 不能为空：至少要有一节内容", kind="invalid_arg")
    parts = [f"# {title}", ""]
    for idx, section in enumerate(sections, start=1):
        heading = str(section.get("heading", "")).strip()
        body = str(section.get("body", "")).strip()
        if not heading or not body:
            raise ToolExecutionError(f"第 {idx} 节的 heading 或 body 为空", kind="invalid_arg")
        parts.append(f"## {idx}. {heading}")
        parts.append("")
        parts.append(body)
        parts.append("")
    markdown = "\n".join(parts).rstrip()
    return {"title": title, "sections": len(sections), "markdown": markdown}


# ============================================================ ④ notify


class NotifyArgs(BaseModel):
    """`notify` 的入参。

    `channel` 用 Literal 白名单收死：通知发给谁是**合规问题**，
    不能让模型自由填邮箱/电话 —— 它会幻觉出一个看起来合理的地址。
    """

    channel: Literal["team", "supervisor", "archive"] = Field(
        description="通知渠道（白名单）：team=生产班组；supervisor=质检主管；archive=仅归档不发人"
    )
    message: str = Field(
        min_length=1, max_length=2000, description="通知正文，一句话说清结论和需要接收方做什么"
    )
    idempotency_key: str = Field(
        min_length=6,
        max_length=64,
        description="幂等键，同一事件必须复用同一个键，"
        "例如「batch-B20241129-03-fail」。重试时不要换新键",
    )


NOTIFY_DESCRIPTION: Final[str] = (
    "把质检结论通知发给指定渠道（带幂等去重）。"
    "什么时候用：报告生成之后；渠道必须跟结论匹配 —— 合格/归档发 team 或 archive，"
    "超差/不合格必须发 supervisor。"
    "什么时候不用：报告还没生成就不要发通知（通知正文要引用报告结论）；"
    "不确定该发哪个渠道时不要猜，宁可如实告知用户。"
    "参数：channel 只能是 team / supervisor / archive；message 是通知正文；"
    "idempotency_key 是幂等键（同一事件复用同一键，如「batch-B20241129-03-fail」）。"
    "返回：delivered（是否送达）、channel、duplicate（是否重复拦截）；"
    "同一 idempotency_key 重复发送时返回 conflict + duplicate=true，此时不需要重试。"
)


#: 幂等账本：key -> 首次投递结果。模块级 —— 工具的副作用必须真实存在，
#: 幂等才不是文档里的空话。测试用 `reset_notify_log()` 清空。
_NOTIFY_LOG: Final[dict[str, dict[str, Any]]] = {}


def reset_notify_log() -> None:
    """清空幂等账本。仅供测试隔离使用。"""
    _NOTIFY_LOG.clear()


def notify(channel: str, message: str, idempotency_key: str) -> dict[str, Any]:
    """投递通知，按 idempotency_key 去重。

    幂等的意义：Agent 循环里"通知"是最不能重复发生的副作用 ——
    模型重试一次，主管就收到两遍重复告警。工具侧用幂等键把
    "重试"变成"确认已完成"，错误类型用 conflict（不是 error），
    因为重复投递不是故障，是**状态**。
    """
    if channel not in ("team", "supervisor", "archive"):
        raise ToolExecutionError(
            f"未知通知渠道 {channel!r}；可用渠道：team / supervisor / archive",
            kind="invalid_arg",
        )
    if idempotency_key in _NOTIFY_LOG:
        first = _NOTIFY_LOG[idempotency_key]
        raise ToolExecutionError(
            f"idempotency_key {idempotency_key!r} 已投递过（首次 delivered={first['delivered']}），"
            "本次为重复投递，已拦截",
            kind="conflict",
        )
    delivered = channel != "archive"  # archive 只归档，不发人
    result: dict[str, Any] = {
        "delivered": delivered,
        "channel": channel,
        "duplicate": False,
        "message_preview": message[:60],
    }
    _NOTIFY_LOG[idempotency_key] = result
    return result


# ============================================================ 注册表 + 统一执行


@dataclass(frozen=True, slots=True)
class QAToolSpec:
    """质检工具的规格：名字、描述、Pydantic 校验模型、执行函数。

    与 W2 `ToolSpec` 的字段相同（复用 `to_openai_schema` 生成 schema），
    单独再包一层是因为本模块的**执行信封**不同（统一 error_type 枚举）。
    """

    name: str
    description: str
    args_model: type[BaseModel]
    func: Any

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


QA_TOOL_SPECS: Final[tuple[QAToolSpec, ...]] = (
    QAToolSpec(
        "search_knowledge_base",
        SEARCH_KB_DESCRIPTION,
        SearchKnowledgeBaseArgs,
        search_knowledge_base,
    ),
    QAToolSpec("query_metrics", QUERY_METRICS_DESCRIPTION, QueryMetricsArgs, query_metrics),
    QAToolSpec("generate_report", GENERATE_REPORT_DESCRIPTION, GenerateReportArgs, generate_report),
    QAToolSpec("notify", NOTIFY_DESCRIPTION, NotifyArgs, notify),
)

QA_TOOL_REGISTRY: Final[dict[str, QAToolSpec]] = {spec.name: spec for spec in QA_TOOL_SPECS}

QA_TOOLS_SCHEMA: Final[list[dict[str, Any]]] = [spec.to_openai_schema() for spec in QA_TOOL_SPECS]

#: 统一错误结构里 error_type 的封闭集合。超出它的 kind 一律降级为 internal。
ERROR_TYPES: Final[frozenset[str]] = frozenset(
    {"invalid_json", "invalid_arg", "unknown_tool", "not_found", "conflict", "timeout", "internal"}
)


def _normalize_error_type(kind: str | None) -> str:
    """把工具抛出的 kind 规范到封闭枚举里。

    映射表刻意显式列出 —— 新增错误类型时必须先想清楚"模型该做什么"，
    想不清楚就说明这个分类不该存在，落回 internal。
    """
    if kind in ERROR_TYPES:
        return kind
    # W2 tools.py 的旧 kind 名映射到统一枚举
    legacy_map = {
        "execution_error": "internal",
        "city_not_found": "not_found",
        "invalid_arguments": "invalid_arg",
        "invalid_arguments_json": "invalid_json",
        "unknown_tool": "unknown_tool",
    }
    return legacy_map.get(kind or "", "internal")


def execute_qa_tool(name: str, raw_arguments: str) -> dict[str, Any]:
    """统一信封的工具执行：**永不抛异常**，错误是数据不是异常。

    与 W2 `execute_tool` 的差别只有信封形状（今晚的验收项）：
    失败统一为 `{"ok": False, "tool", "error_type", "error"}`，
    error_type 是封闭枚举。执行链路与 W2 相同：JSON 解析 → Pydantic 校验
    → 业务执行，三层各自的失败类型不同（模型的修正动作不同）。
    """
    spec = QA_TOOL_REGISTRY.get(name)
    if spec is None:
        return {
            "ok": False,
            "tool": name,
            "error_type": "unknown_tool",
            "error": {
                "message": f"没有名为 {name!r} 的工具",
                "available_tools": sorted(QA_TOOL_REGISTRY),
            },
        }
    try:
        parsed = json.loads(raw_arguments) if raw_arguments.strip() else {}
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "tool": name,
            "error_type": "invalid_json",
            "error": {
                "message": f"参数不是合法 JSON：{exc.msg}（位置 {exc.pos}）",
                "raw": raw_arguments[:200],
            },
        }
    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "tool": name,
            "error_type": "invalid_arg",
            "error": {"message": f"参数必须是 JSON 对象，收到 {type(parsed).__name__}"},
        }
    try:
        args = spec.args_model.model_validate(parsed)
    except ValidationError as exc:
        fields = [
            {
                "field": ".".join(str(p) for p in err["loc"]),
                "reason": err["msg"],
                "got": err.get("input"),
            }
            for err in exc.errors()
        ]
        return {
            "ok": False,
            "tool": name,
            "error_type": "invalid_arg",
            "error": {"message": "参数不符合工具签名", "fields": fields},
        }
    try:
        data = spec.func(**args.model_dump())
    except ToolExecutionError as exc:
        return {
            "ok": False,
            "tool": name,
            "error_type": _normalize_error_type(exc.kind),
            "error": {"message": exc.message},
        }
    except Exception as exc:  # noqa: BLE001 —— 未预期异常也要变成数据回灌
        return {
            "ok": False,
            "tool": name,
            "error_type": "internal",
            "error": {"message": f"工具内部错误：{type(exc).__name__}: {exc}"},
        }
    return {"ok": True, "tool": name, "data": data}


# ============================================================ 描述改写演练（21:00–21:30）


#: 反例 A：只写"做什么"的坏描述 —— 演练用的反面教材，刻意不含触发场景。
VAGUE_DESCRIPTIONS: Final[dict[str, str]] = {
    "search_knowledge_base": "查询数据。",
    "query_metrics": "查询数据。",
    # 其余保持原描述，变量隔离在两个查数据的工具上
    "generate_report": GENERATE_REPORT_DESCRIPTION,
    "notify": NOTIFY_DESCRIPTION,
}


def build_schema_with_descriptions(overrides: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """生成一份可替换描述的 schema 副本 —— 描述 A/B 演练的机制。

    **描述是模型选工具的唯一依据**（W2 的结论），所以它必须是可替换的
    实验变量：除了覆盖 description，其他一切不动，A/B 才归因干净。
    """
    schema = [spec.to_openai_schema() for spec in QA_TOOL_SPECS]
    if not overrides:
        return schema
    for item in schema:
        name = item["function"]["name"]
        if name in overrides:
            item["function"]["description"] = overrides[name]
    return schema


__all__ = [
    "ERROR_TYPES",
    "QA_TOOL_REGISTRY",
    "QA_TOOL_SPECS",
    "QA_TOOLS_SCHEMA",
    "QAToolSpec",
    "GenerateReportArgs",
    "NotifyArgs",
    "QueryMetricsArgs",
    "SearchKnowledgeBaseArgs",
    "VAGUE_DESCRIPTIONS",
    "build_schema_with_descriptions",
    "execute_qa_tool",
    "generate_report",
    "notify",
    "query_metrics",
    "reset_notify_log",
    "search_knowledge_base",
]
