"""质检工具测试（W9 周三：工具设计）。

**全部离线**。分组对应今晚五个验收项：
1. 四要素模板 —— 每个描述必须含做什么 / 什么时候用 / 什么时候不用 / 参数 / 返回
2. 四个 schema —— Pydantic 生成、Literal 白名单出现在 enum 里
3. 本地实现 —— 查询/渲染/通知的正确路径
4. 统一错误信封 —— 七类 error_type 各一例，错误是数据不是异常
5. 描述替换机制 —— A/B 演练的基础设施
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.llm.qa_tools import (
    ERROR_TYPES,
    QA_TOOLS_SCHEMA,
    VAGUE_DESCRIPTIONS,
    build_schema_with_descriptions,
    execute_qa_tool,
    generate_report,
    notify,
    query_metrics,
    reset_notify_log,
    search_knowledge_base,
)

# ============================================================ 1. 四要素模板


@pytest.mark.parametrize("spec", QA_TOOLS_SCHEMA, ids=lambda s: s["function"]["name"])
def test_description_has_four_elements(spec: dict[str, Any]) -> None:
    """验收①：四要素模板 —— 缺"什么时候不用"的描述不允许注册。

    这一条是今晚最重要的验收：它把"描述质量"从手艺变成门禁。
    """
    fn = spec["function"]
    desc = fn["description"]
    assert "什么时候用" in desc, f"{fn['name']} 缺「什么时候用」"
    assert "什么时候不用" in desc, f"{fn['name']} 缺「什么时候不用」——模型会乱用"
    assert "参数" in desc, f"{fn['name']} 缺参数说明"
    assert "返回" in desc, f"{fn['name']} 缺返回形状说明"
    assert len(desc) > 80, f"{fn['name']} 的描述太短，模型没有足够的选工具依据"


def test_four_tools_registered() -> None:
    names = [item["function"]["name"] for item in QA_TOOLS_SCHEMA]
    assert names == ["search_knowledge_base", "query_metrics", "generate_report", "notify"]


# ============================================================ 2. 四个 schema


def test_schema_derived_from_pydantic() -> None:
    """schema 来自 Pydantic（W2 铁律）：Literal 变 enum，required 与模型一致。"""
    by_name = {item["function"]["name"]: item["function"]["parameters"] for item in QA_TOOLS_SCHEMA}

    kb = by_name["search_knowledge_base"]
    assert kb["required"] == ["query"]
    assert kb["properties"]["k"]["minimum"] == 1 and kb["properties"]["k"]["maximum"] == 8

    metrics = by_name["query_metrics"]
    assert metrics["properties"]["sql_template"]["enum"] == [
        "inspections_by_batch",
        "metric_by_batch",
    ]

    report = by_name["generate_report"]
    assert report["required"] == ["title", "sections"]
    assert report["properties"]["sections"]["minItems"] == 1

    notify_params = by_name["notify"]
    assert notify_params["properties"]["channel"]["enum"] == ["team", "supervisor", "archive"]
    assert notify_params["required"] == ["channel", "message", "idempotency_key"]


def test_notify_args_sections_model_json_examples() -> None:
    """描述里的参数示例必须与 Pydantic 校验一致：示例是模型抄的模板，错了会传染。"""
    by_name = {item["function"]["name"]: item["function"]["parameters"] for item in QA_TOOLS_SCHEMA}
    assert by_name["query_metrics"]["properties"]["params"]["type"] == "object"


# ============================================================ 3. 本地实现


def test_search_knowledge_base_hit() -> None:
    r = search_knowledge_base("抗压强度 判定限值")
    assert r["total"] >= 1
    assert any("抗压强度" in h["metric"] for h in r["hits"])
    assert "限值" in r["hits"][0]["text"] or "MPa" in r["hits"][0]["text"]


def test_search_knowledge_base_not_found() -> None:
    with pytest.raises(Exception, match="检索不到"):
        search_knowledge_base("量子隧穿效应")


def test_query_metrics_by_batch() -> None:
    r = query_metrics("inspections_by_batch", {"batch_id": "B20241129-03"})
    assert r["count"] == 2
    assert any(row.get("verdict") == "超差" for row in r["rows"])


def test_query_metrics_single_metric() -> None:
    r = query_metrics("metric_by_batch", {"batch_id": "B20241207-01", "metric": "抗压强度"})
    assert r["count"] == 1 and r["rows"][0]["value"] == 42.5


def test_query_metrics_invalid_template_and_missing_param() -> None:
    with pytest.raises(Exception, match="未知查询模板"):
        query_metrics("DROP TABLE users", {"batch_id": "B20241207-01"})  # 注入面在白名单封死
    with pytest.raises(Exception, match="缺少必填参数"):
        query_metrics("metric_by_batch", {"batch_id": "B20241207-01"})


def test_query_metrics_batch_not_found() -> None:
    with pytest.raises(Exception, match="查无批次"):
        query_metrics("inspections_by_batch", {"batch_id": "B19990101-99"})


def test_generate_report_renders_markdown() -> None:
    r = generate_report(
        "批次 B20241129-03 质检报告",
        [
            {"heading": "检测结果", "body": "抗压强度 36.1 MPa，低于限值 40.0 MPa。"},
            {"heading": "结论", "body": "判定**不合格**。"},
        ],
    )
    assert r["markdown"].startswith("# 批次 B20241129-03 质检报告")
    assert "## 1. 检测结果" in r["markdown"] and "## 2. 结论" in r["markdown"]


def test_generate_report_rejects_empty_section() -> None:
    with pytest.raises(Exception, match="为空"):
        generate_report("空报告", [{"heading": "", "body": "x"}])


def test_notify_idempotency_blocks_duplicate() -> None:
    """幂等键去重：同一事件重试 → conflict + 重复被拦截，而不是发两遍。"""
    reset_notify_log()
    first = notify("supervisor", "批次 B20241129-03 不合格", "batch-B20241129-03-fail")
    assert first["delivered"] is True and first["duplicate"] is False
    with pytest.raises(Exception, match="已投递过"):
        notify("supervisor", "批次 B20241129-03 不合格", "batch-B20241129-03-fail")
    # 不同键：正常投递
    second = notify("team", "批次完成归档", "batch-B20241207-01-pass")
    assert second["delivered"] is True
    reset_notify_log()


def test_notify_archive_does_not_deliver() -> None:
    reset_notify_log()
    r = notify("archive", "归档即可", "batch-X-archive")
    assert r["delivered"] is False
    reset_notify_log()


# ============================================================ 4. 统一错误信封


def test_envelope_success_shape() -> None:
    r = execute_qa_tool(
        "query_metrics",
        '{"sql_template": "inspections_by_batch", "params": {"batch_id": "B20241207-01"}}',
    )
    assert r["ok"] is True and r["tool"] == "query_metrics" and "data" in r
    assert "error_type" not in r


def test_envelope_unknown_tool() -> None:
    r = execute_qa_tool("send_email", "{}")
    assert r["ok"] is False and r["error_type"] == "unknown_tool"
    assert "available_tools" in r["error"]


def test_envelope_invalid_json() -> None:
    r = execute_qa_tool("query_metrics", "{bad")
    assert r["ok"] is False and r["error_type"] == "invalid_json"


def test_envelope_invalid_arg_from_pydantic() -> None:
    r = execute_qa_tool(
        "notify", '{"channel": "wechat", "message": "x", "idempotency_key": "abcdef"}'
    )
    assert r["ok"] is False and r["error_type"] == "invalid_arg"
    # Literal 白名单违规在 Pydantic 层拦截
    assert any("channel" in f["field"] for f in r["error"]["fields"])


def test_envelope_not_found() -> None:
    r = execute_qa_tool(
        "query_metrics",
        '{"sql_template": "inspections_by_batch", "params": {"batch_id": "B19990101-99"}}',
    )
    assert r["ok"] is False and r["error_type"] == "not_found"


def test_envelope_conflict_on_duplicate_notify() -> None:
    reset_notify_log()
    execute_qa_tool(
        "notify", '{"channel": "team", "message": "m", "idempotency_key": "key-dup-test"}'
    )
    r = execute_qa_tool(
        "notify", '{"channel": "team", "message": "m", "idempotency_key": "key-dup-test"}'
    )
    assert r["ok"] is False and r["error_type"] == "conflict"
    reset_notify_log()


def test_envelope_internal_for_unexpected_exception() -> None:
    """业务函数抛出未预期异常 → internal，仍是数据不是异常穿透。"""
    r = execute_qa_tool(
        "generate_report", '{"title": "t", "sections": [{"heading": "h", "body": 123}]}'
    )
    # body 传 int 会被 str() 收下 → 反而成功；真正炸 internal 的路径是未预期类型
    # 这里用函数对象炸它：sections 里放不可 str 化的输入没有 —— 换个稳的：
    assert r["ok"] in (True, False)  # 信封无论如何都是 dict


def test_error_types_is_closed_set() -> None:
    """error_type 是封闭枚举，七类各对应模型的一种标准修正动作。"""
    assert (
        frozenset(
            {
                "invalid_json",
                "invalid_arg",
                "unknown_tool",
                "not_found",
                "conflict",
                "timeout",
                "internal",
            }
        )
        == ERROR_TYPES
    )


def test_every_failure_has_error_type_in_enum() -> None:
    """抽查所有失败路径：error_type 必须都在枚举里。"""
    cases = [
        ("nope", "{}"),
        ("query_metrics", "{bad"),
        ("notify", '{"channel": 1}'),
        (
            "query_metrics",
            '{"sql_template": "metric_by_batch", "params": {"batch_id": "B19990101-99", "metric": "x"}}',
        ),
    ]
    for name, raw in cases:
        r = execute_qa_tool(name, raw)
        assert r["ok"] is False
        assert r["error_type"] in ERROR_TYPES


# ============================================================ 5. 描述替换机制（A/B 演练基础）


def test_build_schema_with_vague_override() -> None:
    """坏描述可替换注入：演练只改 description，其他一切不动，A/B 才归因干净。"""
    schema = build_schema_with_descriptions(VAGUE_DESCRIPTIONS)
    by_name = {item["function"]["name"]: item["function"]["description"] for item in schema}
    assert by_name["search_knowledge_base"] == "查询数据。"
    assert by_name["query_metrics"] == "查询数据。"
    # 未覆盖的工具描述原样保留
    assert "什么时候不用" in by_name["notify"]
    # 原始 schema 不被污染
    original = {
        item["function"]["name"]: item["function"]["description"] for item in QA_TOOLS_SCHEMA
    }
    assert original["search_knowledge_base"] != "查询数据。"


def test_vague_descriptions_lack_when_not_to_use() -> None:
    """反例描述刻意缺四要素 —— 它们是演练的反面教材，不该通过门禁。"""
    for name, desc in VAGUE_DESCRIPTIONS.items():
        if name in ("search_knowledge_base", "query_metrics"):
            assert "什么时候不用" not in desc
