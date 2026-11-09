"""W11 周一：四个工具 × 五类错误的 20 条路径测试（**全部 mock，零网络零真实磁盘**）。

## 为什么单独开一个文件

`test_real_tools.py` / `test_side_effect_tools.py` 是**功能**测试：它们验证
"工具能不能干对活"。本文件是**契约**测试：它验证"工具干不成活时，给出的那张
错误卡片是不是五类里的某一类、恢复动作对不对"。

两者失败的含义完全不同 —— 功能测试红是"坏了"，契约测试红是"模型会被误导"。
分开文件，红的时候一眼知道该看哪个。

## 20 条的编排（4 工具 × 5 类）

| | 参数错 | 权限错 | 超时 | 空结果 | 内部异常 |
| --- | --- | --- | --- | --- | --- |
| retrieve_spec_chunks | invalid_json | not_permitted | timeout | empty_result(ok) | internal |
| query_metrics_sql | invalid_arg | not_permitted | timeout | not_found | internal |
| generate_report | invalid_arg | not_permitted | timeout | empty_result | internal |
| notify | invalid_arg | not_permitted(403) | timeout | empty_result(空白正文) | internal |

"空结果"在**读工具**上是 ok=true（0 行是事实，W9 T10 的教训），
在**写工具**上是 ok=false（没内容可交付）。两种落点同一类，测试里分别断言。

## mock 的纪律

所有故障都用**注入**触发，不靠"真的把它弄坏"：

- 权限错：`monkeypatch` 让写盘/打开库抛 `PermissionError`（不折腾文件系统 ACL，
  Windows 上 chmod 只读目录实测无效 —— 写文件照样成功，见本文件注释）；
- 超时：把工具内部的超时常量打到 0.001s，并让底层函数 sleep；
- 内部异常：让底层抛一个意料外的 `RuntimeError`。

注入点都在**工具边界之内**（底层函数/常量），不是把整个工具替换成假函数 ——
后者会让测试变成"验证 mock 的行为"，而不是"验证错误归类的行为"。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import app.core.llm.real_tools as rt
import app.core.llm.side_effect_tools as se

# ============================================================ 基础设施


def run(coro: Any) -> dict[str, Any]:
    return asyncio.run(coro)


def assert_error_shape(r: dict[str, Any], *, tool: str, error_type: str) -> None:
    """失败信封的**定版六字段 + tool**，一个都不能少。

    这条断言比 `assert r["error_type"] == X` 严格得多：它守的是
    "所有失败路径长得一样"。只要有一处手搓 dict 漏了 hint/retryable，
    错误分布表和模型侧的行为都会在那一处悄悄退化。
    """
    assert r["ok"] is False, f"{tool} 应当失败：{r}"
    assert r["tool"] == tool
    assert r["error_type"] == error_type, f"{tool} 分类错：{r['error_type']} != {error_type}"
    for key in ("message", "hint", "retryable", "retry_after"):
        assert key in r, f"{tool} 的 {error_type} 信封缺字段 {key}：{sorted(r)}"
    assert isinstance(r["message"], str) and r["message"]
    assert isinstance(r["hint"], str) and r["hint"]
    assert isinstance(r["retryable"], bool)


# ============================================================ ① retrieve_spec_chunks


def test_retrieve_param_error() -> None:
    """参数错：JSON 不合法 → invalid_json，可重试（改 JSON 即可）。"""
    r = run(rt.execute_retrieve_tool('{"query": "抗压强度", '))
    assert_error_shape(r, tool="retrieve_spec_chunks", error_type="invalid_json")
    assert r["retryable"] is True
    assert r["retry_after"] is None


def test_retrieve_permission_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """权限错：索引不可读（EACCES）→ not_permitted，**不可重试**。

    注意：这类错误以前会被归成 internal（可重试），模型会反复重试一个
    永远失败的动作。`classify_exception` 把 PermissionError 单独认出来。
    """

    def boom(*_a: Any, **_k: Any) -> Any:
        raise PermissionError("索引目录不可读（EACCES）")

    monkeypatch.setattr(rt, "_get_retriever", boom)
    r = run(rt.execute_retrieve_tool('{"query": "抗压强度"}'))
    assert_error_shape(r, tool="retrieve_spec_chunks", error_type="not_permitted")
    assert r["retryable"] is False
    assert r["retry_after"] is None


def test_retrieve_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """超时：检索超过 RETRIEVE_TIMEOUT_S → timeout，可重试且带退避秒数。"""
    monkeypatch.setattr(rt, "RETRIEVE_TIMEOUT_S", 0.01)

    def slow(*_a: Any, **_k: Any) -> Any:
        time.sleep(0.2)
        return {}

    monkeypatch.setattr(rt, "retrieve_spec_chunks", slow)
    r = run(rt.execute_retrieve_tool('{"query": "抗压强度"}'))
    assert_error_shape(r, tool="retrieve_spec_chunks", error_type="timeout")
    assert r["retryable"] is True
    assert r["retry_after"] == pytest.approx(2.0)


def test_retrieve_empty_result_is_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """空结果：0 命中 → **ok=true + outcome=empty_result**（0 行是事实，不是故障）。

    这是 W9 T10 的教训：把空结果当失败回灌，模型会无意义重试；
    当事实回灌，它才会如实转述"索引里没有"。
    """
    monkeypatch.setattr(
        rt, "retrieve_spec_chunks", lambda *_a, **_k: {"query": "x", "chunks": [], "count": 0}
    )
    r = run(rt.execute_retrieve_tool('{"query": "量子力学"}'))
    assert r["ok"] is True
    assert r["outcome"] == "empty_result"
    assert "hint" in r and r["hint"]


def test_retrieve_internal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """内部异常：意料外的 RuntimeError → internal，可重试一次。"""

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("向量库连接池耗尽")

    monkeypatch.setattr(rt, "retrieve_spec_chunks", boom)
    r = run(rt.execute_retrieve_tool('{"query": "抗压强度"}'))
    assert_error_shape(r, tool="retrieve_spec_chunks", error_type="internal")
    assert r["retryable"] is True


# ============================================================ ② query_metrics_sql


def test_sql_param_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """参数错：limit 越界 → invalid_arg，附字段明细（hint 之外的结构化指路）。"""
    monkeypatch.setenv("QA_METRICS_DB_PATH", str(rt.build_demo_db(tmp_path / "qa.sqlite")))
    r = run(rt.execute_sql_tool('{"table": "inspections", "limit": 101}'))
    assert_error_shape(r, tool="query_metrics_sql", error_type="invalid_arg")
    assert r["retryable"] is True
    assert r["details"]["fields"][0]["field"] == "limit"


def test_sql_permission_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """权限错：部署未授权该表 → not_permitted（改参数永远过不去）。

    用环境变量模拟部署配置，而不是改代码 —— 这正是真实部署里的用法：
    schema 里四张表都可见，但生产只授权 inspections。
    """
    monkeypatch.setenv("QA_METRICS_DB_PATH", str(rt.build_demo_db(tmp_path / "qa.sqlite")))
    monkeypatch.setenv(rt.PERMITTED_TABLES_ENV, "inspections")
    r = run(rt.execute_sql_tool('{"table": "batches"}'))
    assert_error_shape(r, tool="query_metrics_sql", error_type="not_permitted")
    assert r["retryable"] is False
    assert "inspections" in r["message"]  # 告诉模型哪些表是授权的


def test_sql_timeout_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """超时：查询超过 SQL_TIMEOUT_S → timeout，可退避重试。"""
    db = rt.build_demo_db(tmp_path / "qa.sqlite")
    monkeypatch.setenv("QA_METRICS_DB_PATH", str(db))
    monkeypatch.setattr(rt, "SQL_TIMEOUT_S", 0.01)

    def slow_sync(self: Any, sql: str, params: tuple) -> list:  # noqa: ARG001
        time.sleep(0.2)
        return []

    monkeypatch.setattr(rt.ReadOnlySQLTool, "_execute_sync", slow_sync)
    r = run(rt.execute_sql_tool('{"table": "inspections"}'))
    assert_error_shape(r, tool="query_metrics_sql", error_type="timeout")
    assert r["retryable"] is True
    assert r["retry_after"] == pytest.approx(2.0)


def test_sql_empty_result_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """空结果（写侧口径）：查无批次 → not_found，**不可重试**（重试必然还空）。"""
    monkeypatch.setenv("QA_METRICS_DB_PATH", str(rt.build_demo_db(tmp_path / "qa.sqlite")))
    r = run(rt.execute_sql_tool('{"table": "inspections", "where_batch_id": "B19990101-99"}'))
    assert_error_shape(r, tool="query_metrics_sql", error_type="not_found")
    assert r["retryable"] is False
    assert r["hint"]  # 告诉模型换批次号或如实告知


def test_sql_internal_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """内部异常：库文件被外部损坏 → sqlite3.DatabaseError → internal。"""
    db = tmp_path / "broken.sqlite"
    db.write_text("这不是一个 sqlite 文件", encoding="utf-8")
    monkeypatch.setenv("QA_METRICS_DB_PATH", str(db))
    r = run(rt.execute_sql_tool('{"table": "inspections"}'))
    assert_error_shape(r, tool="query_metrics_sql", error_type="internal")
    assert r["retryable"] is True


def test_sql_engine_readonly_maps_to_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    """引擎侧只读拒绝 → not_permitted（第五道闸 mode=ro 的真实错误路径）。

    这条**额外**测一类交叉情形：sqlite 抛 OperationalError("readonly database")，
    必须被认成权限错而不是 internal。以前它会掉进通用 except，模型会重试。
    """
    from app.core.llm.error_contract import classify_exception

    assert classify_exception(sqlite3.OperationalError("attempt to write a readonly database")) == (
        "not_permitted"
    )
    assert classify_exception(sqlite3.OperationalError("database is locked")) == "timeout"
    assert classify_exception(RuntimeError("未知")) == "internal"


# ============================================================ ③ generate_report


@pytest.fixture()
def reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把报告目录与账本指到临时目录（不污染仓库，也不互相干扰）。"""
    import app.core.llm.side_effect_tools as mod

    d = tmp_path / "reports"
    monkeypatch.setattr(mod, "REPORTS_DIR", d)
    monkeypatch.setattr(mod, "NOTIFY_LEDGER_PATH", d / "notify_ledger.jsonl")
    return d


def _report_payload(**overrides: Any) -> str:
    body: dict[str, Any] = {
        "task_id": "qa-perm-01",
        "title": "批次质检报告",
        "sections": [{"heading": "结论", "content": "抗压强度 36.1 MPa，低于限值"}],
    }
    body.update(overrides)
    return json.dumps(body, ensure_ascii=False)


def test_report_param_error(reports_dir: Path) -> None:
    """参数错：某一节 heading 为空 → invalid_arg（形状不合格）。"""
    payload = json.dumps(
        {
            "task_id": "qa-perm-01",
            "title": "t",
            "sections": [{"heading": "", "content": "有正文但没标题"}],
        },
        ensure_ascii=False,
    )
    r = run(se.execute_report_tool(payload))
    assert_error_shape(r, tool="generate_report", error_type="invalid_arg")
    assert r["retryable"] is True


def test_report_permission_error(reports_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """权限错：报告目录不可写（EACCES / 只读介质）→ not_permitted。

    注入点选 `Path.write_text` 而不是改文件系统权限：Windows 上把目录
    设成只读，写文件**照样成功**（实测），这条路径在测试里根本触发不了。
    真实环境的触发是只读挂载/ACL/杀软拦截，抛的都是 PermissionError(OSError)。
    """
    original = Path.write_text

    def deny(self: Path, *a: Any, **k: Any) -> int:
        if str(self).endswith(".md"):
            raise PermissionError(13, "Permission denied", str(self))
        return original(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", deny)
    r = run(se.execute_report_tool(_report_payload()))
    assert_error_shape(r, tool="generate_report", error_type="not_permitted")
    assert r["retryable"] is False


def test_report_timeout_error(reports_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """超时：写盘挂住超过 REPORT_WRITE_TIMEOUT_S → timeout，可退避重试。"""
    monkeypatch.setattr(se, "REPORT_WRITE_TIMEOUT_S", 0.01)

    def slow_write(*_a: Any, **_k: Any) -> dict[str, Any]:
        time.sleep(0.2)
        return {}

    monkeypatch.setattr(se, "generate_report_file", slow_write)
    r = run(se.execute_report_tool(_report_payload()))
    assert_error_shape(r, tool="generate_report", error_type="timeout")
    assert r["retryable"] is True
    assert r["retry_after"] == pytest.approx(2.0)


def test_report_empty_result(reports_dir: Path) -> None:
    """空结果：sections 为空数组 → **empty_result**（不是 invalid_arg）。

    参数形状合法（是个合法数组），缺的是内容 —— 恢复动作是"先去取数据"，
    不是"把字段改个形状"。这条分类差异决定了模型下一步做什么。
    """
    r = run(se.execute_report_tool(_report_payload(sections=[])))
    assert_error_shape(r, tool="generate_report", error_type="empty_result")
    assert r["retryable"] is False


def test_report_internal_error(reports_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """内部异常：渲染期抛意料外异常 → internal。"""

    def boom(*_a: Any, **_k: Any) -> str:
        raise RuntimeError("模板引擎崩了")

    monkeypatch.setattr(se, "render_report_markdown", boom)
    r = run(se.execute_report_tool(_report_payload()))
    assert_error_shape(r, tool="generate_report", error_type="internal")
    assert r["retryable"] is True


# ============================================================ ④ notify


def _notify_payload(**overrides: Any) -> str:
    body: dict[str, Any] = {
        "channel": "team",
        "message": "批次 B20241129-03 抗压强度 36.1 MPa，判定不合格",
        "idempotency_key": "batch-B20241129-03-fail",
    }
    body.update(overrides)
    return json.dumps(body, ensure_ascii=False)


class _FakeHTTP:
    """替换 `_post_json` 的两种故障注入（不起真 server，测归类不测传输）。"""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls = 0

    def __call__(self, endpoint: str, payload: dict, timeout: float = 5.0) -> dict:
        self.calls += 1
        if self.mode == "permission":
            from urllib.error import HTTPError

            raise HTTPError(endpoint, 403, "Forbidden", {}, None)  # type: ignore[arg-type]
        if self.mode == "timeout":
            raise TimeoutError("timed out")
        raise OSError("端点不可达")


def test_notify_param_error(reports_dir: Path) -> None:
    """参数错：channel 不在白名单 → invalid_arg（改渠道即可）。"""
    r = run(se.execute_notify_http_tool(_notify_payload(channel="wechat")))
    assert_error_shape(r, tool="notify", error_type="invalid_arg")
    assert r["retryable"] is True
    assert "team" in r["message"]  # 告诉模型有哪些合法渠道


def test_notify_permission_error(reports_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """权限错：端点 403 → not_permitted，且**不重试**（403 重试多少次都是 403）。"""
    fake = _FakeHTTP("permission")
    monkeypatch.setattr(se, "_post_json", fake)
    r = run(se.execute_notify_http_tool(_notify_payload()))
    assert_error_shape(r, tool="notify", error_type="not_permitted")
    assert r["retryable"] is False
    assert fake.calls == 1  # ← 关键：只打一次，没白烧重试


def test_notify_timeout_error(reports_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """超时：每次请求都 TimeoutError → 重试 1 次后 timeout，可退避重试。"""
    fake = _FakeHTTP("timeout")
    monkeypatch.setattr(se, "_post_json", fake)
    r = run(se.execute_notify_http_tool(_notify_payload()))
    assert_error_shape(r, tool="notify", error_type="timeout")
    assert r["retryable"] is True
    assert fake.calls == 2  # 1 次执行 + 1 次重试
    assert r["retry_after"] == pytest.approx(2.0)


def test_notify_empty_result(reports_dir: Path) -> None:
    """空结果：正文只有空白字符 → empty_result（Pydantic 的 min_length 挡不住）。"""
    r = run(se.execute_notify_http_tool(_notify_payload(message="   ")))
    assert_error_shape(r, tool="notify", error_type="empty_result")
    assert r["retryable"] is False


def test_notify_internal_error(reports_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """内部异常：端点不可达（连接被拒）→ internal，可重试（可能只是服务没起来）。"""
    fake = _FakeHTTP("unreachable")
    monkeypatch.setattr(se, "_post_json", fake)
    r = run(se.execute_notify_http_tool(_notify_payload()))
    assert_error_shape(r, tool="notify", error_type="internal")
    assert r["retryable"] is True


# ============================================================ 横向纪律


ALL_TOOLS: list[tuple[str, Callable[[], dict[str, Any]]]] = []


def test_twenty_paths_covered() -> None:
    """这条测试守的是**本文件自己**：20 条路径真的写了 20 条。

    没有这条，"20 条绿"很容易变成"18 条绿 + 2 条我以为是绿的"。
    比例尺：4 工具 × 5 类。
    """
    import ast

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    tests = [
        n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    ]
    # 只数 4×5 的那 20 条：命名规则 test_<tool>_<category>_error / _result
    per_tool = {
        "retrieve": [t for t in tests if t.startswith("test_retrieve_")],
        "sql": [t for t in tests if t.startswith("test_sql_")],
        "report": [t for t in tests if t.startswith("test_report_")],
        "notify": [t for t in tests if t.startswith("test_notify_")],
    }
    for tool, names in per_tool.items():
        assert len(names) >= 5, f"{tool} 的错误路径不足 5 类：{names}"
    assert sum(len(v) for v in per_tool.values()) >= 20


def test_every_error_type_has_spec() -> None:
    """五类 × error_type 的每个值都必须在分类表里登记（不能有"裸奔"的类型）。"""
    from app.core.llm.error_contract import ERROR_SPECS, ERROR_TYPES

    assert set(ERROR_SPECS) == set(ERROR_TYPES)
    for spec in ERROR_SPECS.values():
        assert spec.recovery, "每类都必须说清靠什么恢复"
        assert spec.hint
        if not spec.retryable:
            assert spec.retry_after is None, "不可重试就不该给退避秒数（自相矛盾的信号）"


# ============================================================ 重试预算：代码说了算


def test_retry_budget_suppresses_after_limit() -> None:
    """**「超时 1 次后不再重试」是代码强制，不是提示词建议。**

    第一次失败：retryable=True（模型可以再试）；
    第二次同类失败：预算用尽，retryable 被翻成 False，且 hint 明确写
    "不要再调用这个工具重试同一动作"。

    这条测试是本周设计的落地点：如果它红了，说明"能不能重试"又回到
    模型手里去了 —— 那本周的第一步就白走了。
    """
    from app.core.llm.error_contract import apply_retry_budget, tool_error

    counts: dict[str, int] = {}
    first = apply_retry_budget(tool_error("query_metrics_sql", "timeout", "超时"), counts)
    assert first["retryable"] is True
    assert first["retry_after"] == pytest.approx(2.0)

    second = apply_retry_budget(tool_error("query_metrics_sql", "timeout", "又超时"), counts)
    assert second["retryable"] is False
    assert second["retry_after"] is None  # 不可重试就不给退避秒数
    assert second["retry_suppressed"] is True
    assert "不要再调用" in second["hint"]

    third = apply_retry_budget(tool_error("query_metrics_sql", "timeout", "再超时"), counts)
    assert third["retryable"] is False


def test_retry_budget_is_per_tool_and_type() -> None:
    """预算按 (tool, error_type) 计**累计**，不是按参数 ——

    模型最常见的错误动作是**换个措辞重试同一个坏工具**（参数微调但错误类型
    不变）。按参数计数拦不住它，按错误类型计数能。同时不同工具互不牵连。
    """
    from app.core.llm.error_contract import apply_retry_budget, tool_error

    counts: dict[str, int] = {}
    for _ in range(2):
        apply_retry_budget(tool_error("notify", "timeout", "超时"), counts)
    # notify 的 timeout 预算已用尽
    assert apply_retry_budget(tool_error("notify", "timeout", "x"), counts)["retryable"] is False
    # 同一工具的另一个类不受影响
    assert apply_retry_budget(tool_error("notify", "internal", "x"), counts)["retryable"] is True
    # 另一个工具的同名类不受影响
    assert (
        apply_retry_budget(tool_error("query_metrics_sql", "timeout", "x"), counts)["retryable"]
        is True
    )


def test_retry_budget_never_touches_non_retryable_or_success() -> None:
    """不可重试的错误与成功结果都不该被预算改写（幂等、无副作用）。"""
    from app.core.llm.error_contract import apply_retry_budget, tool_error, tool_ok

    counts: dict[str, int] = {}
    perm = tool_error("query_metrics_sql", "not_permitted", "无权限")
    for _ in range(5):
        assert apply_retry_budget(dict(perm), counts)["retryable"] is False
    assert counts == {}  # 不可重试的不计数

    ok = tool_ok("query_metrics_sql", {"rows": []})
    assert apply_retry_budget(ok, counts) is ok
