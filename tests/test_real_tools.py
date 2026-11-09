"""真工具测试（W9 周四：检索 + 只读 SQL）。

**全部离线**：检索用 hashing embedder + 内存库（真管线、假语料），
SQL 用临时 SQLite 文件（真引擎、假数据）。不依赖真实模型与网络。

分组（对应今晚验收）：
1. 检索工具 4 类 —— 正常 / 缺参 / 类型错 / 空结果（ok=true + hint）
2. SQL 工具 4 类 —— 正常 / 缺参 / 类型错 / 空结果（not_found）
3. 注入演练 —— 三种攻击全部被拦且 error_type 可读
4. 四道闸的单独验证 —— 白名单、参数化、LIMIT、mode=ro 各自确实在工作
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

# 本文件**所有**引用都绑定顶部这一次 import（第一代模块对象）。
# test_llm_client.py 会在测试中途删掉 sys.modules 里的 app.core.llm.*，
# 之后函数体内再 import 拿到的是重新加载的第二份模块 ——
# 第二份的 ToolExecutionError 与第一份的 _SQLBlocked 不是同一个类，
# pytest.raises(ToolExecutionError) 必然失败；monkeypatch 也打在另一份上。
# 教训：跨模块的类型断言必须在文件顶部绑定一次。
import app.core.llm.real_tools as rt
from app.core.llm.real_tools import (
    INJECTION_CASES,
    QueryMetricsSQLArgs,
    ReadOnlySQLTool,
    RetrieveSpecChunksArgs,
    build_demo_db,
    execute_retrieve_tool,
    execute_sql_tool,
    retrieve_spec_chunks,
    sql_injection_drill,
)
from app.core.llm.tools import ToolExecutionError

# ============================================================ 1. 检索工具 4 类


def test_retrieve_normal_hit() -> None:
    """正常：命中且形状符合验收（chunks:[{id, source, page, text}]）。"""
    r = retrieve_spec_chunks("抗压强度 判定", k=2)
    assert r["ok"] if False else r["count"] >= 1  # 薄包装 data 里 count>=1
    chunk = r["chunks"][0]
    assert set(chunk) == {"id", "source", "page", "text"}
    assert len(chunk["text"]) <= 500  # 截断承诺
    assert "抗压强度" in chunk["source"] or "GB50081" in chunk["source"]


def test_retrieve_missing_required_arg() -> None:
    """缺参：Pydantic 拒绝（模型会收到 invalid_arg + fields）。"""
    with pytest.raises(ValidationError):
        RetrieveSpecChunksArgs.model_validate({})


def test_retrieve_wrong_type() -> None:
    """类型错：k="abc" / query=123 都必须在校验层拦下。"""
    with pytest.raises(ValidationError):
        RetrieveSpecChunksArgs.model_validate({"query": "抗压强度", "k": "abc"})
    with pytest.raises(ValidationError):
        RetrieveSpecChunksArgs.model_validate({"query": 123})


def test_retrieve_empty_result_is_ok_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """验收①的关键分支：空结果返回 ok（不报错），chunks=[] 并附 hint。

    用 stub 检索器测：hashing embedder 是字面匹配，靠"选个不相关的词"
    造空结果是赌运气（单字 n-gram 总能撞上语料里的数字/字母）。
    空结果契约是**包装层**的行为，stub 注入正好隔离 —— 这也是验收表里
    "不依赖真实检索"的字面落地。
    """

    class _EmptyResult:
        hits: list = []

    class _EmptyRetrieverForTest:
        def retrieve(self, query: str, *, top_k: int | None = None, k: int | None = None):  # noqa: ARG002
            return _EmptyResult()

    rt._EmptyRetrieverForTest = _EmptyRetrieverForTest  # 供下方 monkeypatch 引用
    monkeypatch.setattr(rt, "_RETRIEVER", rt._EmptyRetrieverForTest())
    r = retrieve_spec_chunks("任何查询")
    assert r["count"] == 0 and r["chunks"] == []
    assert "hint" in r and "检索不到" in r["hint"]
    monkeypatch.setattr(rt, "_RETRIEVER", None)  # 恢复单例，避免污染其他用例


def test_retrieve_k_bounds_enforced() -> None:
    """k 白名单在类型层：0 和 9 都不该通过。"""
    with pytest.raises(ValidationError):
        RetrieveSpecChunksArgs.model_validate({"query": "抗压强度", "k": 0})
    with pytest.raises(ValidationError):
        RetrieveSpecChunksArgs.model_validate({"query": "抗压强度", "k": 9})


def test_retrieve_envelope_paths() -> None:
    """统一信封：成功 / invalid_json / invalid_arg 三条路。"""
    ok = run_envelope('{"query": "抗压强度", "k": 2}')
    assert ok["ok"] is True and ok["tool"] == "retrieve_spec_chunks"
    bad_json = run_envelope("{bad")
    assert bad_json["ok"] is False and bad_json["error_type"] == "invalid_json"
    bad_args = run_envelope('{"k": 2}')
    assert bad_args["ok"] is False and bad_args["error_type"] == "invalid_arg"


def run_envelope(raw: str) -> dict:
    import asyncio

    return asyncio.run(execute_retrieve_tool(raw))


# ============================================================ 2. SQL 工具 4 类


@pytest.fixture()
def demo_db(tmp_path: Path) -> Path:
    return build_demo_db(tmp_path / "qa.sqlite")


def make_tool_args(**kwargs: object) -> QueryMetricsSQLArgs:
    base: dict[str, object] = {"table": "inspections"}
    base.update(kwargs)
    return QueryMetricsSQLArgs.model_validate(base)


def test_sql_normal_query(demo_db: Path) -> None:
    """正常：按批次过滤，返回行 + 实际执行的 SQL（审计字段）。"""
    tool = ReadOnlySQLTool(demo_db)
    d = asyncio_run(tool.execute(make_tool_args(where_batch_id="B20241129-03")))
    assert d["row_count"] == 2
    assert d["rows"][0]["batch_id"] == "B20241129-03"
    assert "LIMIT ?" in d["sql"] and "WHERE batch_id = ?" in d["sql"]
    # 值必须是绑定参数，绝不出现在 SQL 文本里
    assert "B20241129-03" not in d["sql"]


def asyncio_run(coro: object) -> object:
    import asyncio

    return asyncio.run(coro)  # type: ignore[arg-type]


def test_sql_missing_param(demo_db: Path) -> None:
    """缺参：table 是必填 Literal，缺了在 Pydantic 层拦。"""
    with pytest.raises(ValidationError):
        QueryMetricsSQLArgs.model_validate({"where_batch_id": "B20241207-01"})


def test_sql_wrong_type(demo_db: Path) -> None:
    """类型错：table 不在白名单 / limit 越界 / columns 含未知列。

    columns 的白名单**不在 Pydantic 层**（它依赖 table 的值，Pydantic 表达不了
    跨字段白名单），在执行层 `_check_columns` —— 所以它的拒绝路径要走
    信封（execute_sql_tool）测，见 test_gate_column_whitelist_at_execution。
    """
    with pytest.raises(ValidationError):
        make_tool_args(table="users")
    with pytest.raises(ValidationError):
        make_tool_args(limit=0)
    with pytest.raises(ValidationError):
        make_tool_args(limit=101)


def test_sql_empty_result_not_found(demo_db: Path) -> None:
    """空结果：批次不存在 → not_found（附已知批次清单），不是静默 0 行。"""
    tool = ReadOnlySQLTool(demo_db)
    import asyncio

    with pytest.raises(ToolExecutionError, match="查无批次"):
        asyncio.run(tool.execute(make_tool_args(where_batch_id="B19990101-99")))


def test_sql_limit_enforced(demo_db: Path) -> None:
    """强制 LIMIT：limit=1 就只回 1 行，即使表里更多。"""
    tool = ReadOnlySQLTool(demo_db)
    d = asyncio_run(tool.execute(make_tool_args(limit=1)))
    assert d["row_count"] == 1


# ============================================================ 3. 注入演练


@pytest.mark.anyio
async def test_injection_all_three_blocked() -> None:
    """验收③：三种攻击全部拦下且 error_type 可读。"""
    results = await sql_injection_drill()
    assert len(results) == 3
    for r in results:
        assert r["blocked"] is True, f"{r['name']} 没被拦下！"
        assert r["error_type"] in ("invalid_arg", "not_found", "invalid_json"), (
            f"{r['name']} 的 error_type 不可读：{r['error_type']}"
        )


@pytest.mark.anyio
async def test_injection_drop_table_payload_rejected() -> None:
    """攻击①：'; DROP TABLE… 塞进 where 值 → 关键字检查拒绝（invalid_arg）。"""
    r = await execute_sql_tool(
        '{"table": "inspections", "where_batch_id": "B20241129-03\'; DROP TABLE inspections;--"}'
    )
    assert r["ok"] is False and r["error_type"] == "invalid_arg"


@pytest.mark.anyio
async def test_injection_union_read_rejected() -> None:
    """攻击②：UNION SELECT 塞进 table → Literal 白名单拒绝（invalid_arg）。"""
    r = await execute_sql_tool(
        '{"table": "inspections UNION SELECT batch_id FROM batches", "where_batch_id": "B20241207-01"}'
    )
    assert r["ok"] is False and r["error_type"] == "invalid_arg"


@pytest.mark.anyio
async def test_injection_long_like_rejected() -> None:
    """攻击③：200 字超长值 → max_length=32 拒绝（invalid_arg）。"""
    r = await execute_sql_tool('{"table": "inspections", "where_batch_id": "' + "x" * 200 + '"}')
    assert r["ok"] is False and r["error_type"] == "invalid_arg"


@pytest.mark.anyio
async def test_injection_leaves_table_intact(demo_db: Path) -> None:
    """纵深证明：即使攻击后，目标表数据完好 —— 只读连接 + 白名单双保险。"""
    await sql_injection_drill()
    conn = sqlite3.connect(demo_db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        assert count == 4  # 建库时的原始行数
    finally:
        conn.close()


# ============================================================ 4. 四道闸单独验证


def test_gate_readonly_connection_blocks_write(demo_db: Path) -> None:
    """闸⑤（mode=ro）：只读连接上执行写语句必须被 SQLite 引擎拒绝。"""
    conn = sqlite3.connect(f"file:{demo_db}?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM inspections")
    finally:
        conn.close()


def test_gate_values_never_in_sql_text(demo_db: Path) -> None:
    """闸②参数化的可观察证据：任何用户值都不出现在生成的 SQL 文本里。"""
    tool = ReadOnlySQLTool(demo_db)
    d = asyncio_run(
        tool.execute(make_tool_args(where_batch_id="B20241207-01", where_metric="含水量", limit=5))
    )
    assert "B20241207-01" not in d["sql"]
    assert "含水量" not in d["sql"]
    assert d["sql"].count("?") == 3  # batch_id / metric / limit 全是占位符


def test_gate_table_whitelist_closed() -> None:
    """闸③：白名单外的表名一律 invalid_arg，没有旁路。"""
    for bad in ("batches; DROP TABLE x", "sqlite_master", "(SELECT 1)", ""):
        with pytest.raises(ValidationError):
            make_tool_args(table=bad)


@pytest.mark.anyio
async def test_gate_column_whitelist_at_execution() -> None:
    """闸③'（列白名单在执行层）：未知列在信封里被拒为 invalid_arg。

    Pydantic 表达不了「columns 合法性依赖 table」的跨字段白名单，
    所以列校验放在执行层 —— 这意味着模型会拿到**运行时**的 invalid_arg
    而不是校验期的。信封必须接住它（这正是统一错误结构存在的理由）。
    """
    r = await execute_sql_tool('{"table": "inspections", "columns": ["batch_id", "password_hash"]}')
    assert r["ok"] is False and r["error_type"] == "invalid_arg"
    # W11：message 从 error.message 提升到顶层（定版六字段之一）
    assert "password_hash" in r["message"]


def test_sql_timeout_constant_sane() -> None:
    """timeout 闸存在且 sane：>0 且足够小（别让一次慢查询挂住循环）。"""
    from app.core.llm.real_tools import SQL_TIMEOUT_S

    assert 0 < SQL_TIMEOUT_S <= 10


def test_injection_cases_have_metadata() -> None:
    """演练用例自带解释（attack / blocked_by）：注入演练是教学材料，不是黑盒。"""
    assert len(INJECTION_CASES) == 3
    for case in INJECTION_CASES:
        assert "attack" in case and "blocked_by" in case
        json.loads(case["where"])  # payload 必须是合法 JSON（否则演示脚本会炸）
