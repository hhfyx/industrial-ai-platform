"""并发审批测试：本模块最值钱的一条测试（docs/module-contract-approval.md §五/§七）。

验收标准（方案 §一 第 4 条）：

    20 个请求同时 approve → 1 个成功、19 个 409，
    且 approval_logs 只多 1 条。

**断言"日志只多 1 条"而不是"19 个都失败"** —— 失败可能是 500 而不是 409，
那不算对。所以三件事分开数：

    ok == 1, conflicts == 19（全是 VERSION_CONFLICT）,
    server_errors == 0, logs_added == 1

为什么必须 MySQL：SQLite 默认串行化写入，20 个请求排队执行，
"只有一个成功"根本测不出来（方案文档「提醒 1」）。

运行前提（见 tests/biz_fixture.py）：

    docker compose -f docker-compose.biz.yml up -d mysql
    $env:BIZ_DB_URL="mysql+aiomysql://biz:biz_dev@127.0.0.1:3307/biz_contracts"
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.biz.schemas import ApprovalAction
from app.biz.service import ContractService
from app.core.errors import AppError
from tests.biz_fixture import (
    count_logs,
    db_available,
    run_biz,
    seed_contract,
    truncate_all,
)

pytestmark = pytest.mark.skipif(
    not db_available(),
    reason="MySQL 容器未运行：docker compose -f docker-compose.biz.yml up -d mysql",
)


def _svc(engine) -> ContractService:
    return ContractService(sessionmaker=async_sessionmaker(engine, expire_on_commit=False))


def test_concurrent_approve_only_one_wins() -> None:
    """20 并发 approve 同一合同：1 成功 + 19×409 + 日志只多 1 条。"""
    contract = seed_contract(amount_cents=128_000_000, prefix="CONC")
    contract_id = contract["id"]
    assert contract["status"] == "pending"
    version_after_submit = contract["version"]  # submit 后已是 1

    assert count_logs(contract_id) == 1  # submit 留痕

    async def _fire_20(engine) -> tuple[list[dict], list[str], list[BaseException]]:
        svc = _svc(engine)

        async def _one(i: int):
            try:
                return await svc.approve(
                    contract_id,
                    ApprovalAction(
                        comment=f"并发审批-{i}",
                        expected_version=version_after_submit,  # 20 人读到的同一瞬间
                        actor=f"approver-{i}",
                    ),
                )
            except AppError as exc:
                return exc

        results = await asyncio.gather(*(_one(i) for i in range(20)))
        ok: list[dict] = []
        conflicts: list[str] = []
        errors: list[BaseException] = []
        for r in results:
            if isinstance(r, AppError):
                if r.code == "VERSION_CONFLICT":
                    conflicts.append(str(r))
                else:
                    errors.append(r)
            elif isinstance(r, dict):
                ok.append(r)
            else:
                errors.append(r)
        return ok, conflicts, errors

    ok, conflicts, errors = run_biz(_fire_20)

    assert len(errors) == 0, f"出现内部错误: {errors!r}"
    assert len(ok) == 1, f"应有且仅有 1 个成功，实际 {len(ok)}"
    assert len(conflicts) == 19, f"应 19 个 VERSION_CONFLICT，实际 {len(conflicts)}"

    winner = ok[0]
    # 恰好 +1（submit 0→1，approve 1→2）：19 个失败事务全部回滚，没有半提交
    assert winner["version"] == version_after_submit + 1
    assert winner["status"] == "pending"  # 两级合同：第一级通过仍是 pending

    # 真正的验收点：日志只多 1 条（1 submit + 1 approve = 2）
    assert count_logs(contract_id) == 2, "19 个失败事务的日志必须全部回滚"


def test_two_level_chain_end_to_end() -> None:
    """两级审批完整链：manager 通过 → 仍 pending → director 通过 → approved。"""
    contract = seed_contract(amount_cents=10_000_000, prefix="E2E")
    assert [n["role"] for n in contract["nodes"]] == ["manager", "director"]
    cid = contract["id"]

    async def _chain(engine) -> tuple[dict, dict]:
        svc = _svc(engine)
        first = await svc.approve(cid, ApprovalAction(expected_version=1, actor="manager-1"))
        second = await svc.approve(cid, ApprovalAction(expected_version=2, actor="director-1"))
        return first, second

    first, second = run_biz(_chain)
    assert first["status"] == "pending"
    assert first["nodes"][0]["status"] == "approved"
    assert first["nodes"][1]["status"] == "waiting"

    assert second["status"] == "approved"
    assert second["version"] == 3  # submit(1) + approve(2) + approve(3)

    # 审计轨迹完整：submit → approve → approve
    assert count_logs(cid) == 3


def test_expected_version_conflict_is_409() -> None:
    """过期版本被拒：expected_version=0 但表里已是 1 → 409 VERSION_CONFLICT。"""
    contract = seed_contract(amount_cents=5_000_000, prefix="VER")
    cid = contract["id"]

    async def _late_approve(engine) -> None:
        await _svc(engine).approve(cid, ApprovalAction(expected_version=0, actor="late"))

    with pytest.raises(AppError) as exc_info:
        run_biz(_late_approve)
    assert exc_info.value.code == "VERSION_CONFLICT"
    assert exc_info.value.status_code == 409


def test_truncate_between_tests() -> None:
    """清场生效：TRUNCATE 后日志计数回到 0（防夹具间数据串味）。"""
    truncate_all(run_biz(lambda e: _noop(e), pool_size=2))
    assert count_logs(1) == 0


async def _noop(engine) -> None:
    return None
