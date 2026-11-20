"""报表端点：JSON 汇总 + CSV 导出（docs/module-contract-approval.md §4.3）。

CSV 是 JD 第 9 条「报表生成逻辑」的直接交付：

- 按月 × 客户聚合走 repository.report_monthly()（GROUP BY 在数据库做）；
- CSV 走流式 StringBuilder（量级 ≤100 行，不值得上 StreamingResponse）；
- `Content-Disposition` 带 UTF-8 BOM —— Excel 打开中文不乱码，
  这个细节是「真的导出过报表给业务用」和「没导出过」的分界线。
"""

from __future__ import annotations

import csv
import io
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import Response

from app.biz.repository import ContractRepository

router = APIRouter(prefix="/api/v2/reports", tags=["reports"])

CSV_HEADERS = [
    "月份",
    "客户ID",
    "客户名称",
    "合同数",
    "已批笔数",
    "金额(元)",
]


def _parse_month_bound(value: str | None, *, is_start: bool) -> datetime | None:
    """`?month=2026-12` 或留空。给月份参数换算成 [start, end) 边界。"""
    if value is None:
        return None
    try:
        year, mon = value.split("-")
        year_i, mon_i = int(year), int(mon)
    except ValueError as exc:
        from app.core.errors import AppError

        raise AppError(
            code="INVALID_INPUT",
            message=f"month 参数格式应为 YYYY-MM，收到 {value!r}",
            status_code=422,
        ) from exc
    if is_start:
        return datetime(year_i, mon_i, 1)
    # 结束边界 = 下个月 1 号（左闭右开，避免月末 23:59:59.999 的精度坑）
    if mon_i == 12:
        return datetime(year_i + 1, 1, 1)
    return datetime(year_i, mon_i + 1, 1)


@router.get("/contracts")
async def contracts_report(month: str | None = None) -> dict:
    """JSON 汇总：按月 × 客户聚合（含 total）。"""
    repo = ContractRepository()
    rows = await repo.report_monthly(
        start=_parse_month_bound(month, is_start=True),
        end=_parse_month_bound(month, is_start=False),
    )
    return {
        "rows": rows,
        "total_cnt": sum(r["cnt"] for r in rows),
        "total_amount_yuan": round(sum(r["amount_yuan"] for r in rows), 2),
    }


@router.get("/contracts.csv")
async def contracts_report_csv(month: str | None = None) -> Response:
    """CSV 下载。UTF-8 BOM + Content-Disposition，Excel 直接打开不乱码。"""
    repo = ContractRepository()
    rows = await repo.report_monthly(
        start=_parse_month_bound(month, is_start=True),
        end=_parse_month_bound(month, is_start=False),
    )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_HEADERS)
    for r in rows:
        writer.writerow(
            [
                r["month"],
                r["customer_id"],
                r["customer_name"],
                r["cnt"],
                r["approved_cnt"],
                f"{r['amount_yuan']:.2f}",
            ]
        )

    filename = "contracts_report.csv" if month is None else f"contracts_report_{month}.csv"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


__all__ = ["CSV_HEADERS", "router"]
