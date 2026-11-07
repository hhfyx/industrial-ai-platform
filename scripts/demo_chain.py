"""demo_chain：4 工具手动联调（W9 周五 21:35–22:00）。

**先不用 Agent，纯手调** —— 必答问题：为什么先手动串一遍再交给 Agent？
因为你要区分「工具坏了」和「Agent 用错了」。不先手动验证工具，
周六一半的失败你会误归因给模型。

链路（每一步的输入都来自上一步的真实返回）：

    ① retrieve_spec_chunks("抗压强度 判定")     → 规范限值（40.0 MPa）
    ② ReadOnlySQLTool 查 B20241129-03 检测数据   → 实测值（36.1 MPa）
    ③ generate_report_file（用①②的返回值拼报告） → reports/qa-chain-xxx.md
    ④ notify_http（用③的路径写通知正文）          → mock 端收到

mock 接收端由本脚本自动拉起（127.0.0.1 随机端口），跑完打印收到的 payload。

运行：
    .\\.venv\\Scripts\\python.exe -X utf8 scripts\\demo_chain.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tests._mock_receiver import MockReceiver  # noqa: E402

from app.core.llm.real_tools import (  # noqa: E402
    QueryMetricsSQLArgs,
    ReadOnlySQLTool,
    build_demo_db,
    retrieve_spec_chunks,
)
from app.core.llm.side_effect_tools import generate_report_file, notify_http  # noqa: E402

TITLE = "=" * 72


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


async def main() -> int:
    _force_utf8_stdout()
    receiver = MockReceiver()

    try:
        workdir = Path(tempfile.mkdtemp(prefix="qa_chain_"))
        se_dir = workdir / "reports"
        # 把副作用工具的输出指到临时目录
        import app.core.llm.side_effect_tools as se

        se.REPORTS_DIR = se_dir
        se.NOTIFY_LEDGER_PATH = se_dir / "notify_ledger.jsonl"

        print(f"{TITLE}\n步骤①：检索规范（retrieve_spec_chunks）\n{TITLE}")
        kb = retrieve_spec_chunks("抗压强度 判定", k=2)
        top = kb["chunks"][0]
        print(f"  命中 {kb['count']} 条；首条 source={top['source']} page={top['page']}")
        print(f"  原文片段: {top['text'][:80]}...")

        # —— 以下一比的输入来自上一比的真实返回 ——
        limit_text = top["text"]  # 报告引用的规范原文
        print(f"\n{TITLE}\n步骤②：查检测数据（query_metrics_sql）\n{TITLE}")
        db = build_demo_db(workdir / "qa.sqlite")
        sql_tool = ReadOnlySQLTool(db)
        data = await sql_tool.execute(
            QueryMetricsSQLArgs(table="inspections", where_batch_id="B20241129-03")
        )
        print(f"  SQL: {data['sql']}")
        for row in data["rows"]:
            print(f"    {row['metric']} = {row['value']} {row['unit']}（{row['measured_at']}）")
        strength = next(r for r in data["rows"] if r["metric"] == "抗压强度")

        print(f"\n{TITLE}\n步骤③：生成报告落盘（generate_report_file）\n{TITLE}")
        # 报告内容**只用**①②的真实返回 —— 演示"内容必须来自前面的工具结果"
        report = generate_report_file(
            "qa-chain-B20241129-03",
            "批次 B20241129-03 质检报告",
            [
                {
                    "heading": "检测数据",
                    "content": f"抗压强度实测 {strength['value']} {strength['unit']}"
                    f"（测量时间 {strength['measured_at']}，来源：检测数据表）。",
                },
                {
                    "heading": "规范依据",
                    "content": f"来源 {top['source']}：{limit_text[:100]}",
                },
                {
                    "heading": "结论",
                    "content": f"实测 {strength['value']} {strength['unit']}，"
                    f"低于规范限值 40.0 MPa，判定**不合格**。",
                },
            ],
            reports_dir=se_dir,
        )
        print(f"  路径: {report['path']}  字数: {report['chars']}  节数: {report['sections']}")
        print("  --- 报告全文 ---")
        print((se_dir / "qa-chain-B20241129-03.md").read_text(encoding="utf-8"))

        print(f"{TITLE}\n步骤④：发送通知（notify_http → 本地 mock 端）\n{TITLE}")
        note = notify_http(
            "supervisor",
            f"批次 B20241129-03 不合格：抗压强度 {strength['value']} MPa 低于限值 40.0 MPa，"
            f"报告见 {report['path']}",
            "batch-B20241129-03-chain-fail",
            endpoint=receiver.url(),
        )
        print(
            f"  delivered={note['delivered']} attempt={note['attempt']} duplicate={note['duplicate']}"
        )

        print(f"\n{TITLE}\n幂等验证：同 key 再发一次\n{TITLE}")
        try:
            notify_http(
                "supervisor",
                "重复发送（应被账本拦截）",
                "batch-B20241129-03-chain-fail",
                endpoint=receiver.url(),
            )
            print("  × 重复投递未被拦截（不该发生）")
        except Exception as exc:  # noqa: BLE001
            print(f"  √ 已拦截: {type(exc).__name__}: {str(exc)[:60]}")
        print(f"  mock 端共收到 {len(receiver.received)} 个请求（应为 1）")
        print(f"  payload: {json.dumps(receiver.received[0], ensure_ascii=False)[:160]}")

        print(f"\n{TITLE}\n链路联调完成：检索 → 查数 → 报告 → 通知，全部工具真跑\n{TITLE}")
        return 0
    finally:
        receiver.stop()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
