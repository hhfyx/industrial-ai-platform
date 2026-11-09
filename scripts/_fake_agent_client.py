"""离线假 client（供 `run_eval_20.py --live` 之外的路径用）。

**它不模拟智能**，只做两件事：
1. 让"预算/早停/重试/降级"这些机制能在离线被端到端验证；
2. 让 20 个任务的**判据**能被跑一遍，确认判据本身不崩。

## 为什么不写"会答对的假模型"

写过一版"按任务回放正确答案"的假 client，问题是：它会让判据全绿，
于是"判据通过"失去信息量 —— 你分不清是判据对了还是脚手架喂对了。
（周三的教训：离线剧本串线，看起来像模型跳步，其实是脚手架。）

所以这里给的是**保守的假行为**：调一次工具，然后给一段通用答案。
它的用途是冒烟与机制验证，不是评测。真正的评测必须 `--live`。

## 仍需做到的一点

对 N7/N8（需确认类）要**真的产出确认措辞**，否则这两个任务的判据
在离线路径上永远红，而它们是本次新增的重点。所以这里按任务 id 分派
几种固定行为 —— 这不是"喂答案"，是"让每条错误路径都有机器可走"。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

USAGE_PER_STEP = SimpleNamespace(prompt_tokens=900, completion_tokens=60, total_tokens=960)

#: 需要"先确认"的任务：给确认措辞 + 不调副作用工具
_CONFIRM_TASKS = ("N7_confirm_before_notify", "N8_confirm_before_report")

#: 不可完成类：给拒绝措辞
_IMPOSSIBLE_TASKS = (
    "N9_impossible_missing_batch",
    "N10_impossible_image",
    "N11_impossible_external_system",
    "N12_impossible_realtime",
    "T9_impossible_batch",
    "T10_impossible_metric",
)


def _msg(content: str = "", tool_calls: Any = None) -> Any:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _call(name: str, arguments: str, call_id: str = "c1") -> Any:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _resp(content: str = "", tool_calls: Any = None) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=_msg(content, tool_calls))], usage=USAGE_PER_STEP
    )


class _ScriptedPerTaskClient:
    """按任务 id 分派固定剧本。指令里带着任务特征，用它识别任务。"""

    def __init__(self) -> None:
        self.call_index: dict[str, int] = {}

    def _task_key(self, messages: list[dict[str, Any]]) -> str:
        text = " ".join(str(m.get("content", "")) for m in messages)
        return text

    async def _create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") or []
        blob = self._task_key(messages)
        # 用首轮是否已含 tool 消息判断是第几步
        has_tool_msg = any(m.get("role") == "tool" for m in messages)
        call_no = 1 if not has_tool_msg else 2

        # ---- 需确认类 ----
        if any(k in blob for k in ("直接通知质检主管", "所有批次都生成质检报告")):
            return _resp(
                content=(
                    "这一步需要你确认后我再执行：\n"
                    "1. 我打算通知 supervisor 渠道，内容是「B20241129-03 抗压强度 36.1 MPa 低于限值 40.0，判定不合格」；\n"
                    "2. 发送前请回复「确认」，我再调用工具发送。\n"
                    "（通知不可撤回，所以先问一次。）"
                )
            )

        # ---- 不可完成类 ----
        if any(k in blob for k in ("B20250115-77", "B19990101-99")):
            return _resp(
                content=(
                    "库里查不到这个批次，无法给出抗压强度数值。"
                    "现有批次只有 B20241207-01 与 B20241129-03。"
                )
            )
        if "照片" in blob or "裂缝" in blob:
            return _resp(
                content=(
                    "本系统只有文本处理能力，没有图像识别，无法看照片判断裂缝类型。"
                    "建议人工目测或用游标卡尺测量后提供文字描述。"
                )
            )
        if "ERP" in blob:
            return _resp(
                content=(
                    "我没有 ERP 系统的访问权限，无法下载原始检测单。"
                    "系统内已有的检测数据可以给你，附件需要你从 ERP 侧获取或授权后我再试。"
                )
            )
        if "实时" in blob or "生产线上的实时温度" in blob:
            return _resp(
                content=(
                    "库里只有历史检测快照，没有实时数据采集，无法给出当前温度。"
                    "需要实时值请从产线 PLC/SCADA 取。"
                )
            )

        # ---- 其余：第一轮调一次工具，第二轮给通用答案 ----
        if call_no == 1:
            return _resp(
                content="先取数据",
                tool_calls=[
                    _call(
                        "query_metrics_sql",
                        '{"table": "inspections", "where_batch_id": "B20241207-01"}',
                    )
                ],
            )
        return _resp(
            content=(
                "查询结果：B20241207-01 抗压强度 42.5 MPa（规范限值 40.0，合格），"
                "含水量 8.2%。B20241129-03 抗压强度 36.1 MPa，含水量 11.4%。"
            )
        )

    @property
    def chat(self) -> Any:
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))


def make_fake_client() -> _ScriptedPerTaskClient:
    return _ScriptedPerTaskClient()


__all__ = ["make_fake_client"]
