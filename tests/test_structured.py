"""`structured_call` 的四条核心测试。

任务要求的四条：
  (1) 一次成功
  (2) 一次修复后成功
  (3) 两次修复后成功
  (4) 全失败抛可识别异常

全部离线：用 `tests/fake_llm.py` 的 `FakeLLMClient` 按脚本回放，
**不碰网络、不碰环境变量**。

异步测试用 `@pytest.mark.anyio`（本仓库不装 pytest-asyncio，
`anyio` 是 starlette 的既有依赖，见 tests/conftest.py 的说明）。

除了四条主路径，另补几条踩坑回归：
  · 代码围栏包裹（模型最常见的小动作）
  · json_object 模式仍输出多余字段
  · 修复提示必须要求「只修报错的字段」，且不能改坏正确字段
  · fallback 降级不抛异常
  · content=None 必须当失败，不能静默成空串
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from app.core.llm import structured as S
from tests.fake_llm import (
    FakeCall,
    FakeChoice,
    FakeLLMClient,
    FakeMessage,
    FakeResponse,
    json_reply,
)


class Ticket(BaseModel):
    """与周一/周二同一个任务，保证三个实验周口径一致。"""

    问题类型: Literal[
        "登录与账号",
        "支付与账单",
        "功能异常",
        "性能与可用性",
        "退换货",
        "咨询与售前",
        "服务与支持",
    ] = Field(description="工单的核心诉求类型")
    紧急程度: Literal["P0", "P1", "P2", "P3"] = Field(
        description="P0 阻断 / P1 可绕过 / P2 局部 / P3 咨询"
    )
    涉及产品: Literal[
        "智云控制台",
        "数据同步服务",
        "移动端App",
        "支付网关",
        "开放API",
        "报表中心",
        "消息推送服务",
    ] = Field(description="业务实际被阻断的产品")


GOOD = {"问题类型": "功能异常", "紧急程度": "P0", "涉及产品": "开放API"}
PROMPT = "工单：开放API /v2/orders 接口持续 502，客户订单同步断了 3 小时。"


@pytest.fixture
def patch_client(monkeypatch: pytest.MonkeyPatch):
    """把 `structured_call` 内部的 client 换成 FakeLLMClient。

    注意 patch 的是 `structured` 模块里**已经 import 进来的名字**
    （`from app.core.llm.client import get_llm_client` 之后，
    `structured.get_llm_client` 是模块级属性），而不是 `client.py` 里的原名。
    patch 错地方是这类测试最常见的假绿。
    """

    def _apply(fake: FakeLLMClient) -> FakeLLMClient:
        monkeypatch.setattr(S, "get_llm_client", lambda *a, **k: fake)
        return fake

    return _apply


# ============================================================ (1) 一次成功


@pytest.mark.anyio
async def test_success_first_try(patch_client) -> None:
    fake = patch_client(FakeLLMClient(replies=[json_reply(**GOOD)]))
    result = await S.structured_call(PROMPT, Ticket)

    assert isinstance(result, Ticket)
    assert result.问题类型 == "功能异常"
    assert result.紧急程度 == "P0"
    assert result.涉及产品 == "开放API"
    assert fake.call_count == 1


# ============================================================ (2) 一次修复成功


@pytest.mark.anyio
async def test_repair_once_then_success(patch_client) -> None:
    bad = json_reply(问题类型="功能异常", 紧急程度="P9", 涉及产品="开放API")
    fake = patch_client(FakeLLMClient(replies=[bad, json_reply(**GOOD)]))

    result = await S.structured_call(PROMPT, Ticket)

    assert isinstance(result, Ticket)
    assert result.紧急程度 == "P0"
    assert fake.call_count == 2

    repair_text = fake.calls[1].all_texts()
    assert "紧急程度" in repair_text
    assert "P9" in repair_text
    assert "只修改上面报错的字段" in repair_text


# ============================================================ (3) 两次修复成功


@pytest.mark.anyio
async def test_repair_twice_then_success(patch_client) -> None:
    bad1 = json_reply(问题类型="功能异常", 紧急程度="P9", 涉及产品="开放API")
    bad2 = json_reply(问题类型="功能异常", 涉及产品="开放API")
    fake = patch_client(FakeLLMClient(replies=[bad1, bad2, json_reply(**GOOD)]))

    result = await S.structured_call(PROMPT, Ticket, max_repair=2)

    assert isinstance(result, Ticket)
    assert result.紧急程度 == "P0"
    assert fake.call_count == 3
    assert "紧急程度" in fake.calls[2].all_texts()


# ============================================================ (4) 全失败抛异常


@pytest.mark.anyio
async def test_exhausted_raises_identifiable_error(patch_client) -> None:
    bad = json_reply(问题类型="功能异常", 紧急程度="P9", 涉及产品="开放API")
    fake = patch_client(FakeLLMClient(replies=[bad]))

    with pytest.raises(S.StructuredOutputError) as excinfo:
        await S.structured_call(PROMPT, Ticket, max_repair=2)

    err = excinfo.value
    assert err.attempts == 3
    assert fake.call_count == 3
    assert err.last_raw
    assert err.last_errors
    assert isinstance(err, ValueError)


# ============================================================ 踩坑回归


@pytest.mark.anyio
async def test_fenced_json_is_accepted(patch_client) -> None:
    """模型把 JSON 包在 ```json 围栏里 —— 解析器容错掉，不浪费修复轮。"""
    fenced = "```json\n" + json_reply(**GOOD) + "\n```"
    fake = patch_client(FakeLLMClient(replies=[fenced]))

    result = await S.structured_call(PROMPT, Ticket)

    assert result.问题类型 == "功能异常"
    assert fake.call_count == 1


@pytest.mark.anyio
async def test_extra_fields_do_not_crash(patch_client) -> None:
    """json_object 模式仍可能输出多余字段，Pydantic 默认忽略它们。"""
    noisy = json_reply(**GOOD, category="故障", summary="接口 502", suggested_team="后端")
    fake = patch_client(FakeLLMClient(replies=[noisy]))

    result = await S.structured_call(PROMPT, Ticket)

    assert result.问题类型 == "功能异常"
    assert fake.call_count == 1


@pytest.mark.anyio
async def test_repair_prompt_preserves_correct_fields(patch_client) -> None:
    """修复提示必须包含「其余字段原样保留」，这是收敛的关键。"""
    bad = json_reply(问题类型="功能异常", 紧急程度="P9", 涉及产品="开放API")
    fake = patch_client(FakeLLMClient(replies=[bad, json_reply(**GOOD)]))
    await S.structured_call(PROMPT, Ticket)

    assert "原样保留" in fake.calls[1].all_texts()


@pytest.mark.anyio
async def test_fallback_returns_instead_of_raising(patch_client) -> None:
    """给了 fallback 就走降级，不抛异常。"""
    bad = json_reply(问题类型="功能异常", 紧急程度="P9", 涉及产品="开放API")
    patch_client(FakeLLMClient(replies=[bad]))
    degraded = Ticket(问题类型="功能异常", 紧急程度="P2", 涉及产品="开放API")

    result = await S.structured_call(PROMPT, Ticket, max_repair=1, fallback=degraded)

    assert result is degraded


@pytest.mark.anyio
async def test_content_none_raises_not_silently_empty(patch_client, monkeypatch) -> None:
    """content=None（推理吃光 max_tokens / 被截断）必须当失败，不能静默成空串。

    为什么这里**不**走修复轮：截断是确定性的 —— 同样的 max_tokens 重试仍会截断，
    重试只是多花一次钱。所以它必须直接抛，让上层去调 max_tokens 或降级，
    而不是被本模块的修复循环吞掉、白烧两轮。
    """
    import tests.fake_llm as F

    async def patched(self, **kwargs):
        self._parent.calls.append(
            FakeCall(messages=list(kwargs.get("messages") or []), kwargs=dict(kwargs))
        )
        return FakeResponse(choices=[FakeChoice(message=FakeMessage(content=None))])

    monkeypatch.setattr(F.FakeCompletions, "create", patched)
    fake = patch_client(FakeLLMClient(replies=["{}"]))

    with pytest.raises(ValueError, match="content=None"):
        await S.structured_call(PROMPT, Ticket, max_repair=2)

    # 关键：既然第一次就截断，就不该再烧后面 2 轮修复的钱
    assert fake.call_count == 1


def test_schema_hint_lists_enums() -> None:
    """schema 说明必须把枚举值渲染进去 —— 模型看不到枚举就会自创分类名。"""
    hint = S.schema_hint(Ticket)
    assert "问题类型" in hint
    assert "登录与账号" in hint
    assert "P0" in hint


def test_parse_json_handles_fence_and_noise() -> None:
    """解析器的四级容错。"""
    assert S.parse_json(json_reply(**GOOD))["问题类型"] == "功能异常"
    assert S.parse_json("```json\n" + json_reply(**GOOD) + "\n```")["问题类型"] == "功能异常"
    assert (
        S.parse_json("好的，结果是：\n" + json_reply(**GOOD) + "\n希望有帮助")["问题类型"]
        == "功能异常"
    )
    with pytest.raises(ValueError):
        S.parse_json("")
    with pytest.raises(ValueError):
        S.parse_json("这里根本没有 JSON")
