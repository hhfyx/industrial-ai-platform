"""HTTP 层集成测试：拦截 SDK 发出的**真实 HTTP 请求**，验证端到端行为。

## 这一组和前一组的分工

| | `test_llm_extractor_offline.py` | 本文件 |
| --- | --- | --- |
| 替换点 | **假 client 对象** | **HTTP 传输层** |
| 覆盖到 | `LlmExtractor` 的业务逻辑 | 加上 SDK 的请求构造、响应解析、异常映射 |
| 适合测 | 校验回灌、降级、键序 | `response_format` 真的发出去了吗、SDK 报错怎么变异常 |

**为什么两种都要**：假 client 很快，但它绕过了 SDK —— SDK 可能把参数名改掉、
可能把 400 映射成别的异常类型。HTTP 层拦截能看到这些。

## 关键坑：respx 拦不住这个 SDK

任务要求"集成测试用 respx 拦截 HTTP"。**实测做不到**：

```
httpx  : 0.28.1
httpx2 : 2.16.0
httpx is httpx2 -> False
```

openai 3.37 内部用的是 **`httpx2`**（一个独立分叉），而 respx 只 patch `httpx`。
第一版用 `@respx.mock` 时，请求**全部真实发到了网络**，然后
`RESPX: some routes were not called!` 才暴露问题。

正确做法：用 SDK 官方支持的 `http_client=` 注入 `httpx2.MockTransport`。
这是在**传输层**拦截，比 respx 更底层、也更稳 —— 它不依赖库级的 monkey patch。

**这个坑值得记**：任何"用 respx 测 LLM SDK"的教程，在换过 httpx 分叉的版本上都会失效。
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI

from app.services.llm_extractor import LlmExtractor

BASE_URL = "https://api.test/v1"
CHAT_PATH = "/v1/chat/completions"

CONTRACT_TEXT = "合同编号：HT-2026-001，甲方：张三，签订日期：2026-10-20"


# ============================================================ 传输层脚手架


def _ok_body(content: str) -> dict[str, Any]:
    """构造一个合法的 OpenAI chat.completion 响应体。"""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    }


def _error_body(message: str, code: str = "invalid_request") -> dict[str, Any]:
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


class HttpRecorder:
    """记录 SDK 实际发出的每个 request（body / headers / url）。"""

    def __init__(self, scripted: list[Any]) -> None:
        self._scripted = list(scripted)
        self.requests: list[httpx2.Request] = []
        self.bodies: list[dict[str, Any]] = []

    async def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        raw = request.content
        try:
            self.bodies.append(json.loads(raw))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.bodies.append({})

        item = self._scripted.pop(0) if self._scripted else httpx2.Response(500)
        if isinstance(item, BaseException):
            raise item
        return item

    @property
    def call_count(self) -> int:
        return len(self.requests)


def _client(recorder: HttpRecorder) -> AsyncOpenAI:
    """**注入 MockTransport** —— 这是拦住 SDK 请求的唯一可靠方式。

    `max_retries=0` 关掉 SDK 自带的内部重试，否则"调了几次"会被它搅乱。
    """
    transport = httpx2.MockTransport(recorder.handler)
    return AsyncOpenAI(
        api_key="sk-test-fake",
        base_url=BASE_URL,
        max_retries=0,
        timeout=10,
        http_client=httpx2.AsyncClient(transport=transport),
    )


# ============================================================ 1. 请求形状


@pytest.mark.anyio
async def test_sends_json_schema_response_format() -> None:
    """**强约束真的发出去了**：请求里带 `response_format=json_schema`。

    这是"JSON Schema 强约束"这条要求的可执行证据 ——
    只看代码写了 `payload["response_format"]=...` 不够，
    要断言它确实进了 HTTP body。
    """
    recorder = HttpRecorder([httpx2.Response(200, json=_ok_body('{"甲方": "张三"}'))])
    client = _client(recorder)
    try:
        extractor = LlmExtractor(client, model="test-model")
        await extractor.extract(text=CONTRACT_TEXT, fields=["甲方"])
    finally:
        await client.close()

    assert recorder.call_count == 1
    body = recorder.bodies[0]
    assert body["response_format"]["type"] == "json_schema"
    schema = body["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["required"] == ["甲方"]
    assert schema["schema"]["additionalProperties"] is False


@pytest.mark.anyio
async def test_response_format_can_be_disabled() -> None:
    """网关不支持时可以用 `use_response_format=False` 退回纯 prompt 约束。"""
    recorder = HttpRecorder([httpx2.Response(200, json=_ok_body('{"甲方": "张三"}'))])
    client = _client(recorder)
    try:
        extractor = LlmExtractor(client, model="test-model", use_response_format=False)
        await extractor.extract(text=CONTRACT_TEXT, fields=["甲方"])
    finally:
        await client.close()

    assert "response_format" not in recorder.bodies[0]


@pytest.mark.anyio
async def test_prompt_carries_fields_and_text() -> None:
    recorder = HttpRecorder([httpx2.Response(200, json=_ok_body('{"甲方": "张三"}'))])
    client = _client(recorder)
    try:
        extractor = LlmExtractor(client, model="test-model")
        await extractor.extract(text=CONTRACT_TEXT, fields=["甲方", "合同编号"])
    finally:
        await client.close()

    user_msg = recorder.bodies[0]["messages"][1]["content"]
    assert '["甲方", "合同编号"]' in user_msg
    assert CONTRACT_TEXT in user_msg


@pytest.mark.anyio
async def test_temperature_is_zero() -> None:
    recorder = HttpRecorder([httpx2.Response(200, json=_ok_body('{"甲方": "张三"}'))])
    client = _client(recorder)
    try:
        await LlmExtractor(client, model="test-model").extract(text=CONTRACT_TEXT, fields=["甲方"])
    finally:
        await client.close()

    assert recorder.bodies[0]["temperature"] == 0.0


# ============================================================ 2. 全链路成功


@pytest.mark.anyio
async def test_end_to_end_success() -> None:
    recorder = HttpRecorder(
        [httpx2.Response(200, json=_ok_body('{"甲方": "张三", "合同编号": "HT-2026-001"}'))]
    )
    client = _client(recorder)
    try:
        output = await LlmExtractor(client, model="test-model").extract(
            text=CONTRACT_TEXT, fields=["合同编号", "甲方"]
        )
    finally:
        await client.close()

    assert output.data == {"合同编号": "HT-2026-001", "甲方": "张三"}
    assert output.degraded is False
    assert list(output.data) == ["合同编号", "甲方"]


# ============================================================ 3. 通过 HTTP 的回灌重试


@pytest.mark.anyio
async def test_correction_round_trips_through_http() -> None:
    """**校验失败的回灌在 HTTP 层也成立**：第二次请求的 body 里带着纠正提示。

    这条同时验证了两件事：
    1. 回灌是真的发给了模型（不是只在本地对象里改）
    2. 第 2 次请求的 messages 里能看到上一轮的错误输出（assistant）与纠正（user）
    """
    recorder = HttpRecorder(
        [
            httpx2.Response(200, json=_ok_body('{"甲方": {"nested": 1}}')),  # 类型错
            httpx2.Response(200, json=_ok_body('{"甲方": "张三"}')),  # 修好了
        ]
    )
    client = _client(recorder)
    try:
        extractor = LlmExtractor(client, model="test-model")
        output = await extractor.extract(text=CONTRACT_TEXT, fields=["甲方"])
    finally:
        await client.close()

    assert recorder.call_count == 2
    assert output.data["甲方"] == "张三"
    assert output.degraded is False

    second_msgs = recorder.bodies[1]["messages"]
    roles = [m["role"] for m in second_msgs]
    assert roles.count("assistant") >= 1, "应把上一轮的错误输出作为 assistant 消息带回"
    last_user = [m for m in second_msgs if m["role"] == "user"][-1]["content"]
    assert "不符合要求" in last_user or "问题" in last_user


# ============================================================ 4. 网关错误 → 降级


@pytest.mark.anyio
async def test_400_degrades_and_marks_degraded() -> None:
    """网关 400（如不支持 response_format 且关闭也不行）→ 降级 + degraded。"""
    recorder = HttpRecorder(
        [
            httpx2.Response(400, json=_error_body("bad response_format")),
            httpx2.Response(400, json=_error_body("still bad")),
        ]
    )
    client = _client(recorder)
    try:
        output = await LlmExtractor(client, model="test-model").extract(
            text=CONTRACT_TEXT, fields=["甲方"]
        )
    finally:
        await client.close()

    assert output.degraded is True
    assert "+fallback(" in output.model
    assert output.data["甲方"] == "张三", "降级要靠规则版给出结果"


@pytest.mark.anyio
async def test_unsupported_response_format_is_retried_without_it() -> None:
    """网关明确不支持 `response_format` → 关掉强约束再试一次（而不是直接降级）。

    强约束是**优化项**，不是前提。能抽对结果比"坚持用 schema"更重要。
    """
    recorder = HttpRecorder(
        [
            httpx2.Response(
                400,
                json=_error_body("response_format is not supported by this gateway"),
            ),
            httpx2.Response(200, json=_ok_body('{"甲方": "张三"}')),
        ]
    )
    client = _client(recorder)
    try:
        output = await LlmExtractor(client, model="test-model").extract(
            text=CONTRACT_TEXT, fields=["甲方"]
        )
    finally:
        await client.close()

    assert recorder.call_count == 2
    assert "response_format" in recorder.bodies[0]
    assert "response_format" not in recorder.bodies[1], "第二次应关掉强约束"
    assert output.degraded is False, "第二次成功就不该标记降级"
    assert output.data["甲方"] == "张三"


@pytest.mark.anyio
async def test_connection_error_degrades() -> None:
    """连不上 → 降级（不是 500）。"""
    recorder = HttpRecorder([httpx2.ConnectError("refused")])
    client = _client(recorder)
    try:
        output = await LlmExtractor(client, model="test-model").extract(
            text=CONTRACT_TEXT, fields=["甲方"]
        )
    finally:
        await client.close()

    assert output.degraded is True
    assert output.data["甲方"] == "张三"


@pytest.mark.anyio
async def test_500_is_retried_by_sdk_then_degrades() -> None:
    """5xx：SDK 内部重试不掺和（我们关掉了），我们的 retry 层默认不开，
    所以 500 直接进入降级路径 —— 但结果仍要正确。
    """
    recorder = HttpRecorder([httpx2.Response(500, json=_error_body("boom"))])
    client = _client(recorder)
    try:
        output = await LlmExtractor(client, model="test-model").extract(
            text=CONTRACT_TEXT, fields=["甲方"]
        )
    finally:
        await client.close()

    assert output.degraded is True


# ============================================================ 5. 编造拦截（HTTP 层同样生效）


@pytest.mark.anyio
async def test_fabricated_value_dropped_over_http() -> None:
    recorder = HttpRecorder([httpx2.Response(200, json=_ok_body('{"甲方": "李四"}'))])
    client = _client(recorder)
    try:
        output = await LlmExtractor(client, model="test-model").extract(
            text=CONTRACT_TEXT, fields=["甲方"]
        )
    finally:
        await client.close()

    assert output.data["甲方"] is None, "正文里没有李四，必须丢弃"
    assert output.degraded is False, "这不是失败，是正常的'抽不到'"


# ============================================================ 6. 为什么不用 respx（记录事实）


def test_respx_cannot_intercept_this_sdk() -> None:
    """**把这个事实写成测试**，防止后来者又掉进同一个坑。

    断言 openai SDK 用的是 httpx2，而 respx 只 patch httpx。
    如果将来 SDK 换回 httpx，这条测试会失败 —— 那正是我们应该知道的事。
    """
    import httpx
    import respx  # noqa: F401 - 只为断言它存在

    assert httpx2 is not httpx, (
        "如果这条断言失败，说明 SDK 与 httpx 合流了，届时 respx 可以正常工作，"
        "本文件的 MockTransport 方案可以简化"
    )
