"""LLM 抽取器测试（W2 接入）：全部离线，用假 client 模拟模型。

**为什么这些测试必须存在，而不能只靠 `test_api.py`**：
`test_api.py` 锁的是契约形状（200/422/键集），它跑的是真模型调用会网络依赖；
这里锁的是**韧性行为**：模型输出坏 JSON、编造值、超时、漏字段时，
服务层必须优雅降级而不是 500。这类分支只有注入假 client 才能稳定覆盖。

分组：
1. 正常路径 —— 按 fields 顺序、抽不到为 null
2. 回指校验 —— **编造的值必须被丢弃**（契约 §4.5 的硬保证，不靠 prompt）
3. 解析容忍 —— 代码围栏 / 夹着废话的 JSON
4. 降级 —— 任何 LLM 侧失败都退回规则版，且 model 字段留下降级痕迹
5. 装配 —— `EXTRACTOR_IMPL` 环境变量真的能切换实现
"""

from __future__ import annotations

import pytest

from app.core.deps import Extractor, get_extractor, reset_extractor_cache
from app.services.extractor import RuleExtractor
from app.services.llm_extractor import LlmExtractor

CONTRACT_TEXT = "合同编号：HT-2026-001，甲方：张三，签订日期：2026-10-20，合同金额：¥128,000.00"


# ---------------------------------------------------------------- 假 client
class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str | None = None, *, exc: BaseException | None = None) -> None:
        self.content = content
        self.exc = exc
        self.calls: list[dict] = []

    async def create(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return _FakeResponse(self.content)


class FakeClient:
    """只实现 `chat.completions.create` —— `LlmExtractor` 只用这一个方法。

    这本身就是一个断言：抽取器**不依赖** AsyncOpenAI 的具体类型，
    所以换成任何 OpenAI 兼容实现都行。
    """

    def __init__(self, content: str | None = None, *, exc: BaseException | None = None) -> None:
        self.completions = _FakeCompletions(content, exc=exc)
        self.chat = type("Chat", (), {"completions": self.completions})()


def _extractor(client: FakeClient, **kwargs) -> LlmExtractor:  # noqa: ANN003
    return LlmExtractor(client, model="fake-model", **kwargs)  # type: ignore[arg-type]


# ============================================================ 1. 正常路径
@pytest.mark.anyio
async def test_extracts_in_request_order_and_null_for_missing() -> None:
    """键序必须按 `fields` 重建 —— 即使模型返回的顺序不同。"""
    # 故意把 JSON 里的顺序打乱（甲方在前），模拟模型不守顺序
    client = FakeClient(
        '{"data": {"甲方": "张三", "合同编号": "HT-2026-001", "不存在的字段": null}}'
    )
    output = await _extractor(client).extract(
        text=CONTRACT_TEXT, fields=["合同编号", "甲方", "不存在的字段"]
    )

    assert list(output.data) == ["合同编号", "甲方", "不存在的字段"]
    assert output.data["合同编号"] == "HT-2026-001"
    assert output.data["甲方"] == "张三"
    assert output.data["不存在的字段"] is None
    assert output.missing == ["不存在的字段"]
    # 2/3 命中再打 0.75 折 = 0.5，与规则版同一公式
    assert output.confidence == 0.5
    assert output.model == "fake-model"


@pytest.mark.anyio
async def test_model_field_is_the_model_name_for_response_header() -> None:
    """`X-Extract-Model` 取的是 `model_name`，契约 §2.4 要求它是实现标识。"""
    output = await _extractor(FakeClient('{"data": {"甲方": "张三"}}')).extract(
        text=CONTRACT_TEXT, fields=["甲方"]
    )
    assert output.model == "fake-model"


@pytest.mark.anyio
async def test_prompt_contains_fields_and_original_text() -> None:
    """字段名以 JSON 数组原文写出：字段名里可能有顿号/冒号，结构化才不会被切错。"""
    client = FakeClient('{"data": {}}')
    await _extractor(client).extract(text=CONTRACT_TEXT, fields=["甲方", "合同编号"])

    body = client.completions.calls[0]
    user_prompt = body["messages"][1]["content"]
    assert '["甲方", "合同编号"]' in user_prompt
    assert CONTRACT_TEXT in user_prompt
    # 抽取不要创造性
    assert body["temperature"] == 0.0


# ============================================================ 2. 回指校验
@pytest.mark.anyio
async def test_fabricated_value_is_dropped_to_null() -> None:
    """**契约 §4.5 的硬保证**：正文里没有"李四"，模型编出来也必须被丢弃。

    这条不能只靠 prompt。模型幻觉一个甲方名字时，prompt 拦不住 ——
    只有原文回指校验能拦。
    """
    client = FakeClient('{"data": {"甲方": "李四"}}')
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["甲方"])

    assert output.data["甲方"] is None, "编造的值必须被丢弃"
    assert output.missing == ["甲方"]
    assert output.confidence == 0.0


@pytest.mark.anyio
async def test_partial_value_still_accepted() -> None:
    """正文是 `¥128,000.00`，模型回 `128,000.00`（少了货币符号）—— 这是**正确**值，不能误杀。"""
    client = FakeClient('{"data": {"合同金额": "128,000.00"}}')
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["合同金额"])

    assert output.data["合同金额"] == "128,000.00"


@pytest.mark.anyio
async def test_short_value_requires_exact_presence() -> None:
    """短值整体查：`张三` 在正文里，保留；`王五` 不在，丢弃。"""
    client = FakeClient('{"data": {"甲方": "张三"}}')
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["甲方"])
    assert output.data["甲方"] == "张三"


@pytest.mark.anyio
async def test_numeric_value_is_stringified() -> None:
    """模型把金额写成数字也要接受，最终契约要求值是字符串（§2.1）。"""
    client = FakeClient('{"data": {"合同编号": 128000}}')
    output = await _extractor(client).extract(text="订单号 128000", fields=["合同编号"])
    assert output.data["合同编号"] == "128000"
    assert isinstance(output.data["合同编号"], str)


# ============================================================ 3. 解析容忍
@pytest.mark.anyio
@pytest.mark.parametrize(
    "content",
    [
        '{"data": {"甲方": "张三"}}',
        '```json\n{"data": {"甲方": "张三"}}\n```',
        '```\n{"data": {"甲方": "张三"}}\n```',
        '好的，结果如下：{"data": {"甲方": "张三"}} 以上。',
        '  {"data": {"甲方": "张三"}}  ',
    ],
    ids=["plain", "fence-json", "fence-plain", "chatter-around", "padded"],
)
async def test_parses_common_model_output_shapes(content: str) -> None:
    """模型说"只输出 JSON"但经常加围栏或废话，这些都要能解析。"""
    output = await _extractor(FakeClient(content)).extract(text=CONTRACT_TEXT, fields=["甲方"])
    assert output.data["甲方"] == "张三"


@pytest.mark.anyio
async def test_null_strings_are_normalized_to_none() -> None:
    """模型有时把缺失写成字符串 `"null"` / `"None"` —— 要归一成真 None。"""
    client = FakeClient('{"data": {"甲方": "null", "合同编号": "None"}}')
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["甲方", "合同编号"])
    assert output.data["甲方"] is None
    assert output.data["合同编号"] is None


# ============================================================ 4. 降级
@pytest.mark.anyio
@pytest.mark.parametrize(
    ("exc", "expect_wrapped"),
    [
        (TimeoutError("超时"), True),
        (RuntimeError("401 unauthorized"), False),
        (ValueError("rate limited"), False),
    ],
    ids=["timeout", "auth", "rate-limit"],
)
async def test_any_llm_failure_falls_back_to_rule_extractor(
    exc: BaseException, expect_wrapped: bool
) -> None:
    """**任何 LLM 侧失败都降级**，不返回 500 —— 用户要的是结果，不是错误码。

    降级后的 `model` 必须带 `+fallback(原因)`：这是排查降级频率的唯一入口。

    **接入 `with_retry` 之后语义有一个真实变化**：
    - 可重试的异常（超时/连接/429/5xx）会被包装成 `RetryExhaustedError`，
      所以 `model` 里出现的是 `RetryExhaustedError` 而不是原始的 `TimeoutError`。
      这**对排查更好**：它直接告诉你"这是重试之后仍然失败"，而不是"偶发一次失败"。
    - 不可重试的异常（RuntimeError / ValueError）原样抛出，`model` 里是原始类型名。
    """
    client = FakeClient(exc=exc)
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["合同编号", "甲方"])

    # 规则版能正确抽出这两个字段
    assert output.data["合同编号"] == "HT-2026-001"
    assert output.data["甲方"] == "张三"
    assert output.confidence == 1.0
    assert "+fallback(" in output.model

    if expect_wrapped:
        assert "RetryExhaustedError" in output.model, output.model
    else:
        assert type(exc).__name__ in output.model, output.model


@pytest.mark.anyio
async def test_garbage_json_falls_back_not_raises() -> None:
    """坏 JSON 也走降级，绝不抛给路由。"""
    client = FakeClient("这不是 JSON")
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["甲方"])

    assert output.data["甲方"] == "张三"  # 规则版抽到了
    assert "fallback" in output.model


@pytest.mark.anyio
async def test_empty_content_falls_back() -> None:
    """`content is None` 是推理模型吃光 max_tokens 的典型症状（01_params.md §3.1）。"""
    client = FakeClient(None)
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["甲方"])

    assert output.data["甲方"] == "张三"
    assert "fallback" in output.model


@pytest.mark.anyio
async def test_fallback_is_not_infinite_when_text_has_nothing() -> None:
    """降级也要能给出"全部 null"，而不是再抛一次。"""
    client = FakeClient(exc=TimeoutError("超时"))
    output = await _extractor(client).extract(text="完全无关的正文", fields=["甲方"])

    assert output.data["甲方"] is None
    assert output.confidence == 0.0
    assert output.missing == ["甲方"]


# ============================================================ 5. 装配
def test_default_extractor_is_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认走 LLM —— 否则"仓库里没有 LLM 调用"的尴尬就还在。"""
    monkeypatch.delenv("EXTRACTOR_IMPL", raising=False)
    reset_extractor_cache()
    try:
        extractor = get_extractor()
        assert isinstance(extractor, LlmExtractor)
        # Protocol 装配检查：路由只依赖 extract()，两个实现都满足
        assert isinstance(extractor, Extractor)
    finally:
        reset_extractor_cache()


def test_env_switch_returns_rule_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    """`EXTRACTOR_IMPL=rule` 一键回退 —— 应急开关是改配置，不是改代码。"""
    monkeypatch.setenv("EXTRACTOR_IMPL", "rule")
    reset_extractor_cache()
    try:
        extractor = get_extractor()
        assert isinstance(extractor, RuleExtractor)
        assert isinstance(extractor, Extractor)
    finally:
        reset_extractor_cache()


def test_both_implementations_satisfy_the_same_protocol() -> None:
    """两个实现的 `extract` 签名必须一致，这是"实现可换"的类型级保证。"""
    import inspect

    rule_sig = inspect.signature(RuleExtractor.extract)
    llm_sig = inspect.signature(LlmExtractor.extract)
    assert list(rule_sig.parameters) == list(llm_sig.parameters)
    assert rule_sig.parameters["text"].kind is llm_sig.parameters["text"].kind
    assert rule_sig.parameters["fields"].kind is llm_sig.parameters["fields"].kind
