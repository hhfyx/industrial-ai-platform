"""LLM 抽取器的离线测试（W2 主交付的**唯一无法伪造的成果**）。

**这个文件要证明的事**：`LlmExtractor` 的行为在完全离线、无 API key 的前提下可验证。
这之所以重要，是因为"能离线测绿" ≈ "实现真的可替换" ——
W4 加检索、W9 加工具、W12 加缓存都不会破坏已有契约。

## 分组

1. **强约束** —— JSON Schema 的形态；动态校验模型对缺键/多键/类型错的判定
2. **正常路径** —— 键序按 fields 重建、抽不到为 null、回指校验拦编造
3. **校验回灌** —— 坏输出后把 Pydantic 错误喂回去，第二轮修好
4. **降级** —— 两轮仍失败 / 网络异常 → 规则版 + `degraded=True`
5. **容错** —— 代码围栏、外层包装、null 字符串、网关不支持 response_format
6. **装配** —— `EXTRACTOR_IMPL` 切换、Protocol 一致性

全部用 `tests.fake_llm.FakeLLMClient`，不联网。
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.core.deps import Extractor, get_extractor, reset_extractor_cache
from app.services.extractor import RuleExtractor, extract_rule_based
from app.services.llm_extractor import (
    MAX_CORRECTION_ROUNDS,
    LlmExtractor,
    build_schema_model,
    json_schema_for,
)
from tests.fake_llm import FakeLLMClient, json_reply, wrapped_reply

CONTRACT_TEXT = "合同编号：HT-2026-001，甲方：张三，签订日期：2026-10-20，合同金额：¥128,000.00"


def _extractor(client: FakeLLMClient, **kwargs) -> LlmExtractor:  # noqa: ANN003
    return LlmExtractor(client, model="fake-model", **kwargs)  # type: ignore[arg-type]


# ============================================================ 1. 强约束


def test_json_schema_lists_all_fields_as_required() -> None:
    """**required 必须列出全部字段**：缺键和"值为 null"对下游是不同的东西。

    前者会让前端少一个展示位，后者才是"抽不到"。
    """
    schema = json_schema_for(["甲方", "合同金额"])
    assert schema["type"] == "object"
    assert schema["required"] == ["甲方", "合同金额"]
    assert schema["properties"]["甲方"]["type"] == ["string", "null"]


def test_json_schema_forbids_additional_properties() -> None:
    """`additionalProperties: False` 让"多返回键"在模型层就被挡住。"""
    assert json_schema_for(["甲方"])["additionalProperties"] is False


def test_schema_model_rejects_missing_key() -> None:
    """**缺键必须是校验错误。**

    踩过的坑：字段默认值写成 `None` 时，pydantic 会用默认值补齐，
    于是"模型漏了一个字段"永远发现不了 —— 必须用 `...`（Required）。
    """
    model = build_schema_model(["甲方", "金额"])
    assert model.model_fields["甲方"].is_required() is True

    with pytest.raises(ValidationError, match="Field required"):
        model.model_validate({"甲方": "张三"})  # 缺 金额


def test_schema_model_rejects_extra_key() -> None:
    model = build_schema_model(["甲方"])
    with pytest.raises(ValidationError, match="Extra inputs"):
        model.model_validate({"甲方": "张三", "多余": "x"})


def test_schema_model_rejects_nested_structure() -> None:
    """嵌套对象不是字符串 —— 契约要求值恒为 `string | null`。

    注意边界：**标量（数字/布尔）会在解析阶段被归一成字符串**（见
    `test_scalar_values_are_coerced_not_rejected`），只有嵌套结构才是
    真正需要模型去修的"结构理解错误"。
    """
    model = build_schema_model(["甲方"])
    with pytest.raises(ValidationError):
        model.model_validate({"甲方": {"nested": 1}})


def test_schema_model_accepts_null_and_string() -> None:
    model = build_schema_model(["甲方"])
    assert model.model_validate({"甲方": None}).model_dump() == {"甲方": None}
    assert model.model_validate({"甲方": "张三"}).model_dump() == {"甲方": "张三"}


# ============================================================ 2. 正常路径


@pytest.mark.anyio
async def test_success_path_uses_one_call_and_marks_not_degraded() -> None:
    client = FakeLLMClient(replies=[json_reply(合同编号="HT-2026-001", 甲方="张三")])
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["合同编号", "甲方"])

    assert output.data == {"合同编号": "HT-2026-001", "甲方": "张三"}
    assert output.degraded is False
    assert client.call_count == 1, "成功路径应只调一次"


@pytest.mark.anyio
async def test_data_key_order_follows_request_not_model_output() -> None:
    """**键序按 fields 重建**，绝不信模型给的顺序（契约 §2.1）。"""
    # 模型故意反着给
    client = FakeLLMClient(replies=[json_reply(甲方="张三", 合同编号="HT-2026-001")])
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["合同编号", "甲方"])

    assert list(output.data) == ["合同编号", "甲方"]


@pytest.mark.anyio
async def test_missing_field_becomes_none() -> None:
    """抽不到就是 null；模型明确给 null 也是 null。"""
    client = FakeLLMClient(replies=[json_reply(合同编号="HT-2026-001", 甲方=None)])
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["合同编号", "甲方"])

    assert output.data["甲方"] is None
    assert output.missing == ["甲方"]
    assert output.confidence == 0.375  # 1/2 命中 × 0.75


@pytest.mark.anyio
async def test_fabricated_value_is_dropped() -> None:
    """契约 §4.5：**永不编造**。正文里没有"李四"，模型给了也要丢弃。"""
    client = FakeLLMClient(replies=[json_reply(甲方="李四")])
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["甲方"])

    assert output.data["甲方"] is None, "编造的值必须被丢弃"
    assert output.confidence == 0.0


@pytest.mark.anyio
async def test_partial_value_still_accepted() -> None:
    """正文是 `¥128,000.00`，模型回 `128,000.00` —— 正确值，不能误杀。"""
    client = FakeLLMClient(replies=[json_reply(合同金额="128,000.00")])
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["合同金额"])
    assert output.data["合同金额"] == "128,000.00"


# ============================================================ 3. 校验回灌


@pytest.mark.anyio
async def test_invalid_type_triggers_correction_and_recovers() -> None:
    """**核心验收**：第一轮类型错 → 把 Pydantic 错误回灌 → 第二轮修好。

    断言三件事：
    1. 确实调了 2 次
    2. 第 2 次的 messages 里**包含上一轮的错误输出和错误原因**
    3. 最终结果正确且 degraded=False
    """
    client = FakeLLMClient(
        replies=['{"合同编号": {"nested": 1}}', json_reply(合同编号="HT-2026-001")]
    )
    extractor = _extractor(client)
    output = await extractor.extract(text=CONTRACT_TEXT, fields=["合同编号"])

    assert client.call_count == 2
    assert output.data["合同编号"] == "HT-2026-001"
    assert output.degraded is False

    # 回灌内容检查
    second = client.calls[1]
    texts = second.all_texts()
    assert "合同编号" in texts
    assert "valid string" in texts or "字符串" in texts, "应把 Pydantic 的报错原样回灌"
    assert "nested" in texts, "应带上模型上一轮给的值，它才知道改哪里"
    # 上一轮的错误输出也在对话里（assistant 角色）
    assert any(m.get("role") == "assistant" for m in second.messages)


@pytest.mark.anyio
async def test_trace_records_the_correction_round() -> None:
    """`last_trace` 要留下可诊断的记录（轮次、字段、原因）。"""
    client = FakeLLMClient(
        replies=['{"合同编号": {"nested": 1}}', json_reply(合同编号="HT-2026-001")]
    )
    extractor = _extractor(client)
    await extractor.extract(text=CONTRACT_TEXT, fields=["合同编号"])

    assert len(extractor.last_trace) == 1
    trace = extractor.last_trace[0]
    assert trace.round_index == 0
    assert trace.errors[0]["field"] == "合同编号"
    assert trace.errors[0]["got"] == {"nested": 1}


@pytest.mark.anyio
async def test_invalid_json_triggers_correction() -> None:
    """坏 JSON 也走回灌 —— 它不是"该降级"的理由，模型能改。"""
    client = FakeLLMClient(replies=["这不是 JSON", json_reply(合同编号="HT-2026-001")])
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["合同编号"])

    assert client.call_count == 2
    assert output.data["合同编号"] == "HT-2026-001"
    assert output.degraded is False


@pytest.mark.anyio
async def test_missing_key_triggers_correction() -> None:
    """漏字段也要回灌（这正依赖 required 生效）。"""
    client = FakeLLMClient(
        replies=[
            json_reply(合同编号="HT-2026-001"),  # 漏了 甲方
            json_reply(合同编号="HT-2026-001", 甲方="张三"),
        ]
    )
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["合同编号", "甲方"])

    assert client.call_count == 2
    assert output.data["甲方"] == "张三"
    assert "Field required" in client.calls[1].all_texts()


# ============================================================ 4. 降级


@pytest.mark.anyio
async def test_persistent_failure_degrades_with_flag() -> None:
    """**两轮仍失败 → 降级规则版并标 degraded: true**（api-spec §2.5）。"""
    client = FakeLLMClient(replies=['{"合同编号": {"nested": 1}}'])  # 永远坏
    extractor = _extractor(client, max_correction_rounds=MAX_CORRECTION_ROUNDS)
    output = await extractor.extract(text=CONTRACT_TEXT, fields=["合同编号"])

    # 1 次初始 + 2 轮回灌
    assert client.call_count == MAX_CORRECTION_ROUNDS + 1
    assert output.degraded is True, "必须明确标记降级"
    assert "+fallback(" in output.model
    # 规则版能抽对（这才是降级的意义：不是没结果，是换个来源给结果）
    assert output.data["合同编号"] == "HT-2026-001"


@pytest.mark.anyio
async def test_network_error_degrades_immediately() -> None:
    """网络/鉴权失败不浪费轮数：直接降级。"""
    client = FakeLLMClient(exc=TimeoutError("boom"))
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["合同编号"])

    assert output.degraded is True
    assert "TimeoutError" in output.model or "RetryExhaustedError" in output.model
    assert output.data["合同编号"] == "HT-2026-001", "降级仍要给出结果"


@pytest.mark.anyio
async def test_degraded_result_still_satisfies_contract_shape() -> None:
    """降级结果也必须是契约形状（键序、string|null），否则契约就漏了一个口子。"""
    client = FakeLLMClient(exc=RuntimeError("boom"))
    output = await _extractor(client).extract(
        text=CONTRACT_TEXT, fields=["合同编号", "甲方", "不存在"]
    )

    assert list(output.data) == ["合同编号", "甲方", "不存在"]
    assert all(v is None or isinstance(v, str) for v in output.data.values())


@pytest.mark.anyio
async def test_empty_fields_returns_empty_without_calling_model() -> None:
    client = FakeLLMClient(replies=["{}"])
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=[])

    assert output.data == {}
    assert client.call_count == 0, "空字段不该花钱调模型"


# ============================================================ 5. 容错


@pytest.mark.anyio
@pytest.mark.parametrize(
    "content",
    [
        '{"合同编号": "HT-2026-001"}',
        '```json\n{"合同编号": "HT-2026-001"}\n```',
        '好的：{"合同编号": "HT-2026-001"} 以上。',
        wrapped_reply(合同编号="HT-2026-001"),
    ],
    ids=["plain", "fence", "chatter", "wrapped-data"],
)
async def test_tolerant_parsing_shapes(content: str) -> None:
    """常见模型"小动作"都要能解析：围栏、废话、`{"data":{...}}` 包装。"""
    client = FakeLLMClient(replies=[content])
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["合同编号"])

    assert output.data["合同编号"] == "HT-2026-001"
    assert output.degraded is False


@pytest.mark.anyio
async def test_null_strings_are_normalized() -> None:
    """模型把缺失写成字符串 `"null"` / `"None"` 也要归一成真 None。"""
    client = FakeLLMClient(replies=[json_reply(甲方="null", 合同编号="None")])
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["甲方", "合同编号"])

    assert output.data["甲方"] is None
    assert output.data["合同编号"] is None


@pytest.mark.anyio
async def test_none_content_raises_instead_of_becoming_empty_string() -> None:
    """`content=None` 必须当失败（推理吃光 max_tokens 的典型症状）。

    绝不能 `or ""` 静默成空串 —— 那会被下游当成"模型说没有"，是语义错误。
    """
    client = FakeLLMClient()
    client.replies = ["placeholder"]
    # 直接改剧本返回 None：模拟 content=None
    original_next = client._next_reply
    client._next_reply = lambda: None  # type: ignore[method-assign]

    try:
        output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["合同编号"])
    finally:
        client._next_reply = original_next  # type: ignore[method-assign]

    # content=None 会先走"解析失败 → 回灌"，最终用尽轮数后降级
    assert output.degraded is True
    assert output.data["合同编号"] == "HT-2026-001"


# ============================================================ 6. 装配与一致性


def test_env_switch_selects_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    """`EXTRACTOR_IMPL=rule` 一键回退 —— 应急开关必须是改配置，不是改代码。"""
    monkeypatch.setenv("EXTRACTOR_IMPL", "rule")
    reset_extractor_cache()
    try:
        assert isinstance(get_extractor(), RuleExtractor)
    finally:
        reset_extractor_cache()


def test_default_extractor_is_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXTRACTOR_IMPL", raising=False)
    reset_extractor_cache()
    try:
        assert isinstance(get_extractor(), LlmExtractor)
    finally:
        reset_extractor_cache()


def test_both_implementations_satisfy_protocol() -> None:
    assert isinstance(RuleExtractor(), Extractor)
    assert isinstance(LlmExtractor(FakeLLMClient(), model="m"), Extractor)  # type: ignore[arg-type]


def test_extract_signatures_are_identical() -> None:
    """**"签名保持不变"的可执行证明。**"""
    import inspect

    rule = inspect.signature(RuleExtractor.extract)
    llm = inspect.signature(LlmExtractor.extract)
    assert list(rule.parameters) == list(llm.parameters)
    assert rule.parameters["text"].kind is llm.parameters["text"].kind
    assert rule.parameters["fields"].kind is llm.parameters["fields"].kind


def test_rule_based_entry_used_for_fallback() -> None:
    """`extract_rule_based` 是降级用的函数式入口，必须与类方法行为一致。"""
    result = extract_rule_based(CONTRACT_TEXT, ["合同编号", "甲方"])
    assert result.data == {"合同编号": "HT-2026-001", "甲方": "张三"}
    assert result.degraded is False


def test_correction_rounds_are_bounded() -> None:
    """轮数必须有硬上限：每一轮都是一次计费调用。"""
    assert 1 <= MAX_CORRECTION_ROUNDS <= 3


def test_json_reply_helper_is_valid_json() -> None:
    assert json.loads(json_reply(甲方="张三")) == {"甲方": "张三"}


@pytest.mark.anyio
async def test_scalar_values_are_coerced_not_rejected() -> None:
    """**数字/布尔值要归一成字符串，而不是让校验失败。**

    契约（§2.1）要求值恒为 `string | null`，但"模型把金额写成 128000
    而不是 "128000""极其常见。严格拒绝它要浪费一整轮回灌（一次计费调用 +
    几秒延迟），而转换是确定性、零成本的，且结果正是契约要的东西。

    边界：**能在本地可靠修的，不要推给模型去修** —— 那是白花钱。
    """
    client = FakeLLMClient(replies=['{"合同金额": 128000, "甲方": null}'])
    output = await _extractor(client).extract(
        text="合同金额：128000 元", fields=["合同金额", "甲方"]
    )

    assert client.call_count == 1, "标量归一应该在本地完成，不该触发回灌"
    assert output.data["合同金额"] == "128000"
    assert isinstance(output.data["合同金额"], str)
    assert output.degraded is False


@pytest.mark.anyio
async def test_float_integer_is_coerced_without_trailing_zero() -> None:
    """`128000.0` 应变成 `"128000"` 而不是 `"128000.0"` —— 尽量贴近原文写法。"""
    client = FakeLLMClient(replies=['{"合同金额": 128000.0}'])
    output = await _extractor(client).extract(text="合同金额：128000 元", fields=["合同金额"])
    assert output.data["合同金额"] == "128000"


@pytest.mark.anyio
async def test_nested_structure_still_requires_correction() -> None:
    """嵌套对象/数组才交给模型去修 —— 那是真正的结构理解错误。"""
    client = FakeLLMClient(
        replies=['{"合同编号": {"nested": 1}}', json_reply(合同编号="HT-2026-001")]
    )
    output = await _extractor(client).extract(text=CONTRACT_TEXT, fields=["合同编号"])

    assert client.call_count == 2, "嵌套结构应触发回灌"
    assert output.data["合同编号"] == "HT-2026-001"
