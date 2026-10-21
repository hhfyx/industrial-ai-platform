"""服务层单元测试（任务 14:30–15:30）：正常 / 空文本 / 超长 / 畸形四种输入。

**为什么先测服务层、再测 API**：服务层是 W2 要整个换掉的那一层。
把它锁死（输入 -> 输出的确定性行为），换实现时这批用例仍要全绿；
而 API 测试锁的是契约形状。两层各锁一件事，才不会"改实现把契约带崩"。

服务层刻意**不知道**长度上限 —— 上限是 `schemas.ExtractRequest` 的职责。
所以这里"超长文本"验的不是报错，而是"服务层不会因为长文本崩掉、也不会截断"。
"""

from __future__ import annotations

import asyncio

import pytest

from app.services.extractor import (
    SIMULATED_LATENCY_MS,
    RuleExtractor,
    compute_confidence,
    extract_value,
)

CONTRACT_TEXT = "合同编号：HT-2026-001，甲方：张三，签订日期：2026-10-20，合同金额：¥128,000.00"
ALL_FIELDS = ["合同编号", "甲方", "签订日期", "合同金额"]


def _run(text: str, fields: list[str]):
    """同步跑 async 抽取器，避免给纯计算测试引入事件循环样板代码。"""
    return asyncio.run(RuleExtractor().extract(text=text, fields=fields))


# ============================================================ ① 正常输入
def test_extracts_all_four_fields_in_request_order() -> None:
    """四个字段全命中，且 `data` 的键顺序与 `fields` 完全一致（契约 §2.1）."""
    output = _run(CONTRACT_TEXT, ALL_FIELDS)

    assert output.data == {
        "合同编号": "HT-2026-001",
        "甲方": "张三",
        "签订日期": "2026-10-20",
        "合同金额": "128,000.00",
    }
    # 顺序是契约的一部分：下游可能按下标取，不能靠"恰好对上"
    assert list(output.data) == ALL_FIELDS
    assert output.confidence == 1.0
    assert output.missing == []
    assert output.model == "rule-v1"


def test_amount_keeps_original_format_and_date_is_normalized() -> None:
    """金额保留 `,`（调用方要对账），日期统一成 `YYYY-MM-DD`（契约承诺）."""
    output = _run("合同金额：人民币 1,234.50 元；签订日期：2026年10月9日", ["合同金额", "签订日期"])

    assert output.data["合同金额"] == "1,234.50"
    assert output.data["签订日期"] == "2026-10-09"


def test_supports_english_field_aliases() -> None:
    """英文字段名走同一套规则（W2 接 LLM 后调用方不必换字段名）."""
    output = _run("contract_no: HT-2026-7781  sign_date: 2026/10/01", ["contract_no", "sign_date"])

    assert output.data == {"contract_no": "HT-2026-7781", "sign_date": "2026-10-01"}


@pytest.mark.anyio
async def test_async_elapsed_ms_covers_simulated_latency() -> None:
    """`elapsed_ms` 必须把内部延迟算进去 —— 否则 W2 换成真实模型调用时这个字段会失真."""
    output = await RuleExtractor().extract(text=CONTRACT_TEXT, fields=ALL_FIELDS)

    assert output.elapsed_ms >= SIMULATED_LATENCY_MS
    assert output.elapsed_ms < SIMULATED_LATENCY_MS + 1000


# ============================================================ ② 空文本
def test_empty_text_returns_all_none_with_zero_confidence() -> None:
    """空文本不抛异常：抽不到就是 `null`，置信度 0（契约 §3.1 的"永不编造"）."""
    output = _run("", ALL_FIELDS)

    assert output.data == dict.fromkeys(ALL_FIELDS)
    assert output.confidence == 0.0
    assert output.missing == ALL_FIELDS


def test_whitespace_only_text_is_treated_as_empty() -> None:
    """只有空白的正文等同于空文本 —— 空白里不可能有可抽的值."""
    output = _run("   \n\t  ", ALL_FIELDS)

    assert set(output.data.values()) == {None}
    assert output.confidence == 0.0


# ============================================================ ③ 超长文本
def test_long_text_is_not_truncated_and_still_extracts() -> None:
    """30000 字正文：服务层不截断、不崩，标签在末尾也照样抽得到.

    长度上限归 schema 管（超长在 API 层返回 422，见 test_api.py）。
    服务层若自己截断，会出现"API 说不超长、值却被切掉"的鬼故事。
    """
    padding = "无关段落。" * 5000  # 30000 字，远超 20000 上限
    text = padding + CONTRACT_TEXT
    assert len(text) > 20000

    output = _run(text, ALL_FIELDS)

    assert output.data["合同编号"] == "HT-2026-001"
    assert output.data["甲方"] == "张三"
    assert output.confidence == 1.0


# ============================================================ ④ 畸形输入
def test_garbage_text_yields_nulls_instead_of_raising() -> None:
    """完全无标签的乱码：全部 `null`，不抛异常（抽取器的失败模式是"没抽到"）."""
    output = _run("%%%@@@###   ！！！", ALL_FIELDS)

    assert set(output.data.values()) == {None}
    assert output.confidence == 0.0
    assert output.missing == ALL_FIELDS


def test_malformed_date_is_dropped_not_returned_as_dirty_value() -> None:
    """形似日期但非法（13 月 45 日）宁可返回 `None`，也不返回脏值污染下游."""
    assert extract_value("签订日期：2026-13-45", "签订日期") is None


def test_label_without_value_returns_none() -> None:
    """只有标签没有值（`甲方：` 后面直接句读符）也算没抽到."""
    assert extract_value("合同编号：，甲方：。", "合同编号") is None
    assert extract_value("合同编号：，甲方：。", "甲方") is None


def test_unknown_field_name_falls_back_to_generic_label_match() -> None:
    """表外字段名不报错：`<字段名>：<值>` 能抽到就抽，抽不到是 `null`（契约 §1.1）."""
    assert extract_value("联系人：李四", "联系人") == "李四"
    assert extract_value("正文里没有这个标签", "联系人") is None


def test_partial_hits_lower_confidence() -> None:
    """缺 1 个必抽字段：命中率 2/3 再打 0.75 折 = 0.5（让下游能按分数排序）."""
    output = _run("合同编号：HT-2026-001，签订日期：2026-10-20", ["合同编号", "签订日期", "甲方"])

    assert output.data["甲方"] is None
    assert output.confidence == 0.5


@pytest.mark.parametrize(
    ("fields", "data", "expected"),
    [
        ([], {}, 0.0),
        (["a"], {"a": "x"}, 1.0),
        (["a"], {"a": None}, 0.0),
        (["a", "b"], {"a": "x", "b": None}, 0.375),
    ],
)
def test_compute_confidence_is_deterministic(
    fields: list[str], data: dict[str, str | None], expected: float
) -> None:
    """置信度公式单独可测：换实现时这个公式是**契约语义**，不能悄悄变."""
    assert compute_confidence(fields, data) == expected


def test_extractor_is_stateless_across_calls() -> None:
    """同一个实例连续抽两次，结果一致 —— 单例（lru_cache）才敢这么用."""
    extractor = RuleExtractor()
    first = asyncio.run(extractor.extract(text=CONTRACT_TEXT, fields=ALL_FIELDS))
    second = asyncio.run(extractor.extract(text=CONTRACT_TEXT, fields=ALL_FIELDS))

    assert first.data == second.data
    assert first.confidence == second.confidence
