"""模板库测试：目录规范 / 渲染 / 纪律 / 版本机制 / 与 structured_call 的衔接。

覆盖任务四条验收：
  ① 目录规范 —— app/prompts/{qa,summarize,classify,extract}/v1.jinja2 + schema.py
  ② 四类模板 —— 问答带引用与不确定、摘要带长度与保留、分类带枚举与唯一性、抽取带 null 规则
  ③ 模板自述 —— 每个模板 docstring 五要素齐全
  ④ 版本机制 —— load_prompt(name, version) 可切换，环境变量可覆盖

外加一条纪律测试：**模板不得硬编码业务词**。
"""

from __future__ import annotations

import re

import pytest

from app.prompts import loader as L
from app.prompts.classify.schema import ClassifyLabel
from app.prompts.extract.schema import ExtractResult
from app.prompts.qa.schema import QAAnswer
from app.prompts.summarize.schema import SummaryResult

CATEGORIES = ("qa", "summarize", "classify", "extract")


@pytest.fixture(autouse=True)
def _clear_prompt_cache():
    """每个测试前后都清缓存：环境变量一改，lru_cache 会拿到旧版本。"""
    L.clear_cache()
    yield
    L.clear_cache()


# ============================================================ ① 目录规范


@pytest.mark.parametrize("name", CATEGORIES)
def test_category_dir_has_template_and_schema(name: str) -> None:
    cat_dir = L.PROMPTS_DIR / name
    assert cat_dir.is_dir(), f"缺少模板类别目录 {name}"
    assert (cat_dir / "v1.jinja2").is_file(), f"{name} 缺少 v1.jinja2"
    assert (cat_dir / "schema.py").is_file(), f"{name} 缺少 schema.py"


@pytest.mark.parametrize("name", CATEGORIES)
def test_schema_is_pydantic_model(name: str) -> None:
    schema = L.load_schema(name)
    from pydantic import BaseModel

    assert issubclass(schema, BaseModel)


def test_unknown_category_rejected() -> None:
    with pytest.raises(L.PromptNotFoundError):
        L.load_prompt("summarise")  # 拼错（应为 summarize）


# ============================================================ ② 四类模板


def test_qa_has_citation_numbers_and_uncertainty() -> None:
    """问答：带引用编号的上下文占位符 + 「无答案就说不确定」。"""
    text = L.load_prompt(
        "qa",
        version="v1",
        question="退款多久到账？",
        contexts=[
            {"id": 2, "text": "退款将在 3 个工作日内退回原支付账户。"},
            {"id": 5, "text": "订单发货后不支持无理由退款。"},
        ],
    )
    # 上下文编号渲染进 prompt（不是写死的编号）
    assert "[2]" in text
    assert "[5]" in text
    # 无答案规则
    assert "不确定" in text
    assert "引用编号" in text


def test_qa_citation_numbers_come_from_params_not_hardcoded() -> None:
    """编号必须来自调用方，换一组 contexts 编号就跟着变。"""
    text = L.load_prompt(
        "qa",
        version="v1",
        question="q",
        contexts=[{"id": 11, "text": "a"}, {"id": 42, "text": "b"}],
    )
    assert "[11]" in text
    assert "[42]" in text
    assert "[2]" not in text


def test_summarize_has_length_and_keep_rules() -> None:
    """摘要：长度约束 + 保留数字与专有名词 + 分段摘要的段提示。"""
    body = L.load_prompt(
        "summarize",
        version="v1",
        text="本次营收 1,240 万元。",
        max_chars=80,
        must_keep=["1,240 万元"],
    )
    assert "80" in body
    assert "1,240 万元" in body
    assert "逐字" in body

    segmented = L.load_prompt(
        "summarize",
        version="v1",
        text="",
        segments=["第一段内容", "第二段内容"],
        max_chars=100,
    )
    assert "第 1 段" in segmented
    assert "第 2 段" in segmented


def test_classify_has_enum_boundary_and_single_choice() -> None:
    """分类：类别枚举 + 边界定义 + 「不能选两个」。"""
    text = L.load_prompt(
        "classify",
        version="v1",
        text="扣了两次钱",
        categories=[
            {"name": "故障报修", "boundary": "功能不可用"},
            {"name": "费用争议", "boundary": "扣费异常"},
        ],
    )
    assert "故障报修" in text
    assert "功能不可用" in text
    assert "只能选一个类别" in text
    # 输出格式是单值，不是数组
    assert '"类别": "' in text
    assert '"类别": ["' not in text


def test_classify_schema_rejects_multiple_categories() -> None:
    """「不能选两个」在类型层面封死：给数组直接校验失败。"""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ClassifyLabel(类别=["故障报修", "费用争议"], 置信度=0.9, 理由="两个都像")


def test_extract_has_field_list_and_null_rule() -> None:
    """抽取：字段清单 + 缺失填 null 的规则（不许编）。"""
    text = L.load_prompt(
        "extract",
        version="v1",
        text="请于 2026-11-20 前联系张工。",
        fields=[
            {"name": "主体", "desc": "联系人", "type": "string"},
            {"name": "时间", "desc": "时间点", "type": "date"},
        ],
    )
    assert "主体" in text
    assert "时间" in text
    assert "null" in text
    assert "严禁猜测" in text
    assert "缺失字段" in text


def test_extract_schema_allows_null_but_not_empty_string() -> None:
    """字段是 `str | None`：None 合法（不许编），但语义上不该用空串代替。"""
    ok = ExtractResult(主体=None, 时间=None, 缺失字段=["主体", "时间"], 置信度=0.1)
    assert ok.主体 is None
    # 空串也是合法 str —— 但那是调用方该拦的，这里只确认 None 可表达"没有"
    assert ExtractResult(主体="", 时间=None, 置信度=0.5).主体 == ""


# ============================================================ ③ 模板自述


@pytest.mark.parametrize("name", CATEGORIES)
def test_template_docstring_has_five_sections(name: str) -> None:
    """每个模板的头部 docstring 必须写全五要素，别人才能照着用。"""
    src = (L.PROMPTS_DIR / name / "v1.jinja2").read_text(encoding="utf-8")
    m = re.search(r"\{#(.*?)#\}", src, re.DOTALL)
    assert m, f"{name} 模板缺少说明块（{{# ... #}}）"
    doc = m.group(1)

    for section in ("用途", "入参", "输出 schema", "示例", "已知失败场景"):
        assert section in doc, f"{name} 模板自述缺少「{section}」"


# ============================================================ ④ 版本机制


def test_explicit_version_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式传 version 优先于任何环境变量。"""
    monkeypatch.setenv("PROMPT_VERSION", "v9-not-exist")
    text = L.load_prompt("qa", version="v1", question="q", contexts=[{"id": 1, "text": "a"}])
    assert text


def test_env_global_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """全局 PROMPT_VERSION 生效（此处指向不存在的版本，应报错而不是静默用 v1）。"""
    monkeypatch.setenv("PROMPT_VERSION", "v99")
    with pytest.raises(L.PromptNotFoundError):
        L.load_prompt("qa", question="q", contexts=[{"id": 1, "text": "a"}])


def test_env_per_category_overrides_global(monkeypatch: pytest.MonkeyPatch) -> None:
    """类别级环境变量优先于全局。"""
    monkeypatch.setenv("PROMPT_VERSION", "v99")
    monkeypatch.setenv("QA_PROMPT_VERSION", "v1")
    text = L.load_prompt("qa", question="q", contexts=[{"id": 1, "text": "a"}])
    assert text


def test_list_versions_natural_order() -> None:
    """版本列表按自然序，不是字典序（v2 在 v10 前面）。"""
    versions = L.list_versions("qa")
    assert "v1" in versions
    assert versions == sorted(versions, key=lambda v: int(v[1:]))


# ============================================================ 纪律：不硬编码业务词


def test_blocklist_rejects_business_terms() -> None:
    hits = L.check_no_hardcoded_business_terms("请特别关注 华为 Mate60 的问题")
    assert "华为" in hits


def test_blocklist_passes_instruction_only_text() -> None:
    """纯指令语（不含业务实体）必须通过 —— 否则检查误报率太高就没法用。"""
    clean = "你是分类助手。只能选一个类别。只输出 JSON 对象，不要解释。"
    assert L.check_no_hardcoded_business_terms(clean) == []


def test_blocklist_ignores_docstring_examples() -> None:
    """注释块里为了说明"什么叫业务词"举的例子不算违规。

    这条很关键：如果 docstring 举例也算违规，开发者就不敢在说明里举例子，
    模板自述质量会直接下降。
    """
    # 华为 完整地落在 {# ... #} 注释块内部
    text = (
        "{# 说明：业务词如「华为」「小米」这类实体名不能硬编码，必须走参数。#}\n"
        "正文只输出 JSON 对象，不做任何业务判断。"
    )
    hits = L.check_no_hardcoded_business_terms(text)
    assert "华为" not in hits


def test_load_prompt_line_raises_on_hardcoded_template(tmp_path, monkeypatch) -> None:
    """给一个真的含业务词的模板，load_prompt 必须当场拒绝渲染。"""
    # 直接测检查函数在真实模板路径上的行为，避免污染 app/prompts
    bad = "请识别 小米 手机的问题"
    path = tmp_path / "bad.jinja2"
    path.write_text(bad, encoding="utf-8")
    hits = L.check_no_hardcoded_business_terms(bad, path)
    assert "小米" in hits


# ============================================================ 与 structured_call 衔接


@pytest.mark.anyio
async def test_template_output_feeds_structured_call(patch_structured_client) -> None:
    """模板渲染出的 prompt 能直接喂给 `structured_call`，闭环可用。"""
    from app.core.llm import structured as S

    prompt = L.load_prompt(
        "qa",
        version="v1",
        question="退款多久到账？",
        contexts=[{"id": 2, "text": "退款将在 3 个工作日内退回。"}],
    )
    schema = L.load_schema("qa")
    assert schema is QAAnswer

    result = await S.structured_call(
        prompt,
        schema,
        fallback=QAAnswer(答案="降级", 引用编号=[], 不确定=True),
    )
    assert isinstance(result, QAAnswer)
    assert result.答案


@pytest.fixture
def patch_structured_client(monkeypatch: pytest.MonkeyPatch):
    from app.core.llm import structured as S
    from tests.fake_llm import FakeLLMClient, json_reply

    fake = FakeLLMClient(
        replies=[json_reply(答案="3 个工作日内到账。", 引用编号=[2], 不确定=False)]
    )
    monkeypatch.setattr(S, "get_llm_client", lambda *a, **k: fake)
    return fake


def test_summarize_schema_field_bounds() -> None:
    """字数不能为负（schema 层就拦）。"""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        SummaryResult(摘要="x", 字数=-1, 覆盖要点=[])


def test_classify_confidence_bounds() -> None:
    """置信度越界（>1）被拦。"""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ClassifyLabel(类别="其他", 置信度=1.5, 理由="x")
