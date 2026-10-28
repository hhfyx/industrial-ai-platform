"""装配层测试：**默认的 llm 实现真的能被构造出来**。

## 这补的是 CI 唯一的盲区

其余所有测试都通过 `dependency_overrides` / autouse fixture 注入了**离线规则版**。
也就是说 `_build_llm_extractor()`（读 env → 校验 key → 造 client）
在 CI 里**一次都没被执行过**。

而默认实现是 `llm`。如果它装配有错：
- 缺 key 时不是报错而是静默通过（那样线上会一直降级而没人知道）
- `base_url` 格式不认（拼出 `//chat/completions`）
- client 构造参数不对（timeout/retries 传错位置）

CI 会全绿，而线上直接挂。**10/27 的事故正是这一层**：
网关凭据过期 → 降级 → 响应体多 `degraded` → 契约测试挂，而测试一行没改。

## 与 test_llm_client.py 的区别

那个文件测的是 `get_llm_client()`（**client** 的注入与单例）。
这里测的是 `get_extractor()`（**抽取器**的装配），
包括"默认走 llm 分支"和"缺 key 必须启动即失败"两条。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.core import deps
from app.core.config import LLMSettings, Settings, get_settings
from app.services.extractor import RuleExtractor
from app.services.llm_extractor import LlmExtractor


@pytest.fixture
def fresh_config() -> Iterator[None]:
    """清掉**配置**与**装配**两层缓存。

    **这是踩到的坑**：只清 `reset_extractor_cache()` 不够 ——
    `get_settings()` 自己也是 `lru_cache(maxsize=1)`，它把**第一次**解析出的
    Settings（可能来自仓库里的 `.env`，模型名 `buddy/glm-5.3`）一直缓存着。
    于是 `monkeypatch.setenv("LLM_MODEL", ...)` 完全不生效，
    抽取器拿到的还是旧模型名。

    生产里"只解析一次"是**对的**（配置不该热更新），但测试必须能重置。
    两层分开清，因为它们缓存的是不同东西：Settings（配置本身）
    和 extractor（由配置派生出的对象）。
    """
    deps.reset_extractor_cache()
    get_settings.cache_clear()
    yield
    deps.reset_extractor_cache()
    get_settings.cache_clear()


def _real_extractor():  # noqa: ANN202
    """取真实的 `get_extractor` 结果。

    `conftest.py` 的 autouse fixture 用 `app.dependency_overrides` 注入规则版，
    那只影响**走 FastAPI 的请求**，不影响直接调用 `deps.get_extractor()` ——
    所以这里直接调，测的是真实装配路径。
    """
    return deps.get_extractor()


# ============================================================ 默认装配


def test_default_extractor_is_llm_extractor(
    monkeypatch: pytest.MonkeyPatch, fresh_config: None
) -> None:
    """**默认实现必须是 LlmExtractor。**

    这条看似平凡，但它守着一个真实的失效模式：如果哪天有人把默认值改成
    `rule`（或环境变量读取写错），"服务在调模型"这句话就不再成立，
    而所有测试仍然全绿 —— 因为它们都注入了规则版。
    """
    monkeypatch.delenv("EXTRACTOR_IMPL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "sk-wiring-test-placeholder")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("LLM_MODEL", "wiring-test-model")

    extractor = _real_extractor()

    assert isinstance(extractor, LlmExtractor), (
        f"默认实现应为 LlmExtractor，实际 {type(extractor).__name__}"
    )
    assert extractor.model_name == "wiring-test-model"


def test_rule_switch_still_works(monkeypatch: pytest.MonkeyPatch, fresh_config: None) -> None:
    """`EXTRACTOR_IMPL=rule` 是应急开关，必须始终可用（否则降级手段就没了）。"""
    monkeypatch.setenv("EXTRACTOR_IMPL", "rule")
    monkeypatch.setenv("LLM_API_KEY", "sk-wiring-test-placeholder")

    assert isinstance(_real_extractor(), RuleExtractor)


# ============================================================ 缺 key 必须失败


def test_missing_api_key_fails_at_startup() -> None:
    """**缺 key 必须在构造时就抛**，不能静默通过。

    这是"早炸好过半夜炸"的具体化。如果缺 key 能装配成功，
    线上会表现为"每个请求都降级到规则版"，日志里只有一堆 warning ——
    看起来服务是好的，实际模型从没被调用过。

    用 `_env_file=None` 排除仓库里的 `.env`（那是开发机配置，
    CI/容器里不存在），确保测的是"真的没有 key"。
    """
    with pytest.raises(ValueError, match="api_key"):
        LLMSettings(_env_file=None, base_url="http://127.0.0.1:1/v1")


def test_blank_api_key_fails_at_startup() -> None:
    """空字符串也要炸 —— 它比"没配"更危险：会一路跑到 401。"""
    with pytest.raises(ValueError, match="LLM_API_KEY 不能为空字符串"):
        LLMSettings(_env_file=None, api_key="   ", base_url="http://127.0.0.1:1/v1")


# ============================================================ 装配不联网


def test_wiring_does_not_make_network_call(
    monkeypatch: pytest.MonkeyPatch, fresh_config: None
) -> None:
    """装配阶段**不发任何请求**。

    `base_url` 指向不可达地址（`127.0.0.1:1`）依然能构造出抽取器 ——
    证明 `get_extractor()` 只做对象构造，不做连通性检查。
    这正是 CI wiring smoke 能在几秒内跑完、且不花钱的原因。
    """
    monkeypatch.setenv("LLM_API_KEY", "sk-wiring-test-placeholder")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("LLM_MODEL", "unreachable-model")
    monkeypatch.delenv("EXTRACTOR_IMPL", raising=False)

    extractor = _real_extractor()

    assert isinstance(extractor, LlmExtractor)
    assert extractor.model_name == "unreachable-model"


# ============================================================ 设置项校验


def test_settings_rejects_invalid_retries() -> None:
    """`max_retries` 越界必须在解析期拦住（>5 会让尾延迟不可控）。"""
    with pytest.raises(ValueError):
        LLMSettings(_env_file=None, api_key="sk-x", max_retries=99)


def test_settings_strips_trailing_slash_on_base_url() -> None:
    """末尾斜杠要去掉 —— SDK 内部拼 `base_url + "/chat/completions"`。"""
    settings = LLMSettings(_env_file=None, api_key="sk-x", base_url="https://api.test/v1/")
    assert settings.base_url == "https://api.test/v1"


def test_settings_masks_api_key() -> None:
    settings = LLMSettings(_env_file=None, api_key="sk-secret-value-1234567890")
    assert str(settings.api_key) == "**********"
    assert "sk-secret-value" not in repr(settings)
    assert settings.api_key.get_secret_value() == "sk-secret-value-1234567890"


def test_parent_settings_constructs_nested_llm() -> None:
    """`Settings().llm` 必须是可用的 `LLMSettings` 实例。

    父级的 `default_factory=LLMSettings` 是**惰性求值**的：
    import 时不构造，只有实例化 `Settings` 才构造 —— 所以 import 阶段不会因为
    缺 key 而失败，符合"启动时才炸"的设计。
    """
    settings = Settings(_env_file=None)
    assert isinstance(settings.llm, LLMSettings)
