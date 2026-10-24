"""应用配置：Settings + LLMSettings 嵌套模型。

本课（W1 周一）只加 LLM 相关的 5 个变量，但**写法要能撑到 W2**：

1. **嵌套模型 `Settings().llm.api_key`**
   `LLMSettings` 自己带 `env_prefix="LLM_"`，所以 .env 里写的是**扁平**的
   `LLM_API_KEY`，不需要 `LLM__API_KEY` 那种双下划线写法。实测（见笔记）：
   嵌套模型的 `model_config.env_prefix` 对 BaseSettings/BaseModel 都生效，
   它会在解析自己字段时回读环境里的 `LLM_*`。

2. **只在 `get_settings()` 里解析，import 时不解析**
   `@lru_cache` 包住 `get_settings()`：全进程只解析一次 .env。
   `import app.main` 不会因为缺 key 而炸；`Settings()` 第一次被调用（启动/首个请求）
   才校验 —— 早炸好过半夜炸。

3. **密钥用 SecretStr**
   `repr()` / `model_dump()` / 日志里恒为 `**********`，只有显式
   `.get_secret_value()` 才拿得到明文。client.py 是唯一该调它的地方。

4. **retries 的上下界是有意义的，不是装饰**
   `max_retries=0` 合法（明确表示不重试）；`>5` 直接把一次请求的尾延迟放大到不可控，
   在 FastAPI 的请求超时下会变成「重试到一半被掐断」——比不重试更糟。
   所以用 `ge=0, le=5` 把这类配置错误挡在启动时。

优先级（pydantic-settings 内建）：构造参数 > 真实环境变量 > .env > 字段默认值。
生产机上 `export LLM_API_KEY=...` 就能盖掉仓库里的 .env，不用改代码。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> app/core -> app -> 项目根（industrial-ai-platform）
BASE_DIR = Path(__file__).resolve().parents[2]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class LLMSettings(BaseSettings):
    """大模型接入配置。字段名去掉 `LLM_` 前缀后即 .env 里的键名。

    `env_prefix="LLM_"` 是这一层的全部魔法：.env 写扁平的 `LLM_API_KEY`，
    这里声明 `api_key`，两边就能对上。换成 `LLM__API_KEY`（双下划线）反而不认 ——
    那是 `env_nested_delimiter` 的写法，两套机制别混用。
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- 凭证与端点 ----
    api_key: SecretStr = Field(description="LLM API key，必填；日志与响应里只有掩码")
    base_url: str = Field(
        default="https://api.deepseek.com/v1",
        min_length=1,
        description="OpenAI 兼容端点，含 /v1；换供应商只改这里",
    )

    # ---- 模型与超时 ----
    model: str = Field(default="deepseek-chat", min_length=1, max_length=128)
    timeout: float = Field(default=30.0, gt=0, le=300, description="单次请求超时（秒）")

    # ---- 重试 ----
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="SDK 内部重试次数（0=不重试）；>5 会放大尾延迟，故设上限",
    )

    # ---------------- 规整：mode="before" 拿到原始值 ----------------
    @field_validator("base_url", mode="before")
    @classmethod
    def _strip_trailing_slash(cls, v: object) -> object:
        """`https://x/v1/` 和 `https://x/v1` 必须等价。

        SDK 内部是 `base_url + "/chat/completions"` 拼的，多一个斜杠会拼出
        `//chat/completions`。多数网关能容忍，但本地网关直接 404 ——
        这属于「不该让使用者记住」的细节，解析时就去掉。
        """
        return v.rstrip("/") if isinstance(v, str) else v

    @field_validator("api_key", mode="before")
    @classmethod
    def _reject_blank_key(cls, v: object) -> object:
        """空字符串比「没配」更危险：没配会在启动时报 missing，空串会一路跑到 401。

        直接归一成报错，让问题停在启动阶段。注意 `.env` 里写 `LLM_API_KEY=`
        得到的正是空串，这个分支不是假想。
        """
        if isinstance(v, str) and not v.strip():
            raise ValueError("LLM_API_KEY 不能为空字符串；不配置请整行删掉")
        return v

    # ---------------- 只读派生 ----------------
    @property
    def api_key_masked(self) -> str:
        """供日志/调试展示用，永远是 `**********`。"""
        return str(self.api_key)

    def public_info(self) -> dict[str, object]:
        """可安全对外暴露的摘要：key 恒为掩码。"""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "api_key": self.api_key_masked,
        }


class Settings(BaseSettings):
    """全局配置。

    `llm` 用 `default_factory=LLMSettings`：这个工厂**只在 `Settings()` 被实例化时**
    才执行（pydantic 的默认值是懒求值的），所以 `import app.core.config` 不会构造它，
    「import 时不解析」的设计目标仍然成立 —— 缺 key 只会在 `get_settings()`
    真正被调用时报错，也就是启动阶段。

    `LLMSettings` 自己带 `env_prefix`，所以它作为工厂被构造时同样是**从环境变量
    读值**的，而不是用字段默认值。实测确认过这一点（见 learn/llm/01_params.md）：
    换成普通默认值 `Field(default=None)` 会直接 ValidationError，
    因为 `None` 不是合法的 `LLMSettings` 实例。
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="industrial-ai-platform", min_length=1, max_length=60)
    log_level: LogLevel = Field(default="INFO", description="日志级别")

    llm: LLMSettings = Field(
        default_factory=LLMSettings,
        description="大模型接入配置，见 LLMSettings（LLM_* 环境变量）",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, v: object) -> object:
        """允许 .env 里写 info / Info，统一转大写再交给 Literal 校验。"""
        return v.upper() if isinstance(v, str) else v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """全进程单例。测试里要重载：`get_settings.cache_clear()`。"""
    return Settings()


def reset_settings_cache() -> None:
    """显式重置（改完环境变量后想重新解析时用）。"""
    get_settings.cache_clear()


__all__ = [
    "BASE_DIR",
    "LogLevel",
    "LLMSettings",
    "Settings",
    "get_settings",
    "reset_settings_cache",
]
