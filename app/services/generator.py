"""图像生成服务：`Generator` 协议 + 假实现 + 真实现的装配位。

## 分层与签名（复用 W1/W2 的既定架构）

与 `extractor.py` 完全同构：

- **services 层不 import fastapi**：`Generator` 是纯 Python 协议，
  路由层通过 `deps` 的 `GeneratorDep` 拿到实现 —— 换实现只动装配，
  路由 / schema / 契约文档 / 测试零改动（api-spec §4.7 同一条承诺）。
- **内部结果用 dataclass，对外响应用 schemas 的 BaseModel**：
  两层形状解耦，`GenerationOutput` 怎么改都不牵动契约校验。

## 真实实现在哪

`SdLoraGenerator` 的推理内核**留壳**：本机没有 diffusers / GPU 权重，
现在写它只能写出"没跑过一次的代码"。签名先定（`generate()` 的出入参），
权重到位后填 `_load_pipeline()` 与 `_run_pipeline()` —— 契约不因实现变。
W2 换 LLM 时验证过这条路：`RuleExtractor` → `LlmExtractor` 同签名替换，
`tests/test_api.py` 一行没改。

**已验证的部分是真的**：FakeGenerator 走完整 HTTP 链路（含错误映射），
四类失败路径在离线测试里全部可复现 —— 那是本周验收"测试绿"的本体。

## 四类失败的归属

| 失败 | 谁抛 | 映射 |
| --- | --- | --- |
| 非法类别 | schema 层（422） | `UNKNOWN_DEFECT_CLASS` —— 在进服务层之前就拦住 |
| OOM | 真实现（torch.cuda.OutOfMemoryError） | 503 `GPU_OOM` |
| 超时 | 真实现（TimeoutError） | 504 `GENERATION_TIMEOUT` |
| 权重未加载 | 装配层（构造时检查） | 503 `MODEL_NOT_LOADED` |

OOM/超时/权重是 **5xx 但可重试语义不同**：503 带 `Retry-After`（等显存释放），
504 不建议立即重试（任务本身太重）。错误码只追加不改旧值（W1 规则）。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from app.schemas.generate import ALLOWED_DEFECT_CLASSES

logger = logging.getLogger(__name__)

# 生成图落盘目录：项目内 data/（与 uploads / chroma / sessions 同一惯例，
# 不落系统 Temp —— 重启就丢）。
OUTPUT_DIR: Final[Path] = Path("data") / "generated"

# 单张生成超时（秒）。768px 单张约 8–15 秒，8 张批处理给足 180s ——
# 超时必须是显式常量而不是"等它自己报错"：挂死的 CUDA 调用不会自己超时。
GENERATION_TIMEOUT_S: Final[float] = 180.0


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    """服务层内部结果（不是对外的 pydantic 模型）。"""

    images: list[dict[str, object]]
    """每张图：{"path": 相对路径, "size": 边长, "seed": 实际种子}。"""

    model: str = "sd15-lora-v0"
    elapsed_ms: int = 0
    degraded: bool = False
    notes: list[str] = field(default_factory=list)
    """给日志/头的补充信息（如"部分图降级为 512"），不进契约体。"""


class GeneratorUnavailableError(RuntimeError):
    """权重未加载 / 设备不可用。装配层或实现抛，路由映射 503 MODEL_NOT_LOADED。"""


class GenerationOOMError(RuntimeError):
    """显存不足。真实现抛，路由映射 503 GPU_OOM（带 Retry-After）。"""


class GenerationTimeoutError(TimeoutError):
    """生成超时。实现抛（asyncio.wait_for 的 TimeoutError 子类），映射 504。"""


@runtime_checkable
class Generator(Protocol):
    """生成器的**唯一接口**（与 `Extractor` 协议同一条规矩）。

    `runtime_checkable` 让测试能 `isinstance(fake, Generator)` 验证装配。
    """

    model_name: str

    async def generate(
        self,
        *,
        defect_class: str,
        count: int,
        size: int,
        seed: int | None = None,
    ) -> GenerationOutput: ...


class FakeGenerator:
    """离线假实现：不碰 GPU、不碰网络，秒回。

    **为什么默认装配它而不是真实现**：CI 是断网的，GPU 权重是几十 GB，
    本机开发也未必有卡。契约测试走 FakeGenerator 才能在 `pytest` 里全绿；
    真实现的验证是**部署后的事**（`scripts/check_llm_endpoint.py` 同一模式）。

    它不是 mock：真实地写盘（几字节的 PNG 头）、真实地计时、
    真实地生成种子 —— 测的是**链路行为**（响应形状、错误映射、头字段），
    不是测"diffusers 调对了没"（那在真机验收里做）。
    """

    model_name: str = "fake-generator"

    def __init__(self, *, output_dir: Path | None = None, latency_ms: int = 5) -> None:
        self._output_dir = output_dir or OUTPUT_DIR
        self._latency_ms = latency_ms

    async def generate(
        self,
        *,
        defect_class: str,
        count: int,
        size: int,
        seed: int | None = None,
    ) -> GenerationOutput:
        started = time.perf_counter()
        # 占住 async 语义（与 RuleExtractor 的 sleep 同一理由）
        await asyncio.sleep(self._latency_ms / 1000)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        base_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        images: list[dict[str, object]] = []
        for i in range(count):
            item_seed = base_seed + i
            path = self._output_dir / f"{defect_class}_{size}_{item_seed}.png"
            # 最小合法 PNG 头：让文件真实存在（UI/静态服务能回 200），
            # 内容当然不是真图 —— Fake 的语义边界要诚实。
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
            images.append({"path": path.as_posix(), "size": size, "seed": item_seed})

        return GenerationOutput(
            images=images,
            model=self.model_name,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )


class SdLoraGenerator:
    """SD1.5 + LoRA 真实现的**壳**：装配与异常映射就位，推理内核待权重到位后填。

    `_load_pipeline` / `_run_pipeline` 是留给真内核的两个注入点；
    本类现在抛 `GeneratorUnavailableError` —— 让"权重没到位"在**装配时**
    就变成 503 MODEL_NOT_LOADED，而不是请求进来后在深层栈里炸。
    """

    model_name: str = "sd15-lora-v0"

    def __init__(self, *, output_dir: Path | None = None) -> None:
        self._output_dir = output_dir or OUTPUT_DIR

    async def generate(
        self,
        *,
        defect_class: str,
        count: int,
        size: int,
        seed: int | None = None,
    ) -> GenerationOutput:
        raise GeneratorUnavailableError(
            "SD+LoRA 推理内核尚未装配（本机无 diffusers/GPU 权重）。"
            "装配后本方法内部调用 _load_pipeline/_run_pipeline，签名不变。"
        )


def validate_defect_class(defect_class: str) -> str:
    """白名单校验（422 UNKNOWN_DEFECT_CLASS 的判定函数）。

    放服务层而不是 schema 的 Literal：类别清单会随权重目录增长，
    Literal 会让"加一个类别"变成改契约文件 —— 白名单是**配置**不是**形状**。
    schema 层只管"非空、长度"，表内表外由这里判。
    """
    if defect_class not in ALLOWED_DEFECT_CLASSES:
        raise ValueError(
            f"defect_class 必须是 {'、'.join(ALLOWED_DEFECT_CLASSES)} 之一，收到 {defect_class!r}"
        )
    return defect_class


def available_classes() -> Sequence[str]:
    return ALLOWED_DEFECT_CLASSES


__all__ = [
    "GENERATION_TIMEOUT_S",
    "OUTPUT_DIR",
    "FakeGenerator",
    "GenerationOOMError",
    "GenerationOutput",
    "GenerationTimeoutError",
    "Generator",
    "GeneratorUnavailableError",
    "SdLoraGenerator",
    "available_classes",
    "validate_defect_class",
]
