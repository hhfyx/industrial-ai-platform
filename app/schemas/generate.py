"""缺陷图像生成的接口契约（`POST /api/v1/generate`，项目一 · W8 周三）。

**为什么单独一个文件**：与 `chat.py` 同一条纪律 —— 每条接口线一个契约文件。
`extract.py` 是冻结的 v1 契约，本项目一的新端点碰它一下都是事故。

## 复用什么、新造什么

| 项 | 来源 | 说明 |
| --- | --- | --- |
| 统一错误体 4 字段 | W1（`ErrorResponse`） | **直接 import，零复制** |
| 错误码机制 | W1（`AppError` + 全局 handler） | 新错误码按"只追加"规则进 §3.1 |
| `request_id` 透传 | W1（中间件 + `RequestIdDep`） | 中间件已全局生效，路由只读 |
| 响应体形状 | **新** | 生成没有 `confidence` 语义，不硬套 extract 的 4 字段 |

## 参数上限为什么是契约的一部分，不是实现细节

`count ≤ 8`、`size ∈ {512, 768}` 写在 schema 里（422 拒绝），不是文档里的"建议"：

- 一次 SD 推理的显存占用 ≈ f(size²)，8 张 768px 在 24G 卡上就是边界；
- **没有上限的生成接口等于自杀**：`count=1000` 的恶意/失误请求会占住 GPU
  直到 OOM，而且这个 OOM 会连带打死排在后面的正常请求。
- 上限进契约的另一个收益：调用方在写代码时就能看到边界（/docs 的 description），
  而不是运行时撞了 500 才知道。

## defect_class 为什么是白名单而不是自由字符串

LoRA 权重按缺陷类别组织（每个类别一份适配器）。自由字符串意味着
"服务端静默用基座模型生成"—— 调用方以为在生成划痕，实际是随机噪声。
**宁可 422，不可静默生成错误类别**：这与 extract 契约"抽不到 = null 不编造"
是同一条产品哲学的生成版。
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# ---------------------------------------------------------------- 约束常量
# 进契约（api-spec.md 项目一段 §1.1 的表格引用这几个数），改它们 = 改契约。

MAX_COUNT = 8
"""单次生成张数上限。SD 推理是批处理：8 张 768px 约 10–15 秒 / 显存近满，
超过这个数应该走异步版（本周不做），不是调大这个数。"""

ALLOWED_SIZES: tuple[Literal[512, 768], ...] = (512, 768)
"""允许的输出边长（正方形）。两档的理由：
- 512 是 SD1.5 的原生分辨率（训练分布内，畸形率最低）
- 768 是放大档，显存 ×2.25、耗时 ×2 左右
其它尺寸（如 600）会让模型出分布外的畸形图 —— 限制取值比报错更早。"""

ALLOWED_DEFECT_CLASSES: tuple[str, ...] = (
    "划痕",
    "凹坑",
    "气泡",
    "裂纹",
    "锈蚀",
    "污渍",
)
"""缺陷类别白名单 = LoRA 适配器的目录名。**新类别 = 加权重 + 加这里一行**，
两件事必须在同一个提交里（与 FOLDERS.md"路径即约定"同一纪律）。"""

DefectClass = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=32),
]


class GenerateRequest(BaseModel):
    """生成请求体。

    `extra="forbid"` 与 extract 同款：多传一个键立刻 422 ——
    `{"defect_class": "划痕", "count": 2, "negative_prompt": "..."}` 这种
    "顺手多传一个"不被静默忽略，否则调用方以为参数生效了。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    defect_class: DefectClass = Field(
        description=(
            f"缺陷类别，白名单：{'、'.join(ALLOWED_DEFECT_CLASSES)}。"
            "白名单 = LoRA 适配器目录名；表外类别返回 422（UNKNOWN_DEFECT_CLASS），"
            "不会静默用基座模型生成。"
        )
    )
    count: int = Field(
        default=1,
        ge=1,
        le=MAX_COUNT,
        description=f"生成张数，1–{MAX_COUNT}。生成是 GPU 批处理，上限是显存边界不是产品策略。",
    )
    size: Literal[512, 768] = Field(
        default=512,
        description=(
            f"输出图边长（正方形），只允许 {ALLOWED_SIZES[0]} 或 {ALLOWED_SIZES[1]}。"
            "512 是 SD1.5 原生分辨率；768 显存 ×2.25。其它尺寸出分布外畸形图。"
        ),
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description=(
            "随机种子，可选。传了则同参数可复现（数据集标注场景需要）；"
            "不传则服务端随机。负数按 422 处理（与 diffusers 的行为一致，提前拦）。"
            "约束用 Field 的 ge 而不是 model_validator：after-validator 的报错 loc "
            "是 _model，字段级约束才能让 details 精确指到 seed。"
        ),
    )


class GeneratedImage(BaseModel):
    """一张生成图的落盘信息。**只回路径与元数据，不回 base64**——
    图像走静态文件服务（/ui 同款 StaticFiles），JSON 响应里塞 base64
    会让 8 张 768px 的响应体膨胀到 10MB+，且客户端无法断点续传。
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="图片相对路径（静态服务根下），可直接 GET")
    size: int = Field(description="输出边长，与请求一致")
    seed: int = Field(description="实际使用的种子（请求未传时为服务端生成的值）")


class GenerateResult(BaseModel):
    """成功响应体。

    与 extract 的 4 字段刻意不同（没有 confidence）：生成的"置信度"不存在，
    硬造一个只会污染两个契约的语义。共有的只有 `request_id` 与 `elapsed_ms`
    —— 那两条是平台级约定，不属于某一条接口线。
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(description="本次请求 id，与响应头 X-Request-ID 一致；报障只报它")
    images: list[GeneratedImage] = Field(
        description="生成结果，长度 = 请求的 count；顺序即批次顺序"
    )
    elapsed_ms: int = Field(
        ge=0,
        description="服务端处理耗时（毫秒，含 GPU 推理与存盘），不含网络往返",
    )
    model: str = Field(description="产出该结果的实现标识（头里同步带 X-Generator-Model）")


__all__ = [
    "ALLOWED_DEFECT_CLASSES",
    "ALLOWED_SIZES",
    "MAX_COUNT",
    "GenerateRequest",
    "GenerateResult",
    "GeneratedImage",
]
