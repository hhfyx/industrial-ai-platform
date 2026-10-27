"""对外的 API 契约（请求体 / 响应体 / 错误体）。

**本文件 = `docs/api-spec.md` 的可执行版本。** 契约里写的字段名、类型、约束，
在这里必须是同一个东西，否则文档就成了摆设。

三条硬约束：

1. **入参 `extra="forbid"`**：多传一个键立刻 422，拼错的字段名不会被静默忽略；
2. **出参 `extra="forbid"`**：出参多返回一个键会变成 500（ResponseValidationError），
   所以出参模型必须是"服务层能保证不多给"的严格形状 —— 这正是契约的意义；
3. **每个字段都有 `description`**：它们会原样出现在 `/openapi.json`，也就是 `/docs` 里，
   写字段说明 = 写文档，不另维护一份。

W2 换 LLM 实现时，本文件**不需要改**（见 api-spec.md §4）。
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

# 约束常量：api-spec.md §1.1 的表格引用的就是这几个数
MAX_TEXT_LENGTH = 20000
MAX_FIELD_COUNT = 50
MAX_FIELD_NAME_LENGTH = 64

# 入参严格（多一个键就报错）；出参同样严格（服务层不许偷偷多给字段）
STRICT_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True)

# 单个字段名：去空白后 1–64 字符。用 Annotated 而不是逐个字段校验，
# 是为了让 pydantic 报错时 loc 精确到 `fields.2`，前端才能高亮到具体某一项。
FieldName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_FIELD_NAME_LENGTH),
]


class ExtractRequest(BaseModel):
    """抽取请求体（api-spec.md §1.1）。

    只有两个字段，刻意保持窄：`text` 是要抽的正文，`fields` 是要抽什么。
    任何 "mode"、"model"、"temperature" 之类实现参数都**不进契约** ——
    换实现时调用方不该被迫改请求。
    """

    model_config = STRICT_CONFIG

    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description=(
            "待抽取的正文纯文本，去首尾空白后不能为空；"
            f"长度上限 {MAX_TEXT_LENGTH} 字符。超长返回 422，不是 413。"
        ),
    )
    fields: list[FieldName] = Field(
        min_length=1,
        max_length=MAX_FIELD_COUNT,
        description=(
            "期望抽取的字段名列表，1–50 项，每项去空白后 1–64 字符，不允许重复。"
            "中英文均可（如 合同编号 / contract_no）。"
            "字段名不在服务端字典里不会报错，只是取值为 null。"
        ),
    )

    @model_validator(mode="after")
    def _fields_must_be_unique(self) -> ExtractRequest:
        """重复字段名直接 422。

        重复项会让 `confidence` 的分母失真（同一个字段被数两次），
        也会让 `data` 的键顺序含义变得含糊，所以宁可报错也不去重。
        """
        seen: set[str] = set()
        duplicated: list[str] = []
        for name in self.fields:
            if name in seen:
                duplicated.append(name)
            seen.add(name)
        if duplicated:
            unique_dup = list(dict.fromkeys(duplicated))
            raise ValueError(f"fields 存在重复字段名：{'、'.join(unique_dup)}")
        return self


class ExtractResult(BaseModel):
    """成功响应体（api-spec.md §2.1）。

    **4 个必选字段**，顺序固定：`request_id` / `data` / `confidence` / `elapsed_ms`。
    外加 1 个**可选**字段 `degraded`（v1.1，见 api-spec §2.5）。

    实现标识（`rule-v1` 或模型名）**只走响应头 `X-Extract-Model`**，
    不进 body —— 这样换模型时响应体的字节形状完全不变。
    """

    model_config = STRICT_CONFIG

    # 注意：**不要**指望 `ConfigDict(ser_json_exclude_none=True)`。
    # 实测 pydantic 2.14 里它只影响 `model_dump_json()`，而 FastAPI 是走
    # `model_dump()` 再交给 JSONResponse 序列化的 —— 结果 `None` 会原样输出成 `null`。
    # 正确做法是在路由上声明 `response_model_exclude_none=True`（见 api/v1/extract.py），
    # 那是 FastAPI 自己的机制，已验证：正常结果不含键、降级结果含 `degraded: true`。

    request_id: str = Field(description="本次请求 id，与响应头 X-Request-ID 一致；报障只报它")
    data: dict[str, str | None] = Field(
        description=(
            "字段名 -> 抽到的值；键与请求 fields 完全一致且顺序一致，抽不到为 null。"
            "值恒为字符串（金额保留原始格式），不做隐式类型转换。"
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "整体置信度 0.0–1.0，越高越可信。规则版：命中比例，有字段缺失时再打 0.75 折；"
            "LLM 版沿用同一公式以保证跨实现可比。"
        ),
    )
    elapsed_ms: int = Field(
        ge=0,
        description="服务端处理耗时（毫秒，含内部延迟），不含网络往返",
    )
    degraded: bool | None = Field(
        default=None,
        description=(
            "可选。true 表示本次结果由兜底实现（规则版）产出，"
            "原因是模型/网络失败或结构化输出两轮校验未通过。"
            "**正常结果为 None 且不出现在 JSON 里**，因此老客户端不受影响。"
        ),
    )


class ErrorDetail(BaseModel):
    """字段级错误明细（api-spec.md §3）。"""

    model_config = STRICT_CONFIG

    field: str = Field(
        description="请求体内的字段路径，如 text、fields.2；跨字段/结构性问题为 _model"
    )
    reason: str = Field(description="给人看的原因，中文，不含堆栈信息")


class ErrorResponse(BaseModel):
    """统一错误体：**所有非 2xx 都是这 4 个字段**（api-spec.md §3）。

    调用方只要写一次 `if (!res.ok) { const {code, request_id} = await res.json() }`
    就能覆盖 400/422/500，这就是"统一"的实际收益。
    """

    model_config = STRICT_CONFIG

    code: str = Field(
        description=(
            "机器可判的稳定错误码：INVALID_JSON(400) / INVALID_INPUT(422) / "
            "INTERNAL_ERROR(500) / RESPONSE_CONTRACT_VIOLATION(500)。只追加不改旧值。"
        )
    )
    message: str = Field(description="给人看的一句话，不含堆栈、不含内部路径")
    request_id: str = Field(description="本次请求 id，与响应头 X-Request-ID 一致")
    details: list[ErrorDetail] = Field(
        default_factory=list,
        description="字段级明细；422 必有，400/500 为空数组",
    )


__all__ = [
    "MAX_FIELD_COUNT",
    "MAX_FIELD_NAME_LENGTH",
    "MAX_TEXT_LENGTH",
    "ErrorDetail",
    "ErrorResponse",
    "ExtractRequest",
    "ExtractResult",
    "STRICT_CONFIG",
]
