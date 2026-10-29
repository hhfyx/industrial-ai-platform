"""实验缓存：让重复跑同一个 prompt 不重复花钱。

**为什么单独做一个文件，而不是塞进 `structured.py`**：
`structured_call` 是线上路径（W4/W5/W7/W9 都调它），缓存是**实验期**的手段。
把缓存逻辑混进主路径会让每个线上调用都过一遍哈希 + 磁盘 IO，且引入
"缓存命中时行为与真实调用不同"的风险。分开之后：

    structured_call（纯函数式，无状态）
        ↑ 实验脚本包一层
    scripts/experiment_cache.py

## 缓存键怎么算

`(model, prompt, schema_json, temperature, max_tokens)` 的 sha256。
**必须把 schema 算进键**：同一个 prompt 换个模型定义就完全不是一回事，
不算进键就会拿到结构不兼容的旧结果 —— 这是缓存最隐蔽的串味方式。

## 为什么不缓存失败

失败的调用也花了钱，理论上也该缓存。但**不缓存**：
失败通常是"那次模型的随机表现"，缓存它会让一个偶发失败在后续所有实验里
都复现成一个确定性失败 —— 实验数据会变成"我第一次跑出来的运气"。
缓存只用于成功的样本。

## 存储

`scripts/.cache/structured/<key>.json`。JSON 而非 pickle：能人读、能 diff、
不会因为换 Python 版本反序列化崩掉。`.cache/` 已在 .gitignore（见下）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

CACHE_DIR = Path(__file__).with_name(".cache") / "structured"

M = TypeVar("M", bound=BaseModel)


def cache_key(
    *,
    model: str,
    prompt: str,
    schema: type[BaseModel],
    temperature: float,
    max_tokens: int,
    extra: dict[str, Any] | None = None,
) -> str:
    """算出缓存键。schema 必须参与哈希（见模块 docstring）。"""
    payload = {
        "model": model,
        "prompt": prompt,
        "schema": schema.model_json_schema(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra": extra or {},
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def load(key: str) -> dict[str, Any] | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # 缓存坏了就当没有 —— 绝不能因为缓存文件损坏让实验挂掉
        return None


def save(key: str, value: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def clear() -> int:
    """清空缓存，返回删掉的条目数。"""
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for p in CACHE_DIR.glob("*.json"):
        p.unlink()
        n += 1
    return n


def stats() -> dict[str, Any]:
    """缓存规模，供实验脚本开头打印。"""
    if not CACHE_DIR.exists():
        return {"entries": 0, "bytes": 0}
    files = list(CACHE_DIR.glob("*.json"))
    return {
        "entries": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }


async def cached_structured_call(
    prompt: str,
    schema: type[M],
    *,
    max_repair: int = 2,
    fallback: BaseModel | None = None,
    force_refresh: bool = False,
    extra: dict[str, Any] | None = None,
) -> M:
    """实验用的带缓存版本。命中缓存时**完全不发请求**。

    注意：这里 import `structured_call` 用的是**延迟 import**，
    避免 scripts/ 被 app 反向依赖（脚本可以 import app，反过来不行）。
    """
    from app.core.config import get_settings
    from app.core.llm.structured import (
        DEFAULT_MAX_TOKENS,
        DEFAULT_TEMPERATURE,
        structured_call,
    )

    settings = get_settings()
    key = cache_key(
        model=settings.llm.model,
        prompt=prompt,
        schema=schema,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        extra=extra,
    )

    if not force_refresh:
        hit = load(key)
        if hit is not None:
            return schema.model_validate(hit["value"])

    result = await structured_call(prompt, schema, max_repair=max_repair, fallback=fallback)

    # 只缓存成功（fallback 降级的结果也缓存，但标 degraded 便于事后筛掉）
    save(
        key,
        {
            "value": result.model_dump(mode="json"),
            "degraded": isinstance(result, BaseModel)
            and fallback is not None
            and result.model_dump(mode="json") == fallback.model_dump(mode="json"),
            "schema": schema.__name__,
        },
    )
    return result  # type: ignore[return-value]


__all__ = [
    "CACHE_DIR",
    "cache_key",
    "cached_structured_call",
    "clear",
    "load",
    "save",
    "stats",
]
