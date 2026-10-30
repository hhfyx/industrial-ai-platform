"""CI 装配冒烟：验证默认实现能装配成 LlmExtractor，且网关不可达时降级。

**为什么是独立脚本，而不是 CI 里的 `python -c "..."` 内联**

原 CI 写的是：

    uv run python -c "
    from app.core.deps import get_extractor
    ...
    "

它有**两个**问题，都实测复现过（2026-10-30）：

1. **`import app` 会 ModuleNotFoundError。**
   `uv run python -c` 执行时**当前目录不会自动进 `sys.path`**
   （而 `pytest` 会，因为 pytest 把 rootdir 加进去，且 `tests/` 是包）。
   实测：容器里 `uv run python scripts/xxx.py` → `No module named 'app'`；
   加 `PYTHONPATH=.` 后立刻通过。
   **所以这一步从加进来那天起就一直是红的，只是没人去看 CI。**

2. **内联多行字符串在 shell 里很脆。**
   YAML 的 `run: |` 里嵌 `python -c "..."`，缩进和引号都靠人肉对齐。

改成文件后：本地和 CI 跑**同一份代码**，也能单独 `python scripts/ci_wiring_check.py` 复现。

判定口径
--------
① 装配：`get_extractor()` 返回 `LlmExtractor`，且 model 来自 env
② 降级：网关不可达时**返回 degraded=True 的结果，而不是抛 500**

运行（CI 环境变量已由 workflow 注入）：
    python scripts/ci_wiring_check.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# 关键：把仓库根加进 sys.path。
# 这不是"图省事" —— CI 里 `uv run python scripts/xxx.py` 的 sys.path[0] 是
# `scripts/`，不是仓库根；不加这行就 `import app` 失败。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def step1_wiring() -> None:
    """① 默认实现必须装配成 LlmExtractor。"""
    from app.core.deps import get_extractor
    from app.services.llm_extractor import LlmExtractor

    ext = get_extractor()
    print(f"step1: got {type(ext).__name__} | model={getattr(ext, 'model_name', '?')}")

    assert isinstance(ext, LlmExtractor), f"默认实现应为 LlmExtractor，实际 {type(ext).__name__}"
    expected_model = os.environ.get("LLM_MODEL")
    assert ext.model_name == expected_model, f"model 应为 {expected_model}，实际 {ext.model_name}"
    print("OK: 装配正确 ->", type(ext).__name__, "|", ext.model_name)


def step2_degrade() -> None:
    """② 网关不可达时必须降级，不能抛异常。"""
    from app.core.deps import get_extractor

    ext = get_extractor()
    out = asyncio.run(ext.extract(text="合同编号 HT-2026-0042。", fields=["合同编号"]))
    degraded = getattr(out, "degraded", None)
    print(f"step2: degraded={degraded} | data={out.data}")

    assert degraded is True, (
        f"网关不可达时应标记 degraded=True，实际 {degraded}（说明降级路径没走通）"
    )
    assert out.data.get("合同编号") == "HT-2026-0042", "规则版应能抽出合同编号"
    print("OK: 降级路径通，规则版结果正确")


def main() -> None:
    print("EXTRACTOR_IMPL =", os.environ.get("EXTRACTOR_IMPL"))
    print("LLM_BASE_URL   =", os.environ.get("LLM_BASE_URL"))
    print("LLM_MODEL      =", os.environ.get("LLM_MODEL"))
    step1_wiring()
    step2_degrade()
    print("ALL OK")


if __name__ == "__main__":
    main()
