"""**部署依赖与运行时的边界测试**：验证 Streamlit 线上环境不会缺包。

## 这个文件存在的理由

2026-11-04 的线上部署失败，报错是：

    File "/mount/src/industrial-ai-platform/app/ui/streamlit_app.py", line 47
        from app.core.llm.client import get_llm_client
    File "/mount/src/industrial-ai-platform/app/core/llm/client.py", line 39, in <module>
        from fastapi import Depends
    ModuleNotFoundError: No module named 'fastapi'

根因：`app/ui/requirements.txt` **刻意不含 fastapi**（UI 用不到 Web 框架），
而 `client.py` 在**模块顶层**导入了它。module-level import 会让整条
import 链失败 —— 即使没有任何代码用到 `Depends`。

## 为什么这个 bug 在本地永远测不出来

本地 `.venv` 装了完整开发依赖（含 fastapi），所以
`pytest tests/` 全绿、`streamlit run` 也正常。

**本地的"全绿"证明不了线上能跑** —— 两边的依赖集合不同。
这类 bug 只有两种发现方式：

1. 真的部署一次（本次就是这样，代价是几轮往返）
2. **在测试里模拟"缺包"**（本文件的做法）

第 2 种能提前发现，成本几乎为零。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 部署依赖文件（Cloud 只认入口目录下这个文件名，见 test_deploy_config.py）
DEPLOY_REQUIREMENTS = PROJECT_ROOT / "app" / "ui" / "requirements.txt"

#: 在"无 fastapi"环境里要成功导入的模块。
#:
#: 这些是 Streamlit 运行路径上真正会 import 的东西 ——
#: 少一个就让线上起不来。
STREAMLIT_IMPORT_PATH = [
    "app.core.llm.client",
    "app.core.llm",
    "app.rag.deploy_config",
    "app.rag.embedding",
]


def test_deploy_requirements_has_no_web_framework() -> None:
    """部署依赖里**不该有** fastapi / uvicorn（UI 用不到）。

    ## 这条与"缺包"的关系

    如果哪天有人为了解决上面那个 `ModuleNotFoundError`，
    **选择把 fastapi 加进部署依赖**，线上确实能起来 ——
    但那是在"为了让 UI 装上 Web 框架"而妥协，会多装
    starlette + uvicorn 等一串东西（免费层内存和安装时间都紧张）。

    正解是把 fastapi 隔离到 `app/core/llm/deps.py`，让 `client.py` 保持纯净。
    这条测试守的就是这个边界：**依赖清单是"能力声明"**，
    声明的应该是 UI 真正需要的东西。
    """
    assert DEPLOY_REQUIREMENTS.exists()
    content = DEPLOY_REQUIREMENTS.read_text(encoding="utf-8").lower()

    # 去掉注释再判断（注释里会提到 fastapi 作为说明）
    code_lines = [line.split("#", 1)[0].strip() for line in content.splitlines()]
    actual = " ".join(line for line in code_lines if line)

    for forbidden in ("fastapi", "uvicorn", "gunicorn"):
        assert forbidden not in actual, (
            f"部署依赖里出现了 {forbidden} —— UI 用不到它。"
            "streamlit_app.py 的 import 链应当不依赖 fastapi；"
            "需要 Depends 的代码请放在 app/core/llm/deps.py"
        )


def test_streamlit_import_path_works_without_fastapi() -> None:
    """**核心回归测试**：模拟线上（无 fastapi）导入整条 UI 依赖链。

    ## 为什么必须在子进程里跑

    直接在当前 pytest 进程里 `sys.meta_path` 插一个 blocker 是可行的，
    但副作用会泄漏到同进程的其他测试（例如真正需要 fastapi 的
    `tests/test_llm_client.py`）。**子进程隔离最干净**，也不怕 `sys.modules`
    已经被导入过。

    ## 判据

    退出码 0 且输出 `OK` —— 说明每个模块都在"没有 fastapi"的环境里导入成功。
    """
    snippet = f"""
import sys

class _BlockFastapi:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "fastapi":
            raise ImportError("simulated: no fastapi on Streamlit Cloud")
        return None

sys.meta_path.insert(0, _BlockFastapi())
sys.path.insert(0, {str(PROJECT_ROOT)!r})

# 模拟 Cloud：免费层 -> 走 hashing 降级（不加载 torch）
import os
os.environ["IS_STREAMLIT_CLOUD"] = "1"

for name in {STREAMLIT_IMPORT_PATH!r}:
    __import__(name)

# 顺带确认 UI 真要用的那个函数拿到了
from app.core.config import get_settings
from app.core.llm.client import get_llm_client
client = get_llm_client(get_settings())
assert type(client).__name__ == "AsyncOpenAI", type(client)

print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", snippet],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    assert result.returncode == 0, (
        "在**没有 fastapi** 的环境里导入 UI 依赖链失败了 —— "
        "这正是线上 ModuleNotFoundError 的复现。\n"
        f"stdout:\n{result.stdout[-1500:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    assert "OK" in result.stdout, result.stdout[-500:]


def test_streamlit_path_has_no_module_level_fastapi() -> None:
    """**精度更高的结构性约束**：只检查 Streamlit 导入链上的模块。

    ## 为什么不能用"全仓库禁 fastapi"

    第一版写成 `app/` 下只允许 `deps.py` 出现 fastapi —— **判据错了**。
    `app/api/v1/*.py`、`app/main.py`、`app/core/errors.py` 等是
    **FastAPI 服务端代码**，在那里顶层 import fastapi 完全正确。

    真正的要求不是"全仓库不许用 fastapi"，而是：

        **Streamlit 的 import 链不能碰到 fastapi。**

    所以这里沿着入口模块的实际依赖往下列 —— 只查这条链。

    ## 这条与上面那条（子进程导入）的分工

    | 测试 | 能发现什么 |
    | --- | --- |
    | 子进程导入（`test_streamlit_import_path_works_without_fastapi`） | 真实运行时崩溃 |
    | 本测试（静态扫描导入链） | 指出**具体是哪个文件哪一行** |

    两者互补：前者是"会不会坏"，后者是"坏在哪"。
    """
    #: Streamlit 入口及其会 import 的模块（相对仓库根，POSIX 风格）。
    #:
    #: 只列**我们自己的**代码 —— 第三方库（streamlit/chromadb 等）不归我们管。
    streamlit_chain = [
        "app/ui/streamlit_app.py",
        "app/core/llm/__init__.py",
        "app/core/llm/client.py",
        "app/core/config.py",
        "app/rag/deploy_config.py",
        "app/rag/embedding/__init__.py",
        "app/rag/embedding/bge.py",
        "app/rag/embedding/hashing.py",
    ]

    violations: list[str] = []
    for relative in streamlit_chain:
        path = PROJECT_ROOT / relative
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped.startswith(("from fastapi", "import fastapi")):
                continue
            # 只报**模块级**（无缩进）的 import；函数内延迟导入是允许的
            if line == line.lstrip():
                violations.append(f"{relative}:{lineno}")

    assert not violations, (
        "Streamlit 导入链上有模块**顶层** import 了 fastapi —— "
        "部署依赖不含 fastapi，线上会崩：\n  "
        + "\n  ".join(violations)
        + "\n\n需要 Depends 请放到 app/core/llm/deps.py（只有 FastAPI 侧会导入）。"
    )


def test_api_layer_is_allowed_to_use_fastapi() -> None:
    """反向确认：**FastAPI 服务端代码允许（且应该）直接用 fastapi**。

    这条防的是"过度纠正"：有人看到上面那条，以为全仓库都不能用 fastapi，
    于是把 `app/api/v1/*.py` 里的 import 也改成延迟导入 —— 那是无谓的复杂化。

    **判据是"哪条运行时链"，不是"哪个包"**。
    """
    api_files = list((PROJECT_ROOT / "app" / "api").rglob("*.py"))
    assert api_files, "app/api 不存在？结构变了"

    has_fastapi = any("fastapi" in f.read_text(encoding="utf-8") for f in api_files)
    assert has_fastapi, "app/api 下居然没有任何 fastapi 引用 —— 是不是有人把路由代码搬走了？"
