"""线上/免费层专用的向量化选择：**把内存与启动开销压到最低**。

## 为什么需要它（一个实测出来的硬约束）

本机实测（`scripts/measure_memory.py`，Windows `tasklist` 取工作集）：

| 阶段 | 进程内存 |
| --- | ---: |
| 只 import torch | ~250 MB |
| + 加载 bge-small-zh 模型 | **561 MB** |
| + Streamlit 自身 | 再加 ~100 MB |
| + Chroma 客户端 | 再加 ~50 MB |

Streamlit Community Cloud 的免费层是 **1 GB**。561 MB 的模型加框架开销，
再叠上并发会话，**峰值很容易撞上 1 GB 被 OOM 杀掉** —— 而 OOM 的表现是
"应用重启、对话丢失、用户看到 Redraw 错误"，不是一条清晰的错误信息。

## 两种应对，代价完全不同

| 方案 | 内存 | 质量 | 代价 |
| --- | --- | --- | --- |
| `bge` CPU 全量 | 561 MB | 好（有语义） | 撞内存上限，随时被杀 |
| **`hashing` 降级** | ~1 MB | 只有字面相似 | **语义检索能力归零** |
| ONNX 量化版 | 未测 | 接近 bge | 额外依赖 + 模型转换 |

**免费层默认 `hashing`**，理由是**诚实**：与其上线一个"看起来能语义检索、
实际经常被 OOM 杀掉"的服务，不如明确降级到"能跑通 + 引用链路完整 +
检索只有字面能力"。这与 `docs/design-decisions.md` 第三条（为什么先 ChromaDB）
是同一条思路：**为一个还没被验证的收益付出确定的复杂度，不划算**。

## 降级必须可见

侧栏显示当前 embedder 及其 `recommended_min_score`
（hashing 是 `0.0`，等于关掉拒答闸门）。**悄悄降级等于骗用户。**
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# 环境变量：部署平台用它强制指定实现
EMBEDDER_ENV = "RAG_EMBEDDER"

# 免费层默认实现。见模块 docstring 的内存实测。
# 想用 bge 时显式设 `RAG_EMBEDDER=bge-small-zh` —— 那是有意识的决定
# （本地跑或付费层），不是默认踩坑。
FREE_TIER_DEFAULT = "hashing"


def resolve_embedder_name(explicit: str | None = None) -> str:
    """决定用哪个 embedder。优先级：显式参数 > 环境变量 > 环境推断。

    ## 为什么要在托管环境上做特殊判断

    同一份代码：本机跑 bge（561 MB 无所谓）、免费层跑 bge（撞 1 GB 被杀）。
    **代码不该指望部署者记得改配置** —— 那不是配置失误，
    是"这个环境装不下这个模型"的硬事实。

    所以检测到已知的低内存托管环境时，默认降级并打 warning。

    ⚠️ **已知局限**：下面的判断依赖平台注入的环境变量名。
    平台改名时会**静默失效**（退回 bge → 又 OOM）。
    所以 `tests/test_deploy_config.py` 断言了这几个变量名 ——
    改动它们必须同时改测试。
    """
    if explicit:
        return explicit

    from_env = os.environ.get(EMBEDDER_ENV)
    if from_env:
        return from_env

    if looks_like_free_hosting():
        logger.warning(
            "检测到低内存托管环境，embedder 默认降级为 %s（bge 约需 561MB，"
            "免费层 1GB 会被 OOM 杀掉）。要强制用 bge 请设 %s=bge-small-zh",
            FREE_TIER_DEFAULT,
            EMBEDDER_ENV,
        )
        return FREE_TIER_DEFAULT

    return FREE_TIER_DEFAULT


def looks_like_free_hosting() -> bool:
    """是否是已知的低内存托管环境。

    检测的是**平台注入的变量**（不是猜）：

    | 变量 | 来源 |
    | --- | --- |
    | `IS_STREAMLIT_CLOUD` | Streamlit Community Cloud |
    | `STREAMLIT_SHARING_MODE` | 同上（老版本用这个） |
    | `K_SERVICE` | Google Cloud Run |
    | `DYNO` | Heroku |

    **不检测 `STREAMLIT_SERVER_*`**：那些本地跑也有，会让本地也降级 ——
    而本地有 16 GB 内存，降级纯属自损。
    """
    markers = (
        "IS_STREAMLIT_CLOUD",
        "STREAMLIT_SHARING_MODE",
        "K_SERVICE",
        "DYNO",
    )
    return any(os.environ.get(marker) for marker in markers)


def describe_runtime() -> dict[str, object]:
    """给 UI 显示"我现在跑在什么环境"。降级必须可见。"""
    detected = looks_like_free_hosting()
    return {
        "embedder_env": os.environ.get(EMBEDDER_ENV),
        "free_hosting_detected": detected,
        "default_embedder": resolve_embedder_name(),
        # 给 UI 直接显示的一句话。**不让 UI 自己拼这段判断**，
        # 否则判断逻辑会出现两份，改一处忘一处。
        "notice": (
            "⚠️ 当前运行在低内存托管环境，向量化已降级为 **hashing**"
            "（只有字面相似，没有语义检索）。用 bge 需 ~561MB，超过免费层 1GB 上限。"
            if detected
            else None
        ),
    }


__all__ = [
    "EMBEDDER_ENV",
    "FREE_TIER_DEFAULT",
    "describe_runtime",
    "looks_like_free_hosting",
    "resolve_embedder_name",
]
