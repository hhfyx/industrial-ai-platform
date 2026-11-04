"""**链路参数指纹**：让"改了会影响结论的常量"必须被显式确认。

## 这个文件解决什么问题（2026-11-04 核实）

`RagConfig`（`app/rag/settings.py`）只覆盖 5 个字段：

    top_k / score_threshold / chunk_chars / overlap_chars / max_context_chars

而**真正决定消融结论**的还有一堆模块常量，改一行不会让任何测试失败：

| 常量 | 位置 | 影响 |
| --- | --- | --- |
| `DEFAULT_RRF_K = 60` | `bm25/fusion.py` | 融合排序，直接改 Recall |
| `DEFAULT_CANDIDATES = 20` | `bm25/fusion.py` | 候选数，改小会截断召回 |
| `DEFAULT_B = 0.75` | `bm25/index.py` | BM25 长度归一 |
| `DEFAULT_MAX_LENGTH = 512` | `bm25/rerank.py` | 精排截断，实测影响巨大 |
| `DEFAULT_BATCH_SIZE = 16` | `bm25/rerank.py` | 只影响速度，不影响质量 |
| `DEFAULT_MIN_RATIO = 0.30` | `generation/compress.py` | 压缩比例 |

## 为什么"改一行不影响测试"是严重问题

因为**所有历史评测数字都绑在这组参数上**。如果有人把 `DEFAULT_RRF_K`
从 60 改成 40（看起来是个无害的调参），那么：

- `docs/ablation-four.md` 里"+15pp"不再成立
- README 里"混合层不划算"的结论可能要翻
- **但没有任何测试会红** —— 因为测试断言的是"函数行为"，不是"参数值"

于是文档和代码悄悄脱节，而读文档的人无法分辨。

## 这个机制的边界（它做不到什么）

**它不阻止改参数** —— 参数本来就该可调（做实验必须能改）。
它做的是：**改了要显式更新指纹，并在提交信息里说明**。

```bash
python -X utf8 scripts/param_fingerprint.py            # 看当前指纹
python -X utf8 scripts/param_fingerprint.py --update    # 改完参数后更新
```

指纹变了但历史结论没重跑 —— 那是人的责任，不是测试能解决的。
但至少**变化是可见的**，而不是静默发生。

## 为什么用"指纹"而不是"断言等于某个值"

断言 `DEFAULT_RRF_K == 60` 会让"合法的调参实验"寸步难行（必须改测试）。
指纹是**哈希**：它不说"应该是什么"，只说"和上次不同了"。

这是**变更检测**（change detection）而不是**值约束**（value constraint）。
两者适用场景不同：前者用于"这个值有历史含义，改它要小心"，
后者用于"这个值绝不能是 X"。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 指纹文件位置。**入库**（不是 gitignore）—— 它的作用就是"跨提交对比"。
FINGERPRINT_PATH: Path = PROJECT_ROOT / "eval" / "results" / "param-fingerprint.json"

#: 被跟踪的参数：`(模块路径, 常量名, 人类可读的影响说明)`。
#:
#: ## 什么该进来
#:
#: 判据：**改变它会让已发布的历史数字失效**。
#:
#: - 排序/召回相关（rrf_k、candidates、b）-> 会
#: - 精排截断（max_length）-> 会（实测影响远大于预期）
#: - 压缩比例（min_ratio）-> 会
#: - 批次大小（batch_size）-> **不会**（只影响速度），但值得记，因为
#:   延迟数字绑在它上面
#:
#: ## 什么不该进来
#:
#: - 与质量无关的实现细节（`EMPTY_TOKEN`、`INDEX_VERSION` 之类）——
#:   把它们加进来会让指纹频繁变动，反而失去"值得警觉的变化"这个意义
TRACKED_PARAMS: tuple[tuple[str, str, str], ...] = (
    (
        "app.rag.bm25.fusion",
        "DEFAULT_RRF_K",
        "RRF 融合常数；改它会直接改变混合层的 Recall（消融 ①→② 那一格）",
    ),
    (
        "app.rag.bm25.fusion",
        "DEFAULT_CANDIDATES",
        "融合候选数；改小会截断召回，让精排无料可排",
    ),
    (
        "app.rag.bm25.index",
        "DEFAULT_B",
        "BM25 长度归一化强度；影响长文档的权重",
    ),
    (
        "app.rag.bm25.rerank",
        "DEFAULT_MAX_LENGTH",
        "精排截断长度；实测截断会把相关段落从'高分'直接打到'归零'",
    ),
    (
        "app.rag.bm25.rerank",
        "DEFAULT_BATCH_SIZE",
        "精排批次；只影响速度，但 latency 结论绑在它上面",
    ),
    (
        "app.rag.generation.compress",
        "DEFAULT_MIN_RATIO",
        "压缩保留比例；影响 ④ 配置的上下文长度",
    ),
    (
        "app.rag.settings",
        "DEFAULT_TOP_K",
        "检索条数；所有 Recall@k 数字的分母",
    ),
    (
        "app.rag.settings",
        "DEFAULT_SCORE_THRESHOLD",
        "拒答阈值；改它会同时改变召回与拒答率",
    ),
)


def collect() -> dict[str, object]:
    """读出所有被跟踪参数**当前的真实值**。

    用 `importlib` 真实导入而不是正则扫源码 ——
    正则会把注释里的同名文本也抓进来，那种"指纹"是假的。
    """
    import importlib

    values: dict[str, object] = {}
    for module_name, attr, _ in TRACKED_PARAMS:
        module = importlib.import_module(module_name)
        if not hasattr(module, attr):
            values[f"{module_name}.{attr}"] = "<MISSING>"
            continue
        value = getattr(module, attr)
        # JSON 只认基础类型；Path 之类转成字符串
        values[f"{module_name}.{attr}"] = (
            value if isinstance(value, int | float | str | bool) else str(value)
        )
    return values


def compute() -> dict[str, object]:
    """当前指纹（含每个值的明文 + 一个整体哈希）。

    **保留明文**很重要：光有哈希，diff 只能说"变了"，
    说不出"从 60 变成 40"。排查时那个信息决定一切。
    """
    values = collect()
    digest = hashlib.sha256(
        json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"digest": digest, "values": values}


def load_baseline() -> dict[str, object] | None:
    """读入库的指纹文件；不存在返回 None（首次建立）。"""
    if not FINGERPRINT_PATH.exists():
        return None
    try:
        return json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def diff(baseline: dict[str, object] | None, current: dict[str, object]) -> list[str]:
    """比较基线指纹与当前值，返回**人类可读的差异**。

    返回空列表 = 没有变化。返回非空 = 有参数被改过。
    """
    if baseline is None:
        return []
    before = baseline.get("values") or {}
    after = current.get("values") or {}
    if not isinstance(before, dict) or not isinstance(after, dict):
        return []

    changes: list[str] = []
    for key in sorted(set(before) | set(after)):
        old = before.get(key, "<新增>")
        new = after.get(key, "<已移除>")
        if old != new:
            changes.append(f"{key}: {old} -> {new}")
    return changes


def write_baseline(current: dict[str, object] | None = None) -> None:
    """把当前指纹写进入库文件（改完参数、确认历史结论已重跑后用）。"""
    payload = current if current is not None else compute()
    FINGERPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


__all__ = [
    "FINGERPRINT_PATH",
    "TRACKED_PARAMS",
    "collect",
    "compute",
    "diff",
    "load_baseline",
    "write_baseline",
]
