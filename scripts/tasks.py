"""任务入口：**一条命令跑一件事，不依赖 make / bash**。

## 为什么要这个文件（而不是只有 Makefile）

`Makefile` 已经写了同样的目标，但实测本机**没有 `make`**：

```
PS> make help
make : 无法将"make"项识别为 cmdlet、函数、脚本文件或可运行程序的名称
```

而周五那条验收写的是「`make eval-retrieval` 一条命令出指标」。
让验收依赖一个没装的工具，等于交付一个跑不起来的东西。

所以这里做两件事：

1. **提供跨平台入口**：`python -X utf8 scripts/tasks.py eval-retrieval`
2. **让 Makefile 直接转发到这里** —— 装了 make 的人仍然可以 `make eval-retrieval`，
   但它调的是同一份代码，**不存在两套会漂移的命令定义**

## 为什么不写成 shell 脚本

Windows 上 `pwsh` / `cmd` / `bash`（Git Bash）三种都存在但不保证有哪个。
`python` 是**唯一确定存在**的那个 —— 这个仓库本来就要用 Python 跑。

## 用法

    python -X utf8 scripts/tasks.py            # 列出所有任务
    python -X utf8 scripts/tasks.py eval-retrieval
    python -X utf8 scripts/tasks.py test
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 解释器：**用当前这个**，而不是硬编 `.venv/Scripts/python.exe`。
# 理由：调用 `tasks.py` 的人已经选好了解释器（可能来自 uv run、也可能来自
# 系统 Python 但已装好依赖）。再硬编一次路径，会在 CI / 别人机器上静默失效。
PY = sys.executable

# 任务表：名字 → (说明, 参数列表)
# **参数列表而不是 shell 字符串**：字符串要走 shell 转义，
# 而 Windows 与 POSIX 的引号规则不同 —— 那正是"我这儿能跑"类问题的来源。
TASKS: dict[str, tuple[str, list[str]]] = {
    # ---- 必备 ----
    "eval-retrieval": (
        "50 条检索评测：Recall@k / MRR / 拒答准确率（不调 LLM）",
        ["scripts/run_retrieval_eval.py"],
    ),
    "eval-retrieval-hashing": (
        "同上但用 hashing（只验证流程，召回无参考价值）",
        ["scripts/run_retrieval_eval.py", "--embedder", "hashing"],
    ),
    "test": ("全套测试（断网也能跑）", ["-m", "pytest", "tests/", "-q"]),
    "lint": ("ruff 检查", ["-m", "ruff", "check", "app", "scripts", "tests"]),
    "fmt": ("ruff 自动修复 + 格式化", None),  # 特殊：两步
    # ---- 调参实验（用来选参数，不是交付物） ----
    "topk": ("TopK 扫描：k 3/5/8/10 的召回、噪音、上下文成本", ["scripts/topk_scan.py"]),
    "token-audit": ("bge tokenizer 审计：越界占比 + 直方图", ["scripts/token_audit.py"]),
    "grid": ("chunk × overlap 网格（先粗后细）+ 缓存验证", ["scripts/grid_search.py"]),
    "filter-eval": (
        "三场景过滤前后命中率 + 写错条件的静默行为",
        ["scripts/filter_eval.py"],
    ),
    "threshold": ("阈值分布：命中 vs 未命中的分数", ["scripts/threshold_scan.py"]),
    "refusal-demo": ("拒答可演示（含已撤销方案的对照）", ["scripts/refusal_demo.py"]),
    # ---- W5 周六主交付 ----
    "strategy-compare": (
        "三策略 × 各自最优参数的全量对比（主表）",
        ["scripts/strategy_compare.py"],
    ),
    "strategy-significance": (
        "三策略差异的显著性检验（Wilson + 配对 McNemar）",
        ["scripts/strategy_significance.py"],
    ),
    "threshold-fine": ("阈值精标（50 条集，19 档扫描）", ["scripts/threshold_fine.py"]),
    "diagnose-semantic": (
        "诊断语义切分：是实现不行，还是语料无从发挥",
        ["scripts/diagnose_semantic.py"],
    ),
    "answer-quality": (
        "答案质量抽检（20 条走完整链路，**会花 LLM 钱**）",
        ["scripts/answer_quality.py"],
    ),
}


def _run(args: list[str]) -> int:
    """跑一个子命令，**把退出码原样传出去**。

    传退出码不是形式主义：`tasks.py eval-retrieval` 失败时如果返回 0，
    CI / 脚本会当成成功 —— 那比直接报错更糟。
    """
    print(f"$ {PY} {' '.join(args)}", flush=True)
    return subprocess.call([PY, *args], cwd=PROJECT_ROOT)


def list_tasks() -> None:
    print("可用任务（python -X utf8 scripts/tasks.py <name>）：\n")
    width = max(len(name) for name in TASKS)
    for name, (description, _args) in TASKS.items():
        print(f"  {name:<{width}}  {description}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv

    if not args or args[0] in ("-h", "--help", "help", "list"):
        list_tasks()
        return 0

    name = args[0]
    if name not in TASKS:
        print(f"未知任务：{name}\n", file=sys.stderr)
        list_tasks()
        return 2

    _description, command = TASKS[name]
    if command is None:  # fmt：两步
        code = _run(["-m", "ruff", "check", "app", "scripts", "tests", "--fix"])
        if code != 0:
            # `--fix` 修不完时仍有退出码 1；格式化仍应继续跑完
            print("（ruff check --fix 有剩余问题，继续格式化）", file=sys.stderr)
        return _run(["-m", "ruff", "format", "app", "scripts", "tests"])

    # 额外参数透传给脚本 —— 这样 `tasks.py eval-retrieval --threshold 0.5` 也能用
    return _run([*command, *args[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
