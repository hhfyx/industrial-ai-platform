"""实测进程内存：为什么免费层跑不了 bge。

## 为什么这个脚本必须存在

"bge 占 561 MB"这个数字如果只写在文档里，它就是**某个人记得的一个数**。
下次有人换模型、升级 torch、或改成 ONNX 版，那个数字就过期了 ——
而文档不会自己更新。

所以这里是可复跑的：跑一遍，拿到当前环境的真实数字。

## 为什么用子进程

内存要在**干净进程**里量。在已经 import 了一堆东西的主进程里测，
测出来的是"当前进程加了模型之后多重"，而不是"只跑模型要多重"。

## 为什么用 `tasklist` 取（Windows）

Python 标准库没有跨平台拿工作集（RSS/pSS）的接口，而**不装 psutil**
是本项目的依赖洁癖（多一个包就多一份供应链风险）。
Windows 下 `tasklist /FO CSV /FI "PID eq N"` 给的是工作集，够用。

Linux/macOS 用 `/proc/<pid>/status` 的 VmRSS，或 `ps -o rss=`。

运行：

    uv run python -X utf8 scripts/measure_memory.py
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 子进程里跑的探针脚本。**每一步都先打印阶段名再干活**，
# 这样主进程能按输出顺序对应到阶段 —— 不用猜。
#
# **这里用占位符替换而不是 `str.format`**：探针本身含 f-string（有 `{}`），
# 用 `.format()` 会把它当成格式占位符并抛 `KeyError: 'stage'`（实测）。
# 用 `__HF_HOME__` 这类不会与代码冲突的标记最省事。
PROBE = r"""
import os, sys

def report(stage):
    print(f"STAGE {stage}", flush=True)

print("PID", os.getpid(), flush=True)
report("baseline")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault(
    "HF_HOME", r"__HF_HOME__"
)

report("torch")
import torch

report("sentence_transformers")
from sentence_transformers import SentenceTransformer

report("model_load")
model = SentenceTransformer("__MODEL_NAME__", device="cpu")

report("encode")
model.encode(["违约金按日万分之五支付。"], normalize_embeddings=True, batch_size=1)

report("settled")
import time
time.sleep(__HOLD_SECONDS__)
"""


def _rss_mb(pid: int) -> float:
    """取某个进程的工作集（MB）。跨平台。"""
    system = platform.system()
    if system == "Windows":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
            capture_output=True,
            text=True,
        ).stdout
        # CSV 形如: "python.exe","1234","Console","0","561,452 K"
        #
        # **必须用 csv 模块解析，不能 `line.split(",")`。**
        # 实测踩到：内存字段是 `"18,008 K"` —— **引号里含逗号**。
        # 裸 split 会把它拆成两个字段，`parts[-1]` 只剩 `"008 K"`，
        # 于是所有阶段都量成 ~1 MB，结论变成"bge 只占 1 MB，免费层完全够"。
        # 这个错误最阴的地方是**数字看起来���小很合理**（小就是好事），
        # 会直接把部署决策做反。
        for row in csv.reader(out.splitlines()[1:]):
            if len(row) < 5:
                continue
            raw = row[-1].replace("K", "").replace(",", "").strip()
            try:
                return float(raw) / 1024
            except ValueError:
                continue
        return -1.0

    if system == "Linux":
        status = Path(f"/proc/{pid}/status")
        if status.exists():
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024  # kB -> MB
        return -1.0

    # macOS 等：用 ps
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
    ).stdout
    try:
        return float(out.strip()) / 1024  # kB -> MB
    except ValueError:
        return -1.0


def main() -> int:
    model_name = "BAAI/bge-small-zh-v1.5"
    hf_home = str(PROJECT_ROOT / ".cache" / "hf")
    hold_seconds = 4

    script = (
        PROBE.replace("__HF_HOME__", hf_home)
        .replace("__MODEL_NAME__", model_name)
        .replace("__HOLD_SECONDS__", str(hold_seconds))
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(script)
        probe_path = handle.name

    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", probe_path],
        stdout=subprocess.PIPE,
        text=True,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )

    stages: dict[str, float] = {}
    # **必须用子进程自己报告的 PID，不能用 `proc.pid`。**
    #
    # 实测踩到：在 Windows 上 `.venv\Scripts\python.exe` 是一个 **launcher**，
    # 它会再启动一个真正的 python 子进程。
    #   proc.pid        → 16072（launcher，5.6 MB，一直都在）
    #   子进程自报 PID  → 28148（真正加载了 torch 的那个）
    #
    # 用 `proc.pid` 去 `tasklist`，量到的是 launcher 的空壳 ——
    # 于是所有阶段都是 0.9 MB，结论会变成"bge 只占 1 MB，免费层完全够"。
    # **这个错误会让整个部署决策反向**，而且数字看起来"非常合理"（小就是好事）。
    pid: int | None = None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("PID "):
                pid = int(line.split()[1])
                continue
            if not line.startswith("STAGE "):
                continue
            stage = line.split(maxsplit=1)[1]
            if pid is None:
                continue
            # 读到阶段名时该阶段的活已经干完了（报告在 import/加载之后）
            stages[stage] = round(_rss_mb(pid), 1)
    finally:
        proc.wait()
        Path(probe_path).unlink(missing_ok=True)

    if pid is None:
        print("探针没有报告 PID，无法测量（脚本内部异常？）", file=sys.stderr)
        return 1

    baseline = stages.get("baseline", 0.0)
    payload = {
        "meta": {
            "platform": platform.system(),
            "python": platform.python_version(),
            "model": model_name,
            "note": "工作集（RSS），在干净子进程里量；数值随 torch 版本变化，须复跑",
        },
        "stages_mb": stages,
        # "相对基线的增量"才是"模型到底要多少"的答案。
        # 绝对数里混了 Python 解释器本身（约 10 MB），增量更说明问题。
        "delta_mb": {stage: round(value - baseline, 1) for stage, value in stages.items()},
    }

    print(f"{'阶段':<24} {'工作集':>10} {'增量':>10}")
    for stage in ("baseline", "torch", "sentence_transformers", "model_load", "encode", "settled"):
        if stage not in stages:
            continue
        delta = stages[stage] - baseline
        print(f"{stage:<24} {stages[stage]:>9.1f}M {delta:>9.1f}M")

    peak = max(stages.values()) if stages else 0.0
    print(f"\n峰值 {peak:.0f} MB。Streamlit 免费层上限 1000 MB。")

    # **判据用"模型自身增量"，不是绝对工作集。**
    #
    # 绝对数里含 Python 解释器本身（~20 MB）与它已 import 的库，
    # 但真正决定"能不能塞进 1 GB"的是**加载模型后新增的那部分** ——
    # 因为在 Streamlit 里，解释器和框架本来就要占一份。
    #
    # 实测：torch+ST+模型+一次 encode 的增量是 **527 MB**。
    # 再叠 Streamlit（~100 MB）+ Chroma（~50 MB）+ 会话状态，
    # 已经在 700 MB 以上 —— 1 GB 上限下**没有余量**，
    # 两个并发会话或一次大文档索引就会 OOM。
    delta_peak = max(payload["delta_mb"].values()) if payload["delta_mb"] else 0.0
    print(f"模型栈增量 {delta_peak:.0f} MB（不含解释器基线）。")

    if delta_peak > 400:
        verdict = (
            "→ **bge 在免费层不安全**。模型栈本身就要 "
            f"{delta_peak:.0f} MB，再叠 Streamlit(~100MB) + Chroma(~50MB) + 会话状态，\n"
            "  1 GB 上限下没有余量 —— 并发会话或大文档索引就会 OOM。\n"
            "  → 免费层默认 hashing（见 app/rag/deploy_config.py，由 tests/test_deploy_config.py 守着）"
        )
    else:
        verdict = "→ 本次实测在安全范围内，但 torch 版本变化会显著改变这个数。"
    print(verdict)

    out = PROJECT_ROOT / "scripts" / "memory_result.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写入 {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
