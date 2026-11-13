"""生成服务的基准实测（W8 周五）：五个数字，全部带口径落盘。

## 这五个数字是什么、为什么必须有

| # | 数字 | 口径要点 |
| --- | --- | --- |
| 1 | 单张 512×512 耗时（**含冷启动**） | 冷启动 = 进程内第一次推理（CUDA 图优化/显存分配都在内） |
| 2 | 单张 512×512 耗时（**预热后**） | 预热后再测，与 #1 的差就是"加载完 ≠ 就绪"的量 |
| 3 | 显存峰值 | FP16、batch=1、768 与 512 两档 |
| 4 | batch=1 vs batch=4 吞吐 | 张/秒；单 worker 排队的代价与收益靠它量化 |
| 5 | 模型加载时间 | 进程起 → pipeline 可用；healthcheck 的 start-period 靠它定 |

**同硬件同精度的对照基线**写进结果 JSON 的 `baseline` 字段——
简历上任何"X× 提升"都从这里除出来，基线变了倍数就作废。

## 本机边界（2026-12-11 记录，如实）

本机 `torch 2.15.0+cpu`（无 CUDA 轮子）、无 diffusers、无 SD 权重 ——
**今天跑不出这五个数字**。本脚本就是"真机一跑就出"的可执行清单：
GPU 环境上 `python -X utf8 scripts/benchmark_generator.py` 一次出全部五项 +
落盘 JSON/MD。在跑通之前，README 与简历一律用保守表述（见
`docs/generator-benchmark.md`），不写编造的倍数。

运行（真机）::

    python -X utf8 scripts/benchmark_generator.py --size 512 --repeats 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def _gpu_env() -> dict[str, Any]:
    """硬件与版本口径 —— 没有这两行，任何耗时数字都不可比。"""
    import torch

    assert torch.cuda.is_available(), (
        "需要 CUDA 版 torch（本机是 CPU 轮子，见模块 docstring 的边界记录）。"
        "安装：pip install torch --index-url https://download.pytorch.org/whl/cu128"
    )
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "vram_total_gb": round(props.total_memory / 1e9, 2),
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "precision": "fp16",
    }


def _diffusers_stack() -> tuple[Any, Any]:
    """加载 SD1.5 + 首个 LoRA。权重目录走环境变量，与容器同口径。"""
    import os

    import torch
    from diffusers import StableDiffusionPipeline

    weights_dir = os.environ.get("SD_WEIGHTS_DIR", "models/sd15")
    base = Path(weights_dir) / "stable-diffusion-v1-5"
    if not base.exists():
        raise FileNotFoundError(
            f"SD1.5 基座不在 {base}。放置：models/sd15/stable-diffusion-v1-5/（或改 SD_WEIGHTS_DIR）"
        )

    t0 = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        base,
        torch_dtype=torch.float16,
        safety_checker=None,
        variant="fp16",  # 权重按 fp16 变体下载（体积减半），加载时必须显式指定
    ).to("cuda")
    # 内存优化按 6GB 卡的量级开：attention slicing 几乎无损、xformers 可选
    pipe.enable_attention_slicing()
    load_s = time.perf_counter() - t0
    return pipe, load_s, weights_dir


def _peak_vram_gb() -> float:
    import torch

    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated() / 1e9, 3)


def run_benchmark(*, size: int, repeats: int, batch_sizes: tuple[int, ...]) -> dict[str, Any]:
    import torch

    env = _gpu_env()
    pipe, load_s, weights_dir = _diffusers_stack()

    generator_fn = lambda seed: torch.Generator("cuda").manual_seed(seed)  # noqa: E731

    # ---- ⑤ 加载时间（含 LoRA 挂载）
    loras = sorted((Path(weights_dir) / "lora").glob("*.safetensors"))
    lora_load_s = 0.0
    if loras:
        t0 = time.perf_counter()
        pipe.load_lora_weights(str(loras[0]))
        lora_load_s = time.perf_counter() - t0

    # ---- #1 冷启动单张
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pipe(
        "基准提示词：industrial surface scratch defect, macro photo",
        num_inference_steps=25,
        height=size,
        width=size,
        generator=generator_fn(0),
    )
    torch.cuda.synchronize()
    cold_ms = int((time.perf_counter() - t0) * 1000)
    vram_512 = _peak_vram_gb()

    # ---- #2 预热后单张（repeats 次，报中位数与 P95）
    times: list[int] = []
    for i in range(repeats):
        t0 = time.perf_counter()
        pipe(
            "基准提示词：industrial surface scratch defect, macro photo",
            num_inference_steps=25,
            height=size,
            width=size,
            generator=generator_fn(i),
        )
        torch.cuda.synchronize()
        times.append(int((time.perf_counter() - t0) * 1000))
    warm_median_ms = int(statistics.median(times))
    warm_p95_ms = sorted(times)[int(len(times) * 0.95) - 1]

    # ---- #4 batch 吞吐（同一 prompt 多 seed，steps 按张数摊）
    throughput: dict[str, Any] = {}
    for batch in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        prompts = ["基准提示词"] * batch
        t0 = time.perf_counter()
        pipe(
            prompts,
            num_inference_steps=25,
            height=size,
            width=size,
            generator=[generator_fn(i) for i in range(batch)],
        )
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        throughput[str(batch)] = {
            "wall_ms": int(wall_s * 1000),
            "images_per_sec": round(batch / wall_s, 3),
            "peak_vram_gb": _peak_vram_gb(),
        }

    return {
        "meta": {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "steps": 25,
            "scheduler": type(pipe.scheduler).__name__,
            "attention_slicing": True,
            "lora_loaded": bool(loras),
            **env,
        },
        "model_load_seconds": round(load_s, 2),
        "lora_load_seconds": round(lora_load_s, 2),
        "cold_start_ms": cold_ms,
        "warm_median_ms": warm_median_ms,
        "warm_p95_ms": warm_p95_ms,
        "warm_all_ms": times,
        "peak_vram_gb_512": vram_512,
        "throughput": throughput,
        "baseline": {
            "说明": "对照基线：同硬件、同精度(FP16)、同 steps、batch=1 的『逐张生成』",
            "per_image_warm_ms": warm_median_ms,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=512, choices=(512, 768))
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--batches", type=str, default="1,4", help="逗号分隔，如 1,4")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(exist_ok=True)
    from datetime import datetime as _dt

    stamp = _dt.now(UTC).strftime("%Y%m%d-%H%M%S")
    result = run_benchmark(
        size=args.size,
        repeats=args.repeats,
        batch_sizes=tuple(int(b) for b in args.batches.split(",")),
    )
    out_json = RESULTS_DIR / f"generator-bench-{stamp}.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"落盘：{out_json}")
    print(
        f"加载 {result['model_load_seconds']}s（LoRA +{result['lora_load_seconds']}s）· "
        f"冷启动 {result['cold_start_ms']}ms · 预热中位 {result['warm_median_ms']}ms · "
        f"VRAM {result['peak_vram_gb_512']}GB"
    )
    for batch, t in result["throughput"].items():
        print(f"  batch={batch}: {t['images_per_sec']} 张/s · {t['peak_vram_gb']}GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
