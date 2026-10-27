"""record_demo：自动录制 30 秒演示 GIF（Playwright 截图 → Pillow 合成）。

## 为什么不用 ScreenToGif

ScreenToGif 是**GUI 桌面程序**：录制前要手动框选屏幕区域、手动点录制/停止按钮。
即使能下载安装，自动化也只能靠猜坐标去点 —— 不可靠，而且**无法复现**：
别人想重录一遍，得重新对着屏幕框一遍。

这条路（Playwright + Pillow）的好处：
- **完全可编程**：每一步都是代码，可复现、可 diff、可进 CI
- **不依赖人工**：不需要有人守着屏幕点按钮
- **画面稳定**：固定 viewport，没有窗口遮挡、通知弹窗、鼠标乱入

代价：只录**浏览器页面**，不能录桌面其他窗口。对本项目够用。

## 一个踩到的硬限制：Playwright 同步 API 不能跨线程

第一版用 `threading.Thread` 在后台线程持续 `page.screenshot()`，
结果**一帧都没抓到**：Playwright 的 sync API 绑定在创建它的线程上，
从别的线程调用会静默失败（异常被我的 try/except 吞了）。

所以改用 **asyncio + 异步 API**：截图任务和页面动作在**同一个事件循环**里，
用 `asyncio.create_task` 并发 —— 这样既保证帧率均匀，又不跨线程。

## 录制脚本（约 30 秒）

    [0.0s]   空页面，拖放区
    [1.2s]   文件已选择
    [2.1s]   点"开始摘要"
    [3-9s]   进度条推进（分段摘要 1/5 … 5/5）
    [9-25s]  摘要逐字出现
    [25s]    点"中断" ← **这是要录的重点**
    [27s]    显示"已中断"提示

## 关键技术点：为什么不怕跑 30 秒

用 `page.route()` 拦截 `/api/v1/summarize/stream` 并**回放预置 SSE**。
前端真的在做 fetch + ReadableStream 解帧（代码路径完全真实），
只是数据来自回放而非真模型。好处：**确定、快、不花 API 费用**，
而且能精确控制每个事件的时间点 —— 真实 LLM 的抖动会让演示画面不稳定。

运行：

    uv run python -X utf8 scripts/record_demo.py
    uv run python -X utf8 scripts/record_demo.py --speed 4 --out docs/demo/_test.gif
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Users\17924\AppData\Local\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]

STATIC_DIR = PROJECT_ROOT / "static"
OUT_DIR = PROJECT_ROOT / "docs" / "demo"
VIEWPORT = {"width": 900, "height": 780}


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def find_browser() -> str | None:
    return next((p for p in CHROME_CANDIDATES if Path(p).exists()), None)


# ============================================================ 回放脚本

MOCK_SUMMARY = (
    "该文档系统规定了合同第 1 至第 50 项交付物的验收标准，以及对应的付款与违约条款。"
    "付款方面，甲方应在每次里程碑验收完成后十五个工作日内完成付款。"
    "违约方面，若第 N 期交付延期，乙方按合同总额的 N‰ 承担违约责任，"
    "比例自第 1 期的 1‰ 逐期递增至第 50 期的 50‰。"
)

# 进度阶段的起止（秒，相对流的开始）
PROGRESS_START = 0.6
PROGRESS_STEP = 1.1
DELTA_START = 6.6
DELTA_STEP = 0.055


def build_sse_script() -> list[tuple[float, str]]:
    """`(相对时间, SSE 帧)` 列表。"""

    def frame(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    out: list[tuple[float, str]] = [
        (
            0.0,
            frame(
                {
                    "accepted": {
                        "filename": "sample50.pdf",
                        "pages": 50,
                        "chars": 5437,
                        "chunks": 5,
                        "warnings": [],
                    }
                }
            ),
        )
    ]

    labels = ["第 1–10 页", "第 11–20 页", "第 21–30 页", "第 31–40 页", "第 41–50 页"]
    for i, label in enumerate(labels, start=1):
        out.append(
            (
                PROGRESS_START + (i - 1) * PROGRESS_STEP,
                frame({"progress": {"done": i, "total": 5, "label": label}}),
            )
        )

    for idx, char in enumerate(MOCK_SUMMARY):
        out.append((DELTA_START + idx * DELTA_STEP, frame({"delta": char})))

    end = DELTA_START + len(MOCK_SUMMARY) * DELTA_STEP
    out.append((end, frame({"meta": {"chunks": 5, "api_calls": 6, "elapsed_ms": int(end * 1000)}})))
    out.append((end + 0.1, "data: [DONE]\n\n"))
    return out


# ============================================================ 录制核心


async def record_with_real_server(
    *,
    fps: int,
    speed: float,
    out_path: Path,
    headless: bool,
    abort: bool,
    port: int,
) -> dict[str, Any]:
    """**推荐路径**：起一个真实 uvicorn（含真实 SSE 端点），前端跑真代码。

    为什么要真服务：前端用的是 `fetch` + `response.body.getReader()`，
    这套代码路径只有在**真的有 SSE 响应**时才会执行。
    如果我在网络层用 `route.fulfill` 一次性塞整个 body，
    前端仍然会走 ReadableStream，但只有一次 read —— 那就看不出流式效果。

    所以：真服务 + 真实分块写出。服务端的 chunk 间隔由环境变量控制，
    这样既真实又可加速。
    """
    from playwright.async_api import async_playwright

    browser_path = find_browser()
    if browser_path is None:
        raise RuntimeError("找不到 Chrome/Edge。可运行 playwright install chromium。")

    url = f"http://127.0.0.1:{port}/ui/"
    frames: list[tuple[float, bytes]] = []
    stats: dict[str, Any] = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=browser_path, headless=headless)
        page = await browser.new_page(viewport=VIEWPORT, device_scale_factor=1)

        t0 = time.perf_counter()
        grab_done = asyncio.Event()

        async def grabber() -> None:
            """按固定间隔截图，直到 grab_done 被置位。"""
            interval = 1.0 / fps
            n = 0
            while not grab_done.is_set():
                target = t0 + n * interval
                wait = target - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    png = await page.screenshot(type="png")
                    frames.append((time.perf_counter() - t0, png))
                except Exception:  # noqa: BLE001, S110 - 页面可能正在导航
                    pass
                n += 1

        grab_task = asyncio.create_task(grabber())

        try:
            await page.goto(url, wait_until="load")
            await asyncio.sleep(1.2)

            # 选择文件
            pdf = (PROJECT_ROOT / "tests" / "fixtures" / "sample50.pdf").read_bytes()
            await page.set_input_files(
                "#file",
                files=[{"name": "sample50.pdf", "mimeType": "application/pdf", "buffer": pdf}],
            )
            await asyncio.sleep(0.9)

            # 点开始（此时会打到真实服务）
            await page.click("#go")

            # 等到摘要出到一定比例再中断 —— 太早中断画面里几乎没有内容。
            # **阈值要跟实际文本长度对齐**：之前写死 200 字，但回放正文只有约 140 字，
            # 于是永远等不到，一路拖到 45 秒超时，录出来 55 秒且后半段是空转。
            from app.rag.replay import DEMO_SUMMARY

            threshold = int(len(DEMO_SUMMARY) * 0.55)
            with contextlib.suppress(Exception):
                await page.wait_for_function(
                    f"() => (document.getElementById('output').textContent || '').length > {threshold}",
                    timeout=30000,
                )

            # 点中断（演示重点）：让摘要先流一会儿，再打断
            if abort:
                await asyncio.sleep(1.0)
                enabled = await page.is_enabled("#abort")
                if enabled:
                    await page.click("#abort")
                    await asyncio.sleep(1.5)

            await asyncio.sleep(0.8)

            # 完成态截图
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            shot = OUT_DIR / "ui_screenshot.png"
            await page.screenshot(path=str(shot), full_page=True)
            stats["screenshot"] = str(shot.relative_to(PROJECT_ROOT))

        finally:
            grab_done.set()
            with contextlib.suppress(Exception):
                await grab_task
            await browser.close()

    count, width = compose_gif(frames, out_path, fps=fps)
    stats.update(
        {
            "frames_captured": len(frames),
            "frames_in_gif": count,
            "width": width,
            "gif_kb": round(out_path.stat().st_size / 1024, 1),
            "record_seconds": round(time.perf_counter() - t0, 1),
            "fps": fps,
        }
    )
    return stats


def compose_gif(
    frames: list[tuple[float, bytes]],
    out_path: Path,
    *,
    fps: int,
    colors: int = 64,
    max_width: int | None = 600,
    frame_step: int = 2,
) -> tuple[int, int]:
    """合成 GIF。相邻帧相同则延长上一帧，避免体积浪费。

    **体积控制的三把刀**（GIF 是 8 位色 + 帧间几乎无压缩，很容易上 MB）：

    | 手段 | 效果（实测） |
    | --- | --- |
    | 降色板 96 → 64 | 明显减小，UI 只有少量颜色，肉眼几乎无差 |
    | 降分辨率 900 → 600 | 约减 55%（面积按平方缩） |
    | 隔帧丢弃（`frame_step=2`，时长翻倍） | 再减约 40% |

    三者叠加：1867 KB → **749 KB**。演示的目的是"看得清在流式、能看到中断"，
    不是逐帧研究像素 —— 这个取舍是值得的。
    """
    from PIL import Image

    if not frames:
        raise ValueError("没有抓到任何帧")

    images: list[Image.Image] = []
    durations: list[int] = []
    frame_ms = int(1000 / fps)
    prev_png: bytes | None = None

    for _t, png in frames:
        if prev_png is not None and png == prev_png:
            durations[-1] += frame_ms
            continue
        prev_png = png
        images.append(Image.open(io.BytesIO(png)).convert("RGB"))
        durations.append(frame_ms)

    # 隔帧丢弃：合并相邻帧的时长，保持总时长不变
    if frame_step > 1 and len(images) > 4:
        kept_img: list[Image.Image] = []
        kept_dur: list[int] = []
        for i in range(0, len(images), frame_step):
            kept_img.append(images[i])
            merged = sum(durations[i : i + frame_step])
            kept_dur.append(merged)
        images, durations = kept_img, kept_dur

    width = min(i.width for i in images)
    height = min(i.height for i in images)
    images = [i.resize((width, height)) for i in images]

    # 降分辨率（等比缩放）
    if max_width is not None and width > max_width:
        scale = max_width / width
        width = max_width
        height = round(height * scale)
        images = [i.resize((width, height), Image.Resampling.LANCZOS) for i in images]

    quantized = [i.quantize(colors=colors, method=Image.Quantize.MEDIANCUT) for i in images]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return len(images), width


def main() -> int:
    _force_utf8_stdout()

    parser = argparse.ArgumentParser(description="录制流式摘要演示 GIF")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out", default=str(OUT_DIR / "streaming_summary.gif"))
    parser.add_argument("--port", type=int, default=8020, help="录制用的服务端口")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--no-abort", action="store_true", help="不点中断按钮")
    args = parser.parse_args()

    print(f"浏览器: {find_browser() or '未找到'}")
    print(f"目标  : {args.out}\n")

    try:
        info = asyncio.run(
            record_with_real_server(
                fps=args.fps,
                speed=1.0,
                out_path=Path(args.out),
                headless=not args.headed,
                abort=not args.no_abort,
                port=args.port,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n录制失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("=" * 56)
    for k, v in info.items():
        print(f"  {k:18} {v}")
    print("=" * 56)

    meta = Path(args.out).with_suffix(".json")
    meta.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {Path(args.out).name}（{info['gif_kb']} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
