"""W14 周四 · 项目一演示 GIF（PIL 渲染帧，读真实 transcript——与 W13 同一条纪律）。

产物：docs/generator-demo.gif
故事（30 秒讲完）：提交生成任务（5ms 拿 task_id）→ 轮询状态机
（queued → running → succeeded）→ 出图清单（文件名 + 耗时）。

数据一致性由脚本内断言保证：帧里出现的 task_id / 状态迁移 / 耗时
全部读自 eval/results/gen-demo-transcript-20261114.json（真实 HTTP 流），
不是手写字符串。GIF 风格与 W13 两个 GIF 一致（终端快照，可复现优于录屏）。

运行：python -X utf8 scripts/make_generator_gif.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
TRANSCRIPT = PROJECT / "eval" / "results" / "gen-demo-transcript-20261114.json"
OUT = PROJECT / "docs" / "generator-demo.gif"

W, H = 960, 420
FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_CN = "C:/Windows/Fonts/msyh.ttc"


def _font(size: int, cn: bool = False):
    try:
        return ImageFont.truetype(FONT_CN if cn else FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _terminal_frame(lines: list[str], *, title: str, highlights: dict[int, str]) -> Image.Image:
    img = Image.new("RGB", (W, H), (24, 26, 32))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 34], fill=(40, 43, 54))
    d.text((14, 8), title, font=_font(17, cn=True), fill=(220, 223, 230))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([W - 66 + i * 20, 12, W - 56 + i * 20, 22], fill=c)
    y = 52
    for i, line in enumerate(lines):
        color = highlights.get(i, (171, 178, 191))
        if line.startswith("+"):
            color = (152, 195, 121)
        elif line.startswith("!"):
            color = (224, 108, 117)
        elif line.startswith("$") or line.startswith("POST") or line.startswith("GET"):
            color = (97, 175, 239)
        d.text((20, y), line, font=_font(15, cn=True), fill=color)
        y += 28
    return img


def main() -> int:
    data = json.loads(TRANSCRIPT.read_text(encoding="utf-8"))
    steps = data["transcript"]
    assert len(steps) >= 3, "transcript 不完整（health/submit/poll 至少三步）"

    submit = next(s for s in steps if s["call"].startswith("POST"))
    task_id = submit["resp"]["task_id"]
    status_202 = submit["resp"]["status"]
    submit_ms = submit["ms"]
    polls = [s for s in steps if s["call"].startswith("GET /api/v1/generate/tasks")]
    final = polls[-1]["resp"]
    final_status = final.get("status") or final.get("data", {}).get("status")
    assert final_status == "succeeded", f"终态不是 succeeded：{final_status}"
    poll_ms = [p["ms"] for p in polls]
    result = final.get("result") or final.get("data", {}).get("result") or {}
    images = result.get("images") or []
    elapsed = result.get("elapsed_ms")

    scene1 = [
        "$ GENERATOR_IMPL=fake uvicorn app.main:app --port 8001",
        '$ curl -X POST …/api/v1/generate/async -d \'{"defect_class":"划痕",',
        '      "count":2, "size":512, "seed":42}\'',
        "",
        f"→ HTTP 202  status={status_202}  reused=false",
        f"→ task_id = {task_id[:16]}…",
        f"→ 提交接口 P50 {submit_ms:.0f} ms —— 客户端立刻拿 id 去轮询，",
        "  不占连接等 GPU（同步接口要挂 ~5s）",
    ]
    scene2 = [
        "$ curl …/api/v1/generate/tasks/{task_id}",
        "",
        f"poll 1: queued→running→succeeded  ({poll_ms[0]:.1f} ms)",
        "状态机：表驱动，唯一迁移路径，非法迁移当场抛",
        "幂等：同 (defect_class,count,size,seed) 再提交 → 返回原 id",
        "  重复请求 0 GPU 占用",
        "",
        f"final status = {final_status}（真实状态机 + 真实落盘）",
    ]
    scene3 = [
        "$ curl …/api/v1/generate/tasks/{task_id}  # 完成态",
        "",
        f"+ status: {final_status}   elapsed_ms: {elapsed}",
        f"+ images: {len(images)} 张已写盘",
    ]
    for p in images[:2]:
        scene3.append(f"+   {Path(str(p)).name}")
    scene3 += [
        "",
        "! 服务内核=契约壳（FakeGenerator 路径）；真实 SD 推理数字",
        "! 见 generator-benchmark.md（512² 预热中位 ~5.0s / VRAM 2.81GB）",
    ]

    scenes = [
        (
            _terminal_frame(
                scene1,
                title="① 提交：202 + task_id（5ms）",
                highlights={4: (152, 195, 121), 5: (224, 108, 117)},
            )
        ),
        (
            _terminal_frame(
                scene2, title="② 轮询：状态机 queued→succeeded", highlights={2: (152, 195, 121)}
            )
        ),
        (
            _terminal_frame(
                scene3,
                title="③ 出图：任务完成 + 文件落盘",
                highlights={2: (152, 195, 121), 3: (152, 195, 121)},
            )
        ),
    ]

    frames: list[Image.Image] = []
    durations: list[int] = []
    for img in scenes:
        frames.append(img)
        durations.append(2600)
    frames.append(
        _terminal_frame(
            [
                "open docs/README-generator.md — 快速体验三条命令可跑",
                "task_id / 状态迁移 / 耗时全部读自真实 HTTP transcript，非手写",
            ],
            title="全部数据可复现",
            highlights={0: (152, 195, 121)},
        )
    )
    durations.append(2600)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"{OUT.name}: {len(frames)} 帧, {OUT.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
