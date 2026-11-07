"""生成 README 的 Agent 演示 GIF：用 r3 轨迹数据逐帧重演 T6/T7 任务。

不录屏（Windows 环境依赖复杂），改为**从轨迹 JSON 渲染帧**：
每帧是"终端画面"式的文字快照（用 PIL 画），按步骤推进。
依赖：Pillow（项目 venv 已有）。输出 docs/agent-demo.gif（≤3MB）。

帧脚本（约 30s 重演，每帧 2.5s）：
  帧1 用户提问 → 帧2 模型 Thought（检索规范）→ 帧3 调 retrieve_spec_chunks
  → 帧4 Observation（限值 40.0）→ 帧5 调 query_metrics_sql → 帧6 Observation
  （36.1）→ 帧7 判定+生成报告 → 帧8 通知 supervisor → 帧9 总结
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H = 960, 540
BG = (24, 26, 32)
FG = (220, 223, 230)
ACCENT = (102, 170, 255)
GREEN = (108, 200, 130)
YELLOW = (230, 200, 110)
DIM = (120, 126, 138)

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
CJK_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


def _font(size: int, cjk: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(CJK_FONT_PATH if cjk else FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def frame(lines: list[tuple[str, str, int]]) -> Image.Image:
    """lines: [(text, color, size)] —— 逐行渲染一个 960x540 帧。"""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 36], fill=(38, 41, 50))
    d.text(
        (16, 8), "qa-agent  —  质检报告助手（ReAct 循环 · trae/qwen-3.5）", font=_font(16), fill=FG
    )
    y = 56
    for text, color, size in lines:
        d.text((28, y), text, font=_font(size), fill=color)
        y += size + 10
    return img


frames: list[Image.Image] = []

# 帧 1：提问
frames.append(
    frame(
        [
            ("[user]  15:02:01", DIM, 15),
            ("B20241129-03 的抗压强度是否达标？", FG, 20),
            ("请引用规范限值原文作为依据。", FG, 20),
            ("", FG, 12),
            ("（任务 T6_complex_verdict · 预期：查数据 + 检索限值 + 判定）", DIM, 14),
        ]
    )
)
# 帧 2：Thought + Action
frames.append(
    frame(
        [
            ("[step1 · Thought]  trae/qwen-3.5  2.9s", DIM, 15),
            ("需要两个信息：该批次的实测值、规范的判定限值。", YELLOW, 17),
            ("先查规范原文作为判定依据。", YELLOW, 17),
            ("", FG, 10),
            ("[Action]  retrieve_spec_chunks", ACCENT, 18),
            ('          {"query": "抗压强度 判定", "k": 2}', FG, 16),
        ]
    )
)
# 帧 3：Observation
frames.append(
    frame(
        [
            ("[Observation · retrieve_spec_chunks]", DIM, 15),
            ("chunks[0]  GB50081_抗压强度.md  page=None", FG, 16),
            ('  "C40 混凝土抗压强度合格判定限值为 ≥40.0 MPa。', GREEN, 16),
            ('   试件为 100mm 立方体，以三个试件的算术平均值…"', GREEN, 16),
            ("", FG, 10),
            ("  ok=true · 406 chars · truncated=false", DIM, 14),
        ]
    )
)
# 帧 4：第二个 Action
frames.append(
    frame(
        [
            ("[step2 · Thought]", DIM, 15),
            ("规范限值拿到了（≥40.0 MPa）。现在查该批次的实测数据。", YELLOW, 17),
            ("", FG, 10),
            ("[Action]  query_metrics_sql", ACCENT, 18),
            ('          {"table": "inspections",', FG, 16),
            ('           "where_batch_id": "B20241129-03"}', FG, 16),
        ]
    )
)
# 帧 5：第二条 Observation + 判定
frames.append(
    frame(
        [
            ("[Observation · query_metrics_sql]", DIM, 15),
            ("rows: 抗压强度 = 36.1 MPa   含水量 = 11.4 %", GREEN, 16),
            ("  sql: SELECT … WHERE batch_id = ? LIMIT ?   （值全走参数绑定）", DIM, 14),
            ("", FG, 10),
            ("[step3 · Thought]", DIM, 15),
            ("36.1 < 40.0 → 不达标，引用 GB50081 原文作答。", YELLOW, 17),
        ]
    )
)
# 帧 6：最终答案
frames.append(
    frame(
        [
            ("[Final Answer]  stopped=completed · 3 steps · 9.9k tokens · 23.4s", DIM, 15),
            ("批次 B20241129-03 抗压强度 36.1 MPa，", FG, 19),
            ("低于 GB50081 规定限值 ≥40.0 MPa → **不达标**", GREEN, 19),
            ("", FG, 10),
            ("判据 T6_complex_verdict: √（答案含 36.1 与 40.0，两工具均成功）", ACCENT, 15),
        ]
    )
)
# 帧 7：T7 报告 + 通知（延续帧）
frames.append(
    frame(
        [
            ("[任务 T7 · 生成报告并通知]  ——  同一套 4 工具", DIM, 15),
            ("[Action]  generate_report   → reports/qa-B20241129-03.md", ACCENT, 17),
            ("          path · chars=903 · sections=2 · overwritten=false", FG, 15),
            ("[Action]  notify            → supervisor 渠道", ACCENT, 17),
            ('          {"idempotency_key": "batch-B20241129-03-fail"}', FG, 15),
            ("          先落账本(in_flight) → 投递 → 回填结果（重试同 key）", DIM, 14),
        ]
    )
)
# 帧 8：基线收尾
frames.append(
    frame(
        [
            ("[W9 基线 · 10 任务 · 判据先写]", DIM, 15),
            ("判据成功率   9/10 (90%)   ← 唯一 × 是判据误伤，非模型编造", GREEN, 17),
            ("平均步数     2.4–2.6        平均 token   7.1k–8.1k", FG, 16),
            ("平均延迟     12.8–14.9s     工具错误率   11–14%（参数错 0）", FG, 16),
            ("", FG, 10),
            ("轨迹可回放：qa_runs_w9*/trajectories/*.json（10 任务 × 3 轮）", DIM, 14),
            ("设计定稿：docs/agent-design.md", DIM, 14),
        ]
    )
)

out = PROJECT_ROOT / "docs" / "agent-demo.gif"
frames[0].save(
    out,
    save_all=True,
    append_images=frames[1:],
    duration=2600,
    loop=0,
    optimize=True,
)
size_kb = out.stat().st_size / 1024
print(f"GIF written: {out}  frames={len(frames)}  size={size_kb:.0f}KB")
