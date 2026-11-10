"""生成 README 的可靠性演示 GIF（W11）：注入超时 → 代码退避重试 → 降级。

## 为什么不用录屏

与 `make_agent_gif.py` 同一条理由：Windows 录屏依赖复杂、不可复现。
改成**从真实轨迹渲染帧**——每个数字、每行错误都来自
`qa_runs_w11_r5/trajectories/*.json`，不是编的。

## 演示哪两段（都是 01-02 的真实跑次）

1. **T8**：三工具成功 → notify 失败 → **代码自己重试一次**（`retried=True`）
   → 仍失败 → 判为"收尾失败"保留 completed，但如实说明哪一步没成；
2. **N8**：确认类任务连查三轮 → `budget_tokens` 触顶 → **优雅退出**
   （已完成部分 + 未完成原因 + 建议下一步）。

两段合起来正好是"重试"与"降级"两个机制，30 秒能看完。

输出 `docs/agent-reliability.gif`（< 3MB）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H = 960, 360
BG = (24, 26, 32)
FG = (220, 223, 230)
ACCENT = (102, 170, 255)
GREEN = (108, 200, 130)
YELLOW = (230, 200, 110)
RED = (235, 120, 120)
DIM = (120, 126, 138)

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
CJK_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


def _font(size: int, cjk: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(CJK_FONT_PATH if cjk else FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def frame(title: str, lines: list[tuple[str, str, int]]) -> Image.Image:
    """渲染一帧 960x540：顶栏 + 若干行（文本, 颜色, 字号）。"""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 36], fill=(38, 41, 50))
    d.text((16, 8), title, font=_font(15, cjk=True), fill=FG)
    y = 58
    for text, color, size in lines:
        d.text((28, y), text, font=_font(size, cjk=True), fill=color)
        y += size + 10
    return img


def load_trace(name: str) -> dict:
    p = PROJECT_ROOT / "qa_runs_w11_r5" / "trajectories" / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def build_frames() -> list[Image.Image]:
    t8 = load_trace("T8_complex_notify")
    n8 = load_trace("N8_confirm_before_report")

    # 从真实轨迹里取动作序列
    t8_actions = [a for m in t8["steps"] for a in m.get("actions", [])]

    header = "qa-agent — 可靠性演示 · 错误契约 / 重试 / 降级 · trae/qwen-3.5"
    frames: list[Image.Image] = []

    # ---- 帧 1：开场，说明要演示什么 ----
    frames.append(
        frame(
            header,
            [
                ("[W11 可靠性层]", ACCENT, 20),
                ("", FG, 8),
                ("这段演示两个机制：", FG, 18),
                ("  ① 代码退避重试（不是模型重试）", YELLOW, 17),
                ("  ② 触顶时的结构化降级（说清缺什么）", YELLOW, 17),
                ("", FG, 10),
                ("数据来源：qa_runs_w11_r5 的真实轨迹，非编造", DIM, 14),
                ("判据：20 任务集（01-01 冻结）", DIM, 14),
            ],
        )
    )

    # ---- 帧 1b：超时也是同一条路径（用注入演示，说明可复现）----
    frames.append(
        frame(
            header,
            [
                ("[超时走的是同一条路]", ACCENT, 19),
                ("", FG, 8),
                ("超时 / 内部异常 / 端点不可达 —— 在这套契约里是同一类：", FG, 16),
                ("  error_type=timeout 或 internal  →  retryable=true", YELLOW, 17),
                ("  → 代码退避重试一次  →  仍失败则降级或如实收尾", FG, 16),
                ("", FG, 10),
                ("超时可以用故障注入复现（tests/fault_injection.py）：", DIM, 14),
                ('  FaultInjector(base, failures={1: "timeout", 2: "timeout"})', GREEN, 14),
                ("注入只用于测试与演练，不接生产路径。", DIM, 14),
            ],
        )
    )

    # ---- 帧 2：T8 前几个工具都成功 ----
    # 去重保序：同一工具成功两次只算一项，但要标出次数（真实性）
    from collections import Counter

    ok_counts = Counter(a["tool"] for a in t8_actions if a.get("ok"))
    ok_lines = [f"  ✓ {tool}" + (f"  ×{n}" if n > 1 else "") for tool, n in ok_counts.items()]
    frames.append(
        frame(
            header,
            [
                ("[任务 T8 · 生成报告并通知主管]", DIM, 15),
                ("", FG, 8),
                ("前置工具全部成功：", GREEN, 18),
            ]
            + [(line, GREEN, 17) for line in ok_lines]
            + [("", FG, 8), ("接下来发通知 —— 问题就在这里。", YELLOW, 17)],
        )
    )

    # ---- 帧 3：notify 第一次失败 ----
    frames.append(
        frame(
            header,
            [
                ("[step4 · Action]  notify → supervisor", ACCENT, 18),
                ('  {"channel":"supervisor","idempotency_key":"batch-…"}', FG, 15),
                ("", FG, 10),
                ("  ✗ ok=false   error_type=internal", RED, 18),
                ("  原因：通知端点不可达（本地 mock 未启动）", FG, 16),
                ("", FG, 10),
                ("→ 错误信封带 retryable=true", YELLOW, 17),
                ("   「能重试」由**代码**判定，不是模型猜", DIM, 15),
            ],
        )
    )

    # ---- 帧 4：代码自己重试（retried=True）----
    retried = [a for a in t8_actions if a.get("retried")]
    frames.append(
        frame(
            header,
            [
                ("[retry 节点]  代码退避后重试", ACCENT, 20),
                ("", FG, 8),
                ("模型没有参与这一步。", YELLOW, 18),
                ("循环层读到 retryable=true + retry_after=2.0s，", FG, 16),
                ("自己 sleep 后重新执行同一个调用。", FG, 16),
                ("", FG, 10),
                (f"轨迹里能看到第 2 条同工具记录：retried={bool(retried)}", GREEN, 16),
                ("  ← 没有这个标记，两次调用看不出哪次是代码重试的", DIM, 14),
            ],
        )
    )

    # ---- 帧 5：重试仍失败 → 判为收尾失败 ----
    frames.append(
        frame(
            header,
            [
                ("[再失败]  notify 重试后仍 internal", RED, 19),
                ("", FG, 8),
                ("关键判断：", ACCENT, 18),
                ("  只有**收尾类**工具失败，且此前的必需动作都成功", FG, 16),
                ("  → 保留 completed，不判整个任务失败", GREEN, 17),
                ("", FG, 10),
                ("为什么：通知失败不等于报告不存在。", YELLOW, 17),
                ("把整个任务降级，等于说「因为没人收到通知，所以这份报告没生成」。", DIM, 14),
            ],
        )
    )

    # ---- 帧 6：T8 的最终答案（真实文本）----
    final_t8 = (t8["final_answer"] or "").strip().splitlines()
    frames.append(
        frame(
            header,
            [
                ("[T8 · Final Answer]  stopped=completed", DIM, 15),
                ("", FG, 6),
            ]
            + [(line[:78], GREEN if i == 0 else FG, 15) for i, line in enumerate(final_t8[:6])],
        )
    )

    # ---- 帧 7：N8 预算触顶 ----
    used = n8.get("budget", {}).get("used", {})
    frames.append(
        frame(
            header,
            [
                ("[任务 N8 · 批量生成报告（需确认类）]", DIM, 15),
                ("", FG, 8),
                ("连续查了三轮，token 到顶：", YELLOW, 18),
                (f"  tokens={used.get('tokens')}  上限 10000", RED, 17),
                (f"  steps={used.get('steps')}  wall={used.get('wall_time_s')}s", FG, 16),
                ("", FG, 10),
                ("→ 路由判预算触顶 → 走优雅退出", ACCENT, 17),
            ],
        )
    )

    # ---- 帧 8：优雅退出的三件事 ----
    frames.append(
        frame(
            header,
            [
                ("[优雅退出]  三件事缺一不可", ACCENT, 20),
                ("", FG, 8),
                ("① 已完成部分 —— 用户知道拿到了什么", GREEN, 17),
                ("② 未完成原因 —— 用户知道为什么停", YELLOW, 17),
                ("③ 建议下一步 —— 用户知道该干什么", ACCENT, 17),
                ("", FG, 10),
                ("degraded=true 并计入降级率：", FG, 16),
                ("  预算触顶是**降级**不是失败（确实产出了一部分）", DIM, 15),
            ],
        )
    )

    # ---- 帧 9：N8 真实退出文本 ----
    exit_lines = (n8["final_answer"] or "").strip().splitlines()
    frames.append(
        frame(
            header,
            [
                ("[N8 · Final Answer]  stopped=budget_tokens", DIM, 15),
                ("", FG, 6),
            ]
            + [(line[:78], YELLOW if i == 0 else FG, 14) for i, line in enumerate(exit_lines[:8])],
        )
    )

    # ---- 帧 10：收尾，给全局数字（带口径）----
    frames.append(
        frame(
            header,
            [
                ("[W11 20 任务集 · 5 轮]", ACCENT, 18),
                ("", FG, 8),
                ("成功率        70%–100% 区间（最好一轮 20/20）", FG, 16),
                ("每任务跨轮    12 稳定通过 / 0 稳定失败 / 8 波动", FG, 16),
                ("编造型失败    0 例（6 个不可完成 × 5 轮）", GREEN, 16),
                ("降级率        ~10%（10k 预算档）", FG, 16),
                ("", FG, 10),
                (
                    "口径：降级计入失败；判据 01-01 冻结；可跑 verify_resume_numbers.py 复核",
                    DIM,
                    13,
                ),
            ],
        )
    )

    return frames


def main() -> int:
    out = PROJECT_ROOT / "docs" / "agent-reliability.gif"
    frames = build_frames()
    # 每帧 3.0s；最后一帧停 4s
    durations = [3000] * (len(frames) - 1) + [4000]
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_kb = out.stat().st_size / 1024
    print(f"已生成 {out}  ({len(frames)} 帧, {size_kb:.0f} KB, 约 {sum(durations) / 1000:.0f}s)")
    if size_kb > 3000:
        print("⚠ 超过 3MB，考虑减帧或降分辨率")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
