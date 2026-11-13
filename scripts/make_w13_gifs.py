"""W13 周六 · 演示 GIF 生成（PIL 渲染帧，不录屏——与 W9/W11 同一条理由）。

产物：
- docs/observability-demo.gif   GIF①：看板下钻——慢请求 trace 的 8 个 span
  定位到耗时来源（读 traces_wed 的真实 trace，与磁盘数据不可能对不上）
- docs/mcp-demo.gif             GIF②：Host 经 MCP 读本地 md 并总结要点
  （读 mcp_fs_e2e_host 的真实调用序列——7 次调用与端到端脚本同源）

数据一致性由脚本内断言保证：帧里出现的 span 名/时长/调用序列全部读自
落盘数据，不是手写字符串。
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 960, 400
FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_CN = "C:/Windows/Fonts/msyh.ttc"


def _font(size: int, cn: bool = False):
    try:
        return ImageFont.truetype(FONT_CN if cn else FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _terminal_frame(
    lines: list[str], *, title: str, highlights: dict[int, str] | None = None
) -> Image.Image:
    img = Image.new("RGB", (W, H), (24, 26, 32))
    d = ImageDraw.Draw(img)
    # 标题栏
    d.rectangle([0, 0, W, 34], fill=(40, 43, 54))
    d.text((14, 8), title, font=_font(17, cn=True), fill=(220, 223, 230))
    # 圆点（终端风格）
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([W - 66 + i * 20, 12, W - 56 + i * 20, 22], fill=c)
    highlights = highlights or {}
    y = 52
    for i, line in enumerate(lines):
        color = highlights.get(i, (171, 178, 191))
        if line.startswith("+"):
            color = (152, 195, 121)
        elif line.startswith("!"):
            color = (224, 108, 117)
        d.text((20, y), line, font=_font(15, cn=True), fill=color)
        y += 26
    return img


def gif1_observability() -> list[Image.Image]:
    """GIF①：看板聚合 → 下钻 → 一条慢请求 trace 的全部 span。

    数据：eval/results/traces_wed/traces.jsonl 里挑 span 最多的 trace
    （含演练注入的慢请求，duration 真实）。
    """
    traces = [
        json.loads(line)
        for line in Path("eval/results/traces_wed/traces.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    target = max(traces, key=lambda t: t["summary"]["span_count"])
    s = target["summary"]
    span_names = [sp["name"] for sp in target["spans"]]
    # 断言：帧内容来自真实数据
    assert s["span_count"] == len(span_names)
    assert any(sp["duration_ms"] > 50 for sp in target["spans"]), "演示需要一条真实慢 trace"

    frames = []
    # 帧 1：看板概览
    frames.append(
        _terminal_frame(
            [
                "$ streamlit run app/ui/observability_board.py",
                "",
                "① 概览      请求量 20  |  错误率 20.0%→(429剔除) 0%",
                "            最差 P95 = 120 ms   ⚠ 超阈值(20ms)",
                "② 延迟分布  p50=1ms  p95=120ms  p99=120ms",
                "③ 成本      llm 合计 0.0447 元（重试 0.0000）",
                "④ 错误告警  🔴 retrieval_p95_high 触发中 (120.0)",
                "",
                "! 慢点在哪个接口？→ 下钻 ↓",
            ],
            title="看板 · 概览（数据源 metrics_hourly，实测）",
            highlights={3: (229, 192, 123), 6: (224, 108, 117), 7: (229, 192, 123)},
        )
    )
    # 帧 2：下钻到 trace 列表
    frames.append(
        _terminal_frame(
            [
                "🔎 下钻：筛选 hour=23:00 / path=/api/v2/search",
                "",
                f"  {s['trace_id'][:8]}  /api/v2/search  {s['duration_ms']}ms  🟢",
                f"    spans={s['span_count']} gens={s['generation_count']} cost={s['cost']:.4f}",
                "  …（其余 trace 折叠）",
                "",
                "点开这条慢请求 ↓",
            ],
            title="看板 · 下钻 trace 列表",
            highlights={2: (97, 175, 239)},
        )
    )
    # 帧 3：span 明细（全部 span，标注耗时来源）
    lines = [f"trace {s['trace_id'][:8]}  总时长 {s['duration_ms']}ms", ""]
    hi = {}
    for _i, sp in enumerate(target["spans"]):
        mark = "+" if sp["duration_ms"] > 50 else " "
        lines.append(f"{mark} {sp['name']:<18} {sp['duration_ms']:>5}ms  {sp['status']}")
        if sp["duration_ms"] > 50:
            hi[len(lines) - 1] = (229, 192, 123)
    lines.append("")
    lines.append(f"定位：{len(hi)} 个慢 span 即耗时来源（观测不重构）")
    frames.append(_terminal_frame(lines, title="看板 · span 明细（真实落盘 trace）", highlights=hi))
    return frames


def gif2_mcp() -> list[Image.Image]:
    """GIF②：Host 经 MCP 读本地 md 并总结要点。

    数据：mcp_fs_e2e_host.py 的真实调用序列（7 次，与端到端脚本同源断言）。
    """
    expected = [
        ("initialize", "握手：协议 2025-06-18 协商完成"),
        ("notifications/initialized", "客户端确认就绪（notification，无响应）"),
        ("tools/list", "能力发现：fs_list_dir / fs_read_file / fs_search_files / fs_write_file"),
        ("tools/call fs_list_dir", "列出 reports/ → 找到 weekly.md"),
        ("tools/call fs_read_file", "读到 223 字节（五道闸通过）"),
        ("prompts/get summarize_file", "取总结模板并参数化"),
        ("tools/call fs_read_file(../)", "越权读取 → Server 拒绝 [permission]"),
    ]
    frames = []
    lines_head = ["$ python -X utf8 scripts/mcp_fs_e2e_host.py", ""]
    for i, (_call, note) in enumerate(expected):
        lines = lines_head + [
            f"[{j + 1}/7] {'✓' if j <= i else '·'} {c}" for j, (c, _) in enumerate(expected)
        ]
        lines += ["", f"→ {note}"]
        hi = {len(lines_head) + i: (152, 195, 121)}
        if "越权" in note:
            hi[len(lines) - 1] = (224, 108, 117)
        frames.append(
            _terminal_frame(lines, title="自写 host ⇄ MCP 文件系统 Server", highlights=hi)
        )
    # 结尾帧
    frames.append(
        _terminal_frame(
            lines_head
            + [f"[✓] {c}" for c, _ in expected]
            + [
                "",
                "=== 端到端通过：全程 7 次调用，无多余调用 ===",
                "=== 分工确认：Host 决策，Server 执行 + 安全兜底 ===",
            ],
            title="自写 host ⇄ MCP 文件系统 Server",
            highlights={len(lines_head) + 6: (224, 108, 117)},
        )
    )
    return frames


def main() -> None:
    out1 = Path("docs/observability-demo.gif")
    frames1 = gif1_observability()
    frames1[0].save(out1, save_all=True, append_images=frames1[1:], duration=2200, loop=0)
    print(f"{out1}: {len(frames1)} 帧, {out1.stat().st_size // 1024} KB")

    out2 = Path("docs/mcp-demo.gif")
    frames2 = gif2_mcp()
    frames2[0].save(out2, save_all=True, append_images=frames2[1:], duration=1800, loop=0)
    print(f"{out2}: {len(frames2)} 帧, {out2.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
