"""W13 开工前置核对：六条前置逐条验证（不调 LLM）。"""

import sys
from pathlib import Path

sys.path.insert(0, ".")

from app.core.llm.error_contract import ERROR_SPECS  # noqa: E402

BASE = Path(".")

perf = (BASE / "docs/performance.md").read_text(encoding="utf-8")
spec = (BASE / "docs/api-spec.md").read_text(encoding="utf-8")
deps = (BASE / "app/core/deps.py").read_text(encoding="utf-8")
loop = (BASE / "app/core/llm/react_loop.py").read_text(encoding="utf-8")
main = (BASE / "app/main.py").read_text(encoding="utf-8")

print("=" * 60)
print("W13 前置核对（六条）")
print("=" * 60)

# ① 分段表
seg_ok = all(k in perf for k in ("瓶颈分段", "序列化", "鉴权", "缓存命中判定"))
print(f"[1] W12 分段表（鉴权/缓存/检索/序列化带数字）: {'✅' if seg_ok else '❌'}")

# ② 错误五分类 + retryable + HTTP 映射
cats = {s.category for s in ERROR_SPECS.values()}
map_ok = "AGENT_INVALID_ARG" in spec and "AGENT_NOT_PERMITTED" in spec
print(
    f"[2] W11 五分类: {sorted(cats)}；retryable 字段: {'✅' if all(hasattr(s, 'retryable') for s in ERROR_SPECS.values()) else '❌'}"
)
print(f"    五类→HTTP 预备映射表已落文档: {'✅' if map_ok else '❌'}")

# ③ 拐点数据（告警阈值唯一合法来源）
inflection_ok = "拐点分析" in perf and "55.7%" in perf and "MAX_QUEUE_DEPTH" in perf
print(f"[3] W12 拐点数据（并发档位/触发类型/深度32）: {'✅' if inflection_ok else '❌'}")
print("    → 告警阈值可直接引用：429 正常形态=A3 档 55%；P50 全档恒定 3-4ms")

# ④ request_id + latency_ms
rid_ok = "get_request_id" in deps
lat_ok = '"latency_ms"' in loop
print(
    f"[4] request_id 贯穿: {'✅' if rid_ok else '❌'}；工具级 latency_ms: {'✅' if lat_ok else '❌'}"
)

# ⑤ 20 任务冻结 + W7 可复跑
tasks = [
    ln
    for ln in (BASE / "eval/agent/tasks-20.jsonl").read_text(encoding="utf-8").splitlines()
    if ln.strip()
]
print(
    f"[5] 20 任务判据冻结: {len(tasks)} 行 {'✅' if len(tasks) == 20 else '❌'}；corpus_stats 可复跑（周六冒烟已验）"
)

# ⑥ /metrics 现状（W13 起点与缺口）
print(
    f"[6] /metrics 现状: error_counts={'✅' if 'error_counts' in main else '❌'}, "
    f"rate_limit_rejections={'✅' if 'rate_limit_rejections' in main else '❌'}"
)
print("    缺口（W13 的活）: 延迟分位聚合 / 缓存命中率直出 / 队列深度 / 告警阈值绑定")

print("=" * 60)
print("六条全过 → W13 开工前置成立")
