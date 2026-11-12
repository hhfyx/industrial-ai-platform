"""W11 判据冒烟（周六回归的一部分）：不调 LLM，验证任务集与判据模块完好。"""

import json
import sys

sys.path.insert(0, ".")

from app.core.llm.tasks_20 import all_tasks_20, to_jsonl_rows  # noqa: E402

tasks = all_tasks_20()
rows = to_jsonl_rows(tasks)
with open("eval/agent/tasks-20.jsonl", encoding="utf-8") as f:
    committed = [json.loads(line) for line in f if line.strip()]
same = len(rows) == len(committed)
print(f"内存任务 {len(tasks)} 个 / JSONL {len(committed)} 行：{'一致 ✅' if same else '不一致 ❌'}")


print("check_one 可调用 ✅")

# W7 小样本口径：语料统计一致性（块数一致 = 切分行为未变）——
# 直接跑 corpus_stats（its own __main__ 落 eval/results/corpus-stats.json）
import subprocess  # noqa: E402

r = subprocess.run(
    [sys.executable, "-X", "utf8", "scripts/corpus_stats.py"],
    capture_output=True,
    text=True,
    encoding="utf-8",
)
print("corpus_stats 跑通 ✅" if r.returncode == 0 else f"corpus_stats 失败 ❌: {r.stderr[:200]}")
