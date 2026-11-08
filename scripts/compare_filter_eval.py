"""周日回归核对：W7 小样本（filter_eval，gold=20）复跑 vs 报告对账。"""

import json
from pathlib import Path

o = json.loads(Path("scripts/filter_eval.json").read_text(encoding="utf-8"))
r = json.loads(Path("eval/results/filter_eval_rerun.json").read_text(encoding="utf-8"))
print("orig: chunks", o["meta"]["chunks"], "gold", o["meta"]["gold"])
print("rerun: chunks", r["meta"]["chunks"], "gold", r["meta"]["gold"])
for so, sr in zip(o["scenarios"], r["scenarios"], strict=False):
    print(
        f"{so['scenario']}: orig {so['hit_before']}->{so['hit_after']} "
        f"({so['hit_rate_after']:.0%}) | rerun {sr['hit_before']}->{sr['hit_after']} "
        f"({sr['hit_rate_after']:.0%})"
    )
same = all(
    so["hit_rate_after"] == sr["hit_rate_after"]
    for so, sr in zip(o["scenarios"], r["scenarios"], strict=False)
)
print("hit rates identical:", same)
