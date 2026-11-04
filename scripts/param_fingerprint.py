"""查看/更新链路参数指纹。

## 用法

```bash
python -X utf8 scripts/param_fingerprint.py           # 看当前值 + 与基线的差异
python -X utf8 scripts/param_fingerprint.py --update  # 改完参数并重跑实验后更新基线
python -X utf8 scripts/param_fingerprint.py --json    # 给 CI 用
```

## 什么时候需要 `--update`

**只有一种情况**：你有意改了链路参数，并且已经

1. 重跑了受影响的实验（`make ablation-four` / `make latency-profile`）
2. 更新了文档里的数字与结论

**跳过 1、2 只做 `--update` 是最坏的做法** —— 它等于把
"文档与代码脱节"这件事盖章确认了。

## 为什么需要这个机制

`app/rag/settings.py` 的 `RagConfig` 只有 5 个字段，
而 `DEFAULT_RRF_K`、`DEFAULT_CANDIDATES`、`DEFAULT_MAX_LENGTH`
这些**决定消融结论**的参数是模块常量 —— 改一行，测试全绿。

详见 `eval/param_fingerprint.py` 的模块 docstring。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.param_fingerprint import (  # noqa: E402
    FINGERPRINT_PATH,
    TRACKED_PARAMS,
    compute,
    diff,
    load_baseline,
    write_baseline,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="链路参数指纹")
    parser.add_argument("--update", action="store_true", help="把当前值写为基线")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    current = compute()

    if args.update:
        write_baseline(current)
        if args.json:
            print(json.dumps(current, ensure_ascii=False, indent=2))
        else:
            print(f"✓ 已更新基线：{FINGERPRINT_PATH}")
            print()
            print("⚠ 确认你已经：① 重跑了受影响的实验 ② 更新了文档结论。")
            print("  只更新指纹而不重跑实验，等于把'文档与代码脱节'盖章确认。")
        return 0

    baseline = load_baseline()
    changes = diff(baseline, current)

    if args.json:
        print(
            json.dumps(
                {"digest": current["digest"], "values": current["values"], "changes": changes},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print("=" * 74)
    print("链路参数指纹")
    print("=" * 74)
    reasons = {f"{m}.{a}": r for m, a, r in TRACKED_PARAMS}
    for key, value in current["values"].items():
        print(f"  {key}")
        print(f"      = {value}")
        if key in reasons:
            print(f"      （{reasons[key]}）")
    print()
    print(f"  digest: {current['digest'][:32]}…")
    print()

    if baseline is None:
        print("  还没有基线。首次建立：--update")
        return 0

    if not changes:
        print("✅ 与基线一致（没有被静默改动的链路参数）")
        return 0

    print("⚠ 与基线**不一致**：")
    for change in changes:
        print(f"    {change}")
    print()
    print("  这些参数影响已发布的评测结论。两种处理：")
    print("    A. 误改 -> 恢复原值")
    print("    B. 有意调参 -> 重跑实验 + 更新文档，再 --update")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
