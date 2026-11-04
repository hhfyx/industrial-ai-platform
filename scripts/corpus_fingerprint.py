"""查看/更新语料指纹。

## 用法

```bash
python -X utf8 scripts/corpus_fingerprint.py           # 看差异
python -X utf8 scripts/corpus_fingerprint.py --update  # 确认影响后更新基线
```

## 什么时候该用 `--update`

**先回答一个问题：这次改动会影响评测吗？**

| 改动 | 处理 |
| --- | --- |
| 措辞/排版 | 重跑评测确认数字没变 -> `--update` |
| 改了某道题的锚点原文 | **该题要重新出**，不是 update 就完事 |
| 新增文档进语料 | 重跑（新块可能挤占 Top-K） |
| 删了文档 | 引用它的题失效，必须处理 |

**直接 `--update` 是最坏的做法** —— 等于把"文档与代码脱节"盖章确认。
详见 `eval/corpus_fingerprint.py`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.corpus_fingerprint import (  # noqa: E402
    FINGERPRINT_PATH,
    compute,
    diff,
    load_baseline,
    write_baseline,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="语料指纹")
    parser.add_argument("--update", action="store_true", help="把当前语料写为基线")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--list", action="store_true", help="列出所有语料文档")
    args = parser.parse_args(argv)

    current = compute()

    if args.list:
        print(f"语料文档 {current['documents']} 篇：")
        for name in sorted(current.get("files") or {}):
            print(f"  {name}")
        return 0

    if args.update:
        write_baseline(current)
        if args.json:
            print(
                json.dumps(
                    {"digest": current["digest"], "documents": current["documents"]},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"✓ 已更新语料指纹（{current['documents']} 篇）：{FINGERPRINT_PATH}")
            print()
            print("⚠ 确认你已经判断过影响：改的是措辞还是锚点原文？需要重跑评测吗？")
        return 0

    baseline = load_baseline()
    changes = diff(baseline, current)

    if args.json:
        print(
            json.dumps(
                {
                    "digest": current["digest"],
                    "documents": current["documents"],
                    "changes": changes,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print("=" * 70)
    print(f"语料指纹：{current['documents']} 篇")
    print("=" * 70)
    print(f"  digest: {current['digest'][:32]}…")
    print()

    if baseline is None:
        print("  还没有基线。首次建立：--update")
        return 0

    if not any(changes.values()):
        print("✅ 与基线一致")
        return 0

    for kind, label in (
        ("changed", "内容被改写"),
        ("added", "新增进语料"),
        ("removed", "从语料消失"),
    ):
        items = changes.get(kind) or []
        if items:
            print(f"⚠ {label}（{len(items)} 篇）：")
            for name in items:
                print(f"    - {name}")
            print()

    print("  历史评测数字绑定在旧语料上。处理方式：")
    print("    措辞改动   -> 重跑评测确认数字没变，再 --update")
    print("    改了锚点   -> 该题要重新出（不是 update 就完事）")
    print("    新增文档   -> 重跑（新块可能挤占 Top-K）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
