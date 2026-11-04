"""**链路参数指纹**测试：参数变了必须被显式确认。

## 背景

`RagConfig` 只覆盖 5 个字段，而决定消融结论的 `DEFAULT_RRF_K`
（60）、`DEFAULT_CANDIDATES`（20）、`DEFAULT_MAX_LENGTH`（512）
等都是**模块常量** —— 改一行不会让任何测试失败。

于是可能出现：有人把 `DEFAULT_RRF_K` 从 60 改成 40，
`docs/ablation-four.md` 里"+15pp"不再成立，README 里
"混合层不划算"的结论可能要翻，**但测试全绿**。

## 这个测试做什么

比对**入库的指纹**与**当前实际值**。不一致就红，并给出
"哪个参数从多少变成多少"。

## 它**不**做什么（重要）

它不阻止改参数 —— 参数本来就该可调（做实验必须能改）。
它也不保证"改了参数后结论仍然正确" —— 那要靠重跑实验。

它只保证**变化是可见的**，而不是静默发生。

改了参数并重跑完实验后，更新指纹即可：

```bash
python -X utf8 scripts/param_fingerprint.py --update
```
"""

from __future__ import annotations

import pytest
from eval.param_fingerprint import (
    FINGERPRINT_PATH,
    TRACKED_PARAMS,
    collect,
    compute,
    diff,
    load_baseline,
)


def test_fingerprint_file_is_committed() -> None:
    """指纹文件必须**入库**（不是 gitignore）。

    它的全部价值在于"跨提交对比" —— 不入库就失去了意义。
    """
    assert FINGERPRINT_PATH.exists(), (
        f"缺参数指纹文件 {FINGERPRINT_PATH.relative_to(FINGERPRINT_PATH.parents[2])}。\n"
        "首次建立：python -X utf8 scripts/param_fingerprint.py --update"
    )


def test_tracked_params_all_resolve() -> None:
    """每个被跟踪的参数都必须能真的读到值。

    写错模块名或属性名时，`collect()` 会填 `<MISSING>` ——
    那样指纹永远"稳定"，但跟踪的东西根本不存在。
    这种"看起来在工作"的机制比没有更糟。
    """
    values = collect()
    missing = [key for key, value in values.items() if value == "<MISSING>"]
    assert not missing, "以下被跟踪的参数读不到（模块名或属性名写错了）：\n  " + "\n  ".join(
        missing
    )


def test_link_params_unchanged_since_baseline() -> None:
    """**核心断言**：链路参数与入库指纹一致。

    ## 失败时怎么办（两种情况，处理完全不同）

    **情况 A：误改**（最常见）
    把参数改回去了 —— `git diff` 看看改哪了，恢复即可。

    **情况 B：有意调参**
    那你必须：
    1. 重跑受影响的实验（`make ablation-four` / `make latency-profile`）
    2. 更新文档里的数字与结论
    3. `python -X utf8 scripts/param_fingerprint.py --update`
    4. 提交信息里写明"为什么改 + 重跑了什么"

    只做第 3 步（更新指纹）而跳过 1、2 是**最坏的情况** ——
    等于把"文档与代码脱节"这件事**盖章确认**了。
    """
    baseline = load_baseline()
    if baseline is None:
        pytest.skip("还没有基线指纹（首次建立前跳过）")

    changes = diff(baseline, compute())
    assert not changes, (
        "链路参数与入库指纹不一致：\n  "
        + "\n  ".join(changes)
        + "\n\n这些参数影响已发布的评测结论（见 eval/param_fingerprint.py 的表）。"
        "\n如果是误改，请恢复；如果是有意调参，必须重跑实验并更新文档后再更新指纹。"
    )


def test_baseline_has_plaintext_values() -> None:
    """指纹必须保留**明文值**，不能只有哈希。

    光有哈希时，diff 只能说"变了"，说不出"从 60 变成 40"。
    排查时那个信息决定一切 —— 它直接指向"哪个结论要重算"。
    """
    baseline = load_baseline()
    if baseline is None:
        pytest.skip("还没有基线指纹")

    values = baseline.get("values")
    assert isinstance(values, dict) and values, "指纹里没有明文 values"
    assert baseline.get("digest"), "指纹里没有 digest"


def test_diff_reports_before_and_after() -> None:
    """`diff()` 的输出要同时包含旧值和新值（可读性自测）。

    这条不测业务，测**报错信息的质量** ——
    一个只说"变了"的报错会让人去翻半天代码。
    """
    baseline = {"values": {"a.b": 60, "c.d": 0.75}}
    current = {"values": {"a.b": 40, "c.d": 0.75}}

    changes = diff(baseline, current)
    assert len(changes) == 1, f"应当只报一处变化，实际：{changes}"
    assert "60" in changes[0] and "40" in changes[0], (
        f"报错里要能看到旧值和新值，实际：{changes[0]}"
    )


def test_diff_handles_new_and_removed_params() -> None:
    """新增/移除参数也要能报（不能静默）。

    跟踪列表变动时，如果 `diff` 静默忽略，
    那么"新加了一个常量"这件事不会被任何人注意到 ——
    而新加的常量可能同样影响结论。
    """
    baseline = {"values": {"a.b": 1}}
    current = {"values": {"a.b": 1, "x.y": 2}}

    changes = diff(baseline, current)
    assert changes, "新增参数应当被报出来"
    assert any("新增" in c for c in changes), f"应当标明是新增：{changes}"


def test_tracked_params_have_reasons() -> None:
    """每个被跟踪的参数都要有**可读的影响说明**。

    没有说明的条目，下一个人不知道它为什么被跟踪，
    就会把它当成噪音删掉。
    """
    for module_name, attr, reason in TRACKED_PARAMS:
        assert reason and len(reason) >= 8, f"{module_name}.{attr} 的说明太短，写清它影响哪个结论"
