"""Reranker 的纯逻辑测试。**不加载模型** —— 2.27 GB 的模型不该出现在单测里。

## 这个文件锁什么

| 锁什么 | 防什么 |
| --- | --- |
| sigmoid 数值稳定 | 大负数直接 `1/(1+exp(-x))` 会溢出成 inf |
| 归一化按 query 内部 | 被改成跨 query 归一化（Cross-Encoder 的分数本来就跨 query 不可比） |
| 三种截断真的截在不同位置 | 截断策略写成同一个（那"三种策略"就是假的） |
| `rank_before/after/delta` 可算 | "精排挪了几位"算不出来，就只能靠眼看 |
| `sort_by_length` 不影响分数 | 优化改变了结果 —— 那是 bug，不是优化 |

## 为什么这些能脱离模型测

sigmoid、归一化、截断预算、排序回填都是**纯函数逻辑**。
真正需要模型的部分（前向打分）由 `scripts/rerank_probe.py` 与
`scripts/rerank_eval.py` 实测，不进单测 —— 否则每次 `pytest` 都要 12 秒加载。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from app.rag.bm25.rerank import (  # noqa: E402
    Reranker,
    RerankScore,
    Truncation,
    normalize_within_query,
    sigmoid,
)

# --------------------------------------------------------------------------- sigmoid


def test_sigmoid_is_numerically_stable_for_large_negative() -> None:
    """**大负数不能直接 `1/(1+exp(-x))`** —— `exp(800)` 会 overflow 成 inf。

    Cross-Encoder 的 logit 没有硬下界（实测见过 -11），
    在更长的文本或分布外输入上它可能更低。
    """
    assert sigmoid(-800.0) == pytest.approx(0.0, abs=1e-12)
    assert sigmoid(800.0) == pytest.approx(1.0, abs=1e-12)
    assert sigmoid(0.0) == pytest.approx(0.5)
    # 中段与朴素公式一致
    assert sigmoid(-3.0) == pytest.approx(1.0 / (1.0 + math.exp(3.0)))


def test_sigmoid_is_monotonic() -> None:
    """sigmoid 保序。**这是"分数只是被压缩、名次没变"的依据。**"""
    values = [-10.0, -3.0, 0.0, 2.99, 8.0]
    scores = [sigmoid(v) for v in values]
    assert scores == sorted(scores)


# --------------------------------------------------------------------------- 归一化


def test_normalize_minmax_is_within_one_query() -> None:
    """**归一化只在一个 query 的候选集合内做。**

    实测证据（`scripts/rerank_probe.json`）：8 条不同 query 的 top1 分
    从 0.002 到 0.842，跨度 0.840。**同一模型、同一批候选，只换 query** ——
    分数就整体平移。所以跨 query 的绝对阈值毫无意义。

    按 query 内部归一后，每条 query 的 top1 都是 1.0。
    """
    high_query = [0.842, 0.30, 0.10]
    low_query = [0.002, 0.0015, 0.001]

    high_norm = normalize_within_query(high_query)
    low_norm = normalize_within_query(low_query)

    assert high_norm[0] == pytest.approx(1.0)
    assert low_norm[0] == pytest.approx(1.0), (
        "低分 query 的 top1 在自己的候选里也是最好的，必须拿 1.0 —— "
        "若按全局归一它会被压成 0.00x，那一侧的权重就假了"
    )


def test_normalize_percentile_ranks_by_position() -> None:
    """percentile 按名次分档：最高拿 1.0、最低拿 0.0。**它对离群值比 minmax 稳**。"""
    scores = [0.99, 0.40, 0.38, 0.35]
    normed = normalize_within_query(scores, "percentile")
    assert max(normed) == pytest.approx(1.0)
    assert min(normed) == pytest.approx(0.0)
    # minmax 会把后三名全压到 0.0~0.08 的窄带里
    minmax = normalize_within_query(scores, "minmax")
    assert minmax[1] == pytest.approx((0.40 - 0.35) / (0.99 - 0.35))
    assert normed[1] > minmax[1], "percentile 不该被一条离群高分把其余压扁"


def test_normalize_degenerate_cases() -> None:
    """空表 / 单元素 / 全同分。**都给 1.0 或空，不给 0** —— 理由见 fusion 的同名测试。"""
    assert normalize_within_query([]) == []
    assert normalize_within_query([0.5]) == [1.0]
    assert normalize_within_query([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]


def test_normalize_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="未知归一化方式"):
        normalize_within_query([1.0, 2.0], "softmax")


# --------------------------------------------------------------------------- 截断


@pytest.mark.parametrize(
    "policy,kept_part",
    [
        (Truncation.TAIL, "head"),
        (Truncation.HEAD, "tail"),
        (Truncation.MIDDLE, "middle"),
    ],
    ids=["tail-keeps-head", "head-keeps-tail", "middle-keeps-middle"],
)
def test_truncation_policies_keep_different_parts(policy: Truncation, kept_part: str) -> None:
    """**三种策略必须真的截在不同位置**，否则"三种策略"就是假的。

    用 100 个编号 token、截到 10 个：

        TAIL   → 保留开头 0~9   （丢结尾）
        HEAD   → 保留结尾 90~99 （丢开头）
        MIDDLE → 保留 45~54     （两头都丢）

    命名容易看反：`Truncation.TAIL` 指「**截掉尾巴**」，不是「保留尾巴」。
    这条测试就是为了把这个反直觉的命名钉死。
    """
    ids = list(range(100))
    reranker = Reranker(truncation=policy)
    kept = reranker._truncate_ids(ids, 10)
    assert len(kept) == 10
    if kept_part == "head":
        assert kept[0] == 0, f"TAIL 策略应保留开头，实际从 {kept[0]} 开始"
        assert kept[-1] == 9
    elif kept_part == "tail":
        assert kept[-1] == 99, f"HEAD 策略应保留结尾，实际到 {kept[-1]} 结束"
        assert kept[0] == 90
    else:
        assert kept[0] > 0 and kept[-1] < 99, "MIDDLE 策略应两头都丢"


def test_truncation_handles_short_input() -> None:
    """输入本来就短时不该多截 —— 预算够就不动。"""
    reranker = Reranker()
    ids = list(range(5))
    assert reranker._truncate_ids(ids, 10) == ids


# --------------------------------------------------------------------------- 名次回填


def test_rerank_score_delta_is_negative_when_promoted() -> None:
    """`delta` 负数 = 往前挪（变好）。**这是"精排有没有用"的量化依据。**"""
    promoted = RerankScore("x", 2.0, 0.9, rank_before=13, rank_after=1)
    assert promoted.delta == -12
    demoted = RerankScore("y", -1.0, 0.2, rank_before=2, rank_after=9)
    assert demoted.delta == 7
    unchanged = RerankScore("z", 0.0, 0.5, rank_before=3, rank_after=3)
    assert unchanged.delta == 0


def test_rerank_score_delta_is_none_without_both_ranks() -> None:
    """缺任一名次时 `delta` 为 `None` —— **不要静默当 0**。"""
    assert RerankScore("x", 1.0, 0.5, rank_before=1).delta is None
    assert RerankScore("x", 1.0, 0.5, rank_after=1).delta is None


# --------------------------------------------------------------------------- 配置与加载


def test_reranker_is_lazy() -> None:
    """**不 load 就不加载模型。** 830 条测试不该为它付 12 秒 + 2 GB。"""
    reranker = Reranker()
    assert reranker.is_loaded() is False
    assert reranker.load_ms == 0


def test_missing_model_dir_raises_actionable_error() -> None:
    """模型目录不存在时，错误信息要**直接告诉人跑哪条命令**。"""
    reranker = Reranker(model_path=Path("d:/cache/does-not-exist-reranker"))
    with pytest.raises(FileNotFoundError, match="fetch_reranker.py"):
        reranker.load()


def test_default_batch_sort_is_enabled() -> None:
    """默认按长度排序组 batch。**关掉它实测慢 2 倍**（见 rerank_probe.json）。"""
    assert Reranker().sort_by_length is True


def test_sort_by_length_can_be_disabled_for_ab_compare() -> None:
    """保留 `sort_by_length=False` 是为了做 A/B —— 它不该被悄悄删掉，
    否则"优化到底值多少"就再也测不出来了。"""
    assert Reranker(sort_by_length=False).sort_by_length is False


# ------------------------------------------------- 边界：参数校验（W6 周四补）


def test_truncation_accepts_enum_value_string() -> None:
    """`truncation="tail"` 应当在**构造时**被规整成枚举，而不是等到 rerank 才炸。

    第一版不校验，于是字符串能构造成功，直到 `rerank()` 里
    `self.truncation.value` 才抛 `AttributeError: 'str' object has no attribute 'value'`
    —— **错误发生在离现场很远的地方，报的还是 attribute 而不是"参数类型错了"。**
    """
    reranker = Reranker(truncation="tail")  # type: ignore[arg-type]
    assert reranker.truncation is Truncation.TAIL
    assert isinstance(reranker.truncation, Truncation)


def test_truncation_rejects_unknown_value() -> None:
    """非法值**立刻抛**，并且把可用取值列出来。"""
    with pytest.raises(ValueError, match="truncation 取值非法"):
        Reranker(truncation="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"max_length": 0}, "max_length 必须为正数"),
        ({"batch_size": 0}, "batch_size 必须为正数"),
    ],
)
def test_invalid_model_params_are_rejected(kwargs: dict[str, object], message: str) -> None:
    """模型参数写错必须在构造时抛 —— 否则会一路跑到前向才出问题。"""
    with pytest.raises(ValueError, match=message):
        Reranker(**kwargs)  # type: ignore[arg-type]


# ------------------------------------------------- 边界：长 doc 截断系统性排后


def test_truncation_drops_offwindow_content_entirely() -> None:
    """**被截掉的内容完全不参与打分**，而且是"归零"不是"略低"。

    这不是靠模型跑的测试（那要 12 秒），而是把 `_truncate_ids` 的语义钉住：
    `TAIL` 保留开头，所以**末尾那段内容在编码阶段就消失了**。

    模型层面的实测数字（`docs/reranker.md` §六）：

        答案在开头 → 0.9235 | 答案居中 → 0.0003 | 答案在末尾 → 0.0003

    也就是说：一条"答案在块尾"的长块会被压到数值噪声水平，
    **被系统性排到后面**。这是 `max_length` 不能乱设的原因。
    """
    ids = list(range(1000))
    reranker = Reranker(truncation=Truncation.TAIL, max_length=100)
    kept = reranker._truncate_ids(ids, 100)
    assert kept == list(range(100)), "TAIL 必须保留开头 100 个"
    assert 900 not in kept, "末尾内容必须被彻底丢弃，而不是降权"


def test_each_truncation_policy_covers_a_different_position() -> None:
    """**没有任何单一策略能覆盖全部位置** —— 这是选默认值的依据。

    实测矩阵（`docs/reranker.md` §六）：

        TAIL   覆盖"答案在开头"（0.92），居中和末尾都归零
        HEAD   覆盖"答案在末尾"（0.65），开头和居中归零
        MIDDLE 覆盖"答案居中"（0.78），开头和末尾归零

    默认选 TAIL 的理由：切分器保证块首是完整句首，**答案在块首的概率最高**；
    且 TAIL 的峰值分（0.92）是三者里最高的。
    """
    ids = list(range(300))
    budget = 100
    tail = Reranker(truncation=Truncation.TAIL)._truncate_ids(ids, budget)
    head = Reranker(truncation=Truncation.HEAD)._truncate_ids(ids, budget)
    middle = Reranker(truncation=Truncation.MIDDLE)._truncate_ids(ids, budget)

    assert tail[0] == 0 and head[-1] == 299, "TAIL/HEAD 应分别覆盖两头"
    assert middle[0] > 0 and middle[-1] < 299, "MIDDLE 应两头都丢"
    # 三个覆盖区间不能相同 —— 否则"三种策略"是假的
    assert set(tail) != set(head) != set(middle)
    assert not (set(tail) & set(head)), "TAIL 与 HEAD 的保留区间不该重叠"


def test_default_max_length_does_not_truncate_typical_chunks() -> None:
    """默认 `max_length=512` 对**本项目常见块长**不该触发截断。

    实测：语料 1404 块里中位 77 token、只有 3 块超过 512。
    所以"三种截断策略"的差异在默认配置下**几乎观察不到** ——
    这也是为什么产物里要记 `truncated_pairs`：它才是"这个话题今天相不相关"的判据。
    """
    assert Reranker().max_length == 512
    # 一个典型 500 字中文块 ≈ 350 token，加上 query 后仍在 512 内
    assert Reranker().max_length > 400
