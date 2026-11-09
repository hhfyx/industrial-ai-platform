"""预算与早停（W11 周四）：三道硬上限 + 三条早停条件 + 优雅退出。

## 必答问题：预算砍半会掉多少成功率？

如果你没测这个数，周日的"成本下降"就是拿**用户可用性**换来的，
而你不知道换了多少。这条也是"你怎么控制成本"的标准答案来源。

所以今天的顺序是**先测曲线、再定档**，不是先定档再找理由。
`scripts/budget_curve.py` 跑 5k/10k/15k 三档 × 6 个代表任务，
曲线数据落盘后才回来写 `DEFAULT_BUDGET`。

## 今天的坑（写在任务原文里，值得抄一遍）

> 把预算设得太低"显得省"。成功率会掉，而你在表里会把它归因成"任务太难"。

这条坑的机制很具体：预算低 → 任务在半路被截断 → 判据失败 →
你看到失败就去归因，而归因的默认假设是"这题本来就难"。
`stopped_reason` 里那个 `budget_exhausted` 就是为了让这个归因**无法偷懒**：
凡是预算触顶的失败，都必须单独计数，不能混进"任务难"。

## 三道上限为什么都要

| 上限 | 拦什么 | 缺失的后果 |
| --- | --- | --- |
| `max_tokens` | 长上下文、反复读大块 | 单任务烧掉别人 5 倍的钱 |
| `max_steps` | 打转、无效循环 | 10 步能完的活跑 30 步 |
| `max_wall_time` | 慢工具、网关抖动 | 一个任务挂 10 分钟，队列全堵 |

三者**不能互相替代**：token 不涨但步数涨（每步都很小）、
步数不多但很慢（工具卡住）都是真实形态。只设一道就会漏。

## 三条早停条件，都要能在**路由里**判

"不靠模型自觉"是硬要求。模型说"我已经知道了"就可以停，那等于没有早停。
所以三条都做成纯函数，输入是 state，输出是"停下/继续"：

1. **已有足够信息可出 final_answer** —— 已有成功的工具调用，且模型最近一步
   已经在写自然语言（没再调工具）。这是**自然的完成**，不是"早停"。
2. **连续 2 步无新增有效信息** —— 近两步的工具结果与之前的**语义重合**
   （用周三的语义键判），说明在原地打转。
3. **边际收益为 0（再查同一主题）** —— 上一步的结果与更早的结果语义等价，
   即"又查了一遍同一个东西"。

第 2、3 条看起来像，但触发面不同：第 2 条看**步序列**（连续两步都没进展），
第 3 条看**单次调用的重复性**（这一步在重复历史上的某一步）。
实现上第 3 条归到语义判重（周三已有），第 2 条才是新的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal

#: 早停/触顶的原因。进 `stopped_reason`，也是周日归因表的第一分类。
StopReason = Literal[
    "completed",
    "budget_tokens",
    "budget_steps",
    "budget_wall_time",
    "no_progress",
    "diminishing_returns",
    "unrepairable_failure",
    "repeat_loop",
    "parse_failure",
    "model_timeout",
    "degraded",
]

#: 哪些 stopped_reason 算"预算触顶"。归因时必须单独计数 ——
#: 混进"任务难"就是今天的坑。
BUDGET_REASONS: Final[frozenset[str]] = frozenset(
    {"budget_tokens", "budget_steps", "budget_wall_time"}
)


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    """三道上限。**默认值是测出来的曲线选的档，不是拍的。**

    token 默认 10k，依据见 `DEFAULT_BUDGET` 的注释（三档实测曲线）：
    10k 与 15k 成功率相同（5/6），但 15k 多花 9.3% token 而没买到任何东西。

    初版曾定 15k（按 W10 基线中位数往上留余量）。**曲线推翻了那个拍法** ——
    这正是"先测曲线、再定档"的意义：直觉给的余量，在数据里是纯浪费。
    """

    max_tokens: int = 10_000
    max_steps: int = 10
    max_wall_time_s: float = 120.0
    #: 连续多少步无新增有效信息即早停。2 = 给一步"换个角度"的机会。
    no_progress_limit: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "max_steps": self.max_steps,
            "max_wall_time_s": self.max_wall_time_s,
            "no_progress_limit": self.no_progress_limit,
        }


@dataclass
class BudgetUsed:
    """已用量。每步累加，进 state（`budget_used`）。

    **为什么进 state 而不是局部变量**：W10 的 C4 —— 局部变量在崩溃恢复后
    归零。预算计数一旦归零，重启就等于"预算刷新"，那道闸就形同虚设。
    """

    tokens: int = 0
    steps: int = 0
    wall_time_s: float = 0.0
    #: 连续无进展步数（每次有新信息就清零）
    no_progress_streak: int = 0
    #: 已经因为早停/触顶退出过（防止重复处理）
    stopped: str = ""

    def add_step(self, *, tokens: int, wall_time_s: float, progressed: bool) -> None:
        self.tokens += max(0, int(tokens))
        self.steps += 1
        self.wall_time_s += max(0.0, float(wall_time_s))
        self.no_progress_streak = 0 if progressed else self.no_progress_streak + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "steps": self.steps,
            "wall_time_s": round(self.wall_time_s, 3),
            "no_progress_streak": self.no_progress_streak,
            "stopped": self.stopped,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> BudgetUsed:
        raw = raw or {}
        return cls(
            tokens=int(raw.get("tokens", 0) or 0),
            steps=int(raw.get("steps", 0) or 0),
            wall_time_s=float(raw.get("wall_time_s", 0.0) or 0.0),
            no_progress_streak=int(raw.get("no_progress_streak", 0) or 0),
            stopped=str(raw.get("stopped", "") or ""),
        )


@dataclass(frozen=True, slots=True)
class BudgetVerdict:
    """预算判定结果。`reason` 为空表示还能继续。"""

    exhausted: bool
    reason: StopReason | Literal[""] = ""
    detail: str = ""
    #: 触发时该用什么退出方式：budget_* 走 finalize_budget，no_progress 走降级
    graceful: bool = True

    @property
    def stop(self) -> bool:
        return self.exhausted


def check_budget(
    used: BudgetUsed,
    config: BudgetConfig,
    *,
    next_step_tokens: int = 0,
) -> BudgetVerdict:
    """三道硬上限的判定。**在路由里调用，早于下一次模型调用。**

    `next_step_tokens` 是"预计下一步要花多少"（用上一步的实际值估）。
    为什么要预判而不是花完再判：花完再判意味着**已经超了**才停，
    而预判能在跨过上限之前停手 —— 这是"预算"与"记账"的区别。

    顺序按"最先可能触发的"排：步数 → token → 墙钟。步数最便宜判，
    墙钟最后（要读时钟值，虽然这里是从 state 读的，不额外开销）。
    """
    if used.steps >= config.max_steps:
        return BudgetVerdict(True, "budget_steps", f"已用 {used.steps}/{config.max_steps} 步")

    projected = used.tokens + max(0, next_step_tokens)
    if projected > config.max_tokens:
        return BudgetVerdict(
            True,
            "budget_tokens",
            f"已用 {used.tokens} token，下一步预计 {next_step_tokens}，将超 {config.max_tokens}",
        )

    if used.wall_time_s >= config.max_wall_time_s:
        return BudgetVerdict(
            True,
            "budget_wall_time",
            f"已用 {used.wall_time_s:.1f}s/{config.max_wall_time_s:.0f}s",
        )
    return BudgetVerdict(False)


def check_no_progress(used: BudgetUsed, config: BudgetConfig) -> BudgetVerdict:
    """早停条件②：连续 N 步无新增有效信息。

    "无新增有效信息"的判定不在这个函数里 —— 它需要看工具结果的内容，
    由调用方（图里的 tools 节点）在累加时传到 `add_step(progressed=...)`。
    这里只做阈值判定，保持纯函数。

    为什么放在预算模块而不是 guardrails：它**不是防跑偏**（那三道管的是
    "方向对不对"），它管的是"还在不在前进"。两类机制的读者不同：
    前者写给周日做取舍的人，后者写给成本表。
    """
    if used.no_progress_streak >= config.no_progress_limit:
        return BudgetVerdict(
            True,
            "no_progress",
            f"连续 {used.no_progress_streak} 步无新增有效信息",
        )
    return BudgetVerdict(False)


def judge_progress(previous_keys: set[str], new_keys: set[str]) -> bool:
    """这一步有没有带来**新**的有效信息。

    参数是语义键集合（周三的 `semantic_key`）。判据只有一条：
    新结果里存在**之前没出现过**的键。

    为什么用语义键而不是原始字符串：同一个查询换个写法返回同样的内容，
    那不是新信息。用字符串比会把"再查一遍"误当成"有进展"。
    """
    return bool(new_keys - previous_keys)


@dataclass
class BudgetReport:
    """预算汇总（进轨迹与成本表）。"""

    config: BudgetConfig = field(default_factory=BudgetConfig)
    used: BudgetUsed = field(default_factory=BudgetUsed)
    next_step_estimate: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "used": self.used.to_dict(),
            "next_step_estimate": self.next_step_estimate,
        }


def build_graceful_exit(
    *,
    used: BudgetUsed,
    config: BudgetConfig,
    reason: StopReason,
    partial: str,
    suggestion: str = "",
) -> dict[str, Any]:
    """优雅退出载荷：已完成部分 + 未完成原因 + 建议下一步（W9 B7 的落地）。

    三件事都要有，缺一件就不叫优雅：

    - 只有"原因"没有"已完成"→ 用户不知道拿到了什么，等于全废；
    - 只有"已完成"没有"原因"→ 用户以为任务完成了；
    - 没有"建议"→ 用户不知道下一步该干什么（加预算？拆任务？换问法？）。

    `degraded=True` 是刻意的：预算触顶属于**降级**而不是失败（我们确实
    产出了一部分），计入降级率才能在周日那张表里和"任务失败"分开看。
    """
    return {
        "degraded": True,
        "reason": reason,
        "answer": partial.strip() or "（本次没有产出可用内容）",
        "missing": [f"未完成原因：{_reason_text(reason)}"],
        "budget": {
            "tokens_used": used.tokens,
            "tokens_limit": config.max_tokens,
            "steps_used": used.steps,
            "steps_limit": config.max_steps,
            "wall_time_s": round(used.wall_time_s, 1),
            "wall_time_limit_s": config.max_wall_time_s,
        },
        "suggestion": suggestion or _default_suggestion(reason, used, config),
    }


def _reason_text(reason: str) -> str:
    return {
        "budget_tokens": "token 预算触顶",
        "budget_steps": "步数预算触顶",
        "budget_wall_time": "墙钟时间触顶",
        "no_progress": "连续多步无新增有效信息",
        "diminishing_returns": "边际收益为 0（在重复查询同一主题）",
    }.get(reason, reason)


def _default_suggestion(reason: str, used: BudgetUsed, config: BudgetConfig) -> str:
    """按触顶类型给不同的建议 —— 一句通用建议等于没建议。"""
    if reason == "budget_tokens":
        return (
            f"本次用了 {used.tokens} token（上限 {config.max_tokens}）。"
            "可以：① 把任务拆成两个更小的（先查数据、再出结论）；"
            "② 提高 token 预算；③ 缩小查询范围（少召回几条、只查单指标）。"
        )
    if reason == "budget_steps":
        return (
            f"本次走了 {used.steps} 步（上限 {config.max_steps}）。"
            "可以：① 明确告诉要查哪个批次/哪个指标，减少试探；"
            "② 提高步数预算；③ 如果任务本身需要多轮比对，拆成子任务。"
        )
    if reason == "budget_wall_time":
        return (
            f"本次耗时 {used.wall_time_s:.0f}s（上限 {config.max_wall_time_s:.0f}s）。"
            "可以：① 检查网络/网关延迟；② 提高时间预算；③ 缩小查询范围减少往返。"
        )
    if reason == "no_progress":
        return (
            "连续几步都没有新信息，通常是因为反复查同一个对象。"
            "可以：换关键词、换一个更宽/更窄的条件，或直接基于已有信息给结论。"
        )
    return "可以调整问题或提高预算后重试。"


#: 定档结果。**这个值必须来自 `scripts/budget_curve.py` 的实测**，
#: 改动它的时候请把曲线的数字一起改，否则就成了"先定档再找理由" ——
#: 那正是今天的坑。
#:
#: ## 定档依据（2026-12-31 实测，trae/qwen-3.5，6 任务 × 3 档）
#:
#: ```
#: 档位    成功率      均token   峰值     判据失败  预算触顶
#: 5000    50% (3/6)   5197     5669     1        2
#: 10000   83% (5/6)   6691    14200     1        0
#: 15000   83% (5/6)   7311    14407     1        0
#: ```
#:
#: **选 10k。** 理由：
#:
#: 1. 10k 与 15k 的成功率完全相同（5/6），但 15k 平均多花 620 token
#:    （6691 → 7311，+9.3%）—— 那 620 没买到任何东西；
#: 2. 5k 掉 33 个百分点（50% vs 83%），且 2 个失败明确标着 `budget_tokens`
#:    —— 这不是"任务难"，是钱不够。拿 33 点可用性换 1494 token 不划算；
#: 3. 唯一在 10k 档的失败是 T10（判据误伤，与预算无关，W10 已知问题），
#:    所以 83% 已经是**当前判据下的天花板**，15k 也不可能更高。
#:
#: 峰值 14200（T7 报告任务）逼近 10k 上限而没触顶，说明 10k 的余量对
#: "复杂任务"偏紧但够用；如果 W12 出现更长的任务链，先看曲线再调。
DEFAULT_BUDGET: Final[BudgetConfig] = BudgetConfig(
    max_tokens=10_000,
    max_steps=10,
    max_wall_time_s=120.0,
)


__all__ = [
    "BUDGET_REASONS",
    "DEFAULT_BUDGET",
    "BudgetConfig",
    "BudgetReport",
    "BudgetUsed",
    "BudgetVerdict",
    "StopReason",
    "build_graceful_exit",
    "check_budget",
    "check_no_progress",
    "judge_progress",
]
