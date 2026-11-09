"""恢复策略矩阵（W11 周二 · 重试、降级与故障注入）。

## 一句话

周一让代码**知道**"能不能重试"；今天让代码**执行**"怎么重试、什么时候停"。

`error_contract` 回答"这一类错能不能再来一次"，本模块回答"再来几次、
谁来再来、再不行怎么办"。两者是分工，不是重复：

| 模块 | 回答的问题 | 谁读 |
| --- | --- | --- |
| `error_contract` | 这一类可恢复吗？retryable / retry_after | 模型 + 代码 |
| `recovery`（本模块） | 谁来恢复？几次？用尽后怎么收场？ | **只有代码** |

## 策略矩阵（先写表，再写节点）

| error_type | 动作 | 谁执行 | 上限 | 退避 | 用尽后 |
| --- | --- | --- | --- | --- | --- |
| `invalid_json` | 改参重试 | **模型** | 1 | — | 交回模型 |
| `invalid_arg` | 改参重试 | **模型** | 1 | — | 交回模型 |
| `unknown_tool` | 换工具 | **模型** | 1 | — | 交回模型 |
| `timeout` | 退避重试 | **代码** | 1 | 带 jitter | 降级 |
| `internal` | 退避重试 | **代码** | 1 | 带 jitter | 降级 |
| `not_found` | 换关键词/换对象 | **模型** | 1 | — | 交回模型 |
| `empty_result` | 换关键词/换对象 | **模型** | 1 | — | 交回模型 |
| `not_permitted` | 跳过并告知 | 代码 | 0 | — | 降级 |
| `conflict` | 跳过并报告 | 代码 | 0 | — | 降级 |

## 为什么"谁执行"这一列才是今天的关键

参数错必须**模型**改 —— 代码不知道正确的参数是什么，替它改就是编造。
超时/内部异常必须**代码**重试 —— 模型对"刚发生的超时"没有元信息，
让它决定就是让它猜（周一必答问题的延续）；而且代码重试可以退避 + 抖动，
模型重试只能靠"换个说法再调一次"，那是撞运气。

**矩阵里每一格的上限都是必需的**：没有上限的重试就是"重试到成功"，
而 W9 的 B7 已经判过 —— 那是"会自己续费的故障"。上限是这条判词的执行体。

## 降级与"重试到成功"的根本区别（今日必答问题）

- **重试到成功**：相信"再试一次就会好"，没有停止条件 → 赌。
- **降级**：承认边界，把**已知的部分**和**缺的部分**分别交出去 →
  用户能判断这份答案能信到什么程度。

所以降级的答案形状必须是三条：`answer`（部分结论）、`missing`（缺什么）、
`reason`（为什么缺）。少任何一条，降级就退化成"随便给个通用回答"——
用户看到一个看起来正常但内容错误的答案，**比明确说"查不到"危险得多**。

`answer` 只能由**已有的事实**拼出来（成功工具调用的结果），
绝不再进一次 LLM 去"润色"或"补全"——那一步就是把缺口填上编造的内容。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Final, Literal

from app.core.llm.error_contract import normalize_error_type
from app.core.llm.retry import compute_delay

#: 恢复动作。四个值对应矩阵的四类"谁执行/去哪"。
RecoveryAction = Literal[
    "auto_retry",  # 代码退避重试同一个调用
    "model_fix",  # 交回模型：改参数 / 换关键词 / 换工具
    "skip_and_tell",  # 不重试：跳过这件事并如实告知（权限/重复）
    "degrade",  # 直接降级（没有可挽回的余地）
]


@dataclass(frozen=True, slots=True)
class RecoverySpec:
    """矩阵的一格：动作 + 谁执行 + 上限 + 退避 + 用尽后。

    `degrade_when_unresolved` 与 `action` 是**两件事**，这里刻意分开：

    - `action` 是**这一刻该做什么**（跳过并告知 / 交回模型 / 代码重试）；
    - `degrade_when_unresolved` 是**如果这件事最终没解决，收场算不算降级**。

    混成一个字段会逼出错误行为：权限错该"跳过后继续用别的工具试"，
    而不是立刻放弃整个任务去降级。但它又确实该在"最终没解决"时降级，
    因为缺的那块数据是任务的一部分。两个维度，两个字段。
    """

    action: RecoveryAction
    limit: int  # 0 表示一次都不该再试
    by_code: bool  # True = 代码执行；False = 交回模型
    jitter: bool = False
    degrade_when_unresolved: bool = False


#: **矩阵本体**。改这里就是改策略；节点里不要再写第二份 if/else。
RECOVERY_MATRIX: Final[dict[str, RecoverySpec]] = {
    "invalid_json": RecoverySpec("model_fix", limit=1, by_code=False),
    "invalid_arg": RecoverySpec("model_fix", limit=1, by_code=False),
    "unknown_tool": RecoverySpec("model_fix", limit=1, by_code=False),
    "not_found": RecoverySpec("model_fix", limit=1, by_code=False),
    "empty_result": RecoverySpec("model_fix", limit=1, by_code=False),
    "timeout": RecoverySpec(
        "auto_retry", limit=1, by_code=True, jitter=True, degrade_when_unresolved=True
    ),
    "internal": RecoverySpec(
        "auto_retry", limit=1, by_code=True, jitter=True, degrade_when_unresolved=True
    ),
    "not_permitted": RecoverySpec(
        "skip_and_tell", limit=0, by_code=True, degrade_when_unresolved=True
    ),
    "conflict": RecoverySpec("skip_and_tell", limit=0, by_code=True, degrade_when_unresolved=False),
}

#: 退避基数与抖动比例。与 W2 的 D3（`retry.compute_delay`）同一套参数，
#: 不另外发明一套 —— 两处退避算法不一致，排查时先要搞清"是哪一层的退避"。
BACKOFF_BASE_S: Final[float] = 0.5
BACKOFF_JITTER: Final[float] = 0.5

#: 单次退避的硬上限（秒）。矩阵给的是"退 1 次"，但抖动公式在基数大时
#: 可能算出很长的等待；图里的 `asyncio.sleep` 必须有个天花板，
#: 否则一次重试就能把 20 任务评测的墙钟时间拖到无法解释。
RETRY_MAX_DELAY_S: Final[float] = 2.0


@dataclass(frozen=True, slots=True)
class Decision:
    """一次失败的处置决定。**纯数据，不含副作用** —— 节点照着它走。"""

    action: RecoveryAction
    error_type: str
    tool: str
    attempt: int  # 已经用掉几次（含本次）
    limit: int
    delay_s: float | None  # auto_retry 才有时值
    reason: str  # 给人看的判定理由（进轨迹，便于事后解释）
    by_code: bool = False  # True = 代码执行；False = 交回模型
    degraded_when_unresolved: bool = False  # 最终没解决时算不算降级缺口

    @property
    def exhausted(self) -> bool:
        return self.attempt >= self.limit


def spec_of(error_type: str) -> RecoverySpec:
    return RECOVERY_MATRIX.get(normalize_error_type(error_type)) or RECOVERY_MATRIX["internal"]


def backoff_delay(attempt: int, *, rng: random.Random | None = None) -> float:
    """第 `attempt` 次失败后的等待秒数（指数退避 + 抖动）。

    复用 `retry.compute_delay`：W2 的 D3 已经把"为什么用乘法抖动"论证过一遍，
    这里不再重复实现——同一件事两份实现，两处口径迟早漂移（W10 的教训）。
    """
    return compute_delay(
        attempt, base=BACKOFF_BASE_S, max_delay=8.0, jitter=BACKOFF_JITTER, rng=rng
    )


def decide(
    envelope: dict[str, Any],
    counts: dict[str, int],
    *,
    rng: random.Random | None = None,
) -> Decision:
    """根据失败信封与历史计数，给出处置决定。**原地更新 `counts`**。

    `counts` 的键是 `f"{tool}|{error_type}"`：按**错误类型**累计而不是按参数。
    模型最常见的错误动作是"换个措辞重试同一个坏工具"，按参数计数拦不住它。

    这个函数是**纯判定**：不 sleep、不执行、不改信封。节点拿它做路由，
    测试也拿它做断言 —— 判定与副作用分开，才能"不起图也能单测路由"。
    """
    error_type = normalize_error_type(envelope.get("error_type") or envelope.get("kind"))
    tool = str(envelope.get("tool") or "?")
    spec = spec_of(error_type)
    key = f"{tool}|{error_type}"
    counts[key] = counts.get(key, 0) + 1
    attempt = counts[key]

    if not envelope.get("retryable", True):
        # 信封说不可重试（权限/冲突/空结果/预算已抑制）：
        # **矩阵不许擅自重试**，这是两层结论必须一致的地方。
        #
        # limit=0 的格子是"一用即尽"（权限错、幂等冲突）。此时取决于它
        # 最终算不算缺口：
        # - 权限错：这块数据永远拿不到 → 立刻 degrade（表里"用尽后"那一列）
        # - 幂等冲突：效果**已经发生**，任务其实完成了 → skip_and_tell，
        #   把"已完成"如实报上去就够，不该标成降级
        if spec.limit == 0 and spec.degrade_when_unresolved:
            action: RecoveryAction = "degrade"
        elif spec.action == "auto_retry":
            # 矩阵想重试但预算已被抑制 → 退化成 degrade
            action = "degrade"
        else:
            action = spec.action
        return Decision(
            action=action,
            error_type=error_type,
            tool=tool,
            attempt=attempt,
            limit=spec.limit,
            delay_s=None,
            reason=(
                f"{error_type} 不可重试，处置为 {action}（矩阵 limit={spec.limit}，"
                f"degrade_when_unresolved={spec.degrade_when_unresolved}）"
            ),
            by_code=spec.by_code,
            degraded_when_unresolved=spec.degrade_when_unresolved,
        )

    if not spec.by_code:
        # 参数错/空结果：代码不知道怎么改，必须交回模型
        return Decision(
            action="model_fix",
            error_type=error_type,
            tool=tool,
            attempt=attempt,
            limit=spec.limit,
            delay_s=None,
            reason=f"{error_type} 需要模型改策略（代码不知道该改成什么）",
            by_code=False,
            degraded_when_unresolved=spec.degrade_when_unresolved,
        )

    if attempt <= spec.limit:
        delay = backoff_delay(attempt, rng=rng) if spec.jitter else BACKOFF_BASE_S
        return Decision(
            action="auto_retry",
            error_type=error_type,
            tool=tool,
            attempt=attempt,
            limit=spec.limit,
            delay_s=delay,
            reason=f"第 {attempt}/{spec.limit} 次可重试失败，代码退避 {delay:.2f}s 后重试",
            by_code=True,
            degraded_when_unresolved=spec.degrade_when_unresolved,
        )

    return Decision(
        action="degrade",
        error_type=error_type,
        tool=tool,
        attempt=attempt,
        limit=spec.limit,
        delay_s=None,
        reason=f"{error_type} 已重试 {spec.limit} 次用尽预算",
        by_code=True,
        degraded_when_unresolved=spec.degrade_when_unresolved,
    )


# ============================================================ 降级载荷


def build_degrade_payload(
    *,
    reason: str,
    actions: list[dict[str, Any]],
    partial_notes: list[str] | None = None,
    unresolved: list[str] | None = None,
) -> dict[str, Any]:
    """构造降级答案：`{answer, missing, degraded, reason}`。

    **`answer` 只拼已有事实，不再进 LLM。** 这是本周最容易被写坏的一行：
    一旦"让模型润色一下部分结论"，缺的那块就会被它顺手补上——
    答案看起来完整了，可信度反而归零。

    `missing` 逐条列出"哪个工具、因为什么没拿到"。
    `reason` 只放 error_type（机器可聚合），细节进 `missing`。
    `unresolved` 是"本任务真正必需、但最终没拿到的能力"（比如权限错导致
    某张表查不了），它与"失败的调用"不是一回事：失败的调用可能后来被重试
    救回来了，那就是已解决，不该出现在 missing 里。

    没有成功的调用时，`answer` 必须明说"没有取得任何结论"，
    不要用"暂无法回答"这类含糊话 —— 含糊话会被读成"稍后再试就有了"。
    """
    succeeded: list[str] = []
    missing: list[str] = []
    seen_missing: set[str] = set()
    for action in actions:
        tool = str(action.get("tool") or "?")
        if action.get("ok"):
            if tool not in succeeded:
                succeeded.append(tool)
            continue
        etype = action.get("error_type") or action.get("kind") or "internal"
        item = f"{tool}（{etype}）"
        if item not in seen_missing:
            seen_missing.add(item)
            missing.append(item)

    if succeeded:
        answer = "以下结论仅基于已成功取得的数据，**不完整、不可作为最终判定**：\n" + "\n".join(
            f"- 已取得：{tool} 的返回结果（见轨迹）" for tool in succeeded
        )
    else:
        answer = "本次没有取得任何可用结论（所有数据获取都失败了）。"

    if partial_notes:
        answer += "\n" + "\n".join(f"- {note}" for note in partial_notes)

    # 真正"没拿到"的，以 unresolved 为准（失败的调用可能已被重试救回）。
    # **是替换而不是求交集**：unresolved 由调用方按"任务还需要什么"给出，
    # 比"哪些调用失败了"更准确。求交集会把两边写法稍有差异的条目全过滤掉
    # ——第一版就是这么写的，结果 missing 恒为空，降级退化成"只说失败不说缺什么"。
    if unresolved is not None:
        deduped: list[str] = []
        for item in unresolved:
            if item not in deduped:
                deduped.append(item)
        missing = deduped

    return {
        "answer": answer,
        "missing": missing,
        "degraded": True,
        "reason": normalize_error_type(reason),
    }


def render_degrade_text(payload: dict[str, Any]) -> str:
    """降级载荷 → 给人看的最终答案文本。

    结构化载荷进 state/trajectory（机器读），这段文本进 final_answer（人读）。
    **两者都保留，不合并**：把 JSON 当答案给人看，用户读不懂；
    把这段文本反解析回去，机器读不准。
    """
    lines = [
        "[降级回答] 任务未完成。下面是**已经确定的部分**和**缺的部分**，",
        "请勿把它当作完整结论使用。",
        "",
        str(payload.get("answer", "")).strip(),
        "",
        "缺失（因为什么没拿到）：",
    ]
    missing = payload.get("missing") or []
    if missing:
        lines.extend(f"  - {item}" for item in missing)
    else:
        lines.append("  - （无）")
    lines.append("")
    lines.append(f"原因分类：{payload.get('reason')}（retryable 已用尽或本就不可重试）")
    return "\n".join(lines)


__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_JITTER",
    "RECOVERY_MATRIX",
    "RETRY_MAX_DELAY_S",
    "Decision",
    "RecoveryAction",
    "RecoverySpec",
    "backoff_delay",
    "build_degrade_payload",
    "decide",
    "render_degrade_text",
    "spec_of",
]
