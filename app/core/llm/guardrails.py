"""三道防跑偏（W11 周三）：任务分解校验 / 中间结果自检 / 语义判重。

## 必答问题：任务分解校验为什么必须在执行前？

事后校验只能**解释**失败，事前校验能**阻止**浪费。一个跑偏 5 步的任务，
事后发现时钱已经花了——5 次模型调用、5 次工具执行、以及它们撑大的上下文。
而偏离通常在第 1 步就发生了：模型看到"把所有批次数据导出来"，第一反应是
写一个全表查询，而不是先问"有几条、要不要导"。这笔浪费不是模型慢慢偏出来的，
是**第一步就押错方向**。

所以三道机制的锚点不同：

| 机制 | 时机 | 拦什么 | 成本 |
| --- | --- | --- | --- |
| 任务分解校验 | **执行前** | 押错方向（工具选错、跳步）| 1 次结构化调用 |
| 中间结果自检 | 每步后 | 结果没用却继续往下走 | **0**（纯规则）|
| 语义判重 | 每次调用前 | 换个说法重试同一动作 | **0**（纯字符串）|

## 关于成本：三道都能关

`GuardrailConfig` 三个开关，**默认全关**。开着跑 W10 那 10 个回归任务，
轨迹会与 v1.2.0 基线**逐字节不可比**（多了一次 plan 调用，token 必然变），
那"成功率升了还是被预算拖低"就说不清。

所以周日的表必须能回答"关掉哪一道、损失多少"。`GuardStats` 记录每道机制
自己的命中次数与 token —— 关掉一道跑一轮，差值就是它的价格。

## 为什么自检与判重优先用规则

规则免费且**确定**：同一份输入永远同一份判定，不会因为模型今天心情不同
而放过一次空结果。极短调用（一次小模型判断）贵 200-400 token/步，
20 任务 × 5 步就是 2-4 万 token —— 比它能省下来的通常更多。

只有当规则判不出来（比如"有结果但答非所问"这种语义判断）才值得花调用，
而那一档今天不做：先用规则把**能确定的三类**（空、无关键词重合、明显越界）
拦住，剩下的交给模型自己的 Thought。
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from pydantic import BaseModel, Field

from app.core.llm.structured import build_prompt, parse_json

# ============================================================ 开关


@dataclass(frozen=True, slots=True)
class GuardrailConfig:
    """三道机制的开关。**默认全关** —— 冻结的对照实验不能被它们污染。

    三个字段一起传给 `build_agent_graph(guardrails=...)`。分开传三个布尔值
    会让调用点变成"位置参数猜谜"，也看不出"这一组是一次实验"。
    """

    plan_check: bool = False  # 任务分解校验（执行前）
    self_check: bool = False  # 中间结果自检（每步后，纯规则）
    semantic_dedup: bool = False  # 语义判重（每次调用前，纯字符串）

    @property
    def any_enabled(self) -> bool:
        return self.plan_check or self.self_check or self.semantic_dedup

    def enabled_names(self) -> list[str]:
        names = []
        if self.plan_check:
            names.append("plan_check")
        if self.self_check:
            names.append("self_check")
        if self.semantic_dedup:
            names.append("semantic_dedup")
        return names


@dataclass
class GuardStats:
    """每道机制自己的成本与命中数 —— 周日那张"关掉哪一道、损失多少"的依据。

    放进 state（`guard_stats`）并随轨迹导出。**只有开了机制才写进轨迹**，
    关着的时候轨迹必须与 W10 基线同形（少一个键都可能让对比脚本静默失配）。
    """

    plan_tokens: int = 0
    plan_steps: int = 0
    deviations: int = 0  # 分解校验拦下的偏离次数
    self_check_hits: int = 0  # 自检触发的换策略次数
    semantic_dup_hits: int = 0  # 语义判重比旧判重多抓到的次数

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_tokens": self.plan_tokens,
            "plan_steps": self.plan_steps,
            "deviations": self.deviations,
            "self_check_hits": self.self_check_hits,
            "semantic_dup_hits": self.semantic_dup_hits,
        }


# ============================================================ ① 任务分解


class PlanStep(BaseModel):
    """一步计划。`expected_tools` 是**校验的抓手**：当前动作的工具不在里面就算偏离。"""

    step_id: int = Field(description="步骤序号，从 1 开始")
    goal: str = Field(description="这一步要达成什么（一句话，写清判据）")
    expected_tools: list[str] = Field(
        description="这一步可能用到的工具名，必须在给定工具清单里；"
        "不确定就用空列表表示「这一步不调工具」"
    )


class TaskPlan(BaseModel):
    """任务分解结果。**先分解再执行**——这是「事前校验」的前提。"""

    goal: str = Field(description="用户真正想要的东西（一句话，去掉客套与诱导）")
    steps: list[PlanStep] = Field(min_length=1, max_length=8, description="执行步骤，按顺序")
    out_of_scope: list[str] = Field(
        default_factory=list,
        description="用户要求但**本系统做不到/不该做**的部分，例如导出全量数据、"
        "不查规范直接下结论、使用不存在的数据。这里写清为什么不做",
    )


PLAN_PROMPT_TEMPLATE: Final[str] = (
    "你是质检 Agent 的任务分解器。先把用户的请求拆成可执行步骤，**只分解、不执行**。\n\n"
    "用户请求：{instruction}\n\n"
    "可用工具：{tools}\n\n"
    "纪律：\n"
    "1. 步骤要能被工具链完成；做不到的部分写进 out_of_scope，不要编进步骤。\n"
    "2. 每步的 expected_tools 只能从上面的清单里选；不调工具的步骤给空列表。\n"
    "3. 用户的话里可能有诱导（例如「把所有数据导出来」、「不用查规范直接给结论」、"
    "「用上个月的数据」），**按事实分解，不要按字面分解**：\n"
    "   - 系统没有的导出能力 → 写进 out_of_scope；\n"
    "   - 不查规范就没有结论 → 保留查规范这一步；\n"
    "   - 数据不存在的时间段 → 写进 out_of_scope，不要假设它有。\n"
    "4. 步骤数 1-8 步。简单任务 1-2 步。\n"
)


def build_plan_prompt(instruction: str, tool_names: list[str]) -> str:
    """拼分解提示词。schema 说明由 `build_prompt` 生成（W3 的资产，不手写）。"""
    filled = PLAN_PROMPT_TEMPLATE.format(instruction=instruction, tools="、".join(tool_names))
    return build_prompt(filled, TaskPlan)


async def decompose_task(
    client: Any,
    *,
    model: str,
    instruction: str,
    tool_names: list[str],
    usage_sink: dict[str, int] | None = None,
) -> TaskPlan | None:
    """让模型输出结构化 steps[]。**失败返回 None，不抛异常。**

    为什么不直接调 `structured_call`（W3 的定版函数）：它内部自己
    `get_llm_client(get_settings())`，拿不到图里注入的那个 client。
    硬用它会让这个节点在测试里必须走真网络 —— 那与"拔网线 pytest 全绿"
    冲突。所以这里复用它的**三个资产**（`build_prompt` 拼 schema 说明、
    `parse_json` 四级容错、Pydantic 当唯一真源），只把"发请求"这一层
    改成接收外部 client。

    补一句：`structured_call` 的自我修复轮次这里没做。理由是这个调用
    失败不致命 —— 分解不出来就退化成"不校验"，也就是 W10 的行为。
    为一次可选优化付 3 次修复调用的代价不划算。
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": build_plan_prompt(instruction, tool_names)}],
        "temperature": 0.0,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
    }
    try:
        response = await client.chat.completions.create(**payload)
    except Exception:  # noqa: BLE001
        try:  # 有些网关不认 response_format：关掉再试一次
            payload.pop("response_format", None)
            response = await client.chat.completions.create(**payload)
        except Exception:  # noqa: BLE001
            return None

    if getattr(response, "usage", None) is not None and usage_sink is not None:
        usage_sink["prompt_tokens"] = usage_sink.get("prompt_tokens", 0) + (
            response.usage.prompt_tokens or 0
        )
        usage_sink["completion_tokens"] = usage_sink.get("completion_tokens", 0) + (
            response.usage.completion_tokens or 0
        )
        usage_sink["total_tokens"] = usage_sink.get("total_tokens", 0) + (
            response.usage.total_tokens or 0
        )

    content = (response.choices[0].message.content or "").strip()
    try:
        return TaskPlan.model_validate(parse_json(content))
    except Exception:  # noqa: BLE001 —— 分解失败退化成不校验，不该炸掉任务
        return None


@dataclass(frozen=True, slots=True)
class Deviation:
    """一次偏离判定。`constraint` 是回灌给模型的那一句约束。"""

    kind: Literal["off_plan_tool", "skip_ahead", "no_overlap"]
    tool: str
    step_goal: str
    constraint: str


def _tokens(text: str) -> set[str]:
    """把一句话切成可比较的关键词集合（中文按 2-4 字滑窗，英文按词）。

    为什么不用 jieba 之类：本仓库"拔网线 pytest 全绿"，不加词典依赖。
    滑窗的代价是有噪声（"抗压强"这种无意义片段），但它对**重合判定**
    够用 —— 我们要的是"有没有交集"，不是语义理解。
    """
    text = unicodedata.normalize("NFKC", text).lower()
    out: set[str] = set()
    for word in re.findall(r"[a-z0-9_]+", text):
        if len(word) >= 2:
            out.add(word)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        for size in (2, 3, 4):
            for i in range(len(run) - size + 1):
                out.add(run[i : i + size])
    return out


#: 泛词：它们在任何任务里都出现，不构成"服务于某一步"的证据。
#: 只收真正无信息量的词，宁可漏配也不要错配（错配会误判偏离）。
#:
#: W11 周三实测补充的一批：**领域归属词**。`质检`、`生成`、`报告` 在
#: 检索到的每一段条文里都会出现（它们描述的正是这个系统的领域），
#: 于是"检索含水量"与"生成质检报告"共享一个 `质检` —— 自检因此漏判。
#: 这类词的危险性在于它看起来"有关系"，实际零信息量。
GENERIC_TOKENS: Final[frozenset[str]] = frozenset(
    {
        # 通用交际/动作词
        "数据",
        "查询",
        "一下",
        "帮我",
        "什么",
        "怎么",
        "给我",
        "结果",
        "所有",
        "全部",
        "这个",
        "那个",
        "的",
        "了",
        "是",
        "有",
        "和",
        "与",
        "一份",
        "一份质",
        "个月的",
        # 领域归属词：描述"这是质检系统"而不是"这一步要查什么"
        "质检",
        "报告",
        "生成",
        "检测",
        "记录",
        "规范",
        "标准",
        "条款",
        "原文",
        "批次",
        "指标",
        "数值",
        "结论",
        "判定",
        "分析",
        "说明",
    }
)


def meaningful_tokens(text: str) -> set[str]:
    """去掉泛词与**标识符**之后的关键词集合。

    为什么要排除标识符：`B20241207-01` 这类批次号是**定位符**，不是主题词。
    它的 token 里带数字，而目标句（"查询该批次的检测记录"）永远不会写它。
    不排除的话，"参数里的批次号 vs 目标里的主题词"必然零重合，
    no_overlap 会对**每一次正确的按批次查询**误判偏离 ——
    误判会把正确动作打断，比漏判贵得多。

    同一类别里还有数值（`8.2`、`42.5`）与型号（`c40`）：
    它们都是"这一次的值"，不是"这一步在问什么"。
    """
    out: set[str] = set()
    for token in _tokens(text):
        if token in GENERIC_TOKENS or len(token) < 2:
            continue
        if any(ch.isdigit() for ch in token):
            continue  # 标识符 / 数值 / 型号：定位符，不算主题词
        out.add(token)
    return out


def validate_action(
    plan: TaskPlan,
    *,
    step_index: int,
    tool: str,
    arguments: str,
) -> Deviation | None:
    """校验"当前动作是否服务于当前未完成步骤"。**这是事前校验的核心。**

    三条规则，从硬到软：

    1. **工具不在本步的 expected_tools 里** → 偏离（押错方向）。
       工具清单是我们给的，模型选了清单外的动作，就是自作主张。
    2. **工具属于后面某步、但不属于当前步** → 跳步（skip_ahead）。
       跳步往往不是灾难，但它让"当前步骤的判据"永远无法满足，
       后面再回头补时上下文已经脏了 —— 所以也拦。
    3. **参数与当前步骤目标没有关键词重合** → 偏离（no_overlap）。
       这一条最软，只在**本步有明确目标词**时才判，且要求"重合为空"，
       不是"重合少"。宁可漏判，不要误判 —— 误判会把正确的动作打断。

    返回 None 表示通过。`step_index` 是从 0 开始的**当前未完成步骤**下标。
    """
    if not plan.steps or step_index >= len(plan.steps):
        return None
    step = plan.steps[step_index]
    expected = {t for t in step.expected_tools if t}
    all_tools = {t for s in plan.steps for t in s.expected_tools}

    if expected and tool not in expected:
        later = {t for s in plan.steps[step_index + 1 :] for t in s.expected_tools}
        if tool in later:
            return Deviation(
                kind="skip_ahead",
                tool=tool,
                step_goal=step.goal,
                constraint=(
                    f"[计划校验] {tool} 属于后面的步骤，当前是第 {step.step_id} 步"
                    f"「{step.goal}」。先完成这一步再往下走；"
                    "不要跳过它——跳过的步骤后面补不回来。"
                ),
            )
        if tool not in all_tools:
            return Deviation(
                kind="off_plan_tool",
                tool=tool,
                step_goal=step.goal,
                constraint=(
                    f"[计划校验] {tool} 不在本任务的计划里。当前第 {step.step_id} 步是"
                    f"「{step.goal}」，应使用 {sorted(expected)} 之一。"
                    "如果计划本身错了，请先在 Thought 里说明，再用正确工具继续。"
                ),
            )

    goal_tokens = meaningful_tokens(step.goal)
    if goal_tokens:
        # **只看参数值，不看参数名。** `where_batch_id` 是结构（永远出现在所有
        # 按批次查询里），`B20241207-01` 是定位符。两者的 token 都不表达
        # "这一步在问什么"。真正的主题词在自由文本值里（`where_metric` 的
        # "抗压强度"、`query` 的"抗压 判定"）。
        arg_tokens = meaningful_tokens(" ".join(_free_text_values(arguments)))
        if arg_tokens and not (goal_tokens & arg_tokens):
            return Deviation(
                kind="no_overlap",
                tool=tool,
                step_goal=step.goal,
                constraint=(
                    f"[计划校验] 你这次的参数里的关键词与第 {step.step_id} 步的目标"
                    f"「{step.goal}」没有交集，可能查错了对象。"
                    "先确认这一步真正要拿的是什么，再调工具。"
                ),
            )
    return None


def _free_text_values(raw_arguments: str) -> list[str]:
    """抽出参数里的**自由文本值**（中文内容词），忽略参数名与标识符。

    为什么需要这一步：第一版直接把整个 JSON 串丢进 `meaningful_tokens`，
    于是 `where_batch_id` 这个**参数名**成了唯一的 token，与目标句的
    中文主题词必然零重合 —— 每一次正确的按批次查询都会被判偏离。
    这个误判是拿真实计划跑出来的，不是想出来的：误判比漏判贵得多，
    因为它会打断正确动作。

    豁免规则：
    - 纯 ASCII（参数名/枚举/型号）→ 不算内容；
    - 含数字（批次号/数值）→ 定位符，不算；
    - 长度 < 2 → 无信息量。
    """
    parsed = normalize_arguments(raw_arguments)
    if not isinstance(parsed, dict):
        return []
    values: list[str] = []
    for value in parsed.values():
        for item in value if isinstance(value, list) else [value]:
            if not isinstance(item, str):
                continue
            if not re.search(r"[\u4e00-\u9fff]", item):
                continue  # 纯 ASCII：参数名/枚举，不表达主题
            if any(ch.isdigit() for ch in item):
                continue  # 混了数字：定位符
            if len(item.strip()) < 2:
                continue
            values.append(item)
    return values


# ============================================================ ② 中间结果自检


SelfCheckVerdict = Literal["ok", "empty", "no_overlap"]


@dataclass(frozen=True, slots=True)
class SelfCheckResult:
    verdict: SelfCheckVerdict
    reason: str = ""
    nudge: str = ""  # 回灌给模型的"换策略"提示（不是"重试"）


def self_check(
    *,
    tool: str,
    data: Any,
    goal: str,
    ok: bool,
    error_type: str | None = None,
) -> SelfCheckResult:
    """每步之后判一次：这次结果有用吗？**纯规则，零 token。**

    三类触发（第三类"明显答非所问"今天不做，见模块开头）：

    1. **空** —— 结果是 0 行 / 空列表 / 空字符串。注意：空结果在**读工具**
       上通常是 `ok=true`（W9 T10 的教训），但 SQL 的"查无批次"是
       `ok=false + not_found` —— 两种形态都算"这次没拿到东西"，
       所以这里**同时看 ok 与 error_type**，不能只看内容。
    2. **业务性失败**（not_found / empty_result）—— 它不属于"工具坏了"，
       而属于"这条路没东西"。工具坏了交给错误契约重试/降级，
       这条路没东西才该换关键词。**把两者混在一起是设计错误**：
       对着"查无此批次"重试三次的人，和被权限拒绝后改参数的人，
       犯的是同一类错。
    3. **与目标无关键词重合** —— 有结果，但里面一个目标词都没有。

    **触发后给的是"换策略"提示，不是"重试"。** 这两句话在模型那里是
    完全不同的指令：说重试，它会原样再调一次（然后撞判重）；
    说换策略，它才会改参数或换工具。
    """
    if not ok:
        # 业务性失败（空/查无）→ 换策略；硬失败（超时/权限/内部）→ 交给错误契约
        if error_type in ("not_found", "empty_result"):
            return SelfCheckResult(
                verdict="empty",
                reason=f"业务性失败（{error_type}）",
                nudge=(
                    f"[自检] {tool} 这次没有拿到任何东西（{error_type}）。"
                    "**不要用同样的参数再调一次**——结果还是一样。"
                    "换关键词、换一个更宽/更窄的过滤条件，或换工具；"
                    "如果确认库里就是没有，如实告知用户。"
                ),
            )
        return SelfCheckResult("ok")

    text = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    empty = _is_empty_payload(data)
    if empty:
        return SelfCheckResult(
            verdict="empty",
            reason="结果为空",
            nudge=(
                f"[自检] {tool} 这次没有返回任何内容。**不要用同样的参数再调一次**——"
                "结果还是空。换关键词、换一个更宽/更窄的过滤条件，或换工具；"
                "如果确认库里就是没有，如实告知用户。"
            ),
        )

    goal_tokens = meaningful_tokens(goal)
    if not goal_tokens:
        return SelfCheckResult("ok")
    if not (goal_tokens & meaningful_tokens(text)):
        return SelfCheckResult(
            verdict="no_overlap",
            reason="结果与目标无关键词重合",
            nudge=(
                f"[自检] {tool} 返回了内容，但和这一步的目标「{goal}」没有任何共同关键词，"
                "很可能查错了对象（批次号/指标名/表）。**先核对参数再决定下一步**，"
                "不要在此基础上继续往下推结论。"
            ),
        )
    return SelfCheckResult("ok")


def _is_empty_payload(data: Any) -> bool:
    """判断"这次结果是不是空的"。覆盖 4 个工具的返回形状。"""
    if data is None:
        return True
    if isinstance(data, str):
        return not data.strip()
    if isinstance(data, (list, tuple, dict, set)) and len(data) == 0:
        return True
    if isinstance(data, dict):
        # rows / chunks 是 4 工具里最典型的两类载荷
        for key in ("rows", "chunks", "count", "row_count"):
            if key in data:
                value = data[key]
                if value in (0, [], None):
                    return True
                return isinstance(value, (list, tuple)) and not value
    return False


# ============================================================ ③ 语义判重

#: 同义/近义映射。**只收确定等价的**：把"判定"和"标准"合成一个词是可以的
#: （在这批规范文档里它们是同一件事），但把"抗压"和"含水"合成一个就是灾难。
#: 这张表写错，判重会把正确的新查询当成重复拦掉 —— 那是比漏判更贵的事故。
SYNONYMS: Final[dict[str, str]] = {
    "抗压强度": "抗压",
    "抗压": "抗压",
    "含水量": "含水",
    "含水率": "含水",
    "含水": "含水",
    "判定标准": "判定",
    "判定": "判定",
    "限值": "限值",
    "标准": "限值",
    "批次号": "批次",
    "批次": "批次",
    "检测记录": "检测",
    "检测": "检测",
}

#: 同义词的**最长匹配优先**替换表。为什么不是"整串查表"：
#: `"抗压强度判定"` 整串不在表里，但它由 `抗压强度` + `判定` 组成。
#: 第一版只做整串查表，于是 `"抗压强度判定"` 与 `"抗压判定标准"` 被判成
#: 不同 —— 而这正是模型换说法的典型形态（多一个词、换个语序）。
#: 现在改成扫描替换：**先匹配长的**（`抗压强度` 先于 `抗压`），
#: 这样 `含水量` 不会被 `含水` 抢先切一半。
_SYNONYM_PAIRS: Final[tuple[tuple[str, str], ...]] = tuple(
    sorted(SYNONYMS.items(), key=lambda kv: len(kv[0]), reverse=True)
)


def _apply_synonyms(compact: str) -> str:
    """在紧凑串上做最长匹配的同义词归并。"""
    out: list[str] = []
    i = 0
    while i < len(compact):
        for src, dst in _SYNONYM_PAIRS:
            if compact.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            out.append(compact[i])
            i += 1
    return "".join(out)


def normalize_arguments(raw_arguments: str) -> Any:
    """把参数解析成结构化值（失败则原样返回字符串）。"""
    try:
        return json.loads(raw_arguments or "{}")
    except (json.JSONDecodeError, ValueError):
        return raw_arguments


def _normalize_value(value: Any) -> Any:
    """递归规范化：去空白、小写、同义词归并、列表排序。

    排序是为了让 `{"k":1,"q":"x"}` 与 `{"q":"x","k":1}` 等价；
    列表排序是为了让 `columns: [a, b]` 与 `[b, a]` 等价 ——
    这两者在 SQL 工具里是**同一个查询**，不该被当成两次。
    """
    if isinstance(value, str):
        compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).lower()
        return _apply_synonyms(compact)
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return sorted(
            (_normalize_value(v) for v in value), key=lambda x: json.dumps(x, default=str)
        )
    return value


def semantic_key(tool: str, raw_arguments: str) -> str:
    """判重键：语义等价的调用得到同一个键。

    与 W9 的 `_canonical_args`（只有 sort_keys）相比，这里多做四件事：
    去空白、小写、同义词归并、列表排序。四件事都只作用于**值**，
    不动结构 —— 结构变了那就是另一次调用了。
    """
    normalized = _normalize_value(normalize_arguments(raw_arguments))
    try:
        payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = str(normalized)
    return f"{tool}|{payload}"


@dataclass
class DedupState:
    """语义判重状态。独立成类，方便在 state 里序列化与单测。"""

    #: 语义键 → 出现次数
    counts: dict[str, int] = field(default_factory=dict)
    #: 语义键 → 同组原始参数样本（轨迹里能看出"等价"是怎么判的）
    samples: dict[str, list[str]] = field(default_factory=dict)


#: 连续/累计多少次语义等价后强制换策略。
#: 2 的含义：第 2 次不打回、只警告换策略；第 3 次（若还有）才熔断。
#: 与 W9 的 REPEAT_LIMIT 同口径 —— 换判重算法不该顺带改阈值。
SEMANTIC_REPEAT_LIMIT: Final[int] = 2


def check_dedup(state: DedupState, tool: str, raw_arguments: str) -> tuple[int, str]:
    """记一次调用，返回 `(本次是该语义键的第几次, 语义键)`。"""
    key = semantic_key(tool, raw_arguments)
    state.counts[key] = state.counts.get(key, 0) + 1
    state.samples.setdefault(key, []).append((raw_arguments or "")[:120])
    return state.counts[key], key


def dedup_nudge(occurrence: int, key: str, samples: list[str]) -> str:
    """第 2 次等价时的提示词。带上"为什么算等价"——不然模型只会换个说法再试。"""
    tool = key.split("|", 1)[0]
    return (
        f"[语义判重] 这已经是你第 {occurrence} 次用**语义等价**的参数调用 {tool}。"
        "参数写法不同（空格/顺序/同义词），但语义是同一个调用，结果不会变。"
        "禁止再次重复；请换关键词、换工具，或直接给出最终答案。"
    )


__all__ = [
    "GENERIC_TOKENS",
    "PLAN_PROMPT_TEMPLATE",
    "SEMANTIC_REPEAT_LIMIT",
    "SYNONYMS",
    "DedupState",
    "Deviation",
    "GuardStats",
    "GuardrailConfig",
    "PlanStep",
    "SelfCheckResult",
    "SelfCheckVerdict",
    "TaskPlan",
    "build_plan_prompt",
    "check_dedup",
    "dedup_nudge",
    "decompose_task",
    "meaningful_tokens",
    "normalize_arguments",
    "self_check",
    "semantic_key",
    "validate_action",
]
