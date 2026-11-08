# LangGraph vs 手写循环：一份有证据的对比（W10 周日）

> 对照实验：W9 手写 ReAct 循环（`qa_runs_w9_r3`）vs W10 LangGraph 图版（`qa_runs_w10` 三轮）。
> 冻结声明见 [weekly/w10.md](weekly/w10.md)。本文每条结论都带本周的实测证据——
> 「两处碍事」比「三处更好」值钱，能说出框架哪里不舒服，才是真的用过它。

## 数据先行（trae/qwen-3.5，同一判据同一 10 任务）

| 指标 | W9 手写 | W10 图版（3 轮均值） | 差异 |
| --- | --- | --- | --- |
| 判据成功率 | 90% | 90%（三轮一致） | 持平 |
| 平均步数 | 2.6 | 2.4 | -0.2 |
| 平均 token | 8072 | 7455 | -7.7% |
| 平均延迟 | 14.9s | 13.4s | -1.5s |
| 工具错误率 | 14% | ~16% | 构成不变（环境 + 预期 not_found） |

迁移的 token 成本实测为 **0**：路由是纯函数读 state，checkpoint 写盘不耗 token，
图版没有增加任何一次模型调用。

## 三处明显更好

### 1. 状态 schema 显式且可测

手写版的状态散在循环局部变量里（`action_seen`、`failure_streak`、`parse_fail_streak`……
全是闭包内变量，坏了只能 print 追）；图版全部进了 `AgentGraphState`，
**每个字段带 reducer 语义声明**（累积 or 设值），并且 Tuesday 起就有测试
直接喂 dict 验证合并语义——不起图就能测状态机。
证据：[test_w10_state.py](../../learn/llm/test_w10_state.py) 的
`test_no_reducer_silent_overwrite`（无 reducer 后写覆盖、静默）与
`test_reducer_merge_semantics`（reducer 合并、可断言）。手写循环没有这个测试面。

### 2. 持久化与恢复开箱即用

手写版要自己设计账本（W9 通知工具的 ledger）+ 自己写落盘 + 自己定义恢复语义
（`stopped_reason` 六态是自己发明的状态机）。图版 `SqliteSaver` + thread_id
直接得到：进程被 kill -9 后从崩溃节点精确续跑。
证据：周四演练（[w10_checkpointer.py](../../learn/llm/w10_checkpointer.py)）——
exit 137 硬杀，恢复时 `next=('step3',)`，step1/2 时间戳保留、一个节点不重；
`get_state_history` 免费得到每步回看（8 条历史），手写版要自己实现审计日志。
唯一要付的学费：默认 `durability="async"` 下硬杀会丢 checkpoint，要显式 `"sync"`。

### 3. 分支声明式，图即文档

手写版的控制流埋在 682 行循环体的 if/elif 里，六条终止路径要靠 docstring 的
ASCII 表解释；图版条件边 + 纯函数路由，路由函数不起图就能单测
（周三 `test_route_functions_are_pure_and_decidable` 直接喂 dict 断言 6 个分支）。
证据：[w10_state_flow.md](../../learn/llm/w10_state_flow.md) 的图
与代码结构一一对应——图的形状就是架构文档，新读者先看图再看代码。
周三教学图（三路分支 + 改写回环）从设计到可跑只花了约定的时间盒，
因为「加一个分支 = 加一条边」，不用动循环骨架。

## 两处碍事（值钱的部分）

### 1. 重执行语义让副作用设计变复杂

**恢复时节点从函数开头重新执行，不是从 interrupt() 那一行之后继续。**
手写循环里副作用放哪都行（循环是线性前进的）；图版里 interrupt 之前的副作用
会执行两次。实测：写文件放在 interrupt() 前，恢复后文件两行
（[w10_hitl.py](../../learn/llm/w10_hitl.py) 的 `double_write` 模式 +
回归测试 `test_side_effect_in_node_before_interrupt_runs_twice`）。
这逼出了新的设计纪律：**副作用必须全部放在恢复判定之后，或整段幂等 + 幂等键**
（与 W9 通知账本思想同构，但现在是结构约束而不是可选习惯）。见
[agent-architecture.md](agent-architecture.md) §1。

### 2. 流式与自定义错误处理的自由度受约束

手写循环里我想怎么截断 observation 就怎么截断（`_shrink_strings` 的首尾策略）、
想在哪埋保险丝就在哪埋（三重保险丝的触发时机是我定的）；图版里这些逻辑
要拆进节点与路由，而**框架的调度语义会掺进来**：`durability` 默认 async
（实测硬杀丢 checkpoint）、错误要靠 `stopped_reason` 置位 + 路由收敛而不是
直接 return——六种停止原因从「return 语句」变成「置位 + 边」，代码长了约 15%，
调试时多了一层「现在到底在哪个 super-step」的心智负担。
（缓解：LangSmith/stream 可视化存在，但离线跑批 + 自定义轨迹格式仍要自己做
`result_from_state` 这层适配。）

## 结论

**W10 之后默认用 LangGraph 写 Agent**：成功率持平、token 零成本迁移、
持久化/恢复/中断开箱即用；代价是重执行语义下的副作用纪律 + 一层调度心智。
两处碍事都不是否决项，而是「用的时候要记得的姿势」——已经固化成测试与文档。
