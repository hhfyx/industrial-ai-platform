# Agent 架构（W10 · LangGraph 迁移）

> W10 的架构事实来源。本文从设计稿起步，随 W10 各晚实测补充。
> 手写循环（W9）的架构见 [agent-design.md](agent-design.md)。
> 对比与取舍结论见 [langgraph-vs-handwritten.md](langgraph-vs-handwritten.md)。

## 0. 状态流转图与字段表（周日 10:50–11:30 补全）

全图（节点/条件边/回环/人工确认节点/checkpointer 位置）见
[learn/llm/w10_state_flow.md](../../learn/llm/w10_state_flow.md)——三张图：
评测图（本文 §4-5 引用的节点）、周三教学图（三路分支+改写回环）、周五 HITL 图。

**AgentGraphState 字段表**（完整定义见 `app/core/llm/agent_graph.py`）：

| 字段 | reducer | 写者 | 语义 |
| --- | --- | --- | --- |
| messages | `operator.add` | model_step / tools / parse_retry / finalize | 对话与回灌轨迹 |
| steps | `operator.add` | model_step / tools | dict 形态 StepRecord，轨迹组装时按 step 号合并 |
| usage_log | `operator.add` | model_step | 每次 API 调用的 usage 增量，token 对账的数据源 |
| step_no | 设值 | model_step | 已完成的模型步数（业务预算的读数） |
| pending_mode / pending_calls | 设值 | model_step | 本次模型输出的分类与待执行调用 |
| action_seen | 设值（整 dict 替换） | tools | (tool,args) 全局重复计数（熔断状态） |
| failure_streak / prev_failure_key | 设值 | tools | 同类失败连击（熔断状态） |
| parse_fail_streak | 设值 | parse_retry | 解析失败连击 |
| stopped_reason / final_answer | 设值 | 各节点（单写者收敛） | 停止原因与最终答案 |

## 0.5 三条终止路径（服务化调用方必读）

图版的「结束」有三条路，W12 服务化按此设计超时与重试：

1. **正常收敛**：`finalize`（stopped_reason=completed）——答案可信。
2. **业务软闸优雅退出**：`finalize_budget`（max_steps）或熔断分支
   （repeat_loop / unrepairable_failure / parse_failure / model_timeout）——
   返回部分结果 + 原因，**不是异常**，调用方可据 `_partial_progress` 内容决定重试。
3. **框架硬闸**：`GraphRecursionError`（异常）——只在图写错时发生；
   服务化时必须在调用层 try 住并报警（它意味着代码 bug，不是任务失败）。

查询执行进度用 `scripts/query_graph_state.py` 的 `query_progress(app, config)`
（当前节点 / 步数 / 是否等待人工 / 部分结果），W12 任务状态接口的直接底座。

## 1. interrupt() 的重执行语义（周五 · 面试可讲的「真踩过」）

**恢复时节点从函数开头重新执行，而不是从 interrupt() 那一行之后继续。**
这不是 bug，是实现方式决定的：挂起 = 整个节点的执行帧被丢弃，恢复 =
带着 resume 值**从头再跑一遍这个节点**。推论：

1. **interrupt() 之前的副作用会执行两次**——实测：写文件放在 interrupt() 前，
   恢复后文件里两行（`w10_hitl.py` 的 `double_write` 模式 + 测试
   `test_side_effect_in_node_before_interrupt_runs_twice`）。
2. **正确写法两条**：副作用全部放在「拿到 resume 值并判定之后」
   （本仓库 `act_node` 的结构）；或者副作用整段幂等 + 幂等键
   （与 W9 通知工具的 `result.delivered` 账本拦截同构）。
3. **interrupt() 之前放什么**：只读计算、组装 payload。payload 要回答
   「人在批准时需要看到什么」——草稿、参数、副作用预告，而不是只丢一句「批不批？」。

一句话版本：**interrupt 的语义是「重新问我一遍」，不是「从我停的地方继续」。**

## 2. 挂起与恢复的机制（周五）

- 挂起：节点内 `interrupt(payload)`，前提是 `compile(checkpointer=...)`——
  没有检查点，挂起的状态无处落盘。
- 恢复：`graph.invoke(Command(resume=value), config)`，`value` 成为
  interrupt() 的返回值。进程可以退出、过夜——**interrupt 不是阻塞等待**，
  它把状态落盘后抛出挂起，靠外部再次调用来恢复。天然适合「人几小时后才批」，
  不适合节点内同步等待。
- 三路径：approve（执行副作用）/ reject（记原因，回退重新生成）/
  revise（`update_state` 改参数后 resume，重跑生成）。

## 3. Checkpointer 的粒度与 durability（周四）

- **粒度 = super-step（节点边界）**。恢复点不是任意指令，是节点之间。
  崩溃演练实证：step3 边界前硬杀，恢复 `next=('step3',)`，step1/2 的
  trace 时间戳保留——一个节点不重。推论：长任务必须拆成节点粒度合理的图，
  节点内 5 分钟的不可中断推理，中断能力对它无效。
- **langgraph 1.2.x 默认 `durability="async"`**：checkpoint 异步落盘，
  硬杀进程会丢。要「崩溃真的能恢复」必须显式 `durability="sync"`。
  「接了 checkpointer」≠「崩溃能恢复」。
- 漏传 thread_id：带 checkpointer 时当场 `ValueError`（不是静默新线程）；
  真正的静默风险是**没接 checkpointer 却以为有持久化**。

## 4. 两层限制（周三）

`recursion_limit`（框架硬闸，超了抛 `GraphRecursionError`，是异常不是返回）
与业务预算 `retry_count` / `should_continue`（软闸，超了优雅返回部分结果 + 原因，
W9 的 B7 纪律）是两层防线：软闸先触发给体面，硬闸只在图写错时兜底。

## 5. State 设计（周二）

累积字段加 reducer（`add_messages` / `operator.add`），设值字段故意不加
（后写覆盖即期望语义）。LLM client / DB 连接不进 state：模块级单例 + 闭包注入，
state 只留 id 与数据，`json.dumps` 自检可序列化。

## 6. 与 W9 手写循环的对照（总表见 learn/llm/w10_monday_cheatsheet.md）

| 维度 | W9 手写 | W10 LangGraph |
| --- | --- | --- |
| 循环 | 自己写 while | 声明图，框架跑 |
| 路由 | 循环内 if/elif | conditional edges（纯函数路由） |
| 持久化 | 自己写账本/落库 | checkpointer（thread_id 索引） |
| 中断 | 自己设计状态机 | interrupt / Command(resume) |
| 副作用幂等 | 账本拦截（delivered 才拦） | 同一思想 + 重执行语义下的位置约束 |
