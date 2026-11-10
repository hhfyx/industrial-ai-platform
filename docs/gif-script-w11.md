# GIF 脚本 · 可靠性演示（W11）

> 录 **"注入超时 → 代码退避重试 → 仍失败 → 结构化降级"**。
> 为什么是这一段：它是本周唯一一段**能看见机制在工作**的画面。
> 成功率数字需要解释，降级提示不需要 —— 观众一眼就看懂
> "它知道自己没做成，并且说清了缺什么"。

## 一、录制前准备

```powershell
cd D:\git\网页\industrial-ai-platform
# 1. 确认中断与降级可用（离线，不烧配额）
.\.venv\Scripts\python.exe -m pytest tests\test_recovery_w11.py -q

# 2. 确认降级载荷能单独渲染（这一步不调模型）
.\.venv\Scripts\python.exe -X utf8 -c "from app.core.llm.recovery import build_degrade_payload, render_degrade_text; print(render_degrade_text(build_degrade_payload(reason='timeout', actions=[{'tool':'query_metrics_sql','ok':False,'error_type':'timeout'}], unresolved=['query_metrics_sql（timeout：已重试 1 次用尽预算）'])))"
```

## 二、录制脚本（按时间轴）

| 秒 | 屏幕 | 台词/字幕 |
| --- | --- | --- |
| 0–4 | 终端：`python -X utf8 scripts/run_eval_20.py --live --only T7_complex_report` | 「20 任务集里最脆的一个」|
| 4–10 | 轨迹逐行打出：`query_metrics_sql` ✓ → `retrieve_spec_chunks` ✓ | 「工具正常」|
| 10–16 | **注入超时**（改用带故障注入的脚本，见下）→ 第一行 `✗ timeout` | 「第 4 次调用：超时」|
| 16–22 | 第二行 `↻ retried=True`，代码自己退避 0.32s 后重试 | 「**模型没参与**——重试是代码做的」|
| 22–30 | 再失败 → 打出降级块（见下）| 「仍失败 → 降级。注意它说了缺什么」|

## 三、降级画面的实际文本（照抄，不要改）

```
[降级回答] 任务未完成。下面是**已经确定的部分**和**缺的部分**，
请勿把它当作完整结论使用。

以下结论仅基于已成功取得的数据，**不完整、不可作为最终判定**：
- 已取得：query_metrics_sql 的返回结果（见轨迹）
- 已取得：retrieve_spec_chunks 的返回结果（见轨迹）

缺失（因为什么没拿到）：
  - query_metrics_sql（timeout：已重试 1 次用尽预算）

原因分类：timeout（retryable 已用尽或本就不可重试）
```

**这三行是关键**，字幕要点出来：

1. `已取得：…` —— 它没有假装什么都没做；
2. `缺失：…（timeout：已重试 1 次用尽预算）` —— 它说清了**为什么**；
3. `请勿把它当作完整结论使用` —— 它没有给一个看起来正常的错答案。

## 四、注入超时的可复现命令

故障注入只在 `tests/fault_injection.py`，**不进生产路径**。
录屏时用一个小脚本临时包一层：

```python
# 临时脚本（录完即删，不入库）
import asyncio, sys

sys.path.insert(0, ".")
from tests.fault_injection import FaultInjector
from app.core.llm.agent_registry import execute_agent_tool
from app.core.llm.agent_graph import build_agent_graph
from app.core.config import get_settings
from app.core.llm.client import get_llm_client
from app.core.llm.react_loop import build_system_prompt


async def main():
    s = get_settings()
    inj = FaultInjector(execute_agent_tool, failures={1: "timeout", 2: "timeout"})
    app = build_agent_graph(get_llm_client(s), executor=inj)
    state = await app.ainvoke(
        {
            "instruction": "B20241207-01 的抗压强度是多少？",
            "model": s.llm.model,
            "max_steps": 10,
            "messages": [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": "B20241207-01 的抗压强度是多少？"},
            ],
        },
        config={"configurable": {"thread_id": "gif::degrade"}},
    )
    print("stopped:", state["stopped_reason"])
    for m in state.get("steps", []):
        for a in m.get("actions", []):
            print(
                f"  {a['tool']:22s} ok={a['ok']} type={a.get('error_type')} retried={a.get('retried', False)}"
            )
    print()
    print(state["final_answer"])


asyncio.run(main())
```

先用 `--live` 跑一次确认真实降级文本，再录屏。

## 五、录制参数

| 项 | 值 | 理由 |
| --- | --- | --- |
| 时长 | 30s，循环播放 | 与 W10 的 21s GIF 同一节奏 |
| 分辨率 | 1280×720 | README 里显示清晰、文件不至于太大 |
| 字号 | 16–18px | 手机上看也清楚 |
| 主题 | 深色终端 | 轨迹的白绿红对比明显 |
| 帧率 | 12 fps | 文字滚动够顺，文件小 |

## 六、不要做（四条）

1. **不要加速到看不清** —— 这个 GIF 的价值是"能读到字"，不是"显得快"；
2. **不要剪掉失败画面** —— 只展示成功的那段，等于把"可靠性"演示成"运气好"；
3. **不要用真实端点故障冒充注入** —— 那不可复现，也说不清是机制还是环境；
4. **不要在字幕里写"成功率 100%"** —— 区间是 70–100%，最好的一轮才是 100%。

## 七、录完之后

```powershell
# 确认 GIF 已生成、体积合理（目标 < 3 MB）
Get-Item docs\agent-reliability.gif | Select-Object Name, Length

# 落盘后确认 README 引用路径对得上
Select-String -Path README.md -Pattern 'agent-reliability.gif'
```
