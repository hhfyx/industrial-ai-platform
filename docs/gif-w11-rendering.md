# GIF 生成说明 · 可靠性演示（W11）

> 产物：`docs/agent-reliability.gif`（11 帧 / 199 KB / 34 秒 / 960×360，循环）。
> 生成器：`scripts/make_reliability_gif.py`。

## 一、不做录屏，从真实轨迹渲染帧

与 W9 的 `make_agent_gif.py` 同一条理由：Windows 录屏依赖复杂、**不可复现**。
改成"每帧是一张终端画面式的文字快照，用 PIL 画出来"。

**关键差别**：每一帧里的工具名、错误类型、`retried` 标记、token 数
**全部读自 `qa_runs_w11_r5/trajectories/*.json`**，不是手写进脚本的字符串。
所以 GIF 与评测数据不可能对不上。

数据一致性由生成脚本内部的断言保证（重跑生成器即校验）：

```
T8_complex_notify  stopped_reason == completed   ← 帧 6 是这么说的
                  有 retried=True 的记录          ← 帧 4 是这么说的
N8_confirm_before_report  stopped_reason == budget_tokens  ← 帧 7/9
                          budget.used.tokens > 10000       ← 帧 7
```

## 二、演示哪两段（都是 01-02 的真实跑次）

| 段 | 任务 | 展示什么 |
| --- | --- | --- |
| ① | `T8_complex_notify` | 四工具里前三个成功 → notify 失败 → **代码自己重试一次**（`retried=True`）→ 仍失败 → 判为"收尾失败"，保留 completed 但如实说明哪一步没成 |
| ② | `N8_confirm_before_report` | confirm 类连查三轮 → `budget_tokens` 触顶（14257 > 10000）→ **优雅退出**（已完成部分 + 未完成原因 + 建议下一步）|

另有一帧专门说明**超时走的是同一条路径**（同样的 `retryable=true` →
代码退避重试），并给出可复现的注入写法：

```python
FaultInjector(base, failures={1: "timeout", 2: "timeout"})
```

注入只在 `tests/fault_injection.py`，**不接生产路径**——这一句写在帧里，
也写在代码注释里。

## 三、帧清单

| 帧 | 内容 |
| --- | --- |
| 1 | 开场：演示两个机制（代码重试 / 结构化降级）|
| 2 | 超时走同一条路（附注入写法）|
| 3 | T8 前置工具全部成功（去重计数，如实显示 query_metrics_sql ×2）|
| 4 | notify 第一次失败 → 错误信封带 `retryable=true` |
| 5 | **retry 节点：模型没参与，代码自己退避重试** |
| 6 | 重试仍失败 → 判"只是收尾失败"，保留 completed（附理由）|
| 7 | T8 真实最终答案 |
| 8 | N8 预算触顶（真实 token 数）|
| 9 | 优雅退出的三件事（已完成/原因/建议）|
| 10 | N8 真实退出文本 |
| 11 | 收尾：20 任务集的五个数字（**带口径**）|

## 四、参数与约束

| 项 | 值 | 理由 |
| --- | --- | --- |
| 分辨率 | 960×360 | 内容最长末行 y=291，360 留 70px 边距即可；再高就是留白 |
| 帧数 | 11 | 每帧 3s，末帧 4s，共 34s |
| 体积 | 199 KB | 远低于 3MB 目标 |
| 帧率等价 | 每帧 3s | 文字要读完，不能快 |
| 主题 | 深色 | 轨迹的白/绿/红对比明显 |

## 五、四条"不要做"（仍然有效）

1. **不加速到看不清** —— 价值是"能读到字"，不是"显得快"；
2. **不剪掉失败画面** —— 只展示成功，等于把"可靠性"演示成"运气好"；
3. **不用真实端点故障冒充注入** —— 那不可复现，说不清是机制还是环境；
4. **不在字幕里写"成功率 100%"** —— 区间是 70–100%，最好的一轮才是 100%。

第 4 条在帧 11 落实：写的是"70%–100% 区间（最好一轮 20/20）"，
并注明"降级计入失败；判据 01-01 冻结；可跑 verify_resume_numbers.py 复核"。

## 六、重跑

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\make_reliability_gif.py
# → 已生成 docs/agent-reliability.gif  (11 帧, 199 KB, 约 34s)
```

若轨迹文件缺失（`qa_runs_w11_r5/` 被 ignore），需要先重跑：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\run_eval_20.py --live --model trae/qwen-3.5 --outdir qa_runs_w11_r5
```

## 七、与旧版脚本的关系

`docs/gif-script-w11.md` 原先写的是"录屏脚本"。**实际实现改成了轨迹渲染**，
理由是可复现、可校验。旧文件保留，作为"如果将来真的需要录屏"的备用方案
（它里面那四条"不要做"是通用的）。
