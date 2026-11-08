# GIF 录制脚本：interrupt → 批准 → 继续（30 秒）

> 周日 11:30–12:00 的验收产物录制指南。录成 `docs/agent-hitl.gif`，
> 替换占位引用。屏幕录制用 Windows 自带「截图工具 → 录制」或
> `ffmpeg -f gdigrab`（有 ffmpeg 时）。

## 准备（两个终端并排，字体调大到 16pt+）

终端 A（图执行进程）：

```powershell
cd D:\git\网页\industrial-ai-platform
.\.venv\Scripts\python.exe -X utf8 ..\learn\llm\w10_hitl.py write_report approve
```

跑完 A 后立即在终端 B（模拟「人几个小时后回来」——录制约等 5 秒即可）：

```powershell
# B 只是展示查询命令（不真正执行，人工批复就是 A 的恢复阶段）
.\.venv\Scripts\python.exe -X utf8 scripts\query_graph_state.py --db ..\learn\llm\w10_hitl.sqlite --thread "hitl::task-approve"
```

## 分镜（30 秒）

| 时间 | 画面 | 字幕要点 |
| --- | --- | --- |
| 0–8s | 终端 A：invoke 输出 payload（question/draft/will_do） | 「节点内 interrupt() 挂起，进程直接返回」 |
| 8–15s | 终端 B：query_graph_state 输出 waiting_human=true | 「状态已落盘，人几小时后批都行」 |
| 15–25s | 终端 A：Command(resume="approve") 后的 final_answer + 副作用账本 | 「恢复：副作用账本 count=1，只执行一次」 |
| 25–30s | 并排显示 w10_hitl.py 的 act_node 源码 | 「副作用全部在恢复判定之后——重执行语义的正确姿势」 |

## ffmpeg 一条命令（可选，无 GUI 录制）

```powershell
ffmpeg -f gdigrab -framerate 10 -t 30 -i desktop -vf "fps=10,scale=800:-1:flags=lanczos" -loop 0 docs/agent-hitl.gif
```

> 录制期间保持只有这两个终端在桌面最前，避免无关窗口入镜。
