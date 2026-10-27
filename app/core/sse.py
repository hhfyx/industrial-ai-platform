"""客户端断开的**可靠**检测。

这个问题值得单独一个文件，因为**标准写法在这里是失效的**，而失效得很安静。

## 为什么不能用 `await request.is_disconnected()`

任务要求里写的是 `await request.is_disconnected()`。实测（uvicorn + FastAPI 0.143）：

```python
async def gen():
    for i in range(40):
        print(await request.is_disconnected())   # ← 客户端已断开，这里仍是 False
        yield f"data: {...}\\n\\n"
```

客户端在第 3 块后关闭 socket，服务端日志显示：

```
[gen] i=0 is_disconnected=False
[gen] i=1 is_disconnected=False
[gen] i=2 is_disconnected=False
[gen] i=3 is_disconnected=False      ← 客户端已经走了
[gen] i=4 is_disconnected=False
...
[gen] i=9 is_disconnected=False      ← 一直到我们手动停服务
```

**它一次都没返回 True。** 生成器一直跑完，token 一直在烧 ——
而这段代码"看起来"完全符合任务要求。

原因：`is_disconnected()` 内部是"读完 `request._receive` 队列后判是否 disconnect"。
在 `StreamingResponse` 里，`receive` 通道已被 ASGI server 用于其他用途，
生成器去读时拿不到断开事件，于是永远返回 False。

## 可行方案：直接读 ASGI `receive`

```python
msg = await request.receive()          # 断开时得到 {'type': 'http.disconnect'}
```

实测这条路径**能**拿到断开事件：

```
[watch] receive: {'type': 'http.disconnect'}
[watch] DISCONNECT DETECTED
```

但直接在生成器主循环里 `await request.receive()` 有一个致命问题：
**它会阻塞**。客户端没断开时，这个 await 永远不返回，生成器就卡住了 ——
流式输出直接停摆。

所以正确做法是**另起一个后台任务**监听 receive，主循环只读一个标志位。
断开事件到达 → 后台任务置位 → 主循环在下一次检查时 break。

## 为什么用 `asyncio.Event` 而不是 bool

`Event.is_set()` 是非阻塞的，主循环可以随便查、开销近似为零。
用 bool 也行，但 Event 的语义更明确（"有人置位了"），而且
`wait_for(event.wait(), timeout)` 还能支持"等一会儿看看"的用法 ——
将来要做"断开后等 X 秒确认"时不用改结构。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from fastapi import Request

logger = logging.getLogger(__name__)


class DisconnectWatcher:
    """监听客户端断开，提供一个**非阻塞**的 `disconnected` 标志。

    用法：

        watcher = DisconnectWatcher(request)
        watcher.start()
        try:
            async for chunk in ...:
                if watcher.disconnected:
                    break          # ← 非阻塞，随便查
                yield chunk
        finally:
            await watcher.stop()

    `stop()` 必须调用（放 finally）：否则后台任务会一直挂在
    `request.receive()` 上，连接结束后才抛异常退出 —— 那之前它一直占着一个 task。
    """

    def __init__(self, request: Request) -> None:
        self._request = request
        self._event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def disconnected(self) -> bool:
        """**非阻塞**查询。主循环可以每个 chunk 都查，成本近似为零。"""
        return self._event.is_set()

    def start(self) -> None:
        """启动后台监听任务。重复调用是幂等的（不会起第二个）。"""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._watch(), name="sse-disconnect-watcher")

    async def _watch(self) -> None:
        """读 ASGI receive 直到拿到 `http.disconnect`。

        **异常全部吞掉**：监听断开这件事本身绝不能影响主流。
        `receive()` 在连接正常结束、被取消、或 server 关闭时都可能抛，
        那些情况下"断开"就是事实，置位即可。
        """
        try:
            while True:
                message: dict[str, Any] = await self._request.receive()
                if message.get("type") == "http.disconnect":
                    logger.debug("检测到客户端断开（http.disconnect）")
                    self._event.set()
                    return
                # http.request 之类的其他消息对我们无意义，继续读
        except asyncio.CancelledError:
            # stop() 取消了这个任务 —— 正常路径
            raise
        except Exception as exc:  # noqa: BLE001 - 监听器不能影响主流
            logger.debug("disconnect watcher 结束：%s: %s", type(exc).__name__, exc)
            self._event.set()

    async def stop(self) -> None:
        """取消后台任务并等它真正结束。

        `cancel()` 之后必须 `await`：不等的话任务可能在下一次事件循环迭代
        才真正退出，那期间它仍持有 `request.receive()`。
        `contextlib.suppress` 吞掉 CancelledError —— 那是我们自己取消的。
        """
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def wait(self, timeout: float) -> bool:
        """等最多 `timeout` 秒，返回是否已断开。

        用于"想等一会儿但不想忙等"的场景；主循环里用 `disconnected` 属性就够。
        """
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False


__all__ = ["DisconnectWatcher"]
