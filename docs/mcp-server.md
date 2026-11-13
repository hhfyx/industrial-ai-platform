# W13 周四 · MCP：概念、最小 Server 与原始帧

> 实现位置：`scripts/mcp_min_server.py`（最小 Server，3 个工具）+
> `scripts/mcp_handwritten_client.py`（手写逐帧 client）。
> SDK：mcp 2.3.0。本文开篇就是任务书要的"原始帧留档"。

## 〇、Host / Client / Server 的分工（本周面试核心）

MCP 是 **Host ↔ Client ↔ Server** 的协议层：

```
┌─────────────────────────┐
│ Host（Claude Desktop / IDE / 你自己的 Agent）│
│  ┌──────────┐  ┌──────────┐                  │
│  │ 模型+决策 │  │ Client ×N │←── 1:1 ──┐    │
│  └──────────┘  └──────────┘          │    │
└──────────────────────────────────────┼────┘
                                       │ JSON-RPC 2.0（stdio / HTTP）
                            ┌──────────▼──────────┐
                            │ Server（能力提供方）  │
                            │ tools / resources / prompts │
                            └─────────────────────┘
```

- **Host**：跑模型、做决策（"要不要调这个工具"是模型+Host 的事）；
- **Client**：Host 里的连接器，1 个 Client 管 1 条到 Server 的连接，
  负责**能力发现**（tools/list）与**转发调用**（tools/call）；
- **Server**：暴露能力（可执行的工具、可读的数据、可复用的模板），
  **只执行，不决策**。

MCP 标准化的是**能力的发现与调用**（tools/list → tools/call 的帧格式），
**不负责模型怎么决定调用**——那是 Function Calling 的事。两者是
上下层关系不是竞争关系（见 §四对比表）。

## 一、原始帧留档（完整交互实录）

抓帧方式：`scripts/mcp_handwritten_client.py` 用最底层 stdio 读写
（不用 SDK 高层封装），每帧打印。完整输出存
`eval/results/mcp_frames.txt`。以下每帧都是实录摘录。

### 帧 ①：initialize（request）

客户端先声名自己：协议版本、有哪些能力、我叫什么。

```json
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {"roots": {"listChanged": true}},
    "clientInfo": {"name": "handwritten-client", "version": "0.1.0"}
  },
  "id": 1
}
```

### 帧 ②：initialize 响应（server 回 capabilities + serverInfo）

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "capabilities": {
      "prompts":  {"listChanged": false},
      "resources": {"listChanged": false, "subscribe": false},
      "tools":    {"listChanged": false}
    },
    "instructions": "工业 AI 平台的 MCP 最小 Server（W13 周四）。…",
    "protocolVersion": "2025-06-18",
    "serverInfo": {"name": "industrial-ai-mcp", "version": "0.1.0"}
  }
}
```

关键点：**capabilities 是双向协商的**——client 说"我支持 roots"，
server 回"我提供 tools/resources/prompts 三类能力"。周五暴露真工具时
Host 就是读这里的 `capabilities.tools` 决定要不要展示工具入口。

### 帧 ③：notifications/initialized（notification，无 id）

```json
{"jsonrpc": "2.0", "method": "notifications/initialized"}
```

**没有 id 字段 = 无响应**。JSON-RPC 2.0 的约定：notification 是
单向告知，server 不回帧。握手的第三步只是"客户端确认就绪"，
协议层对此保持沉默——手写 client 时最容易在这里等一个永远不会来的响应。

### 帧 ④：tools/list（request/response）

```json
{"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2}
```

响应里每个工具带 `name / description / inputSchema`（JSON Schema）：

```json
{
  "jsonrpc": "2.0", "id": 2,
  "result": {
    "tools": [
      {
        "name": "hello",
        "description": "向指定的人打招呼。参数 name 是要问候的名字。",
        "inputSchema": {
          "type": "object",
          "properties": {"name": {"title": "Name", "type": "string"}},
          "required": ["name"],
          "title": "helloArguments"
        },
        "outputSchema": {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"], "title": "helloOutput"}
      },
      {"name": "ping", "description": "存活探针：…（无参，inputSchema 无 properties）"},
      {"name": "add", "description": "两数相加。…（a: int, b: int）"}
    ]
  }
}
```

**这就是"能力发现"的实体**：Host 不需要读过任何文档，从这一帧就
知道 server 能做什么、每个工具要什么参数。对比 Function Calling：
schema 写在你的 Agent 代码里，别的 Host 看不见——见 §四。

### 帧 ⑤：tools/call hello（带参调用与结果）

```json
{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "hello", "arguments": {"name": "MCP"}}, "id": 3}
```

```json
{
  "jsonrpc": "2.0", "id": 3,
  "result": {
    "content": [{"type": "text", "text": "你好，MCP！来自 industrial-ai-mcp 的问候。"}],
    "isError": false,
    "structuredContent": {"result": "你好，MCP！…"}
  }
}
```

结果不是裸字符串而是 **content 数组**（type=text/image/resource）——
一个工具可以返回多模态多段内容。`structuredContent` 是 2.x 的
结构化输出（与 outputSchema 对应）。

### 帧 ⑥：tools/call ping（无参工具）

`{"name": "ping", "arguments": {}}` → `pong`。无参工具的
inputSchema 是没有 properties 的空 object。

### 帧 ⑦：参数校验失败——**不是 error 帧**（今天最重要的协议事实）

传 `{"a": "x", "b": 1}` 给 add（a 声明为 int）：

```json
{
  "jsonrpc": "2.0", "id": 6,
  "result": {
    "content": [{"type": "text", "text": "Error executing tool add: 1 validation error for addArguments\na\n  Input should be a valid integer, …"}],
    "isError": true
  }
}
```

**参数校验失败走的是 `result.isError=true`，不是 JSON-RPC `error` 帧**。
分界的含义：

| 层 | 用什么表达 | 谁产生 |
| --- | --- | --- |
| 协议层错误（方法不存在、server 内部崩了） | `error: {code, message}` | MCP 框架 |
| **业务层失败**（参数不合法、工具执行失败） | `result.isError=true` + content 里的人话说明 | 工具实现 |

对周五的启示：平台工具（query_metrics 等）的业务失败（无数据、
参数超限）应该返回 `isError=true` 带说明，让模型能读到原因自我修正；
只有真的"这个请求协议层就不对"才让框架抛 error。这与 W11 错误契约
（error_type 由代码判定）是同一思想在协议层的复现。

## 二、三大原语的区别（一句话版 + 实测）

| 原语 | 一句话 | list/call 帧 | 什么时候用 |
| --- | --- | --- | --- |
| **tools** | 工具**会做事**（有副作用或计算） | `tools/list` → `tools/call` | 查指标、抽字段、生成报告——周五的三个工具全是 |
| **resources** | 资源**是数据**（只读，uri 寻址） | `resources/list` → `resources/read` | 把 docs/*.md、usage_log 暴露成可读数据 |
| **prompts** | 提示**是模板**（预写好的可复用 prompt） | `prompts/list` → `prompts/get` | "周报生成"这类固定套路模板 |

本 Server 只实现了 tools（`capabilities` 里三个都声明了，但只有
tools 有注册项——声明能力 ≠ 提供能力，response 里 listChanged=false
如实反映）。resources 留给周五按需加。

## 三、SDK 2.x 的偏差记录（任务书 → 实际）

| 任务书写的（SDK 1.x） | 实际（mcp 2.3.0） | 说明 |
| --- | --- | --- |
| `from mcp.server.fastmcp import FastMCP` | `from mcp.server.mcpserver import MCPServer` | FastMCP 更名，装饰器 `@mcp.tool()` 不变 |
| `mcp.run(transport="stdio")` | 相同 | 不变 |

为什么偏差值得记录：**协议帧（§一）跨 SDK 版本稳定，Python 包装层
在快速演进**。面试讲 MCP 时讲帧和分工，不要背某个版本的 import 路径。

## 四、MCP vs Function Calling 对比表

| 维度 | Function Calling | MCP |
| --- | --- | --- |
| 谁定义 schema | **应用开发者**写在 Agent 代码里 | **Server 开发者**定义，经 tools/list 下发 |
| 谁发现能力 | 没有"发现"——Agent 启动时手里就有一份写死的工具清单 | **运行时发现**：initialize 协商 + tools/list 拉清单 |
| 谁执行 | Agent 进程内的函数（写 schema + 实现 + 注册，三件事都在自己代码里） | Server 进程（Agent 代码里只有一行 Client 连接） |
| 跨宿主可复用 | **不可**——只有你这个 Agent 能用 | **可**——任何支持 MCP 的 Host（Claude Desktop / IDE / 自建 Agent）连上就能发现并调用 |
| 传输方式 | 进程内函数调用（无传输概念） | stdio（本地子进程）/ streamable-http / SSE（远程） |
| 模型侧接口 | 相同：都是"模型看到 schema → 决定调用 → 收到结果" | 相同 |

**必答问题**（同一个"读文件"能力两种做法）：
- **FC**：在你的 Agent 代码里写 schema + 实现 + 注册，只有你这个 Agent 能用；
- **MCP**：写一个 Server，任何支持 MCP 的 Host 都能发现并调用。
  这是"MCP 让工具可复用"的真正含义。

但注意分寸（面试追问时）：MCP 没有"更好"，它解决的是**复用与解耦**
（工具提供方与模型宿主分离）；单 Agent 内部用 FC 更简单直接。
本平台两者共存：Agent 内部工具（W9 的工具注册表）走 FC 思路，
周五把它们暴露成 MCP Server 是**同一批能力的第二种发布形态**。

## 五、复跑

```powershell
# Server 单独起（stdio 挂起等输入，正常现象）
python -X utf8 scripts/mcp_min_server.py

# 手写 client 全帧交互（含抓帧）
python -X utf8 scripts/mcp_handwritten_client.py
# 完整帧存 eval/results/mcp_frames.txt
```
