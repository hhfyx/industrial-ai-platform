# W1 总结：把一个"能跑"的服务做到"能验收、能换、能装"

> 周期：2026-10-17 ～ 10-22（6 天，10/17 周六开工）
> 产物：**39 个测试全绿、覆盖率 99%**、一份冻结的 v1 契约、三阶段 Docker 镜像 **157 MB**
> 关键提交：`a564c71` → `bb7b753`（共 5 个提交，`v0.1.0` 打在 `bb7b753`）

---

## 一句话总结

W1 没有做任何"AI"的部分 —— 整周只有规则版抽取器，没有一次模型调用。它做的是三件
**之后每一步都要靠它撑着**的地基：

1. **先冻结契约再写实现** —— `docs/api-spec.md` 是手写的、不是代码生成的，实现可以整个换掉
2. **把错误形状焊死** —— 所有非 2xx 都是同一个 4 字段体，不依赖框架默认行为
3. **把部署做到底** —— 非 root、健康探针、一条命令起停、镜像压进 200 MB

**这一周最反直觉的一条**：默认配置下（朴素双阶段构建）镜像 242 MB，在基础镜像上删掉
20 MB 文件体积**一个字节都没降** —— 因为它只是叠了一层删除标记，而不是真的变小。

---

## 一、按时间线的实际交付

| 日期 | 提交 | 主题 | 关键产物 |
| --- | --- | --- | --- |
| 10/17 | `a564c71` | 工程骨架 | `pyproject.toml`、`uv.lock`、ruff + pre-commit、`.gitattributes` |
| 10/17 | `b8f1069` | 环境文档 | `docs/SETUP.md`（561 行，含 11 节 + 故障速查表） |
| 10/21 | `349d54a` | 契约 + 规则版实现 | `docs/api-spec.md`、`app/` 全套、39 个测试 |
| 10/22 | `d83edbe` | 容器化 | `Dockerfile`（双阶段）、`.dockerignore`、`docker-compose.yml`、`docker/healthcheck.py` |
| 10/22 | `bb7b753` | 镜像瘦身 | 三段式（`builder` → `slim-base` → `scratch`），**242 → 211 → 157 MB** |

**代码量**（`v0.1.0` 实测）：Python + 基础设施共 **1398 行**，其中应用 727 行、测试 449 行。
断言：**39 个测试通过，`--cov=app` 覆盖率 99%**（266 行语句，仅 1 行未覆盖）。

---

## 二、三个"必答题"及其实测答案

### 2.1 契约为什么要在写实现之前冻结？—— 因为 W2 要整个换掉实现

`docs/api-spec.md` 不是从代码导出的，是**先手写、再照着写代码**。它的价值在 W2 才被验证：

| 换过什么 | `api-spec.md` 改了吗 |
| --- | --- |
| 规则版 → LLM 版抽取器 | ❌ 没改 |
| 加 `structured_call` 底座 | ❌ 没改 |
| prompt 模板 v1 → v2 → v1.1 | ❌ 没改 |
| 新增可选 `degraded`（v1.1） | ✅ 唯独改了这一次，且向后兼容 |

> `degraded` **只在 `true` 时出现**，正常结果里没有这个键 —— 所以老调用方不读它也不会坏。
> 这是 W1 定下的"新增只能是可选字段"规则（api-spec §4 规则 2）在 W2 第一次被真正用上。

**W1 里已经预埋的三条纪律**（写进 docstring，不是写给 reviewer 看的）：

- `app/services/extractor.py` **不 import fastapi / starlette**（已核：import 段只有
  `asyncio` / `re` / `time` / `dataclasses` / `datetime` / `typing`）—— 抽错了层，W2 就没法替换
- 抽取器返回 `str | None`，**不做类型转换** —— 转成 float/date 是调用方的事，
  否则 `data` 的形态会随实现版本漂移
- `extract()` 是 `async` 且**故意 `sleep(100ms)`** —— 抢在 W2 之前把"抽取是 IO"钉进签名，
  换个实现时 `await` 的位置不用动

### 2.2 所有非 2xx 真的都是同一个形状吗？—— 是，但框架默认给了三套

FastAPI + Starlette 默认会吐出至少三种不同形状：

| 来源 | 默认形状 |
| --- | --- |
| `RequestValidationError` | `{"detail": [...]}` |
| `StarletteHTTPException`（404/405） | `{"detail": "..."}` |
| 未捕获 `Exception` | 纯文本 `Internal Server Error` |

W1 的做法是用 **5 个 handler** 全部焊到同一个 `ErrorResponse`：

```text
AppError                  → 带稳定 code，业务层抛它
RequestValidationError    → 422 INVALID_INPUT / 400 INVALID_JSON（按错误类型分流）
StarletteHTTPException    → 兜住 404/405，同样进契约体
ResponseValidationError   → 500 RESPONSE_CONTRACT_VIOLATION
Exception（最外层）        → 500 INTERNAL_ERROR，对外只说一句话
```

**最容易被漏掉的一点**：`Exception` handler 挂在 `ServerErrorMiddleware`，**绕开了所有用户中间件**，
所以 `X-Request-ID` 必须由它自己写 —— 中间件的"后半"在这种情况下根本不会执行。
统一出口 `_error_response()` 让 body 形状和响应头只定义一次。

**为什么不用 `HTTPException`**：它的 `detail` 只能塞字符串，而契约要求 `code`
可被程序判断（`INVALID_INPUT` vs `INTERNAL_ERROR`）。

### 2.3 镜像 242 MB 怎么压到 157 MB？—— 因为 `rm` 根本不减小镜像

这是本周最反直觉的一条，值得单独记：

| 阶段 | 体积 | 关键动作 |
| --- | --- | --- |
| 朴素双阶段 | **242 MB** | venv 43 MB（uvloop 的 `.so` 里躺着 12 MB 调试符号 + 编译好的 `.pyc` 7.6 MB） |
| strip + 在 base 上删 20 MB | **211 MB** | **只降了 venv 的 27 MB**；删 base 一点没少 |
| 重打包 rootfs（squash） | **157 MB** | 从 `scratch` 接住删干净的整个文件系统 |

> **镜像体积 = 各层之和**。在 `python:3.11-slim` 之上再写一层 `RUN rm`，被删的文件
> 仍然躺在下面那层里，`docker images` 不会少一个字节。要真降下来，得把删干净的
> 文件系统 `COPY --from=slim-base / /` 到 `FROM scratch` —— 这才是"重打包"。

配套的手段（按性价比排序）：

1. **strip `.so`** —— uvloop 的 `.so` 13.6 MB 里 12 MB 是符号表，strip 完只剩 1.6 MB
2. **不编译字节码** —— `UV_COMPILE_BYTECODE` 保持未设置（一旦设成 `1`，uv 会为 479 个文件
   生成 `.pyc`，多出 7.6 MB）
3. **删运行时用不到的东西** —— pip/setuptools/ensurepip、idlelib/distutils/tkinter/test、docs/locale、apt/perl 全家桶
4. **保留 `/var/lib/dpkg`（3 MB）** —— 这是包清单，安全扫描器要靠它列 CVE 范围，删了就成黑盒

**代价（必须写清楚）**：单层镜像的增量拉取粒度变粗；容器里没有 apt/pip/perl 了，
排障得用宿主机的同名镜像或临时起一个带工具的调试容器。

---

## 三、这周真正学到的四条机制级认知

### 3.1 装完 Docker Desktop ≠ 引擎已经起来

第一次 `docker run hello-world` 直接报：

```text
open //./pipe/docker_engine: The system cannot find the file specified
```

两个独立原因叠在一起：

1. **Desktop 后台没启动** —— `com.docker.service` 是 `Stopped`，进程列表里没有任何 docker 进程
2. **当前终端 PATH 是安装前缓存的** —— 即使 `docker.exe` 已存在，`docker` 也是"不是内部或外部命令"

修法：`Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"` 等引擎就绪；
PATH 问题用绝对路径 `C:\Program Files\Docker\Docker\resources\bin\docker.exe`。

> **连带踩到的第二个坑**：`docker-credential-desktop` 也在同一个 bin 目录，
> PATH 没补全时拉镜像是 `error getting credentials - err: exec: "docker-credential-desktop"`。
> 补 PATH 时要把 `resources\bin` 整个加进去，不是只加 `docker.exe`。

### 3.2 受限网络下，构建失败会伪装成四种完全不同的错误

| 报错 | 真实原因 |
| --- | --- |
| `failed to authorize ... auth.docker.io` | token 服务不通 |
| `dial tcp ...:443 ... no HTTPS proxy` | 引擎没配代理 |
| 拉基础镜像 `EOF` | 镜像加速源失效 |
| 拉 `docker/dockerfile:1` 超时 | `# syntax=` 指令带来的额外外网依赖 |

修法：给 Docker Desktop 配 `127.0.0.1:7897` 手动代理（Clash 默认口），**配完必须重启引擎**；
不用 BuildKit 新特性就把 `# syntax=` 那行删掉，少一次外网请求。

### 3.3 PowerShell 会把 `.env` 改坏，而且坏得很安静

`.env` 被 `Set-Content` 反复读写之后：中文注释变乱码，**`PORT=8000` 被并进了上一行注释**。
结果是容器里 `PORT` 为空，启动脚本的 `${PORT:-8000}` 静默兜底成 8000 ——
**表面上一切正常，实际 env_file 根本没送进去**。

正确写法（无 BOM UTF-8）：

```powershell
[System.IO.File]::WriteAllText($path, $content, (New-Object System.Text.UTF8Encoding($false)))
```

排查时注意引号 —— 双引号会被 PowerShell 先展开，看到的是宿主机的值：

```powershell
docker compose exec api sh -c 'echo PORT=[$PORT]'    # 单引号
```

### 3.4 端口映射不用 `${PORT}:${PORT}`

把宿主机映射也交给 `PORT`，会让"故意写错 `PORT=abc`"直接变成 **compose 配置解析报错** ——
你永远看不到容器里 uvicorn 的真实失败日志。写死 `8000:8000`、`PORT` 只留给容器内监听，
错误才会发生在容器里：

```text
Error: Invalid value for '--port': 'abc' is not a valid integer.
```

容器随之进入 `Restarting (2)` 循环（`restart: unless-stopped` 在拉它）。

---

## 四、契约的四个具体决定（都能被 W2 直接复用）

| 决定 | 理由 |
| --- | --- |
| **未知字段名返回 `null`，不报 422** | 字段名由调用方定义，服务端字典永远不完备。报错等于把"不支持"的责任推给调用方，正确做法是如实回答"没抽到" |
| **未知顶层键报 422** | 不静默忽略 —— 拼错的字段名会变成"数据莫名丢了一列" |
| **`fields` 重复报 422** | 重复项会让 `confidence` 的分母失真 |
| **金额返回原始字符串** | 服务端不做隐式转换，避免精度/币种歧义；要数值由调用方 `Decimal` 化 |

`confidence` 的定义也写死在契约里：命中比例，必抽字段缺失按比例打折 ——
它不会因为实现换成 LLM 就改变含义。

---

## 五、测试为什么能离线跑绿（这是 W1 埋的伏笔）

W1 的 39 个测试全部不连外部服务，靠的是一个当时还不起眼的选择：

- `tests/conftest.py` 只做一件事 —— 把 anyio 固定到 asyncio 后端。
  **为什么不引入 `pytest-asyncio`**：`anyio` 是 starlette 的既有依赖，自带 pytest 插件，
  "不引入新依赖"本身就是这条测试的意义 —— 契约的验证不该依赖额外的第三方包
- `tests/test_health.py` 里**版本号从 `app.__version__` 读，不写死** ——
  写死 `"0.1.0"` 的话每次升版本这条测试都会红一次，而它想守的是"openapi 能生成"

这两条在 W2 被放大成完整的离线策略（`FakeLLMClient` + `httpx2.MockTransport` + `autouse` fixture）。

---

## 六、已知局限（诚实地写出来）

| 局限 | 说明 |
| --- | --- |
| **规则版抽取器不是 AI** | W1 全周没有一次模型调用；`RuleExtractor` 只是占住接口，字段表覆盖范围写死在 api-spec §1.2 |
| **只有一个 demo 字段集** | 合同编号/日期/金额/甲方 + 兜底，表外字段走 `<字段名>：<值>` 正则 |
| **`.env` 里没有真实配置项** | W1 只用到 `PORT`；`APP_ENV` 当时就注明"当前代码尚未读取" |
| **镜像体积依赖基础镜像** | `python:3.11-slim` 本机实测 **187 MB**，它会随上游 Debian 更新漂移，硬写阈值会变成"定期假失败" |
| **提交历史的时间线有断层** | 10/17 到 10/21 之间隔了 4 天（`b8f1069` → `349d54a`），实际是文档周与实现周拆开做的 |
| **`app.__version__` 在 v0.1.0 tag 上已是 `0.2.0`** | `pyproject.toml` 同为 `0.2.0` —— 版本号在打 tag 前就被提前改了，`v0.1.0` 标签与包内版本号不一致 |

---

## 七、这周回答"你怎么保证部署是安全的"的完整答案

不能只说"加了 `USER`"，得能报出可执行的检查：

```powershell
docker compose up --build -d
curl.exe http://localhost:8000/health          # {"status":"ok"}
docker compose exec api id -u                  # 10001
docker inspect --format "{{.State.Health.Status}}" industrial-ai-api   # healthy
docker images industrial-ai:v0.1 --format "{{.Size}}"                  # 157MB
docker compose down
```

| 层次 | W1 的实际做法 |
| --- | --- |
| **进程权限** | `useradd --uid 10001 --gid 10001`，`USER 10001:10001`，固定 uid 方便宿主机排障时对得上 |
| **暴露面** | 镜像里没有 uv、没有 tests、没有 `.git`、没有 apt/pip/perl |
| **配置边界** | `.env` 同时被 `.gitignore` 和 `.dockerignore` 排除，只在运行时注入，不烤进镜像层 |
| **进程管理** | `exec uvicorn ...` 接管 PID 1，`docker stop` 的 SIGTERM 才能优雅退出 |
| **生产纪律** | CMD 里**禁止 `--reload`**（双倍进程 + 内存泄漏），bind `0.0.0.0` 才能被宿主机访问 |
| **探活** | 用标准库 `urllib.request` 打容器内 `127.0.0.1:$PORT/health`，不为了 curl 去装 libcurl |

> **健康检查为什么打容器内 `127.0.0.1`**：它要判断的是"容器里的进程活没活"，
> 不是"宿主机能不能连"。这两件事经常被混为一谈 —— 容器里的 `127.0.0.1` 是容器自己的回环网卡。

---

## 八、留给 W2 的三份遗产

| 遗产 | W2 怎么用上的 |
| --- | --- |
| `docs/api-spec.md`（v1，冻结） | 换 LLM 实现时一行没改；v1.1 加 `degraded` 是唯一一次变更 |
| `app/core/deps.py` 的 `lru_cache` 单例 | W2 把 `RuleExtractor()` 换成 `LlmExtractor(...)`，只改这一处装配 |
| 三阶段 Dockerfile | W2 加 `openai`/`lxml`/`pypdf` 等运行时依赖后，同一套 Dockerfile 构建出 **231 MB**（已实测） |

> **注意**：W2 的 231 MB 已超出 200 MB 目标，比 W1 多 **74 MB**。这不是 Dockerfile 退化了 ——
> 三段式结构一个都没改，涨的是运行时依赖本身（`openai` / `lxml` / `pypdf` / `tiktoken` /
> `python-docx` / `regex` / `markdown`）。要继续压，只有两条路：换更小的基础镜像
> （`-alpine` 需重新验证全部 wheel 的兼容性），或把 `lxml` 换成纯 Python 解析路径。
>
> **两个数字别混**：`industrial-ai:v0.1` 这个 tag 复用的是同一个名字。在 `v0.1.0`
> 提交上构建 = **157 MB**（本文实测口径）；在 W2 的 HEAD 上构建 = **231 MB**。
> 引用体积时必须同时说清"哪个提交 + 哪个 tag"，否则数字对不上。

---

## 附：产物索引

| 类别 | 路径 |
| --- | --- |
| 契约 | `docs/api-spec.md`（v1，4 字段响应 + 3 个错误码） |
| 环境文档 | `docs/SETUP.md`（561 行，11 节 + 注意事项 + 故障速查表） |
| 应用 | `app/`：`main.py` `api/v1/extract.py` `core/{deps,errors,middleware}.py` `schemas/extract.py` `services/extractor.py` |
| 部署 | `Dockerfile`（三段式）、`.dockerignore`、`docker-compose.yml`、`docker/healthcheck.py`、`.env.example` |
| 测试 | `tests/`（39 个，全离线；覆盖率 99%） |
| 镜像 | `industrial-ai:v0.1` —— **157 MB**，uid 10001，`/health` 与 `/api/v1/extract` 均 200 |
| 标签 | `v0.1.0` → `bb7b753` |
