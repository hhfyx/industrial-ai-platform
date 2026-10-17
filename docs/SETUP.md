# 环境搭建与开发流程

本文档记录 `industrial-ai-platform` 从零环境到运行验收的完整流程，包含**实测验证过的注意事项**与故障排查表。

> 本文档中的所有命令均在本项目的开发机（Windows 11 + PowerShell 5.1）上实际执行验证过。
> 其他环境（Linux / macOS）可参考，但 Windows 特有的问题不适用。

---

## 目录

- [0. 环境基线](#0-环境基线)
- [1. 安装工具链](#1-安装工具链)
- [2. GitHub 网络配置](#2-github-网络配置)
- [3. 创建项目](#3-创建项目)
- [4. 添加依赖](#4-添加依赖)
- [5. 项目结构](#5-项目结构)
- [6. 配置文件](#6-配置文件)
- [7. 质量门禁](#7-质量门禁)
- [8. Git 初始化与提交](#8-git-初始化与提交)
- [9. 运行与验收](#9-运行与验收)
- [10. 推送到 GitHub](#10-推送到-github)
- [11. 日常开发循环](#11-日常开发循环)
- [注意事项汇总](#注意事项汇总)
- [故障速查表](#故障速查表)

---

## 0. 环境基线

开工前先确认现状，避免在错误前提下工作。

```powershell
uv --version
git --version
(Get-ExecutionPolicy -Scope CurrentUser).ToString()
```

本项目开发机的基线：

| 项 | 版本 / 值 |
|---|---|
| Windows | 11 (10.0.26100) |
| PowerShell | 5.1.26100.9168 |
| Python | 3.11.2（另一套 Anaconda 3.9.23 在 PATH 里，需规避） |
| uv | 0.13.1 |
| git | 2.55.0.windows.1（MinGit 便携版） |
| 执行策略 | `RemoteSigned`（CurrentUser 作用域） |

工具安装位置：

| 工具 | 路径 |
|---|---|
| uv | `%USERPROFILE%\.local\bin\uv.exe` |
| git | `%USERPROFILE%\Tools\MinGit\cmd\git.exe` |

> **注意**：如果 `uv` 或 `git` 提示「不是内部或外部命令」，说明 PATH 还没被当前终端加载。
> **关掉终端重新开一个**。仍然不行就用完整路径：`& "$env:USERPROFILE\.local\bin\uv.exe"`。

---

## 1. 安装工具链

> 第 1 节是一次性准备工作。本开发机已完成，**新机器或重装时需要重新执行**。

### 1.1 放开 PowerShell 执行策略

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**为什么必须做**：uv 官方安装脚本内部有执行策略检查，只接受 `Unrestricted` / `RemoteSigned` / `Bypass`。
默认的 `Restricted` 会让安装直接中止：

```
Error: PowerShell requires an execution policy in [Unrestricted, RemoteSigned, Bypass] to run uv.
```

**用 `-Scope CurrentUser`**：不需要管理员权限。用 `LocalMachine` 会弹 UAC。

### 1.2 安装 uv

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

预期输出：

```
downloading uv 0.13.1 (x86_64-pc-windows-msvc)
installing to C:\Users\<user>\.local\bin
  uv.exe
  uvx.exe
  uvw.exe
everything's installed!
```

实测约 4 秒。

> **注意**：安装脚本会自动修改用户 PATH。
> 如果安装到临时目录（例如调试时指定了 `UV_INSTALL_DIR`），那个临时目录会**永久留在 PATH 里**，删除后变成死条目。
> 正式安装一律用默认目录。

### 1.3 安装 Git（便携版，免管理员）

如果系统没有 git，且 `github.com` 不通导致 winget 无法下载安装包，用 MinGit 便携版：

```powershell
$g = "$env:USERPROFILE\Tools\MinGit"
New-Item -ItemType Directory -Path $g -Force | Out-Null
curl.exe -sSL -o "$env:TEMP\mingit.zip" `
  https://registry.npmmirror.com/-/binary/git-for-windows/v2.55.0.windows.1/MinGit-2.55.0-64-bit.zip
Expand-Archive "$env:TEMP\mingit.zip" -DestinationPath $g -Force
Remove-Item "$env:TEMP\mingit.zip" -Force
```

实测 38.8 MB 约 3.4 秒。然后把 `$g\cmd` 加入用户 PATH：

```powershell
$userPath = [Environment]::GetEnvironmentVariable('Path','User')
[Environment]::SetEnvironmentVariable('Path', "$userPath;$g\cmd", 'User')
```

> **注意**：MinGit 是精简版，**没有 bash、没有 man**。纯 git 操作够用；
> 需要完整 Git Bash 则装完整版（需管理员权限）。

---

## 2. GitHub 网络配置

### 2.1 问题现象

在部分网络环境下，`github.com` 的 DNS 会解析到无法连通的 IP（本项目开发机上是 `20.205.243.166`），
表现为 `git push` / `git clone` 超时或连接被重置：

```
fatal: unable to access 'https://github.com/...': Failed to connect to github.com:443
```

但同一个域名的**其他 IP 是通的**。

### 2.2 解决方案：`http.curloptResolve`

让 git 底层的 libcurl **忽略 DNS 解析，直连指定 IP**。TLS 证书仍按域名校验，安全性不受影响。

```powershell
git config --global http.curloptResolve "github.com:443:140.82.113.4"
```

验证：

```powershell
git ls-remote --heads https://github.com/astral-sh/uv.git HEAD
```

从超时变为约 1.5 秒成功。

### 2.3 IP 会变化，需要定期探活

实测中 `140.82.112.4` 前一天可用、次日失效。**推送超时时先测 IP，不要先怀疑账号或 token**。

```powershell
foreach ($ip in '140.82.113.4','140.82.114.4','140.82.112.3','140.82.112.4') {
  $ok = Test-NetConnection -ComputerName $ip -Port 443 -InformationLevel Quiet -WarningAction SilentlyContinue
  "$ip = $ok"
}
```

选择返回 `True` 的 IP 更新配置：

```powershell
git config --global http.curloptResolve "github.com:443:<可用IP>"
```

### 2.4 其余 git 全局配置

```powershell
git config --global init.defaultBranch main
git config --global user.name  "你的名字"
git config --global user.email "你的邮箱"
git config --global core.quotepath false    # 中文文件名正常显示
git config --global core.autocrlf true      # Windows 换行处理
```

### 2.5 各 GitHub 域名的可用性（供参考）

`github.com` 不通时，以下域名通常仍然可用：

| 域名 | 用途 | 本项目实测 |
|---|---|---|
| `api.github.com` | REST API、建仓库、读元数据 | 可用 |
| `codeload.github.com` | 下载 tarball | 可用 |
| `objects.githubusercontent.com` | Release 资源 | 可用 |
| `raw.githubusercontent.com` | raw 文件 | 时通时断 |
| `github.com` | 网页与 git 协议 | 需靠 `curloptResolve` 绕过 |

---

## 3. 创建项目

```powershell
$root = 'D:\git\industrial-ai-platform'
New-Item -ItemType Directory -Path $root -Force | Out-Null
Set-Location $root

uv init --no-package --name industrial-ai-platform .
uv python pin 3.11
```

### 3.1 为什么用 `--no-package`

| | `--package` | `--no-package` |
|---|---|---|
| 目录结构 | 嵌套 `src/app/__init__.py` | 扁平，根目录 |
| 是否生成 `main.py` | 否 | 是 |
| `[project.scripts]` | `app = "app:main"` | 无 |
| 能否 `uvicorn app.main:app` | 不能 | 可以 |

本项目需要 `uvicorn app.main:app` 的模块路径，因此必须用 `--no-package`。
用 `--package` 会导致 `ModuleNotFoundError: No module named 'app.main'`。

### 3.2 为什么必须 `uv python pin`

uv 会自动挑选系统中最合适的 Python。本开发机 PATH 里有 Anaconda 的 3.9.23，
如果不 pin，`.python-version` 会被写成 `3.9`，与 `requires-python = ">=3.11"` 冲突。

`uv python list` 可以查看所有可用解释器。

---

## 4. 添加依赖

```powershell
# 运行时依赖
uv add fastapi "uvicorn[standard]" pydantic pydantic-settings

# 开发依赖
uv add --dev pytest pytest-cov httpx ruff pre-commit
```

实测：运行时依赖约 2.3 秒，开发依赖约 3 秒。

> **注意**：`uvicorn[standard]` 的方括号**必须加引号**。
> PowerShell 会把 `[standard]` 当作通配符或数组语法解析，不加引号会失败。

> **注意**：`httpx` 是 FastAPI `TestClient` 的必需依赖。
> 缺少它会导致 `from fastapi.testclient import TestClient` 抛 ImportError。

### 4.1 依赖下载慢时切换镜像

直连 PyPI 实测约 97 KB/s。需要加速时临时切换（只对当前终端生效，不改全局配置）：

```powershell
$env:UV_DEFAULT_INDEX = "https://mirrors.aliyun.com/pypi/simple/"
```

---

## 5. 项目结构

```
industrial-ai-platform/
├── app/
│   ├── __init__.py
│   └── main.py            # FastAPI 应用入口
├── tests/
│   ├── __init__.py
│   └── test_health.py
├── docs/
│   └── SETUP.md           # 本文档
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock                # 必须提交，保证依赖可复现
```

创建包目录：

```powershell
New-Item -ItemType Directory -Path "app","tests" -Force | Out-Null
New-Item -ItemType File -Path "app\__init__.py","tests\__init__.py" -Force | Out-Null
```

> **注意**：`__init__.py` 不要省略。
> 虽然 Python 3.3+ 支持命名空间包，但 pytest 与 ruff 的 `src` 解析在缺少该文件时行为不一致。

---

## 6. 配置文件

### 6.1 `app/main.py`

```python
"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="Industrial AI Platform",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe used by deployment and CI."""
    return {"status": "ok"}
```

> **注意**：根路径 `/` 未定义是**有意为之**。
> 浏览器访问 `127.0.0.1:8000` 会请求 `/` 和 `/favicon.ico`，日志中出现两条 404 属于正常现象。
> 查看接口请访问 `/docs`。

### 6.2 `pyproject.toml`

在 `uv init` 生成内容的基础上追加：

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["app", "tests"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "SIM",  # flake8-simplify
]
ignore = [
    "E501", # 行宽由 formatter 负责
]

[tool.ruff.lint.isort]
known-first-party = ["app"]

[tool.ruff.format]
quote-style = "double"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.coverage.run]
source = ["app"]
branch = true
```

> **注意**：`ignore = ["E501"]` 不是放宽标准。
> 行宽已经由 `ruff format` 统一处理，lint 再报一遍属于重复检查。
> 但 `line-length` 必须设置，否则 formatter 使用默认的 88。

### 6.3 `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.17.1
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=1024]
      - id: check-merge-conflict
```

> **注意**：**没有这个文件，`pre-commit install` 只会安装一个空钩子**，
> 提交时所有检查显示 `(no files to check) Skipped`，实际上什么也没验证。
> 这是最容易自欺欺人的地方。

> **注意**：`rev` 必须与本地 ruff 版本一致（`uv run ruff --version` 确认）。
> 版本不一致会出现「本地通过、CI 失败」的诡异现象。

### 6.4 `.gitattributes`

```
* text=auto eol=lf
*.png binary
*.jpg binary
*.pptx binary
*.docx binary
*.xlsx binary
```

> **注意**：不加此文件，每次 `git add` 都会输出大量 CRLF 警告。

### 6.5 `.gitignore`

至少需要包含：

```
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
.coverage.*
htmlcov/
coverage.xml
dist/
build/
*.egg-info/
.env
.env.*
.idea/
.vscode/
*.log
```

> **注意**：`.venv/` 必须排除。其中包含数千个文件，提交后仓库体积会失控。
> **注意**：`.env` 必须排除，同时建议提交一份 `.env.example` 作为模板。

### 6.6 `tests/test_health.py`

```python
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_schema_is_available() -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["version"] == "0.1.0"
```

---

## 7. 质量门禁

按顺序执行，每步都要看到明确的成功标志：

```powershell
uv run ruff format .        # → "N files left unchanged" 或 "N files reformatted"
uv run ruff check --fix .   # → "All checks passed!"
uv run pytest -q            # → "2 passed"
```

> **注意**：**先 format 再 check**。
> 顺序反了会先报格式问题，随后又被 formatter 修掉，浪费一轮。

> **注意**：`ruff check` 报 `Found N errors [*]` 表示可自动修复，加 `--fix` 后重跑确认干净。

> **注意**：pytest 输出 `StarletteDeprecationWarning: install httpx2` 属于上游过渡期警告，不影响运行。

---

## 8. Git 初始化与提交

```powershell
git init
uv run pre-commit install
git add -A
git status --short          # 必须检查暂存内容
git commit -m "chore: init project with uv and ruff"
```

> **注意**：`pre-commit install` **必须在 `git init` 之后**执行。
> 顺序反了会报：
> ```
> FatalError: git failed. Is it installed, and are you in a Git repository directory?
> ```

> **注意**：`git add -A` 之后**务必检查 `git status --short`**。
> 确认 `.venv` 没有被加入。本项目首次提交应为 12 个文件；出现数千个文件说明 `.gitignore` 未生效。

> **注意**：首次 commit 会触发 pre-commit 下载钩子环境，
> 看到 `[INFO] Installing environment for ...` 属于正常现象，耗时几十秒到几分钟，**不要中断**。

> **注意**：PowerShell 会把 git 写入 stderr 的内容包裹成红色 `NativeCommandError`，
> 并附带误导性的 `[exit code: 1]`，**但命令可能已经成功**。
> 判断成败应查看实际产物（commit 是否生成），而非退出码颜色。

---

## 9. 运行与验收

**终端 A** —— 启动服务：

```powershell
uv run uvicorn app.main:app --reload
```

预期输出：

```
INFO:     Will watch for changes in these directories: ['D:\git\industrial-ai-platform']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using WatchFiles
INFO:     Application startup complete.
```

**终端 B** —— 验收：

```powershell
curl.exe http://127.0.0.1:8000/health
```

预期输出：

```json
{"status":"ok"}
```

浏览器打开 <http://127.0.0.1:8000/docs> 查看 Swagger UI。

> **注意**：**必须写 `curl.exe`，不能写 `curl`**。
> PowerShell 中 `curl` 是 `Invoke-WebRequest` 的别名，行为完全不同（不显示响应体、解析 HTML 等）。

> **注意**：终端 A 不能关闭，`--reload` 依赖该进程常驻。
> 修改代码后 WatchFiles 会自动重启。

> **注意**：端口被占用时先检查：
> ```powershell
> Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
> ```

---

## 10. 推送到 GitHub

### 10.1 创建远程仓库

**方式一：网页**（推荐）

打开 <https://github.com/new>，填写仓库名，选择 Public，
**不要勾选** README / .gitignore / License（会与本地文件冲突）。

**方式二：REST API**（`github.com` 打不开时使用）

```powershell
$tok = 'github_pat_xxxxx'

@{
  name        = 'industrial-ai-platform'
  description = 'Industrial AI platform backend built with FastAPI'
  private     = $false
  auto_init   = $false
} | ConvertTo-Json | Set-Content "$env:TEMP\repo.json" -Encoding ascii

curl.exe -sS -X POST `
  -H "Authorization: Bearer $tok" `
  -H "Accept: application/vnd.github+json" `
  -H "User-Agent: dsh-agent" `
  -H "Content-Type: application/json" `
  --data-binary "@$env:TEMP\repo.json" `
  https://api.github.com/user/repos
```

成功返回 `HTTP 201` 及仓库信息。

> **注意**：JSON 必须**写入文件后用 `--data-binary @file`** 传递。
> PowerShell 会吃掉内联 JSON 字符串的引号，导致 API 返回：
> ```json
> {"message":"Problems parsing JSON","status":"400"}
> ```

### 10.2 Token 权限要求

| Token 类型 | 所需配置 |
|---|---|
| **classic**（推荐） | 勾选 `repo` 一项即可 |
| **fine-grained** | `Repository access` 必须选 **All repositories**；`Repository permissions` 中 `Contents: Read and write`（推送）+ `Administration: Read and write`（建仓） |

**fine-grained 最常见的两个错误**：

1. `Repository access` 选了 `Public repositories` —— 这是**只读**，必须选 `All repositories`
2. 只给到 `Metadata: Read` —— 会返回 `403 Resource not accessible by personal access token`

### 10.3 配置 remote 并推送

```powershell
git remote add origin https://github.com/<user>/industrial-ai-platform.git
git push -u origin main
```

首次推送会触发浏览器认证：

```
info: please complete authentication in your browser...
```

在浏览器授权后，凭据会存入 Windows 凭据管理器，后续无需重复输入。

> **注意**：出现 `Invalid username or token` 通常是**凭据缓存**问题，不是网络问题。
> 删除旧凭据后重试：
> ```powershell
> cmdkey /delete:LegacyGeneric:target=git:https://github.com
> git push -f origin main
> ```

> **注意**：`-f` 仅在 `--amend` 或 `rebase` 重写历史后需要。
> **普通提交永远不要 `-f`**，多人协作时会覆盖他人提交。

### 10.4 验证推送结果

```powershell
git log --oneline -1
git ls-remote --heads origin main
```

两个 hash 必须完全一致。

> **注意**：**浏览器打不开 `github.com` 不代表推送失败**。
> `http.curloptResolve` 只影响 git，浏览器使用独立 DNS。
> 可通过 API 验证仓库状态：
> ```powershell
> curl.exe -sS -H "Authorization: Bearer $tok" -H "User-Agent: dsh-agent" `
>   https://api.github.com/repos/<user>/industrial-ai-platform
> ```
> 确认 `"visibility": "public"` 与 `"default_branch": "main"`。

---

## 11. 日常开发循环

```powershell
cd D:\git\industrial-ai-platform

# 1. 同步远程（多人协作时）
git pull

# 2. 编码，服务自动重载，浏览器验证

# 3. 提交前手动跑一遍门禁
uv run ruff format .
uv run ruff check .
uv run pytest

# 4. 提交（pre-commit 自动执行）
git add -A
git commit -m "feat: add device registration endpoint"

# 5. 推送
git push
```

### 提交信息约定

采用 [Conventional Commits](https://www.conventionalcommits.org/)：

| 前缀 | 用途 |
|---|---|
| `feat:` | 新功能 |
| `fix:` | 缺陷修复 |
| `docs:` | 仅文档变更 |
| `refactor:` | 重构（不改变行为） |
| `test:` | 仅测试变更 |
| `chore:` | 构建、工具、依赖变更 |
| `perf:` | 性能优化 |
| `ci:` | CI 配置变更 |

---

## 注意事项汇总

### A. 高频错误（按发生概率排序）

1. **PowerShell 中 `curl` 不是 curl**
   ```powershell
   curl http://127.0.0.1:8000/health      # 错误：走 Invoke-WebRequest 别名
   curl.exe http://127.0.0.1:8000/health  # 正确
   ```

2. **`uv init --package` 结构不符合预期** —— 不生成 `main.py`，`uvicorn app.main:app` 无法工作。用 `--no-package`。

3. **`pre-commit install` 在 `git init` 之前执行** —— 直接报 `FatalError: git failed`。

4. **缺少 `.pre-commit-config.yaml` 就 install** —— 装的是空钩子，提交时全部 `Skipped`。

5. **更换 token 后未清理凭据缓存** —— 持续报 `Invalid username or token`。

### B. 网络（特定网络环境适用）

- `github.com` 域名可能解析到不可达 IP，但其他 IP 可达 → 使用 `http.curloptResolve`
- **IP 会变化**，推送超时先测 IP，再怀疑账号
- 命令行走通 ≠ 浏览器可用（DNS 独立），要浏览器访问需修改 hosts（需管理员）
- `api.github.com` / `codeload.github.com` / `objects.githubusercontent.com` 通常比 `github.com` 更稳定

### C. 路径与沙箱

- **项目路径使用纯 ASCII**。中文路径可能触发文件系统授权异常
  （错误形如 `SetNamedSecurityInfoW failed (Win32 5)`，且 `icacls` 显示路径乱码）
- **不要在已有 git 仓库的子目录中执行 `git init`**，会形成嵌套仓库导致命令行为异常

### D. Python 与 uv

- `uv python pin` 不可省略，否则可能被系统中的旧版 Python 接管
- `uvicorn[standard]` 的方括号需要引号
- `.venv` 绝不提交
- `uv.lock` **必须提交**，保证依赖可复现

### E. 安全

- **Token 不要粘贴到任何对话、日志或截图**中
- 更安全的认证方式：让 Git Credential Manager 弹窗输入，或用 SSH key
- `.env` 必须 gitignore，同时提交 `.env.example` 作为模板
- Token 泄露后立即在 <https://github.com/settings/tokens> 撤销

### F. PowerShell 5.1 特性

- 外部命令写入 stderr 的内容会被包装为 `NativeCommandError`，并附带误导性的 `[exit code: 1]`，
  但命令可能已经成功 —— **判断依据看产物，不看退出码**
- 内联 JSON 会被解析器破坏引号 —— 改用 `--data-binary @file`
- `Set-Content` 默认使用 ANSI 编码；写 JSON 给 API 时用 `-Encoding ascii`，
  写含中文的文本时注意 BOM 问题

---

## 故障速查表

| 现象 | 根因 | 解法 |
|---|---|---|
| `SetNamedSecurityInfoW failed (Win32 5)` | 项目路径含非 ASCII 字符 | 改用纯英文路径 |
| `PowerShell requires an execution policy ...` | 执行策略为 `Restricted` | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `uv : 无法将"uv"项识别为 ...` | PATH 未在当前终端生效 | 新开终端，或用完整路径 |
| `ModuleNotFoundError: app.main` | 项目由 `--package` 创建 | 改为 `--no-package` 重建，或调整 uvicorn 模块路径 |
| `.python-version` 为 3.9 | 未执行 `uv python pin` | `uv python pin 3.11` |
| `FatalError: git failed`（pre-commit） | 未执行 `git init` | 先 `git init`，再 `pre-commit install` |
| pre-commit 全部 `Skipped` | 缺少 `.pre-commit-config.yaml` | 创建配置文件 |
| `Problems parsing JSON`（400） | 内联 JSON 引号被破坏 | 改用 `--data-binary @file` |
| `Resource not accessible by personal access token`（403） | token 权限不足 | classic 勾 `repo`；fine-grained 选 `All repositories` |
| `Invalid username or token` | 凭据缓存失效 | `cmdkey /delete:LegacyGeneric:target=git:https://github.com` |
| `Permission to <repo> denied`（403） | token 无写权限 | 同上 |
| `push` 超时 | `github.com` IP 不可达 | 切换 `http.curloptResolve` 的 IP |
| 浏览器无法打开 `github.com` | 浏览器 DNS 解析到不可达 IP | 修改 hosts（需管理员）或更换网络 |
| 根路径 `/` 返回 404 | 未定义根路由 | 属正常现象，接口文档在 `/docs` |
| `curl` 不显示响应体 | 实际调用的是 `Invoke-WebRequest` | 使用 `curl.exe` |
| `git add` 输出大量 CRLF 警告 | 缺少 `.gitattributes` | 添加 `* text=auto eol=lf` |

---

## 当前项目状态

| 项 | 值 |
|---|---|
| 本地路径 | `D:\git\industrial-ai-platform` |
| 远程仓库 | <https://github.com/hhfyx/industrial-ai-platform> |
| Python | 3.11 |
| 启动命令 | `uv run uvicorn app.main:app --reload` |
| 验收端点 | `/health` → `{"status":"ok"}` |
| 接口文档 | <http://127.0.0.1:8000/docs> |
