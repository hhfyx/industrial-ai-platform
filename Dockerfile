# syntax=docker/dockerfile:1
#
# 多阶段构建：
#   builder  —— 装依赖（uv + uv.lock，可复现），装完即弃
#   runtime  —— 只拷产物（虚拟环境 + app 源码），不含编译器、不含 uv、不含 dev 依赖
#
# 为什么这样拆：
#   * 依赖层先于源码 COPY，改业务代码不会击穿 pip/uv 缓存；
#   * runtime 里没有 uv、没有 .git、没有 tests，镜像才可能停在 200 MB 以内；
#   * 进程以 uid 10001 运行，容器被攻破时拿不到 root。

# ---------------------------------------------------------------- builder
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# uv 只存在于 builder 层，最终镜像里没有它
RUN pip install --no-cache-dir uv

# 只拷锁文件：只要 pyproject/uv.lock 不变，这一层就命中缓存
COPY pyproject.toml uv.lock ./
# --frozen 严格按 uv.lock 装（不重新解析版本）
# --no-dev  只装 [project].dependencies，跳过 pytest/ruff 等开发依赖
# --no-install-project 工程本身不是可安装包，源码由 runtime 阶段直接 COPY
RUN uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------- runtime
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8000

WORKDIR /app

# 非 root 用户：固定 uid/gid，方便宿主机排障时对得上
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app

# 只拷两样东西：虚拟环境 + 应用源码
COPY --from=builder /app/.venv /app/.venv
COPY app ./app
COPY docker/healthcheck.py ./healthcheck.py

USER 10001:10001
EXPOSE 8000

# 健康检查用 python 标准库，避免为了 curl 再装一层 apt 包
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "/app/healthcheck.py"]

# 关键点：
#   --host 0.0.0.0 —— 容器内 127.0.0.1 是容器自己的回环，宿主机访问不到
#   --reload       —— 绝对不能加，生产镜像里它是内存泄漏 + 进程数翻倍
#   exec           —— 让 uvicorn 接管 PID 1，docker stop 的 SIGTERM 才能优雅退出
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
