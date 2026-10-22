# 不写 `# syntax=docker/dockerfile:1`：本文件只用标准指令，不依赖 BuildKit frontend
# 的新特性；省掉这一行就少一次对 auth.docker.io 的拉取（受限网络下它会卡住构建）。
#
# 三阶段构建：
#   builder   —— 装依赖（uv + uv.lock，可复现）+ 剥符号表，装完即弃
#   slim-base —— 站在 python:3.11-slim 上，把运行时用不到的包和文件删掉
#   runtime   —— 从 scratch 接住 slim-base 的**整个文件系统**，再只拷产物
#
# 为什么 runtime 要从 scratch 重新接一遍（而不是直接复用 slim-base）：
#   镜像体积是各层之和。在 python:3.11-slim 之上再写一层 RUN rm，被删的文件
#   仍然留在下面的 base 层里，docker images 一分钱都不会少（实测：删掉 20 MB，
#   体积纹丝不动）。把删干净的文件系统 COPY 成一个新层，体积才真的降下来。

# ---------------------------------------------------------------- builder
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
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

# 剥符号表：uvloop 的 .so 里躺着 12 MB 调试符号，strip 之后只剩 1.6 MB。
# binutils 只装在这一层，最终镜像里没有。
RUN apt-get update \
 && apt-get install -y --no-install-recommends binutils \
 && find /app/.venv -name '*.so' -exec strip --strip-unneeded {} + \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------- slim-base
FROM python:3.11-slim AS slim-base

# 删掉运行时永远用不到的东西（一次性、同一层）：
#   * pip / setuptools / ensurepip —— 运行时从不 pip install；
#   * idlelib / distutils / tkinter / test —— 应用用不到；
#   * 文档、man、locale；
#   * apt / perl 全家桶 —— 容器里不会再装包。
# 保留 /var/lib/dpkg（3 MB）：它的包清单是安全扫描器列 CVE 的依据。
RUN set -eux; \
    rm -rf /usr/local/lib/python3.11/site-packages/pip \
           /usr/local/lib/python3.11/site-packages/pip-*.dist-info \
           /usr/local/lib/python3.11/site-packages/setuptools \
           /usr/local/lib/python3.11/site-packages/setuptools-*.dist-info \
           /usr/local/lib/python3.11/site-packages/packaging \
           /usr/local/lib/python3.11/site-packages/packaging-*.dist-info \
           /usr/local/lib/python3.11/site-packages/pkg_resources \
           /usr/local/lib/python3.11/site-packages/wheel \
           /usr/local/lib/python3.11/site-packages/wheel-*.dist-info \
           /usr/local/lib/python3.11/ensurepip \
           /usr/local/lib/python3.11/idlelib \
           /usr/local/lib/python3.11/distutils \
           /usr/local/lib/python3.11/lib2to3 \
           /usr/local/lib/python3.11/pydoc_data \
           /usr/local/lib/python3.11/tkinter \
           /usr/local/lib/python3.11/turtledemo \
           /usr/local/lib/python3.11/test \
           /usr/share/doc \
           /usr/share/man \
           /usr/share/locale \
           /usr/lib/apt \
           /usr/bin/apt /usr/bin/apt-get /usr/bin/apt-cache /usr/bin/apt-config /usr/bin/apt-mark \
           /usr/bin/sqv \
           /usr/bin/perl5* \
           /usr/lib/x86_64-linux-gnu/perl-base \
           /usr/share/perl5 \
           /usr/lib/x86_64-linux-gnu/libapt-pkg.so* \
           /usr/lib/x86_64-linux-gnu/libdb-5.3.so; \
    find /usr/local/lib/python3.11 -name '__pycache__' -type d -prune -exec rm -rf '{}' +

# 非 root 用户：固定 uid/gid，方便宿主机排障时对得上
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app

# ---------------------------------------------------------------- runtime
# 从 scratch 接住上面那个删干净的文件系统：这一层才真实反映"删掉的东西没了"
FROM scratch AS runtime

COPY --from=slim-base / /

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    PORT=8000

WORKDIR /app

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
