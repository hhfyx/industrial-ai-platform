"""smoke_llm：第一次真实调用 + 认清两种报错长相。

做三件事（对应 20:50–21:10 那一格的验收）：

    ① 正常调用一次  -> 打印 model / usage / latency / 回答文本
    ② 错 key 一次    -> 打印异常类型 + HTTP 状态 + 错误体
    ③ 错 model 一次  -> 同上

**为什么"认清报错"要单独做一遍**：接入 LLM 的绝大多数时间不是花在 happy path，
而是花在"这句报错到底是我的 key 错了、模型名错了，还是字段名错了"。
这三类错误的**异常类型几乎一样**（都是 `openai.APIStatusError` 的子类），
靠断网/重试完全区分不出来 —— 只有把 `status_code` + `body["error"]["code"]`
打出来，才知道该改代码、改配置还是改请求。

运行（先设环境变量，或在项目根用 uv 读 .env）：

    $env:LLM_API_KEY="..."; $env:LLM_BASE_URL="..."; $env:LLM_MODEL="..."
    uv run python -X utf8 scripts/smoke_llm.py

故意失败的两次**不会**让退出码非 0：它们是"预期失败"，被 try/except 吃掉并打印。
```

不设 `PYTHONUTF8` 时，Windows 控制台打印中文回答会乱码（GBK 控制台），
所以脚本自己把 stdout 切成 UTF-8，免得"调用成功但屏幕上全是问号"。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

# 直接 `python scripts/smoke_llm.py` 时，sys.path[0] 是 scripts/ 而不是项目根，
# `import app` 会 ModuleNotFoundError。uv run 不装包（本项目没有 [build-system]），
# 所以这里手动把项目根插到最前面 —— 脚本得能独立跑起来，才算"冒烟"。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import APIConnectionError, APIStatusError, AsyncOpenAI  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.llm.client import get_llm_client  # noqa: E402

QUESTION = "用一句话解释 HTTP 429 是什么意思。"


def _force_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，打印中文回答会抛 UnicodeEncodeError / 变问号。

    `reconfigure(encoding="utf-8")` 只在 3.7+ 且 stdout 是 TextIOWrapper 时可用；
    管道/重定向场景下可能是别的类型，所以用 getattr 兜底。
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):  # pragma: no cover - 极端环境
            reconfigure(encoding="utf-8")


def _error_detail(exc: APIStatusError) -> dict[str, Any]:
    """从 `APIStatusError` 里挖出 `{message, type, code}`。

    **踩过的坑（实测 openai 3.37 / 本机网关）**：`exc.body` 已经是**去掉外层
    `"error"` 信封之后**的内层对象，直接用 `exc.body["error"]["code"]` 会一直拿到
    None —— 而 None 看起来像"网关没返回 code"，会把人往错误方向带。
    这里对两种形态都兼容：先按扁平读，再尝试解包 `"error"` 键。
    """
    body = exc.body if isinstance(exc.body, dict) else {}
    if isinstance(body.get("error"), dict):
        body = body["error"]
    return {
        "message": body.get("message"),
        "type": body.get("type"),
        "code": body.get("code"),
    }


async def call_once(
    client: AsyncOpenAI,
    *,
    model: str,
    label: str,
    api_key_override: str | None = None,
) -> dict[str, Any]:
    """打一次 chat.completions，返回**结构化观测结果**（成功或失败都返回，不抛）。

    失败结果里保留 `status` 和 `code`，这两项是"认清报错"的关键；
    异常类名保留是因为 OpenAI SDK 的分类（`AuthenticationError` / `NotFoundError`）
    本身就是给程序分支用的 —— 但**不要只靠它**，见模块 docstring。
    """
    # api_key_override 只用于"故意用错 key"这一组：
    # 另建一个 client 而不是改 settings，避免污染后面还要用的正常 client。
    if api_key_override is not None:
        client = AsyncOpenAI(
            api_key=api_key_override,
            base_url=client.base_url,
            timeout=client.timeout,
            max_retries=0,  # 故意错的请求不要重试，否则要等三次超时
        )

    started = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": QUESTION}],
            max_tokens=100,
        )
    except APIStatusError as exc:
        # 有 HTTP 响应但状态码非 2xx：401 错 key / 404 错 model / 429 限流 …
        latency_ms = int((time.perf_counter() - started) * 1000)
        err = _error_detail(exc)
        return {
            "label": label,
            "ok": False,
            "kind": "APIStatusError",
            "exc_type": type(exc).__name__,
            "status": exc.status_code,
            "code": err.get("code"),
            "type": err.get("type"),
            "message": err.get("message") or str(exc.message),
            "latency_ms": latency_ms,
        }
    except APIConnectionError as exc:
        # 连不上：DNS / 端口 / 代理 / 超时。与上面**完全不是一类问题**。
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "label": label,
            "ok": False,
            "kind": "APIConnectionError",
            "exc_type": type(exc).__name__,
            "status": None,
            "code": None,
            "type": None,
            "message": str(exc),
            "latency_ms": latency_ms,
        }

    latency_ms = int((time.perf_counter() - started) * 1000)
    choice = resp.choices[0]
    usage = resp.usage
    return {
        "label": label,
        "ok": True,
        "kind": "OK",
        "exc_type": None,
        "model": resp.model,
        "finish_reason": choice.finish_reason,
        "content": (choice.message.content or "").strip(),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
        "latency_ms": latency_ms,
    }


def report(result: dict[str, Any]) -> None:
    """把一次结果打成人类可读的一段。成功与失败用同一套排版，便于对照。"""
    print(f"\n======== [{result['label']}] ========")
    if result["ok"]:
        print(f"  model        : {result['model']}")
        print(f"  latency_ms   : {result['latency_ms']}")
        print(f"  finish_reason: {result['finish_reason']}")
        print(f"  usage        : {json.dumps(result['usage'], ensure_ascii=False)}")
        print(f"  content      : {result['content']}")
    else:
        print(f"  kind         : {result['kind']}  (exc_type={result['exc_type']})")
        print(f"  status       : {result['status']}")
        print(f"  error.code   : {result['code']}")
        print(f"  error.type   : {result['type']}")
        print(f"  message      : {result['message']}")
        print(f"  latency_ms   : {result['latency_ms']}")


async def main() -> int:
    _force_utf8_stdout()

    settings = get_settings()
    llm = settings.llm
    print("======== 配置（key 只显示掩码）========")
    print(f"  base_url    : {llm.base_url}")
    print(f"  model       : {llm.model}")
    print(f"  timeout     : {llm.timeout}s")
    print(f"  max_retries : {llm.max_retries}")
    print(f"  api_key     : {llm.api_key_masked}")

    client = get_llm_client(settings)

    results: list[dict[str, Any]] = []

    # ① 正常调用
    results.append(await call_once(client, model=llm.model, label="正常调用"))

    # ② 错 key：注意 SDK 会在客户端就拒绝空串，所以给一个格式合法但无效的 key
    results.append(
        await call_once(
            client,
            model=llm.model,
            label="故意错 key",
            api_key_override="sk-definitely-not-a-valid-key-000",
        )
    )

    # ③ 错 model
    results.append(
        await call_once(client, model="no-such-vendor/no-such-model-xyz", label="故意错 model")
    )

    for r in results:
        report(r)

    # 退出码只反映"正常调用是否成功"：后两次是预期失败，不算用例失败。
    ok = results[0]["ok"]
    print("\n======== 结论 ========")
    print(f"  正常调用成功 : {ok}")
    print(f"  错 key   状态 : {results[1]['status']} / code={results[1]['code']}")
    print(f"  错 model 状态 : {results[2]['status']} / code={results[2]['code']}")
    print(
        "\n  判据：两者异常类型都属 APIStatusError，区分只能看 status_code + error.code。\n"
        "  401/invalid_api_key -> 改凭证；404/model_not_found -> 改模型名。"
    )

    await client.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
