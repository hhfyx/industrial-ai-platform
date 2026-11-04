"""**拿到 API key 后，一条命令验证它能不能用。**

## 这个脚本解决什么问题

部署到 Streamlit Cloud 之前，最怕的是"点下部署、等 3 分钟、发现 key 不对"。
而 key 的问题有好几类，症状**完全不同**、修法也完全不同：

| 症状 | 真实原因 | 修法 |
| --- | --- | --- |
| `401 Unauthorized` | key 错 / 过期 / 多了空格 | 重新复制 key |
| `404 model_not_found` | 模型名错 | 查 `/models` 拿真 id |
| `402 / 429` | 余额不足 / 限流 | 充值或等一会儿 |
| 超时 / 连接重置 | 网络不通（国内可能要代理） | 开代理，或换国内厂商 |
| 500 / 502 | 服务端问题 | 换时间重试 |
| **能返回但答案是空的** | `max_tokens` 太小，被推理吃光 | 加大 max_tokens |

**最后一条最容易误判**：HTTP 200、usage 有数、但 `content` 是 `None`。
看起来"成功"，实际什么都拿不到。

## 为什么单独写一个脚本，而不是直接跑 answer-quality

`answer-quality` 要跑 20 条真实问答（花几元钱、几分钟）。
这个脚本只做**三次最小调用**（总计约 ¥0.0001），
把"key 能不能用"和"问答质量好不好"分成两个独立的检查。

**先用这个确认连通，再花钱跑质量** —— 否则可能花了钱才发现 key 是错的。

## 它会检查的六件事

1. **base_url 形态** —— 是本地还是公网？公网端点部署后必须可达
2. **连通性** —— 能不能连上（含代理是否正确）
3. **鉴权** —— key 有效吗
4. **模型名** —— 这个名字真的存在吗（并列出可用的前几个）
5. **能不能出字** —— 不只是 HTTP 200，要有非空 content
6. **能不能带上下文作答** —— 模拟真实的 RAG prompt（给一段材料问问题）

## 用法

```bash
# ① 用当前 .env 里的配置（本地网关）
python -X utf8 scripts/check_llm_endpoint.py

# ② 直接用参数测（部署前的官方 API，推荐先这样验一遍）
python -X utf8 scripts/check_llm_endpoint.py \
    --base-url https://api.deepseek.com/v1 \
    --api-key sk-xxxx \
    --model deepseek-chat
```

退出码：`0` 全通；`1` 有检查项失败（方便接到部署脚本里）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from app.core.llm.client import is_loopback_base  # noqa: E402

#: 给模型的一小段材料 —— 模拟真实 RAG prompt 的最小形态。
RAG_PROBE_CONTEXT = (
    "【材料】逾期交付的，按日万分之五支付违约金。首字延迟指从发起请求到第一个非空内容的时间。"
)
RAG_PROBE_QUESTION = "逾期交付的违约金怎么算？只回答一句，并标注引用的编号。"


def _fail(message: str) -> None:
    print(f"  ✗ {message}")


def _ok(message: str) -> None:
    print(f"  ✓ {message}")


def _key_shape_warning(api_key: str) -> None:
    """key 形态预检 —— 只提醒，不阻断（有的厂商 key 确实不带前缀）。

    实测踩到过：把 **UUID 形态的其它 ID**（`api-37102921-f6c2-...`）
    当成 API key 粘进来。那种情况下请求必然 401，但如果只报 401，
    人会先去怀疑"key 是不是过期了""是不是没充值"，而不是"粘错东西了"。

    所以这里把**可零成本判断的形态信息**先说出来。
    """
    if not api_key:
        return
    # **先查长度，再查前缀。** 顺序反过会让 `sk-abc`（粘漏了的 key）被
    # `startswith("sk-")` 直接放行 —— 而"带对了前缀但明显太短"恰恰是
    # copy 到一半就松手这种最常见的失误。测试抓过这个顺序 bug。
    if len(api_key) < 16:
        print(f"  ⚠ key 只有 {len(api_key)} 个字符，偏短 —— 可能复制不全")
        print()
        return
    if api_key.startswith("sk-"):
        return  # 最常见形态，无需提醒
    # UUID 形态是 36 位（8-4-4-4-12），带 `api-` 之类前缀时到 40 位左右。
    # **上界取 45 而不是 40** —— 之前写 range(30, 40) 会把 40 位的漏掉（差一位）。
    looks_like_uuid = api_key.count("-") >= 4 and len(api_key) <= 45
    if looks_like_uuid:
        print("  ⚠ 这个 key 是 **UUID 形态**，不像常见 API key（通常以 `sk-` 开头）")
        print("     DeepSeek 的 key 在 platform.deepseek.com → API keys 里创建")
        print("     （如果是别的厂商，忽略这条提醒）")
        print()


async def _check_endpoint_reachable(base_url: str, api_key: str) -> bool:
    """① 连通性 + ② 鉴权 + ③ 模型名，一次 `/models` 全测了。

    **这一步不花 token**（`/models` 通常是免费的元信息接口）。
    所以"连通/鉴权/模型名"这三个最容易错的地方，能用零成本的方式先排除。
    """
    url = f"{base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(
            # 与 `_build_client` 同一套判断：本地绕过代理、公网尊重代理。
            # 这里必须**保持一致**，否则会出现"脚本能通、应用不通"的诡异现象。
            trust_env=not is_loopback_base(base_url),
            timeout=20.0,
        ) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError as exc:
        _fail(f"连不上 {url}")
        print(f"      原因：{type(exc).__name__}: {exc}")
        if is_loopback_base(base_url):
            print("      → 本地地址：确认那个端口上真的有服务在跑")
        else:
            print("      → 公网地址：国内网络可能需要代理；确认代理已开启后重试")
        return False

    if resp.status_code != 200:
        _fail(f"`/models` 返回 HTTP {resp.status_code}")
        body = resp.text[:300]
        print(f"      响应：{body}")
        if resp.status_code == 401:
            print("      → key 无效：检查是否复制完整、有无多余空格/引号")
        elif resp.status_code == 404:
            print("      → base_url 路径可能不对：通常要以 `/v1` 结尾")
        return False

    try:
        payload = resp.json()
    except ValueError:
        _fail("`/models` 返回的不是 JSON（base_url 可能指到了网页而不是 API）")
        print(f"      响应前 200 字：{resp.text[:200]}")
        return False

    models = [item.get("id") for item in payload.get("data", []) if item.get("id")]
    _ok(f"连通 + 鉴权通过（{resp.status_code}），可用模型 {len(models)} 个")
    if models:
        print(f"      例如：{', '.join(models[:5])}{' …' if len(models) > 5 else ''}")
    return True


async def _check_listed_model(base_url: str, api_key: str, model: str) -> bool:
    """模型名是否在可用列表里。

    **这条单独检查的理由**：模型名写错在 DSH 网关上返回的是
    `502 未配置凭据`（完全误导），在别的网关上可能是 `404`。
    先在列表里对一遍，比等报错快得多。
    """
    url = f"{base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(
            trust_env=not is_loopback_base(base_url), timeout=20.0
        ) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
        if resp.status_code != 200:
            return True  # 列表拿不到就不下结论，交给后面的真实调用判断
        models = {item.get("id") for item in resp.json().get("data", [])}
    except (httpx.HTTPError, ValueError):
        return True

    if not models:
        return True
    if model in models:
        _ok(f"模型名 `{model}` 在可用列表里")
        return True

    _fail(f"模型名 `{model}` **不在**可用列表里")
    # 给最可能的候选：同名但前缀不同（`buddy/glm-5.3` vs `buddy-pool/glm-5.3`）
    short = model.split("/")[-1]
    similar = [m for m in models if m.endswith(short) or short in m][:5]
    if similar:
        print(f"      可能是：{', '.join(similar)}")
    return False


async def _check_generation(client: AsyncOpenAI, model: str) -> bool:
    """④ 真的能出字（不只是 HTTP 200）。

    ## 为什么要检查 content 非空

    实测踩到：`max_tokens=5` 时，模型把预算全用在推理上，
    HTTP 200、`usage` 有数字，但 `content` 是 `None`。
    只看状态码会以为"通了"，实际什么都没拿到。
    """
    started = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "只回复两个字：收到"}],
            max_tokens=64,
        )
    except Exception as exc:  # noqa: BLE001 - 要把 SDK 的各种异常都翻译成人话
        _fail(f"生成调用失败：{type(exc).__name__}")
        text = str(exc)
        print(f"      {text[:300]}")
        if "401" in text:
            print("      → 鉴权失败：key 无效")
        elif "404" in text or "model" in text.lower():
            print("      → 模型名可能不对：看上面那步的可用列表")
        elif "429" in text:
            print("      → 限流或余额不足")
        elif "timeout" in text.lower() or "connect" in text.lower():
            print("      → 网络问题：检查代理")
        return False

    elapsed = int((time.perf_counter() - started) * 1000)
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        _fail(f"返回 200 但 **content 为空**（{elapsed}ms）")
        print("      → 通常是 max_tokens 太小、被推理过程吃光；加大后重试")
        return False

    tokens = resp.usage.total_tokens if resp.usage else "?"
    _ok(f"能出字：{content[:30]!r}（{elapsed}ms，{tokens} tokens）")
    return True


async def _check_rag_style(client: AsyncOpenAI, model: str) -> bool:
    """⑤ 带材料作答 —— 这是真实链路的最小形态。

    单纯"能回复"不代表"能在 RAG 场景下用"。这一步给一小段材料 + 一个问题，
    看它会不会从材料里取答案。**这是最接近线上行为的检查。**
    """
    started = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "只依据【材料】回答。找不到就回答「材料中未提到」。",
                },
                {
                    "role": "user",
                    "content": f"{RAG_PROBE_CONTEXT}\n\n问题：{RAG_PROBE_QUESTION}",
                },
            ],
            max_tokens=256,
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"带材料的调用失败：{type(exc).__name__}: {str(exc)[:200]}")
        return False

    elapsed = int((time.perf_counter() - started) * 1000)
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        _fail("带材料的调用返回空内容")
        return False

    # 关键：答案里要出现材料中的数值（万分之五），否则说明它没读材料
    hit = "万分之五" in content or "0.05" in content
    print(f"      答案：{content[:120].replace(chr(10), ' ')}")
    if hit:
        _ok(f"能从材料里取到答案（{elapsed}ms）")
        return True

    _fail("答案里没有出现材料中的关键数字（万分之五）—— 可能没读材料，或改写了表述")
    print("      → 不一定是致命问题（模型可能换算了），但值得人工看一眼")
    return False


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="验证 LLM 端点能不能用")
    parser.add_argument("--base-url", help="覆盖 .env 里的 LLM_BASE_URL")
    parser.add_argument(
        "--api-key",
        help=(
            "覆盖 .env 里的 LLM_API_KEY。**建议不要用这个参数** —— "
            "命令行参数会明文进 PowerShell 历史文件，改用 --prompt-key 交互输入"
        ),
    )
    parser.add_argument(
        "--prompt-key",
        action="store_true",
        help="交互式输入 key（输入时不回显、不进命令历史，推荐）",
    )
    parser.add_argument("--model", help="覆盖 .env 里的 LLM_MODEL")
    parser.add_argument(
        "--provider",
        choices=["deepseek", "zhipu", "moonshot", "openai", "local"],
        help=(
            "用预设的 base_url + model（省去手拼参数、也避免拼错）。"
            "deepseek <- 推荐；local <- 用 .env 里的本地网关"
        ),
    )
    args = parser.parse_args(argv)

    # ── 厂商预设：把"容易拼错的两项"固化下来 ─────────────────────────────
    #
    # 为什么需要：base_url 少写 `/v1`、模型名多写一个 `-chat`，
    # 都会得到一个看似"key 无效"或"模型不存在"的报错，
    # 而真正的原因只是参数拼错。预设把这两项锁死。
    _PRESETS: dict[str, tuple[str, str]] = {
        "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat"),
        "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
        "moonshot": ("https://api.moonshot.cn/v1", "moonshot-v1-8k"),
        "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    }
    if args.provider and args.provider != "local":
        preset_base, preset_model = _PRESETS[args.provider]
        # 显式传的参数优先于预设（允许"用 deepseek 的端点但换模型"）
        args.base_url = args.base_url or preset_base
        args.model = args.model or preset_model
    elif args.provider == "local":
        args.provider = None  # local 就是"什么都不覆盖，走 .env"

    # ── key 的来源优先级：交互输入 > 命令行 > 配置 ────────────────────────
    #
    # **为什么要有 `--prompt-key`**（而不是只留 `--api-key`）：
    # 把 key 写在命令行里，PowerShell 会把它明文追加到
    # `%APPDATA%\Microsoft\Windows\PowerShell\PSReadLine\ConsoleHost_history.txt`。
    # 那是纯文本，任何能读该文件的人或程序都能拿到 —— 而且**删命令不会删历史**。
    # 用 `getpass` 交互输入则连回显都没有，更不会进历史。
    prompted_key: str | None = None
    if args.prompt_key:
        import getpass

        prompted_key = getpass.getpass("请粘贴 API key（输入时不显示，回车确认）：").strip()
        if not prompted_key:
            print("✗ 没有输入 key，已退出")
            return 1

    # 读配置。**key 一律不打印**（只打印长度与后 4 位）。
    from app.core.config import get_settings

    settings = get_settings()
    base_url = args.base_url or settings.llm.base_url
    model = args.model or settings.llm.model
    loopback = is_loopback_base(base_url)

    if prompted_key or args.api_key:
        # 只要给了 key，就必须同时知道打到哪个端点 —— 缺的项从配置补。
        api_key = prompted_key or args.api_key or ""
        source = "交互输入 + 配置补全" if prompted_key else "命令行参数 + 配置补全"

        # ── 端点/key 不匹配检测（这段来自真实误报）──────────────────────
        #
        # 实测踩到：用户 `--prompt-key` 输入了 DeepSeek 的 key（`sk-` 开头），
        # 但没传 `--base-url`，于是从 `.env` 补到了 **本地网关**
        # `http://127.0.0.1:8326/v1`。网关当然不认 DeepSeek 的 key，
        # 返回 `401 {"code":"invalid_api_key"}` —— 脚本于是报「key 无效」。
        #
        # **这个误报比不检查更糟**：用户会去核对自己有没有复制错，
        # 而不是发现"打错服务器了"。所以这里必须在发请求前就说清楚。
        #
        # 判据：厂商官方 key（`sk-` + 足够长）+ 本地端点 = 必然错配。
        looks_like_vendor_key = api_key.startswith("sk-") and len(api_key) >= 32
        if loopback and looks_like_vendor_key:
            print("=" * 78)
            print("⚠ **端点与 key 不匹配** —— 停在这里，不要继续排查 key")
            print("=" * 78)
            print(f"  你给的 key : `sk-` 开头，{len(api_key)} 位（像厂商官方 key）")
            print(f"  base_url   : {base_url}  ← **本地网关**")
            print()
            print("  本地网关只认 DSH 会话 key，不认厂商官方 key —— 必然 401。")
            print("  你刚才看到的「key 无效」是**误报**，真问题是没有指定端点。")
            print()
            print("  想测 DeepSeek 就用预设：")
            print("    --provider deepseek --prompt-key")
            print()
            print("  （脚本不自动猜端点：猜错会给出误导性诊断，比不猜更糟）")
            print()
            return 1
    else:
        api_key = settings.llm.api_key.get_secret_value()
        source = "配置（.env / 环境变量）"

    print("=" * 78)
    print("LLM 端点自检")
    print("=" * 78)
    print(f"  配置来源   : {source}")
    print(f"  base_url   : {base_url}")
    print(
        f"  目标类型   : {'**本地回环**（部署后不可用）' if loopback else '**公网**（部署可用）'}"
    )
    print(f"  模型       : {model}")
    print(
        f"  key        : {'*' * 8}{api_key[-4:] if len(api_key) > 4 else ''}（长度 {len(api_key)}）"
    )
    print()

    # ── 形态预检：key 长得不像 API key 时当场点明，别等 401 ─────────────
    #
    # **为什么值得单独检查**：实测有人把别的 ID（UUID 形态）当成 key 粘进来，
    # 然后一路排查网络、排查 base_url，最后才发现"粘错东西了"。
    # 形态检查是零成本的，能立刻把这类问题分离出来。
    _key_shape_warning(api_key)

    if loopback:
        print("  ⚠ 这是本地地址：本机验证没问题，但 **Streamlit Cloud 上必须换成公网端点**")
        print("     （部署后 127.0.0.1 指 Cloud 容器自己）")
        print()

    # `trust_env` 与应用里完全一致，否则会出现"脚本能通、应用不通"
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30.0,
        max_retries=1,
        http_client=httpx.AsyncClient(trust_env=not loopback, timeout=30.0),
    )

    checks: list[tuple[str, bool]] = []
    try:
        print("① 连通性 / 鉴权")
        reachable = await _check_endpoint_reachable(base_url, api_key)
        checks.append(("连通 / 鉴权", reachable))

        print("\n② 模型名")
        model_ok = await _check_listed_model(base_url, api_key, model) if reachable else False
        checks.append(("模型名", model_ok))

        if not reachable:
            print("\n（连不上就跳过后续检查）")
        else:
            print("\n③ 能否出字")
            gen_ok = await _check_generation(client, model)
            checks.append(("能出字", gen_ok))

            if gen_ok:
                print("\n④ 能否带材料作答（真实 RAG 形态）")
                rag_ok = await _check_rag_style(client, model)
                checks.append(("带材料作答", rag_ok))
    finally:
        await client.close()

    print()
    print("=" * 78)
    failed = [name for name, passed in checks if not passed]
    if not failed:
        print("✅ 全部通过 —— 这个端点和 key 可以用于部署")
        if loopback:
            print()
            print("⚠ 但别忘了：本地网关在 Cloud 上不可达。")
            print("  部署前用官方 API 再跑一遍这个脚本：")
            print("    python -X utf8 scripts/check_llm_endpoint.py \\")
            print("        --base-url https://api.deepseek.com/v1 \\")
            print("        --api-key sk-xxx --model deepseek-chat")
        return 0

    print(f"❌ {len(failed)} 项未通过：{', '.join(failed)}")
    print("  按上面每项的「→」提示处理。")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
