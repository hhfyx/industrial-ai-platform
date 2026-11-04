"""`scripts/check_llm_endpoint.py` 的测试：**不联网、不花钱**。

## 为什么要给一个"诊断脚本"写测试

因为诊断脚本最危险的情况是**它自己说谎**：
报了"✅ 全部通过"，而真实的调用路径其实不通。

这个项目已经有先例（见 `docs/deploy-notes.md` 坑 1）：
内存测量脚本因为解析错字段，报出"峰值 1 MB，免费层完全够"，
而真相是 561 MB —— **小数字看起来是好事，所以没人怀疑**。

所以这里只测两件能离线验的事：

1. **退出码语义** —— 失败必须返回非 0（否则 CI / 部署脚本会被骗）
2. **`trust_env` 与应用一致** —— 否则会出现"脚本能通、应用不通"

真正的端到端检查由 `make check-llm` 做（约 5 秒，近乎免费），
但那需要真实端点，不适合放进单元测试。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "check_llm_endpoint.py"


def test_script_exists_and_is_documented() -> None:
    """脚本要在，且被文档指向 —— 否则它是一个"存在但没人知道"的工具。"""
    assert SCRIPT.exists(), "缺 scripts/check_llm_endpoint.py"

    notes = PROJECT_ROOT / "docs" / "deploy-notes.md"
    assert "check_llm_endpoint.py" in notes.read_text(encoding="utf-8"), (
        "deploy-notes.md 必须指向这个自检脚本，否则读者不知道部署前该先跑它"
    )


def test_unreachable_endpoint_exits_nonzero() -> None:
    """**连不上必须返回非 0 退出码。**

    这是这条测试的全部价值：如果连不上也返回 0，
    接到部署流程里就会"检查通过 → 上去发现不通"。

    用一个**一定没有服务**的端口（1 号端口非特权、不会有人监听），
    并从参数注入，避免读到 `.env` 里的真实配置。
    """
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(SCRIPT),
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--api-key",
            "sk-dummy",
            "--model",
            "dummy-model",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    assert result.returncode != 0, (
        f"连不上却返回了 0 —— 部署脚本会被这个退出码骗过\nstdout:\n{result.stdout[-800:]}"
    )
    # 人话提示（不是裸异常）
    combined = result.stdout + result.stderr
    assert "连不上" in combined or "未通过" in combined, (
        f"失败时应当有中文提示，实际输出：{combined[-500:]}"
    )


def test_bad_key_is_reported_as_401_hint() -> None:
    """key 错时要给出"key 无效"的方向，而不是只丢一个状态码。

    仍然指向不可达端口 —— 这条只验**输出里有可操作的提示**，
    不依赖真实网络。（真实 401 由 `make check-llm` 验。）
    """
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(SCRIPT),
            "--base-url",
            "https://api.invalid-host-for-test.invalid/v1",
            "--api-key",
            "sk-definitely-wrong",
            "--model",
            "whatever",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    # 域名不存在 -> 应当提示网络/代理方向，而不是崩出 traceback
    assert "Traceback" not in combined, f"不该裸崩：{combined[-500:]}"
    assert "→" in combined, "失败时要给出「→」形式的具体建议"


def test_trust_env_matches_application_for_loopback() -> None:
    """脚本与应用的代理判断必须一致（都调 `is_loopback_base`）。

    这一条是**结构性检查**：不是跑脚本，而是读源码确认它用了同一个函数。

    为什么值得单独一条：如果脚本自己写死 `trust_env=False` 或 `True`，
    那么在"本地能通 / 公网不通"（或反之）时会给出**误导性的自检结果** ——
    而自检脚本说谎比没有自检更糟。
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert "is_loopback_base" in source, "脚本没有用 is_loopback_base —— 代理判断可能与应用不一致"
    # 允许换行/空格差异：只看"trust_env 的值由 is_loopback_base 推出"。
    compact = "".join(source.split())
    assert "trust_env=notis_loopback_base" in compact, (
        "脚本的 trust_env 应当由 is_loopback_base 推出（与应用同一判据）"
    )


def _load_checker_module():
    """把脚本当模块导入，好直接测它的纯函数。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_llm_endpoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_key_shape_warning_flags_uuid_form(capsys) -> None:
    """UUID 形态的"key"要当场点明，而不是等一个裸 401。

    ## 这条来自真实踩坑

    实测有人把 `api-37102921-f6c2-4968-a141-c2d1993ed94e`（UUID + 前缀）
    当成 DeepSeek API key 粘进来。服务端当然返回 `401 Authentication Fails`，
    但**只报 401 会让人去怀疑"过期了""没充值""额度用完"**，
    而不是"我粘错东西了"。

    形态检查零成本，能立刻把这类问题分离出来。

    ⚠ 边界：**只提醒、不阻断** —— 有的厂商 key 确实不带 `sk-` 前缀，
    硬阻断会误伤正常配置。
    """
    mod = _load_checker_module()

    def captured(key: str) -> str:
        capsys.readouterr()  # 清掉上一次的输出
        mod._key_shape_warning(key)
        return capsys.readouterr().out

    # UUID 形态（含 40 位那种 —— 之前 range(30, 40) 漏掉过 40 位，差一位）
    assert "UUID" in captured("api-37102921-f6c2-4968-a141-c2d1993ed94e")
    assert "UUID" in captured("37102921-f6c2-4968-a141-c2d1993ed94e")

    # 正常形态：不啰嗦
    assert captured("sk-abcdef1234567890abcdef") == ""
    # 太短：提醒复制不全
    assert "偏短" in captured("sk-abc")
    # 空值：不崩
    assert captured("") == ""


def test_prompt_key_option_exists() -> None:
    """必须提供 `--prompt-key`，避免 key 明文进 PowerShell 历史文件。

    ## 为什么这是安全问题而不是体验问题

    PowerShell 把每条命令明文追加到
    `%APPDATA%\\Microsoft\\Windows\\PowerShell\\PSReadLine\\ConsoleHost_history.txt`。
    用 `--api-key sk-xxx` 传 key，就等于把 key **永久写进一个纯文本文件** ——
    而且**删掉那条命令也不会删历史**。

    `getpass` 交互输入不回显、不进历史，是更安全的方式。
    （保留 `--api-key` 是为了 CI 等非交互场景。）
    """
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "--prompt-key" in result.stdout, (
        "缺 --prompt-key：用户只能用命令行传 key，会明文进 PowerShell 历史"
    )
    # 帮助文本要说明为什么（否则没人会用那个更麻烦的选项）
    assert "历史" in result.stdout, "帮助里要说明 --api-key 会进命令历史"
