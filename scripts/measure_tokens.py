"""measure_tokens：tiktoken 估算 vs 真实 usage 的偏差 + 三种文本的 token 密度。

**为什么这件事必须先测再用：**

用 tiktoken 做成本估算是个常见做法，但它估的是 GPT 的 BPE 词表。
我们的模型（GLM / DeepSeek 系）用的是**另一套 tokenizer** ——
直接拿 tiktoken 算出的数当"真实成本"会系统性偏错，而错误方向（高估还是低估）事先不知道。

所以本脚本做两件事：

1. **偏差表**：同一段文本，`tiktoken 估算` vs `网关返回的真实 usage`，
   分中文 / 英文 / 代码三类。给出每类的偏差系数（倍数），
   这个系数就是"能不能用 tiktoken 做估算"的答案。
   如果中文偏差是 1.8×，说明 tiktoken 对中文明显高估（或低估），
   那成本表就不能直接拿 tiktoken 数 × 单价。

2. **token 密度**：每类文本"每字符多少 token"。它决定"一份 1000 字的中文合同
   大概多少 token"，是做容量规划时的第一手数字。

**一个容易算错的地方（实测已发现）**：`completion_tokens` 里**包含 reasoning_tokens**。
本机网关的响应里有 `completion_tokens_details.reasoning_tokens`，
而 `prompt_tokens + completion_tokens = total_tokens` —— 说明推理 token 是被计入
计费的 completion 侧的。所以"输出很便宜"的直觉在推理模型上不成立：
它可能先在思维链里烧掉大量 token，然后只吐几十个字。

运行：

    uv run python -X utf8 scripts/measure_tokens.py
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from app.core.config import get_settings  # noqa: E402

# ---------------------------------------------------------------- 样本
# 三类文本各取一段真实感的内容，长度相近（便于跨类比较密度）。
# 刻意**不用**随机字符串：token 密度强烈依赖文本的自然程度，
# 随机串会得到完全无代表性的密度。

CHINESE_TEXT = (
    "合同编号：HT-2026-001，甲方：杭州星辰科技有限公司，乙方：上海云脉信息技术有限公司，"
    "签订日期：2026-10-20，合同金额：壹拾贰万捌仟元整（¥128,000.00），"
    "履行期限：2026-11-01 至 2027-10-31。乙方应按约定交付并保证质量，甲方按期付款。"
    "任何一方违约，应赔偿对方因此遭受的直接损失。争议提交杭州仲裁委员会仲裁。"
)

ENGLISH_TEXT = (
    "Contract No. HT-2026-001. Party A: Hangzhou Xingchen Technology Co., Ltd. "
    "Party B: Shanghai Yunmai Information Technology Co., Ltd. "
    "Signed on 2026-10-20 for a total amount of CNY 128,000.00. "
    "The term is from 2026-11-01 to 2027-10-31. Party B shall deliver as agreed and "
    "warrant quality; Party A shall pay on schedule. Either party in breach shall "
    "compensate the other for direct losses incurred."
)

CODE_TEXT = (
    "async def extract(client: AsyncOpenAI, text: str, fields: list[str]) -> dict:\n"
    "    response = await client.chat.completions.create(\n"
    "        model='glm-5.3',\n"
    "        messages=[{'role': 'user', 'content': text}],\n"
    "        temperature=0.0,\n"
    "    )\n"
    "    payload = json.loads(response.choices[0].message.content or '{}')\n"
    "    return {name: payload.get(name) for name in fields}\n"
)

SAMPLES: list[tuple[str, str]] = [
    ("中文", CHINESE_TEXT),
    ("英文", ENGLISH_TEXT),
    ("代码", CODE_TEXT),
]


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def tiktoken_estimate(text: str, encoding_name: str = "cl100k_base") -> int:
    """用 tiktoken 估 token 数。

    `cl100k_base` 是 GPT-3.5/4 的词表，也是绝大多数"tiktoken 估算脚本"默认用的。
    我们用它正是为了**测出它的偏差** —— 如果换一个更接近的词表，
    测出来的就不是"常见做法的偏差"了。
    """
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def real_usage(text: str) -> dict[str, Any]:
    """打一次真实请求，只取 usage。

    `max_tokens=1` 且要求"只回一个字"：我们只关心 **prompt_tokens**（输入侧），
    输出越短越好，省钱也省时间。推理 token 仍然会产生（本机网关默认开），
    所以下面单独记录 `reasoning_tokens` 以揭示它对计费的影响。
    """
    settings = get_settings()
    llm = settings.llm
    payload = {
        "model": llm.model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 16,
        "temperature": 0.0,
        "reasoning_effort": "none",  # 关推理，避免思维链污染 completion 侧
    }
    response = httpx.post(
        f"{llm.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {llm.api_key.get_secret_value()}"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage", {})
    details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "reasoning_tokens": details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0,
    }


def measure_reasoning_overhead() -> dict[str, Any]:
    """对比"开推理"vs"关推理"的 token 消耗 —— 验证 reasoning 计入计费。

    这是成本表里最容易被漏掉的一项：推理模型的思维链**在 completion 侧计费**，
    而用户在最终答案里看不到那些 token。
    """
    settings = get_settings()
    llm = settings.llm
    prompt = "用一句话说明什么是 HTTP 429。"
    out: dict[str, Any] = {}
    for label, extra in (("关推理", {"reasoning_effort": "none"}), ("开推理", {})):
        payload = {
            "model": llm.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.0,
            **extra,
        }
        response = httpx.post(
            f"{llm.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {llm.api_key.get_secret_value()}"},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        details = usage.get("completion_tokens_details") or {}
        out[label] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": details.get("reasoning_tokens", 0),
            "content_chars": len(data["choices"][0]["message"].get("content") or ""),
        }
    return out


def main() -> int:
    _force_utf8_stdout()

    parser = argparse.ArgumentParser(description="token 计量：tiktoken 偏差 + 密度")
    parser.add_argument("--out", default=str(Path(__file__).with_name("token_result.json")))
    args = parser.parse_args()

    settings = get_settings()
    print(f"model={settings.llm.model}  base_url={settings.llm.base_url}")
    print("tiktoken 词表=cl100k_base（GPT 系，非同模型词表）\n")

    rows: list[dict[str, Any]] = []
    print(f"{'文本':<6} {'字符':>5} {'tiktoken':>9} {'真实prompt':>10} {'偏差':>8} {'密度':>12}")
    print("-" * 62)
    for label, text in SAMPLES:
        est = tiktoken_estimate(text)
        real = real_usage(text)
        real_prompt = real["prompt_tokens"]
        # tiktoken 只算了文本本体，真实 prompt 还含 role/message 封装（约 10 tokens）
        ratio = round(est / real_prompt, 3) if real_prompt else None
        density = round(real_prompt / max(len(text), 1), 3)
        row = {
            "label": label,
            "chars": len(text),
            "tiktoken_estimate": est,
            "real_prompt_tokens": real_prompt,
            "tiktoken_vs_real_ratio": ratio,
            "tokens_per_char": density,
            "real_usage": real,
        }
        rows.append(row)
        print(
            f"{label:<6} {len(text):>5} {est:>9} {real_prompt:>10} {ratio:>8} {density:>10} tok/字"
        )

    print("\n偏差解读：ratio = tiktoken估算 / 真实prompt_token")
    print("  > 1 表示 tiktoken 高估（按它算成本会偏贵）")
    print("  < 1 表示 tiktoken 低估（按它算成本会偏便宜，危险）")
    print("  真实 prompt 还含 message 封装（role 等固定开销约 10 token），所以正常应略小于 1")

    print("\n== 推理 token 是否计费 ==")
    overhead = measure_reasoning_overhead()
    for label, row in overhead.items():
        print(
            f"  {label}: prompt={row['prompt_tokens']} completion={row['completion_tokens']}"
            f" reasoning={row['reasoning_tokens']} 可见字数={row['content_chars']}"
        )
    print(
        "  → reasoning_tokens 计入 completion_tokens（计费），"
        "但用户在答案里看不到它。推理模型的真实成本可能远高于'答案长度'暗示的量。"
    )

    result = {
        "meta": {
            "model": settings.llm.model,
            "base_url": settings.llm.base_url,
            "encoding": "cl100k_base",
        },
        "density": rows,
        "reasoning_overhead": overhead,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 明细写入 {Path(args.out).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
