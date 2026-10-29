"""探针：摸清本网关支持哪种 response_format。

三种模式逐一实测，判定「一句话说清方案」的依据：
  M1 json_schema + strict=True   —— 最强约束，模型必须照 schema 出
  M2 json_object                 —— 仅保证是合法 JSON，字段结构靠 prompt
  M3 两者都不传                   —— 只有 prompt 约束（对照组）

对每种模式：
  · 记录是否被接受（HTTP 状态）
  · 是否真的按 schema 出结构
  · 是否混入多余字段（json_object 的已知毛病）
  · 是否被包在 ```json 代码块里（模型最常见的小动作）

运行：python -X utf8 probe_response_format.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx

API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8326/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "deepseek-account/deepseek-flash")
ENDPOINT = f"{BASE_URL}/chat/completions"

OUTPUT = Path(__file__).with_name("probe_response_format.json")

# 用一个「故意需要约束」的 schema：枚举 + 整数范围 + 必填三字段。
# 如果 schema 真生效，模型不可能给出 "P1" 以外的紧急程度、也不可能漏 "涉及产品"。
SCHEMA = {
    "type": "object",
    "properties": {
        "问题类型": {
            "type": "string",
            "enum": [
                "登录与账号",
                "支付与账单",
                "功能异常",
                "性能与可用性",
                "退换货",
                "咨询与售前",
                "服务与支持",
            ],
        },
        "紧急程度": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
        "涉及产品": {
            "type": "string",
            "enum": [
                "智云控制台",
                "数据同步服务",
                "移动端App",
                "支付网关",
                "开放API",
                "报表中心",
                "消息推送服务",
            ],
        },
        "置信度": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["问题类型", "紧急程度", "涉及产品", "置信度"],
    "additionalProperties": False,
}

USER_PROMPT = (
    "工单：开放 API 的 /v2/orders 接口从今天下午开始一直返回 502，客户订单同步已经断了 3 小时。"
)

SYSTEM = "你是客服工单分类助手。只输出 JSON。"


def call(mode: str) -> dict:
    body: dict = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_PROMPT},
        ],
        "max_tokens": 400,
        "temperature": 0.0,
        "reasoning_effort": "none",
    }
    if mode == "json_schema_strict":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "Ticket", "strict": True, "schema": SCHEMA},
        }
    elif mode == "json_object":
        body["response_format"] = {"type": "json_object"}
    elif mode == "none":
        pass
    else:
        raise ValueError(mode)

    try:
        r = httpx.post(
            ENDPOINT, headers={"Authorization": f"Bearer {API_KEY}"}, json=body, timeout=90
        )
    except Exception as exc:  # noqa: BLE001
        return {"mode": mode, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if r.status_code != 200:
        return {"mode": mode, "ok": False, "status": r.status_code, "error": r.text[:400]}

    d = r.json()
    msg = d["choices"][0].get("message") or {}
    content = msg.get("content") or ""
    return {
        "mode": mode,
        "ok": True,
        "status": 200,
        "content": content,
        "usage": d.get("usage", {}),
    }


FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def analyse(res: dict) -> dict:
    """判定这次输出到底有没有被约束住。"""
    if not res.get("ok"):
        return res
    text = res["content"]
    out: dict = {
        "mode": res["mode"],
        "ok": True,
        "status": res["status"],
        "raw": text[:500],
        "content_len": len(text),
    }

    # ① 是不是被代码围栏包了
    out["has_fence"] = "```" in text
    stripped = text.strip()
    m = FENCE.match(stripped)
    if m:
        stripped = m.group(1).strip()

    # ② 能不能直接解析
    try:
        payload = json.loads(stripped)
        out["parsed"] = True
    except json.JSONDecodeError:
        payload = None
        out["parsed"] = False

    if payload is None:
        out["schema_ok"] = False
        out["extra_keys"] = []
        out["missing_keys"] = []
        out["enum_violations"] = []
        return out

    keys = set(payload)
    expected = {"问题类型", "紧急程度", "涉及产品", "置信度"}

    # ③ 多余字段（json_object 的经典毛病）
    out["extra_keys"] = sorted(keys - expected)
    out["missing_keys"] = sorted(expected - keys)

    # ④ 枚举越界
    violations = []
    for field, allowed in (
        ("问题类型", SCHEMA["properties"]["问题类型"]["enum"]),
        ("紧急程度", SCHEMA["properties"]["紧急程度"]["enum"]),
        ("涉及产品", SCHEMA["properties"]["涉及产品"]["enum"]),
    ):
        val = payload.get(field)
        if val is not None and val not in allowed:
            violations.append(f"{field}={val!r}")
    out["enum_violations"] = violations

    # ⑤ 置信度类型与范围
    conf = payload.get("置信度")
    out["confidence_ok"] = isinstance(conf, (int, float)) and 0 <= conf <= 1

    out["payload"] = payload
    out["schema_ok"] = (
        not out["extra_keys"]
        and not out["missing_keys"]
        and not violations
        and out["confidence_ok"]
    )
    return out


def main() -> None:
    if not API_KEY:
        raise SystemExit("缺 LLM_API_KEY")

    print(f"endpoint={ENDPOINT}\nmodel={MODEL}\n")

    results = []
    for mode in ("json_schema_strict", "json_object", "none"):
        res = call(mode)
        a = analyse(res)
        results.append(a)
        if not a.get("ok"):
            print(f"[{mode}] ❌ 不被接受 status={a.get('status')} {a.get('error', '')[:120]}")
            continue
        print(f"[{mode}] ✅ 接受")
        print(f"     parsed={a['parsed']} has_fence={a['has_fence']} schema_ok={a['schema_ok']}")
        print(
            f"     missing={a['missing_keys']} extra={a['extra_keys']} "
            f"enum_violations={a['enum_violations']} conf_ok={a['confidence_ok']}"
        )
        print(f"     raw[:150]={a['raw'][:150]!r}")
        print()

    # 结论行
    print("=" * 60)
    strict = next((r for r in results if r["mode"] == "json_schema_strict"), {})
    jobj = next((r for r in results if r["mode"] == "json_object"), {})
    print("结论：")
    if strict.get("ok"):
        print(f"  json_schema strict 可用（schema_ok={strict['schema_ok']}）")
    else:
        print(f"  json_schema strict 不可用（status={strict.get('status')}）")
    if jobj.get("ok"):
        print(f"  json_object 可用（纯净解析={jobj['parsed']}，多余字段={jobj['extra_keys']}）")
    else:
        print(f"  json_object 不可用（status={jobj.get('status')}）")

    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 写入 {OUTPUT.name}")


if __name__ == "__main__":
    main()
