"""验收 `/extract` 的 degraded 标记：LLM 不可达时必须降级并标记，而不是报错。

运行前先把服务起在 8011（LLM_BASE_URL 指向不可达地址）：
    python -X utf8 -m uvicorn app.main:app --host 127.0.0.1 --port 8011

为什么用 Python 发请求而不是 PowerShell：中文 JSON 在 PowerShell 里
用单引号传会被 GBK 编码破坏（本次实测踩到，返回 422 而不是业务响应）。
"""

from __future__ import annotations

import json
import sys

import httpx

BASE = "http://127.0.0.1:8011"


def main() -> None:
    # ① 健康检查
    r = httpx.get(f"{BASE}/health", timeout=10)
    print(f"/health -> {r.status_code}")

    # ② 正常请求（LLM 不可达 → 必须降级）
    payload = {
        "text": "甲方为星辰科技有限公司，合同编号 HT-2026-0042，签署日期 2026-03-15。",
        "fields": ["合同编号", "签署日期"],
    }
    r = httpx.post(f"{BASE}/api/v1/extract", json=payload, timeout=120)
    print(f"/api/v1/extract -> {r.status_code}")
    body = r.json()
    print(json.dumps(body, ensure_ascii=False, indent=2))

    # ③ 断言契约里的关键字段
    ok = True
    if r.status_code != 200:
        print("❌ 应返回 200（降级不是错误）")
        ok = False
    if body.get("degraded") is not True:
        print("❌ 缺 degraded: true 标记 —— 省钱的代价必须可见")
        ok = False
    for k in ("request_id", "data", "confidence", "elapsed_ms"):
        if k not in body:
            print(f"❌ 缺契约字段 {k}")
            ok = False
    if "合同编号" not in body.get("data", {}):
        print("❌ 降级结果里缺少请求的字段")
        ok = False

    # ④ 422 契约：多传一个键必须被拒
    r2 = httpx.post(
        f"{BASE}/api/v1/extract",
        json={"text": "a", "fields": ["x"], "unexpected": "y"},
        timeout=30,
    )
    print(f"\n多传一个键 -> {r2.status_code}（应为 422）")
    if r2.status_code != 422:
        print("❌ extra=forbid 没生效")
        ok = False

    print("\n" + ("✅ 全部通过" if ok else "❌ 有失败项"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
