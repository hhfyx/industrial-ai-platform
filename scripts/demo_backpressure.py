"""80 并发打 /generate/async，验证队列深度 32 触发 429 + 动态 Retry-After。"""

import json
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8123/api/v1/generate/async"


def submit(i: int) -> tuple[int, str, str]:
    body = json.dumps(
        {"defect_class": "凹坑", "count": 8, "size": 768, "seed": 10_000 + i}
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, "", ""
    except urllib.error.HTTPError as e:
        retry_after = e.headers.get("Retry-After", "-")
        payload = json.loads(e.read().decode("utf-8"))
        return e.code, retry_after, payload.get("message", "")


with ThreadPoolExecutor(max_workers=80) as pool:
    results = list(pool.map(submit, range(80)))

print(Counter(code for code, _, _ in results))
limited = [(ra, msg) for code, ra, msg in results if code == 429]
if limited:
    print("Retry-After:", limited[0][0])
    print("message:", limited[0][1])
