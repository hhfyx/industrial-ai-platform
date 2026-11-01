"""从 ModelScope 拉 `bge-reranker-v2-m3` 到本地缓存。

## 为什么不用 `huggingface_hub`

`huggingface.co` 在本机**不可达**（`getaddrinfo failed`），而 ModelScope 可达。
所以这里直接走 ModelScope 的 `resolve/master/<file>` 下载接口，
把文件按原布局放进一个 HF 风格的目录，让 `transformers` 能按普通本地路径加载。

## 为什么单独写一个脚本而不是 `from_pretrained("BAAI/...")`

那样会触发 HF 的网络请求并卡在超时上（实测 huggingface_hub 会先校验 commit hash）。
本地路径加载不碰网络，这也让"拔网线能跑"这条纪律继续成立。

## 断点续传

2.27 GB 的 safetensors 下载中断了不能从头来。这里用 `Range` 头 +
临时文件追加，已下完的部分不重下。
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

MODEL_ID = "BAAI/bge-reranker-v2-m3"
BASE_URL = f"https://www.modelscope.cn/models/{MODEL_ID}/resolve/master"
TARGET_DIR = Path("d:/cache/modelscope/bge-reranker-v2-m3")

FILES: tuple[str, ...] = (
    "config.json",
    "configuration.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "model.safetensors",
)


def download(name: str, dest: Path) -> None:
    """分块下载 + 断点续传。**大文件必须续传，否则一次网络抖动就白下。**"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    request = urllib.request.Request(f"{BASE_URL}/{name}")
    if existing:
        request.add_header("Range", f"bytes={existing}-")

    try:
        response = urllib.request.urlopen(request, timeout=60)
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {name}: {exc}", flush=True)
        return

    total_header = response.headers.get("Content-Length")
    total = int(total_header) + existing if total_header else None
    mode = "ab" if existing else "wb"
    written = existing
    started = time.time()
    last_report = started

    with dest.open(mode) as handle:
        while True:
            chunk = response.read(1 << 20)  # 1 MB
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            now = time.time()
            if now - last_report >= 5:
                speed = (written - existing) / (now - started) / 1e6 if now > started else 0
                pct = f"{written / total * 100:.1f}%" if total else "?"
                print(
                    f"  {name}: {written / 1e6:.1f}/{total / 1e6:.1f} MB ({pct}) {speed:.1f} MB/s",
                    flush=True,
                )
                last_report = now
    print(f"[ok] {name}: {written / 1e6:.1f} MB", flush=True)


def main() -> int:
    print(f"目标目录：{TARGET_DIR}", flush=True)
    for name in FILES:
        dest = TARGET_DIR / name
        expected = None
        # safetensors 已知大小，用它判断是否已完整
        if name == "model.safetensors":
            expected = 2271071852
        if dest.exists() and expected and dest.stat().st_size == expected:
            print(f"[skip] {name}: 已完整", flush=True)
            continue
        download(name, dest)
    print("\n完成。目录内容：", flush=True)
    for item in sorted(TARGET_DIR.glob("*")):
        print(f"  {item.name}: {item.stat().st_size / 1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
