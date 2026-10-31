"""W4 回归：确认 RAG 层没有弄坏 W2/W3 的东西。"""

import os
import subprocess
import sys

checks = [
    (
        "API 四个端点 + extract 契约形状",
        r"""
import os, sys
sys.path.insert(0, ".")
os.environ.setdefault("LLM_API_KEY", "sk-test-placeholder")
os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:1/v1")
os.environ.setdefault("LLM_MODEL", "test-model")
from app.main import app
paths = app.openapi()["paths"]
need = ["/api/v1/extract", "/api/v1/extract/stream", "/api/v1/chat/stream", "/api/v1/summarize/stream"]
assert not [p for p in need if p not in paths], "端点缺失"
schema = paths["/api/v1/extract"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
assert "$ref" in schema and schema["$ref"].endswith("ExtractResult"), schema
print("OK", len(paths), "paths; extract 契约完整")
""",
    ),
    (
        "W3 四类模板全部可渲染（带全参数）",
        r"""
import sys
sys.path.insert(0, ".")
from app.prompts.loader import load_prompt, list_versions

# classify v2 比 v1 多了 `not_boundary`（"不包括什么"），所以给全字段。
# **StrictUndefined 会立刻报错而不是静默渲染空** —— 这正是它的设计目的，
# 也说明我一开始漏参数就被抓住了。
params = {
    "qa": {"question": "违约金怎么算", "contexts": [{"id": 1, "text": "按日万分之五。"}]},
    "summarize": {"text": "一份很长的合同文档内容。"},
    "classify": {"text": "客户说设备连不上网了。",
                 "categories": [
                     {"name": "故障报修", "boundary": "设备无法使用",
                      "not_boundary": "询问使用方法"},
                     {"name": "咨询", "boundary": "询问用法",
                      "not_boundary": "设备已损坏"}],
                 },
    "extract": {"text": "合同编号：HT-2026-001，甲方：张三。",
                 "fields": [
                     {"name": "合同编号", "type": "string", "desc": "合同的唯一编号"},
                     {"name": "甲方", "type": "string", "desc": "签约甲方的名称"},
                 ]},
}
for name in ("qa", "summarize", "classify", "extract"):
    vs = list_versions(name)
    assert vs, f"{name} 无版本"
    for v in vs:
        assert load_prompt(name, v, **params[name]).strip(), f"{name}/{v} 渲染为空"
print("OK versions:", {n: list_versions(n) for n in ("qa","summarize","classify","extract")})
""",
    ),
    (
        "summarize 链路（Chunk 改名后仍可用）",
        r"""
import sys, asyncio
sys.path.insert(0, ".")
from pathlib import Path
from app.rag.loaders import load_document
from app.rag.summarize import segment_documents, summarize_document
r = load_document(Path("tests/fixtures/sample50.pdf"))
segs = segment_documents(r.documents, chunk_tokens=500)
assert segs
class FC:
    class Chat:
        class C:
            async def create(self, **kw):
                m = type("M", (), {"content": "摘要"})()
                return type("R", (), {"choices": [type("X", (), {"message": m})()]})()
        def __init__(self): self.completions = self.C()
    def __init__(self): self.chat = self.Chat()
res = asyncio.run(summarize_document(r.documents, client=FC(), model="m", chunk_tokens=500))
print("OK segments=", len(segs), "chunk_count=", res.chunk_count)
""",
    ),
    (
        "CLI ask --help",
        r"""
import sys, runpy
sys.path.insert(0, ".")
sys.argv = ["x", "--help"]
try: runpy.run_module("app.cli.ask", run_name="__main__")
except SystemExit as e: assert e.code == 0, f"exit {e.code}"
print("OK cli.ask")
""",
    ),
    (
        "CLI summarize --help",
        r"""
import sys, runpy
sys.path.insert(0, ".")
sys.argv = ["x", "--help"]
try: runpy.run_module("app.cli.summarize", run_name="__main__")
except SystemExit as e: assert e.code == 0, f"exit {e.code}"
print("OK cli.summarize")
""",
    ),
    (
        "W3 A/B 脚手架 --dry-run",
        r"""
import sys, subprocess, os
sys.path.insert(0, ".")
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
p = subprocess.run([sys.executable, "scripts/ab_test.py", "--set", "qa", "--a", "v1", "--b", "v2", "--dry-run"],
                   capture_output=True, text=True, env=env)
assert p.returncode == 0, (p.stderr or "")[-400:]
assert "[done]" in p.stdout, (p.stdout or "")[-300:]
print("OK ab_test dry-run 跑通")
""",
    ),
    (
        "提示词硬编码检查仍然生效",
        r"""
import sys
sys.path.insert(0, ".")
from app.prompts.loader import check_no_hardcoded_business_terms
hits = check_no_hardcoded_business_terms("请分析华为 Mate60 的用户评价。")
assert hits, "硬编码检查失效了"
print("OK 检测到业务词:", hits[:3])
""",
    ),
]

failures = 0
for name, code in checks:
    p = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        capture_output=True,
        text=True,
        cwd=r"D:\git\网页\industrial-ai-platform",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    ok = p.returncode == 0
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if ok:
        lines = [row for row in p.stdout.strip().splitlines() if row.strip()]
        last = lines[-1]
        print("        ", last[:120])
    else:
        failures += 1
        print("        STDERR:", (p.stderr or "").strip()[-260:])

print(f"\n{len(checks) - failures}/{len(checks)} 通过")
sys.exit(1 if failures else 0)
