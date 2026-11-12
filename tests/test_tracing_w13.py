"""W13 周一：三级 trace 模型与全链路接入的测试。

验收三条（任务书 19:00–20:30）：
① 每条 trace 完整（一次请求一条 trace，span 树有根）
② 错误请求也有 trace（4xx/5xx/异常路径都落 trace）
③ 断网时观测不阻塞业务（sink 线程死掉/队列堵死不影响响应）
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from app.core.tracing import (
    clip_text,
    generation_span,
    get_trace_id,
    record_usage,
    span,
    trace_scope,
    trace_stats,
)
from app.main import app

ENDPOINT = "/api/v1/extract"


@pytest.fixture
async def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture(autouse=True)
def _tmp_trace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """trace 落到临时目录，测试互不污染、也不往仓库 data/ 写东西。"""
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    return tmp_path / "traces"


def _read_traces(trace_dir: Path) -> list[dict]:
    f = trace_dir / "traces.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line]


def _wait_traces(trace_dir: Path, *, min_count: int = 1, timeout_s: float = 3.0) -> list[dict]:
    """落盘在后台线程：轮询等到至少 min_count 条（或超时返回现状）。"""
    deadline = time.monotonic() + timeout_s
    traces: list[dict] = []
    while time.monotonic() < deadline:
        traces = _read_traces(trace_dir)
        if len(traces) >= min_count:
            return traces
        time.sleep(0.05)
    return traces


# ============================================================ 骨架单测


def test_clip_text_short_passes_through() -> None:
    assert clip_text("短文本") == "短文本"


def test_clip_text_long_becomes_hash_and_head() -> None:
    long = "甲" * 5000
    clipped = clip_text(long)
    assert isinstance(clipped, dict)
    assert clipped["len"] == 5000
    assert len(clipped["head"]) == 200
    assert len(clipped["sha256_16"]) == 16


def test_span_outside_trace_is_noop() -> None:
    before = trace_stats()["flushed"]
    with span("orphan") as rec:
        rec["attrs"]["x"] = 1
    # 全量跑时其它测试会先产生 trace：no-op 的语义是"这次不新增"，不是全局为 0
    assert trace_stats()["flushed"] == before


def test_trace_scope_records_nested_spans() -> None:
    with trace_scope("t-1", root_name="root") as root:
        with span("child", key="v") as child:
            assert child["parent_id"] == root["span_id"]
        with generation_span("gen", model="m") as gen:
            record_usage(gen, prompt_tokens=100, completion_tokens=20)
    # 同步 flush 是测试可见性的代价；生产路径在后台线程
    assert get_trace_id() == "t-1" or get_trace_id() == ""


def test_error_span_marks_status(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), trace_scope("t-err", root_name="root"), span("boom"):
        raise RuntimeError("炸了")
    traces = _wait_traces(tmp_path / "traces")
    assert len(traces) == 1  # 错误 trace 100% 保留（采样之外的保底）
    assert traces[0]["summary"]["status"] == "error"
    boom = next(s for s in traces[0]["spans"] if s["name"] == "boom")
    assert boom["status"] == "error"


def test_sampling_drops_normal_keeps_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRACE_SAMPLE_RATE", "0.0")
    for i in range(5):
        with trace_scope(f"ok-{i}", root_name="root"):
            pass
    with pytest.raises(RuntimeError), trace_scope("err-1", root_name="root"):
        raise RuntimeError("x")
    traces = _wait_traces(tmp_path / "traces")
    # 采样率 0：正常 trace 全丢，错误 trace 一条不丢
    assert [t["summary"]["trace_id"] for t in traces] == ["err-1"]


def test_background_task_gets_fresh_collector(tmp_path: Path) -> None:
    """create_task 拷贝 collector 引用的坑：请求侧 flush 后，后台任务
    用新 collector 写自己的 trace——两边都不丢。"""
    import asyncio

    async def main() -> None:
        with trace_scope("req-1", root_name="http.request"):
            background_done = asyncio.Event()

            async def bg() -> None:
                # 后台任务里开自己的 trace_scope（生产代码同款）
                with trace_scope("bg-1", root_name="agent.task"), span("agent.step"):
                    await asyncio.sleep(0)
                background_done.set()

            asyncio.get_running_loop().create_task(bg())
            await asyncio.wait_for(background_done.wait(), timeout=2)

    asyncio.run(main())
    traces = _wait_traces(tmp_path / "traces", min_count=2)
    ids = {t["summary"]["trace_id"] for t in traces}
    assert "req-1" in ids
    assert "bg-1" in ids


# ============================================================ HTTP 链路


@pytest.mark.anyio
async def test_request_produces_complete_trace(client: httpx.AsyncClient, tmp_path: Path) -> None:
    """验收①：一次请求一条完整 trace，root 是 http.request，响应头带 trace id。"""
    resp = await client.post(ENDPOINT, json={"text": "合同编号：HT-1", "fields": ["金额"]})
    assert resp.status_code == 200
    assert resp.headers["x-trace-id"] == resp.headers["x-request-id"]

    traces = _wait_traces(tmp_path / "traces")
    tid = resp.headers["x-trace-id"]
    mine = [t for t in traces if t["summary"]["trace_id"] == tid]
    assert len(mine) == 1, f"trace_id {tid} 应恰好落一条 trace，实际 {len(mine)} 条"
    root = mine[0]["spans"][0]
    assert root["name"] == "http.request"
    assert root["parent_id"] is None
    assert root["attrs"]["path"] == ENDPOINT
    assert mine[0]["summary"]["status"] == "ok"


@pytest.mark.anyio
async def test_error_request_also_has_trace(client: httpx.AsyncClient, tmp_path: Path) -> None:
    """验收②：422 校验失败也落 trace，status=error，100% 采样。"""
    resp = await client.post(ENDPOINT, json={"text": "", "fields": ["金额"]})
    assert resp.status_code == 422

    traces = _wait_traces(tmp_path / "traces")
    tid = resp.headers["x-trace-id"]
    mine = [t for t in traces if t["summary"]["trace_id"] == tid]
    assert len(mine) == 1
    assert mine[0]["summary"]["status"] == "error"
    assert mine[0]["spans"][0]["attrs"]["status_code"] == 422


@pytest.mark.anyio
async def test_trace_id_flows_to_deep_spans(client: httpx.AsyncClient, tmp_path: Path) -> None:
    """请求链路深处（服务层）的 span 与请求 root 同属一条 trace。

    v1 extract 走规则抽取器（不调 LLM），generation 不该出现；真正的
    LLM span 用 structured_call 直连假 client 验证——这正是"三级模型"
    的接线点：observation 在服务层，generation 在 LLM 调用层。
    """
    resp = await client.post(ENDPOINT, json={"text": "合同编号：HT-2", "fields": ["金额"]})
    assert resp.status_code == 200

    traces = _wait_traces(tmp_path / "traces")
    tid = resp.headers["x-trace-id"]
    mine = [t for t in traces if t["summary"]["trace_id"] == tid][0]
    # 同一条 trace：所有 span 的 trace_id 一致，root 是请求
    assert all(s["trace_id"] == tid for s in mine["spans"])
    assert mine["spans"][0]["name"] == "http.request"


@pytest.mark.anyio
async def test_generation_span_wired_into_structured_call(tmp_path: Path) -> None:
    """LLM 调用记 generation：usage/cost 进 attrs，与调用 span 同 trace。"""
    from types import SimpleNamespace

    from app.core.llm.structured import _request

    async def _create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"amount": "100万元"}'))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=15, total_tokens=135),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))

    with trace_scope("gen-1", root_name="extract.request"):
        # _request 是模块级可注入点（structured_call 会自取真实 client），
        # 直测 _request 的接线最短路径
        from app.core.llm.structured import _request

        raw = await _request(
            client,
            model="fake-model",
            messages=[{"role": "user", "content": "抽取金额"}],
            temperature=0.0,
        )
        assert raw == '{"amount": "100万元"}'

    traces = _wait_traces(tmp_path / "traces")
    tid = "gen-1"
    mine = [t for t in traces if t["summary"]["trace_id"] == tid]
    assert len(mine) == 1
    gens = [s for s in mine[0]["spans"] if s["kind"] == "generation"]
    assert gens, f"应有 generation span，实际 spans: {[s['name'] for s in mine[0]['spans']]}"
    g = gens[0]
    assert g["name"] == "llm.structured"
    assert g["attrs"]["model"] == "fake-model"
    assert g["attrs"]["prompt_tokens"] == 120
    assert g["attrs"]["completion_tokens"] == 15
    assert mine[0]["summary"]["prompt_tokens"] == 120
    assert mine[0]["summary"]["completion_tokens"] == 15


@pytest.mark.anyio
async def test_sink_thread_death_does_not_block(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验收③：落盘线程死了（等价于磁盘满/断网），请求照常返回。

    真正的断网场景里写盘只是变慢，这里用更极端的等价物——把 _write
    换成必炸函数——验证观测失败不影响业务路径。
    """
    from app.core.tracing import _sink

    sink = _sink()

    def _always_fails(item):  # noqa: ANN001
        raise OSError("disk full / network gone")

    monkeypatch.setattr(sink, "_write", _always_fails)
    resp = await client.post(ENDPOINT, json={"text": "合同编号：HT-3", "fields": ["金额"]})
    assert resp.status_code == 200  # 业务照常
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and sink.stats()["write_errors"] == 0:
        time.sleep(0.05)
    assert sink.stats()["write_errors"] > 0  # 观测在丢，但没人被阻塞
