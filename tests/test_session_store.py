"""会话持久化的离线测试：建表 / 追加 / 恢复 / 幂等 / 零转换。

`tmp_path` 建独立库文件，不碰 `data/`。UI 恢复路径的验收
（"刷新后历史还在"）依赖这里锁住的契约：
turns 表行结构与 UI message dict **同名同义**，恢复即灌回。
"""

from __future__ import annotations

from app.rag.session_store import SessionStore, new_session_id


def _message(role: str, content: str, **extra: object) -> dict:
    """构造一个与 streamlit_app 相同形状的 message dict。"""
    msg: dict = {"role": role, "content": content}
    msg.update(extra)
    return msg


def test_roundtrip_preserves_every_field(tmp_path) -> None:
    """写入一条完整消息 → 恢复 → 每个字段逐项相等（不是抽查有值）。"""
    store = SessionStore(tmp_path / "s.sqlite")
    sid = new_session_id()
    store.create_session(sid)

    original = _message(
        "assistant",
        "违约金按合同第 5 条计算。",
        citations=[{"index": 1, "source": "合同.pdf", "page": 5, "chunk_id": "abc_5_0"}],
        contexts=[{"index": 1, "text": "甲方应支付违约金", "source": "合同.pdf", "page": 5}],
        invalid_refs=[3],
        refused=False,
        timing={"ttft_ms": 210, "total_ms": 1450},
        rewrite_of="",
        created_at="2026-12-07T20:30:00.000+00:00",
    )
    store.append_turn(sid, original)

    loaded = store.load_turns(sid)
    assert len(loaded) == 1
    assert loaded[0] == original, "恢复结果必须与原 dict 逐字段相等（零转换契约）"
    store.close()


def test_seq_ordering_and_multi_turn(tmp_path) -> None:
    """多轮按 seq 严格有序；user/assistant 交替的顺序恢复后不乱。"""
    store = SessionStore(tmp_path / "s.sqlite")
    sid = new_session_id()
    store.create_session(sid)
    msgs = [
        _message("user", "它的公差是多少？", rewrite_of="它的公差是多少？", citations=[]),
        _message("assistant", "HX-200 的公差是 ±0.01mm。"),
        _message("user", "再详细点"),
        _message("assistant", "指 ±0.01mm 是直径公差……"),
    ]
    for m in msgs:
        store.append_turn(sid, m)

    loaded = store.load_turns(sid)
    assert [m["content"] for m in loaded] == [m["content"] for m in msgs]
    assert [m["role"] for m in loaded] == ["user", "assistant", "user", "assistant"]
    store.close()


def test_seq_assigned_monotonically_under_same_store(tmp_path) -> None:
    """seq 由 store 分配且递增 —— 调用方不传序号就不会撞。"""
    store = SessionStore(tmp_path / "s.sqlite")
    sid = new_session_id()
    store.create_session(sid)
    seqs = [store.append_turn(sid, _message("user", f"第{i}问")) for i in range(5)]
    assert seqs == [0, 1, 2, 3, 4]
    store.close()


def test_create_session_is_idempotent(tmp_path) -> None:
    """重复登记同一 session 不报错（UI 每帧都可能调）。"""
    store = SessionStore(tmp_path / "s.sqlite")
    sid = new_session_id()
    store.create_session(sid)
    store.create_session(sid)  # 第二次必须静默
    store.append_turn(sid, _message("user", "你好"))
    assert len(store.load_turns(sid)) == 1
    store.close()


def test_load_unknown_session_returns_empty(tmp_path) -> None:
    """未登记的 session 返回空列表不抛 —— 恢复不存在的会话 = 开新会话。"""
    store = SessionStore(tmp_path / "s.sqlite")
    assert store.load_turns("no-such-session") == []
    assert store.session_exists("no-such-session") is False
    store.close()


def test_default_values_survive_roundtrip(tmp_path) -> None:
    """UI 的早期消息没有 citations/timing 等字段时，落库取默认值且可恢复。"""
    store = SessionStore(tmp_path / "s.sqlite")
    sid = new_session_id()
    store.create_session(sid)
    store.append_turn(sid, _message("user", "只有 role 和 content"))

    loaded = store.load_turns(sid)[0]
    assert loaded["citations"] == []
    assert loaded["contexts"] == []
    assert loaded["invalid_refs"] == []
    assert loaded["refused"] is False
    assert loaded["timing"] == {}
    assert loaded["rewrite_of"] == ""
    assert loaded["created_at"], "created_at 落库时自动补"
    store.close()


def test_rewrite_of_roundtrip(tmp_path) -> None:
    """rewrite_of 进出无损 —— 改写层的可回放落在这一个字段上。"""
    store = SessionStore(tmp_path / "s.sqlite")
    sid = new_session_id()
    store.create_session(sid)
    store.append_turn(sid, _message("user", "HX-200 的公差是多少？", rewrite_of="它的公差是多少？"))
    loaded = store.load_turns(sid)[0]
    assert loaded["content"] == "HX-200 的公差是多少？"
    assert loaded["rewrite_of"] == "它的公差是多少？"
    store.close()
