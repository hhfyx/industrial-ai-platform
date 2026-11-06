"""会话持久化：`sessions` / `turns` 两张表，按 session_id 恢复。

## 表结构与 UI message dict 一一对应（零转换恢复）

Streamlit 的 `st.session_state` 每次刷新重建 —— 持久化后恢复路径必须能把
`session_id` 对应的 turns 灌回 `session_state.messages`。所以 turns 表的列
与 UI 存的 message dict（`streamlit_app._append_answer` 存的那份）
**字段同名同义**：role / content / citations / contexts / invalid_refs /
refused / timing / rewrite_of / created_at。恢复时逐行 `json.loads` 回 dict，
不需要任何字段改名或重排。

## 为什么是 sqlite 而不是 jsonl / 一堆文件

仿 `embedding/cache.py` 的既有结论：单文件、单事务批量写、
`ls` 不会出现一千个文件。会话规模（几十轮 × 每轮几 KB）远够用，
真要清就删 `data/sessions.sqlite`。

## 为什么 turns 是逐行追加而不是整会话覆写

一问一答各一行（与 UI 的 messages 列表逐项对应）。整会话覆写的问题：
"第 5 轮崩了"会把前 4 轮也一起丢。逐行追加让崩溃边界天然清晰，
恢复 = `SELECT ... ORDER BY seq`，与 UI 列表顺序严格一致。

## rewrite_of 字段

本行消息是"改写层的产物"时的出处记录：用户消息行记它被改写**前**的原文
（改写发生在检索输入上，不改用户看到的消息）；为空表示未经改写
（首轮 / 降级 / 歧义 / 普通聊天）。W7 纪律：改写层必须可回放。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH: str = "data/sessions/sessions.sqlite"
"""默认库文件位置（相对项目根）。**放在 sessions/ 子目录里不是随手**：
docker-compose 把 `data/sessions/` 挂成持久 volume（W8 周六）——
库文件在挂载点**之内**才随 volume 存活；放在 `data/` 下则容器重建即丢
（实测踩过：库落在 /app/data/sessions.sqlite，volume 挂的是 /app/data/sessions/，
重建容器历史全丢）。独立文件，**不混进** embedding 缓存库。"""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_db_path() -> Path:
    return _project_root() / DEFAULT_DB_PATH


def new_session_id() -> str:
    """新会话 id。uuid4 短横线全保留 —— 可读性优先于长度，这不是高频路径。"""
    return str(uuid.uuid4())


def _utc_now_iso() -> str:
    """UTC ISO 时间戳。落盘统一 UTC，展示层自己转本地 —— 与 params_result.json 同口径。"""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class SessionStore:
    """会话存储（sqlite 后端）。一个连接 + 一把锁（同 `EmbeddingCache` 的做法）。

    Streamlit 每次交互重跑脚本，但 store 只做毫秒级单行插入，
    不值得引入连接池；`check_same_thread=False` 是给
    `asyncio.to_thread` 里的调用留的（与本仓库 cache.py 同一条理由）。
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              session_id  TEXT PRIMARY KEY,
              created_at  TEXT NOT NULL,
              title       TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS turns (
              seq         INTEGER NOT NULL,          -- 会话内消息序号，0 起，严格递增
              session_id  TEXT NOT NULL REFERENCES sessions(session_id),
              role        TEXT NOT NULL,             -- 'user' | 'assistant'
              content     TEXT NOT NULL,
              citations   TEXT NOT NULL DEFAULT '[]',-- JSON 数组
              contexts    TEXT NOT NULL DEFAULT '[]',-- JSON 数组
              invalid_refs TEXT NOT NULL DEFAULT '[]',
              refused     INTEGER NOT NULL DEFAULT 0,
              timing      TEXT NOT NULL DEFAULT '{}',-- JSON 对象
              rewrite_of  TEXT NOT NULL DEFAULT '',  -- 改写前原文（空 = 未经改写）
              created_at  TEXT NOT NULL,
              PRIMARY KEY (session_id, seq)
            );

            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, seq);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ 写

    def create_session(self, session_id: str, *, title: str = "") -> None:
        """登记会话。重复创建不报错（幂等）—— UI 每帧都可能调。"""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at, title) VALUES (?, ?, ?)",
                (session_id, _utc_now_iso(), title),
            )
            self._conn.commit()

    def append_turn(self, session_id: str, message: dict[str, Any]) -> int:
        """追加一条消息，返回分配的 seq。

        **seq 由本方法分配**（SELECT MAX + 1，在锁内），调用方不传 ——
        两个来源（UI 主流程与恢复逻辑）并发追加时序号也不会撞。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            seq = int(row[0]) + 1
            self._conn.execute(
                "INSERT INTO turns (seq, session_id, role, content, citations, contexts,"
                " invalid_refs, refused, timing, rewrite_of, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    seq,
                    session_id,
                    str(message.get("role", "user")),
                    str(message.get("content", "")),
                    json.dumps(message.get("citations") or [], ensure_ascii=False),
                    json.dumps(message.get("contexts") or [], ensure_ascii=False),
                    json.dumps(message.get("invalid_refs") or [], ensure_ascii=False),
                    1 if message.get("refused") else 0,
                    json.dumps(message.get("timing") or {}, ensure_ascii=False),
                    str(message.get("rewrite_of") or ""),
                    str(message.get("created_at") or _utc_now_iso()),
                ),
            )
            self._conn.commit()
            return seq

    # ------------------------------------------------------------------ 读

    def load_turns(self, session_id: str) -> list[dict[str, Any]]:
        """按序恢复一个会话的全部消息。**结构与 UI message dict 一一对应**。

        未登记的 session 返回空列表（不抛）—— "恢复一个不存在的会话"
        在 UI 里就是"开一个新会话"，不是错误。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, citations, contexts, invalid_refs, refused,"
                " timing, rewrite_of, created_at FROM turns WHERE session_id = ?"
                " ORDER BY seq",
                (session_id,),
            ).fetchall()
        return [
            {
                "role": role,
                "content": content,
                "citations": json.loads(citations),
                "contexts": json.loads(contexts),
                "invalid_refs": json.loads(invalid_refs),
                "refused": bool(refused),
                "timing": json.loads(timing),
                "rewrite_of": rewrite_of,
                "created_at": created_at,
            }
            for (
                role,
                content,
                citations,
                contexts,
                invalid_refs,
                refused,
                timing,
                rewrite_of,
                created_at,
            ) in rows
        ]

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return row is not None

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = [
    "DEFAULT_DB_PATH",
    "SessionStore",
    "default_db_path",
    "new_session_id",
]
