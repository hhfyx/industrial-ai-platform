"""质检 Agent 的两个真工具（W9 周四）：检索薄包装 + 只读 SQL。

**本文件的安全设计只有一个中心思想（周四必答问题）：**

> SQL 工具为什么必须是「表 + 模板 + 参数」而不是"让模型写任意 SQL"？
> ——因为**模型的输出是不可信输入**。让模型写 SQL 等于让外部输入直接
> 变成可执行语句：模型可能被提示注入、可能幻觉出破坏性语句、可能被
> 诱导读不该读的表。「模型写 SQL」在威胁模型里与"让用户写 SQL"同级别。
>
> 白名单 + 参数化 + 只读 + LIMIT 四道闸，是本周唯一涉及真实安全的地方：
> 前三天的工具是假数据（炸了也只是返回错误），今天开始接真东西
> （真检索管线、真 SQLite 文件）——**有真东西的地方才有真安全边界**。

## 模块结构

| 部分 | 内容 | 对应验收 |
| --- | --- | --- |
| `retrieve_spec_chunks` | RAG 检索的**薄包装**：不新增排序逻辑、不改 `retrieve()` 签名 | 20:00–20:25 |
| `ReadOnlySQLTool` | SQLite 只读查询：四道闸（SELECT-only / 参数化 / 表白名单 / 强制 LIMIT）+ timeout + mode=ro | 20:25–20:50 |
| `sql_injection_drill` | 三种注入攻击的演练入口（演示脚本调用） | 20:50–21:10 |

两个工具都接的是**真东西**（真检索管线 / 真磁盘 SQLite），但测试全部
离线（hashing embedder + 内存库 / 临时 SQLite 文件）——延续本仓库
"拔网线 pytest 全绿"的规矩。

## 统一错误信封（W11 周一升级版，契约在 `error_contract.py`）

    成功: {ok: true,  tool, data, outcome: "success"|"empty_result"[, hint]}
    失败: {ok: false, tool, error_type, message, hint, retryable, retry_after[, details]}

五类错误在本文件的具体落点（每类都要有**可触发**的路径，不能只有文档）：

| 类 | error_type | 本文件的触发点 |
| --- | --- | --- |
| 参数错 | invalid_json / invalid_arg | JSON 语法坏 / k 越界 / 表名列名不在白名单 |
| 权限错 | not_permitted | 库文件不可读（EACCES）、引擎侧 readonly 拒绝 |
| 超时 | timeout | 检索超 `RETRIEVE_TIMEOUT_S` / SQL 超 `SQL_TIMEOUT_S` |
| 空结果 | not_found / empty_result | 查无批次 → not_found；检索 0 命中 → ok+empty_result |
| 内部异常 | internal | 其余未预期异常 |

`not_permitted` 是本周新加的一类：它原先被映射成 `invalid_arg`
（"放弃这种写法"），但两者的恢复动作不同 —— 参数错该改参数，
权限错该换方案并告知用户。混在一类里，模型会对一个永远失败的请求反复改参。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, Field

from app.core.llm.error_contract import (
    classify_exception,
    tool_error,
    tool_ok,
)
from app.core.llm.tools import ToolExecutionError

# ============================================================ ① 检索工具（薄包装）


class RetrieveSpecChunksArgs(BaseModel):
    """`retrieve_spec_chunks` 的入参。k 上限 8：检索段按步数乘成本，源头限住。"""

    query: str = Field(
        min_length=2,
        max_length=120,
        description="检索词，用指标名或关键词，例如「抗压强度 判定」「含水量 限值」",
    )
    k: int = Field(default=3, ge=1, le=8, description="返回最相关的 k 条条款，默认 3")


RETRIEVE_DESCRIPTION: Final[str] = (
    "在产品规范知识库（RAG 索引）里检索**规范条款的原文片段**。"
    "什么时候用：需要某项指标的判定依据原文、限值出处、检测方法原文时，"
    "例如「抗压强度的判定标准原文」。"
    "什么时候不用：不要用它查某一批次的检测数值（那是 query_metrics 的事）；"
    "不要在不知道要查什么指标时盲搜。"
    "参数：query 是指标名/关键词组合（如「抗压强度 判定」），k 是返回条数（默认 3，最多 8）。"
    "返回：chunks 列表，每条含 id、source（来源文档）、page（页码，可能为 null）、"
    "text（截断到 500 字的原文）；**查不到时仍返回 ok=true 且 chunks 为空列表**，"
    "并附 hint 说明当前索引里有哪些文档 —— 空结果不是错误，是「索引里没有」这个事实。"
)


def _build_retriever() -> Any:
    """构造真检索器：hashing embedder + 内存库 + 内置语料。

    走的是**真管线**（load_markdown → DelimiterSplitter → HashingEmbedder →
    InMemoryVectorStore → Retriever），语料是内置的三份假规范文档
    （写到临时目录再 index_file，让 loader 按正常路径解析）。
    真实部署时把语料目录换成 docs 库即可，工具层零改动。
    """
    import tempfile

    from app.rag.embedding.hashing import HashingEmbedder
    from app.rag.indexing import Indexer
    from app.rag.retrieval import Retriever
    from app.rag.splitters.recursive import RecursiveSplitter  # 兼容别名，实为 DelimiterSplitter
    from app.rag.stores.memory import InMemoryVectorStore

    embedder = HashingEmbedder()
    store = InMemoryVectorStore()
    indexer = Indexer(embedder=embedder, store=store, splitter=RecursiveSplitter())
    retriever = Retriever(embedder=embedder, store=store)

    # 内置语料：三份规范文档。真实系统这里换成 docs 目录。
    corpus = {
        "GB50081_抗压强度.md": (
            "# 抗压强度判定\n\nC40 混凝土抗压强度合格判定限值为 ≥40.0 MPa。"
            "试件为 100mm 立方体，以三个试件的算术平均值作为该组强度值。\n\n"
            "## 复测规则\n\n单块低于限值但平均值达标时允许复测一次，复测仍不达标判为不合格。"
        ),
        "GB50082_含水量.md": (
            "# 含水量限值\n\n烘干法测定，105±5℃ 烘至恒重。含水率 ≤10.0% 判为合格。\n\n"
            "## 超差处理\n\n含水率超过限值应复测一次，复测仍超差判为不合格并通知质检主管。"
        ),
        "QCVIS_外观.md": (
            "# 外观检验\n\n目测检查表面蜂窝、麻面、裂缝。"
            "允许个别气孔，出现贯穿性裂缝直接判不合格，无需复测。"
        ),
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for name, content in corpus.items():
            (tmp_dir / name).write_text(content, encoding="utf-8")
            indexer.index_file(tmp_dir / name)
    return retriever


# 模块级单例：索引一次，进程内复用。检索器无状态（store 只读），并发安全。
_RETRIEVER: Any | None = None


def _get_retriever() -> Any:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = _build_retriever()
    return _RETRIEVER


def get_shared_retriever() -> Any:
    """共享检索器的公开入口（W12 周一：/api/v2/search 冷端点复用）。

    `_get_retriever` 是工具层的私有装配（下划线约定）；冷端点是第二个
    真实调用方，公开一个**具名入口**而不是跨模块摸私有名 ——
    私有名被外部引用的那一刻它就不是私有的了，约束会静默失效。
    语义与 `_get_retriever` 完全一致：单例、无状态、并发安全。
    """
    return _get_retriever()


#: 单条 chunk 文本的进上下文上限。检索管线不截断（嵌入要全文），
#: 但**回灌给模型**的 observation 必须有界 —— 与 react_loop 的两级截断同思想。
CHUNK_TEXT_MAX_CHARS: Final[int] = 500

#: 单次检索的超时（秒）。W11 周一新增：检索也是 IO，也会挂住循环。
#: 取 5s —— 比 SQL 的 3s 宽（向量检索比本地 SQLite 慢一个量级），
#: 但远小于循环的 TOOL_STEP_TIMEOUT_S（10s），保证超时能被归类成 timeout
#: 而不是被外层的通用超时兜底成 internal（那会丢掉"可退避重试"的语义）。
RETRIEVE_TIMEOUT_S: Final[float] = 5.0


def retrieve_spec_chunks(query: str, k: int = 3) -> dict[str, Any]:
    """RAG 检索的薄包装。**不新增排序逻辑，不改 retrieve() 签名** —— 这是任务原文。

    为什么是"薄包装"而不是"重新实现一个检索工具"：
    排序、相似度口径、top_k 语义已经在 W4-W7 的检索管线里定过版
    （`SearchHit.score` 是余弦、`DEFAULT_TOP_K` 有实测出处）。
    工具层再排一遍序 = 第二个排序真相来源，两边迟早漂移。
    工具层只做三件事：调管线 → 换形状 → 截文本。

    **空结果返回 ok=true + chunks=[] + hint**：检索无命中是正常业务状态
    （索引里就是没有），不是故障。把它当错误回灌，模型会无意义地重试；
    把它当事实回灌（附 hint 指路），模型能转行（换词 / 如实告知用户）。
    成功信封上会带 `outcome="empty_result"`（见 `execute_retrieve_tool`），
    让统计能把它和"有结果"分开数，同时又不改变模型看到的行为。
    """
    retriever = _get_retriever()
    result = retriever.retrieve(query, top_k=k)
    chunks = [
        {
            "id": hit.chunk.id,
            "source": hit.chunk.source,
            "page": hit.chunk.page,
            "text": hit.chunk.text[:CHUNK_TEXT_MAX_CHARS],
        }
        for hit in result.hits
    ]
    data: dict[str, Any] = {"query": query, "chunks": chunks, "count": len(chunks)}
    if not chunks:
        data["hint"] = (
            f"知识库中检索不到与「{query}」相关的内容。"
            "可以换关键词重试，或直接告知用户当前索引没有覆盖该主题。"
        )
    return data


# ============================================================ ② 只读 SQL 工具


class QueryMetricsSQLArgs(BaseModel):
    """`query_metrics_sql` 的入参。

    `table` 是 Literal 白名单：模型只能选表，不能写表名 ——
    "UNION SELECT 读别的表" 这类攻击在参数层面就不成立。
    `columns` 可选：不传 = 全列（模板里展开）。
    """

    table: Literal["inspections", "batches"] = Field(
        description="要查询的表（白名单）：inspections=检测记录（批次×指标×数值）；"
        "batches=批次台账（批次号、型号、生产日期）"
    )
    columns: list[str] = Field(
        default_factory=list,
        description="要返回的列（可选）。inspections 的列：batch_id, metric, value, unit, measured_at；"
        "batches 的列：batch_id, model, produced_at。不传返回全部列",
    )
    where_batch_id: str = Field(
        default="",
        max_length=32,
        description="按批次号过滤，如「B20241129-03」。留空返回全表（受 LIMIT 限制）",
    )
    where_metric: str = Field(
        default="",
        max_length=32,
        description="按指标名过滤（仅 inspections 表有效），如「抗压强度」。留空不过滤",
    )
    limit: int = Field(default=20, ge=1, le=100, description="返回行数上限，默认 20，最大 100")


QUERY_METRICS_SQL_DESCRIPTION: Final[str] = (
    "用**预置的只读 SQL 模板**查询质检数据库（SQLite，只读连接）。"
    "什么时候用：需要具体批次的检测数值、数值明细、批次台账信息时，"
    "例如「B20241129-03 查了哪些指标、数值多少」。"
    "什么时候不用：不要用它查规范条文和限值（那是 retrieve_spec_chunks 的事）；"
    "你不能自己写 SQL —— 本工具只接受表名和过滤参数，语句由模板生成。"
    "参数：table 只能选 inspections / batches；columns 可选列名子集；"
    "where_batch_id / where_metric 是过滤值（会参数化绑定，不是拼接）；limit 默认 20 最大 100。"
    "返回：columns（实际列名）、rows（行数据，值原生类型）、row_count、sql（实际执行的只读语句，供审计）。"
    "表或批次不存在返回 not_found 或空 rows（附 hint），不是错误。"
)


#: 每张表的列白名单：columns 参数里的每个名字都要在这里（第二道表级闸）。
_ALLOWED_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "inspections": ("batch_id", "metric", "value", "unit", "measured_at"),
    "batches": ("batch_id", "model", "produced_at"),
}

#: 已知批次号（演示数据；真实部署查台账表）。用于把"空结果"归类成 not_found 与否。
_KNOWN_BATCHES: Final[tuple[str, ...]] = ("B20241207-01", "B20241129-03")

#: **部署级权限白名单**（W11 周一新增）：schema 里能看到的表 ≠ 本次部署授权查的表。
#: 默认两张表都授权；生产环境用 `QA_METRICS_PERMITTED_TABLES=inspections`
#: 就能把台账表锁掉 —— 模型仍会在 schema 里看到 batches（schema 是静态的），
#: 但执行层会以 not_permitted 拒绝，并明确告诉它"换方案而不是改参数"。
#: 这与"表不在 Literal 白名单里"（invalid_arg）是两件事：前者是**没被授权**，
#: 后者是**参数写错了**，恢复动作不同，所以不能混成一类。
PERMITTED_TABLES_ENV: Final[str] = "QA_METRICS_PERMITTED_TABLES"


def permitted_tables() -> tuple[str, ...]:
    """当前部署授权的表。环境变量缺省 = 全部表。"""
    raw = os.environ.get(PERMITTED_TABLES_ENV, "").strip()
    if not raw:
        return tuple(_ALLOWED_COLUMNS)
    granted = tuple(t.strip() for t in raw.split(",") if t.strip())
    # 只保留真实存在的表名，防止环境变量写错时把权限放大
    return tuple(t for t in granted if t in _ALLOWED_COLUMNS)


#: 单条 SQL 的执行超时（秒）。SQLite 默认没有语句级超时，
#: 用中断回调模拟：超时后 interrupt() 让 step() 返回 SQLITE_INTERRUPT。
SQL_TIMEOUT_S: Final[float] = 3.0


class _SQLBlocked(ToolExecutionError):
    """SQL 安装闸门拒绝执行的统一异常，kind 直接用七类枚举里的名字。"""


class ReadOnlySQLTool:
    """SQLite 只读查询工具：**四道闸 + timeout + mode=ro**。

    ## 四道闸（任何一道不过，语句根本不会到 SQLite）

    | 闸 | 实现 | 拦什么 |
    | --- | --- | --- |
    | ① SELECT-only | 语句由**模板**拼装，模板只生成 SELECT；参数值走 `?` 绑定 | 任何写操作、多语句（execute 一次只跑一条） |
    | ② 参数化 | 所有值占位符 `?` + `params` 元组，**绝不 f-string 拼值** | `'; DROP TABLE…` 这类经典注入 —— 值永远是值，不是代码 |
    | ③ 表白名单 | `table` 参数是 Literal；`columns` 逐个对 `_ALLOWED_COLUMNS` 校验 | UNION SELECT 读别的表、子查询越权 |
    | ④ 强制 LIMIT | limit 参数 ge=1 le=100，模板里 `LIMIT ?` 参数化 | 超长 LIKE / 全表扫描把上下文撑爆 |

    ## mode=ro 的位置

    打开连接用 `sqlite3.connect(f"file:{path}?mode=ro", uri=True)` ——
    **这是第五道闸，也是最硬的一道**：前四道是"我的代码的纪律"，
    mode=ro 是"数据库引擎的物理权限"。就算我的模板有 bug 拼出了写语句，
    SQLite 引擎本身也会拒绝（readonly database 错误）。代码纪律与引擎
    权限双层防护，缺一层都不叫安全边界。

    ## 为什么不用一条正则拦"危险关键词"

    黑名单正则（拦 DROP/DELETE/;）是**必输的防御**：绕过姿势无穷
    （注释、编码、pragma、attach……），而且黑名单挡不住合法但致命的
    SELECT（`SELECT load_extension(...)`）。白名单模板从根上不存在
    "用户可控的语句结构"，所以没有可绕的面。
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    # ---------- 闸门（私有，先于任何 SQL 执行） ----------

    def _check_table(self, table: str) -> tuple[str, ...]:
        allowed = _ALLOWED_COLUMNS.get(table)
        if allowed is None:
            raise _SQLBlocked(
                f"表 {table!r} 不在白名单里；允许的表：{sorted(_ALLOWED_COLUMNS)}",
                kind="invalid_arg",
            )
        # 闸③'（W11 周一新增）：部署级权限。schema 里能看到的表不代表本次部署授权查。
        # 用 not_permitted 而不是 invalid_arg —— 改参数永远过不去，模型该换方案。
        granted = permitted_tables()
        if table not in granted:
            raise _SQLBlocked(
                f"表 {table!r} 在当前部署里未被授权查询；已授权的表：{list(granted)}。"
                "这是权限问题，不是参数问题 —— 改参数无效，请换已授权的表或如实告知用户",
                kind="not_permitted",
            )
        return allowed

    def _check_columns(self, table: str, columns: list[str]) -> list[str]:
        allowed = _ALLOWED_COLUMNS[table]
        if not columns:
            return list(allowed)
        unknown = [c for c in columns if c not in allowed]
        if unknown:
            raise _SQLBlocked(
                f"列 {unknown} 不在表 {table} 的白名单里；允许的列：{list(allowed)}",
                kind="invalid_arg",
            )
        # 保留调用方给定的顺序，但只允许白名单里的列
        return list(columns)

    @staticmethod
    def _check_no_forbidden_payload(values: tuple[Any, ...]) -> None:
        """参数值里不允许出现 SQL 结构关键字。

        参数化本身已经让值永远是值（第二道闸），这一条是**纵深防御**：
        模型可能把 `; DROP TABLE x` 塞进 where_batch_id —— 值绑定后它只是
        一段没有意义的字符串，查不到任何行。但"查不到"和"被拒绝"是两种
        不同的回灌：拒绝（invalid_arg）告诉模型"这个输入形状不对"，
        比让它对着 0 行结果瞎猜更快收敛。
        """
        pattern = re.compile(
            r";|--|/\*|\b(drop|delete|insert|update|alter|create|attach|pragma)\b", re.IGNORECASE
        )
        for value in values:
            if isinstance(value, str) and pattern.search(value):
                raise _SQLBlocked(
                    f"过滤参数包含 SQL 结构关键字，已拒绝：{value[:40]!r}。"
                    "过滤值只能是普通文本（批次号/指标名）",
                    kind="invalid_arg",
                )

    # ---------- 模板拼装（值不参与拼装，只有列名/表名参与，而它们都过了白名单） ----------

    def _build_query(self, args: QueryMetricsSQLArgs) -> tuple[str, tuple[Any, ...]]:
        """从**白名单内的标识符 + 参数化值**拼一条 SELECT。

        参与字符串拼装的只有：表名、列名 —— 两者都过了白名单校验，
        不存在"模型可控的标识符"。所有**值**（batch_id/metric/limit）
        全部走 `?` 占位符绑定。这就是"表 + 模板 + 参数"的字面落地。
        """
        self._check_table(args.table)  # 闸③：表白名单（拒绝在列校验之前，报错更聚焦）
        cols = self._check_columns(args.table, args.columns)

        where_clauses: list[str] = []
        params: list[Any] = []
        if args.where_batch_id:
            where_clauses.append("batch_id = ?")
            params.append(args.where_batch_id)
        if args.where_metric and args.table == "inspections":
            where_clauses.append("metric = ?")
            params.append(args.where_metric)

        sql = f"SELECT {', '.join(cols)} FROM {args.table}"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        # 强制 LIMIT（第四道闸）：占位符绑定，模型给的 limit 是值不是结构
        sql += " LIMIT ?"
        params.append(args.limit)

        self._check_no_forbidden_payload(tuple(params))
        return sql, tuple(params)

    # ---------- 执行 ----------

    def _execute_sync(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        """同步执行（mode=ro），供 to_thread + 超时包装调用。"""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")  # 引擎层再锁一层：连接级只读
        try:
            cur = conn.execute(sql, params)
            columns = [desc[0] for desc in cur.description or ()]
            rows = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
            return rows
        finally:
            conn.close()

    async def execute(self, args: QueryMetricsSQLArgs) -> dict[str, Any]:
        """带超时地执行一次查询，返回统一信封的 data 部分。"""
        sql, params = self._build_query(args)
        started = time.perf_counter()
        try:
            rows = await asyncio.wait_for(
                asyncio.to_thread(self._execute_sync, sql, params),
                timeout=SQL_TIMEOUT_S,
            )
        except TimeoutError:
            raise _SQLBlocked(
                f"查询超过 {SQL_TIMEOUT_S}s 被强制中断（通常是过滤条件太宽导致全表扫描）",
                kind="timeout",
            ) from None
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        data: dict[str, Any] = {
            "sql": sql,
            "columns": list(_ALLOWED_COLUMNS[args.table])
            if not args.columns
            else list(args.columns),
            "rows": rows,
            "row_count": len(rows),
            "elapsed_ms": elapsed_ms,
        }
        if not rows:
            data["hint"] = (
                f"查询返回 0 行。已知批次号：{list(_KNOWN_BATCHES)}。"
                "确认批次号是否正确，或改用 retrieve_spec_chunks 查规范信息。"
            )
            if args.where_batch_id and args.where_batch_id not in _KNOWN_BATCHES:
                raise _SQLBlocked(
                    f"查无批次 {args.where_batch_id!r}；已知批次：{list(_KNOWN_BATCHES)}",
                    kind="not_found",
                )
            # 周六首轮改进（改动 #2）：批次存在但**指标过滤**无匹配 →
            # hint 明说"该批次没有这个指标的数据"，并提示"转述 0 行结果
            # 就是正确答案"。首轮 T10 实测：模型收到 0 行后多花一步
            # 全查再自己归纳——hint 直接给结论能省一步。
            if args.where_metric:
                data["hint"] = (
                    f"批次 {args.where_batch_id} 存在，但没有指标 {args.where_metric!r} 的检测记录。"
                    "0 行就是这个问题的答案：如实告知用户该批次没有此指标数据，"
                    "不要编造数值。可去掉 where_metric 查看该批次已有的指标。"
                )
        return data


# ============================================================ 统一信封（W11 周一升级）

#: 本模块的工具名常量：信封里 tool 字段的唯一来源（拼错会在测试里立刻暴露）。
RETRIEVE_TOOL: Final[str] = "retrieve_spec_chunks"
SQL_TOOL: Final[str] = "query_metrics_sql"


async def execute_retrieve_tool(raw_arguments: str) -> dict[str, Any]:
    """`retrieve_spec_chunks` 的统一信封执行。永不抛异常，错误是数据。

    `model_validate_json` 会把 JSON 语法错误也归到 ValidationError
    （fields 里 loc 为空、reason 带 Invalid JSON）—— 但它与"参数形状错"
    是两种不同的模型修正动作（重生成合法 JSON vs 改字段），
    所以先用显式 `json.loads` 分离 invalid_json。

    W11 周一的五类落点：参数错（两条）、超时（`RETRIEVE_TIMEOUT_S`）、
    空结果（ok + outcome）、内部异常（兜底）。权限错在本工具的真实
    触发是"索引不可读"，由 `classify_exception` 从异常里认出来。
    """
    from pydantic import ValidationError

    try:
        json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return tool_error(
            RETRIEVE_TOOL,
            "invalid_json",
            f"参数不是合法 JSON：{exc.msg}（位置 {exc.pos}）",
            details={"raw": (raw_arguments or "")[:200]},
        )
    try:
        args = RetrieveSpecChunksArgs.model_validate_json(raw_arguments or "{}")
    except ValidationError as exc:
        return tool_error(
            RETRIEVE_TOOL,
            "invalid_arg",
            "参数不符合工具签名",
            details={"fields": _field_details(exc)},
        )
    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(retrieve_spec_chunks, args.query, args.k),
            timeout=RETRIEVE_TIMEOUT_S,
        )
    except TimeoutError:
        return tool_error(
            RETRIEVE_TOOL,
            "timeout",
            f"检索超过 {RETRIEVE_TIMEOUT_S}s 被强制中断（索引规模或 query 过宽）",
        )
    except ToolExecutionError as exc:
        return tool_error(RETRIEVE_TOOL, exc.kind, exc.message)
    except Exception as exc:  # noqa: BLE001 —— 未预期异常也要变成数据回灌
        return tool_error(
            RETRIEVE_TOOL,
            classify_exception(exc),
            f"检索管线错误：{type(exc).__name__}: {exc}",
        )
    empty = not data.get("chunks")
    return tool_ok(RETRIEVE_TOOL, data, empty=empty)


def _field_details(exc: Any) -> list[dict[str, Any]]:
    """Pydantic 的 ValidationError → 可读字段明细（hint 之外的结构化指路）。"""
    return [
        {
            "field": ".".join(str(p) for p in err["loc"]),
            "reason": err["msg"],
            "got": err.get("input"),
        }
        for err in exc.errors()
    ]


async def execute_sql_tool(raw_arguments: str) -> dict[str, Any]:
    """`query_metrics_sql` 的统一信封执行。永不抛异常，错误是数据。

    四道闸的拒绝统一映射为 invalid_arg（模型该做的是换参数写法）；
    数据侧的 not_found 单独成类（模型该做的是换批次号或如实告知）；
    引擎侧权限拒绝（mode=ro 拦下写操作、库文件不可读）映射为
    not_permitted —— 这是 W11 周一新增的一类，以前它被塞进 invalid_arg，
    导致模型对着一个永远不可能成功的请求反复改参数。
    """
    from pydantic import ValidationError

    try:
        json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return tool_error(
            SQL_TOOL,
            "invalid_json",
            f"参数不是合法 JSON：{exc.msg}（位置 {exc.pos}）",
            details={"raw": (raw_arguments or "")[:200]},
        )
    try:
        args = QueryMetricsSQLArgs.model_validate_json(raw_arguments or "{}")
    except ValidationError as exc:
        return tool_error(
            SQL_TOOL,
            "invalid_arg",
            "参数不符合工具签名",
            details={"fields": _field_details(exc)},
        )
    # 数据库路径：环境变量优先（部署时注入），缺省用模块旁的演示库
    # （首次使用时自动建库 —— 演示数据与 qa_tools 的假数据一致）。
    db_path = os.environ.get("QA_METRICS_DB_PATH") or str(_DEFAULT_DB_PATH)
    if not Path(db_path).exists():
        build_demo_db(db_path)
    tool = ReadOnlySQLTool(db_path)
    try:
        data = await tool.execute(args)
    except ToolExecutionError as exc:
        return tool_error(SQL_TOOL, exc.kind, exc.message)
    except Exception as exc:  # noqa: BLE001
        return tool_error(
            SQL_TOOL,
            classify_exception(exc),
            f"SQL 执行错误：{type(exc).__name__}: {exc}",
        )
    return tool_ok(SQL_TOOL, data)


# ============================================================ 演示数据库


#: 演示库路径（临时文件，测试与演示共用同一建库函数）。
_DEFAULT_DB_PATH: Final[Path] = Path(__file__).parent / "_qa_demo.sqlite"


def build_demo_db(db_path: str | Path | None = None) -> Path:
    """建一个演示用 SQLite 库（两张表 + 假数据）。测试用临时目录传 path。

    数据与 qa_tools.py 的假数据表一致（B20241207-01 合格 / B20241129-03 超差），
    这样周四和周三的工具能互相印证。
    """
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                batch_id TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                measured_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                produced_at TEXT NOT NULL
            );
            DELETE FROM inspections;
            DELETE FROM batches;
            INSERT INTO batches VALUES
                ('B20241207-01', 'C40 预制构件', '2024-12-07'),
                ('B20241129-03', 'C40 预制构件', '2024-11-29');
            INSERT INTO inspections VALUES
                ('B20241207-01', '抗压强度', 42.5, 'MPa', '2024-12-08 09:30'),
                ('B20241207-01', '含水量', 8.2, '%', '2024-12-08 10:15'),
                ('B20241129-03', '抗压强度', 36.1, 'MPa', '2024-11-30 09:00'),
                ('B20241129-03', '含水量', 11.4, '%', '2024-11-30 09:40');
            """
        )
        conn.commit()
    finally:
        conn.close()
    return path


# ============================================================ 注入演练（20:50–21:10）


#: 三种攻击的构造 + 各自被哪道闸拦住。演示与测试共用这一份。
INJECTION_CASES: Final[list[dict[str, str]]] = [
    {
        "name": "经典写注入",
        "payload": "B20241129-03'; DROP TABLE inspections;--",
        "attack": "把 DROP 塞进字符串值里，靠引号逃逸出值上下文",
        "blocked_by": "闸②参数化（值永远不是代码）+ 纵深：闸⑤值关键字检查直接拒绝",
        "where": '{"table": "inspections", "where_batch_id": "B20241129-03\'; DROP TABLE inspections;--"}',
    },
    {
        "name": "UNION 越权读表",
        "payload": "inspections UNION SELECT batch_id, model, produced_at, 1, 1 FROM batches",
        "attack": "把 batches 表的列 UNION 进结果里读出来",
        "blocked_by": "闸③表白名单（table 只能是 Literal 枚举值，语句结构模型不可控）",
        "where": '{"table": "inspections UNION SELECT batch_id, model, produced_at, 1, 1 FROM batches", "where_batch_id": "B20241129-03"}',
    },
    {
        "name": "超长 LIKE 撑爆上下文",
        "payload": "x" * 200,
        "attack": "超长过滤值 + 无 LIMIT 会让结果集和上下文一起爆炸",
        "blocked_by": "闸④强制 LIMIT（≤100）+ Pydantic max_length=32 拦超长值",
        "where": '{"table": "inspections", "where_batch_id": "' + "x" * 200 + '"}',
    },
]


async def sql_injection_drill() -> list[dict[str, Any]]:
    """跑三种注入攻击，返回「攻击名 → 被哪道闸拦下 → error_type」的表。

    每种攻击都要断言**两件事**：被拒绝（ok=false）且 error_type 可读。
    只拦下但返回含糊错误，模型同样会瞎猜 —— 那不是合格的边界。
    """
    results: list[dict[str, Any]] = []
    for case in INJECTION_CASES:
        r = await execute_sql_tool(case["where"])
        blocked = r["ok"] is False
        results.append(
            {
                "name": case["name"],
                "attack": case["attack"],
                "blocked_by": case["blocked_by"],
                "blocked": blocked,
                "error_type": r.get("error_type"),
                "message": (r.get("error") or {}).get("message", "")[:80],
            }
        )
    return results


__all__ = [
    "CHUNK_TEXT_MAX_CHARS",
    "INJECTION_CASES",
    "PERMITTED_TABLES_ENV",
    "QUERY_METRICS_SQL_DESCRIPTION",
    "RETRIEVE_DESCRIPTION",
    "RETRIEVE_TIMEOUT_S",
    "RETRIEVE_TOOL",
    "SQL_TIMEOUT_S",
    "SQL_TOOL",
    "ReadOnlySQLTool",
    "RetrieveSpecChunksArgs",
    "QueryMetricsSQLArgs",
    "build_demo_db",
    "execute_retrieve_tool",
    "execute_sql_tool",
    "permitted_tables",
    "retrieve_spec_chunks",
    "sql_injection_drill",
]
