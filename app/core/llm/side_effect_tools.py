"""质检 Agent 的两个副作用工具（W9 周五）：报告落盘 + 通知出站。

**本模块的中心设计是"副作用分离"（21:15–21:35 的验收项）**：

> 通知/写文件是有副作用的动作：**先落记录、再执行，重试不重复。**

顺序不能反。先执行再记账的话，"执行成功但记账前进程死了"会留下一个
已发出但无记录的通知 —— 重试时没人拦它，主管收到两遍。
先记账（幂等账本里写入"进行中"）再执行，任何时刻崩溃，恢复方都能从
账本判断该重做还是该停。这就是 W8 D4（投递幂等）在 Agent 侧的复现：
**幂等不是"重试不报错"，是"重试不重复产生效果"。**

## 与周三假工具的关系

周三 `qa_tools.py` 的 `generate_report` / `notify` 是**哑实现**
（渲染返回字符串、通知写进程内 dict）。今天换真实现：

| 工具 | 周三（哑） | 周五（真） |
| --- | --- | --- |
| `generate_report` | 返回 markdown 字符串 | 落盘 `reports/{task_id}.md`，返回路径与字数 |
| `notify` | 进程内 dict 账本 | HTTP POST 到 mock 接收端 + 文件账本 + 重试 1 次 |

对外**签名与描述不变**（四要素描述原样复用，只改"返回"段）——
换实现不换契约，Agent 层（周六接入）无需感知。

## 报告工具的诚实性约束

内容必须来自前面的工具结果（在描述里写清"不要编造数据"）——
报告是质检结论的载体，编造数据比工具失败严重得多。工具层能做的
是**留痕**：把报告全文落盘，事后可审计"模型写进报告的数字是否
来自 query_metrics 的返回"。

## W11 周一的五类错误落点（契约在 `error_contract.py`）

| 类 | error_type | 本文件的触发点 |
| --- | --- | --- |
| 参数错 | invalid_json / invalid_arg | JSON 坏 / channel 不在白名单 / sections 为空 |
| 权限错 | **not_permitted** | 报告目录不可写（EACCES / 只读介质）；通知端点返回 401/403 |
| 超时 | **timeout** | 报告写盘超 `REPORT_WRITE_TIMEOUT_S`；通知 HTTP 超时（重试后仍超）|
| 空结果 | **empty_result** | sections 全部为空（没有内容可写）；通知正文为空（没有内容可发）|
| 内部异常 | internal | 其余未预期异常 |

两条与 W10 不同的判断，理由都写在触发点上：

1. **端口不可达 = internal，401/403 = not_permitted。** 前者重试可能好，
   后者重试一万次都是 403 —— 归错类，模型就会对一个永远失败的端点反复重试。
2. **sections 空 = empty_result 而不是 invalid_arg。** 参数形状是对的
   （是个合法的空数组），缺的是**内容**。恢复动作是"先去取数据"，
   不是"把参数改个形状"。
"""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, Field, ValidationError

from app.core.llm.error_contract import classify_exception, tool_error, tool_ok
from app.core.llm.tools import ToolExecutionError

# ============================================================ ① 报告工具（落盘真实现）


class ReportSection(BaseModel):
    """报告的一节。字段名用 content（与周三 body 同义，描述里注明）。"""

    heading: str = Field(min_length=1, max_length=60, description="小节标题，如「检测结果比对」")
    content: str = Field(min_length=1, max_length=4000, description="小节正文，Markdown 文本")


class GenerateReportFileArgs(BaseModel):
    """`generate_report` 落盘版的入参。task_id 是报告文件的主键。"""

    task_id: str = Field(
        min_length=4,
        max_length=64,
        description="任务标识，报告文件名的主键，如「qa-B20241129-03」。同一 task_id 重跑会覆盖",
    )
    title: str = Field(
        min_length=1, max_length=80, description="报告标题，如「批次 B20241129-03 质检报告」"
    )
    sections: list[ReportSection] = Field(
        min_length=1,
        max_length=12,
        description="报告小节列表。**所有数值必须来自前面工具的返回**，不要编造数据；"
        "数据缺失的指标写「数据缺失，本批次不可判定」，不要填默认值",
    )


GENERATE_REPORT_FILE_DESCRIPTION: Final[str] = (
    "把已整理好的小节渲染成质检报告并**写入 reports/ 目录**（Markdown 文件）。"
    "什么时候用：当判定结论已经形成、各小节的正文内容已经明确时。"
    "什么时候不用：不要用它替你做判断（判定逻辑在你的推理里，不在工具里）；"
    "结论未定时不要先渲染报告。"
    "**内容纪律：报告里的所有数值和结论必须来自你前面调用工具拿到的结果，"
    "不要编造数据；查不到的指标如实写「数据缺失」，不要填默认值。**"
    "参数：task_id 是文件名主键（如 qa-B20241129-03）；title 是报告标题；"
    "sections 是 [{heading, content}] 列表，content 为 Markdown 正文。"
    "返回：path（报告文件的相对路径）、chars（正文字数）、sections（节数）、"
    "overwritten（是否覆盖了同 task_id 的旧报告）。"
    "**拿到 path 之后必须在最终答案里把路径原样写出来**"
    "（例如「已写入 reports/qa-B20241129-03.md」）—— 用户要的是那份文件，"
    "只说「报告已生成」等于没交付，他找不到东西。"
    "sections 为空或正文为空返回 invalid_arg。"
)


def render_report_markdown(title: str, sections: list[dict[str, str]]) -> str:
    """渲染 Markdown。与周三 qa_tools.generate_report 同一排版，抽出复用。"""
    parts = [f"# {title}", ""]
    for idx, section in enumerate(sections, start=1):
        heading = str(section.get("heading", "")).strip()
        body = str(section.get("content") or section.get("body", "")).strip()
        if not heading or not body:
            raise ToolExecutionError(f"第 {idx} 节的 heading 或 content 为空", kind="invalid_arg")
        parts.append(f"## {idx}. {heading}")
        parts.append("")
        parts.append(body)
        parts.append("")
    return "\n".join(parts).rstrip()


#: 报告输出目录（相对工作目录）。真实部署时从配置注入。
REPORTS_DIR: Final[Path] = Path("reports")

#: 报告写盘超时（秒）。W11 周一新增：写盘也是 IO，机械盘/NFS/杀软扫描都能挂住。
#: 取 5s —— 远小于循环的 TOOL_STEP_TIMEOUT_S（10s），保证归类成 timeout
#: 而不是被外层兜底成 internal（那会丢掉"可退避重试"的语义）。
REPORT_WRITE_TIMEOUT_S: Final[float] = 5.0


def generate_report_file(
    task_id: str, title: str, sections: list[dict[str, str]], *, reports_dir: Path | None = None
) -> dict[str, Any]:
    """渲染并落盘报告。返回路径与字数（验收①的返回要求）。

    **先建目录再写文件**；文件名是 `{task_id}.md` —— 同 task_id 重跑覆盖
    （报告是"当前结论"的载体，不是追加日志；历史版本交给 git/备份）。

    两类"没有内容"在这里区分开（W11 周一）：
    - `sections` 是空列表 → **empty_result**（参数形状合法，缺的是内容）；
    - 某一节的 heading/content 为空串 → **invalid_arg**（参数形状不合格）。
    前者该"先去取数据"，后者该"把这一节补全"，恢复动作不同。
    """
    if not sections:
        raise ToolExecutionError(
            "报告没有任何小节（sections 为空），没有内容可写。"
            "先调用检索/查询工具拿到结论，再据此组织小节",
            kind="empty_result",
        )
    markdown = render_report_markdown(title, sections)
    target_dir = reports_dir or REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{task_id}.md"
    overwritten = path.exists()
    path.write_text(markdown, encoding="utf-8")
    return {
        "path": str(path).replace("\\", "/"),
        "chars": len(markdown),
        "sections": len(sections),
        "overwritten": overwritten,
    }


# ============================================================ ② 通知工具（HTTP + 幂等 + 重试）


NOTIFY_HTTP_DESCRIPTION: Final[str] = (
    "把质检结论通知**发送到通知服务**（HTTP 端点，带幂等去重与自动重试）。"
    "什么时候用：报告生成之后；渠道必须跟结论匹配 —— 合格/归档发 team 或 archive，"
    "超差/不合格必须发 supervisor。"
    "什么时候不用：报告还没生成就不要发通知（通知正文要引用报告结论）；"
    "不确定该发哪个渠道时不要猜，宁可如实告知用户。"
    "参数：channel 只能是 team / supervisor / archive；message 是通知正文；"
    "idempotency_key 是幂等键（同一事件复用同一键，如「batch-B20241129-03-fail」），"
    "重试时不要换新键。"
    "返回：delivered（是否送达）、channel、attempt（第几次尝试成功）、"
    "duplicate（是否为幂等拦截）。同一 idempotency_key 的重复发送会被账本拦截"
    "（conflict + duplicate=true），此时不需要重试。"
    "端点不可达时自动重试 1 次；仍失败返回结构化错误（error_type=internal），"
    "已写账本的记录不会重复投递。"
)


class NotifyHTTPArgs(BaseModel):
    """`notify` HTTP 版的入参。与周三 NotifyArgs 同形（channel 白名单不变）。"""

    channel: str = Field(
        description="通知渠道（白名单）：team=生产班组；supervisor=质检主管；archive=仅归档不发人"
    )
    message: str = Field(
        min_length=1, max_length=2000, description="通知正文，一句话说清结论和需要接收方做什么"
    )
    idempotency_key: str = Field(
        min_length=6,
        max_length=64,
        description="幂等键，同一事件必须复用同一个键，"
        "例如「batch-B20241129-03-fail」。重试时不要换新键",
    )


#: 通知渠道白名单（与周三一致）。
_ALLOWED_CHANNELS: Final[tuple[str, ...]] = ("team", "supervisor", "archive")

#: mock 接收端地址。本地演示用；真实部署换成通知服务的 URL。
NOTIFY_ENDPOINT: Final[str] = "http://127.0.0.1:8920/notify"

#: 幂等账本文件（JSON 行）。**文件而不是进程内 dict**：
#: 进程重启后账本还在，重试才真的不会重复投递。
NOTIFY_LEDGER_PATH: Final[Path] = Path("reports") / "notify_ledger.jsonl"


def _load_ledger() -> dict[str, dict[str, Any]]:
    """读幂等账本。文件不存在 = 空账本；坏行跳过并保留其余（账本自身可恢复）。"""
    if not NOTIFY_LEDGER_PATH.exists():
        return {}
    ledger: dict[str, dict[str, Any]] = {}
    for line in NOTIFY_LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            ledger[record["idempotency_key"]] = record
        except (json.JSONDecodeError, KeyError):
            continue  # 半行坏数据不该让整个账本失效
    return ledger


def _append_ledger(record: dict[str, Any]) -> None:
    """追加一条账本记录（JSON 行）。**先落记录、再执行**的"先"。"""
    NOTIFY_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTIFY_LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _mark_ledger_result(idempotency_key: str, result: dict[str, Any]) -> None:
    """把执行结果回填进账本（同 key 追加一条带 result 的记录；读时取最后一条）。"""
    ledger = _load_ledger()
    first = ledger.get(idempotency_key, {})
    _append_ledger({**first, "idempotency_key": idempotency_key, "result": result})


def _http_status_kind(code: int) -> str | None:
    """HTTP 状态码 → error_type。**归类逻辑只写这一处**。

    为什么抽成函数：这段判断原先埋在 `_post_json` 的 except 里，
    于是只有"从 _post_json 内部抛出"的错误才会被正确归类；任何从别处
    冒上来的 `HTTPError`（`urllib` 的 `HTTPError` 同时是 `URLError` 的子类）
    都会掉进"端点不可达"的分支，被重试 2 次再报 internal —— 403 明明该
    立刻停手。把映射抽出来，两层（`_post_json` 与 `notify_http`）共用同一份，
    归类就不会因为异常从哪个口子冒出来而改变。

    | code | 类 | retryable | 理由 |
    | --- | --- | --- | --- |
    | 401/403 | not_permitted | False | 授权问题，重试无意义 |
    | 429 | timeout | True | 限流，退避后可能好（复用 timeout 类，恢复动作就是退避）|
    | 其余 4xx | None（不改判）| — | 客户端请求本身有问题，交给上层按 internal 处理 |
    """
    if code in (401, 403):
        return "not_permitted"
    if code == 429:
        return "timeout"
    return None


def _post_json(endpoint: str, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
    """同步 HTTP POST（urllib，无新依赖）。供 to_thread 调用。

    **回环地址必须绕过系统代理**（W6 周日 httpx 同款问题的 urllib 翻版）：
    urlopen 默认读环境代理（HTTP_PROXY/HTTPS_PROXY），断网演练（死代理）
    或真实代理场景下，发往 127.0.0.1 mock 端的请求会被扔进代理再弹回来，
    表现为"端点不可达"。判据与 client.py 一致：目标是本机 → 不走代理。

    **HTTP 错误码翻译成异常**（W11 周一）：`urllib` 把 4xx/5xx 抛成
    `HTTPError`，而它同时是 `URLError` 的子类 —— 不先分流的话，
    403（权限）会掉进和"端点不通"同一个 except，被归成 internal 可重试，
    模型就会对着一个永远 403 的端点反复重试。

    这里只分两类，因为**error_type 说的是"发生了什么"，retryable 说的才是
    "能不能再试"**：

    - 401/403 → `not_permitted`（权限，不可重试）
    - 429/5xx → `internal`（服务端自己出错，**可重试** —— 由分类表给 retryable=True）
    - 其余 4xx → 原样抛出（客户端请求本身错了，属未预期，归 internal）
    """
    from urllib.parse import urlparse

    host = (urlparse(endpoint).hostname or "").lower()
    is_loopback = host in ("127.0.0.1", "localhost", "::1", "0.0.0.0")
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        if is_loopback:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            resp_ctx = opener.open(req, timeout=timeout)
        else:
            resp_ctx = urllib.request.urlopen(req, timeout=timeout)
        with resp_ctx as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        kind = _http_status_kind(exc.code)
        if kind == "not_permitted":
            raise ToolExecutionError(
                f"通知端点拒绝授权（HTTP {exc.code}）：{endpoint}",
                kind="not_permitted",
            ) from exc
        if kind == "timeout":
            raise ToolExecutionError(
                f"通知端点限流（HTTP {exc.code}）：{endpoint}；请退避后重试",
                kind="timeout",
            ) from exc
        raise
    return json.loads(body) if body.strip() else {"status": "no-body"}


def notify_http(
    channel: str,
    message: str,
    idempotency_key: str,
    *,
    endpoint: str = NOTIFY_ENDPOINT,
    retries: int = 1,
) -> dict[str, Any]:
    """投递通知：先记账 → 执行（失败重试 1 次）→ 回填账本。

    **幂等的三层防线**（W8 D4 的 Agent 侧复现）：
    1. 账本拦截：同 key 已有成功记录 → conflict + duplicate=true，不发包；
    2. 接收端去重：mock 端按 payload 里的 idempotency_key 自己也查一遍
       （网络层账本可能丢，接收端是最后防线）；
    3. 键纪律：重试复用同 key（写进描述，模型照做）。
    """
    if channel not in _ALLOWED_CHANNELS:
        raise ToolExecutionError(
            f"未知通知渠道 {channel!r}；可用渠道：{' / '.join(_ALLOWED_CHANNELS)}",
            kind="invalid_arg",
        )
    # W11 周一：空白正文是**没有内容可发**（empty_result），不是参数形状错。
    # Pydantic 的 min_length=1 只挡得住空串，挡不住 "   "。
    # 两者恢复动作不同：形状错该改字段，空内容该先去取结论再发。
    if not message.strip():
        raise ToolExecutionError(
            "通知正文为空（只有空白字符），没有内容可发送。"
            "先用检索/查询工具拿到结论，再据此写通知正文",
            kind="empty_result",
        )

    # ---- 第一层：账本拦截（先落记录的读取面）----
    # 拦截规则（周六首轮改进，改动 #1）：只有**已确认送达**的记录才拦。
    # in_flight（意图已记、结果未知——比如上一次端点不可达）不拦，
    # 复用同 key 重发是安全的：接收端按 key 去重（第二层防线），
    # 真重发过也不会重复。旧规则"key 存在即拦"实测造成的假象：
    # T8 里模型看到"已投递过"就向上报告"已送达"，而首次投递从未成功。
    ledger = _load_ledger()
    previous = ledger.get(idempotency_key)
    if previous is not None and previous.get("result", {}).get("delivered") is True:
        raise ToolExecutionError(
            f"idempotency_key {idempotency_key!r} 已成功投递过，本次为重复投递，已拦截",
            kind="conflict",
        )
    if previous is not None:
        # in_flight / 未确认：允许复用同 key 重发（接收端去重兜底），不换新键
        pass

    # ---- 先落记录（状态=进行中），再执行 ----
    _append_ledger(
        {
            "idempotency_key": idempotency_key,
            "channel": channel,
            "status": "in_flight",
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    payload = {"channel": channel, "message": message, "idempotency_key": idempotency_key}
    last_error = ""
    saw_permission = False
    saw_timeout = False
    for attempt in range(1, retries + 2):  # 1 次执行 + retries 次重试
        try:
            response = _post_json(endpoint, payload)
            result = {
                "delivered": bool(response.get("accepted", True)),
                "channel": channel,
                "attempt": attempt,
                "duplicate": bool(response.get("duplicate", False)),
                "endpoint": endpoint,
            }
            _mark_ledger_result(idempotency_key, result)
            return result
        except ToolExecutionError as exc:
            # 权限错**不重试**：403/401 重试多少次都是 403，白烧一次往返。
            # 直接上抛让信封归类 not_permitted。
            if exc.kind == "not_permitted":
                saw_permission = True
                last_error = exc.message
                break
            saw_timeout = True
            last_error = exc.message
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            # HTTPError 是 URLError 的子类，可能从别的口子冒上来（比如被替换/
            # 包装过的传输层）—— 这里用同一个映射再认一次，保证 403 无论
            # 从哪条路径出现都不会被当成"端点不可达"去重试。
            if isinstance(exc, urllib.error.HTTPError):
                kind = _http_status_kind(exc.code)
                if kind == "not_permitted":
                    saw_permission = True
                    last_error = f"HTTP {exc.code} 拒绝授权：{endpoint}"
                    break
                if kind == "timeout":
                    saw_timeout = True
                    last_error = f"HTTP {exc.code} 限流：{endpoint}"
                    continue
            if isinstance(exc, (TimeoutError, socket.timeout)):
                saw_timeout = True
            last_error = f"{type(exc).__name__}: {exc}"
            # 重试**复用同一个 key**（第三层防线）——接收端可能已收到
            # 但响应丢了，换新键会让它真的发两遍。

    if saw_permission:
        raise ToolExecutionError(
            f"通知端点 {endpoint} 拒绝授权：{last_error}。"
            "这是权限问题，重试无效；请换渠道（如 archive 归档）或如实告知用户通知发送失败",
            kind="not_permitted",
        )
    if saw_timeout:
        raise ToolExecutionError(
            f"通知端点 {endpoint} 重试 {retries + 1} 次仍超时/不可用：{last_error}。"
            "账本已记录本次投递意图，恢复后不会重复发送；可退避后重试",
            kind="timeout",
        )
    raise ToolExecutionError(
        f"通知端点 {endpoint} 重试 {retries + 1} 次仍失败：{last_error}。"
        "账本已记录本次投递意图，恢复后不会重复发送；请稍后重试或转人工。",
        kind="internal",
    )


# ============================================================ 统一信封（W11 周一升级）

#: 工具名常量：信封里 tool 字段的唯一来源。
REPORT_TOOL: Final[str] = "generate_report"
NOTIFY_TOOL: Final[str] = "notify"


def _field_details(exc: Any) -> list[dict[str, Any]]:
    """Pydantic ValidationError → 字段明细（hint 之外的结构化指路）。"""
    return [
        {
            "field": ".".join(str(p) for p in err["loc"]),
            "reason": err["msg"],
            "got": err.get("input"),
        }
        for err in exc.errors()
    ]


def _is_empty_report_payload(exc: Any) -> bool:
    """校验失败是不是**只有** "sections 为空" 这一个原因。

    为什么不能直接在 Pydantic 里把 `min_length` 改成 0：那会改掉**正常行为** ——
    schema 是在告诉模型"至少要有一节"，这条约束该留着当护栏。但模型偶尔仍会
    发一个空数组进来，此时它的错误动作是"先去取数据"，不是"把字段改个形状"。
    所以在**归类这一层**把这条单独的校验错误翻译成 empty_result：

        sections=[]                    → empty_result（缺内容）
        sections=[{...}] 但正文为空     → invalid_arg（形状不合格）
        section 里缺 heading           → invalid_arg

    只有"唯一的错误就是 sections 长度不足"时才翻译，其他组合一律 invalid_arg。
    """
    errors = exc.errors()
    if len(errors) != 1:
        return False
    err = errors[0]
    loc = [str(p) for p in err["loc"]]
    return loc == ["sections"] and err.get("type") in ("too_short", "missing")


async def execute_report_tool(raw_arguments: str) -> dict[str, Any]:
    """`generate_report` 落盘版的统一信封。永不抛异常。

    五类落点：invalid_json / invalid_arg（sections 里空字段）、
    **empty_result**（sections 为空 —— 参数形状合法，缺的是内容）、
    **not_permitted**（目录不可写 / 只读介质）、**timeout**（写盘挂住）、
    internal（其余）。写盘超时用 `REPORT_WRITE_TIMEOUT_S` 卡。
    """
    try:
        json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return tool_error(
            REPORT_TOOL,
            "invalid_json",
            f"参数不是合法 JSON：{exc.msg}（位置 {exc.pos}）",
            details={"raw": (raw_arguments or "")[:200]},
        )
    try:
        args = GenerateReportFileArgs.model_validate_json(raw_arguments or "{}")
    except ValidationError as exc:
        if _is_empty_report_payload(exc):
            return tool_error(
                REPORT_TOOL,
                "empty_result",
                "报告没有任何小节（sections 为空），没有内容可写",
            )
        return tool_error(
            REPORT_TOOL,
            "invalid_arg",
            "参数不符合工具签名",
            details={"fields": _field_details(exc)},
        )
    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(
                generate_report_file,
                args.task_id,
                args.title,
                [s.model_dump() for s in args.sections],
            ),
            timeout=REPORT_WRITE_TIMEOUT_S,
        )
    except TimeoutError:
        return tool_error(
            REPORT_TOOL,
            "timeout",
            f"报告写盘超过 {REPORT_WRITE_TIMEOUT_S}s 被中断（磁盘/网络盘响应过慢）",
        )
    except ToolExecutionError as exc:
        return tool_error(REPORT_TOOL, exc.kind, exc.message)
    except Exception as exc:  # noqa: BLE001
        return tool_error(
            REPORT_TOOL,
            classify_exception(exc),
            f"报告写入失败：{type(exc).__name__}: {exc}",
        )
    return tool_ok(REPORT_TOOL, data)


async def execute_notify_http_tool(raw_arguments: str) -> dict[str, Any]:
    """`notify` HTTP 版的统一信封。永不抛异常；conflict（幂等拦截）原样分类。

    五类落点：invalid_json / invalid_arg（channel 不在白名单）、
    **not_permitted**（端点 401/403 —— 重试无效，`notify_http` 内部已不重试）、
    **timeout**（端点超时/5xx/429，重试后仍失败）、internal（端点不可达）。
    """
    try:
        json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return tool_error(
            NOTIFY_TOOL,
            "invalid_json",
            f"参数不是合法 JSON：{exc.msg}（位置 {exc.pos}）",
            details={"raw": (raw_arguments or "")[:200]},
        )
    try:
        args = NotifyHTTPArgs.model_validate_json(raw_arguments or "{}")
    except ValidationError as exc:
        return tool_error(
            NOTIFY_TOOL,
            "invalid_arg",
            "参数不符合工具签名",
            details={"fields": _field_details(exc)},
        )
    try:
        data = await asyncio.to_thread(
            notify_http, args.channel, args.message, args.idempotency_key
        )
    except ToolExecutionError as exc:
        return tool_error(NOTIFY_TOOL, exc.kind, exc.message)
    except Exception as exc:  # noqa: BLE001
        return tool_error(
            NOTIFY_TOOL,
            classify_exception(exc),
            f"通知发送错误：{type(exc).__name__}: {exc}",
        )
    return tool_ok(NOTIFY_TOOL, data)


# ============================================================ mock 接收端（20:25–20:50）


MOCK_RECEIVER_SCRIPT: Final[
    str
] = '''"""通知 mock 接收端：打印收到的 payload，并按 idempotency_key 去重。

启动（演示脚本会自动拉起，也可手动）：
    python scripts/mock_notify_receiver.py [--port 8920]
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

SEEN: set[str] = set()
LOG: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        key = payload.get("idempotency_key", "")
        duplicate = key in SEEN
        SEEN.add(key)
        LOG.append({**payload, "duplicate_at_receiver": duplicate})
        print(f"[receiver] {'DUPLICATE ' if duplicate else ''}{json.dumps(payload, ensure_ascii=False)}", flush=True)
        body = json.dumps({"accepted": True, "duplicate": duplicate}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """GET /log：查看接收端收到的全部请求（幂等验证用）。"""
        body = json.dumps(LOG, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass  # 静默默认日志，payload 打印已经够了


if __name__ == "__main__":
    port = 8920
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print(f"[receiver] listening on 127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


def write_mock_receiver(path: Path | None = None) -> Path:
    """把 mock 接收端写成脚本文件。演示与联调共用。"""
    target = path or Path("scripts") / "mock_notify_receiver.py"
    target.write_text(MOCK_RECEIVER_SCRIPT, encoding="utf-8")
    return target


__all__ = [
    "GENERATE_REPORT_FILE_DESCRIPTION",
    "MOCK_RECEIVER_SCRIPT",
    "NOTIFY_ENDPOINT",
    "NOTIFY_HTTP_DESCRIPTION",
    "NOTIFY_LEDGER_PATH",
    "NOTIFY_TOOL",
    "REPORTS_DIR",
    "REPORT_TOOL",
    "REPORT_WRITE_TIMEOUT_S",
    "GenerateReportFileArgs",
    "NotifyHTTPArgs",
    "execute_notify_http_tool",
    "execute_report_tool",
    "generate_report_file",
    "notify_http",
    "render_report_markdown",
    "write_mock_receiver",
]
