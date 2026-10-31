"""知识库问答 UI（Streamlit）。

## 这个文件为什么这么长，以及为什么把逻辑都抽到 `app/rag/` 里

**Streamlit 的代码有个特点：它每次交互都重跑整个脚本。**
这意味着"把业务逻辑写在 `app.py` 里"会立刻变成灾难 ——
session_state 里塞满中间变量、缓存键写错、重跑时索引被重复建。
所以这里的规矩是：

| 层 | 在哪里 | 职责 |
| --- | --- | --- |
| 编排 | `app/rag/query.py` · `app/rag/indexing.py` | 检索、计时、调模型、解析引用 |
| 展示 | **本文件** | 排版、进度条、展开原文、四类错误提示 |
| 状态 | `st.session_state` | 对话历史、已索引文档、耗时记录 |

**本文件不 import 任何 rag 内部实现细节**（不碰 splitter / embedder / store），
只用 `QAEngine` + `Indexer` + `get_embedder` + `get_store` 这四个装配点。
这样"UI 改版"和"检索逻辑改进"可以互不影响。

## 必答题的答案怎么在界面上呈现

任务要求：从提问到首 token 的 1.5 秒里，检索占多少。

界面上**三个数字同时显示**，不合并：

    🔍 检索 11ms（编码 10 + 相似度 1）｜ ⏱ 首字 1243ms ｜ 总计 2870ms

并额外显示"检索占首字 X%"。**这个比例就是 W8 的决策依据** ——
如果它是 1%，那么"优化检索"对延迟毫无帮助，W8 该做的是流式之外的别的东西。
把三个数字合成一个"耗时 1.2s"，就再也回答不了这个问题。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.llm.client import get_llm_client  # noqa: E402
from app.rag.deploy_config import describe_runtime, resolve_embedder_name  # noqa: E402
from app.rag.embedding import get_embedder  # noqa: E402
from app.rag.indexing import Indexer, IndexReport  # noqa: E402
from app.rag.loaders import SUPPORTED_EXTENSIONS  # noqa: E402
from app.rag.query import QAEngine, StreamDone  # noqa: E402
from app.rag.stores import get_store  # noqa: E402
from app.rag.stores.chroma import DEFAULT_COLLECTION  # noqa: E402
from app.ui.evidence import find_evidence_spans, highlight  # noqa: E402

# 上传临时目录：**必须在项目内**（D 盘），不能落到系统 Temp ——
# 那是"重启后文件没了"以及"清理脚本把用户上传删了"的来源。
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

# 历史裁剪：保留最近这么多轮完整对话（任务要求 3 轮）。
MAX_HISTORY_TURNS = 3

PAGE_TITLE = "知识库问答 · industrial-ai-platform"


# ============================================================ 缓存：重跑不重建


@st.cache_resource(show_spinner=False)
def _load_embedder(name: str) -> Any:  # noqa: ANN401
    """embedder 是**进程级单例**（模型加载 16 秒），必须跨重跑复用。

    `st.cache_resource` 而不是 `st.cache_data`：前者缓存"对象"（模型、连接），
    后者缓存"可序列化的数据"。把 400 MB 的模型塞进 cache_data 会被 pickle 拷一份。

    `name` 作参数是为了**换 embedder 时自动换缓存** —— 不然用户在侧栏改了设置、
    却还在用旧模型，而界面上没有任何提示。
    """
    return get_embedder(name)


@st.cache_resource(show_spinner=False)
def _load_store(backend: str, collection: str) -> Any:  # noqa: ANN401
    """向量库同样跨重跑复用（Chroma 每次开库有 0.4 秒开销）。"""
    if backend == "chroma":
        return get_store("chroma", collection_name=collection)
    return get_store("memory")


def _current_min_score(embedder_name: str) -> str:
    """读取该 embedder 的拒答阈值（只读属性，不加载模型）。

    **返回字符串而不是 float**：hashing 的值是 `0.0`，直接显示成"0.0"
    会让人以为是"阈值设错了"，所以显示成 "0.0（已关闭）"。
    """
    try:
        value = get_embedder(embedder_name).recommended_min_score
    except Exception:  # noqa: BLE001 - 显示用，取不到就不显示原因
        return "未知"
    return f"{value}（已关闭）" if value == 0.0 else f"{value}"


def _engine(embedder_name: str, backend: str, collection: str, top_k: int) -> QAEngine:
    """每次都重建 `QAEngine`，但它持有的 embedder/store 来自缓存。

    **为什么 engine 不缓存**：`top_k` 是用户可调参数，缓存它会导致
    "改了 top_k 不生效"。engine 的构造是纯内存操作（微秒级），不值得缓存。
    """
    return QAEngine(
        embedder=_load_embedder(embedder_name),
        store=_load_store(backend, collection),
        top_k=top_k,
    )


# ============================================================ 状态初始化


def _init_state() -> None:
    """初始化 `session_state`。

    **必须用 `setdefault` 而不是直接赋值**：Streamlit 每次交互重跑整个脚本，
    直接赋值会把用户刚才的对话清空 —— 那是这个框架最经典的新手坑。
    """
    st.session_state.setdefault("messages", [])  # [{role, content, ...}]
    st.session_state.setdefault("indexed", {})  # {source: IndexReport}
    st.session_state.setdefault("timings", [])  # [dict] 每次问答的耗时
    st.session_state.setdefault("pending", None)  # 待处理的提问（见下）
    st.session_state.setdefault("last_error", None)


# ============================================================ 侧栏：配置与知识库


def _render_sidebar(embedder_name: str, backend: str) -> tuple[str, str, int]:
    """侧栏：配置 + 知识库文档列表 + 危险操作。

    返回 `(embedder_name, backend, top_k)` —— 用户可能在这一帧里改了它们。
    """
    with st.sidebar:
        st.subheader("⚙️ 配置")

        embedder_name = st.selectbox(
            "向量化模型",
            options=["hashing", "bge-small-zh"],
            index=0 if embedder_name == "hashing" else 1,
            help="hashing 离线但无语义；bge-small-zh 需下载 92MB 模型（首次约 16 秒）",
        )
        backend = st.selectbox(
            "向量库",
            options=["memory", "chroma"],
            index=0 if backend == "memory" else 1,
            help="memory 重启即丢；chroma 持久化到 data/chroma",
        )
        top_k = st.slider("检索条数 top_k", min_value=1, max_value=10, value=5)

        # ---- 降级必须可见 ----
        # 见到"运行在低内存托管环境"时，把这件事摆在侧栏最显眼的位置。
        # **悄悄降级等于骗用户**：他以为在做语义检索，实际只有字面匹配。
        runtime = describe_runtime()
        if runtime["notice"]:
            st.warning(runtime["notice"])

        # 当前 embedder 的拒答阈值也显示出来 —— 它是 0.0 就说明"没有拒答闸门"，
        # 用户该知道这件事，否则会疑惑"为什么问什么都给答案"。
        st.caption(
            f"当前向量化：**{embedder_name}**（拒答阈值 {_current_min_score(embedder_name)}）"
        )

        st.divider()
        st.subheader("📚 当前知识库")
        # **必须用 selectbox 刚返回的 `backend`。**
        #
        # Streamlit 在同一次渲染里，上面的 selectbox 已经返回了用户刚选的新值；
        # 而函数参数里那个是"上一次渲染时"的旧值。写成
        # `_load_store(某个旧变量, ...)` 会显示**切换前的库** ——
        # 用户切到 chroma 看到"空"，实际还在问 memory。
        # 这种 bug 不报错，只是数字对不上，最难查。
        store = _load_store(backend, DEFAULT_COLLECTION)
        total = store.count()
        indexed: dict[str, IndexReport] = st.session_state.indexed

        if total == 0:
            st.caption("（空）还没有索引任何文档。")
        else:
            st.metric("总块数", total)
            if indexed:
                for source, report in indexed.items():
                    status = "✅" if report.ok else "⚠️"
                    st.caption(f"{status} **{source}** · {report.pages} 页 · {report.chunks} 块")
            else:
                # 持久化库里已有数据、但本次会话没索引过（比如重启后打开）
                st.caption("（库中已有数据，但本次会话未执行索引）")

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🧹 清空对话", use_container_width=True):
                st.session_state.messages = []
                st.session_state.timings = []
                st.rerun()
        with col_b:
            if st.button("🗑 重建索引", use_container_width=True):
                # **重建 = 清空 + 重来**。不残留旧块，
                # 否则改 embedder 后新旧向量混在一个库里（维度不同会直接报错，
                # 维度相同时更糟：静默返回无法解释的结果）。
                store.clear()
                st.session_state.indexed = {}
                st.session_state.messages = []
                st.session_state.timings = []
                st.rerun()

        st.divider()
        st.caption("数据来源：`data/uploads/` · 索引库：`data/chroma/`")

    return embedder_name, backend, top_k


# ============================================================ 上传与索引


def _save_uploads(files: list[Any]) -> list[Path]:
    """把上传的文件落盘，返回路径列表。

    **必须先落盘再解析**：我们的 loader 接口是 `load_document(path)`，
    改成接收 bytes 要改三个 loader。而落盘还有个额外好处 ——
    出问题时文件还在，可以直接拿去复现。

    文件名用**原始名字**（不是 uuid）：`source` 字段会用文件名，
    它是用户看到引用时的标识。用 uuid 会让界面显示"3f2a1b.pdf"。
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for uploaded in files:
        target = UPLOAD_DIR / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        saved.append(target)
    return saved


def _render_upload(embedder_name: str, backend: str, collection: str) -> None:
    """上传 + 索引，带三阶段进度条。"""
    st.subheader("📤 上传文档")
    files = st.file_uploader(
        "支持 PDF / DOCX / Markdown，可多选",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if not files:
        return

    if st.button(f"⚡ 索引这 {len(files)} 个文件", type="primary"):
        engine_embedder = _load_embedder(embedder_name)
        store = _load_store(backend, collection)
        indexer = Indexer(embedder=engine_embedder, store=store)

        paths = _save_uploads(files)
        status = st.status("正在索引…", expanded=True)
        progress = st.progress(0.0, text="准备中")

        # 三个可见阶段对应任务要求：解析 N 页 / 切出 M chunk / 编码耗时
        stage_labels = {
            "load": "① 解析",
            "split": "② 切分",
            "embed": "③ 编码",
            "store": "④ 入库",
        }

        def on_stage(stage: str, _ratio: float, message: str) -> None:
            """把领域层的阶段事件翻译成进度条。

            **进度按文件数算，不按阶段算**：阶段只有 4 个，
            按阶段画进度条会在每个文件里从 0 跑到 100 再归零。
            真实进度需要知道"第几个文件 / 共几个"，那个信息在这里。
            """
            label = stage_labels.get(stage, stage)
            status.write(f"{label} {message}")
            progress.progress(min(done_files / len(paths), 1.0), text=f"{label} {message}")

        done_files = 0
        total_started = time.perf_counter()
        for path in paths:
            try:
                report = indexer.index_file(path, on_stage=on_stage)
            except Exception as exc:  # noqa: BLE001 - 单文件失败不该毁掉整批
                # **四类健壮性之二：解析失败要指出哪个文件**
                st.session_state.last_error = {
                    "kind": "parse",
                    "file": path.name,
                    "message": f"{type(exc).__name__}: {exc}",
                }
                status.write(f"❌ **{path.name}** 解析失败：{type(exc).__name__}: {exc}")
                done_files += 1
                continue

            st.session_state.indexed[path.name] = report
            done_files += 1
            progress.progress(done_files / len(paths), text=f"已完成 {done_files}/{len(paths)}")

            # 每个文件的四段耗时都写出来 —— 这就是"编码耗时"要求的落点
            status.write(
                f"✅ **{path.name}** · {report.pages} 页 → {report.chunks} 块 · "
                f"解析 {report.load_ms}ms / 切分 {report.split_ms}ms / "
                f"编码 {report.embed_ms}ms / 入库 {report.store_ms}ms"
            )
            for warning in report.warnings:
                # **扫描件/空页警告必须显示**：不显示的话用户以为全部索引成功了
                status.write(f"⚠️ {warning}")

        total_ms = int((time.perf_counter() - total_started) * 1000)
        progress.progress(1.0, text=f"完成，共 {total_ms}ms")
        status.update(label=f"索引完成（{total_ms}ms）", state="complete", expanded=True)
        st.rerun()


# ============================================================ 检索与生成


def _render_timing_badge(timing: dict[str, Any]) -> None:
    """三个耗时数字并列显示。**这是必答题在界面上的落点。**

    刻意**不合并成一个"耗时"**：合并之后就再也回答不了
    "检索占首 token 多少"，而 W8 的优化方向全看这个比例。
    """
    ttft = timing.get("ttft_ms")
    retrieval = timing.get("retrieval_ms", 0)
    total = timing.get("total_ms", 0)
    share = timing.get("retrieval_share")

    parts = [
        f"🔍 检索 **{retrieval}ms**（编码 {timing.get('embed_ms', 0)} + "
        f"相似度 {timing.get('search_ms', 0)}）"
    ]
    if ttft is not None:
        parts.append(f"⏱ 首字 **{ttft}ms**")
        if share is not None:
            parts.append(f"检索占首字 **{share * 100:.1f}%**")
    parts.append(f"⏳ 总计 **{total}ms**")
    st.caption("　｜　".join(parts))


def _render_chunk_with_evidence(chunk: Any, answer_text: str) -> None:
    """展开显示 chunk 原文，并把"答案里那句话"高亮出来。

    ## 为什么需要高亮，而不是"整段展开就够了"

    任务说"整段展开 + 页码标注即可"。但一段 500 字的条款里找"按日万分之五"
    这五个字，人眼要扫三秒 —— 而这三秒恰恰是"引用可不可信"的关键体验：
    **找不到，用户就不信**。

    所以用 `app/ui/evidence.py` 的 LCS 匹配把依据句标出来。

    ## 两条降级路径都要说清

    | 情况 | 表现 |
    | --- | --- |
    | 找到依据 | 粗体高亮命中的那段 |
    | **找不到**（模型改写幅度大） | 高亮为空，**显式提示"未能在原文中定位依据"** |

    第二种不能静默 —— 用户看到"没有任何高亮"会以为是 UI 坏了，
    或者更糟：以为整段都是依据。**必须把"没定位到"这件事说出来。**
    """
    spans = find_evidence_spans(answer_text, chunk.text)
    body = chunk.text[:1200] + ("…" if len(chunk.text) > 1200 else "")

    if spans:
        # 只在**截断范围内**的 span 才高亮：超过 1200 字的区间如果画出来，
        # 标记会落在省略号之后，看起来像标错了位置。
        visible = [s for s in spans if s.start < 1200]
        st.markdown(highlight(body, visible))
    else:
        # `st.text` 而不是 `st.markdown`：**没有高亮时原文里可能有 `#` `|` `*`**，
        # 用 markdown 渲染会把它们吃掉，用户拿去对照 PDF 就搜不到了。
        st.text(body)
        st.caption("（未能在原文中定位到支撑句 —— 可能是模型做了较大改写，请人工核对）")


def _render_sources(contexts: list[Any], answer: Any) -> None:
    """「参考来源」：编号 + 文件名 + 页码，可展开看原文并高亮依据句。

    **只列被真正引用的块**（`Answer.citations` 里的），不是全部检索结果。
    列一堆没被用到的块会让人以为"这些也是依据"。
    """
    if not contexts:
        if answer.invalid_refs:
            st.warning(
                f"⚠️ 回答里出现了不存在的编号 {answer.invalid_refs}，"
                "这些引用无法追溯到原文 —— 说明模型在编，请核对。"
            )
        return

    with st.expander(f"📖 参考来源（{len(contexts)} 条）", expanded=False):
        for citation, chunk in zip(answer.citations, contexts, strict=False):
            st.markdown(
                f"**[{citation.index}]** `{citation.source}` · **P{citation.page}**"
                if citation.page is not None
                else f"**[{citation.index}]** `{citation.source}`"
            )
            _render_chunk_with_evidence(chunk, answer.text)
            st.divider()

    if answer.invalid_refs:
        st.warning(
            f"⚠️ 另有编号 {answer.invalid_refs} 不在检索结果里，已忽略 —— 本次回答可能有编造成分。"
        )


async def _stream_answer(
    engine: QAEngine, question: str, model: str, extra: dict[str, Any]
) -> None:
    """流式问答：边出字边渲染，并实时显示检索耗时。

    **关键点：`retrieved` 事件在模型出第一个字之前就到了。**
    所以用户不用盯着空白等 TTFT —— 那段时间里界面已经在显示
    "检索到 5 块，最高分 0.63"。
    """
    client = get_llm_client(get_settings())

    status_slot = st.empty()
    text_slot = st.empty()
    parts: list[str] = []
    done: StreamDone | None = None

    async for event, payload in engine.answer_stream(
        question,
        client=client,
        model=model,
        extra_request=extra,
    ):
        if event == "retrieved":
            chunks, scores, timing = payload  # type: ignore[misc]
            top = f"{max(scores):.3f}" if scores else "—"
            status_slot.caption(
                f"🔍 检索 **{timing.retrieval_ms}ms**（编码 {timing.embed_ms} + "
                f"相似度 {timing.search_ms}）· 命中 {len(chunks)} 块 · 最高分 {top}"
            )
        elif event == "delta":
            parts.append(payload)  # type: ignore[arg-type]
            text_slot.markdown("".join(parts) + "▌")
        elif event == "done":
            done = payload  # type: ignore[assignment]

    final_text = done.answer.text if done else "".join(parts)
    text_slot.markdown(final_text)

    if done is None:
        # 生成器没给出 done 就结束了 —— 说明上游有 bug，必须显示而不是静默空白
        st.error("生成过程异常结束（没有收到完成事件）。请重试。")
        return

    _render_timing_badge(done.timing.as_dict())
    _render_sources(done.contexts if not done.answer.refused else [], done.answer)

    # 写进历史：**存的是纯文本 + 引用 + 耗时**，不是 Streamlit 对象。
    # 存对象会让 session_state 里堆积不可序列化的东西，重跑时行为诡异。
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_text,
            "citations": [
                {"index": c.index, "source": c.source, "page": c.page, "label": c.label}
                for c in done.answer.citations
            ],
            "contexts": [
                {"index": c.index, "text": c.text[:1200], "source": c.source, "page": c.page}
                for c in (done.contexts if not done.answer.refused else [])
            ],
            "invalid_refs": done.answer.invalid_refs,
            "refused": done.answer.refused,
            "timing": done.timing.as_dict(),
        }
    )
    st.session_state.timings.append(done.timing.as_dict())


def _render_history(turns: int) -> None:
    """渲染历史对话。`turns` 之外的旧对话**不渲染但仍留在 state 里**。

    **为什么"不渲染"而"不删除"**：任务要求"历史裁剪（最近 3 轮 + 摘要）"。
    本周不做指代消解（W8），所以裁剪只影响**展示**；
    真正传给模型的上下文由 `_history_for_model` 决定。
    把两者分开，是为了将来做"滚动摘要"时不用改渲染逻辑。
    """
    messages = st.session_state.messages
    visible = messages[-(turns * 2) :] if turns > 0 else messages

    hidden = len(messages) - len(visible)
    if hidden > 0:
        st.caption(f"（已折叠 {hidden} 条更早的消息，保留最近 {turns} 轮）")

    for message in visible:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("timing"):
                    _render_timing_badge(message["timing"])
                contexts = message.get("contexts") or []
                if contexts:
                    with st.expander(f"📖 参考来源（{len(contexts)} 条）", expanded=False):
                        for ctx in contexts:
                            st.markdown(
                                f"**[{ctx['index']}]** `{ctx['source']}` · **P{ctx['page']}**"
                                if ctx.get("page") is not None
                                else f"**[{ctx['index']}]** `{ctx['source']}`"
                            )
                            # 历史里也要高亮：用户翻回去看旧回答时，
                            # 同样需要"哪句是依据"。用纯 dict 复现同样的逻辑，
                            # 不依赖 Chunk 对象（历史里存的是可序列化的 dict）。
                            spans = find_evidence_spans(message["content"], ctx["text"])
                            if spans:
                                st.markdown(highlight(ctx["text"], spans))
                            else:
                                st.text(ctx["text"])
                                st.caption("（未能在原文中定位到支撑句，请人工核对）")
                            st.divider()
                if message.get("invalid_refs"):
                    st.warning(f"⚠️ 编号 {message['invalid_refs']} 不在检索结果里，已忽略。")


def _history_for_model(turns: int) -> list[dict[str, str]]:
    """把历史压成模型能用的消息列表。

    **只取最近 `turns` 轮**，且**只取纯文本** —— 引用与耗时是给人看的，
    塞进 prompt 只会浪费 token 并干扰模型。

    **本周不做指代消解**（W8 才做）：所以历史里出现的"它/那个条款"
    模型仍然无法关联到具体文档。历史在这里的作用只是"让对话连贯"，
    不是"让模型理解指代"。
    """
    messages = st.session_state.messages
    if turns <= 0:
        return []
    recent = messages[-(turns * 2) :]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


# ============================================================ 主流程


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="📚", layout="wide")
    _init_state()

    st.title("📚 知识库问答")
    st.caption("上传文档 → 索引 → 提问。答案带 [n] 引用，可展开查看原文。")

    settings = get_settings()
    # 默认 embedder 由 `deploy_config` 决定，不是写死 "hashing"：
    # - 本地（无平台标记）→ hashing（保守，不触发 92 MB 下载）
    # - Streamlit Cloud → hashing，且侧栏显式提示"已降级"
    # - 设了 `RAG_EMBEDDER` → 用它（用户显式决定）
    #
    # **写死 "hashing" 也能跑**，但那会让"部署侧该降级"只存在于文档里而不在代码里 ——
    # 文档会被忘掉，代码不会。
    embedder_name = st.session_state.get("embedder_name", resolve_embedder_name())
    backend = st.session_state.get("backend", "memory")
    # 上一次生效的 embedder —— 用来检测"用户换了模型但库没重建"
    previous_embedder = st.session_state.get("active_embedder")

    embedder_name, backend, top_k = _render_sidebar(embedder_name, backend)
    st.session_state.embedder_name = embedder_name
    st.session_state.backend = backend
    st.session_state.active_embedder = embedder_name

    collection = DEFAULT_COLLECTION
    store = _load_store(backend, collection)

    # ---- 换 embedder 但库没重建 → 必须警示 ----
    #
    # **为什么这是个真问题**：向量库里的向量是旧 embedder 产的。
    # 换模型后：
    #   - 维度不同（256 vs 512）→ Chroma 会直接报错（还算好）
    #   - 维度相同 → **静默返回无法解释的结果**，且不报任何错
    #
    # 所以这里不自动清库（那会毁掉用户数据），而是**明确警示 + 给一个按钮**。
    if previous_embedder is not None and previous_embedder != embedder_name and store.count() > 0:
        st.warning(
            f"⚠️ 你从 **{previous_embedder}** 切到了 **{embedder_name}**，"
            f"但向量库里还有 {store.count()} 条旧模型的向量。\n\n"
            "继续提问会得到无法解释的结果（同维度时静默出错，不同维度时直接报错）。"
        )
        if st.button("🗑 清空向量库并重建索引", type="primary"):
            store.clear()
            st.session_state.indexed = {}
            st.session_state.messages = []
            st.session_state.timings = []
            st.rerun()

    _render_upload(embedder_name, backend, collection)

    # ---- 四类健壮性之一：上一次的解析错误要还看得见 ----
    if st.session_state.last_error:
        err = st.session_state.last_error
        st.error(f"❌ **{err['file']}** 解析失败：{err['message']}")
        if st.button("知道了，清除这条提示"):
            st.session_state.last_error = None
            st.rerun()

    st.divider()
    st.subheader("💬 对话")
    _render_history(MAX_HISTORY_TURNS)

    # `store` 在上面的"换 embedder 检测"里已经取过（`st.cache_resource` 是同一对象），
    # 这里直接复用，不再重复调用 —— 重复调用虽然无副作用，但会让"到底用哪个 store"
    # 出现两个可能的来源，是将来改错的入口。
    # ---- 四类健壮性之三：知识库为空时，不给输入框（而不是让用户问了再报错）----
    if store.count() == 0:
        with st.chat_message("assistant"):
            st.info(
                "知识库还是空的。请先在上面上传文档并点「索引」。\n\n"
                "如果已经上传过但这里仍显示为空，检查侧栏的「向量库」是否与索引时一致 —— "
                "`memory` 每次重启都会清空。"
            )
        return

    question = st.chat_input("问点什么…（例如：违约金怎么算）")
    if not question:
        return

    # 用户消息立刻上屏
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    engine = _engine(embedder_name, backend, collection, top_k)
    extra: dict[str, Any] = {}
    # 本机网关默认开推理模式，思维链会吃光 max_tokens 导致 content=None/q=null
    # （实测见 learn/llm/01_params.md §3.1）。glm 系需要显式关掉。
    if "glm" in settings.llm.model.lower():
        extra["reasoning_effort"] = "none"

    with st.chat_message("assistant"):
        try:
            import asyncio

            asyncio.run(_stream_answer(engine, question, settings.llm.model, extra))
        except Exception as exc:  # noqa: BLE001 - 四类健壮性之四：模型报错可重试
            # **不吞掉错误，也不让整个页面崩**：给出原因 + 一个重试按钮。
            # 直接 `raise` 会让 Streamlit 显示一大片红色栈，用户不知道能做什么。
            st.error(
                f"生成失败：{type(exc).__name__}: {exc}\n\n"
                "可能是网关凭据/网络问题。可以点下面的按钮重试。"
            )
            if st.button("🔄 重试这一问"):
                st.session_state.pending = question
                st.rerun()

    st.rerun()


if __name__ == "__main__":
    main()
