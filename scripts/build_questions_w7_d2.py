"""W7 第二批出题：20 条多跳 + 20 条拒答（2026-12-02 周三）。

## 多跳（m-01..m-20）

两个锚点必须落在**不同 chunk**（validate_golden.py 强制）；
notes 写清"缺任一块答不出"的理由。只出 2 跳 —— 系统取 top5，
3 跳以上测了只会得 0，不是系统差是题不合法。

## 拒答（r-01..r-20）

同领域但语料未覆盖。每条三道门缺一不可：
1. 全文 grep：关键词出现 0 次（证据写进 notes）
2. 向量 top20：bge 实测 top1 分数与来源（脚本自动跑、自动记录）
3. 同领域约束：问的是 RAG/LLM 工程，不是"今天天气"

拒答题最隐蔽的失败：文档里用不同表述写了同一事实。
grep 只能证明关键词没有，向量 top20 才能证明"语义上也没有近邻"——
两道门一起过才许入库。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_retrieval_eval as rre  # noqa: E402
from retrieval_harness import load_raw_corpus  # noqa: E402

OUT = PROJECT_ROOT / "eval" / "questions-120.jsonl"

# ---------------------------------------------------------------- 多跳 20 条
# 每条：两个 key（各在一处），notes 必须写清"缺任一块答不出"
MULTI_HOP: list[dict[str, str]] = [
    {
        "id": "m-01",
        "doc": "deploy-notes.md",
        "keys": ["内存字段含逗号", "`Popen.pid` 不是被测进程"],
        "question": "内存测量那两次差点量错的坑，分别错在哪？",
        "gold_answer": "坑1：tasklist CSV 内存字段含逗号；坑2：p.pid 是 launcher 不是被测进程",
        "notes": "两坑各在一节，只知其一得不出完整答案",
    },
    {
        "id": "m-02",
        "doc": "deploy-notes.md",
        "keys": ["334 KB", "必须放在入口目录"],
        "question": "部署依赖文件的两个坑合起来看，最终方案落在哪里？",
        "gold_answer": "hash 行太长(334KB)易装失败；文件必须放入口目录否则被 uv.lock 抢掉",
        "notes": "坑5 与坑5.1 分处两节，合并才能答",
    },
    {
        "id": "m-03",
        "doc": "deploy-notes.md",
        "keys": ["缓存减少的是", "悄悄降级等于骗用户"],
        "question": "LRU 缓存为什么救不了免费层，降级为什么必须显示？",
        "gold_answer": "缓存省计算不省常驻内存（528MB 还在）；悄悄降级等于骗用户",
        "notes": "方案表与降级展示分处两节",
    },
    {
        "id": "m-04",
        "doc": "streamlit-ui.md",
        "keys": ["检索约 9~11 ms", "1128 ms"],
        "question": "首字延迟里检索的占比是怎么算出来的？",
        "gold_answer": "9ms 检索 / 1128ms 真实 TTFT ≈ 0.80%；分母必须用历史实测",
        "notes": "分子在一节、分母口径在另一节",
    },
    {
        "id": "m-05",
        "doc": "streamlit-ui.md",
        "keys": ["解析 714 ms 是 PDF 逐页抽取", "编码 300 ms"],
        "question": "50 页 PDF 全流程里，解析和索引各自花了多久？",
        "gold_answer": "解析 714ms；索引 1015ms（其中编码 300ms）",
        "notes": "五个数字表与冷启动说明分处两节",
    },
    {
        "id": "m-06",
        "doc": "streamlit-ui.md",
        "keys": ["拒答话术 + **不调模型**", "有编号但找不到出处就是 bug"],
        "question": "空检索和引用编号越界这两类异常，界面分别怎么处理？",
        "gold_answer": "空检索：拒答话术且不调模型；编号越界：显式警告，静默丢弃视为 bug",
        "notes": "场景表与引用规则分处两节，缺一答不全",
    },
    {
        "id": "m-07",
        "doc": "streamlit-ui.md",
        "keys": ["只渲染最近", "只取纯文本"],
        "question": "多轮历史的展示与喂模型分别怎么处理？",
        "gold_answer": "都取最近 3 轮；喂模型只取纯文本（引用/耗时给人看）",
        "notes": "展示层与传模型层分处两节",
    },
    {
        "id": "m-08",
        "doc": "why-i-use-langchain.md",
        "keys": ["**只支持 `tool_choice=auto` 和 `none`**", "对扁平 schema 能强制执行"],
        "question": "LangChain 结构化输出在本网关失败的完整归因是什么？",
        "gold_answer": "function_calling 被网关能力堵死；嵌套 schema 不强制而扁平可以",
        "notes": "三层归因表与扁平对照分处两节，缺一归因不完整",
    },
    {
        "id": "m-09",
        "doc": "why-i-use-langchain.md",
        "keys": ["95 行", "它赢的地方"],
        "question": "代码量对比与自研承认的劣势各是什么？",
        "gold_answer": "LangChain 约 95 行更短；做 agent+工具+流式时 Runnable 层有优势",
        "notes": "对比表与 3.3 节分处两处",
    },
    {
        "id": "m-10",
        "doc": "prompt-ab.md",
        "keys": [
            '**真正的元凶是"两步走"结构。**',
            "| qa-02 | 1.00 | **0.33** | ⬇ 掉了 3 个要点里的 2 个 |",
        ],
        "question": "问答组 v2 掉分的机制和最典型的那条各是什么？",
        "gold_answer": "两步走结构让答案写短；qa-02 从 1.00 掉到 0.33",
        "notes": "机制分析与逐条表分处两节",
    },
    {
        "id": "m-11",
        "doc": "prompt-ab.md",
        "keys": ['**模型的失败模式是把"意图"当成"现象"来判。**', "往往先描述现象再提问题"],
        "question": "分类混淆的机制和用户输入的特征是什么？",
        "gold_answer": "模型把意图当现象判；用户先描述现象再提问题",
        "notes": "失败模式与输入特征分处两节",
    },
    {
        "id": "m-12",
        "doc": "prompt-ab.md",
        "keys": ["部分实现", "0 引用"],
        "question": "v2 加引用要求后，预期收益与实际效果差在哪？",
        "gold_answer": "预期引用更完整，实际 qa-08 类上下文无关题仍 0 引用",
        "notes": "改动表与逐条归因分处两节",
    },
    {
        "id": "m-13",
        "doc": "system-prompt.md",
        "keys": ["**① Token 翻倍**：204 → 423 token / 次（+219）", "**324 汉字**"],
        "question": "严格版 prompt 的 token 代价折算成字数是多少？",
        "gold_answer": "204→423 token（+219）；按 0.676 token/字 ≈ 多 324 字",
        "notes": "代价表与换算分处两节",
    },
    {
        "id": "m-14",
        "doc": "system-prompt.md",
        "keys": ["把「拒绝」拆成四档（共 5 条相关输入", "显式拒绝率 0.00 → 0.60"],
        "question": "判定器返工前后，拒绝这个动作的统计口径和结论差在哪？",
        "gold_answer": "二元口径三指标全打平；拆四档后显式拒绝 0→0.60 差异显现",
        "notes": "拆档设计与差异结果分处两节，缺一不可",
    },
    {
        "id": "m-15",
        "doc": "system-prompt.md",
        "keys": ["#3、#9、#19", "拒答条款套用"],
        "question": "正常工单里出现无法判断的三条，共同原因是什么？",
        "gold_answer": "#3/#9/#19 正文没写产品名，拒答条款被套用到可推断字段",
        "notes": "条目列表与归因分处两节",
    },
    {
        "id": "m-16",
        "doc": "chunking-boundaries.md",
        "keys": [
            "**症状**：`docs/w4-checklist.md` 切出 **44 块 = 文档段数**",
            '**修法**：判据改成"**标题行**"',
        ],
        "question": "w4 清单切分不合并的根因与修法各是什么？",
        "gold_answer": "把有 level 当节开头导致 44 块不合并；改成按标题行判定",
        "notes": "症状与修法分处两节",
    },
    {
        "id": "m-17",
        "doc": "chunking-boundaries.md",
        "keys": ["39 块带路径", "照抄 `recursive` 的做法"],
        "question": "代码块丢 section_path 的范围和根因是什么？",
        "gold_answer": "40 块里 39 块有路径、唯一缺的是代码块；根因是照抄 recursive 做法",
        "notes": "症状与根因分处两节",
    },
    {
        "id": "m-18",
        "doc": "prompt-templates.md",
        "keys": ["缓存键和渲染参数分开", "tuple 不是对象"],
        "question": "_freeze 崩 ctx.id 的根因和修法各是什么？",
        "gold_answer": "tuple 无属性访问；键用冻结元组、渲染前 thaw 回 dict",
        "notes": "根因与修法分处两节",
    },
    {
        "id": "m-19",
        "doc": "structured-output.md",
        "keys": ["自创了 `category / priority", "看不到枚举"],
        "question": "探针实验里模型自创字段与枚举缺失的关系是什么？",
        "gold_answer": "模型看不到枚举时自创了 category/priority/summary 字段",
        "notes": "探针结果与枚举实证分处两节",
    },
    {
        "id": "m-20",
        "doc": "ab-scaffold.md",
        "keys": ["**自己编的输入会不知不觉地配合 prompt。**", "先于 prompt 存在的"],
        "question": "输入池为什么要从开发语料抽取，而不是按需求自己写？",
        "gold_answer": "自己编会配合 prompt 变成自证；语料先于 prompt 存在，不知道我要什么",
        "notes": "危害与原理分处两节，缺一理由不完整",
    },
]

# ---------------------------------------------------------------- 拒答 20 条
# 全部为 RAG/LLM 工程同领域主题；关键词将在脚本里自动 grep 确认 0 次
REFUSAL: list[dict[str, str]] = [
    {
        "id": "r-01",
        "question": "pgvector 的 HNSW 索引建图参数 efConstruction 一般怎么取？",
        "kw": ["pgvector", "efConstruction"],
        "notes_hint": "向量库选型同领域；语料只讲 Chroma（Milvus 被设计文档覆盖过，已由 grep 拦下）",
    },
    {
        "id": "r-02",
        "question": "Qdrant 的 payload 过滤条件语法怎么写？",
        "kw": ["Qdrant", "payload 过滤"],
        "notes_hint": "向量库选型同领域",
    },
    {
        "id": "r-03",
        "question": "LlamaIndex 的句子窗口检索怎么配置？",
        "kw": ["LlamaIndex", "句子窗口"],
        "notes_hint": "RAG 框架同领域，语料刻意不用框架",
    },
    {
        "id": "r-04",
        "question": "vLLM 的 PagedAttention 解决的是什么问题？",
        "kw": ["vLLM", "PagedAttention"],
        "notes_hint": "推理服务同领域",
    },
    {
        "id": "r-05",
        "question": "FlashAttention 的分块计算为什么能省显存？",
        "kw": ["FlashAttention"],
        "notes_hint": "注意力实现同领域",
    },
    {
        "id": "r-06",
        "question": "PostgreSQL 里 pg_trgm 和全文检索怎么选？",
        "kw": ["pg_trgm", "PostgreSQL"],
        "notes_hint": "存储检索同领域",
    },
    {
        "id": "r-07",
        "question": "Elasticsearch 的 BM25 里 k1 和 b1 分别调什么？",
        "kw": ["Elasticsearch", "k1"],
        "notes_hint": "语料讲 BM25 原理但没讲 ES 实现",
    },
    {
        "id": "r-08",
        "question": "Weaviate 的向量化模块怎么配置？",
        "kw": ["Weaviate"],
        "notes_hint": "向量库选型同领域",
    },
    {
        "id": "r-09",
        "question": "推理引擎的 continuous batching 与静态 batching 相比省在哪？",
        "kw": ["continuous batching", "静态 batching"],
        "notes_hint": "推理服务同领域；语料只提过一次上游动态 batching 现象，未展开机制",
    },
    {
        "id": "r-10",
        "question": "GGUF 格式的模型文件里量化等级存在哪个字段？",
        "kw": ["GGUF", "量化等级"],
        "notes_hint": "推理格式同领域；语料只讲过放弃 ONNX 路线，GGUF 零覆盖",
    },
    {
        "id": "r-11",
        "question": "Haystack 的 pipeline 组件怎么串？",
        "kw": ["Haystack"],
        "notes_hint": "RAG 框架同领域",
    },
    {
        "id": "r-12",
        "question": "RAGAS 评测框架的 faithfulness 指标怎么算？",
        "kw": ["RAGAS"],
        "notes_hint": "评测框架同领域，语料自研指标未用它",
    },
    {
        "id": "r-13",
        "question": "MLflow 的实验追踪怎么接进训练脚本？",
        "kw": ["MLflow"],
        "notes_hint": "MLOps 同领域",
    },
    {
        "id": "r-14",
        "question": "Airflow 的 DAG 调度怎么定义依赖？",
        "kw": ["Airflow", "DAG 调度"],
        "notes_hint": "编排同领域",
    },
    {
        "id": "r-15",
        "question": "gRPC 的服务端流式接口在 proto 里怎么定义？",
        "kw": ["gRPC", "proto"],
        "notes_hint": "流式同领域，语料讲 SSE 不讲 gRPC",
    },
    {
        "id": "r-16",
        "question": "Redis 的 LFU 淘汰策略参数怎么调？",
        "kw": ["LFU", "Redis"],
        "notes_hint": "缓存同领域",
    },
    {
        "id": "r-17",
        "question": "Kafka 的 ISR 副本同步机制是什么？",
        "kw": ["Kafka", "ISR"],
        "notes_hint": "消息队列同领域",
    },
    {
        "id": "r-18",
        "question": "DVC 的数据集版本管理怎么和 git 配合？",
        "kw": ["DVC"],
        "notes_hint": "数据版本同领域",
    },
    {
        "id": "r-19",
        "question": "TensorRT 的 INT8 量化精度损失怎么评估？",
        "kw": ["TensorRT", "INT8"],
        "notes_hint": "推理优化同领域",
    },
    {
        "id": "r-20",
        "question": "FAISS 的 IVF-PQ 索引两阶段量化原理是什么？",
        "kw": ["FAISS", "IVF-PQ"],
        "notes_hint": "向量索引同领域",
    },
]

assert len({it["id"] for it in MULTI_HOP}) == len(MULTI_HOP) == 20
assert len({it["id"] for it in REFUSAL}) == len(REFUSAL) == 20


def extract_quote(lines: list[str], key: str, doc: str, item_id: str) -> str:
    hits = [ln for ln in lines if key in ln]
    if not hits:
        raise SystemExit(f"[{item_id}] key 在 {doc} 找不到：{key!r}")
    if len(hits) > 1:
        raise SystemExit(f"[{item_id}] key 在 {doc} 命中 {len(hits)} 行，需更精确：{key!r}")
    line = hits[0].strip()
    if len(line) <= 80:
        quote = line
    else:
        start = max(0, line.find(key) - 15)
        quote = line[start : start + 80].strip()
    if not 15 <= len(quote) <= 80:
        raise SystemExit(f"[{item_id}] quote 长度 {len(quote)} 越界：{quote!r}")
    return quote


def build_multi_hop(lines_by_doc: dict[str, list[str]]) -> list[dict]:
    records = []
    for it in MULTI_HOP:
        doc = it["doc"]
        keys = it["keys"] if isinstance(it["keys"], list) else [it["keys"], it.get("keys2")]
        keys = [k for k in keys if k]
        assert len(keys) == 2, f"[{it['id']}] 需要 2 个 key"
        quotes = [extract_quote(lines_by_doc[doc], k, doc, it["id"]) for k in keys]
        if quotes[0] == quotes[1]:
            raise SystemExit(f"[{it['id']}] 两锚点取到同一行")
        records.append(
            {
                "id": it["id"],
                "type": "multi_hop",
                "question": it["question"],
                "anchors": [
                    {"source": doc, "page": None, "head12": q[:12], "quote": q} for q in quotes
                ],
                "gold_chunk_ids": [],
                "gold_answer": it["gold_answer"],
                "notes": it["notes"],
            }
        )
    return records


def grep_confirm(full_by_doc: dict[str, str], kw: list[str]) -> tuple[bool, str]:
    """全文 grep：所有关键词出现 0 次才算过。返回（通过, 证据文本）。"""
    full = "\n".join(full_by_doc.values())
    hits = {k: full.count(k) for k in kw}
    if any(n > 0 for n in hits.values()):
        return False, "、".join(f"{k}={n}次" for k, n in hits.items() if n > 0)
    return True, "、".join(kw) + " 全文 0 次"


def vector_confirm(items: list[dict], threshold_hint: float = 0.575) -> list[str]:
    """向量 top20 实测：bge 检索，记录 top1 分数与来源。

    这一步是 grep 的补充：文档里用不同表述写了同一事实时，
    grep 是 0 但向量分会很高。top1 >= 阈值提示的条目会被点名复查。
    """
    from app.rag.chunking import ChunkingConfig, build_strategy
    from app.rag.embedding import get_embedder
    from app.rag.stores.memory import InMemoryVectorStore

    chunks = build_strategy(
        "recursive", cfg=ChunkingConfig(chunk_chars=500, overlap_chars=50)
    ).chunk(load_raw_corpus(), ChunkingConfig(chunk_chars=500, overlap_chars=50))
    embedder = get_embedder("bge-small-zh")
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    report: list[str] = []
    for it in items:
        hits = store.search(embedder.embed_query([it["question"]])[0], top_k=20)
        top = hits[0]
        scores = [h.score for h in hits]
        above = sum(1 for s in scores if s >= threshold_hint)
        it["vector_top1"] = round(top.score, 4)
        it["vector_top1_source"] = top.chunk.source
        flag = " ⚠高分近邻，需人工复查" if above else ""
        report.append(
            f"[{it['id']}] top1={top.score:.4f}({top.chunk.source}) "
            f"top20 中 ≥{threshold_hint} 的有 {above} 条{flag}"
        )
    return report


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    docs = load_raw_corpus()
    lines_by_doc: dict[str, list[str]] = {}
    full_by_doc: dict[str, str] = {}
    for d in docs:
        lines_by_doc.setdefault(d.source, []).extend(d.text.split("\n"))
        full_by_doc[d.source] = full_by_doc.get(d.source, "") + d.text + "\n"

    # ---- 多跳 ----
    mh = build_multi_hop(lines_by_doc)
    print(f"多跳取证完成：{len(mh)} 条")

    # ---- 拒答：门 1 全文 grep ----
    refusal_records: list[dict] = []
    failed_grep: list[str] = []
    for it in REFUSAL:
        ok, evidence = grep_confirm(full_by_doc, it["kw"])
        if not ok:
            failed_grep.append(f"[{it['id']}] 关键词在语料出现：{evidence}")
            continue
        refusal_records.append(
            {
                "id": it["id"],
                "type": "refusal",
                "question": it["question"],
                "anchors": [],
                "gold_chunk_ids": [],
                "gold_answer": "资料未覆盖，应拒答并说明无依据",
                "notes": f"拒答：{it['notes_hint']}；grep 证据：{evidence}",
            }
        )
    if failed_grep:
        print("✗ grep 未过的拒答候选（不入库）：")
        for f in failed_grep:
            print(f"  {f}")
        raise SystemExit(1)
    print(f"拒答 grep 门通过：{len(refusal_records)} 条")

    # ---- 拒答：门 2 向量 top20（联网/加载模型约 20 秒）----
    print("向量确认中（加载 bge + 建索引，约 30 秒）…")
    vector_items = [{"id": r["id"], "question": r["question"]} for r in refusal_records]
    for line in vector_confirm(vector_items):
        print(f"  {line}")
    vmap = {v["id"]: v for v in vector_items}
    for r in refusal_records:
        v = vmap[r["id"]]
        r["notes"] += f"；向量 top1={v['vector_top1']}（{v['vector_top1_source']}）"

    # ---- 泄题检查（多跳问题 vs 自己的 quote）----
    gold_items = [
        rre.GoldItem(id=r["id"], type=r["type"], question=r["question"], anchors=r["anchors"])
        for r in mh
    ]
    problems = rre.check_no_leak(gold_items)
    if problems:
        print(f"✗ 泄题 {len(problems)} 条：")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)

    # ---- 与已有 80 条事实题的 quote 查重 ----
    # 注意：多跳的锚点与周二某条事实题指向**同一行原文**是合法的 ——
    # 多跳问的是"两处合并"，事实题问的是"单处事实"，考的东西不同。
    # 真正要拦的是：**同一题型内**的重复标注，以及多跳两锚点互为同文。
    existing = [
        json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    # 只拦完全同题：question 归一化后相同（换问法也算重复，README §〇）
    def norm(text: str) -> str:
        return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")

    existing_qs = {norm(e["question"]) for e in existing}
    dup_q = [r["id"] for r in mh if norm(r["question"]) in existing_qs]
    if dup_q:
        print(f"✗ 与事实题问题重复：{dup_q}")
        raise SystemExit(1)

    # ---- 合并交错落盘 ----
    merged = existing + mh + refusal_records
    by_type: dict[str, list[dict]] = {}
    for r in merged:
        by_type.setdefault(r["type"], []).append(r)
    interleaved: list[dict] = []
    while any(by_type.values()):
        for t in ("fact", "multi_hop", "refusal"):
            if by_type[t]:
                interleaved.append(by_type[t].pop(0))
    OUT.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in interleaved),
        encoding="utf-8",
    )
    from collections import Counter

    kinds = Counter(r["type"] for r in interleaved)
    print(f"✓ 合计 {len(interleaved)} 条 → {OUT.relative_to(PROJECT_ROOT)}：{dict(kinds)}")


if __name__ == "__main__":
    main()
