# RAG 三段式架构（W4 骨架版）

> 对应提交：`feat: rag skeleton and document schema`
> 代码：`app/rag/` · 测试：`tests/test_rag_skeleton.py`（48 条）·
> 决策：[design-decisions.md](design-decisions.md) ·
> 数字用来复跑：`uv run python -X utf8 scripts/rag_cost_compare.py`

---

## 一、全景图

一张图说清"每段的输入、输出、谁在做工"。**箭头上的字就是类型名** ——
类型对不上，链路就断了，这比任何文字描述都严格。

```
┌────────────────────────── 索引段（离线，一次） ──────────────────────────┐
│                                                                        │
│   文件            ① 解析              ② 切分              ③ 向量化       │
│  .pdf ──────────▶ loaders/ ───────▶ splitters/ ────────▶ embedding/    │
│  .docx            load_document       RecursiveSplitter   HashingEmbedder│
│  .md              │                   │                   │            │
│                   ▼                   ▼                   ▼            │
│              List[Document]        List[Chunk]        List[List[float]]│
│              (text,page,           (+id 内容派生)      (L2 归一化)      │
│               source,metadata)                                         │
│                                                                        │
│                                          ④ 入库                       │
│                                    stores/ VectorStore.upsert()        │
│                                           │                            │
└───────────────────────────────────────────┼────────────────────────────┘
                                            ▼
                                   ┌──────────────────┐
                                   │   向量库（持久）   │
                                   │  ChromaDB / 内存  │
                                   └────────┬─────────┘
                                            │
┌────────────────────────── 检索段（在线，每次） ─────────────────────────┐
│                                           │                            │
│   用户问题           ① query 向量化        │  ② 相似度 + ③ TopK          │
│   "违约金怎么算" ───▶ 同一个 embedder ────▶ store.search(vector, k)      │
│                        │                  │                            │
│                        └──────────────────┴──▶ List[SearchHit]         │
│                                                 (chunk + score)        │
└─────────────────────────────────────────────────┬──────────────────────┘
                                                  │
┌────────────────────────── 生成段（在线，每次） ──┼──────────────────────┐
│                                                  ▼                     │
│  ① 拼上下文 build_context         ② 调用模型          ③ 引用回填         │
│  [1] 来源：合同.pdf:p7   ───▶  client.chat ───▶  Answer(text,           │
│  违约金按合同总额…                .completions       citations)         │
│  [2] 来源：合同.pdf:p9           .create()         [1] 合同.pdf 第 7 页 │
└────────────────────────────────────────────────────────────────────────┘
```

### 三段的一句话职责

| 段 | 职责 | 输入 → 输出 | 什么时候跑 |
| --- | --- | --- | --- |
| **索引** | 把文件变成"能被单独命中的块" | 文件 → `List[Chunk]` + 向量 → 库 | 上传/同步时，**一次** |
| **检索** | 从已有块里挑最相关的几条 | `str` → `List[SearchHit]` | 每次提问 |
| **生成** | 拼资料、调模型、标引用 | `List[Chunk]` + `str` → `Answer` | 每次提问 |

**这张表最重要的一列是最后一列**。索引慢不等于 RAG 慢：
把索引耗时算进"问答延迟"，就会为"首字快"去砍块数，
而真正该优化的是检索的 top_k 与生成段的上下文长度。

---

## 二、每段的输入输出（逐个签字）

### ① 解析：`Path` → `List[Document]`

```python
Document(source: str, page: int | None, text: str, metadata: dict)
```

| 已有能力 | 位置 | 说明 |
| --- | --- | --- |
| PDF（逐页、页码、去页眉页脚、跨页断句标记） | `loaders/pdf.py` | W4 之前就在，本格**直接复用** |
| DOCX（一段一 Document、表格整张一块） | `loaders/docx.py` | 同上 |
| Markdown（按标题切逻辑块） | `loaders/markdown.py` | 同上 |

这一步的输出**不是**"干净的纯文本"，而是"带出处的文本"。
`page` 为 `None` 是合法值（Markdown/DOCX 没有物理页），
**但不许伪造 `page=1`** —— 理由见下一节。

### ② 切分：`List[Document]` → `List[Chunk]`

```python
Chunk(text, page, source, metadata, index, pages, id=内容派生)
```

| 规则 | 取值 | 为什么 |
| --- | --- | --- |
| 块长 | 500 字符 | ≈ 一段完整论述（中文 ≈ 350 token）；A4 合同的一个条款通常 200–600 字，500 让**大多数条款整块落在一个块里** |
| 重叠 | 50 字符（10%） | 跨块断句只能靠重叠缓解；存储涨 10% 是必须付的钱 |
| 边界优先级 | 段落 → 换行 → 句末标点 → 空格 → 硬切 | 越靠前越语义完整；硬切只在"整段没有标点"时发生 |
| `id` | `sha256(source, page, text)[:16]` | 幂等入库、跨进程可复现 |

**`index` 不参与 `id` 计算**：否则改一次 `chunk_chars` 会让整库 id 全部失效 ——
"只调了个参数"表现成"索引需要重建"，幂等性就被参数调优打穿了。

### ③ 向量化：`List[str]` → `List[List[float]]`

```python
Embedder.embed(texts: Sequence[str]) -> list[list[float]]
```

接口**只有批量**，没有 `embed_one`：单条调用一旦存在，调用方就会一条一条调，
本地模型的批处理收益直接归零。也**不是 async** —— 本地 embedder 是纯 CPU 计算，
套一层协程只会掩盖真正的性能位置。

当前装配 `HashingEmbedder`（字符 n-gram 哈希 + L2 归一化）。它是**占位实现**：

| 保证 | 不保证 |
| --- | --- |
| 相同字面 → 相同向量（跨进程一致，用 blake2b 不用内置 `hash()`） | **语义相似**（"付款"与"支付"无关） |

> **所以：本实现下任何召回率数字都没有参考价值。** 拿它调参 = 优化噪声。
> 换真模型（sentence-transformers）后跑同一套测试，差异只来自向量质量 ——
> 这正是现在把接口定下来的用途。

### ④ 入库：`Chunk + vector` → `store.count()`

```python
VectorStore.upsert(chunks, vectors) -> int   # 必须幂等
VectorStore.search(vector, top_k)  -> list[SearchHit]
VectorStore.count() / clear()
```

- **必须是 `upsert` 不是 `insert`**：`Chunk.id` 内容派生，重跑同一份文档是常态。
  语义若是 insert，"重跑一次"条数就翻倍，而检索看起来还挺正常。
- **`chunks` 与 `vectors` 数量不一致必须抛异常**，不能截断到较短的 ——
  静默截断会把"漏了一条"伪装成"索引成功但检索差一点"。
- `SearchHit` 自带 `Chunk` 快照（不只 `id + score`）：生成层不需要第二次回查，
  也就不存在"回查时索引已经变了"的窗口。

### ⑤ 检索：`str` → `List[SearchHit]`

```python
Retriever(embedder=同一个, store=同一个).retrieve(query, top_k=4)
```

三条硬规则：

1. **必须用与索引完全相同的 embedder** —— 不在同一空间时余弦算出来是噪声，**且不报错**。
   所以 `Retriever` 构造时要求显式传 embedder，**不给默认值**（默认值在这里是有害的）。
2. **空 query 直接返回空**，不调 embedder：零向量会返回一堆"分数全 0"的块，
   看起来检索成功，实际是随便挑的。
3. `top_k` 默认 4（≈ 2000 字 ≈ 1400 token）。取 10 会把上下文推到 3500 token，
   而在召回率未验证的前提下，多给的 6 条大多是噪声 —— **噪声进上下文会实实在在拉低答案质量**。

### ⑥ 生成：`List[Chunk]` + `str` → `Answer`

```python
Answer(text, citations: list[Citation], context_chars, used_chunks)
Citation(index, source, page, chunk_id) → label = "合同.pdf 第 7 页"
```

- **上下文里每块带编号 `[n]`，要求模型在答案里标注 `[n]`**。
  编号 → `Chunk` 的映射由我们掌握，模型只负责"引用哪一条"，
  不负责"那一条在哪个文件、哪一页"（它会把页码说错，而且很自信）。
- **超预算整块丢弃，不截断块**：截一半的块会让模型读残句，
  然后照着残句给出一个"看起来有依据"的错答案 —— 比少给一块糟得多。
- **零命中时直接返回固定话术，不调模型**：让模型对着空上下文回答，它唯一能做的就是编。

---

## 三、为什么 `page` 从第一步就必须在结构里

这是本次骨架里唯一"后悔就来不及"的设计点。

**事后补不回来的机制**：文本一旦切块、向量化、入库，页码只存在于切分那一刻的
局部变量里。等上线后发现"用户想点开原文第几页核对"，你面对的是几千条向量 ——
唯一出路是**全量重新解析 + 重新切分 + 重新向量化**。

按本次实测的量级算（每页 1200 字、每块 500 字）：

| 文档规模 | 块数 ≈ | 重跑一次要做什么 |
| --- | ---: | --- |
| 50 页 | ~120 | 解析 + 切分 + 120 次 embed + 120 次写入 |
| 5000 页（一个知识库） | ~12000 | 同样四步 ×100，**且期间检索结果是新旧混杂的** |

所以规则写成代码里的约束，而不是文档里的一句话：

- `Document.page`：`int | None`，`< 1` 直接 `ValueError`
- `Chunk.page` + `Chunk.pages`：**两个字段**，`page` 给引用展示（`p7`），
  `pages` 给覆盖范围（`p7-9`）。合成一个会丢掉其中一个真相。
- `page=None` 一路传到底，`location` 退化成文件名 —— **不伪造页码**
- `metadata` 扁平化时 `page` 用 `-1` 编码 `None`，`restore_location` 是唯一的解码点

测试锁住的不变量（`tests/test_rag_skeleton.py`）：

```
test_chunk_page_is_carried_through_splitting   页码一路传到块上
test_chunk_keeps_none_page_for_non_paged_sources   None 不许变成 1
test_chunk_location_covers_multiple_pages       跨页读成 p7-9
test_metadata_roundtrip_through_store_encoding  编码/解码不漂移
```

---

## 四、目录映射：文件 ↔ 责任

```
app/rag/
├── schema.py          ★ Document / Chunk / Metadata（结构级契约，本次新增）
├── __init__.py        re-export 契约；LoadResult 留在这里
│
├── loaders/           ① 解析（W4 之前已有，本格未改逻辑）
│   ├── __init__.py    LOADERS 注册表：加格式只改这里一行
│   ├── pdf.py · docx.py · markdown.py
│
├── splitters/         ② 切分（本次新增）
│   ├── base.py        ChunkSize（含 overlap<chunk 约束）+ Splitter Protocol
│   └── recursive.py   RecursiveSplitter：语义边界优先 + 尾部重叠
│
├── embedding/         ③ 向量化（本次新增）
│   ├── base.py        Embedder Protocol（dimension + 批量 embed）
│   ├── hashing.py     HashingEmbedder（占位实现，见 §2 ③ 的边界说明）
│   └── __init__.py    get_embedder() 装配点
│
├── stores/            ④ 入库 + 检索底座（本次新增）
│   ├── base.py        VectorStore Protocol + SearchHit
│   ├── memory.py      InMemoryVectorStore（O(n·d) 暴力全扫，几千条内）
│   └── __init__.py    get_store() 装配点
│
├── generation/        ⑥ 生成（本次新增）
│   └── base.py        build_context / generate_answer / Answer / Citation
│
├── indexing.py        Indexer + IndexReport（串起 ①②③④）
├── retrieval.py       Retriever + RetrievalResult（串起 ⑤）
├── pipeline.py        RagPipeline（唯一装配点：强制共享 embedder 与 store）
│
├── summarize.py       map-reduce 摘要（**W4 之前的代码，本次只改名**）
└── replay.py          DEMO_REPLAY 回放（未改）
```

### 本次做的一处改名（必须说清）

`summarize.Chunk` → `summarize.SummarySegment`，`chunk_documents` → `segment_documents`。

**为什么非改不可**：`Chunk` 这个名字现在属于"要落库、有稳定主键"的索引单位。
摘要那边的"分段"没有 id、不落库、只活在单次请求里。两者形状相似但生命周期完全不同 ——
名字不分开，代价是具体的：想给检索块加个 `score`，会发现自己在改摘要链路的类型。

**改动范围**：4 个文件（`summarize.py` / `api/v1/summarize.py` /
`scripts/latency_breakdown.py` / `tests/test_rag.py`），逻辑零改动，375 条测试全绿。
参数名 `chunk_tokens` **保留不改** —— 它是 `/summarize/stream` 已暴露给前端的字段，
改它等于改 HTTP 契约。**内部命名可以迁移，对外契约不该跟着动。**

---

## 五、装配：唯一正确的接法

```python
pipeline = build_pipeline()  # 默认 memory store + hashing embedder
pipeline.index_file("合同.pdf")  # 索引段
retrieved = pipeline.retrieve("违约金怎么算")  # 检索段（不调模型，联调先看它）
answer = await pipeline.answer("违约金怎么算", client=client, model=model)  # 生成段
```

`RagPipeline` 只有 40 行、没有状态机 —— 它存在的唯一价值是**把三条隐性规则写死在一处**：

1. embedder 必须索引/检索共用
2. store 必须共用同一个实例
3. 切分参数不能中途换

这三条里任何一条被忘记，链路**都不会报错，只会答案变差**。所以它们两条挂在构造
函数上（`Retriever` 强制传 embedder），一条挂在测试上
（`test_pipeline_shares_embedder_and_store_between_phases`）。

---

## 六、当前能做什么 / 不能做什么（别把骨架当成品）

| 能力 | 状态 | 缺什么 |
| --- | --- | --- |
| 三段链路跑通（离线，无 key） | ✅ 已验证，48 条测试 | — |
| 幂等入库（重复索引不翻倍） | ✅ 已验证 | — |
| 页码溯源到 `p7` / `p7-9` | ✅ 已验证 | — |
| 答案带 `[n]` 引用 + 引用列表 | ✅ 已验证（假 client） | 真模型的引用准确率**未测** |
| **语义检索** | ❌ | 真 embedding 模型（换 `get_embedder` 一行） |
| **持久化**（重启不丢） | ❌ | ChromaDB（换 `get_store` 一行） |
| **增量更新/删除** | ❌ | `VectorStore` 没有 `delete_by_source` —— 接口宁缺勿滥，有调用方再加 |
| **RAG 效果的召回率/准确率** | ❌ **现在测了也不算数** | 真模型 + 评测集 |

**"未验证"和"已实现"要分开写**。上表里 ✅ 的每一行都有对应测试名，
❌ 的每一行都写清了缺什么 —— 不写"待完善"。

---

## 七、下一步（按依赖顺序，不是按重要程度）

1. **换真 embedding 模型** → 立刻跑一次召回率基线（这一步之前所有质量数字都无意义）
2. **接 ChromaDB** → 落持久化，然后才有"重启后检索还对不对"这个测试
3. **加 `delete_by_source`** → 增量更新，接口等调用方出现再加
4. **评测集** → 20 条问题 × 标准答案（做法复用 `eval/prompt-sets/` 的采集纪律）
5. **API 端点** → `POST /api/v1/rag/ask`，契约先写进 `docs/api-spec.md` 再写代码
