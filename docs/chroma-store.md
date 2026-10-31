# ChromaDB 持久化向量库

> 代码：`app/rag/stores/chroma.py` · 测试：`tests/test_chroma_store.py`（39 条）·
> 数据可复跑：`scripts/chroma_acceptance.py` · 上游：[embedding.md](embedding.md)

---

## 0. 必答题：`distances` 在 cosine 空间下是什么含义

**cosine 距离 = 1 - 余弦相似度。越小越像。方向与相似度相反。**

搞反阈值逻辑就全错 —— 而且不会报错，只会"筛出来的是最不像的"。

### 实测（三个构造向量，查询为 `[1,0,0]`）

| 候选 | 与查询的余弦 | Chroma `distance` | `1 - distance` |
| --- | ---: | ---: | ---: |
| A 同向 | **1.0** | **0.0** | 1.0 |
| B 正交 | 0.0 | 1.0 | 0.0 |
| C 反向 | **-1.0** | **2.0** | -1.0 |

三条结论，逐条都是坑：

1. **`distance = 1 - cosine`，误差 0.00e+00** —— 不是近似式，是定义式
2. **范围是 [0, 2]，不是 [0, 1]** —— 因为余弦可以是负数。默认 space 是 `l2`（范围 [0,4]），
   两者数值完全不可比
3. **返回顺序是距离升序**（A → B → C），即越像越靠前

### 最常见的误用

```python
# ❌ 以为 distance < 0.8 是"足够像"
if distance < 0.8:
    ...
# 这句在 cosine 空间里的真实含义是：相似度 > 0.2 就算命中
# 比你以为的宽松得多，几乎不会筛掉任何东西

# ❌ 方向写反
if distance > 0.8:
    ...
# 筛出来的是**最不像**的那一批
```

### 所以本项目的做法：把口径统一成相似度

`SearchHit.score` 被接口钉死为**余弦相似度（越大越像）**：

| 实现 | 底层的数值 | 对外 `score` |
| --- | --- | --- |
| `ChromaStore` | distance（越小越像） | **`1 - distance`** |
| `InMemoryVectorStore` | 余弦（越大越像） | 原值 |

上层只认一种口径，写阈值时不需要知道底下是谁。
`test_both_stores_agree_on_score_direction` 断言两个实现返回**同一组分数和同一个顺序** ——
这条防的正是"换库那天阈值逻辑突然全错"。

> 周三时 `SearchHit.score` 的注释写的是"语义由具体 store 决定"。
> 周四改成"钉死为余弦相似度"，理由就是这道必答题：
> **两种口径混在同一个类型里，调用方必然有一半概率搞反。**

---

## 1. 持久化：必须显式给 path

```python
chromadb.Client()  # ❌ 内存/临时，进程退出就没了
chromadb.PersistentClient(path="data/chroma")  # ✅
```

**这是 W4 最高频的"数据怎么丢了"**。实测：写 2 条 → `del` 掉 client →
新建 `PersistentClient` 打开同一 path → `count()` 仍是 2，metadata 也还在。

### 默认路径的两个决定

| 决定 | 理由 |
| --- | --- |
| 相对项目根（`data/chroma`），**不是 `~/.cache/chroma`** | 备份脚本会带上；换机器 `cp -r` 就够。Chroma 默认写到用户目录，那是"换台机器数据就没了"的典型来源 |
| 用 `Path(__file__).parents[3]` 而不是 `Path.cwd()` | cwd 取决于你从哪启动 uvicorn（项目根 / tests / Docker 的 `/app`），用 cwd 拼相对路径会让"数据在哪个目录"随启动方式变化 |

第二条是"数据怎么丢了"的第二种成因，比第一条更隐蔽：**同一条命令在不同的
工作目录下会打开两个不同的库**，而两边都不报错。

---

## 2. collection 与元数据规范

### 建 collection

```python
client.get_or_create_collection(
    name="industrial_rag",
    metadata={"hnsw:space": "cosine"},
    embedding_function=<守卫，见 §4>,
)
```

三条硬要求：

| 项 | 值 | 不这么做会怎样 |
| --- | --- | --- |
| `hnsw:space` | **`cosine`** | 默认是 `l2`，distance 范围变 [0,4]，数值含义完全不同 |
| 名字长度 | **≥ 3 字符** | `"t"` 直接 `ValueError: Expected a name containing 3-512 characters`（实测） |
| 首尾字符 | 字母数字 | `"合同"` 之类会因非 `[a-zA-Z0-9._-]` 被拒 |

### 元数据字段规范（定版）

| 字段 | 类型 | 例子 | 用途 |
| --- | --- | --- | --- |
| `source` | str | `合同A.pdf` | 按文档过滤 / 引用展示 |
| `page` | int | `7` / `-1` | 按页码过滤；`-1` 编码"无页" |
| `pages` | str | `6,7` | 跨页块覆盖范围（逗号分隔） |
| `chunk_index` | int | `3` | 块序号，与 `store_id` 对齐 |
| `doc_id` | str | `a1b2c3d4e5` | 按文档批量删除 |
| `content_id` | str | 16 位哈希 | **内容派生 id**，判定"是不是同一内容" |
| `block` | str | `text`/`table`/`heading` | 区分表格块 |

**全部是扁平标量**（str/int/float/bool）—— 这条约束是 Chroma 逼出来的：
它的 `where` 只支持比较与逻辑组合，嵌套结构不支持。

`pages` 用逗号字符串而不是列表，是因为 Chroma 不接受 list 类型的 metadata 值
（它只存标量）。这在结构上就把"元数据必须可移植"这件事钉住了。

### 两种 id 的分工（本格最重要的设计）

`Chunk.id`（内容派生）与 `Chunk.store_id`（位置派生）**两者都要有**：

| | `id` | `store_id` |
| --- | --- | --- |
| 格式 | `sha256(source,page,text)[:16]` | `{doc_id}_{page}_{chunk_index}`，如 `a1b2c3d4e5_7_2` |
| 变了会怎样 | 内容变 → 新记录（旧记录成孤儿） | 位置不变 → **原地覆盖** |
| 用途 | 引用去重、跨库一致性 | **向量库主键**、按文档/页码定位 |
| 可读性 | 无 | **有**（打开库就能看出是哪份文档的哪一块） |
| 存在哪 | metadata 的 `content_id` | Chroma 的主键位 |

**为什么存储主键用位置派生**：任务要求"可复现"，且要让"第 7 页第 2 块"
在库里直接可查。用哈希做主键，排查时只能拿哈希反查，而那段代码通常还没写。

**代价（必须写清）**：改切分参数后 `store_id` **会指向不同内容**。
`chunk_chars` 从 500 改到 800，第 3 块可能从"违约条款"变成"付款条款"，
但两条记录的 `store_id` 都是 `xxx_7_2` —— upsert 会**静默覆盖**。

不是 bug，是取舍的另一面。应对方式是**重建前按 source 删干净**
（`delete_by_source`），而不是指望 id 帮你发现不一致。

---

## 3. 入库：批量 + 自定义 embedding

### 为什么不让 Chroma 自己算 embedding

不传 `embedding_function` 时 Chroma 会加载它自带的 ONNX MiniLM。本次探测时
**建集合直接卡住下载，跑到超时被 kill**。两个后果：

1. 离线环境建不了库
2. 更糟：如果有人忘了传 `embeddings=`，Chroma 会**静默**用那个模型算向量 ——
   库里一半向量来自 bge、一半来自 MiniLM，**检索结果变得无法解释**

所以传一个"会抛异常"的守卫 EF。实测守卫生效：`query_texts=` 会立刻
`RuntimeError: Chroma 试图自己算 embedding`。

守卫必须实现**五个方法**（Chroma 1.5.9 的完整 EF 合同，缺一个就炸或刷警告）：

| 方法 | 缺了会怎样 |
| --- | --- |
| `__call__` | 写入路径炸 |
| `embed_query` | `query_texts=` 时 `AttributeError`（炸得让人看不懂） |
| `name()` | `get_or_create_collection` 时 `AttributeError` |
| `__init__` | 每次建集合刷一条 DeprecationWarning |
| `get_config` / `build_from_config` | 同上 |

> "每次跑测试都刷一条废警告"的结果是所有人开始无视警告，
> 那会让真正的警告一起被忽略。所以这些都补齐了 —— 现在 39 条测试零警告。

### 批量入库

```python
vectors = embedder.embed([c.text for c in chunks])  # 一次 encode 100 条
store.upsert(chunks, vectors)  # 一次 upsert 100 条
```

逐条 `add` 在 100 条时就能看出差距，在 15,000 条（5000 页）时是
"几十秒 vs 几分钟"，而且中间失败会留下半份索引。

### 100 条真实入库验收

`scripts/chroma_acceptance.py`：5 份文档 × 20 条互不相同的条款 = 100 条。

| 检查 | 结果 |
| --- | --- |
| 入库条数 | 100 |
| **重启后 count** | **100** ✓ 持久化生效 |
| 落盘大小 | 0.7 MB |
| where `source=合同B` | 命中 20 条，全部来自该文档 ✓ |
| `search_in_pages(2..3)` | 命中 20 条，页码全部在区间内 ✓ |

---

## 4. 检索验证：10 条已知答案

**本周不做量化**，所以判断是人工的：看 top3 里有没有对应原文。

| 问题 | top1 | 命中位置 | 文档对 |
| --- | --- | --- | :---: |
| 违约了要赔多少钱 | 合同B_p1 | top1 | ✓ |
| 什么时候付钱 | 合同A_p1 | top1 | ✓ |
| 验收不合格怎么办 | 合同A_p2 | **未命中** | ✓ |
| 软件的著作权归谁 | 合同D_p1 | top1 | ✓ |
| 保密要保密多久 | 合同E_p2 | top2 | ✓ |
| 地震导致延期谁负责 | 合同B_p5 | top1 | ✓ |
| 打官司去哪里仲裁 | 合同E_p4 | top3 | ✓ |
| 设备送到什么地址 | 合同C_p1 | **未命中** | ✓ |
| 保修期多长 | 合同B_p1 | top1 | ✓ |
| 工程师多久能到现场 | 合同B_p4 | top1 | ✓ |

**top3 内容命中 8/10（80%）；top1 文档正确 10/10。**

### 两条"未命中"的真实原因（人工判断）

**① "验收不合格怎么办"** —— top1 命中"验收不通过：乙方须在十个工作日内完成整改"。
**语义上其实答对了**，只是没命中我指定的那一段（"书面确认验收合格"）。
这是**字符串匹配判据的局限**，不是检索缺陷。

**② "设备送到什么地址"** —— top3 是"设备清单 / 到货验收 / 序列号登记"，
而"文一西路 969 号"（交付地点）没进 top3。**这是真实的检索失败**：
"设备"这个词把三条都带"设备"的条款顶了上来，而"地址"的语义权重不够。

### 三个必须记的教训

**教训一：语料重复会把 top_k 名额吃光。**

第一版让 5 份文档用**完全相同**的 20 条条款，结果 top3 全被同一段文本占满
（内容一样、只是 source 不同，分数完全相同），10 条问题里 2 条"未命中"。
**那次未命中不是检索失败，是语料缺陷把名额吃掉了。**

这是 RAG 里的真实陷阱：知识库里有几份近似文档（不同版本的合同、同一条款的副本）时，
top_k 被同一段内容占满 —— 看起来"检索到了"，实际上下文窗口浪费在重复内容上。

**教训二：gold 写错时，"未命中"看起来和检索失败一模一样。**

我写了 `("要派几个工程师到现场", "2 名工程师", ...)`，
而 20 条条款里**根本没有"2 名工程师"**（那是周三 A/B 集里的句子）。
结果这条被当成了检索缺陷去查 —— 时间全花在一个不存在的目标上。

所以脚本现在开头就调 `assert_gold_exists()`：gold 不在语料里直接退出。
**测试数据本身出错，是最贵的错**：它不报错，只让所有结论失真。

**教训三：只判"内容对不对"会放过"文档错"。**

所以 `KNOWN_ANSWERS` 每条都带期望文档，判两件事：
内容在不在 top3、top1 的 source 对不对。只看内容的话，
5 份文档里内容相同的块会被当成命中，掩盖"过滤没生效"这类问题。

---

## 5. 一个测试抓出来的真实 bug（`index_text` 的覆盖语义）

写完 Chroma 后跑全套测试，**周三的 bge 端到端测试挂了**：
它想入 3 条不同条款再检索，结果命中不对。

根因不在 Chroma，在 `RagPipeline.index_text`：

```python
pipeline.index_text("第一条 付款条款", source="合同")  # 库里有 1 条
pipeline.index_text("第二条 违约条款", source="合同")  # 还是 1 条！
pipeline.index_text("第三条 质保条款", source="合同")  # 还是 1 条
```

因为每段文本都是该 source 的"第 0 块"，`store_id` 都是 `xxx_-1_0` ——
后一次静默覆盖前一次。而**每次 `IndexReport.ok` 都是 `True`**。

### 修复：显式 `append` 开关 + index 偏移

| 用法 | 语义 | store_id |
| --- | --- | --- |
| `index_text(t, source=s)`（默认） | **这段就是该 source 的全部内容** —— 先删后写 | `_0` |
| `index_text(t, source=s, append=True)` | 追加到已有内容后面 | `_{已有块数}` |

`append=True` 时用 `store.count_by_source(s)` 算偏移量，
让新块的 `index` 接在已有块后面，`store_id` 就不冲突了。

**默认是覆盖而不是追加**，因为"重新上传同一份文件"是更常见的操作 ——
旧版本必须消失，否则"改了合同再上传"会让新旧两版同时被检索到。

### 这个 bug 的价值

**一次"看起来与 Chroma 无关"的测试，抓出了索引语义的缺陷。**
这正是不该把旧测试删掉、也不该让新 store 只跑新测试的理由 ——
换 store 时跑全套测试，才能让这种"跨层的语义冲突"浮出来。

---

## 6. where 过滤：范围查询的写法坑

```python
# ❌ 直接报错：ValueError: Expected operator expression to have exactly one operator
where = {"page": {"$gte": 1, "$lte": 3}}

# ✅ 必须 $and + 两个独立的比较
where = {"$and": [{"page": {"$gte": 1}}, {"page": {"$lte": 3}}]}
```

这不是文档里显眼的地方，是实测踩到的。所以接口上把它做成**独立方法**
（`search_in_pages`）而不是往 `search` 里塞 kwargs：

**两种语义混在一个参数里，"传了但没生效"会变成沉默故障。**
`search(where=...)` 是精确匹配，`search_in_pages(page_from, page_to)` 是范围，
各自有明确的类型与校验（`page_from > page_to` 直接抛，不静默返回空）。

---

## 7. 明确未做 / 未验证

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 多进程 / 多 worker 写入 | ❌ 未验证 | Chroma 的 SQLite 后端写并发差，多 worker 同时写会锁。这是"该换 Milvus"的第一个信号 |
| 阈值（相似度下限）选多少 | ❌ 未定 | 需要评测集才能定；现在 `top_k` 是唯一的截断手段 |
| 混合检索（向量 + BM25） | ❌ 未做 | "设备送到什么地址"这类**关键词强、语义弱**的查询，纯向量会失败 —— 那 2 条未命中就是证据 |
| 元数据过滤的性能 | ❌ 未测 | where 是否走索引、10 万条时会不会退化成全扫 |
| 索引重建的耗时 | ❌ 未测 | 15,000 条要多久、期间检索可用性 |
| collection 迁移 / 版本 | ❌ 未做 | 换 embedder 后旧 collection 怎么处理（目前只能 `clear` 重建） |
| 真实业务文档 | ❌ 未验证 | 100 条语料是我写的合成合同条款，页密度与真实文档不同 |
