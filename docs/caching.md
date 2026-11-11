# 缓存三层设计与幂等（docs/caching.md，W12 周二）

> 本周主题：**同一件事只付一次钱**。三层缓存覆盖三次成本：
> embedding（编码成本）、检索（查询理解成本）、答案（模型调用成本）。
> 全文的关键不是"加了缓存"，而是"缓存什么时候必须**不**命中"。

## 0. 一张总表

| 层 | 后端 | key 组成 | TTL | 省的是什么 | 代码 |
| --- | --- | --- | --- | --- | --- |
| embedding | sqlite（按模型分文件） | `sha256(model \x1f dim \x1f text)` | 永久 | 编码 CPU（bge ~10ms/条） | `embedding/cache.py`（W4 已有） |
| 检索结果 | 内存 LRU（2000 条） | `sha256(规范化query \x1f 配置指纹 \x1f k)` | 30 min | embed+search（实测 ~11ms → ~1ms） | `rag/result_cache.py` + `cached_qa.py`（本周） |
| 答案 | 内存 LRU（2000 条） | `sha256(规范化query \x1f 上下文指纹 \x1f prompt指纹 \x1f model)` | 5 min（拒答 2 min） | **一次模型调用 ≈ 2700 token** | 同上 |
| 生成幂等 | 内存 dict | `(defect_class, count, size, seed, model_version)` | 任务 TTL 1h | 一次 GPU 推理 | `generation_queue.py`（W8 已有，本周补 model_version） |
| 会话幂等 | sqlite turns 表 | `(session_id, role='user', content)` | 永久 | 一次重复问答 | `session_store.py`（本周） |

## 1. 配置指纹：本周最关键的一个决定

**检索结果缓存的 key 必须绑定"当前链路配置"**，否则改了 top_k 之后缓存
还在返回旧 k 的结果 —— 旧结果冒充新链路，无任何报错。

这是 BM25 分词器指纹（W6 周一技术决定 ③）的放大版：

| | W6 的坑 | 本周的坑（被指纹挡住） |
| --- | --- | --- |
| 不一致的两侧 | 索引文件 vs 查询分词器 | 缓存 vs 当前链路配置 |
| 症状 | 分数系统性偏低，人怀疑"BM25 不适合中文" | 效果"没变化"，人怀疑"参数没生效" |
| 防线 | 索引文件存分词器指纹 | 缓存 key 存链路配置指纹 |

指纹**显式列出**七个字段（strategy / chunk_chars / overlap_chars /
top_k / score_threshold / max_context_chars / embedder 名）而不收
RagConfig 对象 —— 收对象的话，RagConfig 加字段时这里不会自动跟上，
"新参数影响结果却没进指纹"就是串味的起点。显式列出 = 加参数的人
被迫路过这一行。实现：`result_cache.retrieval_config_fingerprint`，
一致性测试①守着（换配置必 miss）。

**索引内容变化不在指纹里**（块 id 是内容派生的）—— 语料一变必须显式失效：
`ResultCache.invalidate_retrieval_all()` 同时清检索层与答案层
（只清检索层是"半失效"，旧检索上的旧答案全部过期 —— 这条写在实现里，
是缓存一致性最常见的缺口）。

## 2. 答案层 key 为什么含"会话上下文指纹"（必答问题）

多轮对话里同一条 query 在不同上下文中答案不同。不加指纹，用户第 2 轮问
"它呢"会拿到第 1 轮缓存的答案 —— **最容易被用户发现、最难复现的 bug**。

指纹 = 最近 4 条用户消息的哈希（`context_fingerprint`，窗口取 4 足够：
RAG 问答的指代距离很少超过 2 轮）。效果：

- 上下文变了（真多轮）→ key 变 → miss → 重新答 ✅
- 同一轮重复提交（网络重试/双击）→ key 不变 → 命中 ✅

一致性测试④（`test_context_change_forces_answer_miss`）守着这条。

## 3. prompt 指纹：从全文哈希，不靠版本号

答案 key 里的 prompt 段是 `sha256(SYSTEM_PROMPT + ANSWER_PROMPT)` 全文 ——
**不手维护版本号**。版本号会忘改（改了 prompt 忘 bump = 旧答案继续生效），
全文哈希不会。一致性测试②守着。

## 4. 负缓存与"别把超时缓存成空"

| 结果类型 | 缓存吗 | TTL | 理由 |
| --- | --- | --- | --- |
| 检索 0 命中 | ✅ | 与正常条目同（30min） | 空是检索的真相，失效条件与正常条目一致（配置/语料） |
| 拒答（分数闸门） | ✅ | **2 min** | 拒答由 `min_score` 决定，阈值不在 key 指纹里；短 TTL 保证调阈值实验尽快绕开旧条目 |
| 超时/异常/降级 | ❌ **永不** | — | 把"服务暂时坏了"缓存成"没找到"，2 分钟内用户拿到确定错误的答案 |

"异常不缓存"**不是靠自觉，是靠结构**：`CachedQAEngine` 没有 try/except
包住 engine 调用 —— 异常向上抛，写缓存的代码在正常返回之后才执行，
异常路径根本到不了写点。测试⑤（`test_exception_path_never_caches`）
验证：抛 TimeoutError 后缓存条目数必须为 0。

## 5. 检索层缓存为什么存 id+分数而不是整块

`CachedRetrieval` 存 `(chunk_ids, scores)`；命中时用
`store.get_by_content_id`（既有接口，引用可校验的第三跳）回查块文本。

- **为什么不存 Chunk 列表**：语料重切后同一 id 的内容可能变，
  回查保证"缓存 key 没变但语料变了"的窗口内拿到**当前**块内容；
- **回查失败怎么办**：任一块取不到 → 整条当 miss 走全链路
  （`store_miss_on_hit` 计数），**绝不返回残缺结果** ——
  宁可贵一次，不给调用方一个静默变短的命中列表。测试⑥守着。

## 6. 幂等（两条线）

**项目一（生成）**：幂等键 `(defect_class, count, size, seed, model_version)`。
W8 已有前四项，本周补 `model_version = generator.model_name` ——
不加它的坑：换模型版本后相同参数命中旧模型的产物并复用。
与配置指纹同一条设计：**派生数据的 key 必须绑定产生它的生产者版本**。
seed=None 仍不参与幂等（"每次要新的"语义，W8 原判不动）。

**项目二（问答）**：`SessionStore.append_turn` 对
`(session_id, role='user', content)` 内容级去重 —— 重复提交返回已有 seq，
不产生第二条。边界三条（都有测试）：

1. 幂等的范围是 **session**：两个会话问同一句话各自独立；
2. 只去重 **user** 消息：assistant 两条同内容是合法的（不同轮次产物）；
3. assistant 行是系统产物，append 调用点唯一，天然不需要去重。

## 7. 命中率与成本关系（周二 16:30 的数据）

离线实测（`scripts/cache_cost_compare.py`，20 条 query × 2 轮，hashing）：

| 轮 | mean | p50 | p99 | 命中率 |
| --- | --- | --- | --- | --- |
| 冷 | 0.11 ms | 0.10 ms | 0.16 ms | 0/20 |
| 热 | 0.01 ms | 0.01 ms | 0.01 ms | **20/20 = 100%** |

检索延迟下降 **94.4%**。注意口径的诚实性：

- hashing 查询编码毫秒级，这个数字**证明机制生效**，不代表生产量级；
- 生产链路（bge + Chroma）：检索 ~11ms → 命中 ~1ms（回查 + key 计算），
  同样是 ~90% 下降，绝对值小但**在高 QPS 下是 CPU 容量差**；
- **token 账（推算口径，非账单）**：检索层不省模型 token（省 embed CPU）；
  答案层命中一次 = 省 1 次模型调用 ≈ 2700 token（6000 字上下文 + 200 输出）。
  命中率 70% 时：每 100 次问答省 70 × 2700 ≈ **189k token**。
  真实账单要联网跑（usage_log 口径），那是"已部署"那周的验收项。

## 8. 单实例边界（写明，与限流同一条纪律）

内存 LRU 只对**单个进程**有效。多实例时命中率稀释但**不出错**
（key 含配置指纹，错的是量不是值 —— 这与限流不同，限流是"量错了
就超额"，缓存是"量少了但值永远对"）。所以 Redis 化的优先级低于限流：
上第二个实例那天先 Redis 化限流，缓存可以等命中率数据说话。

## 9. 已知边界（如实）

- **只缓存非流式路径**：`answer_stream`（UI 流式）不缓存 —— 流式的
  delta 序列缓存语义（半截命中怎么续）是另一个议题，W13 再议；
- **拒答条目走 5min 而非 2min 的例外**：`cached_qa._store_answer` 把
  refused 条目标 negative=True，机制层按 2min 存；但分数闸门拒答发生在
  `QAEngine.answer` 内部，阈值不在 key 里 —— 短 TTL 是**缓解**不是根治，
  根治需要把 min_score 也放进答案 key（会破坏"同参数必命中"），权衡后选缓解；
- **LRU 容量 2000 条是拍脑袋的**：等 `/metrics` 的命中率与淘汰计数有真实
  数据后再调 —— 与 top_k=5 的选法同一逻辑：没有证据不动它。
