# 分层抽样 20 条（周四全链路调管线用）

> 种子 20261202，重跑必得同批。来源 questions-120.jsonl（120 条）。

| 层 | 配额 | 实际 |
| --- | ---: | ---: |
| fact 大文档 | 4 | 4 |
| fact 小文档 | 4 | 4 |
| multi_hop | 6 | 6 |
| refusal（含 3 条最高分近邻） | 6 | 6 |

事实题覆盖文档：chunking-boundaries.md、chunking-predictions.md、deploy-notes.md、prompt-ab.md、prompt-templates.md、structured-output.md、system-prompt.md、why-i-use-langchain.md

## 拒答特抽（阈值边缘，最先暴露误放）

- r-01: pgvector 的 HNSW 索引建图参数 efConstruction 一般怎么取？
- r-03: LlamaIndex 的句子窗口检索怎么配置？
- r-06: PostgreSQL 里 pg_trgm 和全文检索怎么选？
- r-16: Redis 的 LFU 淘汰策略参数怎么调？
- r-18: DVC 的数据集版本管理怎么和 git 配合？
- r-20: FAISS 的 IVF-PQ 索引两阶段量化原理是什么？

## 完整清单

- f-01（fact）四类模板 A/B 里最容易混的是哪两个意图类别，混了几次？
- f-03（fact）抽取组 v2 大幅胜出，真实原因是什么？
- f-22（fact）json_mode 路线失败时，模型的输出实际长什么样？
- f-31（fact）给模板加新版本的正确流程是什么？
- f-35（fact）这组 prompt A/B 实验里，两版之间唯一动过的东西是什么？
- f-47（fact）切分器对代码块做常规 strip 会破坏什么？
- f-74（fact）按文档结构标记来切分的策略，预判会在哪种文档上吃亏？
- f-76（fact）三种切分策略的预判排序是怎样的？
- m-02（multi_hop）部署依赖文件的两个坑合起来看，最终方案落在哪里？
- m-10（multi_hop）问答组 v2 掉分的机制和最典型的那条各是什么？
- m-11（multi_hop）分类混淆的机制和用户输入的特征是什么？
- m-16（multi_hop）w4 清单切分不合并的根因与修法各是什么？
- m-18（multi_hop）_freeze 崩 ctx.id 的根因和修法各是什么？
- m-19（multi_hop）探针实验里模型自创字段与枚举缺失的关系是什么？
- r-01（refusal）pgvector 的 HNSW 索引建图参数 efConstruction 一般怎么取？
- r-03（refusal）LlamaIndex 的句子窗口检索怎么配置？
- r-06（refusal）PostgreSQL 里 pg_trgm 和全文检索怎么选？
- r-16（refusal）Redis 的 LFU 淘汰策略参数怎么调？
- r-18（refusal）DVC 的数据集版本管理怎么和 git 配合？
- r-20（refusal）FAISS 的 IVF-PQ 索引两阶段量化原理是什么？
