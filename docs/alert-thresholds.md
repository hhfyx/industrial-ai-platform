# W13 告警阈值表（唯一合法来源：W12 拐点数据）

> 本文件是 W13 可观测面板的阈值**唯一合法来源**。每一条阈值都从
> `performance.md` 第二轮压测（2026-01-09，两轮复测差 <4%）推导，
> 不允许"拍脑袋"值进入本表。改本表的唯一方式：用文末复测命令
> 重新实测，替换基线与阈值，并在修订记录里写明测点。

## 一、阈值推导表（hashing 口径）

基线全部来自 [performance.md §九](performance.md)（两轮对照）与 §十（拐点判定链）。

| 指标 | 实测基线（两轮） | 告警阈值 | 推导依据 |
| --- | --- | --- | --- |
| 检索 P50（服务端 X-Elapsed-Ms） | 1ms 恒定 | **> 3ms 持续 5min** | 基线 1ms，3 倍余量仍远低于客户端感知；恒定值漂移 = 服务层异变 |
| 检索 P95（服务端） | 2ms 恒定 | **> 6ms 持续 5min** | 同上；服务器口径不受客户端/网络噪声污染 |
| 检索 P95（端到端） | 7–10ms（A1–A3） | **> 20ms 持续 5min** | 基线 2 倍。注意：A3 档 50 并发下基线本身 9–10ms，阈值必须高于它 |
| extract P50（端到端） | 2.4–2.7s，**上游网关决定** | **不设阈值** | 延迟是 LLM 网关的（W5：检索占 1%）。对它告警 = 给上游波动买单。观察对象改为服务端 X-Elapsed-Ms |
| 5xx 计数 | **0**（全档 × 两轮） | **> 0 即告** | 拐点判定链证明 5xx 不是本系统的"正常坏法"——任何 5xx 都是异常 |
| 429 占比 | 55.7% @50 并发（配额函数） | **不告 429 本身** | 429 = 限流器按预期挡人（保护行为）。告它 = 每次压测都被叫醒 |
| 浅队列 429 | 0（历史 429 全部深度触顶） | **告** | 背压判据一：浅队列拒绝 = 不是限流配额，是别的东西坏了 |
| 队列排空 | 深度 32 触顶后排空，探针恢复 202 | **触顶后 60s 不排空 → 告** | 背压判据二/三：队列永不排空才是故障的形状（performance.md §五） |
| 缓存命中率 | ~100%（预热稳态） | **< 70% 持续 5min → 告** | 命中率塌了意味着指纹/key 逻辑变了（W12 周二的配置指纹教训） |

**判读约定（继承 W12，不新造）**：429 记 success 不记 failure；
延迟与错误率必须成对报。

## 二、bge 口径：现状与边界

W12 全部压测数字是 **hashing 语料**口径（[performance.md §十三局限 1]）。
bge 形态的阈值**本表暂不给出服务级数字**，给出的是已实测的锚点与边界：

### 2.1 已实测（2026-11-12，本机，模型本地缓存命中）

| 项 | 数值 | 口径 |
| --- | --- | --- |
| 模型加载（首次 embed 含加载） | 28.5s | BAAI/bge-small-zh-v1.5，D:\cache\hub 缓存 |
| 单条 encode P50 / P95（预热后，60 条） | **16.7ms / 18.7ms** | 本机 CPU；W6 记录为 10.5/11.2ms——**同模型同机不同时点差 60%**，证明 bge 阈值必须当日复测，不能抄历史值 |
| 维度 | 512 | 与 embedding/base.py 一致 |

### 2.2 服务级 bge 压测的前置改动（W13 待办，不是本文件能"声明完成"的）

服务端共享检索器目前**硬编码** `HashingEmbedder + InMemoryVectorStore`
（`app/core/llm/real_tools.py::_build_retriever`，L93-135）。要把
`/api/v2/search` 切到 bge+Chroma 口径，需要：

1. `_build_retriever` 读 `RAG_EMBEDDER` 环境变量装配（`deploy_config.resolve_embedder_name` 已有现成函数）；
2. store 换 `ChromaStore`（内嵌 PersistentClient，chroma-data 卷已持久化）;
3. 语料从内置三份假规范换成真实 docs 库并重建索引（bge 下索引时间 = 1.4ms/条 × 全库）;
4. 复跑 `python -X utf8 scripts/load_test.py --profile all`，把 §一表里
   带 "hashing 口径" 标注的行整体替换。

**在 4 完成之前**，§一表只对 hashing 形态生效；bge 部署形态沿用
"基线 2 倍" 的推导规则，但基线值必须来自 4 的实测，禁止拿 2.1 的
16.7ms 直接外推服务级 P95（服务级还要加 store 回查、序列化、并发争用）。

## 三、口径声明（照抄进面板配置）

- 本表阈值仅对 **hashing embedder + 内存库** 形态生效；
- bge+Chroma 形态上线前必须完成 §2.2 的四步并重测；
- 单机 locust 环回注入（无网络变量）；公网形态 P95 会更高；
- 全部延迟数字为热缓存口径；冷态对照见 performance.md §十一；
- extract（完整链路）阈值挂在服务端 elapsed 上，不挂端到端。

## 四、复测命令（改阈值的唯一入口）

```powershell
# 压测配置（压测专用放大限流：全局 200 / key 30）
$env:PLATFORM_API_KEYS='bench-key-0000000000,demo-key-12345678'
$env:PLATFORM_GLOBAL_QPS='200'; $env:PLATFORM_KEY_QPS='30'
uvicorn app.main:app --port 8100

python -X utf8 scripts/load_test.py --profile all   # 五档
python -X utf8 scripts/queue_observe.py             # 队列观察（另起 8101）

# bge 单条锚点复测（不改服务装配也能跑）
$env:HF_ENDPOINT='https://hf-mirror.com'
python -X utf8 -c "..." # 见 scripts 内 embed 基准；60 条预热后取 P50/P95
```

## 五、部署验收记录（2026-11-12，W13 开工日实测）

W12 留的三条部署尾巴，今日全部补上，修复过程暴露两个真 bug：

| 项 | 结果 | 修了什么 |
| --- | --- | --- |
| Docker 守护进程 | ✅ Docker Desktop 29.9.2 启动，25s 就绪 | W12 遗留"守护进程未起" |
| 双服务 compose | ✅ w12-api-rag / w12-api-gen / w12-nginx 全部 healthy | **修复①**：services 编排缺 `env_file: .env`，`LLM_API_KEY` 从未注入，容器启动即 ValidationError（crash-loop）。两个服务各补一行 |
| 单服务 compose | ✅ industrial-ai-api healthy（8000） | 清理了占用容器名的孤儿实例 |
| `nginx -t` | ✅ syntax ok / test successful | W12 遗留"未跑"（alpine 容器内执行） |
| `/api/v2/ready` 外部可达 | ✅ 经 nginx :8080 返回 `{"status":"ready","checks":{...}}` | **修复②**：Dockerfile 漏建 `data/generated` 目录，具名 volume 以 root 属主初始化，uid 10001 写入 PermissionError → 500。补 mkdir + chown（与 W8 sqlite 坑同根：volume 属主） |

运维注意（今天踩到的，写进面板 runbook）：
- **容器重建后必须 `docker restart w12-nginx`**：nginx 启动时解析 upstream
  域名并缓存 IP，api 容器重建换了 IP 后它继续打旧地址 → 502；
- readiness 语义是"依赖单例已构造才 200"（cold.py：探针不触发加载），
  新起的栈要先各打一次 generate + extract 真实请求，ready 才会翻 200——
  面板的 readiness 告警要配一条"启动后 5min 仍 503"而不是"启动即 503"。

## 六、修订记录

| 日期 | 变更 | 依据 |
| --- | --- | --- |
| 2026-11-12 | 初版：hashing 阈值表 + bge 锚点 + 部署验收记录 | performance.md 第二轮数据（01-09）；bge 微基准（当日）；本机部署实测 |
