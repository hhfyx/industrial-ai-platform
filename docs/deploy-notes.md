# 部署踩坑记录

> 目标：Streamlit Community Cloud（免费层）· 备选：本地 `docker compose`
> 实测数据：`scripts/memory_result.json`（由 `scripts/measure_memory.py` 产出，可复跑）
>
> **状态：2026-11-04 首次真机部署成功。**
> `https://industrial-ai-platform-9mxgvsbziodwgcqxnepcwm.streamlit.app`
> 逐项验证结果见 [§线上验证结果](#线上验证结果2026-11-04-首次真机部署)。

---

## 一句话结论

**免费层跑不了 bge。** 模型栈实测增量 **528 MB**，加上 Streamlit 自身（~100 MB）
与 Chroma（~50 MB），1 GB 上限下**没有余量** —— 并发会话或一次大文档索引就会 OOM。
所以线上默认降级为 `hashing`（只有字面检索），且**降级在 UI 里显式可见**。

**部署成功后暴露的第二个问题**（2026-11-04）：`hashing` 的字面匹配会
让"越没内容的块排得越前"（短标题、目录、图片路径全部挤进 Top-K）。
已修（短块合并 + 目录排除），但**这是清噪音，不是让检索变好** ——
详见 [§检索质量](#检索质量线上事故与修复2026-11-04)。

---

## 坑 1：`tasklist` 的 CSV 里内存字段含逗号（差点把结论做反）

**这个坑最值得记，因为它的错误结果看起来非常合理。**

```powershell
tasklist /FI "PID eq 44520" /FO CSV
# "python.exe","44520","Console","2","5,648 K"
#                                  ↑ 引号里有个逗号
```

第一版这么解析：

```python
parts = [p.strip('"') for p in line.split(",")]
raw = parts[-1]  # 得到 " K"，不是 "5,648 K"
```

结果：所有阶段都量成 ~1 MB，脚本打印"峰值 1 MB，免费层完全够"。

**为什么危险**：小数字看起来是好事。如果我信了它，
今天就会给免费层配上 bge，然后在线上随机 OOM —— 而那时我已经把
"部署完成"写进 README 了。

修法：用 `csv.reader` 解析（它正确处理引号内的逗号）。

> 教训与 `docs/weekly/w3.md` 里那条 `connect_ex` 用反是同一种：
> **验证手段本身也会错，而且错的时候不会报错。**

## 坑 2：`.venv\Scripts\python.exe` 是 launcher，`Popen.pid` 不是被测进程

```python
p = subprocess.Popen([sys.executable, "-c", "..."])
# p.pid        = 44520  ← launcher，一直 5.6 MB
# 子进程自报   = 31272  ← 真正加载了 torch 的那个
```

用 `p.pid` 去 `tasklist`，量到的是 launcher 的空壳 —— 同样得到"1 MB"。

修法：让探针脚本第一行打印 `os.getpid()`，主进程用它去查询。

**这两条合起来说明一件事**：内存测量这条链路有两个独立的错误源
（字段解析 + 进程身份），任何一个出错都会给出"很小、很乐观"的数字。
**所以修完之后我用"两次独立测量"交叉验证**：单独跑一次得到 561 MB，
这次脚本跑得到 528 MB（增量口径），量级吻合。

## 坑 3：`[client] maxUploadSize` 是无效配置，Streamlit 会静默忽略

```toml
[client]
maxUploadSize = 200     # ❌ 只打一行警告，然后用默认值继续跑
```

正确位置是 `[server]`。表现是"我明明调了上限"和"还是传不上去"同时成立。

**这类"配置写错但进程起来了"的问题比崩溃难查** —— 它没有任何错误信号。
`tests/test_deploy_config.py::test_streamlit_config_has_no_invalid_keys` 守着这条。

## 坑 4：requirements 带 3777 个 hash 行，装依赖容易失败

`uv export` 默认输出带 `--hash` 的版本（334 KB）。Streamlit Cloud 按
`requirements.txt` 装依赖时，**任何平台差异（轮子名字、Python 小版本）
都会让 hash 校验失败**，而错误信息是 `THESE PACKAGES DO NOT MATCH THE HASHES` ——
看起来像"包被篡改了"。

修法：`uv export --no-hashes`。可复现性对**部署**不重要（本地 `uv.lock` 才是），
部署要的是"能装上"。

## 坑 5：完整 requirements 含 torch（900 MB），装不完

`uv export` 出的是完整运行时依赖。torch CPU 版约 900 MB，
Streamlit Cloud 的安装时长与磁盘配额都吃紧。

所以部署侧用 `app/ui/requirements.txt`（12 个包，**不含 torch**）。
文件头写了"为什么不装"，因为总有人会"顺手补上缺的包" ——
`test_deploy_requirements_excludes_torch` 挡着。

### ⚠️ 坑 5.1：这份文件**必须放在入口目录**，放根目录会被 `uv.lock` 抢掉（W7 周一修正）

这一条是后来才发现的，**它差点让整个部署计划失败，而且在部署前没有任何症状**。

原来的计划是：文件叫 `requirements-deploy.txt`，部署时
"在 Cloud 的 **Advanced settings** 里指定它"。

**那句话是错的，两处都错：**

| 我以为 | 实际 |
| --- | --- |
| Advanced settings 里能指定 requirements 文件 | **没有这个选项** —— Advanced settings 只有 **Secrets** 和 **Python version** |
| `requirements-deploy.txt` 这个文件名能被识别 | **不能** —— Cloud 只认 `uv.lock` / `Pipfile` / `environment.yml` / `requirements.txt` / `pyproject.toml` |

也就是说，**那份文件从来就不会被读到**。而 Cloud 实际会做什么，官方文档写得很清楚：

> Community Cloud will use the **first** dependency file it finds.
> Community Cloud will search the directory where your **entrypoint file** is,
> **then** it will search the **root** of your repository. In each location,
> dependency files are prioritized in this order:
>
> `uv.lock` > `Pipfile` > `environment.yml` > `requirements.txt` > `pyproject.toml`

本仓库根的 `uv.lock` 有 **1.1 MB**，而它在根目录里的优先级**高于** `requirements.txt`。所以：

| 放哪里 | Cloud 实际读谁 | 结果 |
| --- | --- | --- |
| 只有根目录的 `requirements.txt` | **`uv.lock`**（优先级更高） | 装**完整**依赖 → 含 torch 900 MB → **超时 / OOM** |
| **`app/ui/requirements.txt`**（入口目录） | **它自己**（入口目录先被搜） | 只装 12 个包 → 能跑起来 |

**结论：根目录没法修。** 除非删掉 `uv.lock`（但它是本地可复现性的依据），
否则根目录永远命中 `uv.lock`。唯一可用的机制就是"入口文件所在目录"这一层。

修法：把文件从 `requirements-deploy.txt`（根）挪到 **`app/ui/requirements.txt`**，
即**入口文件 `app/ui/streamlit_app.py` 的同目录**，并改成 Cloud 认识的名字。

> **这个坑为什么最值得记**：错误不是崩溃、不是报错，而是
> "计划装 12 个包，实际装了 900 MB 的 torch"。
> **在真正点下部署按钮之前，它不会产生任何信号** ——
> 文档写着 A，Cloud 做的是 B，而 A 看起来完全合理。
>
> `tests/test_deploy_config.py::test_deploy_manifest_is_where_cloud_actually_looks`
> 现在守着三件事：名字必须是 `requirements.txt`、位置必须在入口目录、
> 旧文件名不得复活。

---

## 降级设计：为什么是 hashing，不是 ONNX

任务给的三个选项：ONNX 量化版 / LRU 缓存 / 退化到 API embedding。

| 方案 | 内存 | 质量 | 为什么选/不选 |
| --- | --- | --- | --- |
| **hashing（选它）** | ~1 MB | 只有字面相似 | **零依赖、零下载、确定能跑** |
| ONNX 量化版 | 未测（估计 100~200 MB） | 接近 bge | 需要 `onnxruntime` + 模型转换 + 平台兼容性验证。**在没验证前它只是"可能更小"** |
| LRU 缓存 | 不解决问题 | — | 缓存减少的是**计算**，不是**常驻内存**。模型还是要占 528 MB |
| API embedding | 小 | 依赖网络 | 索引阶段要把**全文**发出去；且免费层出网、限流、凭据都是新问题 |

**选 hashing 的理由是"确定的代价 vs 不确定的收益"**：
ONNX 量化版可能省下 300 MB，但需要我先验证它能装、能跑、精度下降多少 ——
那是半天到一天的工作，而它不改变"这个 demo 想演示什么"（能跑通 + 能讲清 + 能指出问题）。

> ⚠️ **这不是"ONNX 不好"**，是"今天没时间验证它"。
> 这条写进 `README` 的"已知问题"清单，作为明确的待办。

### 降级必须可见

侧栏显示：
```
⚠️ 当前运行在低内存托管环境，向量化已降级为 hashing
（只有字面相似，没有语义检索）。用 bge 需 ~561MB，超过免费层 1GB 上限。
```

并显示当前 `recommended_min_score`（hashing 是 `0.0`，等于关掉拒答闸门）。

**悄悄降级等于骗用户**：他以为在做语义检索，实际只有字面匹配。
`tests/test_deploy_config.py::test_describe_runtime_gives_a_visible_notice`
断言这句提示存在且含内存数字。

---

## 部署步骤（Streamlit Community Cloud）

> **分两半**：① 仓库侧（已就绪，见下）② 网页侧（需要你的 GitHub 账号，
> 我无法代办 —— 这是唯一需要你动手的部分）。

### ① 仓库侧（已完成，不用你动）

| 项 | 状态 |
| --- | --- |
| 仓库是 public | ✅ `https://github.com/hhfyx/industrial-ai-platform` |
| 入口文件 | ✅ `app/ui/streamlit_app.py` |
| 部署依赖 | ✅ **`app/ui/requirements.txt`**（入口目录，Cloud 会先搜这里） |
| Streamlit 配置 | ✅ `.streamlit/config.toml`（`maxUploadSize=200` 在 `[server]` 段） |
| 降级逻辑 | ✅ 检测到托管环境自动走 `hashing`，且 UI 里**可见** |

### ② 网页侧（需要你做，约 5 分钟）

```text
1. 打开 https://share.streamlit.io   （可能需要科学上网）
2. 用 GitHub 账号登录并授权 Streamlit
3. 点 "New app" → "Deploy a public app from GitHub"
4. 填三个字段：
     Repository : hhfyx/industrial-ai-platform
     Branch     : main
     Main file path : app/ui/streamlit_app.py      ← 必须手填这一整行
5. 点 "Advanced settings"（可选，见下方 Secrets）
6. 点 "Deploy!"
```

> **为什么 Main file path 要手填**：Cloud 默认找仓库根的 `app.py` / `streamlit_app.py`，
> 而本仓库根有 `app/` **包** —— 在根放一个 `app.py` 会遮蔽它，import 直接坏掉。

### ②·5 拿到 key 后，**先跑自检再部署**（省一次 3 分钟的等待）

```bash
python -X utf8 scripts/check_llm_endpoint.py \
    --base-url https://api.deepseek.com/v1 \
    --api-key sk-xxxx \
    --model deepseek-chat
```

或者用 make（读 `.env` 里的配置，适合本地验证）：

```bash
make check-llm
```

它检查五件事，**约 5 秒、近乎免费**（只发 3 次最小调用）：

| # | 检查 | 失败的典型原因 |
| --- | --- | --- |
| 1 | 连通性 / 鉴权 | key 错、base_url 少了 `/v1`、国内需代理 |
| 2 | 模型名在可用列表里 | 名字写错（会**列出相似的名字**给你选） |
| 3 | 能否出字 | `max_tokens` 太小被推理吃光（返回 200 但 content 为空） |
| 4 | 能否带材料作答 | 模型不读上下文 —— **这是最接近线上的检查** |
| 5 | 代理判断一致 | 脚本与应用的 `trust_env` 判据相同，避免"脚本能通、应用不通" |

**为什么要先跑这个**：`make answer-quality` 要跑 20 条真实问答（几元钱、几分钟）。
先花 5 秒确认 key 能用，再花钱测质量 —— 否则可能花了钱才发现 key 是错的。

> 脚本打印 key 时只显示后 4 位（`********E5Ro`），不会把 key 泄露到日志里。

### ③ Secrets（**如果不填，提问会 502**）

在 Advanced settings 的 **Secrets** 框里粘贴（**根级，不要放进 `[section]`**）：

```toml
LLM_API_KEY = "sk-..."
LLM_BASE_URL = "https://..."
LLM_MODEL = "..."

# 可选：强制 embedder
# RAG_EMBEDDER = "hashing"     # 默认就是它；想试 bge 才改
```

> **为什么是根级而不是 `[section]`**：Streamlit 官方文档明确 ——
> 根级 secrets "will be available as **environment variable**"
> （`os.environ["db_username"] == st.secrets["db_username"]`），
> **而 section 里的不会**。本项目用 pydantic-settings 读 `LLM_*` 环境变量，
> 所以只要放在根级，**不需要改任何代码**。

> ### ⚠️ 这里有一个我无法代办的真实阻塞
>
> 本地跑的 `LLM_BASE_URL=http://127.0.0.1:8326/v1` 是**你机器上的网关**。
> 部署到 Cloud 之后，`127.0.0.1` 指的是 **Cloud 容器自己** —— 那里什么都没有。
>
> **所以线上要能用，必须提供一个公网可达的 LLM 端点 + key。**
> 没有 key 的后果不是"app 起不来"，而是**上传、检索、引用都正常，只有生成答案报 502**
> —— 这种"看起来能跑但一问就错"的状态比彻底失败更难解释。
>
> 三个选项见下方 §线上可用的三条路。

---

## 线上可用的三条路（**这一步必须你决定**）

线上 demo 要能"问答"，就需要一个**公网可达**的 LLM 端点。三个选项，代价与效果都不同：

| 选项 | 做法 | 代价 | 线上效果 |
| --- | --- | --- | --- |
| **A. 官方 API + key**（推荐） | 注册 DeepSeek / 智谱 等官方 API，充值几元，把 key 填进 Secrets | **几元钱**（本项目 20 条约 ¥0.5~2） | **完整可用**：流式、引用、拒答全都能演示 |
| B. 只演示检索层 | 不填 LLM key，UI 明确标注"生成不可用" | 0 | 上传 + 检索 + 引用可用；**一问就 502** —— 需要有清晰提示否则很糟 |
| C. 自建公网网关 | 把本机网关暴露到公网 | 有**安全风险**（key 泄露），且需要固定 IP / 内网穿透 | 完整可用，但**不建议** |

**我的建议是 A**：`deepseek-account/deepseek-flash`（本机网关已验证可用）
对应的官方 API 充值几元就够演示很久。

> **为什么不建议 C**：本机网关的 `LLM_API_KEY` 是 DSH 的会话凭据。
> 把它连同内网穿透一起暴露到公网，等于把一个能白嫖你额度的入口开放给所有人。
> **省钱不值得冒这个。**

### 不管选哪个，这两件事都要做

1. **确认仓库 public** —— 免费层只能读 public 仓库
2. **首屏要写清当前能力边界** —— 降级提示必须可见（`deploy_config.describe_runtime()` 已实现）

---

## 我本机那个"502"已经修了（W6 周日）

> ⚠️ 本节以下的旧记录里写"本机网关 chat 端点返回 502 未配置凭据"，
> **那个问题已经在 W6 周日解决**，且根因与凭据无关：
>
> | # | 真实原因 | 修法 |
> | --- | --- | --- |
> | 1 | **本地回环请求走了系统代理** —— httpx（openai SDK 底层）默认 `trust_env=True`，把打 `127.0.0.1:8326` 的请求扔进 Clash 代理 | `_build_client` 显式传 `trust_env=False` |
> | 2 | **模型名写错** —— `.env` 里写 `buddy/glm-5.3`，正确是 `buddy-pool/glm-5.3`（少 `-pool`） | 改成 `buddy-pool/glm-5.3` |
>
> **两个坑返回的是同一个误导性错误**（`502 未配置凭据，请先登录`），
> 这正是一周来一直往"账号问题"方向查的原因。
>
> 修完后本机端到端已跑通：20 条问答，**严格正确率 75%**、幻觉引用 0 个。
> 见 [answer-quality.md](answer-quality.md)。

---

## 备选：本地 docker compose

W1 的模板可以直接复用（`Dockerfile` + `docker-compose.yml`），但**有两处必须改**：

| 位置 | 改什么 | 为什么 |
| --- | --- | --- |
| `Dockerfile` | 加 `streamlit` 与 RAG 依赖 | 现在的镜像只装 API 依赖 |
| `docker-compose.yml` | 加一个 `ui` service，端口 `8501:8501` | API 与 UI 是两个进程 |
| `.dockerignore` | 确保 `data/`、`.cache/` 不进镜像 | 向量库与模型权重是可重建的派生物，进镜像会让体积翻倍 |

**这条我今天没做**（时间分配给了 README 与决策文档）。记在这里作为待办，
并写进 README 的"已知问题"。

---

## 线上验证清单（2026-11-04 已逐条跑过）

| # | 步骤 | 期望 | 实测 |
| --- | --- | --- | --- |
| 1 | 打开首页 | 看到"📚 知识库问答"与侧栏降级提示 | ✅ |
| 2 | 上传文档 | 进度条走完，显示"N 页 → M 块" | ✅ 21.9 KB Markdown → 60 块 |
| 3 | 提问 | 逐字出字（不是一次性出现） | ✅ 首字 675~1021ms |
| 4 | 看耗时徽标 | 检索 ~4ms / 首字 Xms / 总计 | ✅ |
| 5 | 展开「参考来源」 | 看到原文 + **依据句高亮** | ✅ |
| 6 | 问一个知识库里没有的问题 | 如实说没有，不是编答案 | ✅ 「材料中没有提到」 |

> **清单已跑完，但有一点必须理解清楚**：
> 线上走 `hashing`（无 torch），检索质量**低于**本机用 bge 测出的
> 75% 正确率（见 [answer-quality.md](answer-quality.md)）。
> 那 75% 是 bge 的成绩，**不能直接标到线上**。
>
> 线上真实能力的边界见上文 §检索质量 → "边界：这只清了噪音，没让检索变好"。

### 一条给演示用的建议

线上演示时**问用词与原文一致的问题**：

| 好问题（能答对） | 坏问题（会答不出） |
| --- | --- |
| 「缺陷类型有哪些」 | 「模型有什么不足」 |
| 「数据量从多少增加到多少」 | 「这个方案有什么局限」 |
| 「用了哪个 Text Encoder」 | 「基座模型是什么」 |

右列的"坏"不是真的坏 —— 是 `hashing` 换不了说法。
演示时避开，比被问倒好。

> ⚠️ **但"在 Cloud 上跑"现已验证**（2026-11-04），差别只剩检索质量一项，
> 不再是"未验证"状态。

---

## 线上验证结果（2026-11-04 首次真机部署）

> **这一节是实测记录，不是预期。** 之前整份文档都是"按官方文档推导"，
> 现在是"真的部署过了"。

### 部署形态

| 项 | 值 |
| --- | --- |
| 平台 | Streamlit Community Cloud（免费层） |
| 应用地址 | `https://industrial-ai-platform-9mxgvsbziodwgcqxnepcwm.streamlit.app` |
| 入口 | `app/ui/streamlit_app.py` |
| 依赖文件 | **`app/ui/requirements.txt`（确认被读到）** |
| embedder | `hashing`（降级如设计） |
| LLM | DeepSeek 官方 API（`deepseek-chat`） |

### 逐项结论

| # | 项目 | 结果 | 证据 |
| --- | --- | --- | --- |
| 1 | 仓库克隆 | ✅ | 构建日志 `Cloning into '/mount/src/industrial-ai-platform'...` |
| 2 | **依赖文件被读到** | ✅ | 日志 `Using uv pip install` + `Resolved 113 packages`，**下载清单里没有 torch** —— 证明命中的是精简清单而非根目录 `uv.lock` |
| 3 | 进程启动 | ✅ | 页面渲染出「📚 知识库问答」 |
| 4 | embedder 降级 | ✅ | 侧栏显示「向量化已降级为 hashing」 |
| 5 | 文档索引 | ✅ | 21.9 KB 的 Markdown → 60 块 |
| 6 | 检索命中正文 | ✅ | 引用指向正文段，不只是标题 |
| 7 | LLM 调用 | ✅ | 首字 675~1021ms，流式输出 |
| 8 | 引用有效性 | ✅ | `[1]`~`[10]` 全部落在 top_k 范围内 |
| 9 | 拒答 | ✅ | 问「缺陷的定义标准」→ 如实答「材料中没有提到」 |

### 线上答案质量（真实样例）

输入：`D:\git\实践\test\工业缺陷生成模型全链路报告.md`（21906 字节）
问题：`缺陷`

输出（节选）：

> 原始数据来源于工业产线拍摄的大图（约 3000×3000），包含各类产品缺陷；
> 通过 YOLO 标注框 → 前景裁剪的处理流程，提取出 512×512 的缺陷切片 [1]。
> 缺陷类型共 6 种，合计 3323 张 [2]。
> 各缺陷类型分布：type 0 为 219 张（6.6%），type 1 为 329 张（9.9%）…
> 优化后数据量由 1806 张增至 3323 张（+84%），缺陷类型由 4 种增至 6 种 [7]。
> **关于"缺陷"的具体定义或判定标准，材料中没有提到。**

**这段值得单独说**：精确数字 + 正确引用 + **主动声明边界**（定义标准没有）。
"该答的答了、不该编的没编"是 RAG 最理想的表现形态。

### 部署过程中真实修掉的 4 个问题

按发现顺序，每个都在本地**无法复现**：

| # | 问题 | 为什么本地测不出来 | 修法 |
| --- | --- | --- | --- |
| 1 | `requirements-deploy.txt` 名字和位置都不被 Cloud 认 | 本地不经过 Cloud 的依赖解析 | 挪到 `app/ui/requirements.txt` |
| 2 | `.streamlit/secrets.toml` 没被 gitignore | 本地没有泄露风险 | 加 ignore + 测试守护 |
| 3 | `client.py` 顶层 `from fastapi import Depends` | **本地 `.venv` 装了 fastapi** | 隔离到 `app/core/llm/deps.py` |
| 4 | 短块 + 目录污染检索 | **本地测试文档没有目录和标签行** | 短块合并 + 目录排除（见 §检索质量） |

> 第 3、4 条有个共同点：**本地全绿证明不了线上能用**。
> 一个因为依赖集合不同，一个因为测试文档和真实文档不同。
> 现在第 3 条有 `tests/test_deploy_runtime_boundary.py` 守着
> （子进程屏蔽 fastapi 后导入整条 UI 链），本地就能抓到。

### 仍未验证（诚实清单）

| 项 | 说明 |
| --- | --- |
| 并发会话下的内存 | `st.cache_resource` 是进程级单例，多会话共享；真实峰值需线上观测 |
| Chroma 磁盘配额 | 免费层磁盘有限，`data/chroma` 长期增长会撞上限 |
| 冷启动耗时 | 免费层闲置后会休眠，唤醒需重新加载 |
| 大文档（>100 页 PDF） | 只测了 21.9 KB 的 Markdown |

---

## 检索质量（线上事故与修复，2026-11-04）

> 这一节记录**部署成功之后**才暴露的问题 ——
> 应用能跑 ≠ 能答对。**这是两类完全不同的事。**

### 现象

问「缺陷有哪些」，系统拒答，理由是：

> 上下文仅包含报告标题、目录、数据集来源和预处理流程标题等信息

**它没说错** —— 检索回来的 5 块里，2 块是纯标题、1 块是图片路径。

### 根因：字面检索下"越没内容的块排得越前"

`hashing` / BM25 算**字符重叠比例**。9 字符的块里 4 个实词占绝大比重，
500 字符的正文被大量无关字稀释 —— **短块天然赢**。

实测（同一份文档）：

```
问题「训练损失怎么变化」
  #1  **训练命令**：        (9 字符)
  #2  **预期训练进度**：     (11 字符)
  #4  ### 5.3 训练损失曲线   ← 真正的答案排第 4
```

不是排序略有偏差，是**系统性倒挂**：越没内容的块排得越前。

目录块是同一问题的极端情况 —— 它是全文**最泛**的一块，
每个主题词都在里面出现过，所以不管问什么都排第一：

```
问题「用了什么基座模型」
  #1  目录块 (0.259)
  #5  #### 3.2.3 Text Encoder (CLIP ViT-L/14)  ← 真正的答案
```

### 修复（`app/rag/loaders/markdown.py`）

| 修复 | 判据 | 常量 |
| --- | --- | --- |
| 短块合并 | 文本块 < 40 字符就并入相邻块 | `MIN_CHUNK_CHARS = 40` |
| 目录排除 | 锚点链接占非空行比例 ≥ 0.5 | `TOC_ANCHOR_RATIO = 0.5` |
| 图片行丢弃 | 整行只有 `![alt](path)` | — |

两个常量都是**实测标定**的，不是拍的：
40 覆盖了最长的标签行（28 字符）又不误并正常段落（中位数 150+）；
0.5 下目录块实测 0.89，而正文里偶尔一个交叉引用是 0.10。

### 修复效果（同一份文档、同一批问题）

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 块数 | 105 | **60** |
| 短块（<40 字符） | **17 个** | **0 个** |
| 最短块 | **3 字符** | **40 字符** |
| 「训练损失怎么变化」正确答案 | 第 4 名 | **第 2 名** |
| 「缺陷有哪些」缺陷类型表 | 第 3 名 | **第 2 名** |
| 「用了什么基座模型」第一名 | **目录块** | 正文块 |

### 边界：这只清了噪音，没让检索变好

**真正的答案仍未稳定进前三。** 问「用了什么基座模型」，
`Text Encoder（CLIP ViT-L/14）` 还是可能不在 Top-3。

因为 `hashing` 按字符重叠排序，**不理解**「基座模型」和
「Text Encoder (CLIP ViT-L/14)」是同一件事。
要真正解决必须换语义 embedding（付费层，或 API embedding 按量计费）。

**所以线上 demo 的诚实说法是**：

> 在"问题用词与原文一致"时能答对；换说法就可能答不出。
> 这是免费层字面检索的能力边界，不是缺陷隐藏。

演示时问用词一致的问题（如「缺陷类型有哪些」而非「模型的结构缺陷指什么」），
效果稳定。这条边界写在这里，比演示时被问倒好。

---

## 未做 / 未验证（诚实清单）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| ~~真实线上链接~~ | ✅ **已创建（2026-11-04）** | `https://industrial-ai-platform-9mxgvsbziodwgcqxnepcwm.streamlit.app` |
| ~~线上全链路跑通~~ | ✅ **已跑通（2026-11-04）** | 见上文 §线上验证结果：索引、检索、流式、引用、拒答全部实测 |
| ~~线上依赖文件是否真的被读到~~ | ✅ **已确认** | 构建日志 `Resolved 113 packages` 且**下载清单无 torch** |
| 并发会话下的内存 | ❌ 未测 | `st.cache_resource` 是进程级单例，多会话共享模型；真实峰值需线上观测 |
| Chroma 在免费层的磁盘配额 | ❌ 未测 | 免费层磁盘有限，`data/chroma` 长期增长会撞上限 |
| 冷启动耗时 | ❌ 未测 | 免费层闲置休眠，唤醒需重新加载 |
| 大文档（>100 页 PDF） | ❌ 未测 | 只测了 21.9 KB 的 Markdown |
| ONNX 量化版 | ❌ 未做 | 见上文"为什么选 hashing" |
| docker compose 的 UI service | ❌ 未做 | 记在上文"备选"一节 |
| 语义检索（换 embedding） | ❌ 未做 | 免费层能力边界，见 §检索质量的"边界"一节 |

---

# W8 周六：项目二容器化 + 部署（2026-12-12）

## 权重策略三选一：**两条线各自定案**（这不是一道题，是两道）

任务书给的三选一（镜像内打包 / 启动时下载 / ONNX 量化）只对**容器部署**成立。
但本项目有**两条部署线**，它们的约束完全不同，答案是两个：

| 部署线 | 运行环境 | 选择 | 代价（写明，不藏） |
| --- | --- | --- | --- |
| **A：Streamlit Cloud**（已上线） | 1GB 内存、无 GPU、无持久盘 | **不打包、不下载、不 ONNX —— 用 hashing 字面检索** | 检索只有字面相似，换说法答不出（见 §检索质量边界）。**换来的是：零内存压力、零冷启动下载、确定能跑** |
| **B：容器部署**（本周新增） | 自控内存（mem_limit 6g）、可挂 volume | **bge-small-zh 原生 FP16，volume 挂载（=任务书的选项②）** | 镜像小（92MB 权重在盘上不在镜像里）、**冷启动含 16s 模型加载**（healthcheck start-period 已覆盖）、依赖挂载路径存在 |

**为什么不选镜像内打包（选项①）**：bge 权重 92MB 看似可以烤进镜像，
但换来的是"换 embedding 模型 = 重 build + 重 push 整个镜像"——
而 embedding 是这个项目**最常换**的组件（W5 调过一轮）。
92MB 的下载只发生一次（volume 持久化），打包省的只是那一次。

**为什么不选 ONNX（选项③）**：内存最小是真的（bge FP16 528MB → ONNX int8 估 ~150MB），
但精度下降幅度**没有实测数字**，而 W7 的全部评测数字（Recall@5 40.0%）都是
bge FP16 跑出来的——换 ONNX 意味着评测口径作废重测。在"内存够用"的容器里，
用精度损失换不急需的内存是负优化。**它留在待办里，触发条件写清楚：
容器内存压到 <1GB 或要部署到边缘设备时再做，且必须重跑 W7 评测。**

### 必答问题：部署版和评测版差在哪（W8 周六正式回答）

| 维度 | 评测版（W7 数字所属） | 部署版 A（Cloud） | 部署版 B（容器） |
| --- | --- | --- | --- |
| embedder | bge-small-zh FP16 | **hashing**（字面） | **bge-small-zh FP16（同评测版）** |
| Recall@5 口径 | **40.0% 只适用于这一列** | 无此指标（字面检索，未跑过该评测） | **40.0% 可外推**（同模型同参数） |
| 切分参数 | chunk=500/overlap=50 | 同 | 同 |
| top_k / 阈值 | 5 / 0.575 | 5 / hashing=0.0（闸门关闭） | 同评测版 |

**结论（也是周口径说明的原文）**：

> W7 的 Recall@5 40.0% **只描述容器部署版（B）**。
> Cloud 版（A）为内存约束降级为 hashing，**不能引用该数字**——
> 它的能力边界单独描述（"问题用词与原文一致时能答对"）。
> 两条线的降级都**在 UI 可见**（`deploy_config.describe_runtime()`），
> 不存在"看起来是语义检索实际是字面检索"的混淆。

这条口径 B7 那张"部署版和评测版差在哪"的问法，面试时就是拿这张表回答。

## compose 编排升级（api + chroma 双服务）

**为什么 chroma 从"容器内进程"拆成独立 service**：
W4 之前 chroma 跑在 api 进程内（`data/chroma` 本地目录），单容器没问题。
拆出来的三个理由：
1. **数据生命周期分离**——向量库要跨容器重建存活（volume 持久化），
   api 容器随时可以重建而索引不丢；
2. **内存账本清晰**——chroma 常驻 ~50-80MB，混在 api 里会让 mem_limit 定值
   变成猜测；
3. **healthcheck 才有意义**——单容器 healthcheck 只能探"进程活着"，
   双服务才能表达"api 依赖 chroma 就绪"（`depends_on: service_healthy`），
   这正是 W1 的 F4 题（"容器健康≠依赖就绪"）的落地。

升级见 [`docker-compose.yml`](../docker-compose.yml)（api + chroma 双服务 +
两个 volume + depends_on 条件就绪）。验证结果见下文 §容器全链路验证。

## 容器全链路验证（2026-12-12，方案 B 实测）

> **环境**：Docker Desktop（WSL2 后端）· 镜像 `industrial-ai:v0.2` ·
> chroma 内嵌 + 具名 volume · LLM 走宿主机网关（`host.docker.internal:8326`，glm-5.3）
> 语料：工业缺陷生成模型全链路报告.md（21.9 KB → 60 段 → 68 块）

### 五个耗时（全链路，bge FP16 + 真实 LLM）

| # | 环节 | 耗时 | 口径 |
| --- | --- | --- | --- |
| 1 | 索引（含 chroma 首次建库） | **2.2 s** | 60 段 → 68 块 + 向量化 + 入库 |
| 2 | 单问全链路（检索+生成） | **3,985 ms** | 检索 ~10ms，其余是 LLM 生成 |
| 3 | 多轮改写调用 | **7,552 ms** | structured_call（含网关排队；改写把「它的分布」扩成完整问题） |
| 4 | 二轮问答（改写后检索） | **4,734 ms** | 命中分布表，引用 [4][5] |
| 5 | 10 并发 wall | **6.7 s** | 0 错误 0 拒答，见下方压测表 |

### 10 并发压测（16:30 格）

| 指标 | 数值 |
| --- | --- |
| 错误率 | **0/10** |
| 拒答 | 0 |
| P50 | **4,282 ms** |
| P95 | **6,594 ms** |
| min / max | 2,023 / 6,614 ms |
| wall（10 条全部完成） | 6.7 s |

**瓶颈段判定：上游 LLM 网关，不是本地**。证据：10 条并发 wall（6.7s）
≈ 最慢单条（6.6s）——请求几乎同时完成，说明它们在上游排队；
本地检索 ~10ms/条、CPU 占用极低。**结论：并发瓶颈在网关的并发度，
扩容方向是网关加并发配额或换更大网关，不是给 api 容器加副本。**

### 本次部署修掉的 4 个新坑（都是容器特有，本地测不出）

| # | 现象 | 报错原文 | 根因 | 修法 |
| --- | --- | --- | --- | --- |
| 1 | 容器反复重启（Restarting (1)） | `ModuleNotFoundError: No module named 'openai'` | GPU 镜像只装了"生成栈"，但 `app.main` 的 import 链会拉 chat/stream 路由 → openai | `gpu-requirements.txt` 补齐 import 链（openai/jinja2/python-multipart）——**镜像能起 = 全链可导入** |
| 2 | 起动即崩 `Restarting (1)` | `ValidationError for LLMSettings: api_key Field required` | `app.main` 导入链读 Settings，api_key 必填；GPU 容器不做问答但进程要起 | compose 里 `LLM_API_KEY` 占位值（不产生真实调用） |
| 3 | 索引时 chroma 崩 | `chromadb.errors.InternalError: (code: 14) unable to open database file` | 具名 volume 首次挂载继承**挂载点属主**；镜像里 `data/` 是 root 的，uid 10001 进程写 sqlite 被拒 | Dockerfile 里 `mkdir -p data/{chroma,sessions,uploads} && chown 10001` —— **volume 属主问题必须在镜像里解**，不是 compose 里 |
| 4 | 容器内问答挂起 | 连接超时（无报错，表现为"模型慢"） | `.env` 的 `LLM_BASE_URL=127.0.0.1:8326` 在容器里指向容器自己 | compose `environment` 覆盖为 `http://host.docker.internal:8326/v1` |

### 双向校验在线上真实工作了（附注）

压测日志里出现一条：`答案引用了不存在的编号 [64]（可用：[1..5]）→ 重答后仍越界 → 标记 degraded`。
这是 W8 周二的引用完整性层在真实 LLM 输出上的第一次实战拦截——
glm-5.3 在某条压测问题里编造了 [64]，重答没修好，降级标记生效。
**校验层在本地测试全绿之外，第一次证明了它在脏输出面前的价值。**

## W8 改动上云（2026-12-12 第二次部署）

`f2f0716`（周一~周五全周）与 `86f3e93`（周六部署）推上 main 后 Cloud 自动重建。

| 项 | 结果 |
| --- | --- |
| 重建触发 | ✅ push → Cloud 自动构建（无需手动操作） |
| 应用响应 | ✅ 303 → `/-/login`（**平台新增 viewer 登录门**，已登录 GitHub 的浏览器无感；能响应跳转 = 应用本体在运行） |
| W8 新代码 | ✅ 零新增第三方依赖——session_store（sqlite3 标准库）/ query_rewrite（复用 structured_call）/ dedupe（difflib 标准库），`app/ui/requirements.txt` 一行没改 |
| 会话持久化上云 | ✅ sqlite 落容器盘（Cloud 免费层无持久 volume，重启即清——与本地容器行为差异，见下方口径表） |

**Cloud 免费层的会话持久化边界（如实）**：`data/sessions.sqlite` 在 Cloud 是**临时盘**，
应用休眠/重启后会话历史清零。这与本地容器（volume 持久化）行为不同。
线上演示多轮指代时**同一会话内**的改写与上下文是完整的（内存里），跨重启不保证。

---

# W12 周三：双服务编排、Nginx 反代与故障演练

> `docker-compose.services.yml`（双服务）+ `deploy/nginx.conf`（统一入口）。
> 单服务 `docker-compose.yml` 保留：本地开发与 CI 的最小可运行单位。

## 一、为什么这样拆两个服务

拆分依据是**故障域与伸缩维度**，不是"一进程一容器"的形式主义：

| | api-rag（文档智能面） | api-gen（图像生成面） |
| --- | --- | --- |
| 资源画像 | CPU + 大内存（bge/chroma/会话库），mem_limit 4g | GPU 独占 + 单 worker 串行（generation_queue.py 的设计），mem_limit 8g |
| 故障面 | OOM 拖垮问答 —— 单进程内无解 | 生成面 OOM **不拖垮**问答面（隔离的收益） |
| 伸缩 | 可横向扩副本（内存允许时） | 只有一个 GPU，扩副本无意义 |
| 流式 | SSE 三端点（对反代有特殊要求） | 无流式 |

## 二、healthcheck 为什么探 /ready 而不是 /health

`depends_on: condition: service_healthy` 要的是"**能接流量**"（模型已加载），
不是"**进程活着**"（模型加载 30s 期间 liveness 也是 200）。
用 liveness 做 readiness 的活 = 编排器在加载窗口把流量切进未就绪的服务
—— liveness/readiness 分离（W12 周一必答问题）在编排层的直接应用。
`start_period: 45s` 覆盖加载窗口，`interval: 15s × retries: 5` = 最坏 75s 摘流。

## 三、Nginx 配置的四个"为什么"（deploy/nginx.conf）

1. **SSE 不缓冲四件套**：`proxy_buffering off`（攒满再发 = SSE 失效）、
   `proxy_cache off`、`proxy_read_timeout 300s`（长摘要 3 分钟+）、
   `proxy_http_version 1.1`（默认 1.0 会破坏 SSE 长连接）。
   应用侧 `X-Accel-Buffering: no`（W2 的 E1、W8 的 F3）是**声明**，
   nginx 配置是**执行** —— 两边都写，谁生效都一样。
2. **超时与业务预算对齐**：Agent 预算 `max_wall_time=120s`（冻结值）→
   `/api/v2/agent-tasks` 的 `proxy_read_timeout 180s`；生成单任务
   `GENERATION_TIMEOUT_S=180s` → `/api/v1/generate` 用 240s。
   **反代 60s 默认值会掐断一个预算内还有 60s 的任务** —— 超时必须是
   "业务预算 + 余量"，不是拍脑袋数字。
3. **不重试非幂等请求**（必答问题）：`proxy_next_upstream off` 挂在
   生成与 Agent 提交两条路由上。nginx 默认对 error/timeout 重试 ——
   对 `POST /generate` 重试 = 生成两次 = 双倍 GPU 时间，而客户端只看到
   一次请求。与 W8 的 D4（GPU 独占的串行重试代价）、W11 的重试矩阵
   （retryable 由代码判定）是同一原则在反代层的体现。
   `POST /extract`/`chat` 同理（在 `/` 路由里也只允许 GET 的 error 重试，
   POST 的重试由客户端决定 —— 客户端知道哪次带了幂等键）。
4. **路由表**：`/api/v1/generate → gen_backend`，其余 → `rag_backend`。
   v2 的 `/agent-tasks` 属于问答面（LLM 调用，无 GPU）。

## 四、取消语义的两个面（端到端联调实测出的细节）

| 状态 | DELETE agent-tasks | DELETE generate/tasks |
| --- | --- | --- |
| queued | 直接 cancelled | 直接 cancelled |
| running | **协作式**：标记 + 执行协程在工具步边界退出，保留 partial_answer | 协作式：worker 在**每张图之间**检查（CUDA 无安全取消点，硬杀留脏显存），已生成张数如实保留 |
| succeeded/failed | 409 TASK_FINISHED | 409 TASK_FINISHED |
| cancelled（重复删） | **200 幂等回显**（目标状态已达成） | 200 幂等回显 |

第一版把"已取消再取消"也回了 409 —— 那是错的：取消明明生效了，
409 是撒谎。两个面的语义统一为"**取消是否真的达成了目标状态**"。

进度百分比是**真实进度**（已完成张数/总张数，worker 每张回写），
不是时间估算 —— GPU 任务时长大方差，估算是撒谎。

## 五、故障演练三条记录（scripts/fault_drill.py，实测）

| # | 演练 | 观察 | 结论 |
| --- | --- | --- | --- |
| ① | 生成面挂掉，问答面行为 | 问答面 extract HTTP 200，与生成面端口零交互（proxy_pass 按路径分流） | 分流在 nginx 层：生成面挂了只有 `/api/v1/generate` 502，问答面**天然不受影响** |
| ② | kill worker 进程 | gen health=unreachable（连接拒绝）；rag health=200 | 进程死 → 探针连不上 → unhealthy → 摘流/重启；摘流延迟 ≤ 15s×5 = 75s（readiness 间隔与重试次数的直接含义） |
| ③ | 服务重启后任务状态 | 重启前提交的 Agent 任务，重启后 GET → **404** | 内存任务表重启即丢（周一冻结清单写明的约束，现在是实证）。轮询方 404 = 状态不可知，不是失败；落库是"已部署"周任务 |

## 六、遗留（如实）

- nginx `nginx -t` 语法校验**未在本机跑**（Docker 守护进程 W11 起就没起来，
  本机也无 nginx 二进制）—— 做了静态自检（括号配对 / location 覆盖 /
  关键指令计数），首次 `up` 时以容器内校验为准；
- 端到端联调走的是**两个裸 uvicorn**（8100/8101），nginx 形态的等价物是
  路由表（逻辑一致），真实反代链路的联调要等守护进程起来；
- Agent 执行的 `instrumented_executor` 包装只回写工具步快照，
  模型步的 current_node 不更新（0.5s 内轮询能看到滞后一步）——
  可接受，落库时一并解决。
