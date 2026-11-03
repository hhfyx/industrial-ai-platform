# 部署踩坑记录

> 目标：Streamlit Community Cloud（免费层）· 备选：本地 `docker compose`
> 实测数据：`scripts/memory_result.json`（由 `scripts/measure_memory.py` 产出，可复跑）

---

## 一句话结论

**免费层跑不了 bge。** 模型栈实测增量 **528 MB**，加上 Streamlit 自身（~100 MB）
与 Chroma（~50 MB），1 GB 上限下**没有余量** —— 并发会话或一次大文档索引就会 OOM。
所以线上默认降级为 `hashing`（只有字面检索），且**降级在 UI 里显式可见**。

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

## 线上验证清单（有链接后逐条跑）

| # | 步骤 | 期望 |
| --- | --- | --- |
| 1 | 打开首页 | 看到"📚 知识库问答"与侧栏降级提示 |
| 2 | 上传 `tests/fixtures/sample50.pdf` | 进度条走完，显示"50 页 → 50 块"与四段耗时 |
| 3 | 提问"违约金怎么算" | 逐字出字（不是一次性出现） |
| 4 | 看耗时徽标 | 检索 ~10ms / 首字 Xms / 总计 |
| 5 | 展开「参考来源」 | 看到 P7 之类页码 + 原文 + **依据句高亮** |
| 6 | 问一个知识库里没有的问题 | 「知识库中未找到相关内容」，不是编答案 |

> ✅ **步骤 3~5 的能力已在本机验证过**（W6 周日修掉 502 之后）：
> 20 条真实问答跑通，流式、引用、拒答都验过，见 [answer-quality.md](answer-quality.md)。
>
> ⚠️ **但"在 Cloud 上跑"仍未验证** —— 差别有两处，都不是代码问题：
> 1. 线上走 `hashing`（无 torch），检索质量低于本机测的 bge
> 2. 线上需要一个公网 LLM 端点（见上文三条路）
>
> 所以这张清单**只能在拿到链接后逐条跑**，现在仍是"待验证"。

---

## 未做 / 未验证（诚实清单）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 真实线上链接 | ❌ **未创建** | 需要 GitHub 账号授权 + 一个公网 LLM 端点。仓库侧已就绪（见 §部署步骤 ①） |
| 线上全链路跑通 | ❌ 未验证 | 本机已通（75% 正确率），但线上是 `hashing` + 公网端点，两个变量都换了 |
| **线上依赖文件是否真的被读到** | ⚠️ **未实机验证** | 我按官方文档的检索规则修正了（入口目录 + `requirements.txt`），**但只有真正部署一次才能确认**。若 Cloud 仍装 torch，说明我对规则的理解有误 —— 那时看构建日志的包列表 |
| ONNX 量化版 | ❌ 未做 | 见上文"为什么选 hashing" |
| docker compose 的 UI service | ❌ 未做 | 记在上文"备选"一节 |
| 并发会话下的内存 | ❌ 未测 | `st.cache_resource` 是进程级单例，多会话共享模型；真实峰值需线上观测 |
| Chroma 在免费层的磁盘配额 | ❌ 未测 | 免费层磁盘有限，`data/chroma` 长期增长会撞上限 |
