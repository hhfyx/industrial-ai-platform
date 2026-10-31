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

所以部署侧用 `requirements-deploy.txt`（10 个包，**不含 torch**）。
文件头写了"为什么不装"，因为总有人会"顺手补上缺的包" ——
`test_deploy_requirements_excludes_torch` 挡着。

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

```powershell
# ① 确认仓库是 public（免费层只能读 public repo）
#    https://github.com/hhfyx/industrial-ai-platform

# ② 入口文件与依赖
#    app/ui/streamlit_app.py   ← Streamlit 会自动找 streamlit_app.py
#    requirements-deploy.txt   ← 在 Cloud 的 "Advanced settings" 里指定

# ③ 运行时配置（Advanced settings → Secrets）
LLM_API_KEY = "sk-..."          # 必填，否则提问会 502
LLM_BASE_URL = "https://..."
LLM_MODEL = "..."

# ④ 可选：强制 embedder
RAG_EMBEDDER = "hashing"        # 默认就是它；想试 bge 才改
```

> **入口文件路径要在 Cloud 里手填 `app/ui/streamlit_app.py`** ——
> Streamlit 默认找仓库根的 `app.py`/`streamlit_app.py`，
> 而本仓库根有 `app/` 包（放一个 `app.py` 会遮蔽它，import 直接坏掉）。

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

> ⚠️ **步骤 3~5 我无法在本地验证**：本机网关 chat 端点返回
> `502 {'message': '未配置凭据，请先登录'}`。所以线上验证依赖部署侧配好
> `LLM_API_KEY`。这条是**未验证**，不是"已完成"。

---

## 未做 / 未验证（诚实清单）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 真实线上链接 | ❌ **未创建** | 需要 GitHub 凭据 + Streamlit Cloud 账号授权，我这边只有本地 git |
| 线上全链路跑通 | ❌ 未验证 | 同上，且依赖 chat 端点凭据 |
| ONNX 量化版 | ❌ 未做 | 见上文"为什么选 hashing" |
| docker compose 的 UI service | ❌ 未做 | 记在上文"备选"一节 |
| 并发会话下的内存 | ❌ 未测 | `st.cache_resource` 是进程级单例，多会话共享模型；真实峰值需线上观测 |
| Chroma 在免费层的磁盘配额 | ❌ 未测 | 免费层磁盘有限，`data/chroma` 长期增长会撞上限 |
