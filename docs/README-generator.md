# 项目一：缺陷图像生成服务（SD + LoRA · FastAPI 异步封装）

> 这是本仓库的**项目一**。项目二（文档知识库 RAG 平台）见[主 README](../README.md)。
> 两个项目共享一个仓库（同一套工程底座：统一错误体、request_id 贯穿、契约冻结纪律），
> 但**部署完全独立**——Dockerfile、compose、端口（8001 vs 8000）互不干扰。

---

## 一句话

**给缺陷检测数据集做数据增广的生成服务**：按缺陷类别（划痕/凹坑/气泡/裂纹/锈蚀/污渍）
生成 512²/768² 的工业缺陷图像，SD1.5 + 每类别一个 LoRA 适配器，
FastAPI 封装成同步 + 异步两个端点，**单 worker 串行消费**保护 GPU。

## 三层结构

### 模型层

| 项 | 值 | 说明 |
| --- | --- | --- |
| 基座 | SD 1.5（FP16） | 消费级显卡的原生分辨率，6GB 卡可跑 |
| 微调 | **LoRA（每类别一个适配器）** | **只训练低秩新增参数**——基座冻结。6 个类别 = 6 个 `.safetensors`，**换类别 = 换适配器文件，基座不动** |
| 权重 | volume 挂载（不进镜像） | `models/sd15/` 目录；换权重不用重 build 镜像 |

**"消费级显卡可跑"的口径**（如实）：
- 推理：RTX 4050 Laptop **6GB** 实测可载入 SD1.5 FP16（显存余量见
  [generator-benchmark.md](generator-benchmark.md) 的五数字表——**推理耗时的格子目前待真权重实测**，
  本 README 不引用未实测的数字）；
- **训练不在本服务范围内**：LoRA 适配器是离线训练的产物，本服务只负责**装配与推理**。

### 工程层

```
POST /api/v1/generate            同步：请求挂起直到出图（小批量场景）
POST /api/v1/generate/async      异步：202 + task_id 立刻返回（大批量场景）
GET  /api/v1/generate/tasks/{id} 查询：queued → running → succeeded / failed
```

| 设计 | 为什么 |
| --- | --- |
| **单 worker 串行** | GPU 是独占资源：多 worker 并发只会互相抢显存 → OOM，吞吐反降。"并发高"的正确位置在**入口**（5 个请求毫秒级各拿 task_id），推理侧排队 |
| **推理放 `asyncio.to_thread`** | SD 一跑 8~15 秒，不丢线程会冻结事件循环——连状态查询接口都会卡死 |
| **状态机表驱动** | `queued → running → succeeded/failed` 唯一迁移路径，非法迁移当场抛；没有 running→queued 的"重试"（重试=新任务） |
| **幂等复用** | 相同 `(defect_class, count, size, seed)` 且已成功 → 直接返回原 task_id，不重复占 GPU。`seed` 不传则视为"每次要新的"，不参与幂等 |
| **错误码与项目二同源** | `GPU_OOM`(503+Retry-After) / `GENERATION_TIMEOUT`(504) / `MODEL_NOT_LOADED`(503) / `UNKNOWN_DEFECT_CLASS`(422)——统一错误体 4 字段，一张错误码表 |
| **白名单类别** | 表外类别 422，**绝不静默用基座模型生成**（调用方以为在生成划痕，实际是噪声） |

契约与错误码表全文：[api-spec.md 项目一段](api-spec.md)。

### 部署层

```powershell
docker compose -f compose-gpu.yml up -d     # GPU 版（项目一）
docker compose up -d                        # CPU 版（项目二，另一个文件互不误起）
```

| 项 | 值 |
| --- | --- |
| 镜像 | `industrial-generator:v0.1`，**实测 12.8GB**（逐层分解见 benchmark 文档；其中 12.1GB 是 CUDA 运行时基座，-runtime 而非 -devel） |
| 权重 | **0 字节进镜像**（`docker history` 逐层验证），volume 挂载 |
| GPU 直通 | 容器内 `torch.cuda.is_available() == True`（实测） |
| 端口 | `8001`（与项目二 8000 错开） |
| healthcheck | start-period 150s（覆盖 SD 权重加载的 30~90s） |

## 实测数字（唯一不可伪造的资产）

见 [generator-benchmark.md](generator-benchmark.md)——**五个数字的表格已建好，
部署验证清单 7 项已全部打勾**（GPU 直通 / compose healthy / 503 路径 / 异步状态机全链路）。
推理耗时的数字格子**待真权重实测**，本文档的纪律是：**没实测的格子留空，不写估算**。

## 已知问题（与效果数字同样重要）

1. **推理内核是"签名定版的壳"**：`SdLoraGenerator` 的 `_load_pipeline/_run_pipeline`
   两个注入点已就位、异常映射与装配已实测，但**真实 diffusers 推理没跑过一张图**——
   因为写它的时候本机没有 diffusers 与权重。诚实的说法是"接口层全绿、内核待装配"，
   不是"生成服务已完成"。
2. **镜像 12.8GB 超出 <2GB 目标**：CUDA 运行库不可削减；减体积的三条路径
   （cu118 / 裸 CUDA 基座 / 多阶段裁剪）已写在 benchmark 文档，均未实施。
3. **无鉴权**：与项目二一致的本地/演示定位，公网部署前必须加。
4. **异步任务的进度是字符串**（"3/4"）：结构化到 per-image 进度是 v2 的事。

## 快速体验（FakeGenerator，无 GPU 也能跑）

```powershell
# 默认 GENERATOR_IMPL=fake：真实写盘、真实状态机、毫秒级响应
docker compose -f compose-gpu.yml up -d
curl -X POST http://localhost:8001/api/v1/generate/async `
  -H "Content-Type: application/json" `
  -d '{"defect_class":"划痕","count":2,"size":512,"seed":42}'
# → {"task_id":"...","status":"queued",...} 立刻返回
curl http://localhost:8001/api/v1/generate/tasks/<task_id>
# → {"status":"succeeded","result":{"images":[...],"elapsed_ms":...}}
```
