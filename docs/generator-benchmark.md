# 生成服务基准实测（W8 周五）—— 五个数字 + 口径

> **状态（2026-12-11）：待真机实测。** 本文档的数字表格已经建好，
> 每一行都是"跑通后填"的状态。**没填的格子就是没测的** —— 不写估算值，
> 不写"大概"，不引用别人的公开 benchmark 冒充自己的实测。
> 简历上引用任何倍数之前，这里的表格必须先填满。
>
> 复跑命令：`python -X utf8 scripts/benchmark_generator.py --size 512 --repeats 8`

## 一、为什么本机今天测不出（如实记录）

| 项 | 现状 | 影响 |
| --- | --- | --- |
| GPU | RTX 4050 Laptop，6GB，驱动 616.92（CUDA 13.4） | 硬件在位 |
| torch | `2.15.0+cpu`（CPU 轮子） | `torch.cuda.is_available() == False` |
| diffusers | 未安装 | 生成栈缺失 |
| SD1.5 + LoRA 权重 | 未下载（几十 GB 级） | 无推理对象 |

补齐路径（按顺序）：
1. `pip install torch --index-url https://download.pytorch.org/whl/cu128`（CUDA 轮子，~2.5GB）
2. `pip install diffusers transformers accelerate safetensors`
3. 权重放到 `models/sd15/stable-diffusion-v1-5/` + `models/sd15/lora/`
4. 跑上面的基准脚本

## 二、五个数字（跑通后填）

| # | 指标 | 口径 | 数值 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | 单张 512² 耗时（**含冷启动**） | FP16 · steps=25 · attention_slicing · 含 CUDA 图优化与显存分配 | — | ⬜ 待测 |
| 2 | 单张 512² 耗时（**预热后**中位数） | 同上，8 次取中位 | — | ⬜ 待测 |
| 3 | 显存峰值 | FP16 · batch=1 · 512² / 768² 分开报 | — | ⬜ 待测 |
| 4 | batch=1 vs batch=4 吞吐 | 张/秒（wall time），显存分开报 | — | ⬜ 待测 |
| 5 | 模型加载时间 | 进程起 → pipeline 可用（含 LoRA 挂载，另列） | — | ⬜ 待测 |

**必答问题的回答（当前版本，数据齐后更新）**：

简历上那句"130×"的基线到底是什么？——**今天无法给出实测答案**，如实改写：

- 基线候选 A：**逐张在线生成**（传统做法：不缓存、不 batch、每请求重新规划显存）。
  候选 B：**离线批处理**（预生成一个大池子，运行时只做检索）。
- 这两条基线的"倍数"相差一个量级，**没实测之前不写任何倍数**。
- 本项目架构上真正成立、且不依赖未跑数字的表述是：
  > 「异步任务队列（202 + 轮询）替代同步生成：接口 P50 从『等待整批推理完成』
  > 降为『提交毫秒级返回』；GPU 侧单 worker 串行 + 幂等复用，重复请求不再占用 GPU」。
  这三句都有 Thursday 的测试与代码背书，可以先写。
- 「130×」这类倍数**留到基准表填满之后**，按 `baseline.per_image_warm_ms` 除出来。

## 三、口径纪律（写数字时的硬规则）

1. **硬件型号 + 精度 + batch + steps 必须同句出现**。"单张 1.8 秒"没有意义，
   "RTX 4050 / FP16 / batch=1 / steps=25 / 512² 下 1.8 秒"才可复现。
2. **冷启动与预热分开**。"第一次 47s、之后 1.8s" —— 差的 45s 是加载，
   把它混进单张耗时会让人以为推理就这么慢。
3. **是否含 VAE 解码必须写**。`pipe()` 返回的是 PIL 图（含 VAE 解码）；
   只测 UNet 去耗的话数字会好看一截，但那不是用户等的时间。
4. **数字与数字比之前先对口径**。同一硬件、同精度、同 steps 才能相除；
   跨硬件的倍数是营销，不是工程。

## 四、部署验证清单（compose-gpu.yml，跑通后逐条打勾）

| # | 步骤 | 期望 | 实测 |
| --- | --- | --- | --- |
| 1 | `docker run --rm --gpus all industrial-generator:v0.1 python -c "import torch; print(torch.cuda.is_available())"` | True（GPU 直通） | ✅ **True**（2026-12-11） |
| 2 | `docker build -f Dockerfile.gpu -t industrial-generator:v0.1 .` | 构建成功 | ✅（体积见下表） |
| 3 | `docker compose -f compose-gpu.yml up -d` | 容器 healthy | ✅ `Up (healthy)` |
| 4 | `curl :8001/health` | 200 | ✅ `{"status":"ok"}` |
| 5 | `POST :8001/api/v1/generate`（无权重） | 503 `MODEL_NOT_LOADED`（可诊断） | ✅ |
| 6 | `POST :8001/api/v1/generate/async` → 轮询 | queued → failed + 结构化错误码 | ✅ 全链路（状态机在容器内工作） |
| 7 | 权重目录不挂载时启动 | 503 `MODEL_NOT_LOADED`，不是静默假成功 | ✅ |

## 五、镜像体积：**实测 12.8GB，未达 <2GB 目标（如实记录）**

| 层 | 体积 | 说明 |
| --- | --- | --- |
| `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` 基座 | 12.1 GB | CUDA 12.8 运行库 + cuDNN9 + conda + pytorch 自带 cu 轮子 |
| `apt-get install libgl1 libglib2.0-0` | 201 MB | opencv 系图像库运行时依赖 |
| `pip install diffusers/transformers/fastapi…` | 283 MB | 生成栈 + API 栈 |
| `COPY app + healthcheck + useradd` | ~6 MB | 应用源码 |
| **合计** | **12.8 GB** | **权重 0 GB（volume 挂载，未进镜像 ✅）** |

**为什么超了、为什么这是当前最优**：
- 目标"镜像 < 2GB"的隐含假设是"CPU 推理或外部 GPU 服务"。
  SD1.5 FP16 推理**必须**有 CUDA 运行库 + cuDNN（~10GB，无从削减），
  这是 `-runtime` 而非 `-devel` **之后**的下限（-devel 还要再加 ~6GB）。
- 权重不进镜像这条验收**达成**：`.dockerignore` 排除了 `*.safetensors/checkpoints/`，
  `docker history` 证明最大层是基座，没有任何权重层。
- 可行的减体积路径（未做，如实登记）：
  1. `torch==2.8.0+cu118`（CUDA 11.8 运行库更小，但 4050 驱动要兼容性确认）
  2. 换 `nvidia/cuda:12.8-runtime` 裸基座 + 只装 CPU 版依赖 + torch cu 轮子（省 conda 全家桶 ~2GB）
  3. 多阶段：runtime 阶段只 COPY site-packages 的子集（省 conda/pip 缓存 ~1GB）
- **结论表述**：镜像 12.8GB（不含权重），其中 12.1GB 是 CUDA 运行时基座 ——
  写简历时写「权重与镜像分离（volume 挂载），镜像 12.8GB 全部为运行时依赖」，
  不写"<2GB"。
