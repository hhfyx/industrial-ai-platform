# 技术报告一：缺陷图像生成服务（SD + LoRA · 数据增广）

> W14 周三产出。**本报告每一句都可验证**：数字带台账编号
> （`docs/w14-number-ledger.md` #n），命令可在仓库根目录复跑。
> 页脚有 commit hash 与日期。

---

## 1. 背景（为什么做这个）

工业缺陷检测的数据集天然稀缺：产线大图里缺陷占比极小，标注成本高，
同一类缺陷的可训练样本往往只有几百张。检测模型的瓶颈通常不在架构而在数据——
**同类缺陷的形态多样性不足，模型学到的是"这几张图"而不是"这类缺陷"**。

本项目的答案是：用扩散模型按缺陷类别生成可控的增广图像。
约束条件决定了全部技术选型：

| 约束 | 选型 | 放弃的替代 |
| --- | --- | --- |
| 消费级显卡（RTX 4050 6GB / 训练卡 11GB+） | SD 1.5（512² 原生分辨率） | SDXL（显存翻倍起） |
| 每类样本少、类别会增长 | LoRA 每类别一个适配器 | 全参数微调 / 每类一个完整模型 |
| 类别可控（生成"划痕"就得是划痕） | `<defect>` token + prompt 控制 | 无条件 DDPM（不可控，32² 分辨率） |

---

## 2. 数据层

**来源与预处理**：产线拍摄大图（约 3000×3000）→ YOLO 标注框 → 前景裁剪
（5% padding）→ 等比缩放 512²。

**规模**（出处：训练工作区报告 §1.5 目录计数与 §1.3 类型分布表，不可本仓复跑）：

| 数据目录 | 张数 | 内容 |
| --- | ---: | --- |
| `00_class_0` | 1,687 | 类型 0/1/8/9（产品表面） |
| `02_class_2` | 111 | 类型 0/1/8/9 |
| `05_class_5` | 8 | 类型 0/1 |
| `fg_class_1` | 876 | 类型 a（标签破洞） |
| `fg_class_4` | 641 | 类型 b（颜色胶布） |
| **合并后合计** | **3,323** | 6 种缺陷类型 |

**如实登记一处源数据不一致**：训练工作区报告 §1.3 的类型分布表各行加总为
3,164，与其自称的合计 3,323 差 159（疑为 02/05 两目录的类型细分未单列）。
本报告因此**以目录计数为准**（上表加总自洽），类型分布只引占比量级
（type 9 占 23.9% 最大、type 0 占 6.6% 最小）。引用数据规模时用"3,323 张、
6 类（目录计数）"，不引用类型分布表的逐行数字。

**合并带来 +84% 数据量与 +2 类缺陷**（1806→3323 张，4→6 类）——
多源合并是数据层最大的一次增量，代价是需要处理各源的目录结构与标签体系差异。

---

## 3. 模型层

### 3.1 架构

SD Inpainting（UNet 9ch 输入：noisy_latent + zero_mask + zero_masked_image）
+ LoRA（rank=8, alpha=8）挂在 UNet attention 层（to_q/to_k/to_v/to_out.0）
与 Text Encoder（q/k/v/out_proj）。

**只训练低秩新增参数：约 2.18M**（UNet ~1.8M + Text Encoder ~0.4M），
基座 84M VAE 与 UNet 全部冻结。6 个类别 = 6 个适配器文件
（UNet ~5MB + TE ~2MB），**换类别 = 换文件，基座不动**——这是"类别会增长"
约束的直接产物。

### 3.2 训练提速（126× 的出处与口径）

**这是「130×」传闻的真实出处，基线说得清**（台账 #48，周三破案）：

| 指标 | 优化前 | 优化后 | 提升 |
| --- | ---: | ---: | ---: |
| 单步耗时 | 39 s/step | 0.31 s/step | **126×** |
| 每 epoch | 19.5 h | 17 min | 69× |
| GPU 显存 | 13 GB | 7 GB | −46% |

- **基线**：初始实现的训练步内逐步调用 VAE encoder 编码当前 batch 图片；
- **对照**：`precompute_latents()` 一次性 VAE 编码全部 3323 张缓存到
  `latents_cache.pt`，训练步直接读 latent，VAE 释放（省 ~2.5GB）；
- **口径**：训练侧（训练工作区报告 §5.4 实测，训练卡 11GB+ 显存档）。
  **本仓库无法复跑**（训练脚本不在本仓库），引用必须注明出处。
- 附加：FP16 权重 + `enable_gradient_checkpointing()`。
  **没有使用 AMP**（autocast/GradScaler 未启用）——任务书里"预计算 + AMP"
  的 AMP 部分是预设错误，本报告如实不写。

### 3.3 训练结果

Loss 0.1387（epoch 1）→ 0.0889（epoch 13），降幅 35.9%，仍在下降（报告 §5.3）。
生成模式两种：txt2img（纯噪声 + prompt 控类）、img2img（参考图加噪再去噪，
strength 0.85 保留部分结构）。

---

## 4. 工程层（服务化）

**如实声明服务边界**（台账 §十二第 4 条）：`SdLoraGenerator` 是**契约壳**
——装配、异常映射、白名单、错误码、状态机全部就位并有测试，
但 `generate()` 抛 `GeneratorUnavailableError`：**服务化的是"队列 + 状态机 + 契约"，
不是"diffusers 调用"**。真实 diffusers 推理在独立基准脚本中实测
（见 §5），两条链路不互相冒充。

| 设计 | 为什么 |
| --- | --- |
| 同步 + 异步双端点 | `POST /api/v1/generate`（挂起等图）与 `POST /api/v1/generate/async`（202 + task_id 立回）服务两类场景 |
| **单 worker 串行** | GPU 独占：并发 worker 抢显存 → OOM，吞吐反降。"并发"的正确位置在入口（毫秒级各拿 task_id），推理侧排队。**实测佐证**：batch=4 吞吐 0.158 张/s **低于** batch=1 的 0.195（台账 #46，两轮复现）——连 batch 化都不划算 |
| 推理放 `asyncio.to_thread` | 单张 ~5s，不丢线程会冻结事件循环，连状态查询都卡死 |
| 状态机表驱动 | queued→running→succeeded/failed 唯一迁移路径，非法迁移当场抛；无 running→queued"重试"（重试=新任务） |
| 参数级幂等 | `(defect_class, count, size, seed)` 相同且已成功 → 返回原 task_id，0 GPU 占用 |
| 白名单类别 | 表外 422，绝不静默用基座生成 |

---

## 5. 部署与实测

**GPU 基准**（台账 #43–47；`SD_WEIGHTS_DIR=models/sd15 python -X utf8
scripts/benchmark_generator.py --size 512 --repeats 8`；真实 diffusers 推理）：

| 指标 | 值 | 口径 |
| --- | --- | --- |
| 单张 512²（含冷启动） | 6,010 ms（两轮 5,826–6,010） | RTX 4050 6GB / FP16 / steps=25 / attention_slicing |
| 单张 512²（预热中位） | **约 5.0s（两轮 4,867–5,002ms，漂移 2.8%）** | 8 次/轮取中位；含 VAE 解码 |
| 显存峰值 | 2.81 GB（两轮逐位一致） | batch=1, 512² |
| batch 吞吐 | batch=4 **0.158 反低于** batch=1 0.195 张/s | 6GB 卡上 batch 化不划算 |
| 模型加载 | 2.67–3.45 s | 进程起 → pipeline 可用 |

**镜像与部署**（台账出处 generator-benchmark.md §五 + 部署验证清单 7 项全过）：
镜像实测 12.8GB（其中 12.1GB 是 CUDA 12.8 运行时基座，`-runtime` 而非 `-devel`
的下限，**权重 0 字节进镜像**——`docker history` 逐层验证）；容器内
`torch.cuda.is_available() == True`；无权重启动 503 `MODEL_NOT_LOADED`
（可诊断，不是静默假成功）。

**消费级显卡口径的一句话**：推理在 6GB 卡实测可跑（5.0s/张、2.81GB）；
训练需要 11GB+ 显存档（优化前 13GB，预计算后 7GB）——**两个数字不混用**。

---

## 6. 已知局限（与效果数字同样重要）

1. **服务内核是契约壳**：异步队列/状态机/错误码在 FakeGenerator 上全链路测试
   （部署验证 7 项含容器内状态机验证），但 `SdLoraGenerator.generate()` 尚未接
   diffusers——"接口层全绿、内核待装配"是唯一诚实的表述。
2. **生成多样性评估是主观的**：txt2img/img2img 样本经人工目检分类正确性，
   无 FID/CLIP-score 等量化指标——增广效果对下游检测模型的提升**未闭环验证**
   （生成的图还没喂回检测器测 mAP）。
3. **训练侧数字不可本仓复跑**（脚本在训练工作区）；`defectfill2` 环境在位，
   复现需该工作区。
4. **无鉴权**（本地/演示定位）；镜像 12.8GB 的三条减体积路径均未实施。

---

## 7. 复现命令

```bash
# GPU 基准（本仓，需 GPU + SD1.5 权重）
SD_WEIGHTS_DIR=models/sd15 python -X utf8 scripts/benchmark_generator.py --size 512 --repeats 8

# 服务契约（FakeGenerator，无 GPU 可跑）
docker compose -f compose-gpu.yml up -d
curl -X POST http://localhost:8001/api/v1/generate/async -H "Content-Type: application/json" \
  -d '{"defect_class":"划痕","count":2,"size":512,"seed":42}'

# 数字台账总核对
python -X utf8 scripts/tasks.py report-numbers
```

训练侧复现（需训练工作区）：`conda activate defectfill2` →
`python train_defect_generator.py --dataset_root defect_merged --epochs 30 --lr 1e-4`。

---

*报告生成：2026-11-14（W14 周三）· commit `5e9caf5`（后续以仓库最新为准）·
数字台账 `docs/w14-number-ledger.md` · 复现入口 `scripts/report_numbers.py` ·
训练侧数据出自训练工作区报告（已注明出处与不可复跑边界）*
