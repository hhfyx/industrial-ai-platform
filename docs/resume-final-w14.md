# 直聘简历 · 精简单页版（W14 最终版）

> **2026-11-20 产出**。目标：一页 A4 装下两段项目 + 企业业务模块，砍修饰留硬货。
> 每个数字仍可溯源（台账 `docs/w14-number-ledger.md` #n）。
> 裁剪原则：**同一件事只说一次**；技能栏只留项目里真用到的；负结果只留最有信息量的。

---

黄浩锋
男 | 15989527280 | 1792472354@qq.com | github.com/hhfyx/industrial-ai-platform
求职意向：AI 应用开发实习生 | 期望城市：广州

## 教育经历

广州城市理工学院 | 本科 | 人工智能 | 2023–2027

## 专业技能

- **LLM 应用**：RAG 全链路（切分/混合检索/精排/压缩）、LangGraph Agent
  （工具调用/检查点/人机中断）、评测体系（黄金集防泄漏、Recall@k / MRR、
  LLM-as-judge + 人工校准 30/30）、Prompt 工程、SSE 流式
- **工程**：Python（FastAPI / asyncio / Pydantic v2）、MySQL（SQLAlchemy 2.0 /
  Alembic / 事务与乐观锁）、Docker（GPU 直通）、GitHub Actions、
  pytest 1300+ 条离线测试
- **算法**：PyTorch、SD+LoRA 微调、YOLO、OpenCV

## 项目经历

### 项目一｜文档知识库 RAG 平台：检索 → Agent → 可观测（个人）

github.com/hhfyx/industrial-ai-platform
Python / FastAPI / LangGraph / bge / ChromaDB / Docker · [公网 demo 在线]

从零构建“切分→混合检索→引用生成”全链路并服务化（鉴权/限流/缓存/双容器编排），
自建评测体系量化每一步；公网 demo 在线（免费层降级字面检索，降级显式可见）。

- **评测体系**：自建 120 条黄金集（80 事实/20 多跳/20 拒答，13 份未参与调参
  文档出题，机器门禁防泄漏）；**Recall@5 56.0%**、MRR 0.427（judge 三档
  校准 30/30）；**答案正确率随检索命中层数 67%→38%→15% 单调衰减**——
  检索质量决定生成上限
- **负结果如实入报告**：精排 +17.5pp 但延迟 ×147（不显著即不上线）；
  混合检索 −5pp（RRF 挤掉独有召回）；拒答分数区间完全重叠→判定
  “调参不可修”转二段判定
- **Agent 层**：LangGraph 四工具 + 检查点 + 人机中断（副作用恰好一次）；
  20 任务评测 5 轮成功率 **70%–100%**（降级计入失败），不可完成任务
  编造 **0 例**（数值回查工具返回，轨迹可复跑）
- **可观测**：trace 三级（12 span/请求）+ 逐笔成本账本 + 聚合对账——
  **对账抓出自己漏埋的 LLM 入口（成本低估 45%）**；MCP Server 五道闸
  路径安全，5 条攻击全拒

### 项目二｜缺陷图像生成服务：SD+LoRA · 异步队列（个人）

github.com/hhfyx/industrial-ai-platform/blob/main/docs/README-generator.md
Python / FastAPI / diffusers / PEFT / Docker（GPU 直通）

缺陷检测数据集增广：YOLO 裁剪 **3,323 张/6 类**切片，SD1.5 + 每类一个
LoRA 适配器（只训练 2.18M 参数，基座冻结），异步队列单 worker 保护 GPU。

- **训练提速 126×（口径完整）**：每步在线 VAE 编码 → 一次性预计算 latent
  缓存盘，单步 39s→0.31s、显存 13GB→7GB（训练卡实测）
- **推理实测（RTX 4050 6GB）**：单张 512² **约 5.0s**（两轮 4.87–5.00s）、
  显存 2.81GB；**batch=4 吞吐反降 15–19%**（反直觉结论，两轮复现）
- **异步化（架构收益非 GPU 加速）**：提交 P50 **4.2ms**（202+轮询）
  vs 同步阻塞 ~5s；幂等复用（重复请求 0 GPU 占用）；GPU 镜像 12.8GB
  **权重 0 字节进镜像**（docker history 逐层验证）

### 项目三｜企业业务模块：合同审批（个人）

github.com/hhfyx/industrial-ai-platform/blob/main/docs/module-contract-approval.md
MySQL 8 / SQLAlchemy 2.0 / Alembic / FastAPI

- **5 表建模**（客户/合同/审批节点/审计日志/幂等键）：金额用「分」存整数、
  `version` 乐观锁、审计日志只 INSERT；5 态状态机纯函数实现（非法迁移 409），
  金额阈值路由审批链（≥10 万两级审批）
- **并发正确性实测**：单事务改三表 + `expected_version` 校验——
  **20 并发 approve 恰好 1 成功、19×409、日志只多 1 条**（MySQL 活体测试）；
  报表按月×客户 GROUP BY 聚合 + CSV 导出（UTF-8 BOM，Excel 直开不乱码）

## 个人特长

- **口径先行**：每个数字标 n/配置/复现命令（自建数字台账 + 一键复现脚本）
- **负结果也是结果**：不显著不声称、无效优化如实写——压测报告里没有
  失败的优化等于没做实验
- **文档随代码走**：84 篇技术文档、13 篇周复盘随 commit 入库，可回溯
