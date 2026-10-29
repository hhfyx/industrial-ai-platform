v0.3.0 — W3 Prompt 工程周

本版本把「模型输出」从"祈祷它按格式回"变成"有契约、有校验、有可复现对比"。

## 核心交付

**1. structured_call（签名定版，后续四周共用）**
   async def structured_call(prompt, schema, *, max_repair=2, fallback=None) -> M
   三层防线：prompt 渲染 schema / json_object 挡格式 / Pydantic 校验（唯一真源）
   校验失败把 Pydantic 报错原文回灌，最多 2 轮，用尽抛 StructuredOutputError(ValueError)

**2. 四类模板库 app/prompts/**
   qa / summarize / classify / extract，每类 v1.jinja2 + schema.py
   load_prompt(name, version) 三级版本优先级（显式 > 类别环境变量 > 全局）
   模板不得硬编码业务词 —— 做成会报错的检查（黑名单外置文件）

**3. 40 条输入池 + A/B 脚手架**
   eval/prompt-sets/ 4 类 × 10 条，全部从 docs/README 语料抽取，每条带 source
   数据泄漏检查机器可验（扫已用实验输入，零重叠）
   temperature/seed 写死不作为参数；结论阈值 MIN_SIGNAL_DELTA=0.20 写在代码里

## 四类 A/B 实测结论（全量口径，失败记 0）

   qa         0.5717 -> 0.5350   (-3.7pp, 无显著差异)
   classify   0.8000 -> 0.8000   (打平)
   extract    0.1333 -> 0.7667   (+63.3pp, 大胜)
   summarize  0.9000 -> 0.3000   (-60pp, 70% 超时)

定版：只有 extract 升 v1.1（重跑确认 +0.6334 与原实验完全一致）。

归因（不是"效果好"，是具体改动）：
- extract 胜在契约修复（固定槽位 -> 任意字段清单），不是措辞
- qa/summarize 败在「分步结构」—— 推理模型自己已在分步，再加一层
  导致答案被压缩 / 思维链过长撞超时

## 关键认知（本周最重要的三条）

1. 本网关 response_format 是「形式接受、语义不强制」：
   嵌套 schema 0/10，扁平 schema 8/10。不是"完全不强制"，是"对嵌套不强制"。

2. LangChain 四种配置在嵌套 schema 上全部 0/10：
   - function_calling 网关直接回 only tool_choice auto and none are supported
   - 叠加"不会自我修复"，所以不用它 —— 是有效区间不重叠，不是框架恐惧症

3. 成本指标必须按成功样本归一：
   失败的请求不产出 token，"总成本更低"可能只等于"有几次没跑起来"。
   摘要 v2 原报省 62%，按每条成功归一后实际贵 37%。

## 已验证

- 327 条测试全离线（拔网线全绿），覆盖率 79%
- ruff check / ruff format 干净
- docker compose up --build：容器 healthy、/health 200、
  /api/v1/extract 返回 200 + degraded:true、规则版结果正确、多传一键 422

## 未验证（诚实标注）

- **容器内的 LLM 路径跑不通**（本周未验证，不是已修复）：
  容器内 127.0.0.1 指容器自己；改成 host.docker.internal 后网络通了，
  但网关回 502「未配置凭据」—— 本机网关鉴权绑宿主进程登录态，容器是另一个身份。
  详见 docs/container-networking.md。上真实容器部署前必须先解决。
- 摘要任务延迟不适合线上：v1 P95 55s / v2 29s，根因是 reasoning=high。
  W4 开工第一件事（见 docs/weekly/w3.md 5.1）。
- reasoning=off 的对照组始终没跑。
- 每类 10 条样本，1 条 = 10pp，<20pp 的差异不构成结论。

## 明镜：本周自我推翻五次

判定/统计口径错误四次（拒绝率二元口径、不确定+引用误判、
汇总只算成功条目、成本未按成功归一）+ 单样本代表一整类一次。
全部记在 docs/weekly/w3.md 3.1。

## 文档

docs/{prompt-patterns,system-prompt,structured-output,prompt-templates,
ab-scaffold,prompt-ab,why-i-use-langchain,container-networking}.md
docs/weekly/{w1,w2,w3}.md
