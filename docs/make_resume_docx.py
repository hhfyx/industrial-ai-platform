"""生成简历 DOCX（W14 周五定稿版的 Word 形态）。

版式目标：单页 A4、紧凑行距、与原版"直聘简历"的信息结构一致。
姓名/联系方式保持原版；数字全部来自 docs/resume-final-w14.md（台账编号可追溯）。
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn

DARK = RGBColor(0x1F, 0x24, 0x2B)
GRAY = RGBColor(0x5A, 0x62, 0x6B)
BLUE = RGBColor(0x1A, 0x5F, 0xB4)

doc = Document()

# 页面：A4 窄边距
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(1.1)
sec.left_margin = sec.right_margin = Cm(1.4)

# 正文默认字体
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(8.5)
style.font.color.rgb = DARK
style.paragraph_format.space_after = Pt(1)
style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
style.paragraph_format.line_spacing = 1.04
style.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")


def set_eastasia(run, font="微软雅黑"):
    run.font.name = run.font.name or "Calibri"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)


def para(text="", size=8.5, bold=False, color=None, space_after=1, space_before=0,
         align=None, bullet=False):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    if bullet:
        pf.left_indent = Cm(0.35)
        pf.first_line_indent = Cm(-0.35)
        text = "•  " + text
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    r.font.color.rgb = color or DARK
    set_eastasia(r)
    return p


def heading(text, size=10.5, space_before=4):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(2)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = True
    r.font.color.rgb = BLUE
    set_eastasia(r)
    # 下边框线
    pPr = p._p.get_or_add_pPr()
    bd = pPr.makeelement(qn("w:pBdr"), {})
    bottom = pPr.makeelement(qn("w:bottom"), {qn("w:val"): "single", qn("w:sz"): "6",
                                              qn("w:space"): "1", qn("w:color"): "1A5FB4"})
    bd.append(bottom)
    pPr.append(bd)


def mixed(p, parts):
    """parts: list of (text, bold, color|None)"""
    for text, bold, color in parts:
        r = p.add_run(text)
        r.font.size = Pt(8.5)
        r.bold = bold
        r.font.color.rgb = color or DARK
        set_eastasia(r)


# ============ 抬头 ============
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(0)
r = p.add_run("黄浩锋"); r.font.size = Pt(17); r.bold = True; r.font.color.rgb = DARK; set_eastasia(r)

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(1)
mixed(p, [("男  |  15989527280  |  1792472354@qq.com  |  ", False, GRAY),
          ("github.com/hhfyx/industrial-ai-platform", True, BLUE),
          ("  |  求职意向：AI 应用开发实习生  |  期望城市：广州", False, GRAY)])

# ============ 教育经历 ============
heading("教育经历")
p = para(space_after=0)
mixed(p, [("广州城市理工学院", True, None), ("    本科 · 人工智能    2023 – 2027", False, GRAY)])

# ============ 专业技能 ============
heading("专业技能")
skills = [
    ("LLM 应用：", "RAG 全链路（切分三策略实测、bge 向量、BM25+向量 RRF 混合、Cross-Encoder 精排、上下文压缩）；评测体系（黄金集防泄漏、Recall@k / MRR / NDCG、拒答三分类、LLM-as-judge + 人工校准 30/30、McNemar / Wilson 置信区间）；Agent 工程（LangGraph 状态图、工具调用、检查点与人机中断、重试降级矩阵、预算控制）；Prompt 工程（JSON Schema 强约束、模板版本机制、单维度 A/B）；模型接入（OpenAI 兼容 API、Function Calling、SSE 流式、重试降级、Token 成本对账）"),
    ("工程：", "Python（asyncio / Pydantic v2 / FastAPI）、C++、Docker（GPU 直通、多阶段构建）、GitHub Actions（CI）、pytest 1300+ 条离线测试（拔网全绿）、ruff + pre-commit"),
    ("算法：", "PyTorch、SD+LoRA 微调、YOLO、OpenCV、ROS / AMCL / A*"),
]
for label, body in skills:
    p = para(bullet=True, space_after=1)
    mixed(p, [(label, True, None), (body, False, None)])

# ============ 项目一 ============
heading("项目经历")
p = para(space_after=0)
mixed(p, [("项目一｜文档知识库 RAG 平台：检索 → Agent → 可观测（个人）", True, None)])
p = para(space_after=0)
mixed(p, [("github.com/hhfyx/industrial-ai-platform", True, BLUE)])
p = para(space_after=2)
mixed(p, [("Python / FastAPI / LangGraph / bge / rank_bm25 / ChromaDB / Streamlit / Docker", False, GRAY)])

para("从零构建“切分→混合检索→精排→引用生成”全链路并服务化（鉴权 / 限流 / 缓存 / 双容器编排），"
     "自建 120 条黄金集完成量化评测与失败归因；公网 demo 在线（免费层降级为字面检索，降级显式可见）。",
     space_after=2)

bullets1 = [
    [("检索与生成：", True, None),
     ("递归切分（chunk 500/50，约 1.2k 块），bge 向量 + BM25 RRF 融合 + 可选精排/压缩（实测后默认关闭）；答案强制带 [n] 引用并原文高亮，引用编号双向校验（越界重答→仍越界标 degraded）、上下文近重复去重", False, None)],
    [("评测体系：", True, None),
     ("120 条黄金集（80 事实 / 20 多跳 / 20 拒答），题源全部来自 13 份未参与调参的文档，机器门禁防跨集泄漏；Recall@5 56.0%（事实 66.2% / 多跳 15.0%）、Recall@10 64.0%、MRR 0.427（100 可答题，judge 三档校准一致率 30/30）；答案正确率随检索命中层数单调衰减 67%→38%→15%——检索质量决定生成上限", False, None)],
    [("四层消融严格累加：", True, None),
     ("精排 +17.5pp 但延迟 119ms→17.5s（×147，同集同配置 p50 相比）；混合检索 −5pp——用“两路候选独有占比 86%、RRF 偏好共识”解释“加上就好”的直觉为何失效；对照全部 p>0.05，不显著即不上线", False, None)],
    [("拒答：", True, None),
     ("可答题 top1 分数 [0.513, 0.629] 与拒答题区间完全重叠，阈值 0.575 下误拒 12 + 幻觉 14 同时爆发——据此判定“调参不可修”，转向二段判定", False, None)],
    [("Agent 层（LangGraph）：", True, None),
     ("四工具 + Sqlite 检查点 + 人机中断（副作用恰好一次）；自建 20 任务评测集（判据冻结），5 轮成功率 70%–100% 区间（降级计入失败），6 个不可完成任务编造 0 例（答案数值回查工具返回，轨迹可复跑）；预算-成功率曲线实测拐点 10k", False, None)],
    [("可观测：", True, None),
     ("trace 三级埋点（12 span/请求）+ 逐笔成本账本 + 聚合对账——对账抓出自己埋的两个真缺口（一处 LLM 入口漏埋点，成本低估 45%）；告警阈值全部指向实测出处；MCP Server 五道闸路径安全，5 条攻击全拒", False, None)],
    [("工程纪律：", True, None),
     ("1300+ 条离线测试、CI 接入、tasks.py run-eval 一条命令出全部指标，报告随附 commit 与语料指纹", False, None)],
]
for parts in bullets1:
    p = para(bullet=True, space_after=1)
    mixed(p, parts)

# ============ 项目二 ============
p = para(space_after=0, space_before=4)
mixed(p, [("项目二｜缺陷图像生成服务：SD+LoRA · 异步队列（个人）", True, None)])
p = para(space_after=0)
mixed(p, [("github.com/hhfyx/industrial-ai-platform/blob/main/docs/README-generator.md", True, BLUE)])
p = para(space_after=2)
mixed(p, [("Python / FastAPI / diffusers / PEFT（LoRA）/ Docker（GPU 直通）/ SD1.5", False, GRAY)])

para("给缺陷检测数据集做数据增广：产线大图 YOLO 裁剪出 3,323 张、6 类缺陷切片（多源合并 +84%），"
     "SD1.5 + 每类别一个 LoRA 适配器（只训练 2.18M 低秩参数，基座冻结），FastAPI 封装为同步 + 异步端点，"
     "单 worker 串行保护 GPU。", space_after=2)

bullets2 = [
    [("训练提速（126×，口径完整）：", True, None),
     ("初始实现每步在线 VAE 编码 → 改为一次性 VAE 预计算全部 latent 缓存盘，单步 39s→0.31s（126×）、每 epoch 19.5h→17min、显存 13GB→7GB（训练卡 11GB+ 档实测）", False, None)],
    [("推理实测（消费级显卡口径）：", True, None),
     ("RTX 4050 6GB · FP16 · steps=25 下单张 512² 约 5.0s（两轮 4.87–5.00s）、显存 2.81GB；batch=4 吞吐反降约 15–19%（反直觉结论，两轮复现）——6GB 卡上 batch 化不划算", False, None)],
    [("异步化收益（架构，不是 GPU 加速）：", True, None),
     ("提交接口 P50 4.2ms（202 + task_id + 轮询）vs 同步阻塞 ~5s；GPU 单 worker 串行 + 参数级幂等复用（重复请求 0 GPU 占用）；状态机表驱动，非法迁移当场抛", False, None)],
    [("部署：", True, None),
     ("GPU 镜像实测 12.8GB（其中 12.1GB 为 CUDA 运行时基座，权重 0 字节进镜像，docker history 逐层验证）；容器内 GPU 直通、无权重启动返回可诊断 503 而非静默假成功", False, None)],
]
for parts in bullets2:
    p = para(bullet=True, space_after=1)
    mixed(p, parts)

out = r"D:\git\网页\industrial-ai-platform\docs\resume-final-w14.docx"
doc.save(out)
print("saved:", out)
