"""生成简历 DOCX（W14 精简单页版 · 无括号，2026-11-20）。

与 md 文本版同步：全文无圆括号——括号补充改为直接写进句子或删除。
单页 A4：正文 8pt / 行距 1.0 / 边距 1.2cm。
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn

DARK = RGBColor(0x1F, 0x24, 0x2B)
GRAY = RGBColor(0x5A, 0x62, 0x6B)
BLUE = RGBColor(0x1A, 0x5F, 0xB4)

BODY = 8
doc = Document()

sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(0.9)
sec.left_margin = sec.right_margin = Cm(1.2)

style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(BODY)
style.font.color.rgb = DARK
style.paragraph_format.space_after = Pt(0.5)
style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
style.paragraph_format.line_spacing = 1.0
style.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")


def set_eastasia(run):
    rPr = run._element.get_or_add_rPr()
    fonts = rPr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.insert(0, fonts)
    fonts.set(qn("w:eastAsia"), "微软雅黑")


def mixed(p, parts):
    for text, bold, color in parts:
        r = p.add_run(text)
        r.font.size = Pt(BODY)
        r.bold = bold
        r.font.color.rgb = color or DARK
        set_eastasia(r)


def para(text="", bold=False, color=None, size=None, space_after=0.5, space_before=0,
         align=None, bullet=False):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    if bullet:
        pf.left_indent = Cm(0.3)
        pf.first_line_indent = Cm(-0.3)
        text = "• " + text
    r = p.add_run(text)
    r.font.size = Pt(size or BODY)
    r.bold = bold
    r.font.color.rgb = color or DARK
    set_eastasia(r)
    return p


def bullet(parts, space_after=0.5):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.left_indent = Cm(0.3)
    pf.first_line_indent = Cm(-0.3)
    r = p.add_run("• ")
    r.font.size = Pt(BODY)
    r.font.color.rgb = DARK
    set_eastasia(r)
    mixed(p, parts)
    return p


def heading(text, space_before=3):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(1.5)
    r = p.add_run(text)
    r.font.size = Pt(10)
    r.bold = True
    r.font.color.rgb = BLUE
    set_eastasia(r)
    pPr = p._p.get_or_add_pPr()
    bd = pPr.makeelement(qn("w:pBdr"), {})
    bottom = pPr.makeelement(qn("w:bottom"), {qn("w:val"): "single", qn("w:sz"): "6",
                                              qn("w:space"): "1", qn("w:color"): "1A5FB4"})
    bd.append(bottom)
    pPr.append(bd)


# ============ 抬头 ============
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(0)
r = p.add_run("黄浩锋"); r.font.size = Pt(15); r.bold = True; r.font.color.rgb = DARK; set_eastasia(r)

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(1)
mixed(p, [("男 | 15989527280 | 1792472354@qq.com | ", False, GRAY),
          ("github.com/hhfyx/industrial-ai-platform", True, BLUE),
          (" | 求职意向：AI 应用开发实习生 | 期望城市：广州", False, GRAY)])

# ============ 教育 ============
heading("教育经历")
p = para(space_after=0)
mixed(p, [("广州城市理工学院", True, None), ("    本科 · 人工智能    2023 – 2027", False, GRAY)])

# ============ 技能 ============
heading("专业技能")
skills = [
    ("LLM 应用：", "RAG 全链路——切分、混合检索、精排、压缩；LangGraph Agent——工具调用、检查点、人机中断；评测体系——黄金集防泄漏、Recall@k / MRR、LLM-as-judge + 人工校准 30/30；Prompt 工程、SSE 流式"),
    ("工程：", "Python——FastAPI、asyncio、Pydantic v2；MySQL——SQLAlchemy 2.0、Alembic、事务与乐观锁；Docker 含 GPU 直通；GitHub Actions；pytest 1300+ 条离线测试"),
    ("算法：", "PyTorch、SD+LoRA 微调、YOLO、OpenCV"),
]
for label, body in skills:
    bullet([(label, True, None), (body, False, None)])

# ============ 项目一 ============
heading("项目经历")
p = para(space_after=0)
mixed(p, [("项目一｜文档知识库 RAG 平台：检索 → Agent → 可观测 · 个人项目", True, None)])
p = para(space_after=0)
mixed(p, [("github.com/hhfyx/industrial-ai-platform", True, BLUE)])
p = para(space_after=1.5)
mixed(p, [("Python / FastAPI / LangGraph / bge / ChromaDB / Docker · 公网 demo 在线，免费层降级字面检索且降级显式可见", False, GRAY)])

para("从零构建切分、混合检索、引用生成的完整链路并服务化，覆盖鉴权、限流、缓存与双容器编排，自建评测体系量化每一步。",
     space_after=1)

bullet([("评测体系：", True, None),
        ("自建 120 条黄金集——80 事实、20 多跳、20 拒答，题源来自 13 份未参与调参的文档，机器门禁防跨集泄漏；Recall@5 56.0%、MRR 0.427，judge 三档校准一致率 30/30；答案正确率随检索命中层数 67%→38%→15% 单调衰减，检索质量决定生成上限", False, None)])
bullet([("负结果如实入报告：", True, None),
        ("精排 +17.5pp 但延迟 147 倍，不显著即不上线；混合检索 −5pp，RRF 挤掉独有召回；拒答分数区间完全重叠，据此判定调参不可修，转向二段判定", False, None)])
bullet([("Agent 层：", True, None),
        ("LangGraph 四工具 + 检查点 + 人机中断，副作用恰好一次；20 任务评测 5 轮成功率 70%–100%，降级计入失败；不可完成任务编造 0 例，答案数值回查工具返回，轨迹可复跑", False, None)])
bullet([("可观测：", True, None),
        ("trace 三级埋点，单请求 12 个 span；逐笔成本账本 + 聚合对账——对账抓出自己漏埋的 LLM 入口，成本低估 45%；MCP Server 五道闸路径安全，5 条攻击全拒", False, None)])

# ============ 项目二 ============
p = para(space_after=0, space_before=2)
mixed(p, [("项目二｜缺陷图像生成服务：SD+LoRA · 异步队列 · 个人项目", True, None)])
p = para(space_after=0)
mixed(p, [("github.com/hhfyx/industrial-ai-platform/blob/main/docs/README-generator.md", True, BLUE)])
p = para(space_after=1.5)
mixed(p, [("Python / FastAPI / diffusers / PEFT / Docker GPU 直通", False, GRAY)])

para("缺陷检测数据集增广：YOLO 裁剪 3,323 张、6 类切片，SD1.5 加每类一个 LoRA 适配器，只训练 2.18M 参数、基座冻结，异步队列单 worker 保护 GPU。",
     space_after=1)

bullet([("训练提速 126×：", True, None),
        ("每步在线 VAE 编码改为一次性预计算 latent 落盘缓存，单步 39s→0.31s，显存 13GB→7GB，训练卡实测", False, None)])
bullet([("推理实测：", True, None),
        ("RTX 4050 6GB、FP16 下单张 512² 约 5.0s，两轮 4.87–5.00s；显存 2.81GB；batch=4 吞吐反降 15–19%，反直觉结论两轮复现，6GB 卡上 batch 化不划算", False, None)])
bullet([("异步化：", True, None),
        ("提交接口 P50 4.2ms，202 + task_id + 轮询，对比同步阻塞约 5s——架构收益而非 GPU 加速；参数级幂等复用，重复请求 0 GPU 占用；GPU 镜像 12.8GB，权重 0 字节进镜像，docker history 逐层验证", False, None)])

# ============ 项目三 ============
p = para(space_after=0, space_before=2)
mixed(p, [("项目三｜企业业务模块：合同审批 · 个人项目", True, None)])
p = para(space_after=0)
mixed(p, [("github.com/hhfyx/industrial-ai-platform/blob/main/docs/module-contract-approval.md", True, BLUE)])
p = para(space_after=1.5)
mixed(p, [("MySQL 8 / SQLAlchemy 2.0 / Alembic / FastAPI", False, GRAY)])

bullet([("5 表建模：", True, None),
        ("客户、合同、审批节点、审计日志、幂等键；金额用「分」存整数，version 乐观锁，审计日志只 INSERT；5 态状态机纯函数实现，非法迁移当场 409；金额阈值路由审批链，10 万以上两级审批", False, None)])
bullet([("并发正确性实测：", True, None),
        ("单事务改三表 + expected_version 校验——20 并发 approve 恰好 1 成功、19 个 409、日志只多 1 条，MySQL 活体测试；报表按月与客户 GROUP BY 聚合，CSV 导出带 UTF-8 BOM，Excel 直接打开不乱码", False, None)])

out = r"D:\git\网页\industrial-ai-platform\docs\resume-final-w14.docx"
doc.save(out)
print("saved:", out)
