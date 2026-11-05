"""W7 事实型黄金集出题管道（2026-12-01 周二）。

## 它解决什么

标注最大的坑不是"出不出题"，是 **quote 手抄必错**（弯引号/全角空格/漏一个字），
README §六坑二整节都在修这个。所以本管道把"写字"和"取证"分开：

- **人写**：question（口语化改写）、key（答案所在行的定位子串）、
  gold_answer、notes（为什么算事实题）
- **脚本取证**：从 loader 清洗后的原文里找到 key 所在行，
  **整行或按窗口切片**作为 quote —— 天然逐字，不可能抄错

## 红线一（题源）

只用 13 份从未出题的文档（FREE_DOCS，与调参集 50+20+5+5 条零交集）。
清单与配额见 eval/README.md §十。

## 校验链

生成后自动跑三道：quote 全文唯一 → rre.check_no_leak（防泄题）→
落盘后由 validate_golden.py 对索引反查。任何一道红了就改人写的部分，
**不改检查**。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_retrieval_eval as rre  # noqa: E402
from retrieval_harness import load_raw_corpus  # noqa: E402

OUT = PROJECT_ROOT / "eval" / "questions-120.jsonl"

QUOTA: dict[str, int] = {
    "prompt-ab.md": 9,
    "deploy-notes.md": 9,
    "why-i-use-langchain.md": 7,
    "prompt-templates.md": 8,
    "system-prompt.md": 8,
    "chunking-boundaries.md": 8,
    "ab-scaffold.md": 5,
    "streamlit-ui.md": 7,
    "structured-output.md": 6,
    "container-networking.md": 4,
    "chunking-predictions.md": 4,
    "release-notes-v0.3.0.md": 3,
    "release-notes-v0.3.1.md": 2,
}
assert sum(QUOTA.values()) == 80

# ---------------------------------------------------------------- 出题表
# 每条：question（口语化）、key（定位子串）、gold_answer（可判定）、notes（为什么是事实题）
ITEMS: list[dict[str, str]] = [
    # ---- prompt-ab.md（9）----
    {
        "id": "f-01",
        "doc": "prompt-ab.md",
        "key": "各混 1 次",
        "question": "四类模板 A/B 里最容易混的是哪两个意图类别，混了几次？",
        "gold_answer": "功能咨询→故障报修；两版各混 1 次",
        "notes": "事实题：混淆矩阵结论唯一，一句原文可判",
    },
    {
        "id": "f-02",
        "doc": "prompt-ab.md",
        "key": "v2 有 7 条超时（70% 失败率）",
        "question": "摘要组换 v2 之后失败情况怎么样？",
        "gold_answer": "7 条超时，失败率 70%",
        "notes": "事实题：实测数字唯一",
    },
    {
        "id": "f-03",
        "doc": "prompt-ab.md",
        "key": "不是 v2 模板写得好",
        "question": "抽取组 v2 大幅胜出，真实原因是什么？",
        "gold_answer": "不是模板好，是 v1 的字段契约对不上",
        "notes": "事实题：诚实归因段一句原文可判",
    },
    {
        "id": "f-04",
        "doc": "prompt-ab.md",
        "key": "任何 <20pp 的差异都不该当结论",
        "question": "10 条样本的 A/B 结果要差到什么程度才算有信号？",
        "gold_answer": "至少差 2 条（20pp）",
        "notes": "事实题：阈值写死，数值唯一",
    },
    {
        "id": "f-05",
        "doc": "prompt-ab.md",
        "key": "反向边界让「功能咨询」更宽",
        "question": "v2 给分类模板加的那条排除性描述，实际效果是什么？",
        "gold_answer": "功能咨询变宽，把提建议也吸了过来",
        "notes": "事实题：归因唯一（cl-07 一条）",
    },
    {
        "id": "f-06",
        "doc": "prompt-ab.md",
        "key": "纳入统计",
        "question": "初版汇总的统计口径错在哪？",
        "gold_answer": "只把成功条目纳入统计，超时多的版本得分虚高",
        "notes": "事实题：口径修正段唯一表述",
    },
    {
        "id": "f-07",
        "doc": "prompt-ab.md",
        "key": "契约不匹配",
        "question": "抽取组 v1 输得那么惨，根因是什么？",
        "gold_answer": "不是模型差，是 v1 schema 契约不匹配",
        "notes": "事实题：一句原文可判",
    },
    {
        "id": "f-08",
        "doc": "prompt-ab.md",
        "key": "规律：v2 倾向于",
        "question": "抽取模板遗留的真实待改项是什么？",
        "gold_answer": "v2 倾向提炼而非照抄，会压缩字段值",
        "notes": "事实题：待改项唯一",
    },
    {
        "id": "f-09",
        "doc": "prompt-ab.md",
        "key": "| **平均得分（全量）** | **0.1333** | **0.7667** | Δ = **+0.6334** |",
        "question": "抽取组两版跑分，把失败也计 0 的那种算法下各拿多少分？",
        "gold_answer": "v1 0.1333 → v2 0.7667（+0.6334）",
        "notes": "事实题：表格数值唯一",
    },
    # ---- deploy-notes.md（9）----
    {
        "id": "f-10",
        "doc": "deploy-notes.md",
        "key": "模型栈实测增量 **528 MB**",
        "question": "免费层为什么跑不了 bge？",
        "gold_answer": "模型栈增量 528 MB，加上其余组件 1 GB 无余量",
        "notes": "事实题：实测数字唯一",
    },
    {
        "id": "f-11",
        "doc": "deploy-notes.md",
        "key": "内存字段含逗号",
        "question": "内存测量踩的第一个坑是什么？",
        "gold_answer": "tasklist CSV 内存字段含逗号，差点把结论做反",
        "notes": "事实题：坑标题唯一",
    },
    {
        "id": "f-12",
        "doc": "deploy-notes.md",
        "key": "正确位置是 `[server]`",
        "question": "上传大小上限配置写在哪才真正生效？",
        "gold_answer": "[server] 段；写 [client] 会被静默忽略",
        "notes": "事实题：正确位置唯一",
    },
    {
        "id": "f-13",
        "doc": "deploy-notes.md",
        "key": "零依赖、零下载、确定能跑",
        "question": "低内存环境的降级方案为什么选 hashing？",
        "gold_answer": "零依赖、零下载、确定能跑",
        "notes": "事实题：选型理由唯一",
    },
    {
        "id": "f-14",
        "doc": "deploy-notes.md",
        "key": "必须放在入口目录",
        "question": "部署用的依赖文件必须放在哪里？",
        "gold_answer": "入口文件所在目录；放根目录会被 uv.lock 抢掉",
        "notes": "事实题：位置唯一（坑 5.1）",
    },
    {
        "id": "f-15",
        "doc": "deploy-notes.md",
        "key": "优先级更高",
        "question": "仓库根目录同时有 requirements.txt 和 uv.lock 时，Cloud 实际读哪个？",
        "gold_answer": "uv.lock（优先级更高）",
        "notes": "事实题：表格判定唯一",
    },
    {
        "id": "f-16",
        "doc": "deploy-notes.md",
        "key": "悄悄降级等于骗用户",
        "question": "嵌入模型降级为什么必须在界面上显示出来？",
        "gold_answer": "悄悄降级等于骗用户：用户以为语义检索，实际只有字面",
        "notes": "事实题：设计原则唯一表述",
    },
    {
        "id": "f-17",
        "doc": "deploy-notes.md",
        "key": "约 5 秒、近乎免费",
        "question": "部署前跑端点自检的成本是多少？",
        "gold_answer": "约 5 秒，只发 3 次最小调用，近乎免费",
        "notes": "事实题：数字唯一",
    },
    {
        "id": "f-18",
        "doc": "deploy-notes.md",
        "key": "根级，不要放进",
        "question": "Streamlit Secrets 里的配置要粘贴到哪一层？",
        "gold_answer": "根级；放进 [section] 不会变成环境变量",
        "notes": "事实题：位置唯一",
    },
    # ---- why-i-use-langchain.md（7）----
    {
        "id": "f-20",
        "doc": "why-i-use-langchain.md",
        "key": "| LangChain `with_structured_output` | **0/10** | **8/10** |",
        "question": "LangChain 的结构化输出在嵌套与扁平 schema 下各成功几条？",
        "gold_answer": "嵌套 0/10；扁平 8/10",
        "notes": "事实题：对照表数值唯一",
    },
    {
        "id": "f-21",
        "doc": "why-i-use-langchain.md",
        "key": "**只支持 `tool_choice=auto` 和 `none`**",
        "question": "本网关对 tool_choice 的支持到什么程度？",
        "gold_answer": "只支持 auto 和 none",
        "notes": "事实题：网关能力边界唯一",
    },
    {
        "id": "f-22",
        "doc": "why-i-use-langchain.md",
        "key": "两个业务字段都答对了",
        "question": "json_mode 路线失败时，模型的输出实际长什么样？",
        "gold_answer": "业务字段都对，但没套规定的 JSON 外壳",
        "notes": "事实题：错误形态唯一",
    },
    {
        "id": "f-23",
        "doc": "why-i-use-langchain.md",
        "key": "网关能力缺失",
        "question": "三层归因里 function_calling 那一层的失败责任在谁？",
        "gold_answer": "网关能力缺失，与框架无关",
        "notes": "事实题：归因表唯一",
    },
    {
        "id": "f-24",
        "doc": "why-i-use-langchain.md",
        "key": "对扁平 schema 能强制执行",
        "question": "扁平 schema 对照实验纠正了什么结论？",
        "gold_answer": "本网关对扁平能强制执行，只是对嵌套不强制",
        "notes": "事实题：修正结论唯一",
    },
    {
        "id": "f-25",
        "doc": "why-i-use-langchain.md",
        "key": "1 条样本足以",
        "question": "从单条样本得出结论这件事，代价是什么？",
        "gold_answer": "基于 1 条样本的归因写错了整个小节",
        "notes": "事实题：教训唯一",
    },
    {
        "id": "f-26",
        "doc": "why-i-use-langchain.md",
        "key": "**如果我今天要做的是「多步骤 agent + 工具调用 + 流式 + 内存」**",
        "question": "什么场景下 LangChain 的 Runnable 层仍有明显优势？",
        "gold_answer": "多步骤 agent + 工具调用 + 流式 + 内存",
        "notes": "事实题：适用场景唯一",
    },
    # ---- prompt-templates.md（8）----
    {
        "id": "f-27",
        "doc": "prompt-templates.md",
        "key": "对外 API 契约",
        "question": "模板的输出 schema 为什么不放 app/schemas/ 目录？",
        "gold_answer": "那里放的是对外 API 契约；模板 schema 是另一回事",
        "notes": "事实题：目录分工唯一",
    },
    {
        "id": "f-28",
        "doc": "prompt-templates.md",
        "key": "加上 ③ 才稳",
        "question": "分类模板防数组输出，靠到第几层才稳？",
        "gold_answer": "三层齐上才稳；只有措辞时约 1/10 出数组",
        "notes": "事实题：实测比例唯一",
    },
    {
        "id": "f-29",
        "doc": "prompt-templates.md",
        "key": "逼着编造",
        "question": "抽取模板的字段类型设计里，None 的意义是什么？",
        "gold_answer": "None 是合法值；写必填 str 会逼模型编造",
        "notes": "事实题：设计关键唯一",
    },
    {
        "id": "f-30",
        "doc": "prompt-templates.md",
        "key": "没有测试的规范等于没有规范",
        "question": "模板头部五要素自述靠什么机制保证不缺失？",
        "gold_answer": "测试逐模板检查五个小标题存在",
        "notes": "事实题：机制唯一",
    },
    {
        "id": "f-31",
        "doc": "prompt-templates.md",
        "key": "新增 `v2.jinja2`（可选 `v2_schema.py`），**不改 v1**",
        "question": "给模板加新版本的正确流程是什么？",
        "gold_answer": "新增 v2.jinja2 与可选 v2_schema.py，不改 v1",
        "notes": "事实题：流程唯一",
    },
    {
        "id": "f-32",
        "doc": "prompt-templates.md",
        "key": "tuple 不是对象",
        "question": "lru_cache 参数冻结踩的坑，根因是什么？",
        "gold_answer": "tuple 没有属性访问；缓存键与渲染参数要分开",
        "notes": "事实题：根因唯一",
    },
    {
        "id": "f-33",
        "doc": "prompt-templates.md",
        "key": "没有一个数字被改写",
        "question": "真实模型验证里，摘要模板的数字保留表现如何？",
        "gold_answer": "数字逐字保留，没有一个被改写成约数",
        "notes": "事实题：实测结论唯一",
    },
    {
        "id": "f-34",
        "doc": "prompt-templates.md",
        "key": "模糊样本给了 0.45",
        "question": "边界模糊的样本在分类模板上的置信度表现如何？",
        "gold_answer": "给 0.45 低分交人工复核；未出现选两个类别",
        "notes": "事实题：数值唯一",
    },
    # ---- system-prompt.md（8）----
    {
        "id": "f-35",
        "doc": "system-prompt.md",
        "key": "唯一变量 = 约束密度",
        "question": "这组 prompt A/B 实验里，两版之间唯一动过的东西是什么？",
        "gold_answer": "约束密度",
        "notes": "事实题：唯一变量唯一",
    },
    {
        "id": "f-36",
        "doc": "system-prompt.md",
        "key": "系统提示字符数",
        "question": "宽松版与严格版的提示词长度各是多少？",
        "gold_answer": "289 vs 664 字符",
        "notes": "事实题：表格数值唯一",
    },
    {
        "id": "f-37",
        "doc": "system-prompt.md",
        "key": "不是推理能力",
        "question": "这组实验为什么把推理档位关掉？",
        "gold_answer": "测的是指令遵循；思维链会干扰格式约束",
        "notes": "事实题：理由唯一",
    },
    {
        "id": "f-38",
        "doc": "system-prompt.md",
        "key": "唯一能区分",
        "question": "四要素里，边界句的特殊作用是什么？",
        "gold_answer": "唯一能区分「任务」和「闲聊」的一句话",
        "notes": "事实题：作用唯一",
    },
    {
        "id": "f-39",
        "doc": "system-prompt.md",
        "key": "硬猜**一个具体值",
        "question": "没有拒答条款时，模糊输入会发生什么？",
        "gold_answer": "模型硬猜一个具体值",
        "notes": "事实题：行为唯一",
    },
    {
        "id": "f-40",
        "doc": "system-prompt.md",
        "key": "显式拒绝率 0.00 → 0.60",
        "question": "四档判定改完之后，两版真正把越界请求挡在门外（合法 JSON 加拒绝话术）的比例各是多少？",
        "gold_answer": "0.00 → 0.60（笼统搪塞 0.40 → 0.00）",
        "notes": "事实题：数值唯一",
    },
    {
        "id": "f-41",
        "doc": "system-prompt.md",
        "key": "零过度拒绝",
        "question": "严格版对正常工单的过度拒绝率是多少？",
        "gold_answer": "0.00，20 条无一被误拒",
        "notes": "事实题：实测数值唯一",
    },
    {
        "id": "f-42",
        "doc": "system-prompt.md",
        "key": "尺子没刻度",
        "question": "判定器返工这件事留下的教训是什么？",
        "gold_answer": "指标定义错了，数据再真也是废的",
        "notes": "事实题：教训唯一",
    },
    # ---- chunking-boundaries.md（8）----
    {
        "id": "f-43",
        "doc": "chunking-boundaries.md",
        "key": '**检索时按章节过滤，生成时告诉用户"在第 1.2 节"**',
        "question": "section_path 存进 metadata 有哪两个实际用途？",
        "gold_answer": "检索按章节过滤；生成时告诉用户在第几节",
        "notes": "事实题：用途两条且唯一",
    },
    {
        "id": "f-44",
        "doc": "chunking-boundaries.md",
        "key": "层级信息只存在于结构里",
        "question": "层级差异为什么不能用切分参数弥补？",
        "gold_answer": "层级只存在于结构里，切完就销毁",
        "notes": "事实题：机制唯一",
    },
    {
        "id": "f-45",
        "doc": "chunking-boundaries.md",
        "key": "判据改成",
        "question": "把「有 heading_level」当成节开头这个错，怎么修的？",
        "gold_answer": "判据改成标题行；level 只用来筛深度",
        "notes": "事实题：修法唯一",
    },
    {
        "id": "f-46",
        "doc": "chunking-boundaries.md",
        "key": "无主孤儿块",
        "question": "表格把节切断后，分隔线会遇到什么问题？",
        "gold_answer": "成了无主孤儿块，不知道属于哪一节",
        "notes": "事实题：后果唯一",
    },
    {
        "id": "f-47",
        "doc": "chunking-boundaries.md",
        "key": "而 `strip()` 恰好吃掉 Python",
        "question": "切分器对代码块做常规 strip 会破坏什么？",
        "gold_answer": "strip 吃掉 Python 唯一的结构性信息（缩进）",
        "notes": "事实题：原因唯一",
    },
    {
        "id": "f-48",
        "doc": "chunking-boundaries.md",
        "key": "2 × 块数",
        "question": "半句率这个指标的分母是怎么定的？",
        "gold_answer": "2 × 块数：每块两个边界都可能错",
        "notes": "事实题：定义唯一",
    },
    {
        "id": "f-49",
        "doc": "chunking-boundaries.md",
        "key": "| **对照：等长硬切** | — | 31 | **33.87%** |",
        "question": "等长硬切和按标点切的半句率各是多少？",
        "gold_answer": "33.87% vs 7.14%，差一个量级",
        "notes": "事实题：对照数值唯一",
    },
    {
        "id": "f-50",
        "doc": "chunking-boundaries.md",
        "key": "命中行 / 候选大字号行的比例",
        "question": "PDF 标题启发式的 confidence 这个数衡量什么？",
        "gold_answer": "命中行/候选大字号行比例，判断判据稳不稳",
        "notes": "事实题：定义唯一",
    },
    # ---- ab-scaffold.md（5）----
    {
        "id": "f-51",
        "doc": "ab-scaffold.md",
        "key": "偷偷改来",
        "question": "A/B 脚手架的 temperature 和 seed 为什么写成常量？",
        "gold_answer": "允许调它，等于允许有人偷偷改来调好看",
        "notes": "事实题：理由唯一",
    },
    {
        "id": "f-52",
        "doc": "ab-scaffold.md",
        "key": "自己编的输入",
        "question": "40 条输入为什么从开发语料抽而不是自己编？",
        "gold_answer": "自己编会不知不觉配合 prompt，那是自证不是测试",
        "notes": "事实题：纪律理由唯一",
    },
    {
        "id": "f-53",
        "doc": "ab-scaffold.md",
        "key": "若 `len(dist) == len(cls)`",
        "question": "类别分布的校验专门防的是什么？",
        "gold_answer": "每类各 1 条的手工凑平衡，直接判失败",
        "notes": "事实题：断言唯一",
    },
    {
        "id": "f-54",
        "doc": "ab-scaffold.md",
        "key": "见过并针对性优化过",
        "question": "prompt 实验集为什么不能拿去当 W7 评测集？",
        "gold_answer": "调参时见过并优化过，评测会虚高",
        "notes": "事实题：边界理由唯一",
    },
    {
        "id": "f-55",
        "doc": "ab-scaffold.md",
        "key": "set_usage_sink",
        "question": "用量统计怎么拿到而不动定版的函数签名？",
        "gold_answer": "set_usage_sink 旁路钩子，默认 None 零开销",
        "notes": "事实题：机制唯一",
    },
    # ---- streamlit-ui.md（7）----
    {
        "id": "f-56",
        "doc": "streamlit-ui.md",
        "key": "检索约 9~11 ms",
        "question": "一次提问的 1.5 秒首字延迟里，检索占了多少？",
        "gold_answer": "9~11 ms，占 0.36%~0.80%，不到 1%",
        "notes": "事实题：实测数值唯一",
    },
    {
        "id": "f-57",
        "doc": "streamlit-ui.md",
        "key": "荒谬的数字",
        "question": "算检索占比时，分母为什么不能用假客户端测出来的延迟？",
        "gold_answer": "模型不可能 29 ms 出字，那是个荒谬的数字",
        "notes": "事实题：理由唯一",
    },
    {
        "id": "f-58",
        "doc": "streamlit-ui.md",
        "key": "编码 300 ms",
        "question": "50 页 PDF 的索引阶段，耗时大头是什么？",
        "gold_answer": "编码 300 ms（50 块 × 约 6 ms/块）",
        "notes": "事实题：数值唯一",
    },
    {
        "id": "f-59",
        "doc": "streamlit-ui.md",
        "key": "不计入请求耗时",
        "question": "冷启动的模型加载算不算进请求耗时？",
        "gold_answer": "不算，11 秒是启动成本",
        "notes": "事实题：口径唯一",
    },
    {
        "id": "f-60",
        "doc": "streamlit-ui.md",
        "key": "检索结果在 TTFT",
        "question": "用户等首字的这几秒里，命中的块是什么时候显示出来的？",
        "gold_answer": "TTFT 之前、调模型之前就上屏，避免空白等待",
        "notes": "事实题：顺序唯一",
    },
    {
        "id": "f-61",
        "doc": "streamlit-ui.md",
        "key": "只列被引用的块",
        "question": "回答下面的来源列表为什么不像别的应用那样把所有召回都贴上去？",
        "gold_answer": "列全部检索结果会让人以为都是依据",
        "notes": "事实题：设计理由唯一",
    },
    {
        "id": "f-62",
        "doc": "streamlit-ui.md",
        "key": "静默返回无法解释的结果",
        "question": "换了嵌入模型之后，重建索引时为什么必须清空向量库？",
        "gold_answer": "维度不同会报错；维度相同更糟，静默返回无法解释的结果",
        "notes": "事实题：后果唯一",
    },
    # ---- structured-output.md（6）----
    {
        "id": "f-63",
        "doc": "structured-output.md",
        "key": "参数接受，但不强制",
        "question": "本网关对 json_schema strict 的真实行为是什么？",
        "gold_answer": "参数接受，但不强制执行",
        "notes": "事实题：能力边界唯一",
    },
    {
        "id": "f-64",
        "doc": "structured-output.md",
        "key": "押在本地 Pydantic 校验",
        "question": "response_format 不可信时，输出正确性靠什么保证？",
        "gold_answer": "本地 Pydantic 校验；json_object 只挡围栏废话",
        "notes": "事实题：方案唯一",
    },
    {
        "id": "f-65",
        "doc": "structured-output.md",
        "key": "手写会和模型漂移",
        "question": "schema 说明为什么由代码生成而不是手写？",
        "gold_answer": "手写会和模型漂移；生成永远同步",
        "notes": "事实题：理由唯一",
    },
    {
        "id": "f-66",
        "doc": "structured-output.md",
        "key": "「你错了，请重试」是废话",
        "question": "修复重试时为什么要回灌报错原文？",
        "gold_answer": "空泛重试模型不知错在哪；原文给字段路径和实际值",
        "notes": "事实题：机制理由唯一",
    },
    {
        "id": "f-67",
        "doc": "structured-output.md",
        "key": "等不起 5 轮",
        "question": "修复重试的上限为什么定在 2 轮？",
        "gold_answer": "每轮真实计费秒级延迟，线上用户等不起 5 轮",
        "notes": "事实题：理由唯一",
    },
    {
        "id": "f-68",
        "doc": "structured-output.md",
        "key": "缓存键必须含 schema",
        "question": "实验缓存的键由哪些字段构成？",
        "gold_answer": "model/prompt/schema_json/temperature/max_tokens 的 sha256",
        "notes": "事实题：构成唯一",
    },
    # ---- container-networking.md（4）----
    {
        "id": "f-69",
        "doc": "container-networking.md",
        "key": "指的是容器自己",
        "question": "容器里写 127.0.0.1 访问宿主服务，实际访问到的是谁？",
        "gold_answer": "容器自己，不是宿主机",
        "notes": "事实题：原因一唯一",
    },
    {
        "id": "f-70",
        "doc": "container-networking.md",
        "key": "鉴权跟着宿主进程的登录态走",
        "question": "本机网关的鉴权依赖什么？",
        "gold_answer": "宿主进程的登录态；容器是另一身份，不被认可",
        "notes": "事实题：原因二唯一",
    },
    {
        "id": "f-71",
        "doc": "container-networking.md",
        "key": "设计要验证的东西",
        "question": "容器里出现降级标记，为什么决定不去修？",
        "gold_answer": "降级路径本来就是设计要验证的东西",
        "notes": "事实题：理由唯一",
    },
    {
        "id": "f-72",
        "doc": "container-networking.md",
        "key": "互相独立",
        "question": "容器里必然降级的两个原因之间是什么关系？",
        "gold_answer": "互相独立：网络可达不够，还要解决鉴权",
        "notes": "事实题：关系唯一",
    },
    # ---- chunking-predictions.md（4）----
    {
        "id": "f-73",
        "doc": "chunking-predictions.md",
        "key": "分隔符分布",
        "question": "递归字符切分策略依赖的是什么信号？",
        "gold_answer": "分隔符分布（段落/句号/逗号），在字符层",
        "notes": "事实题：预判表唯一",
    },
    {
        "id": "f-74",
        "doc": "chunking-predictions.md",
        "key": "标题不承载语义",
        "question": "按文档结构标记来切分的策略，预判会在哪种文档上吃亏？",
        "gold_answer": "标题不承载语义的文档",
        "notes": "事实题：预判唯一",
    },
    {
        "id": "f-75",
        "doc": "chunking-predictions.md",
        "key": "短句堆叠",
        "question": "语义切分策略预判的短板是什么？",
        "gold_answer": "短句堆叠/术语密集的文档，且有两个结构性代价",
        "notes": "事实题：预判唯一",
    },
    {
        "id": "f-76",
        "doc": "chunking-predictions.md",
        "key": "按标题层级（MD）",
        "question": "三种切分策略的预判排序是怎样的？",
        "gold_answer": "标题层级(MD) > 递归字符(PDF) > 语义",
        "notes": "事实题：排序唯一",
    },
    # ---- release-notes-v0.3.0.md（3）----
    {
        "id": "f-77",
        "doc": "release-notes-v0.3.0.md",
        "key": "签名定版",
        "question": "v0.3.0 的核心交付是什么？",
        "gold_answer": "structured_call 签名定版，后续四周共用",
        "notes": "事实题：交付唯一",
    },
    {
        "id": "f-78",
        "doc": "release-notes-v0.3.0.md",
        "key": "327 条测试全离线",
        "question": "v0.3.0 已验证部分的测试规模是多少？",
        "gold_answer": "327 条测试全离线，覆盖率 79%",
        "notes": "事实题：数字唯一",
    },
    {
        "id": "f-79",
        "doc": "release-notes-v0.3.0.md",
        "key": "不是已修复",
        "question": "v0.3.0 的未验证项是怎么标注的？",
        "gold_answer": "容器内 LLM 路径跑不通，标「未验证」而非「已修复」",
        "notes": "事实题：标注唯一",
    },
    # ---- release-notes-v0.3.1.md（2）----
    {
        "id": "f-80",
        "doc": "release-notes-v0.3.1.md",
        "key": "不改任何运行时行为",
        "question": "v0.3.1 相对 v0.3.0 的变更性质是什么？",
        "gold_answer": "补周报遗漏与决策文档，不改运行时行为",
        "notes": "事实题：性质唯一",
    },
    {
        "id": "f-81",
        "doc": "release-notes-v0.3.1.md",
        "key": "前两档数据被覆盖",
        "question": "消融实验分两次跑踩了什么坑？",
        "gold_answer": "每次新建 store，前两档数据被覆盖",
        "notes": "事实题：坑唯一",
    },
]

assert len({it["id"] for it in ITEMS}) == len(ITEMS) == 80


def extract_quote(lines: list[str], key: str, doc: str, item_id: str) -> tuple[str, str]:
    """从行列表里取 key 所在行 → 整行或窗口切片。返回（quote, head12）。"""
    hits = [ln for ln in lines if key in ln]
    if not hits:
        raise SystemExit(f"[{item_id}] key 在 {doc} 找不到：{key!r}")
    if len(hits) > 1:
        raise SystemExit(f"[{item_id}] key 在 {doc} 命中 {len(hits)} 行，需更精确：{key!r}")
    line = hits[0].strip()
    if len(line) <= 80:
        quote = line
    else:
        start = max(0, line.find(key) - 15)
        quote = line[start : start + 80].strip()
    if not 15 <= len(quote) <= 80:
        raise SystemExit(f"[{item_id}] quote 长度 {len(quote)} 越界：{quote!r}")
    return quote, quote[:12]


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    docs = load_raw_corpus()
    lines_by_doc: dict[str, list[str]] = {}
    full_by_doc: dict[str, str] = {}
    for d in docs:
        lines_by_doc.setdefault(d.source, []).extend(d.text.split("\n"))
        full_by_doc[d.source] = full_by_doc.get(d.source, "") + d.text + "\n"

    per_doc = Counter(it["doc"] for it in ITEMS)
    for doc, n in per_doc.items():
        if n != QUOTA[doc]:
            raise SystemExit(f"配额不符：{doc} 实为 {n}，配额 {QUOTA[doc]}")

    # 逐条取证 + quote 全文唯一
    records: list[dict] = []
    problems: list[str] = []
    for it in ITEMS:
        quote, head12 = extract_quote(lines_by_doc[it["doc"]], it["key"], it["doc"], it["id"])
        n = full_by_doc[it["doc"]].count(quote)
        if n != 1:
            problems.append(f"[{it['id']}] quote 在 {it['doc']} 出现 {n} 次")
        records.append(
            {
                "id": it["id"],
                "type": "fact",
                "question": it["question"],
                "anchors": [{"source": it["doc"], "page": None, "head12": head12, "quote": quote}],
                "gold_chunk_ids": [],
                "gold_answer": it["gold_answer"],
                "notes": it["notes"],
            }
        )

    # 泄题检查（问题 vs 自己的 quote）
    gold_items = [
        rre.GoldItem(id=r["id"], type="fact", question=r["question"], anchors=r["anchors"])
        for r in records
    ]
    problems.extend(rre.check_no_leak(gold_items))

    if problems:
        print(f"✗ {len(problems)} 个问题：")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)

    # 轮转交错写盘：不同文档的题穿插，避免按文档成块
    by_doc: dict[str, list[dict]] = {}
    for r in records:
        by_doc.setdefault(r["anchors"][0]["source"], []).append(r)
    interleaved: list[dict] = []
    docs_order = list(by_doc)
    while any(by_doc[d] for d in docs_order):
        for d in docs_order:
            if by_doc[d]:
                interleaved.append(by_doc[d].pop(0))
    OUT.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in interleaved),
        encoding="utf-8",
    )
    print(f"✓ 写出 {len(interleaved)} 条 → {OUT.relative_to(PROJECT_ROOT)}")
    final_counts = Counter(r["anchors"][0]["source"] for r in interleaved)
    print(
        f"  覆盖 {len(final_counts)} 份文档："
        + "、".join(f"{d}({n})" for d, n in final_counts.items())
    )


if __name__ == "__main__":
    main()
