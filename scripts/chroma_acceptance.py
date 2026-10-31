"""Chroma 持久化验收：100 条真实文本入库 + 10 条已知答案的 top3 检索。

**这个脚本验的是"能不能用"，不是"效果好不好"**（任务明确说本周不做量化）。
所以判断标准是**人工**的：每条问题，看 top3 里有没有对应的原文。

## 为什么用 bge 而不是 HashingEmbedder

`HashingEmbedder` 只有字面相似。用它做"人工判断命中"，判断的是字面重叠而不是语义 ——
那在真实使用里几乎没有价值（用户不会用原文的词提问）。
所以这个脚本强制用 `bge-small-zh`，缺依赖时直接退出并说明。

## 两件必须验证的事

1. **重启后数据还在**：入完 100 条，`del` 掉 store，新建 `PersistentClient` 重开，
   `count()` 必须还是 100。这是"W4 最高频的数据怎么丢了"的直接反证。
2. **where 过滤真的生效**：按 `source` 与按 `page` 区间各查一次，
   断言返回的每一条都满足过滤条件。

运行：

    uv run python -X utf8 scripts/chroma_acceptance.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 语料：5 份文档 × 20 条条款 = 100 块。
#
# **第一版设计踩了坑，这里保留教训**：最初让 5 份文档用**完全相同**的 20 条条款，
# 结果 top3 全被同一段文本占满（内容一样、只是 source 不同，分数完全相同），
# 10 条问题里有 2 条因此"未命中"——但那次未命中**不是检索失败**，
# 是语料重复把 top_k 名额吃掉了。
#
# 这是 RAG 里的真实陷阱：**当知识库里有几份近似文档（不同版本的合同、
# 同一条款的多个副本），top_k 会被同一段内容占满**，看起来"检索到了"，
# 实际上下文窗口被浪费在重复内容上。所以本脚本的语料必须让每份文档内容不同。
DOCUMENTS: list[tuple[str, str]] = [
    ("合同A_技术服务.pdf", "付款与验收"),
    ("合同B_运维服务.pdf", "违约与质保"),
    ("合同C_设备采购.pdf", "采购与交付"),
    ("合同D_软件开发.pdf", "知识产权与变更"),
    ("合同E_保密协议.pdf", "保密与争议"),
]

# 5 个主题域，各 20 条。**每条都是不同的文本**，100 条无重复。
_CLAUSES_BY_DOMAIN: dict[str, list[str]] = {
    "付款与验收": [
        "第五条 付款计划：甲方按里程碑分期支付合同价款，验收合格后十五个工作日内付款。",
        "第七条 验收标准：每一期交付物须经甲方书面确认验收合格，方可进入付款流程。",
        "付款比例：预付款百分之三十，验收款百分之六十，质保金百分之十。",
        "发票要求：乙方须开具增值税专用发票，税率百分之六。",
        "付款方式：银行转账至合同约定的对公账户，不接受第三方代收。",
        "逾期付款：甲方逾期付款的，按日万分之三支付逾期利息。",
        "验收流程：乙方提交验收申请后五个工作日内甲方组织验收。",
        "验收整改：验收不通过的，乙方须在十个工作日内完成整改并重新提交。",
        "部分验收：经甲方同意可就已完成部分单独验收并分期付款。",
        "付款前提：每笔付款以乙方提交完整交付文档为前提条件。",
        "账户变更：一方变更收款账户须提前五个工作日书面通知。",
        "结算依据：以双方签字的验收单作为结算依据。",
        "质保金退还：质保期届满且无质量问题的，三十日内无息退还。",
        "费用调整：税率政策变化时，合同价款按不含税额重新计算。",
        "预算上限：本合同总价不含甲方另行采购的硬件费用。",
        "付款审批：甲方内部审批流程不计入付款期限。",
        "对账要求：乙方须于每月末提交当月对账单。",
        "交付文档：含用户手册、部署手册、测试报告各一份。",
        "预付款保函：预付款超过五十万元的，乙方须提供等额保函。",
        "尾款条件：尾款支付以系统上线运行满三十日为前提。",
    ],
    "违约与质保": [
        "第六条 违约责任：乙方逾期交付的，按日万分之五支付违约金，累计不超过合同总额的百分之十。",
        "第十三条 质保期：系统验收合格之日起 12 个月，质保期内免费修复缺陷。",
        "终止条件：连续两次里程碑延期超过三十日的，甲方有权单方解除合同。",
        "质量瑕疵：修复后仍不满足验收标准的，甲方可要求更换或退款。",
        "响应时限：质保期内故障响应不超过四小时，重大故障不超过一小时。",
        "免责情形：因甲方提供的环境不满足要求导致的故障不属质保范围。",
        "赔偿上限：任何一方的赔偿责任不超过合同总额。",
        "违约金抵扣：违约金可直接从应付款项中扣除。",
        "延保条款：质保期届满后可续签年度维保，费用为合同额的百分之八。",
        "备件供应：乙方承诺停产后继续供应备件不少于五年。",
        "缺陷分级：缺陷分为致命、严重、一般三级，分别对应不同修复时限。",
        "复现要求：甲方报障须提供可复现的操作步骤。",
        "第三方软件：第三方软件的缺陷由乙方协助向原厂报修。",
        "升级义务：质保期内乙方免费提供安全补丁升级。",
        "现场支持：重大故障时乙方须派工程师到场，四十八小时内到位。",
        "验收后缺陷：验收后发现的隐蔽缺陷仍由乙方负责修复。",
        "服务中断：单次服务中断超过八小时的，按日顺延质保期。",
        "责任划分：因甲方违规操作导致的损坏由甲方承担。",
        "不可抗力：因地震、洪水、疫情等不可抗力导致延期，双方互不承担违约责任。",
        "履约保证金：乙方须缴纳合同额百分之五的履约保证金。",
    ],
    "采购与交付": [
        "第十二条 交付地点：杭州市余杭区文一西路 969 号，乙方负责运输及现场安装调试。",
        "附件一 设备清单：工业网关 20 台，型号 GW-300，单价 3800 元。",
        "到货验收：设备到货后三个工作日内双方共同开箱验货。",
        "包装要求：设备须采用原厂包装，运输途中的损坏由乙方承担。",
        "运输保险：乙方须为运输货物投保，保额不低于货值。",
        "安装条件：甲方须在设备到货前完成机房供电与网络布线。",
        "调试标准：安装完成后须连续运行七十二小时无故障方视为调试完成。",
        "备件清单：随机提供易损件一套，含电源模块与风扇。",
        "序列号登记：设备序列号须在交付清单中逐台登记。",
        "装箱单据：每箱须附装箱单、合格证与保修卡。",
        "仓储费用：甲方原因导致延迟收货的，仓储费用由甲方承担。",
        "风险转移：货物毁损风险自甲方签收之时转移。",
        "型号变更：未经甲方书面同意，乙方不得变更设备型号。",
        "替代方案：原型号停产的，乙方须提供性能不低于原型号的替代品。",
        "计量校验：仪器类设备须提供第三方计量检定证书。",
        "培训计划：为甲方运维人员提供 3 天培训，含上机实操与考核。",
        "文档交付：设备随机文档须为简体中文。",
        "现场条件：安装现场须具备接地电阻小于四欧姆的接地条件。",
        "搬迁支持：设备搬迁由乙方提供有偿技术支持。",
        "回收义务：设备报废后乙方须提供合规回收渠道。",
    ],
    "知识产权与变更": [
        "第八条 知识产权：本项目产生的软件著作权归甲方所有，乙方保留通用组件的复用权。",
        "第四条 变更管理：任何需求变更须经双方书面确认，涉及费用另议。",
        "第十五条 数据归属：设备运行数据归甲方所有，乙方仅可在脱敏后用于模型优化。",
        "开源合规：乙方使用的开源组件须符合其许可证要求并提供清单。",
        "第三方侵权：因乙方交付物侵犯第三方权利的，由乙方承担全部责任。",
        "商标使用：未经对方书面许可，任何一方不得使用对方商标。",
        "变更评估：变更请求须在五个工作日内完成影响评估。",
        "变更上限：累计变更金额超过合同额百分之二十的，须重新签订补充协议。",
        "模型权属：乙方预训练的通用模型权重仍归乙方所有。",
        "衍生成果：基于甲方数据训练的模型归甲方独占使用。",
        "源代管：核心模块源代码可托管至双方认可的第三方平台。",
        "交付形式：软件以可执行程序加部署脚本形式交付。",
        "反向工程：甲方不得对交付软件进行反向工程。",
        "版本控制：乙方须将交付代码纳入版本管理并保留提交记录。",
        "变更冻结点：上线前三十日进入变更冻结期。",
        "紧急变更：生产环境故障的紧急变更可先执行后补流程。",
        "文档同步：每次变更须同步更新相关技术文档。",
        "验收基准：变更后的验收以补充协议约定的标准为准。",
        "人员变更：乙方更换核心成员须提前通知并完成交接。",
        "知识转移：项目结束前乙方须完成不少于两轮知识转移。",
    ],
    "保密与争议": [
        "第九条 保密义务：双方对合同履行中知悉的商业秘密负有保密义务，保密期三年。",
        "第十一条 争议解决：协商不成的，提交杭州仲裁委员会按其规则仲裁。",
        "保密范围：含技术方案、报价、客户名单与未公开的经营数据。",
        "保密例外：已公开信息、独立开发获得的信息不受保密条款约束。",
        "员工约束：双方须使其员工及分包方遵守同等保密义务。",
        "保密期限：合同终止后保密义务继续有效三年。",
        "资料返还：合同终止后三十日内返还或销毁全部保密资料。",
        "泄露通知：发现泄露后二十四小时内书面通知对方。",
        "适用法律：本合同适用中华人民共和国法律。",
        "仲裁语言：仲裁程序以中文进行。",
        "仲裁费用：仲裁费用由败诉方承担。",
        "诉讼例外：涉及知识产权侵权的可向有管辖权的法院起诉。",
        "继续履行：争议期间除争议部分外合同继续履行。",
        "送达地址：双方确认的送达地址适用于仲裁文书送达。",
        "审计权：甲方有权每年一次对乙方相关记录进行审计。",
        "宣传限制：未经对方同意不得对外宣传合作关系。",
        "数据跨境：不得将甲方数据传输至境外服务器。",
        "安全事件：发生数据安全事件须在两小时内报告。",
        "分级授权：按最小必要原则授予数据访问权限。",
        "合规要求：双方须遵守网络安全法与数据安全法的相关规定。",
    ],
}

# 10 条**已知答案**的问题：(问题, 期望命中的原文片段, 期望来自哪份文档)
#
# **每条 gold 都必须先验证"它在语料里真的存在"** —— 这不是形式主义：
# 第一版我写了 `("要派几个工程师到现场", "2 名工程师", ...)`，
# 而那 20 条条款里根本没有 "2 名工程师"（那是周三 A/B 集里的句子）。
# 结果这条"未命中"被当成了检索缺陷，实际是**测试数据本身错了**。
# 所以脚本开头加了 `_assert_gold_exists()`，gold 不在语料里就直接退出。
#
# 另外两条"未命中"是**真的检索失败**（人工判断后确认）：
# - "验收不合格怎么办"：top1 命中了"验收不通过…十个工作日内完成整改"，
#   语义上其实**答对了**，只是没命中我指定的那一段。这类要人工看内容，
#   不能只看字符串匹配 —— 见 `notes` 字段。
# - "设备送到什么地址"：top3 全是"设备清单/到货验收/序列号登记"，
#   而"文一西路 969 号"那条（交付地点）没进 top3。这是真实的检索失败。
KNOWN_ANSWERS: list[tuple[str, str, str]] = [
    ("违约了要赔多少钱", "按日万分之五支付违约金", "合同B_运维服务.pdf"),
    ("什么时候付钱", "十五个工作日内付款", "合同A_技术服务.pdf"),
    ("验收不合格怎么办", "书面确认验收合格", "合同A_技术服务.pdf"),
    ("软件的著作权归谁", "软件著作权归甲方所有", "合同D_软件开发.pdf"),
    ("保密要保密多久", "保密期三年", "合同E_保密协议.pdf"),
    ("地震导致延期谁负责", "互不承担违约责任", "合同B_运维服务.pdf"),
    ("打官司去哪里仲裁", "杭州仲裁委员会", "合同E_保密协议.pdf"),
    ("设备送到什么地址", "文一西路 969 号", "合同C_设备采购.pdf"),
    ("保修期多长", "12 个月", "合同B_运维服务.pdf"),
    # 原来这里写的是 "2 名工程师"，语料里**不存在**那句 —— 修正为真实存在的条款
    ("工程师多久能到现场", "四十八小时内到位", "合同B_运维服务.pdf"),
]


def build_corpus() -> list[tuple[str, str, int, int]]:
    """生成 100 条 (text, source, page, chunk_index)。

    **每份文档 20 条互不相同的条款**（见 `_CLAUSES_BY_DOMAIN` 的说明：
    第一版让 5 份文档内容相同，top3 被重复内容占满，导致假"未命中"）。
    页码按 4 条一页均分，一份文档 5 页。
    """
    rows: list[tuple[str, str, int, int]] = []
    for source, domain in DOCUMENTS:
        clauses = _CLAUSES_BY_DOMAIN[domain]
        for index, clause in enumerate(clauses):
            page = 1 + index // 4  # 每 4 条一页，1..5
            rows.append((clause, source, page, index))
    return rows


def assert_gold_exists() -> None:
    """**开跑前先验 gold 在语料里存在**，否则直接退出。

    这条检查是踩坑后加的：gold 写错（拼错、抄了别的数据集的句子）时，
    "未命中"看起来完全像检索失败，会让人去调 top_k、换模型 ——
    时间全花在一个不存在的目标上。

    **测试数据本身出错，是最贵的错**：它不会报错，只会让所有结论失真。
    """
    rows = build_corpus()
    missing: list[str] = []
    wrong_source: list[str] = []
    for question, expected, expected_source in KNOWN_ANSWERS:
        in_corpus = [row for row in rows if expected in row[0]]
        if not in_corpus:
            missing.append(f"{question}（期望片段 {expected!r} 不在任何条款里）")
        elif not any(row[1] == expected_source for row in in_corpus):
            actual = sorted({row[1] for row in in_corpus})
            wrong_source.append(f"{question}（期望 {expected_source}，实际在 {actual}）")
    if missing or wrong_source:
        for line in missing + wrong_source:
            print(f"  ✗ gold 错误：{line}", file=sys.stderr)
        raise SystemExit("gold 与语料不同步，先修测试数据再跑")


def main() -> int:
    try:
        from app.rag.embedding import get_embedder
        from app.rag.schema import Chunk
        from app.rag.stores.chroma import ChromaStore
    except ImportError as exc:
        print(f"缺少依赖：{exc}\n先运行：uv sync", file=sys.stderr)
        return 1

    # **先验数据，再跑检索**：gold 写错时，"未命中"看起来和检索失败一模一样。
    assert_gold_exists()

    embedder = get_embedder("bge-small-zh")

    store_path = PROJECT_ROOT / "data" / "chroma_acceptance"
    # **每次从干净状态开始**：库里若留着上次的 100 条，这次的 upsert 会覆盖它们，
    # count 仍然是 100 —— 那样"重复入库不翻倍"和"持久化生效"两件事会混在一起，
    # 无法判断哪一条性质被验到了。
    shutil.rmtree(store_path, ignore_errors=True)

    store = ChromaStore(store_path, collection_name="acceptance")
    rows = build_corpus()
    print(f"语料：{len(rows)} 条，来自 {len(DOCUMENTS)} 份文档")

    chunks = [
        Chunk(text=text, page=page, source=source, index=index)
        for text, source, page, index in rows
    ]
    # **批量向量化**：一次 encode 100 条，而不是 100 次单条调用。
    # 实测 batch=64 比 batch=1 快 4 倍（见 docs/embedding.md），
    # 100 条时是 0.3 秒 vs 1.4 秒。
    vectors = embedder.embed([c.text for c in chunks])
    stored = store.upsert(chunks, vectors)
    print(f"入库：{stored} 条，库内共 {store.count()} 条")
    print(f"落盘目录：{store_path}（{_dir_size(store_path):.1f} MB）")

    # ---------------------------------------------------------- ① 持久化验证
    del store
    reopened = ChromaStore(store_path, collection_name="acceptance")
    persisted = reopened.count()
    print(f"\n① 重启后 count = {persisted}（应为 {len(rows)}）")
    assert persisted == len(rows), f"持久化失败：{persisted} != {len(rows)}"

    # ---------------------------------------------------------- ② where 过滤
    probe = embedder.embed_query(["违约金怎么算"])[0]
    by_source = reopened.search(probe, top_k=20, where={"source": "合同B_运维服务.pdf"})
    assert by_source, "按 source 过滤没命中任何东西"
    assert all(h.chunk.source == "合同B_运维服务.pdf" for h in by_source), (
        "按 source 过滤返回了别的文档"
    )
    print(f"② where source=合同B：命中 {len(by_source)} 条，全部来自该文档 ✓")

    in_pages = reopened.search_in_pages(probe, page_from=2, page_to=3, top_k=20)
    assert in_pages, "按页码区间过滤没命中"
    assert all(2 <= (h.chunk.page or 0) <= 3 for h in in_pages), "页码区间过滤返回了区间外的块"
    print(f"   search_in_pages 2..3：命中 {len(in_pages)} 条，页码全部在区间内 ✓")

    # ---------------------------------------------------------- ③ 10 条已知答案
    print("\n③ 10 条已知答案的 top3（人工判断）")
    print(f"   {'问题':22} {'top1':14} {'top2':14} {'top3':14}  命中位置  文档对")
    hits_at = {1: 0, 2: 0, 3: 0, "miss": 0}
    details: list[dict] = []
    for question, expected, expected_source in KNOWN_ANSWERS:
        query_vector = embedder.embed_query([question])[0]
        hits = reopened.search(query_vector, top_k=3)
        position = "未命中"
        for rank, hit in enumerate(hits, start=1):
            if expected in hit.chunk.text:
                position = f"top{rank}"
                hits_at[rank] += 1
                break
        else:
            hits_at["miss"] += 1
        # 判"文档对不对"：只看内容会自动放过"内容相同但文档错"的情况
        source_ok = "✓" if hits and hits[0].chunk.source == expected_source else "✗"
        locs = [f"{h.chunk.source[:4]}p{h.chunk.page}" for h in hits]
        print(f"   {question:22} {locs[0]:14} {locs[1]:14} {locs[2]:14}  {position:8}  {source_ok}")
        details.append(
            {
                "question": question,
                "expected": expected,
                "expected_source": expected_source,
                "position": position,
                "top1_source_correct": hits[0].chunk.source == expected_source if hits else False,
                "top3": [
                    {
                        "score": round(h.score, 4),
                        "source": h.chunk.source,
                        "page": h.chunk.page,
                        "text": h.chunk.text[:30],
                    }
                    for h in hits
                ],
            }
        )

    total = len(KNOWN_ANSWERS)
    source_ok_count = sum(1 for d in details if d["top1_source_correct"])
    print(
        f"\n   命中分布：top1={hits_at[1]}  top2={hits_at[2]}  top3={hits_at[3]}  "
        f"未命中={hits_at['miss']}（共 {total} 条）"
    )
    print(f"   top3 内命中率：{(total - hits_at['miss']) / total:.0%}")
    print(f"   top1 文档正确：{source_ok_count}/{total}")

    payload = {
        "meta": {
            "vectors": len(rows),
            "documents": len(DOCUMENTS),
            "embedder": embedder.name,
            "dimension": embedder.dimension,
            "collection": "acceptance",
            "distance_space": "cosine",
            "note": "本周不做量化，命中位置为人工判断的参考记录",
            "corpus_design": (
                "5 份文档 × 20 条互不相同的条款。第一版让 5 份内容相同，"
                "top3 被重复内容占满导致假未命中 —— 那是语料缺陷不是检索失败，"
                "教训保留在脚本注释里"
            ),
        },
        "persistence": {"count_after_reopen": persisted},
        "where_filter": {
            "by_source": len(by_source),
            "page_range_2_3": len(in_pages),
        },
        "known_answers": {
            "total": total,
            "top1": hits_at[1],
            "top2": hits_at[2],
            "top3": hits_at[3],
            "miss": hits_at["miss"],
            "top1_source_correct": source_ok_count,
            "details": details,
        },
    }
    out = PROJECT_ROOT / "scripts" / "chroma_acceptance.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写入 {out.relative_to(PROJECT_ROOT)}")

    # 清理：验收库不进仓库（100 条 × 512 维 ≈ 200 KB，但不该进 git）
    shutil.rmtree(store_path, ignore_errors=True)
    return 0


def _dir_size(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024 / 1024


if __name__ == "__main__":
    raise SystemExit(main())
