"""query 侧指令前缀 A/B：加 vs 不加，用**真实链路**测，不用手工脚本。

**为什么这个脚本要跑真实链路（`RagPipeline` + `BGEEmbedder`）**：
手工脚本里我可以随手 `normalize_embeddings=True`，但那证明不了**生产代码**是这么干的。
这个脚本走 `Retriever.retrieve()` → `embed_query()`，所以它测的是
"跑起来会是什么结果"，不是"理论上应该是什么结果"。

## 输出什么

两个查询集（easy / hard）× 两种模式（不加前缀 / 加前缀）的 hit@1、hit@3、MRR，
以及逐条 rank 变化。

- **easy 集**：查询词与原文高度重叠 —— 考字面命中，前缀加不加都该赢
- **hard 集**：换说法、几乎不共享关键词 —— 考语义，**前缀的收益应该主要体现在这里**

如果 hard 集前缀加不加没差别，说明前缀没起作用；如果 easy 集加了反而变差，
说明前缀在"字面匹配"场景是噪声。两个结果都是信息。

运行：

    uv run python -X utf8 scripts/bge_prefix_ab.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 语料：20 段合同条款。长度接近真实切块（30~60 字），不是"一句话样本"。
PASSAGES: list[str] = [
    "第五条 付款计划：甲方按下列里程碑分期支付合同价款，验收合格后十五个工作日内付款。",
    "第六条 违约责任：乙方逾期交付的，按日万分之五支付违约金，累计不超过合同总额的百分之十。",
    "第七条 验收标准：每一期交付物须经甲方书面确认验收合格，方可进入付款流程。",
    "第八条 知识产权：本项目产生的软件著作权归甲方所有，乙方保留通用组件的复用权。",
    "第九条 保密义务：双方对合同履行中知悉的商业秘密负有保密义务，保密期三年。",
    "第十条 不可抗力：因地震、洪水、疫情等不可抗力导致延期，双方互不承担违约责任。",
    "第十一条 争议解决：协商不成的，提交杭州仲裁委员会按其规则仲裁。",
    "第十二条 交付地点：杭州市余杭区文一西路 969 号，乙方负责运输及现场安装调试。",
    "第十三条 质保期：系统验收合格之日起 12 个月，质保期内免费修复缺陷。",
    "第十四条 人员安排：乙方须派驻不少于 2 名工程师现场支持，驻场期与实施期一致。",
    "附件一 设备清单：工业网关 20 台，型号 GW-300，单价 3800 元。",
    "附件二 培训计划：为甲方运维人员提供 3 天培训，含上机实操与考核。",
    "第一条 服务范围：乙方为甲方提供工业设备预测性维护系统的开发与部署服务。",
    "第二条 合同金额：合同总价人民币 128 万元整，含税。",
    "第三条 工期：自合同签订之日起 120 个自然日内完成全部交付。",
    "第四条 变更管理：任何需求变更须经双方书面确认，涉及费用另议。",
    "第十五条 数据归属：设备运行数据归甲方所有，乙方仅可在脱敏后用于模型优化。",
    "第十六条 转包限制：乙方不得将本项目主体工作转包第三方，分包须经甲方同意。",
    "补充说明：本合同一式两份，双方各执一份，自双方法定代表人或授权代表签字盖章之日起生效。",
    "风险提示：若乙方连续两次里程碑延期超过 30 日，甲方有权单方解除合同。",
]

# easy：查询词与原文高度重叠。"违约金" vs "违约金"。
EASY: list[tuple[str, str]] = [
    ("违约金怎么算", "按日万分之五"),
    ("什么时候付款", "十五个工作日内付款"),
    ("验收不合格怎么办", "书面确认验收合格"),
    ("软件著作权归谁", "著作权归甲方所有"),
    ("保密期多长", "保密期三年"),
    ("地震导致延期算违约吗", "互不承担违约责任"),
    ("出了纠纷去哪仲裁", "杭州仲裁委员会"),
    ("设备送到哪里", "文一西路 969 号"),
    ("质保多久", "12 个月"),
    ("要派几个工程师", "不少于 2 名工程师"),
    ("网关多少钱", "3800 元"),
    ("培训几天", "3 天培训"),
    ("合同总价多少", "128 万元"),
    ("工期多长时间", "120 个自然日"),
    ("数据归谁", "设备运行数据归甲方"),
    ("能不能转包", "不得将本项目主体工作转包"),
    ("延期太久能解约吗", "单方解除合同"),
    ("需求变了怎么处理", "需求变更须经双方书面确认"),
]

# hard：**换说法**，与目标 passage 几乎不共享关键词，但与某些干扰项可能重叠。
HARD: list[tuple[str, str]] = [
    ("钱什么时候能到账", "十五个工作日内付款"),
    ("对方拖了活儿要赔多少", "按日万分之五支付违约金"),
    ("东西坏了谁来修", "质保期内免费修复缺陷"),
    ("出了争执找谁评理", "杭州仲裁委员会"),
    ("谁能主张代码的所有权", "软件著作权归甲方所有"),
    ("中途要加功能怎么办", "需求变更须经双方书面确认"),
    ("工程队要在现场待多久", "不少于 2 名工程师现场支持"),
    ("机器放在哪幢楼", "文一西路 969 号"),
    ("换人做行不行", "不得将本项目主体工作转包第三方"),
    ("数据能不能喂给模型", "脱敏后用于模型优化"),
    ("总花销", "128 万元整"),
    ("停水停电能不能免责", "因地震、洪水、疫情等不可抗力"),
    ("甲方可以随时叫停吗", "甲方有权单方解除合同"),
    ("这个活儿交给谁干", "预测性维护系统的开发与部署"),
]


def locate(needle: str) -> int:
    """用唯一子串定位 gold passage。

    **用子串而不是人工写编号**：手写 `[3, 7, 12]` 这种下标，
    只要语料顺序改一次就全错，而错误会表现成"模型效果变差了"。
    子串定位找不到就抛 —— 那说明语料和查询集不同步，必须先修。
    """
    for index, passage in enumerate(PASSAGES):
        if needle in passage:
            return index
    raise AssertionError(f"语料里找不到 {needle!r}，查询集与语料不同步")


def evaluate(pipeline, pairs: list[tuple[str, str]], with_instruction: bool) -> dict:
    """跑一遍检索，返回 hit@1 / hit@3 / MRR / 逐条 rank。

    **直接调 `Retriever` 而不是 `pipeline.answer`**：这里只关心检索，
    调模型只会引入 30 秒延迟和一次计费，对"前缀有没有用"毫无帮助。
    """
    embedder = pipeline.embedder
    original = getattr(embedder, "query_instruction", "")
    if not with_instruction:
        # 关掉前缀：这是本脚本唯一"作弊"的地方，改的是 embedder 的配置，
        # 不是检索逻辑 —— 所以两条路径走的仍然是同一份代码。
        embedder.query_instruction = ""
    try:
        ranks: list[int] = []
        rows: list[dict] = []
        for query, needle in pairs:
            gold = locate(needle)
            hits = pipeline.retrieve(query, top_k=len(PASSAGES)).hits
            order = [next(i for i, p in enumerate(PASSAGES) if p == hit.chunk.text) for hit in hits]
            rank = order.index(gold) + 1
            ranks.append(rank)
            rows.append(
                {
                    "query": query,
                    "rank": rank,
                    "gold_score": round(hits[order.index(gold)].score, 4),
                    "top1": PASSAGES[order[0]][:26],
                    "top1_score": round(hits[0].score, 4),
                }
            )
    finally:
        embedder.query_instruction = original

    n = len(ranks)
    return {
        "n": n,
        "hit@1": round(sum(1 for r in ranks if r == 1) / n, 4),
        "hit@3": round(sum(1 for r in ranks if r <= 3) / n, 4),
        "mrr": round(sum(1 / r for r in ranks) / n, 4),
        "ranks": ranks,
        "rows": rows,
    }


def main() -> int:
    try:
        from app.rag.embedding import get_embedder
        from app.rag.pipeline import RagPipeline
    except ImportError as exc:  # pragma: no cover - 依赖缺失是环境问题
        print(f"缺少依赖：{exc}\n先运行：uv sync", file=sys.stderr)
        return 1

    embedder = get_embedder("bge-small-zh")
    pipeline = RagPipeline(embedder=embedder)
    reports = [pipeline.index_text(text, source=f"合同#{i}") for i, text in enumerate(PASSAGES)]
    assert all(r.ok for r in reports), "索引失败"

    results: dict[str, dict] = {}
    for set_name, pairs in (("easy", EASY), ("hard", HARD)):
        for mode, with_instruction in (("no_prefix", False), ("with_prefix", True)):
            results[f"{set_name}/{mode}"] = evaluate(pipeline, pairs, with_instruction)

    print(f"{'集合':6} {'前缀':12} {'hit@1':>7} {'hit@3':>7} {'MRR':>7} {'Δhit@1':>8}")
    for set_name in ("easy", "hard"):
        base = results[f"{set_name}/no_prefix"]
        for mode in ("no_prefix", "with_prefix"):
            row = results[f"{set_name}/{mode}"]
            delta = row["hit@1"] - base["hit@1"]
            print(
                f"{set_name:6} {mode:12} {row['hit@1']:>7.3f} {row['hit@3']:>7.3f} "
                f"{row['mrr']:>7.3f} {delta:>+8.3f}"
            )

    print("\nhard 集逐条（rank 1 = 命中）:")
    for base_row, prefix_row in zip(
        results["hard/no_prefix"]["rows"], results["hard/with_prefix"]["rows"], strict=True
    ):
        flag = ""
        if base_row["rank"] != prefix_row["rank"]:
            flag = " ← 前缀改善" if prefix_row["rank"] < base_row["rank"] else " ← 前缀变差"
        print(
            f"  {base_row['query']:22} 不加={base_row['rank']:>2}  加={prefix_row['rank']:>2}{flag}"
        )

    out = PROJECT_ROOT / "scripts" / "bge_prefix_ab.json"
    payload = {
        "meta": {
            "model": getattr(embedder, "model_name", "unknown"),
            "dimension": embedder.dimension,
            "max_seq_length": getattr(embedder, "max_seq_length", None),
            "instruction": getattr(embedder, "query_instruction", ""),
            "corpus_size": len(PASSAGES),
            "note": "走真实链路（RagPipeline + Retriever.embed_query），不是手工脚本",
            "raw_vectors": False,
        },
        "results": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写入 {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
