"""黄金集校验：**坏数据当场报错，带病出题跑不进评测**。

## 一句话定位

`python -X utf8 scripts/tasks.py validate-golden --file eval/questions-120.jsonl`

它是 W7 出题环节的守门人，对应 eval/README.md §六那五条自检的自动化版：

1. **格式校验**：字段齐全、type 合法、锚点数按题型锁死、id 唯一且带正确前缀
2. **锚点反查**：每个 quote 能在**当前索引**里恰好解析到一块
   （反查失败 = 坏数据，报错列出全部失败条目，不是遇到第一个就停）
3. **泄题检查**：问题与 quote 的最长公共子串 ≥5 字且含 ≥4 汉字 → 拒收
4. **拒答证据**：refusal 的 notes 必须留"关键词全文 0 次"的搜索记录
   （metrics.md §六的防刷设计：标错拒答集会让检索失败计成拒答成功）
5. **多跳确实跨块**：两个锚点必须落在不同 chunk（同块按事实型重标）

## 为什么校验必须对着索引，不是对着 md 文件

eval/README.md §六坑一：检索用的是 loader **清洗后**的文本，
`grep` 原文通过 ≠ 索引里找得到。本项目第一次标注就栽在这里
（50 条全过、脚手架跑起来 8 条解析失败）。

## 为什么一次报出全部错误

出题是批量的：120 条里错 8 条时，逐个修复要跑 8 遍校验，
每遍都要加载模型建索引。全部收集、一次报出，改一轮就能过。

## 退出码

0 = 全过；1 = 有错误（数量打印在 stderr）。**绝不"警告但通过"**——
坏数据不会报错，只会让所有结论失真，那比崩溃贵得多。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_retrieval_eval as rre  # noqa: E402
from retrieval_harness import load_raw_corpus  # noqa: E402

from app.rag.chunking import ChunkingConfig, build_strategy  # noqa: E402

#: 锚点数按题型锁死 —— metrics.md §一防刷第 2 条：
#: "把 gold 标得多而宽"是最容易的 Recall 刷法，多标一个锚点就报错。
EXPECTED_ANCHORS: dict[str, int] = {"fact": 1, "multi_hop": 2, "refusal": 0}

#: id 前缀按题型锁死。事实题 f-、多跳 m-、拒答 r-，与 questions-50 一致。
EXPECTED_PREFIX: dict[str, str] = {"fact": "f-", "multi_hop": "m-", "refusal": "r-"}

#: 拒答证据检查的适用范围：**冻结日期（2026-11-30）之后标注的集**。
#: questions-50 是老集 —— 它的 notes 只写了"关键词计数见自检 B"，
#: 那是当时的规范。规范先于数据存在（README 开篇第一句），
#: 追溯老数据等于用今天的尺子量昨天的题；W7 集才按 metrics.md §六执行。
REFUSAL_EVIDENCE_REQUIRED: tuple[str, ...] = ("questions-120.jsonl",)

#: refusal 的 notes 里必须出现的证据关键词。
#: 全文搜索是唯一能证明"语料确实没有"的手段 ——
#: "我记得没写过"与"确实没写"之间差着一次 grep（README §四）。
REFUSAL_EVIDENCE: tuple[str, ...] = ("0 次", "全文")


def validate_format(
    path: Path, *, require_refusal_evidence: bool = False
) -> tuple[list[dict[str, Any]], list[str]]:
    """格式校验。返回（条目, 错误列表）。格式不合法的行直接拒收。

    `require_refusal_evidence` 只对冻结（2026-11-30）之后标注的集开启 ——
    老集（questions-50）的拒答 notes 按当时的规范，不追溯。
    """
    errors: list[str] = []
    items: list[dict[str, Any]] = []
    if not path.exists():
        return items, [f"文件不存在：{path}"]

    seen: set[str] = set()
    # utf-8-sig：PowerShell 的 `Set-Content -Encoding utf8` 会写 BOM，
    # 裸 utf-8 读会在第一行炸"Unexpected UTF-8 BOM" —— 宽容它，别让人卡在编码上
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        prefix = f"{path.name}:{lineno}"

        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{prefix} 不是合法 JSON：{exc}")
            continue
        items.append(raw)

        # 字段齐全（任务书格式：id/question/gold_anchors/gold_answer/type/notes）
        for key in ("id", "type", "question", "anchors", "gold_answer"):
            if key not in raw:
                errors.append(f"{prefix} 缺字段 {key!r}")

        kind = raw.get("type")
        if kind not in EXPECTED_ANCHORS:
            errors.append(f"{prefix} type 非法：{kind!r}（允许：fact/multi_hop/refusal）")
            continue

        # 锚点数锁死 —— 反"标得宽"
        n = len(raw["anchors"] or [])
        if n != EXPECTED_ANCHORS[kind]:
            errors.append(
                f"{prefix} [{raw['id']}] {kind} 应有 {EXPECTED_ANCHORS[kind]} 个锚点，实为 {n}"
                + (
                    "（多标 = 把 Recall 刷上去，锚点数按题型锁死）"
                    if n > EXPECTED_ANCHORS[kind]
                    else ""
                )
            )

        # id 规则：唯一 + 前缀与 type 一致
        item_id = str(raw["id"])
        if item_id in seen:
            errors.append(f"{prefix} id 重复：{item_id}")
        seen.add(item_id)
        if not item_id.startswith(EXPECTED_PREFIX[kind]):
            errors.append(
                f"{prefix} id 前缀与 type 不符：{item_id} 应以 {EXPECTED_PREFIX[kind]!r} 开头"
            )

        # gold_answer 必填且非空 —— 人工复核靠它
        if not str(raw.get("gold_answer") or "").strip():
            errors.append(f"{prefix} [{item_id}] gold_answer 为空")

        # 锚点字段结构
        for i, anchor in enumerate(raw["anchors"] or []):
            for key in ("source", "quote"):
                if not str(anchor.get(key) or "").strip():
                    errors.append(f"{prefix} [{item_id}] 第 {i + 1} 个锚点缺 {key}")
            quote = str(anchor.get("quote") or "")
            if quote and not (15 <= len(quote) <= 80):
                errors.append(
                    f"{prefix} [{item_id}] quote 长度 {len(quote)} 越界（15~80，"
                    "太短易撞段落、太长会被 300 切分切碎 —— README §二）"
                )

        # 拒答必须留全文搜索证据 —— 反"标错拒答集"（metrics.md §六）
        if kind == "refusal" and require_refusal_evidence:
            notes = str(raw.get("notes") or "")
            if not all(token in notes for token in REFUSAL_EVIDENCE):
                errors.append(
                    f"{prefix} [{item_id}] refusal 的 notes 必须留全文搜索证据"
                    f"（包含 {'、'.join(REFUSAL_EVIDENCE)!r}），例如："
                    "「关键词 pgvector/向量索引类型 全文 0 次」。"
                    "没搜过就出题 = 拒答集可能标错 = 拒答准确率不可信"
                )

    return items, errors


def validate_anchors(items: list[dict[str, Any]], chunks: list[Any]) -> list[str]:
    """锚点反查 + 多跳跨块。复用 run_retrieval_eval 的解析器 —— 同一份代码，
    校验与评测看到的失败才一致。"""
    errors: list[str] = []

    for item in items:
        kind = item["type"]
        if kind == "refusal":
            continue
        item_id = str(item["id"])
        resolved: list[str] = []
        for i, anchor in enumerate(item["anchors"]):
            try:
                chunk = rre.resolve_anchor(anchor, chunks)
            except rre.AnchorResolutionError as exc:
                errors.append(f"[{item_id}] 第 {i + 1} 锚点反查失败：{exc}")
                continue
            resolved.append(chunk.id)

        if kind == "multi_hop" and len(resolved) == 2 and resolved[0] == resolved[1]:
            errors.append(
                f"[{item_id}] 多跳两锚点落在同一个 chunk 里 —— 按事实型重标"
                "（README §三：同块能读完的不是多跳）"
            )

    # quote 必须逐字唯一：同一 source 下，两条题的 quote 相同 = 标注重复
    seen_quotes: dict[str, str] = {}
    for item in items:
        for anchor in item["anchors"] or []:
            key = f"{anchor.get('source')}::{anchor.get('quote')}"
            if key in seen_quotes and seen_quotes[key] != str(item["id"]):
                errors.append(
                    f"[{item['id']}] quote 与 [{seen_quotes[key]}] 完全相同 —— "
                    "同一事实被标了两次（换问法也不行，README §〇）"
                )
            seen_quotes.setdefault(key, str(item["id"]))

    return errors


def validate_leak(items: list[dict[str, Any]]) -> list[str]:
    """泄题检查：复用 check_no_leak 的判据（最长公共子串 ≥5 字且 ≥4 汉字）。"""
    errors: list[str] = []
    for item in items:
        if item["type"] == "refusal":
            continue
        gold_like = rre.GoldItem(
            id=str(item["id"]),
            type=item["type"],
            question=str(item["question"]),
            anchors=list(item["anchors"] or []),
        )
        problems = rre.check_no_leak([gold_like])
        errors.extend(problems)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="黄金集校验：格式 + 锚点反查 + 泄题 + 证据")
    parser.add_argument(
        "--file",
        type=Path,
        default=PROJECT_ROOT / "eval" / "questions-120.jsonl",
        help="要校验的 jsonl（默认 W7 报成绩集）",
    )
    args = parser.parse_args(argv)

    # Windows GBK 控制台打不出 ✓/✗ —— stdout 重配为 UTF-8（与 tasks.py 同款约定）
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    # 拒答证据检查只对冻结后标注的集开启（见 REFUSAL_EVIDENCE_REQUIRED）
    require_evidence = args.file.name in REFUSAL_EVIDENCE_REQUIRED
    items, errors = validate_format(args.file, require_refusal_evidence=require_evidence)
    print(f"格式校验：{len(items)} 条，{len(errors)} 个错误")

    # 锚点反查需要索引；空集或格式已坏时跳过（格式错修完再看反查才有意义）
    if items and not errors:
        chunk_cfg = ChunkingConfig(chunk_chars=500, overlap_chars=50)
        chunks = build_strategy("recursive", cfg=chunk_cfg).chunk(load_raw_corpus(), chunk_cfg)
        print(f"锚点反查：{len(chunks)} 块（chunk=500 overlap=50 recursive，冻结口径）")
        errors.extend(validate_anchors(items, chunks))
        errors.extend(validate_leak(items))

    if errors:
        print(f"\n✗ {len(errors)} 个问题：", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        print("\n改数据，不改检查。", file=sys.stderr)
        return 1

    print(f"✓ {args.file.name} 全部通过（{len(items)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
