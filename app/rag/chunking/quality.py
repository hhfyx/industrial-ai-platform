"""切分质量的**量化代理指标**：半句率。

## 这个指标在测什么

一段文本被切成块之后，最刺眼的两类错都在**块的边界**上：

1. **块首是半句**：`…验收标` + `准及相应的付款…` —— 后一块从半个词开始
2. **块尾是半句**：`…支付合同价` —— 前一块停在一句话的中间

两者都是"切点落错了位置"。指标就是数这两类块占多少：

    半句率 = (首字符可疑的块数 + 末字符可疑的块数) / (2 × 块总数)

分母取 `2 × 块数` 而不是块数：每块有两个边界，两个都可以错。

## 为什么它是"代理"而不是"真值"

**它只能看见标点，看不见语义。** 一块以 `。` 结尾，也可能正好把一个
"若 A，则 B"的因果句拆开（`若甲方未按期付款。` —— 有句号但论述不完整）。
判据二（论述单元是否被拦腰切）永远需要人读。

但这不等于指标没用。它的价值在**方向性**：
- 半句率 **0.9** → 切分器在词中间切断（LangChain 那种贪心截断）
- 半句率 **0.0** → 每个边界都落在标点上（`DelimiterSplitter` 的设计目标）

两个数差一个量级时，它足以**指出问题存在**；W4 的 B2 就是这样发现
"LangChain 切出半个词"的 —— 那件事光看块长分布是发现不了的。

## 判据为什么这么写（每条都有反例在铺垫）

| 块首 | 判定 | 为什么 |
| --- | --- | --- |
| 句末标点开头（`。` `！`） | **可疑** | 标点属于上一句，它出现在块首说明上一块被切在标点之前 |
| 逗号/顿号/分号开头 | **可疑** | 强边界不该有块以弱标点起头 |
| 无空格的后接拉丁字母/数字（`05mm` 开头而前块以 `0.` 结尾） | **可疑** | 数字被拦腰切 |
| 标题（`#` / `##`）、代码围栏 | **安全** | 结构边界天然不以标点起头 |
| 表格行（`|` 开头） | **安全** | 表格行是结构单位 |
| 其余 | 安全 | 中文句子的正常起点 |

| 块尾 | 判定 | 为什么 |
| --- | --- | --- |
| `。！？；` 结尾 | **安全** | 句末 |
| `\n` 结尾 | **安全** | 段落/表格行结束 |
| 数字或单位结尾（`0.05mm` / `95%`） | **可疑** | 数字被切断的最强信号 |
| 逗号结尾 | **可疑** | 强边界不该停在弱标点 |
| 闭合标点（`）` `」` `】`）后无标点 | **可疑** | 括号收尾却没句号，通常是被硬切 |

## 已知会误判的三类（写出来，避免有人把 0.0 当完美）

1. **列表项**（`- 要点`）本身不带句号 → 误报为半句尾
2. **表格块**不以标点结尾 → 但按上面的判据 `|` 行是安全的，整块不以 `|` 结尾时会误报
3. **代码块**以 `}` / 行尾结束 → 要靠 `is_code` 白名单排除

所以 `SentenceRate` 的分组统计**必须按块类型分开**，
把三类特殊块混进平均值会让这个指标失去意义。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.schema import Chunk

# 句末标点：块尾落在这些字符上是"完整的一句话"
SENTENCE_END = frozenset("。！？；.!?;")
# 弱标点：块首/块尾出现它们说明切点落在句中
WEAK_PUNCT = frozenset("，、,：:—…")
# 标题 / 结构性块首字符
STRUCTURAL_START = frozenset("#>|`-*")

# 数字或单位的结尾（`0.05mm` / `95%` / `±0.02mm`）
_NUMBER_TAIL = re.compile(r"[0-9０-９][A-Za-zμ℃°‰%]*$|[0-9０-９]$|%$|‰$|℃$")


@dataclass
class ChunkVerdict:
    """单个块的判定。**保留名字而不是只留布尔**：报告里要能列出具体是哪些块。"""

    index: int
    text_head: str
    text_tail: str
    head_suspect: bool
    tail_suspect: bool
    reason: str = ""

    @property
    def suspect(self) -> bool:
        return self.head_suspect or self.tail_suspect


@dataclass
class SentenceRate:
    """一组块的半句率报告。"""

    total: int = 0
    head_suspect: int = 0
    tail_suspect: int = 0
    verdicts: list[ChunkVerdict] = field(default_factory=list)
    skipped: int = 0
    """被白名单排除的块数（表格 / 代码 / 标题 / 单块）。"""

    @property
    def rate(self) -> float:
        """半句率 = 可疑边界数 / 全部边界数。**分母是 `2 × 评估块数`。**

        空集合返回 0.0（而不是 1.0 或 NaN）：没有块就没有可疑边界。
        返回 NaN 会让"打印成 nan"变成一次误诊。
        """
        boundaries = 2 * self.total
        if boundaries <= 0:
            return 0.0
        return round((self.head_suspect + self.tail_suspect) / boundaries, 4)

    def as_dict(self) -> dict[str, object]:
        return {
            "chunks": self.total,
            "skipped": self.skipped,
            "head_suspect": self.head_suspect,
            "tail_suspect": self.tail_suspect,
            "sentence_rate": self.rate,
            # 前 5 条可疑块的具体位置，便于人工复核（只给数字等于让人去翻文件）
            "examples": [
                {
                    "index": v.index,
                    "head": v.text_head,
                    "tail": v.text_tail,
                    "reason": v.reason,
                }
                for v in self.verdicts
                if v.suspect
            ][:5],
        }

    def summary(self) -> str:
        """一行可读的句子，直接贴进报告。"""
        return (
            f"半句率 {self.rate:.2%}"
            f"（{self.total} 块评估、{self.skipped} 块跳过；"
            f"块首可疑 {self.head_suspect}、块尾可疑 {self.tail_suspect}）"
        )


def check_chunk(chunk: Chunk) -> ChunkVerdict | None:
    """单个块 → 判定。`None` 表示按白名单跳过（不该被计入半句率）。"""
    if chunk.is_table or chunk.is_code:
        # 表格行与代码行的"边界"不是句子边界 —— 计入指标会把 0 拉高
        return None
    text = chunk.text.strip()
    if not text:
        return None
    # **整块只有一个"行"且是标题**：标题不带句号是正常的
    if chunk.block == "heading" and "\n" not in text:
        return None

    head = text[0]
    tail = text[-1]
    head_bad = False
    tail_bad = False
    reasons: list[str] = []

    # ---- 块首
    if head in SENTENCE_END:
        head_bad = True
        reasons.append(f"块首是句末标点 {head!r}")
    elif head in WEAK_PUNCT:
        head_bad = True
        reasons.append(f"块首是弱标点 {head!r}")
    elif head.isdigit() and tail == "":
        head_bad = True
        reasons.append("块首是裸数字")

    # ---- 块尾
    if tail in SENTENCE_END or tail == "\n" or tail in "）)」』】":
        tail_bad = False
    elif tail in WEAK_PUNCT:
        tail_bad = True
        reasons.append(f"块尾停在弱标点 {tail!r}")
    elif _NUMBER_TAIL.search(text):
        tail_bad = True
        reasons.append(f"块尾是数字/单位：{text[-12:]!r}")
    elif tail.isalpha() and tail not in STRUCTURAL_START:
        # 中文正文以汉字结尾且无标点：可能是列表项，也可能是被硬切。
        # **判为可疑**：宁可多报一条让人复核，也不要漏掉真正的断句。
        tail_bad = True
        reasons.append(f"块尾无标点：{text[-12:]!r}")

    return ChunkVerdict(
        index=chunk.index,
        text_head=text[:20],
        text_tail=text[-20:],
        head_suspect=head_bad,
        tail_suspect=tail_bad,
        reason="；".join(reasons),
    )


def measure(chunks: list[Chunk]) -> SentenceRate:
    """一组块 → 半句率报告。"""
    report = SentenceRate()
    for chunk in chunks:
        verdict = check_chunk(chunk)
        if verdict is None:
            report.skipped += 1
            continue
        report.total += 1
        report.head_suspect += int(verdict.head_suspect)
        report.tail_suspect += int(verdict.tail_suspect)
        report.verdicts.append(verdict)
    return report


def measure_by_kind(chunks: list[Chunk]) -> dict[str, dict[str, object]]:
    """按块类型分组统计。

    **为什么必须分组**：三类块的边界判据不同（表格以行结束、代码以 `}` 结束、
    正文以 `。` 结束）。混在一起求平均，等于用一个数描述三件不同的事 ——
    而那个平均数没有任何一份文档能对上。
    """
    groups: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        groups.setdefault(chunk.block, []).append(chunk)
    return {kind: measure(items).as_dict() for kind, items in groups.items()}


__all__ = [
    "ChunkVerdict",
    "SENTENCE_END",
    "SentenceRate",
    "check_chunk",
    "measure",
    "measure_by_kind",
]
