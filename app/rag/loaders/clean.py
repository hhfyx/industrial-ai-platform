"""清洗：**只做"排版噪声"的去除，不做"语义改写"**。

这一层最大的风险不是"洗得不干净"，是**洗过头**。原因很具体：
前端显示的引用是 `合同.pdf 第 7 页` + 一段原文摘录，用户会拿这段摘录去 PDF 里
搜索。**只要清洗动了正文一个字，用户就搜不到那句** —— 那一刻他不是在怀疑代码，
是在怀疑整份报告的可靠性。

所以这里的规则是"书面化"的，而且是**可被测试断言**的：

> 清洗后的文本，必须是清洗前文本的**子序列**（去掉空白后逐字比对）。
> 唯一的例外是被显式判定为噪声的整行（页眉/页脚/页码行）。

`tests/test_loaders.py::test_cleaning_never_rewrites_body_text` 就是在守这条。
一旦有人往里加"顺便把某个词换掉"的逻辑，那条测试会立刻红。

---

## 四类清洗，各自的判据不同（混在一起就必然出错）

| 噪声 | 判据 | 为什么不能用另一种判据 |
| --- | --- | --- |
| 页眉 页脚 | **按出现频率**：首/末行在 ≥60% 的页上重复 | 页码每页都不同，频率永远抓不到 |
| 页码行 | **按形态**：`第 3 页` / `- 3 -` / `Page 3 of 50` | 它只出现一次，频率统计看不见它 |
| 断行 | **按上下文**：上一行没有句末标点，下一行以小写/汉字开头 | 纯按长度切会把完整句子也粘上 |
| 多余空白 | 纯正则，无上下文 | — |

**两类判据必须分开**，这是这一格最容易踩的坑：把页码行丢进频率计数器，
它统计值永远是 1，于是永远删不掉；把页眉丢进形态正则，正文里恰好长得像
`第 3 页` 的引用会被误删。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- 1. 空白压缩

# 连续的水平空白（含全角空格）
_MULTI_WS = re.compile(r"[ \t\u3000]+")
# 连续 3 个以上换行压成 2 个：保留"段落之间有空行"这个事实，但不保留空了几行
_MULTI_NL = re.compile(r"\n{3,}")


def normalize_text(raw: str) -> str:
    """最小清洗：统一换行、压缩空白、去首尾空白。

    **刻意不做的事**：不删行（那是 `strip_repeated_lines` / `strip_page_numbers`
    的活）、不修断词、不做任何需要上下文的重排 —— 这些都要带行级信息，
    压成一个大字符串之后再处理就晚了。

    **为什么压缩空白是安全的**：PDF 抽出的多空格是排版坐标的残留
    （`drawString(60, y)` 与 `drawString(120, y)` 之间会被抽成一大段空白），
    它不是原文内容。压缩它不会让用户搜不到句子 —— 用户搜索时也不会输入 5 个空格。
    """
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")  # 全角空格 → 半角
    text = _MULTI_WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


# ---------------------------------------------------------------- 2. 页眉页脚


def strip_repeated_lines(
    pages: list[list[str]], *, min_repeat_ratio: float = 0.6
) -> list[list[str]]:
    """去掉**每页都出现**的首/末行（页眉 / 页脚 / 水印）。

    只取首尾各 1 行，是踩过坑的：最初写成"首 2 行 + 末 2 行"，
    测试用的 4 行页面直接覆盖了**全部**行，正文三行全被当成候选删掉，
    每页只剩标题。那种 bug 很隐蔽：结果看起来"有内容"（标题还在），
    只是内容少了一大截。真正的页眉页脚就在首/末行，取 1 行足够。

    少于 3 页不做统计：2 页重复恰好 100%，会把正文当页脚删掉。
    """
    if len(pages) < 3:
        return pages

    threshold = max(2, int(len(pages) * min_repeat_ratio))
    first_lines: dict[str, int] = {}
    last_lines: dict[str, int] = {}
    for lines in pages:
        non_empty = [ln.strip() for ln in lines if ln.strip()]
        if not non_empty:
            continue
        for bucket, line in ((first_lines, non_empty[0]), (last_lines, non_empty[-1])):
            if len(line) < 2:
                continue
            bucket[line] = bucket.get(line, 0) + 1

    drop: set[str] = set()
    for bucket in (first_lines, last_lines):
        for line, count in bucket.items():
            if count >= threshold:
                drop.add(line)

    cleaned: list[list[str]] = []
    for lines in pages:
        kept = list(lines)
        # 只在两端剥：正文中间的重复行不动（可能是正常的重复表格行）
        while kept and kept[0].strip() and kept[0].strip() in drop:
            kept.pop(0)
        while kept and kept[-1].strip() and kept[-1].strip() in drop:
            kept.pop()
        cleaned.append(kept)

    if drop:
        logger.info("剔除疑似页眉页脚 %d 行：%s", len(drop), list(drop)[:3])
    return cleaned


# ---------------------------------------------------------------- 3. 页码行

_PAGE_NUMBER = re.compile(
    r"^\s*(?:第\s*\d+\s*页(?:\s*[/／]?\s*共\s*\d+\s*页)?|[-–—]\s*\d+\s*[-–—]|page\s+\d+(?:\s+of\s+\d+)?|\d{1,4})\s*$",
    re.IGNORECASE,
)


def strip_page_numbers(lines: list[str]) -> list[str]:
    """剔除页码行。**必须单独做**：页码每页都不同，频率统计永远抓不到它。

    只剥首尾各 1 行：正文里出现一个孤立数字（金额 `128000`）是正常内容。
    """
    if not lines:
        return lines
    kept = list(lines)
    if kept and _PAGE_NUMBER.match(kept[0]):
        kept.pop(0)
    if kept and _PAGE_NUMBER.match(kept[-1]):
        kept.pop()
    return kept


# ---------------------------------------------------------------- 4. 合并断行

# 行首是这些字符 → 上一行**不可能**在标点处结束（是小写字母、汉字、逗号等续行特征）
_CONTINUATION_HEAD = re.compile(r"^[\u4e00-\u9fff a-z0-9(（\[【,，、;；:：\"'「『]")
# 行尾是这些标点 → 这一行说完了，不该和下一行粘
_SENTENCE_END = re.compile(r"[。！？.!?;；:：…】）)\"'」』】]\s*$")
# 行尾是这些字符 → 话没说完（字母/数字/汉字/逗号/顿号/连字符）
_UNFINISHED_END = re.compile(r"[A-Za-z0-9\u4e00-\u9fff,，、\-—–]\s*$")
# 行尾是**拉丁字母**：英文断行常见形态，下一行往往以大写字母续上
_LATIN_TAIL = re.compile(r"[A-Za-z]\s*$")
# 看起来像"新章节标题"的行，绝不与上一行拼接
_HEADING_LIKE = re.compile(
    r"^\s*(?:第\s*[0-9一二三四五六七八九十百]+\s*[章节条部分]\b|"
    r"(?:附件|附录|表格|表|图)\s*[0-9一二三四五六七八九十]+|"
    r"[0-9]{1,2}(?:\.[0-9]{1,2})+\s|"
    r"[一二三四五六七八九十]+[、.]|"
    r"#+\s)"
)


def looks_unfinished(text: str) -> bool:
    """这段文本是否"话没说完"（页末断句的判据）。

    第一步看有没有句末标点；第二步看结尾字符类型。
    会在"页末正好是个短句但没加标点"时误判，但成本极低（只是多拼一次），
    误判后果远小于"句子被永久切断"。
    """
    tail = text.rstrip()
    if not tail:
        return False
    if _SENTENCE_END.search(tail):
        return False
    return bool(_UNFINISHED_END.search(tail))


def can_join(previous: str, following: str) -> bool:
    """判断两段是否该拼接（**跨行** 与 **跨页** 共用这一条判据）。

    四条同时成立才拼：
    1. 上一段没说完（无句末标点）
    2. 上一段**本身不是标题**（标题天然不带句号，否则会被当成"没说完"）
    3. 下一段不以"标题样"开头（否则那是新章节，不是断句）
    4. 下一段首字符是续行特征（小写/汉字/逗号…）

    第 2 条是实测补上的：`第 1 章 · 第 1 页主题` 这类标题行结尾没有标点，
    按规则 1 会被判定"没说完"，于是把下一句正文粘到标题后面
    （`第 1 章 · 第 1 页主题本节是第 1 页的正文…`）。读者看到的是标题和正文
    糊成一行 —— 而这种错**看起来像排版，不像 bug**，最容易留在库里。

    第 3 条防的是跨页场景：页末没标点非常常见，但下一页很可能是
    `第 6 条 违约责任`。把它接到上一句屁股后面，会造出一句原文里根本不存在的
    句子，而且读起来通顺，比乱码更难发现。
    """
    if not previous.strip() or not following.strip():
        return False
    if not looks_unfinished(previous):
        return False
    prev_body = previous.strip()
    if _HEADING_LIKE.match(prev_body):
        return False
    head = following.lstrip()
    if _HEADING_LIKE.match(head):
        return False
    if _CONTINUATION_HEAD.match(head):
        return True
    # 英文断行的常见形态：上一行以字母结尾，下一行以**大写字母**开头
    # （新句子或专有名词）。只有"上一行确实断在拉丁字母上"时才允许这样拼 ——
    # 否则中文句子后面跟一个英文大写词（`Total amount`）会被误粘。
    return bool(_LATIN_TAIL.search(prev_body) and head[:1].isupper())


def merge_wrapped_lines(lines: list[str]) -> list[str]:
    """把 PDF 硬换行粘回段落。

    PDF 里的"换行"绝大多数是**排版行为**（一行放不下就换行），不是段落边界。
    不合并的话，"甲方应于第 1 次里程碑验收后 / 十五个工作日内完成付款" 会被
    当成两段 —— 切分时可能正好切在中间，检索命中半句。

    段落边界靠**空行**表示（`normalize_text` 已经保证连续空行压成 1 个）。
    所以这里只在"非空行 + 非空行"之间判断，空行一律保留为段落分隔。
    """
    out: list[str] = []
    for line in lines:
        if not line.strip():
            out.append("")
            continue
        if out and out[-1] and can_join(out[-1], line):
            # 中文之间不加空格（加了会让用户搜不到原文），英文之间加一个
            separator = " " if needs_space(out[-1], line) else ""
            out[-1] = out[-1].rstrip() + separator + line.lstrip()
        else:
            out.append(line)
    return out


def needs_space(left: str, right: str) -> bool:
    """两段拼接时要不要补空格：英文词被切断时需要，中文不需要。

    判断依据是**两侧字符类型**：`...Machine` + `Learning...` 必须补空格，
    否则会粘成 `MachineLearning`，那是新词 —— 同样是数据损坏。
    中文之间加空格则会让用户搜不到原文（用户不会输入那个空格）。

    行尾连字符（`Machine-` + `Learning`）直接拼：那种情况下连字符本身就是
    断词标记，保留它会把 `Machine-Learning` 这种真词判错。

    **PDF loader 也调这个函数**（跨页拼接）。两处各写一份的代价是：
    改了一处忘了另一处，然后"跨行拼对了、跨页拼错了"——
    这种不一致比彻底错更难查。
    """
    if not left or not right:
        return False
    last, first = left.rstrip()[-1], right.lstrip()[0]
    if last == "-":
        return False
    return bool(re.match(r"[A-Za-z0-9]", last)) and bool(re.match(r"[A-Za-z0-9]", first))


__all__ = [
    "can_join",
    "looks_unfinished",
    "merge_wrapped_lines",
    "needs_space",
    "normalize_text",
    "strip_page_numbers",
    "strip_repeated_lines",
]
