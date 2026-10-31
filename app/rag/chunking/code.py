"""代码块切分：**按函数/结构边界切，保留缩进与语言标记**。

## 为什么代码不能按正文的递归切分

正文的分隔符优先级是 `\\n\\n → \\n → 。！？； → ，` —— **中文句读**。
拿它切代码会做两件错事：

| 错法 | 后果 |
| --- | --- |
| 按 `\\n\\n` 切 | 函数之间通常只有一个空行，能切；但**空行在函数体内也常见**，会把一个函数劈开 |
| 按 `，` / `；` 切 | 直接切在 `f(a, b)` 的参数中间、`for i in range(n);` 的行尾 —— 代码语法碎了，`is_code` 语义消失 |
| 去掉缩进 | Python 的缩进是**语法**。丢了缩进的块在模型眼里永远报 `IndentationError`，它读到的是"语法错误的代码" |

第三条最隐蔽：切分器会 `strip()` 每一块，而 `strip()` 恰好吃掉 Python 唯一的
结构性信息。所以本模块**保留每行原始缩进**，只在整块首尾去空行。

## 切分边界按语言

| 语言 | 边界判据 |
| --- | --- |
| Python | 顶格（缩进 0）的 `def` / `class` / `async def` / 装饰器 `@` |
| JS/TS/Go/Java/C | 顶格或以 `}` 结束的 `function` / `func` / `class` / `interface` / `=>` 块 |
| 未知语言 | **只在空行处切**（最保守的通用边界） |

未知语言不做"猜语法"：猜错的代价是产出一个语法碎片，
而"按空行切"至少保证块边界落在人写的段落之间。

## 为什么代码块不参与正文递归切分

代码的"句子"是**行**，不是中文句号。混进正文切分器会让 `def` 与它的
`return` 分到两块 —— 那种块对检索毫无价值（命中了也读不出逻辑）。

所以三条策略都必须**先按 `block="code"` 分流**，代码块只走这里。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 代码块的最长字符数。**与正文预算分开**：代码的信息密度高，
# 一块 300 字的代码 ≈ 600 字正文的信息量。
DEFAULT_CODE_CHARS: int = 400

# Python 顶格结构边界
_PY_BOUNDARY = re.compile(r"^(?:async\s+)?(?:def|class)\s+\w|^@|^if\s+__name__")

# 其他语言的顶格结构边界
_OTHER_BOUNDARY = re.compile(
    r"^(?:export\s+)?(?:async\s+)?(?:function|func|class|interface|struct|enum)\b"
    r"|^(?:public|private|protected|static|final)\b"
    r"|^(?:const|let|var)\s+\w+\s*=\s*(?:\(|async|function)"
)

# 语言标记（去掉 ```python 里的反引号后的那个词）
_FENCE_LANG = re.compile(r"^\s*(?:```|~~~)\s*([A-Za-z0-9_+-]*)")

PYTHON_LANGS = frozenset({"python", "py", "python3", "py3"})
JS_LANGS = frozenset(
    {
        "javascript",
        "js",
        "typescript",
        "ts",
        "tsx",
        "jsx",
        "go",
        "java",
        "c",
        "cpp",
        "cs",
        "rust",
        "rs",
    }
)


@dataclass(frozen=True, slots=True)
class CodeBlock:
    """一个切出来的代码块。`language` 与 `indent_preserved` 都要能传出去。"""

    text: str
    language: str
    """语言标记（`python` / `javascript` / 空串）。**保留它是为了下游能重新渲染**：
    一条引用里的代码片段如果丢了语言标记，前端就无法高亮，模型也少了一条先验。"""


def detect_language(first_line: str) -> str:
    """从围栏首行取语言标记（`` ```python `` → `python`）。"""
    match = _FENCE_LANG.match(first_line)
    return (match.group(1) or "").lower() if match else ""


def strip_fences(text: str) -> tuple[str, str]:
    """去掉围栏行，返回 `(代码本体, 语言)`。

    只有**首尾都是围栏**时才去掉；单个围栏（未闭合）原样保留 ——
    未闭合是源文档的问题，不该由切分器替它"修好"（那会掩盖问题）。
    """
    lines = text.split("\n")
    if not lines:
        return text, ""
    language = detect_language(lines[0])
    starts = bool(_FENCE_LANG.match(lines[0]))
    ends = len(lines) > 1 and bool(_FENCE_LANG.match(lines[-1]))
    if starts and ends:
        return "\n".join(lines[1:-1]), language
    return text, language


def split_code(text: str, *, max_chars: int = DEFAULT_CODE_CHARS) -> list[CodeBlock]:
    """按结构边界切代码块。

    ## 三条不变量（测试守着）

    1. **不切断函数**：一个 `def` 块内的所有行必须落在同一块（除非它本身超预算）
    2. **保留缩进**：每行原样保留，不做 `strip()`
    3. **保留语言标记**：`CodeBlock.language` 要带出去

    第 1 条在"函数比预算还长"时无法满足 —— 那时按**行**切而不是按字符切，
    保证至少不在行中间断开。这是唯一能做的：把一个 500 行的函数切成两块，
    比在函数第 200 个字符处切断要好得多。

    ## 围栏行：**留在文本里**

    第一版把它去掉了（"围栏是 Markdown 语法，不是代码"）。但实测这个判断是错的：

    - **语言标记就在围栏上**（`` ```python ``）。去掉之后 `CodeBlock.language`
      只能靠 `detect_language` 从原文里取一次再丢掉，下游（前端渲染 / 模型）
      拿到的是"一段没有语言标识的裸代码"。
    - 下游若要把片段拼回 Markdown（引用展示、拼上下文），没有围栏就**渲染成正文**，
      代码里的 `#` 会变成标题、`*` 会变成斜体 —— 内容被吃掉。

    所以这里保留围栏，只在**切分成多块时**给后续块补上围栏
    （否则第 2 块没有开围栏，同样渲染不出来）。
    """
    body, language = strip_fences(text)
    if not body.strip():
        return []

    boundaries = _boundary_pattern(language)
    chunks = _group_by_boundary(body, boundaries, max_chars)
    if not chunks:
        return []

    fence = _fence_line(text)
    blocks: list[CodeBlock] = []
    for piece in chunks:
        # 每一块都补完整围栏：单块时围栏本来就在原文里，多块时后续块必须补。
        # **用原文的围栏行**（含 `~~~` 与语言标记），不自己拼三引号 ——
        # 原文可能用四个反引号或 `~~~`，自己拼会改掉格式。
        wrapped = f"{fence}\n{piece}\n{fence}" if fence else piece
        blocks.append(CodeBlock(text=wrapped, language=language))
    return blocks


def _fence_line(text: str) -> str:
    """取原文的围栏行（`` ``` `` / `~~~` + 语言标记）。

    没有围栏（未闭合 / 纯代码片段）时返回空串 —— 那时补一个自己拼的围栏
    等于给这段文本加上它原本没有的 Markdown 语义。
    """
    lines = text.split("\n")
    if lines and _FENCE_LANG.match(lines[0]):
        return lines[0].strip()
    return ""


def _boundary_pattern(language: str) -> re.Pattern[str]:
    """按语言选边界判据。**未知语言退化为"空行边界"**。

    未知语言不猜语法（见模块 docstring）：猜错的代价是语法碎片，
    而空行边界至少落在人写的段落之间。
    """
    if language in PYTHON_LANGS:
        return _PY_BOUNDARY
    if language in JS_LANGS:
        return _OTHER_BOUNDARY
    return re.compile(r"^$")  # 空行：最保守的通用边界


def _group_by_boundary(body: str, boundary: re.Pattern[str], max_chars: int) -> list[str]:
    """按边界切分，再贪心合并到 `max_chars`。

    **不切行**：每一行要么整行在前块，要么整行在后块。
    行中切断会让代码变成语法碎片（`ret` + `urn x`），那比块超预算糟得多。

    ## 一个必须写清的顺序问题（实测第一版做错了）

    `def load(...)` 这种 3 行的小函数，在 `max_chars=40` 时**结构组本身就超预算**。
    第一版的流程是"贪心合并 → 遇到超预算的组就按行切"，
    于是 `def load` / `with open` / `return` 被切成三块 —— **函数被拆散**，
    而那正是本模块要防的第一条错。

    正确顺序是：**先判断"这一组能不能装进一块"**，
    装不下才降级到按行切（并接受"这个函数太大"的事实）。

    优先级是硬的：**宁可块超预算，也不拆函数**。
    块超预算的后果是"下游按预算截断时可能整块丢掉"，
    而拆函数的后果是"命中的块里读不出逻辑" —— 后者更糟，因为它是**静默的**。
    """
    # ① 成组：一组 = 一个结构单元（一个函数、一个类、或一个空行分隔的段）
    groups: list[list[str]] = []
    current: list[str] = []
    for line in body.split("\n"):
        is_boundary = bool(boundary.match(line))
        if is_boundary and current and any(ln.strip() for ln in current):
            groups.append(current)
            current = []
        current.append(line)
    if current and any(ln.strip() for ln in current):
        groups.append(current)

    # ② 合并：**只有"整组装得下"才合并进缓冲**；单组超预算时独占一块
    #    （宁可超预算也不拆函数）。只有"单组远大于预算"时才按行切，
    #    阈值取 2 倍预算：一个 800 字的函数在 400 预算下独占一块是合理的，
    #    一个 4000 字的函数就必须切了（否则它永远进不了上下文）。
    out: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for group in groups:
        group_text = "\n".join(group)
        group_len = len(group_text) + 1

        if group_len > max_chars:
            if buffer:
                out.append("\n".join(buffer))
                buffer, buffer_len = [], 0
            if group_len <= max_chars * 2:
                # 略超预算：**整组一块**，不拆函数
                out.append(group_text)
            else:
                # 远超预算（> 2 倍）：按行切，这是唯一的选择
                out.extend(_slice_by_lines(group, max_chars))
            continue

        if buffer and buffer_len + group_len > max_chars:
            out.append("\n".join(buffer))
            buffer, buffer_len = [], 0
        buffer.extend(group)
        buffer_len += group_len

    if buffer:
        out.append("\n".join(buffer))

    # ③ 去整块首尾的空行（**不动行内缩进**）
    return [piece.strip("\n") for piece in out if piece.strip()]


def _slice_by_lines(group: list[str], max_chars: int) -> list[str]:
    """按**行**累积到预算就断。行中间永不断开（缩进与语法都靠行）。"""
    out: list[str] = []
    line_buffer: list[str] = []
    line_len = 0
    for line in group:
        line_len += len(line) + 1
        if line_buffer and line_len > max_chars:
            out.append("\n".join(line_buffer))
            line_buffer, line_len = [line], len(line) + 1
        else:
            line_buffer.append(line)
    if line_buffer:
        out.append("\n".join(line_buffer))
    return out


__all__ = [
    "DEFAULT_CODE_CHARS",
    "CodeBlock",
    "detect_language",
    "split_code",
    "strip_fences",
]
