"""中文分词：**索引侧与查询侧共用的唯一分词器**。

## 为什么这个模块是 BM25 全部风险所在

BM25 靠**字面重合**打分。它算的是"query 里的词在块里出现了几次"，
所以"词"的边界一旦两边不一致，分数就是错的 —— 而且**不报错**：

    索引时把「同轴光」切成 → 同轴 / 光
    查询时把「同轴光」切成 → 同轴 / 光        ← 一致，能匹配
    但用户查「同轴光镜头」→ jieba 可能切成   → 同轴 / 光 / 镜头   ← 也能匹配

    真正的坑在这里：索引时切「同轴光」，查询时用户写的是「同轴」，切成 同轴
    → 索引里根本没有 `同轴` 这个 token（只有 `同轴` + `光` 两个），勉强能中
    但如果自定义词典只在索引侧生效、查询侧没加载，同一个字符串会被切成不同的词。

所以规矩是：**任何时候都不允许各自调用 `jieba.lcut`**。
只允许走 `tokenize()`，它保证两侧用同一个词典、同一套停用词、同一种过滤。

## 三件必须一起做对的事

| 事 | 做错的后果 |
| --- | --- |
| 自定义词典（领域术语） | 「同轴光」被切成「同轴/光」，与 query 的「同轴光」不是一个 token |
| 停用词表 | 「的/了/是」在所有块里都有，IDF 趋 0，但它们仍占 BM25 的 TF 与长度归一 |
| 两侧同一个 tokenizer | 最隐蔽：两侧都"能跑"，只是分数系统性偏移 |

## 为什么自定义词典要**强制加载**而不是"建议加载"

`jieba.add_word` 是**进程级全局状态**。如果它只在建索引的脚本里被调用，
那么"查询进程"就没有这些词 —— 两个进程的分词结果不一致，
而错误表现是"BM25 召回率比预期低"，会被归因到 BM25 本身不行。

所以词典在**模块导入后由 `tokenize()` 首次调用时**加载（`_ensure_dict`），
索引脚本、查询进程、测试、Web 服务拿到的都是同一份。加载是幂等的。

## 词典放哪、怎么加

`app/rag/bm25/domain_dict.txt`，一行一个词，`#` 开头是注释。
可加词频（`词 频率 词性`），本项目的词都是"能切出来就行"，只用词本身。

加词的判据不是"这个词重要"，而是**"jieba 默认会把它切碎，而它在本项目语料里
是一个整体概念"**。比如「同轴光」默认切成「同轴/光」——
在光学语境里「同轴光」是一个器件属性，切碎后它与「同轴」的块、
「光」的块都会产生虚假匹配。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

DICT_PATH: Path = Path(__file__).resolve().parent / "domain_dict.txt"

STOPWORDS: frozenset[str] = frozenset(
    {
        # 中文虚词与高频连接词：IDF 趋 0，留着只会占 TF 与长度归一
        "的",
        "了",
        "是",
        "在",
        "和",
        "与",
        "或",
        "而",
        "及",
        "等",
        "也",
        "都",
        "就",
        "还",
        "把",
        "被",
        "让",
        "给",
        "对",
        "从",
        "到",
        "为",
        "以",
        "于",
        "之",
        "其",
        "这",
        "那",
        "个",
        "些",
        "有",
        "没",
        "不",
        "很",
        "会",
        "能",
        "要",
        "该",
        "着",
        "过",
        "得",
        "地",
        "吗",
        "呢",
        "啊",
        "吧",
        # 英文虚词
        "the",
        "a",
        "an",
        "of",
        "to",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "as",
        "it",
        "this",
        "that",
    }
)
"""停用词表。

**这串东西对 BM25 不是"锦上添花"，是必需的**：BM25 的长度归一因子
`|D| / avgdl` 里 `|D|` 是 token 数 —— 停用词占掉的每 1 个 token
都在稀释真正有区分度的词。中文里「的」出现频率极高，
不剔掉会让长块和短块的分数差被虚词抹平。

**为什么不用 jieba 自带的停用词表**：它面向通用语料，会剔掉
「不」「没」这类在本项目里**有语义**的词（「不返回 usage」「没有 413」）。
判据是"在这批文档里 IDF 是否趋 0"，不是"通用中文里是否常见"。
"""

_MODEL_NAME = "jieba"

_dict_loaded = False


def _ensure_dict() -> None:
    """把领域词典灌进 jieba 的全局词典树。**幂等，首次调用时执行。**"""
    global _dict_loaded
    if _dict_loaded:
        return
    import jieba

    if DICT_PATH.exists():
        # `load_userdict` 接受一行一个词，也接受 `词 频率 词性`。
        jieba.load_userdict(str(DICT_PATH))
    _dict_loaded = True


_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:[-.][A-Za-z0-9_]+)*|[0-9]+(?:\.[0-9]+)+")
r"""**型号 / 标识符 / 数值的保护网**。

这是实测出来的，不是预防性的：`INVALID_INPUT` 被 jieba 切成 `invalid` / `input`，
`bge-small-zh` 被切成 `bge` / `small` / `zh`，`4.2` 被切成 `4` / `.` / `2`。

前两种切法**都会让 BM25 在该强的地方变弱**：
用户查 `INVALID_INPUT` 时，索引里只剩 `invalid` 和 `input` 两个极常见的词，
IDF 很低；而 `INVALID_INPUT` 这个整体标识符的 IDF 本该很高。
型号匹配是 BM25 相对向量检索**唯一的结构性优势**（见 README 的必答问题），
在这里切碎等于把优势自己拆掉。

两段选择并列：

- `[A-Za-z][A-Za-z0-9_]*(?:[-.][A-Za-z0-9_]+)*` → `INVALID_INPUT`、`bge-small-zh`、
  `deepseek-chat`、`v0.4.1`。普通英文单词（`def`、`and`）也走这支，
  但它们不含分隔符，整体切出来与 jieba 的结果一致，无副作用。
- `[0-9]+(?:\.[0-9]+)+` → `4.2`、`3.11`、`0.19`。小数/版本号是一个数，
  拆成 `4` `.` `2` 会让 `.` 被当标点丢掉、只剩两个孤零零的数字。

**单段数字（`3209`）不在此列**：jieba 本来就整体给出，不需要额外处理。
"""


def _tokenize_plain(text: str) -> list[str]:
    """对**不含标识符的**片段做分词 + 过滤。"""
    import jieba

    tokens: list[str] = []
    for raw in jieba.lcut(text):
        token = raw.strip().lower()
        if not token:
            continue
        if token in STOPWORDS:
            continue
        if _PUNCT_ONLY.match(token):
            continue
        # 中文单字：多为切分噪声（"多/跳"、"幂/等/性"的前身）
        if len(token) == 1 and _is_cjk(token):
            continue
        tokens.append(token)
    return tokens


@lru_cache(maxsize=4096)
def tokenize(text: str) -> tuple[str, ...]:
    """切成小写 token 元组。**索引与查询都必须走这里。**

        `lru_cache` 是必要的：建索引时 1592 个块要切，查询时每个 query 切一次，
        而词典树在老版本 jieba 上是纯 Python 的前缀扫描 ——
        重复切同一段文本（测试里反复跑）会明显变慢。

        ## 返回**元组**而不是列表

        BM25 的 `BM25Okapi` 直接吃 `list[list[str]]`，元组会被当成可迭代，
        但它内部有 `corpus` 的 set 运算。返回不可变元组，让"调用方顺手改一下
        token 流"这种事在类型层面就做不到 —— 那会破坏索引与查询的一致。

        ## 处理顺序（顺序本身就是设计）

        **先按 `_IDENTIFIER` 把文本切成"标识符段"与"普通段"交替的序列**，
    标识符段整体作为一个 token，普通段才交给 jieba。

        为什么不用"占位符替换"（那是更常见的做法）：占位符本身含数字，
        jieba 会把占位符里的数字单独切出来，结果比不做还糟 —— 实测过，
        `INVALID_INPUT` 会变成 `('0', '错误码')`。区间扫描没有这个副作用，
        也不需要维护占位符映射。

        ## 归一化 / 丢弃

        - **小写**：`HTTP` 与 `http` 在 BM25 里是两个 token，但语义是一个
        - **保留 `_` `-` `.`**：见 `_IDENTIFIER` 的说明
        - 丢弃：纯标点、停用词、中文单字碎片
        - **英文单字符不丢** —— `C` 可能是语言名，`R` 可能是版本号的一部分
    """
    _ensure_dict()
    tokens: list[str] = []
    position = 0
    for match in _IDENTIFIER.finditer(text):
        if match.start() > position:
            tokens.extend(_tokenize_plain(text[position : match.start()]))
        identifier = match.group(0).lower()
        if identifier and identifier not in STOPWORDS:
            tokens.append(identifier)
        position = match.end()
    if position < len(text):
        tokens.extend(_tokenize_plain(text[position:]))
    return tuple(tokens)


_PUNCT_ONLY = re.compile(r"^[\W_]+$")
r"""整个 token 都是标点/下划线时丢弃。

**但 `_` 单独作为一个 token 要丢，`INVALID_INPUT` 保留**：
`\W` 在 re 里对 Unicode 中文是「非单词字符」，中文标点会被它匹配到；
而 `INVALID_INPUT` 里有 `\w`（字母数字），所以不匹配 —— 正是想要的。
"""


_CJK_RANGE = ("\u4e00", "\u9fff")


def _is_cjk(token: str) -> bool:
    return all(_CJK_RANGE[0] <= char <= _CJK_RANGE[1] for char in token)


def dict_words() -> list[str]:
    """读词典里的词。**唯一的解析规则**，`describe()` 与体检脚本都用它。

    规则只有两条：跳过空行与 `#` 注释行，其余整行是一个词。
    没有"最多取第一列"那种宽松处理 —— 因为宽松解析正是上一版
    把一整段 `>` 说明文字当成词的原因（见词典文件末尾的说明）。
    """
    if not DICT_PATH.exists():
        return []
    words: list[str] = []
    for line in DICT_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # `词 频率 词性` 是 jieba 的扩展格式，本项目没有用；只用词本身。
        words.append(stripped.split()[0])
    return words


def describe() -> dict[str, object]:
    """给索引元数据 / 日志用：**这份索引是按哪个分词器切的**。

    没有这一项，两个用不同词典建出来的索引在库里长得一模一样，
    而 BM25 分数会差一截 —— 那是最难查的一类回归。
    """
    _ensure_dict()
    return {
        "tokenizer": _MODEL_NAME,
        "dict_path": str(DICT_PATH.name),
        "dict_words": len(dict_words()),
        "stopwords": len(STOPWORDS),
    }


__all__ = ["DICT_PATH", "STOPWORDS", "describe", "dict_words", "tokenize"]
