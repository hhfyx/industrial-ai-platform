"""策略三：语义切分（`SemanticChunking`）。

## 它依赖什么、在哪退化

依赖 **句间语义突变**：把文本切成句子，算每句的向量，相邻两句的相似度
在整个文档里**相对最低**的那些位置就是候选断点。

退化点（预判见 `docs/chunking-predictions.md`，这里是实现上的对策）：

| 文档形态 | 失效方式 | 本实现的对策 |
| --- | --- | --- |
| 并列条款（每句一个主题） | **切得过碎**：每句之间相似度都低，每句都成断点 | `semantic_min_chars` 把过短的段并回上一段 |
| 单主题技术文档 | **退化成硬切**：相似度一直高，永不到达断点 | `semantic_max_chars` 在句子边界强制断开 |
| 短句为主的中文 | 句向量区分度低（hashing 下尤其糟），相似度噪声压过真实边界 | **无法在实现层解决** —— 只能如实标注它依赖 embedder 质量 |

## 两个结构性代价（写出来是为了将来能推翻这个决定）

**① 成本**：索引阶段要为**每个句子**算一次向量。20 页 ≈ 600 句，
bge 批处理约 1 秒；5000 页知识库 ≈ 15 万句 —— 索引时间翻倍。
注意它**不影响入库的向量数**（仍是 ~15000），所以这笔钱是纯额外开销。

**② 依赖**：它**不能在没有 embedder 的环境里工作**。CI / 免费层（hashing）下
它是在"字面相似"的分数上找语义边界 —— 在错误信号上做决策。
所以 `hashing` 下它的结论**没有参考价值**，必须与 bge 分组统计。

## 为什么用分位数而不是绝对阈值

相似度的绝对值是 **embedder 相关**的（bge 相关区间 0.578~0.676、
hashing 0.088~0.484，见 `embedding/base.py` 那张实测表）。
写死 `similarity < 0.5` 会让"换个模型就一个块都切不出来"——
而那种失效不报错，只是块变长，最容易被归因到"模型不行"。
取**分位数**则对分布的平移/缩放免疫。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from app.rag.chunking.base import ChunkingConfig, inherited_pages, make_chunk
from app.rag.embedding import Embedder
from app.rag.embedding.vector import cosine
from app.rag.schema import Chunk, Document

logger = logging.getLogger(__name__)

# 中文句子边界：句末标点（保留标点在前一句末尾）。
# 与 `delimiter.py` 的原子片切法同一个口径，但这里**必须保留标点**：
# 断点位置就是句子位置，丢了标点就无法把句子拼回连续的原文。
_SENTENCE_END = re.compile(r"([。！？；\n])")


def split_sentences(text: str) -> list[str]:
    """把一段文本切成句子。**保留原换行，空句不产出**。

    ## `\n` 必须原样留在句尾，不能当"边界后丢掉"（实测踩过）

    第一版把 `\n` 当成与句号同等的分隔符，切完 `strip()` ——
    结果拼回块时**换行全没了**。在 Markdown 上后果很严重：

        # W4 开工清单> 从 W3 的未决项里提炼出的**可执行**清单。> 为什么必须做…

    标题与正文挤在同一行，引用标记 `>` 变成正文里的普通字符 ——
    块进了索引之后，模型看到的是**结构已经塌掉的文本**。
    实测这一条在 `docs/w4-checklist.md` 的语义样本上，**48 块里有 20 多块受影响**。

    所以：换行**作为边界但不丢弃**，切点落在 `\n` 之后、`\n` 留在前一句末尾。
    性质：`"".join(sentences)` 能还原原文（由测试守着）。

    ## 为什么 `\n` 也算边界

    中文正文里的换行常常就是分句（尤其 loader 从 PDF 抽出来的行）。
    代价是"一句被硬换行断开的长句会被拆成两句"，
    但那与"在句子边界断点"并不冲突 —— 拆分位置本来就是候选断点。

    ## 两条不变量（由 `tests/test_chunking.py` 守着）

    1. **`"".join(split_sentences(t)) == t.strip()`** —— 切分不丢字符。
       这是"块覆盖了原文"的可验证形式；丢了字符，覆盖率就是假的。
    2. **不含空白句** —— 空块会污染索引。

    第 1 条坑了三次，每次都是"丢字符"：

    | 版本 | 写法 | 结果 |
    | --- | --- | --- |
    | 第一版 | `buffer.strip()` 落袋 | 行尾换行全丢 → Markdown 塌成 `# 标题> 正文` |
    | 第二版 | `strip(" \\t")` 落袋 | `"。"` 之后的 `\n` 成纯空白新句被丢弃 |
    | 第三版 | 空白片段并入上一句 | **丢了换行前的正文**：覆盖率只剩 72%（4377/6066） |
    | 第四版 | 空白片段**不落袋、留在 buffer 里往下带** | 对 |

    ## 最终写法的两条规则

    1. **只在 `buffer` 里有非空白内容时才落袋**（`flush()`）。
       缓冲里的纯空白（连续空行、行首缩进）**不落袋、也不清空** —— 它属于下一句的开头。
    2. **每一片都 `+=` 到 buffer**，包括偶数片（分隔符之间的正文）。
       写成 `buffer = piece` 会把上一轮留下的空白覆盖掉 —— 那正是第三版丢换行的原因。

    这样 `"".join(sentences)` 严格等于 `text.strip()`，且不会产出空白句。
    """
    pieces = _SENTENCE_END.split(text)
    sentences: list[str] = []
    buffer = ""

    def flush() -> None:
        """把 buffer 落成一句。**纯空白的 buffer 不落袋、也不清空。**"""
        nonlocal buffer
        if buffer.strip():
            sentences.append(_rstrip_spaces_only(buffer))
            buffer = ""

    for index, piece in enumerate(pieces):
        buffer += piece  # 偶数片是正文，奇数片是边界本身 —— 两者都要进 buffer
        if index % 2 == 1:
            # **Markdown 表格行内部不切**：它含 `|` 与空格，在标点处切断会毁掉结构。
            # 实测：`…静默降分？ |` 被 `？` 切开后，行尾那个孤立的 `|`
            # 被下一步当空片段吞掉 —— 表格行少了收尾管道符，整行不再是合法表格行。
            if _inside_table_row(buffer):
                continue
            flush()

    flush()
    return sentences


def _inside_table_row(buffer: str) -> bool:
    """当前累积的文本是否**还在一行 Markdown 表格里**。

    判据：这一行的第一个非空白字符是 `|`，且还没出现换行。
    表格行的唯一结束方式是换行 —— 在行内任何句末标点处断句都是错的
    （`它会不会替我做重排？（我想要自己控制） |` 是**一整个表格单元格**）。

    代价：一个单元格里若有多句话，不会被拆开。这不是损失 ——
    表格单元格本就该整体保留（`Chunk` 那层已有"表格整张不切"的契约）。
    """
    current_line = buffer.rsplit("\n", 1)[-1]
    return current_line.lstrip().startswith("|")


def _rstrip_spaces_only(text: str) -> str:
    """去掉首尾空格/制表符，**但保住行尾的表格管道符**。

    `buffer.strip(" \t")` 对普通句子是对的，对 Markdown 表格行是错的：
    `"| a | b |"` 剥掉空格后 `|` 还在，但 `"…？ |"` 这种"标点紧贴管道符"
    的情形会被切碎 —— `？` 是边界，切开后剩 `" |"`，strip 就把它吃空，管道符消失。
    实测 `docs/w4-checklist.md` 的末段表格因此少了行尾 `|`（177→176 字）。

    所以：**行尾保留任何非空白的结尾字符**，只剥首尾空格。
    """
    stripped = text.strip(" \t")
    if stripped.endswith("|"):
        # 结尾是管道符：只剥开头，结尾原样
        return text.lstrip(" \t")
    return stripped


class SemanticChunking:
    """按句间向量相似度找断点，再合并成块。"""

    name = "semantic"

    def __init__(self, embedder: Embedder) -> None:
        """**embedder 必须显式传入，不给默认值。**

        与 `Retriever` 同一条理由：默认值会让"用了哪个模型切的"变成隐式状态。
        而语义切分的结论**完全由 embedder 决定**（hashing 下等于在噪声上找边界）——
        给它一个默认值，等于让这个前提条件从视野里消失。
        """
        self.embedder = embedder

    def chunk(self, docs: list[Document], cfg: ChunkingConfig) -> list[Chunk]:
        """切分。

        ## 为什么不跨 Document 合并

        递归字符与标题策略都在**单个 Document 内部**工作；语义策略看似可以跨页
        合并句子（"这句话没说完"），但那条路需要判断"上一页结尾是否未完"——
        那是 loader 已经做过的事（跨页断句被拼接、`pages=(6,7)` 记下来）。
        在这里再做一次只会重复推理，而且会把两个来源的句子混进同一块，
        让 `source` / `page` 归属变成猜测。

        所以：**逐 Document 切**，`pages` 从该 Document 继承。
        """
        if not docs:
            return []

        chunks: list[Chunk] = []
        for doc in docs:
            if not doc.text.strip():
                continue
            # 表格整张不切（契约要求）：按句切会把表格行列关系彻底打散
            if doc.is_table:
                chunks.append(
                    make_chunk(
                        text=doc.text.strip(),
                        doc=doc,
                        index=len(chunks),
                        strategy=self.name,
                        pages=inherited_pages(doc),
                        extra_metadata={"semantic_segments": 1, "semantic_breakpoints": 0},
                    )
                )
                continue

            # **代码块不参与语义切分**：句向量的相似度对代码行没有意义
            # （`def load():` 与 `    with open(...)` 在字面与语义上都"不像"，
            # 但它们是同一个逻辑单元）。代码走结构边界切（见 `chunking/code.py`）。
            if doc.is_code:
                from app.rag.chunking.code import split_code

                for block in split_code(doc.text):
                    chunks.append(
                        make_chunk(
                            text=block.text,
                            doc=doc,
                            index=len(chunks),
                            strategy=self.name,
                            pages=inherited_pages(doc),
                            extra_metadata={
                                "block": "code",
                                "code_language": block.language,
                            },
                        )
                    )
                continue

            segments, breakpoints = self._segment(doc.text, cfg)
            for segment in segments:
                chunks.append(
                    make_chunk(
                        text=segment,
                        doc=doc,
                        index=len(chunks),
                        strategy=self.name,
                        pages=inherited_pages(doc),
                        extra_metadata={
                            "semantic_segments": len(segments),
                            "semantic_breakpoints": breakpoints,
                            "embedder": self.embedder.name,
                        },
                    )
                )
        return chunks

    # ------------------------------------------------------------------ 内部

    def _segment(self, text: str, cfg: ChunkingConfig) -> tuple[list[str], int]:
        """返回 `(语义段列表, 被采纳的断点数)`。"""
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            # 一句话（或没有句子）：没有"句间"可言，整段一块
            return [text.strip()], 0

        vectors = self.embedder.embed(sentences)
        similarities = [cosine(vectors[i], vectors[i + 1]) for i in range(len(sentences) - 1)]

        break_indices = _break_indices(similarities, cfg.semantic_breakpoint_percentile)
        segments = _assemble(sentences, break_indices, cfg)
        return segments, len(break_indices)


def _break_indices(similarities: Sequence[float], percentile: float) -> list[int]:
    """相似度最低的 `percentile%` 位置 → 断点下标（断在 i 与 i+1 之间即返回 i）。

    ## 为什么分位数要对 `len(similarities)` 取整并至少给 1 个

    全取 0 会让语义策略退化成"整篇一块"（然后被 `max_chars` 硬切），
    等于这次 embedding 白花钱。所以下限 1：只要有两句以上，就允许一个断点。
    """
    if not similarities:
        return []
    ratio = max(0.0, min(100.0, percentile)) / 100.0
    count = max(1, int(round(len(similarities) * ratio)))
    count = min(count, len(similarities))

    # 取相似度最低的 count 个位置。**按下标排序**而不是按相似度排序：
    # 后面拼段要按原文顺序，顺序错了会把句子拼成乱序文本。
    lowest = sorted(range(len(similarities)), key=lambda i: similarities[i])[:count]
    return sorted(lowest)


def _assemble(
    sentences: Sequence[str], break_indices: Sequence[int], cfg: ChunkingConfig
) -> list[str]:
    """按断点拼段，并处理"过短合并"与"过长强制断开"两条兜底。"""
    breaks = set(break_indices)
    raw_segments: list[str] = []
    buffer = ""
    for index, sentence in enumerate(sentences):
        buffer += sentence
        if index in breaks:
            raw_segments.append(buffer)
            buffer = ""
    if buffer:
        raw_segments.append(buffer)

    # ---- 统一预算：`chunk_chars` 是三种策略共同的"目标块长"
    #
    # **这里曾经完全没有读 `chunk_chars`**（W5 周六发现）：
    # 只用了 `semantic_min_chars` / `semantic_max_chars` 两个私有兜底值。
    # 后果是**改 `chunk_chars` 对语义切分毫无影响** ——
    # 实测 300 / 500 / 800 三档切出来的块数完全相同（2002），
    # 也就是"三策略对比"里语义策略根本没参与参数比较，
    # 那场比较对它是**不公平**的（它被钉死在 max_chars=800 上）。
    #
    # 修法：把 `chunk_chars` 作为**上限**，两个私有兜底值作为次级约束：
    #   - 过长判定用 `min(chunk_chars, semantic_max_chars)`
    #     （semantic_max_chars 仍然是"单主题文档的强制断开点"，
    #       它不该超过全局预算）
    #   - `semantic_min_chars` 原样保留（"过短合并"与预算无关）
    budget = min(cfg.chunk_chars, cfg.semantic_max_chars)

    # 兜底 1：过短的段并回上一段（对策：并列条款每句都成断点）
    # **合并后要复查是否超预算**：否则一串短句会被并成一个远超 chunk_chars 的巨块，
    # 而那是"修了 A 引出 B"的典型。超了就另起一段。
    merged: list[str] = []
    for segment in raw_segments:
        if (
            merged
            and len(merged[-1]) < cfg.semantic_min_chars
            and len(merged[-1]) + len(segment) <= budget
        ):
            merged[-1] += segment
        else:
            merged.append(segment)

    # 兜底 2：过长的段在句子边界强制断开（对策：单主题技术文档永不到达断点）
    final: list[str] = []
    for segment in merged:
        if len(segment) <= budget:
            final.append(segment)
            continue
        # 按句子重新累加，到 budget 就断（**不截断句子**）
        buffer = ""
        for sentence in split_sentences(segment):
            if buffer and len(buffer) + len(sentence) > budget:
                final.append(buffer)
                buffer = sentence
            else:
                buffer += sentence
        if buffer:
            final.append(buffer)

    return [segment.strip() for segment in final if segment.strip()]


__all__ = ["SemanticChunking", "split_sentences"]
