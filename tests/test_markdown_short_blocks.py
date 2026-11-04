"""Markdown 短块合并的测试 —— **每一条都来自一次实测踩坑**。

## 背景：线上检索质量事故（2026-11-04）

用户把一份 21.9 KB 的 Markdown 报告（《工业缺陷生成模型全链路报告》）
上传到 Streamlit，问「缺陷有哪些」，系统拒答，理由说得很清楚：

    「上下文仅包含报告标题、目录、数据集来源和预处理流程标题等信息」

**它没说错** —— 检索回来的 5 块里，2 块是纯标题、1 块是图片路径。

本地复现后定位到：文档被切成 **113 块**，其中 **17 块短于 40 字符**：

    ---                       (3 字符)
    **训练命令**：             (9 字符)
    **预期训练进度**：          (11 字符)
    ## 附录：项目文件索引       (12 字符)

## 为什么短块是**系统性**灾难（不是"稍微有点噪音"）

字面检索（hashing / BM25）算字符重叠比例。9 字符的块里 4 个实词占绝大比重，
500 字符的正文被大量无关字稀释 —— **短块天然赢**。实测：

    问题「训练损失怎么变化」
      #1  **训练命令**：       (9 字符)
      #2  **预期训练进度**：    (11 字符)
      #4  ### 5.3 训练损失曲线  ← 真正的答案排第 4

这不是排序略有偏差，是**越没内容的块排得越前**。

## 修复后（同一份文档实测）

    105 块 -> 61 块
    短块 17 个 -> 0 个
    最短块 3 字符 -> 40 字符
    「训练损失怎么变化」正确答案从第 4 名 -> 第 2 名
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.loaders.markdown import (
    MIN_CHUNK_CHARS,
    TOC_ANCHOR_RATIO,
    _drop_image_only_lines,
    _is_toc_block,
    _merge_short_blocks,
    load_markdown,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ============================================================ 图片行


def test_image_only_lines_are_dropped() -> None:
    """整行只有图片语法的行要删掉。

    图片对问答零价值（模型看不了图），但在**字面检索**里有害：
    `![图3-2 训练损失曲线](figures/x.png)` 含「训练损失曲线」五个实词，
    会挤进 Top-5 占掉正文的位置。

    实测：这个图片块排在 `缺陷有哪些` 的第 5 名（score 0.089）。
    """
    text = "\n".join(
        [
            "### 5.3 训练损失曲线",
            "",
            "![图3-2 训练损失曲线](report_figures/fig3_2_loss_curve.png)",
            "- 初始 loss：0.1387",
        ]
    )
    out = _drop_image_only_lines(text)

    assert "report_figures" not in out, "图片路径应当被删掉"
    assert "初始 loss" in out, "正文不能受影响"
    assert "### 5.3 训练损失曲线" in out, "标题不能受影响"


def test_image_inline_in_sentence_is_kept() -> None:
    """**混在句子里的图片不能删** —— 它是句子的一部分。

    `见图 3-2：![曲线](x.png)` 删掉图片会留下「见图 3-2：」这种断句。
    判据是"这一行除了图片什么都没有"。
    """
    text = "见图 3-2：![曲线](x.png) 显示 loss 从 0.14 降到 0.09"
    out = _drop_image_only_lines(text)
    assert out == text, "整行不是纯图片，应当原样保留"


# ============================================================ 短块合并


def test_short_blocks_are_merged_away() -> None:
    """短于 `MIN_CHUNK_CHARS` 的文本块必须被合并掉。

    这条是整个修复的核心目的：**不让短块成为独立检索候选**。
    """
    pieces = [
        (1, "text", "# 报告标题"),  # 短
        (2, "text", "## 数据"),  # 短
        (2, "text", "原始数据来源于工业产线拍摄的大图，包含各类产品缺陷，共约三千张。"),  # 长
    ]
    merged = _merge_short_blocks(pieces)

    assert len(merged) == 1, f"三个短块应当并成一块，实际 {len(merged)} 块：{merged}"
    text = merged[0][2]
    assert "# 报告标题" in text and "## 数据" in text and "原始数据" in text, (
        "合并要保留所有内容（不能丢），只是不再独立成块"
    )


def test_merged_block_keeps_shallowest_level() -> None:
    """合并后层级取**更浅**的（数字更小 = 更高层）。

    ## 这条来自一次真实的旧测试失败

    `# 付款计划`（level 1）是短标题，被并进 `## 一、里程碑`（level 2）后
    层级变成了 2，于是 `test_markdown_heading_levels_go_into_metadata`
    里 `assert 1 in levels` 红了。

    语义上取更浅的才对：合并后的块代表"付款计划下的里程碑"，
    它的**主标题**是 level 1。取深的会让下游以为整块都在二级标题下，
    摘要时丢失文档的顶层结构。
    """
    pieces = [
        (1, "text", "# 付款计划"),
        (2, "text", "## 一、里程碑\n下表列出各期交付的时间与金额，含金额与日期等完整信息。"),
    ]
    merged = _merge_short_blocks(pieces)
    assert len(merged) == 1
    assert merged[0][0] == 1, f"应当保留更浅的层级 1，实际 {merged[0][0]}"


def test_table_block_stays_independent() -> None:
    """**表格必须保持独立**（不能被短块吸收，也不能吸收别人）。

    ## 这条也是踩出来的

    第一版让短标题无条件吸收后一块，把
    `### 1.3 缺陷类型分布`（14 字符）和后面那张表**拼成一块**，
    于是 `test_markdown_table_is_separate_block` 红了。

    下游切分器靠 `block="table"` 决定"这张表不许切碎" ——
    表格一旦被拼进正文块，切分器就会按中文句读去切它，
    `| 类型 | 数量 |` 这种行会被切成两半。
    """
    pieces = [
        (2, "text", "### 1.3 缺陷类型分布"),  # 短标题
        (2, "table", "| 类型 | 数量 |\n| --- | --- |\n| type 0 | 219 |"),  # 表格
        (2, "text", "从上表可见各类缺陷的分布情况，其中 type 0 占比最高，值得关注。"),
    ]
    merged = _merge_short_blocks(pieces)

    tables = [p for p in merged if p[1] == "table"]
    assert len(tables) == 1, f"表格应当仍是一个独立块，实际：{merged}"
    # 表格内容必须是**原来的表格**，没有被拼入标题
    assert tables[0][2].startswith("| 类型 | 数量 |"), f"表格内容被污染了：{tables[0][2][:50]}"


def test_short_block_skips_table_to_merge_forward() -> None:
    """短块向后找目标时，要**跳过表格**去找下一个文本块。

    ## 这是第二版修复（第一版太保守）

    第一版规则是"后一块是表格就不合并" -> 真实文档从 51 块**回退到 80 块**，
    短块从 0 涨回 29 个。因为 Markdown 报告里"标题 + 表格"的排列非常常见，
    于是大量短标题都被跳过了。

    正确做法：向后**找最近的普通文本块**，把表格原样留在原位。
    """
    pieces = [
        (2, "text", "### 1.3 分布"),  # 短标题
        (2, "table", "| a | b |\n| --- | --- |\n| 1 | 2 |"),  # 表格挡在中间
        (2, "text", "表中列出各项数据的具体数值与占比，可对照查看整体分布情况。"),
    ]
    merged = _merge_short_blocks(pieces)

    # 结果应是 [表格, 合并后的文本] 两块 —— 标题被并到后面那段正文上
    assert len(merged) == 2, f"应当 2 块（表格 + 合并文本），实际 {len(merged)}：{merged}"
    assert merged[0][1] == "table", "表格位置应当在前面且类型不变"
    assert merged[1][1] == "text"
    assert "### 1.3 分布" in merged[1][2], "短标题应当并进后面的正文"
    assert "表中列出" in merged[1][2]


def test_no_content_is_lost_in_merging() -> None:
    """**合并绝不能丢内容** —— 这是最底线的不变量。

    所有原始文本的字符都必须还在（允许增加换行分隔符）。
    """
    pieces = [
        (1, "text", "# 标题"),
        (2, "text", "## 一节"),
        (2, "text", "## 二节"),
        (2, "text", "这是一段足够长的正文内容，用于验证合并不丢失任何字符，包括标点符号。"),
    ]
    merged = _merge_short_blocks(pieces)
    joined = "\n".join(p[2] for p in merged)
    for _, _, original in pieces:
        assert original in joined, f"内容丢了：{original!r}"


def test_empty_input_is_safe() -> None:
    """空输入返回空列表，不抛异常。"""
    assert _merge_short_blocks([]) == []


def test_merge_loop_terminates() -> None:
    """**极端输入不能让合并死循环**。

    如果整篇都是超短块（比如全是 `---`），无轮数上限就会转不完。
    实现里有 `for _ in range(10)` 兜底，这条测试用 50 个短块压它。
    """
    pieces = [(1, "text", "---") for _ in range(50)]
    merged = _merge_short_blocks(pieces)  # 不能挂
    assert isinstance(merged, list)


# ============================================================ 真实文档回归


def test_real_report_has_no_short_blocks() -> None:
    """**端到端回归**：用那份真实报告验证，短块数必须为 0。

    这份文档就是线上事故的原始输入（`D:\\git\\实践\\test\\工业缺陷生成
    模型全链路报告.md`）。测试里用 fixture 副本，避免依赖仓库外的绝对路径。
    """
    fixture = FIXTURES / "report_short_blocks.md"
    if not fixture.exists():
        pytest.skip("缺少 fixture：report_short_blocks.md")

    result = load_markdown(fixture)
    lengths = [len(d.text) for d in result.documents]

    short = [n for n in lengths if n < MIN_CHUNK_CHARS]
    assert not short, (
        f"仍有 {len(short)} 个短块（最短 {min(short)} 字符）—— "
        "短块在字面检索里会挤掉正文，见本文件 docstring"
    )


# ============================================================ 目录块


def test_toc_block_is_detected() -> None:
    """目录块要能被识别出来。

    ## 为什么目录比短块更危险

    短块靠"短"泄露身份，目录块**又长又像正文**：
    它以 `## 目录` 开头，后面是一串章节名，看起来完全正常。

    但它是**最泛的一块** —— 文档里每个主题词都在里面出现过，
    所以不管问什么，它在字面检索里都排第一。

    实测（那份报告）：问「用了什么基座模型」，
    目录块排 #1（0.259），真正的答案
    `#### 3.2.3 Text Encoder（CLIP ViT-L/14）` 排 #5。
    """
    toc = (
        "## 目录\n\n"
        "1. [数据集](#1-数据集)\n"
        "2. [环境配置](#2-环境配置)\n"
        "3. [模型架构](#3-模型架构)\n"
        "4. [主模型代码](#4-主模型代码)\n"
    )
    assert _is_toc_block(toc) is True


def test_normal_paragraph_with_one_anchor_is_not_toc() -> None:
    """正文里偶尔出现一个锚点链接**不能**被误判成目录。

    判据是**占比**（`TOC_ANCHOR_RATIO`）而不是"含不含锚点" ——
    否则"详见[第 3 节](#3-模型架构)"这类交叉引用会被整段丢掉。
    """
    body = (
        "## 3.2.3 Text Encoder\n\n"
        "将文本 prompt 编码为 UNet 可用的条件向量。\n"
        "关于位置编码的实现，详见[第 3 节](#3-模型架构)。\n"
        "本模块使用 CLIP ViT-L/14 的文本编码器部分，参数量约 123M。\n"
    )
    assert _is_toc_block(body) is False, "正文段落被误判成目录了"


def test_toc_is_excluded_from_index_but_reported() -> None:
    """目录块要**不进入索引**，且必须**在 warnings 里说出来**。

    ## 为什么要"说出来"

    静默丢弃是危险的：用户发现"块数比预期少"会以为有 bug；
    或者问"文档分几章"得不到答案却不知道为什么。

    一条 warning 让这个行为可见（UI 会显示在"当前知识库"下面）。

    ## 为什么不是"降级链接后继续索引"（实测否定）

        修复前：#1 目录块 score=0.259
        只降级：#1 目录块 score=0.255   <- 排名没动，分数几乎没变

    拉高分的是**章节名本身**（「模型架构」「生成结果」与问题里的「模型」重合），
    不是 `#` 符号。
    """
    import tempfile

    content = (
        "# 报告\n\n"
        "## 目录\n\n"
        "1. [数据集](#1-数据集)\n"
        "2. [模型架构](#2-模型架构)\n"
        "3. [生成结果](#3-生成结果)\n"
        "4. [实现过程](#4-实现过程)\n\n"
        "## 1. 数据集\n\n"
        "原始数据来源于工业产线拍摄的大图，包含各类产品缺陷，共约三千张图像。\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as fh:
        fh.write(content)
        path = Path(fh.name)

    try:
        result = load_markdown(path)
        texts = "\n".join(d.text for d in result.documents)

        # 目录的章节名列表不该作为独立可检索内容出现
        assert "1. [数据集]" not in texts, "目录块没被排除"
        # 但正文要在
        assert "原始数据来源于工业产线" in texts, "正文被误删了"
        # 必须报告这件事
        assert any("目录" in w for w in result.warnings), (
            f"跳过目录块必须写进 warnings（否则用户不知道），实际：{result.warnings}"
        )
    finally:
        path.unlink(missing_ok=True)


def test_toc_ratio_threshold_is_reasonable() -> None:
    """阈值本身要合理（防止有人随手改成 0.0 或 1.0）。

    - `0.0` -> 任何含锚点的块都被当目录（会误删正文）
    - `1.0` -> 只有 100% 全是链接才认（目录里带序号/标题就漏掉）
    """
    assert 0.3 <= TOC_ANCHOR_RATIO <= 0.8, (
        f"TOC_ANCHOR_RATIO={TOC_ANCHOR_RATIO} 不在合理区间 [0.3, 0.8]"
    )


def test_real_report_has_toc_removed() -> None:
    """**真实验证**：那份报告的目录块必须被跳过。

    修复前后的对照（同一份文档、同一个问题「用了什么基座模型」）：

        修复前  #1 目录块（0.259）
        修复后  #1 `### 5.4 速度优化对比 / ## 6. 5 模型对比图 …`（0.181）

    目录不再霸占第一名。**注意：真正的答案仍未进前三** ——
    这是 `hashing` 的能力边界（它按字符重叠排序，不理解"基座模型"
    与"Text Encoder (CLIP ViT-L/14)"是同一件事）。
    这条测试只锁定"目录不再污染"，不假装语义检索已经好了。
    """
    fixture = FIXTURES / "report_short_blocks.md"
    if not fixture.exists():
        pytest.skip("缺少 fixture：report_short_blocks.md")

    result = load_markdown(fixture)
    joined = "\n".join(d.text for d in result.documents)
    assert "(#1-数据集)" not in joined, "目录的锚点链接仍在索引内容里"
    assert any("目录" in w for w in result.warnings), "应当报告跳过了目录块"
