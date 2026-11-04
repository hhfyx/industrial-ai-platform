"""**文档与事实一致性**测试：防止"文档说通了、实际没通"。

## 为什么文档需要测试

这个项目已经吃过一次亏：`docs/deploy-notes.md` 里写着
"在 Advanced settings 里指定 requirements 文件"，而**那个选项不存在**。
文档说得斩钉截铁，实际从来没人成功过 —— 这种错误在无人部署前
**没有任何症状**。

部署成功后我又写了大量"已验证"的结论（线上链接、逐项结果）。
这些结论**要么是真的跑过，要么是编的**，而读文档的人无法分辨。

所以这里给关键声明加断言：**要么文档和代码一致，要么测试红。**

## 但不能过度

文档里的**实测数据**（如"105 块 → 60 块"）不该由测试重新推导 ——
那是某一次具体运行的记录，测试环境不同就可能不同。
这里只锁**结构性事实**（链接存在、常量的值、文件名）。
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_NOTES = PROJECT_ROOT / "docs" / "deploy-notes.md"
README = PROJECT_ROOT / "README.md"


def test_deploy_notes_records_the_live_url() -> None:
    """部署文档必须写着**真实的线上地址**。

    没有地址的"已部署"是不可验证的声称 —— 读者没法打开看一眼。

    这条也顺便保证：如果哪天应用被删了或换了地址，改文档时不会漏。
    """
    content = DEPLOY_NOTES.read_text(encoding="utf-8")
    urls = re.findall(r"https://[a-z0-9-]+\.streamlit\.app", content)
    assert urls, "deploy-notes.md 没有记录线上地址 —— 写了'已部署'却没有地址，读者无法验证"
    assert any("industrial-ai-platform" in u for u in urls), f"线上地址看起来不是本项目的：{urls}"


def test_readme_links_to_the_live_app() -> None:
    """README 首屏要给线上链接 —— 这是这个项目最有力的证据。

    简历/面试场景下，"有个能点开的链接"比任何数字都直接。
    """
    content = README.read_text(encoding="utf-8")
    assert "streamlit.app" in content, "README 里没有线上链接"


def test_deploy_notes_states_what_is_unverified() -> None:
    """部署文档必须保留**未验证清单**，不能只剩好消息。

    ## 这条防的是什么

    修完一堆问题之后，很自然的冲动是把"未验证"段落删掉 ——
    反正都通了。但**未验证项不会因为不写就消失**：
    并发内存、磁盘配额、冷启动、大文档，一个都没测。

    留着它们的作用不是自谦，是**给下一个人（或三个月后的自己）
    一个准确的起点** —— 否则他会以为这些也验过了。
    """
    content = DEPLOY_NOTES.read_text(encoding="utf-8")
    assert "未验证" in content or "未测" in content, (
        "deploy-notes.md 里没有'未验证/未测'段落 —— 是不是被'已完成'冲掉了？"
    )
    # 至少要点名这几项（都是真实没测的）
    for item in ("并发", "磁盘"):
        assert item in content, f"未验证清单里应当提到「{item}」"


def test_documented_constants_match_code() -> None:
    """文档里写的常量值必须和代码一致。

    `MIN_CHUNK_CHARS = 40` / `TOC_ANCHOR_RATIO = 0.5` 这两个数
    在文档里被详细解释过（为什么是 40 不是 30）。
    如果有人改了代码没改文档，读者会按错的数字去理解行为。
    """
    from app.rag.loaders.markdown import MIN_CHUNK_CHARS, TOC_ANCHOR_RATIO

    content = DEPLOY_NOTES.read_text(encoding="utf-8")
    assert f"MIN_CHUNK_CHARS = {MIN_CHUNK_CHARS}" in content, (
        f"deploy-notes.md 里的 MIN_CHUNK_CHARS 与代码（{MIN_CHUNK_CHARS}）不一致"
    )
    assert f"TOC_ANCHOR_RATIO = {TOC_ANCHOR_RATIO}" in content, (
        f"deploy-notes.md 里的 TOC_ANCHOR_RATIO 与代码（{TOC_ANCHOR_RATIO}）不一致"
    )


def test_deploy_notes_does_not_claim_unverified_as_verified() -> None:
    """**反向检查**：不能把"未验证"的项目写成已验证。

    这条是价值观测试 —— 它检查特定几个真的没测的项目，
    不能被标成 ✅。

    实测没做过的：并发内存、磁盘配额、冷启动、大文档。
    """
    content = DEPLOY_NOTES.read_text(encoding="utf-8")

    # 找出"仍未验证/未测"表格里的行
    unchecked_items = ["并发", "磁盘", "冷启动", "大文档"]
    for item in unchecked_items:
        # 该行应当带 ❌，不能是 ✅
        pattern = re.compile(rf"\|[^|\n]*{item}[^|\n]*\|([^|\n]*)\|")
        match = pattern.search(content)
        if match is None:
            continue  # 该行可能写法不同，交给上面那条测
        status = match.group(1)
        assert "✅" not in status, (
            f"「{item}」被标成了已完成，但它**从没测过** —— 这是最容易自欺的一类错误"
        )
