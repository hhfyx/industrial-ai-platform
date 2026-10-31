"""兼容导出层：`RecursiveSplitter` 的旧 import 路径。

W4 周五把实现搬到了 [`delimiter.py`](delimiter.py) 并改名为 `DelimiterSplitter`
（名字反映真实核心：**中文分隔符优先级表**，不是"递归"这个技巧）。

但周三的代码、文档、测试引用的是 `app.rag.splitters.recursive` 这个路径。
删掉它会让 `from app.rag.splitters.recursive import RecursiveSplitter` 直接
`ModuleNotFoundError` —— 而那条 import 分散在测试、脚本和文档链接里。

**所以这里留一层转发，而不是去改所有引用**：改引用是机械劳动，
风险是漏改一处；转发层是一行赋值，成本是零。

**什么时候该删**：当下没有任何地方 import 它、文档里也不再提它时。
`tests/test_loaders.py::test_recursive_splitter_alias_still_works` 守着这条转发，
如果哪天真的删了，那条测试会红，删的人必须显式把测试一起删 ——
**这就是"兼容层要有测试"的意义：它让删除变成一个决定，而不是一次意外。**
"""

from __future__ import annotations

from app.rag.splitters.delimiter import DEFAULT_SEPARATORS, DelimiterSplitter

# 同一个类，不是子类。子类会让 isinstance 判断出现两种答案。
RecursiveSplitter = DelimiterSplitter

__all__ = ["DEFAULT_SEPARATORS", "RecursiveSplitter"]
