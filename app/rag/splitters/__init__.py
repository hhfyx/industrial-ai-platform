"""切分：`Document` → `Chunk`。索引链路的第一道工序。

**这一层只做一件事：把长文本切成"检索时能被单独命中"的块。**

不在这里做的事（各有归属，混进来就是将来换不掉的原因）：
- 不做 embedding（那是 `embedding/`，换模型不该动切分逻辑）
- 不做入库（那是 `stores/`，换库不该动切分逻辑）
- 不做清洗（那是 `loaders/`，原始文本的边界一旦模糊就无法排查"是模型读错还是我们切错"）

**为什么强调"切成能被单独命中"**：块太大，一个块里塞了三种主题，命中后答案不准；
块太小，一句话被切成两半，两半各自都不完整。所以 `ChunkSize` 的默认值
取 500 字符 / 50 字符重叠 —— 这个量级是"一段完整论述"的常见长度，
不是拍脑袋：一份 A4 合同条款段通常 200–600 字，取 500 让大多数条款**整段落在一个块里**。

**为什么必须有重叠**：跨块的信息断裂只能靠重叠缓解。一个"若甲方未按期付款，
则乙方有权…"的条件句被切在中间时，重叠至少让下一块还能看到条件的前半句。
代价是存储涨约 10%（50/500），检索可能返回近重复块 —— 后者用 id 去重即可，
前者是必须付的钱。
"""

from __future__ import annotations

from app.rag.splitters.base import ChunkSize, Splitter
from app.rag.splitters.recursive import RecursiveSplitter

__all__ = ["ChunkSize", "RecursiveSplitter", "Splitter"]
