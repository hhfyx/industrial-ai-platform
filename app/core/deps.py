"""依赖注入：把"用哪个抽取器"变成一行装配代码。

**这个文件是 W2 换实现的唯一开关。**
路由只声明 `ExtractorDep`，不知道底下是正则还是 LLM；换实现时只改这里的一行，
路由、schema、契约文档、测试全都不用动。

`get_extractor` 为什么是 `lru_cache` 的单例：抽取器的构造（W2 时是模型 client）
很贵，绝不能每个请求造一个。规则版构造很便宜，但**接口一致比当下省事更重要** ——
现在省一行，W2 就得返工一层。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from app.core.errors import REQUEST_ID_ATTR
from app.services.extractor import RuleExtractor


@lru_cache
def get_extractor() -> RuleExtractor:
    """进程级单例。W2 换实现时把这里改成 `return LlmExtractor(...)` 即可。

    返回类型标注是 `RuleExtractor` 而路由只依赖 `extract()` 方法 ——
    W2 换成 `LlmExtractor` 后请把它抽成共同的 Protocol，那一步再做，
    现在写 Protocol 属于为不存在的第二个实现付设计成本。
    """
    return RuleExtractor()


ExtractorDep = Annotated[RuleExtractor, Depends(get_extractor)]


def get_request_id(request: Request) -> str:
    """读中间件写进 `request.state` 的 id。

    **不要在这里生成**：生成会跟日志里的 id 对不上（中间件那次才是被记录的）。
    中间件一定先跑，所以这里直接取；取不到说明中间件被摘了，属于装配错误。
    """
    return str(getattr(request.state, REQUEST_ID_ATTR, ""))


RequestIdDep = Annotated[str, Depends(get_request_id)]


__all__ = ["ExtractorDep", "RequestIdDep", "get_extractor", "get_request_id"]
