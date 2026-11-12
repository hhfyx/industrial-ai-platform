"""v2 冷端点：检索-only（零 LLM）+ readiness 探针（W12 周一 14:30–15:30）。

## 为什么要有检索-only 端点（冷热分离的"冷"）

压测要把"基础设施延迟"与"模型延迟"分开：
模型路径（ask/extract/summarize）的首 token 延迟 99% 在模型侧（W5 实测，
检索只占 1%）——混在一起压，得到的分位延迟是模型网关的性能曲线，
不是**你的服务**的。冷端点走完整的 embed → search → TopK 链路、
**不调一次模型**，它的 P99 才是"服务本身的延迟"。

冷端点也是压测的**负载发生器标定器**：先压冷端点摸到服务层的吞吐上限，
再压热端点看下降多少 —— 差值就是"加模型之后服务的真实代价"。

## liveness / readiness 分开（必答问题的落点）

- `/health`（已有，保留）：liveness —— 进程活着就 200。
  模型加载 30 秒期间它**必须**是 200，否则编排器在加载窗口反复杀容器，
  形成"永远起不来"死循环（W8 冷启动问题在服务层的正式形态）。
- `/ready`（本文件新增）：readiness —— **模型已加载**才 200，否则 503。
  编排器拿它决定"该不该把流量切进来"；liveness 决定"要不要重启"。
  两个问题、两个探针、两个端口语义，混用一个就死循环。

readiness 的实现不假设"模型"是哪一个：构造极重的组件（SD 权重、bge 模型）
在 `get_generator()`/`get_extractor()` 首次调用时才真正加载（lru_cache 单例），
所以 readiness 检查的是**依赖的单例是否已经构造成功** —— 构造过且没抛 = 就绪。
未构造时不能去触发构造（那会把 readiness probe 变成加载触发器，
一个探针请求把 30 秒的加载提前拉起 —— 探针的职责是观察，不是参与）。

## 鉴权边界

`/ready` 在 `security.PUBLIC_PATH_PREFIXES` 白名单里：
探针请求不该带业务凭证，也不该消耗配额 —— 否则压测打到一半
配额耗尽，readiness 翻 503，编排器把正在压测的服务摘除，
压测报告里多出一段谁也解释不了的空窗。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Final

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field

from app.core.errors import AppError
from app.core.security import remaining_header_for, require_api_key

logger = logging.getLogger(__name__)

# 归因实验开关（W12 周四）：正文实现瘦身 120 / 缓存恒开；
# SEARCH_SUMMARY_CHARS / SEARCH_CACHE_OFF 只为归因矩阵能分批测
# （scripts/attribution_experiment.py），默认值 = 正式形态。
SUMMARY_CHARS: Final[int] = int(os.environ.get("SEARCH_SUMMARY_CHARS", "120"))
CACHE_OFF: Final[bool] = os.environ.get("SEARCH_CACHE_OFF", "") == "1"

# 归因实验开关（W12 周四）：正文实现瘦身 120 / 缓存恒开；
# SEARCH_SUMMARY_CHARS / SEARCH_CACHE_OFF 只为归因矩阵能分批测
# （scripts/attribution_experiment.py），默认值 = 正式形态。
SUMMARY_CHARS: Final[int] = int(os.environ.get("SEARCH_SUMMARY_CHARS", "120"))
CACHE_OFF: Final[bool] = os.environ.get("SEARCH_CACHE_OFF", "") == "1"

router = APIRouter(prefix="/api/v2", tags=["v2"])

READY_DESC: Final[str] = (
    "readiness 探针：**依赖已就绪**（模型单例构造成功）才 200，否则 503。\n\n"
    "与 `/health`（liveness）的区别：liveness 只回答「进程活着吗」，"
    "模型加载期间也是 200；readiness 回答「能接流量吗」，加载完成前必须 503。"
    "两个探针混用 = 编排器在加载窗口反复重启容器（W8 冷启动死循环）。"
)

READY_ERROR_RESPONSES = {
    503: {"description": "依赖未就绪（模型未加载或加载失败）"},
}


class ReadyResponse(BaseModel):
    """readiness 返回体。形状参考 /metrics：给人看 + 给压测脚本判断。"""

    status: str = Field(description='"ready" 或 "not_ready"')
    checks: dict[str, bool] = Field(description="各依赖的就绪状态：generator / extractor")
    elapsed_ms: int = Field(description="本探针执行耗时（毫秒）")


def _singleton_loaded(module: str, name: str) -> bool:
    """探测 lru_cache 单例是否已构造（不触发构造）。

    `cache_info().currsize` 是 lru_cache 的官方内省：>0 = 有缓存实例。
    **绝不调用函数本身** —— 那会把探针变成加载触发器（见模块 docstring）。
    """
    import importlib

    try:
        mod = importlib.import_module(module)
        fn = getattr(mod, name, None)
        if fn is None or not hasattr(fn, "cache_info"):
            return False
        return fn.cache_info().currsize > 0
    except Exception:  # noqa: BLE001 - 探针绝不能自己炸：炸了按"未就绪"处理
        return False


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="readiness 探针（依赖就绪检查）",
    description=READY_DESC,
    responses=READY_ERROR_RESPONSES,
)
async def ready(response: Response) -> ReadyResponse:
    """readiness：检查依赖单例是否已构造，未就绪抛 503。

    ## checks 的两个维度

    - `generator`：图像生成器（SD 权重，最重的那个）；
    - `extractor`：抽取器（LLM client 构造，网络栈 + 凭据解析）。

    两个都就绪才 200。只关心其中一个的调用方读 `checks` 字典 ——
    返回体给事实，状态码给结论，与 degraded 的设计同一条哲学。
    """
    started = time.perf_counter()
    checks: dict[str, bool] = {
        "generator": _singleton_loaded("app.core.deps", "get_generator"),
        "extractor": _singleton_loaded("app.core.deps", "get_extractor"),
    }
    elapsed = max(int((time.perf_counter() - started) * 1000), 1)
    if not all(checks.values()):
        raise AppError(
            code="NOT_READY",
            message="依赖未就绪（模型尚未加载完成），请稍后重试",
            status_code=503,
            headers={"Retry-After": "5"},
        )
    response.headers["Cache-Control"] = "no-store"
    return ReadyResponse(status="ready", checks=checks, elapsed_ms=elapsed)


# ---------------------------------------------------------------- 检索-only（冷端点）


class SearchRequest(BaseModel):
    """检索-only 请求体。与 QA 的差别只有一条：没有要回答的问题，只有查询词。"""

    query: str = Field(min_length=1, max_length=500, description="检索词")
    k: int = Field(default=3, ge=1, le=8, description="返回条数（与 retrieve_spec_chunks 同上限）")


class SearchHitItem(BaseModel):
    """单条命中（**瘦身版**，W12 周四优化一）。

    ## 为什么 text 从 500 字缩到 120 字摘要

    压测分段表（docs/performance.md §四）的最大段是序列化/中间件 ~50% ——
    响应体里 3 条 × 500 字的 chunk 全文是序列化成本的大头。检索-only 的
    主要用途（压测标定 / 命中质量探针 / 热端点开销对比）只需要"命中的是
    什么"的判断依据，不需要全文；**全文另接口**（GET /search/chunk/{id}，
    回查 get_by_content_id）—— "列表轻、详情重"是列表页的标准形态。
    """

    id: str
    source: str
    page: int | None = None
    score: float
    text: str = Field(description="摘要（前 120 字）；全文走 GET /search/chunk/{id}")


class SearchResponse(BaseModel):
    """检索-only 响应。**零 LLM 字段**：没有 answer、没有 confidence 的模型分。"""

    request_id: str
    hits: list[SearchHitItem]
    elapsed_ms: int


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="检索-only（零 LLM 冷端点，压测专用）",
    description=(
        "走完整检索链路（embed → search → TopK），**不调用任何模型**。\n\n"
        "用途：\n"
        "- 压测服务层延迟上限（P99 不含模型侧）；\n"
        "- 标定热端点的模型开销（热 − 冷 = 模型代价）；\n"
        "- 检索质量的回归探针（改切块器/换 embedder 时用它看命中率）。\n\n"
        "查不到时返回 200 + 空 hits（「没有」是事实不是错误，与 retrieve_spec_chunks 同一语义）。"
    ),
    responses={422: {"description": "字段校验失败（INVALID_INPUT）"}},
)
async def search_only(
    payload: SearchRequest,
    request: Request,
    _: str = Depends(require_api_key),
) -> SearchResponse:
    """检索-only 入口。复用 real_tools 的内置语料检索器（同一 Retriever 协议）。

    检索器单例放在函数内 import：`real_tools` 的装配是 W11 已验证的
    "真管线 + 内置语料"，这里不复制装配逻辑 —— 复制是漂移的起点。

    `request_id` 从 request.state 读（中间件已写，与 RequestIdDep 同源）——
    不另起依赖参数是因为本路由的 Depends 槽位已用于鉴权依赖。
    """
    from app.core.deps import get_request_id
    from app.core.llm.real_tools import get_shared_retriever
    from app.rag.result_cache import (
        CachedRetrieval,
        ResultCache,
        fingerprint_from_rag_config,
        get_result_cache,
    )

    rid = get_request_id(request)
    started = time.perf_counter()
    retriever = get_shared_retriever()

    # 结果缓存接入（W12 周四优化二）：cold 端点此前每次都走全链路
    # embed+search，QAEngine 路径有检索层缓存而这里没有 —— 两个入口
    # 一个有缓存一个没有，就是"压测口径与服务口径不一致"的来源。
    # key 与 CachedQAEngine 同一构造（规范化 query + 配置指纹 + k），
    # 命中回查 store（同一条降级纪律：回查失败降级全链路）。
    # CACHE_OFF 开关只服务归因实验，正文恒开。
    config = retriever.config
    cache: ResultCache = get_result_cache()
    config_fp = fingerprint_from_rag_config(config, embedder_name=retriever.embedder.name)
    key = ResultCache.retrieval_key(payload.query, config_fingerprint=config_fp, k=payload.k)

    if not CACHE_OFF:
        cached = cache.get_retrieval(key)
        if cached is not None:
            chunks = [retriever.store.get_by_content_id(cid) for cid in cached.chunk_ids]
            if chunks and all(c is not None for c in chunks):
                elapsed = max(int((time.perf_counter() - started) * 1000), 1)
                hits = [
                    SearchHitItem(
                        id=chunk.id,
                        source=chunk.source,
                        page=chunk.page,
                        score=round(score, 4),
                        text=chunk.text[:SUMMARY_CHARS],
                    )
                    for chunk, score in zip(chunks, cached.scores, strict=True)
                ]
                return SearchResponse(request_id=rid, hits=hits, elapsed_ms=elapsed)
            logger.info("cold 端点缓存命中但回查失败，降级全链路: %s", payload.query[:40])

    result = retriever.retrieve(payload.query, k=payload.k)
    elapsed = max(int((time.perf_counter() - started) * 1000), 1)
    # 写缓存（与 CachedQAEngine 同一形状；空结果也写 = 负缓存语义）
    if not CACHE_OFF:
        cache.put_retrieval(
            key,
            CachedRetrieval(
                chunk_ids=tuple(h.chunk.id for h in result.hits),
                scores=tuple(result.scores()),
                top_k=result.top_k,
                where=tuple(sorted((k2, str(v)) for k2, v in (result.where or {}).items())),
            ),
        )

    hits = [
        SearchHitItem(
            id=hit.chunk.id,
            source=hit.chunk.source,
            page=hit.chunk.page,
            score=round(hit.score, 4),
            text=hit.chunk.text[:SUMMARY_CHARS],
        )
        for hit in result.hits
    ]
    return SearchResponse(request_id=rid, hits=hits, elapsed_ms=elapsed)


@router.get(
    "/rate-limit/status",
    response_model=None,
    summary="当前 key 的配额状态（调试用）",
    description="返回当前调用方 key 的剩余令牌数与其被拒次数（按指纹，无明文 key）。",
)
async def rate_limit_status(
    request: Request,
    _: str = Depends(require_api_key),
) -> dict[str, Any]:
    """配额状态。metrics 的 key 维度视图 —— "谁在挤谁"从这里看。"""
    from app.core.security import get_rate_limiter

    limiter = get_rate_limiter()
    remaining = remaining_header_for(request)
    fp = getattr(request.state, "rate_limit_fingerprint", None)
    return {
        "remaining": remaining,
        "rejected_by_this_key": limiter.rejected_by_key.get(fp, 0) if fp else 0,
        "tracked_keys": len(limiter.key_buckets),
        "total_rejections": sum(limiter.rejected_by_key.values()),
    }


__all__ = ["router"]
