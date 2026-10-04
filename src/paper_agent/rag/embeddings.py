# -*- coding: utf-8 -*-
"""带重试与自动降批的 Embeddings 包装。

- 批次超上限（400 `batch size is invalid`）→ 二分拆批重试；
- 后端偶发错误（400 backend response / 429 / 超时）→ 指数退避重试。

让上层（PaperIndex / 检索）无需关心这些细节。
"""

from __future__ import annotations

import logging
import time

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

_TRANSIENT_HINTS = (
    "backend response failed",
    "timeout",
    "timed out",
    "rate limit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "connection",
    "temporarily",
)


def _is_too_large(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "batch size" in msg or "larger than" in msg


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(hint in msg for hint in _TRANSIENT_HINTS)


class RetryingEmbeddings(Embeddings):
    """把任意 `Embeddings` 包一层重试/拆批。"""

    def __init__(
        self,
        base: Embeddings,
        max_retries: int = 3,
        base_delay: float = 1.0,
        only_transient: bool = True,
    ) -> None:
        self.base = base
        self.max_retries = max(0, max_retries)
        self.base_delay = base_delay
        self.only_transient = only_transient

    # ---------------- 内部 ----------------
    def _call(self, texts: list[str], depth: int = 0) -> list[list[float]]:
        attempt = 0
        while True:
            try:
                return self.base.embed_documents(texts)
            except Exception as exc:  # noqa: BLE001
                if len(texts) > 1 and _is_too_large(exc):
                    mid = len(texts) // 2
                    logger.warning(
                        "embedding 批次过大（%d），二分重试：%s", len(texts), str(exc)[:120]
                    )
                    return self._call(texts[:mid], depth + 1) + self._call(texts[mid:], depth + 1)

                retryable = _is_transient(exc) or not self.only_transient
                if attempt >= self.max_retries or not retryable:
                    raise
                delay = self.base_delay * (2**attempt)
                attempt += 1
                logger.warning(
                    "embedding 调用失败（第 %d 次），%.1fs 后重试：%s",
                    attempt,
                    delay,
                    str(exc)[:140],
                )
                time.sleep(delay)

    def _call_one(self, text: str) -> list[float]:
        attempt = 0
        while True:
            try:
                return self.base.embed_query(text)
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.max_retries or not (_is_transient(exc) or not self.only_transient):
                    raise
                delay = self.base_delay * (2**attempt)
                attempt += 1
                logger.warning("embed_query 失败（第 %d 次），%.1fs 后重试：%s", attempt, delay, str(exc)[:140])
                time.sleep(delay)

    # ---------------- Embeddings 接口 ----------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._call(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._call_one(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        # 基类没有异步实现时退回同步（避免额外判断开销）
        try:
            return await self.base.aembed_documents(list(texts))
        except (NotImplementedError, AttributeError):
            return self.embed_documents(list(texts))

    async def aembed_query(self, text: str) -> list[float]:
        try:
            return await self.base.aembed_query(text)
        except (NotImplementedError, AttributeError):
            return self.embed_query(text)
