# -*- coding: utf-8 -*-
"""PDF 抓取与本地缓存。

只用开放获取链接（arXiv / PMC / 开放 PDF），失败就跳过并在结果里说明原因。
两个入口：`download_pdf()` 落盘缓存（`/ingest` 用），`fetch_pdf_bytes()` 只取内存字节
（`/quick` 用，**不写任何文件**）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from ..core.config import Settings, get_settings
from ..core.schema import Paper
from ..core.utils import safe_filename

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 paper-agent/0.1 (+academic research)"
)

# `safe_filename` 已上收到 utils（纯函数，供 fetch 与 pdf_server 共用）；
# 这里保留导入，旧写法 `from .rag.fetch import safe_filename` 仍然可用。
__all__ = ["cached_pdf", "download_pdf", "fetch_pdf_bytes", "local_pdf_path", "safe_filename"]

# 内存抓取的单篇 PDF 上限（避免一个怪物 PDF 把内存或 embedding 预算吃光）
_MAX_MEMORY_PDF_BYTES = 48 * 1024 * 1024


def local_pdf_path(paper: Paper, settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    return s.papers_dir / f"{safe_filename(paper.paper_id)}.pdf"


def cached_pdf(paper: Paper, settings: Settings | None = None) -> Path | None:
    """本地已缓存且看起来是 PDF 的路径。"""
    path = local_pdf_path(paper, settings)
    if not path.exists() or path.stat().st_size < 128:
        return None
    try:
        with path.open("rb") as fh:
            if not fh.read(4).startswith(b"%PDF"):
                return None
    except OSError:
        return None
    return path


async def _get_pdf_bytes(url: str, timeout: float) -> bytes:
    """GET 一个 URL 并返回原始字节（不落盘）。"""
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": _UA, "Accept": "application/pdf,*/*"},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


async def download_pdf(
    paper: Paper,
    settings: Settings | None = None,
    timeout: float = 90.0,
) -> tuple[Path | None, str]:
    """下载论文 PDF 到 `data/papers/`。

    Returns:
        (path, message)：path 为 None 时 message 说明原因。
    """
    s = settings or get_settings()
    s.ensure_dirs()

    existing = cached_pdf(paper, s)
    if existing:
        return existing, "命中本地缓存"

    url = paper.pdf_url or ""
    if not url:
        return None, "没有 PDF 直链（仅有元数据）"

    path = local_pdf_path(paper, s)
    try:
        content = await _get_pdf_bytes(url, timeout)
    except Exception as exc:  # noqa: BLE001
        return None, f"下载失败 {type(exc).__name__}: {exc}"

    if not content.startswith(b"%PDF"):
        return None, "返回内容不是 PDF（可能需要订阅或被反爬）"

    path.write_bytes(content)
    return path, f"已下载 {len(content) // 1024} KB"


async def fetch_pdf_bytes(
    paper: Paper,
    settings: Settings | None = None,
    timeout: float = 90.0,
    max_bytes: int = _MAX_MEMORY_PDF_BYTES,
) -> tuple[bytes | None, str]:
    """只取 PDF 字节，**不落盘**（`/quick` 专用）。

    已缓存过的论文直接读缓存（本来就在磁盘上，没有再写新文件）；否则按 `pdf_url` 抓到内存。

    Returns:
        (content, message)：content 为 None 时 message 说明原因。
    """
    s = settings or get_settings()

    existing = cached_pdf(paper, s)
    if existing is not None:
        try:
            return existing.read_bytes(), "命中本地缓存（未落盘）"
        except OSError as exc:  # noqa: BLE001 - 缓存读不了就当没有
            logger.warning("读取本地缓存失败（%s）：%s", existing, exc)

    url = paper.pdf_url or ""
    if not url:
        return None, "没有 PDF 直链（仅有元数据）"

    try:
        content = await _get_pdf_bytes(url, timeout)
    except Exception as exc:  # noqa: BLE001
        return None, f"下载失败 {type(exc).__name__}: {exc}"

    if len(content) > max_bytes:
        return None, f"PDF 过大（{len(content) // (1024 * 1024)} MB，超过 {max_bytes // (1024 * 1024)} MB 上限）"
    if not content.startswith(b"%PDF"):
        return None, "返回内容不是 PDF（可能需要订阅或被反爬）"
    return content, f"内存抓取 {len(content) // 1024} KB（未落盘）"
