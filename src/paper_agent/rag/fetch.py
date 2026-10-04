# -*- coding: utf-8 -*-
"""PDF 抓取与本地缓存。

只用开放获取链接（arXiv / PMC / 开放 PDF），失败就跳过并在结果里说明原因。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx

from ..config import Settings, get_settings
from ..schema import Paper

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 paper-agent/0.1 (+academic research)"
)
_SAFE = re.compile(r"[^0-9A-Za-z._-]+")


def safe_filename(paper_id: str) -> str:
    return _SAFE.sub("_", paper_id or "unknown")[:120]


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
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": _UA, "Accept": "application/pdf,*/*"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content = resp.content
    except Exception as exc:  # noqa: BLE001
        return None, f"下载失败 {type(exc).__name__}: {exc}"

    if not content.startswith(b"%PDF"):
        return None, "返回内容不是 PDF（可能需要订阅或被反爬）"

    path.write_bytes(content)
    return path, f"已下载 {len(content) // 1024} KB"
