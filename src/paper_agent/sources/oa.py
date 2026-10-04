# -*- coding: utf-8 -*-
"""全文获取兜底：没有 PDF 直链时，先补链，再退到网页正文。

补链顺序（都失败就返回空，由上层决定是否退到网页正文/题录）：

1. arXiv 链接 → 直接推出 `/pdf/<id>`（不必请求）；
2. **Unpaywall**（`UNPAYWALL_EMAIL`，免 key，覆盖最全的 OA 定位）；
3. **OpenAlex** `best_oa_location.pdf_url`（复用内置解析器）；
4. 落地页 HTML：`citation_pdf_url` meta / `link[type=application/pdf]` / `.pdf` 链接。
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote

from ..core.config import Settings, get_settings
from ..core.htmltext import html_to_text, pdf_url_from_html
from ..core.schema import Paper
from .fetchers import _request, resolve_doi_openalex

logger = logging.getLogger(__name__)

UNPAYWALL_API = "https://api.unpaywall.org/v2"
_HTTP_URL = re.compile(r"^https?://", re.IGNORECASE)
_PDF_SUFFIX = re.compile(r"\.pdf(?:$|[?#])", re.IGNORECASE)
_ARXIV_ABS = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)
_CHARSET = re.compile(r"charset=[\"']?([\w-]+)", re.IGNORECASE)
_HTML_ACCEPT = "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8"
_HTML_HEADERS = {"Accept": _HTML_ACCEPT, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
_MAX_HTML_BYTES = 3_000_000  # 单页上限，防超大页面把内存/时间拖爆


def pdf_url_from_unpaywall(payload: dict[str, Any] | None) -> str:
    """Unpaywall 响应 → PDF 直链（纯函数；`best_oa_location` 优先，其次任一 OA location）。"""
    data = payload or {}
    best = data.get("best_oa_location") or {}
    if isinstance(best, dict) and best.get("url_for_pdf"):
        return str(best["url_for_pdf"])
    for location in data.get("oa_locations") or []:
        if isinstance(location, dict) and location.get("url_for_pdf"):
            return str(location["url_for_pdf"])
    return ""


def _doi_of(paper: Paper) -> str:
    if paper.doi:
        return paper.doi.replace("https://doi.org/", "").strip()
    if paper.paper_id.startswith("doi:"):
        return paper.paper_id.split(":", 1)[1].strip()
    return ""


def _decode(content: bytes, content_type: str) -> str:
    """按 HTTP 头 / `<meta charset>` 猜编码解码（中文站常见 GBK/GB2312）。"""
    charset = ""
    match = _CHARSET.search(content_type or "")
    if match:
        charset = match.group(1)
    if not charset:
        head = content[:4096].decode("ascii", errors="ignore")
        match = _CHARSET.search(head)
        charset = match.group(1) if match else "utf-8"
    try:
        return content.decode(charset, errors="replace")
    except LookupError:  # 服务端给了不认识的 charset
        return content.decode("utf-8", errors="replace")


async def fetch_html(url: str, settings: Settings | None = None) -> tuple[str, str, bool]:
    """抓一个网页，返回 `(最终 URL, 文本, 是否 PDF)`。

    - 带重试（复用检索层 `_request`：429/5xx/超时退避）；
    - 内容按 `Content-Type`/magic 判断是不是 PDF（是的话上层可直接当直链用）；
    - 超长页面截断到 `_MAX_HTML_BYTES`。
    """
    s = settings or get_settings()
    resp = await _request(url, None, s, retries=1, headers=_HTML_HEADERS)
    content = resp.content[:_MAX_HTML_BYTES]
    content_type = str(resp.headers.get("content-type") or "")
    if content[:4] == b"%PDF" or "application/pdf" in content_type.lower():
        return str(resp.url), "", True
    return str(resp.url), _decode(content, content_type), False


async def discover_pdf_url(paper: Paper, settings: Settings | None = None) -> tuple[str, str]:
    """给没有 PDF 直链的论文再找一次全文链接。

    Returns:
        `(pdf_url, note)`；`pdf_url` 为空表示没找到，`note` 说明尝试过哪些途径
        （找到时形如「补链：Unpaywall」，没找到时形如「补链未果（已试 …）」）。
    """
    s = settings or get_settings()
    if paper.pdf_url:
        return paper.pdf_url, "已有直链"

    url = paper.url or ""
    tried: list[str] = []

    # ① arXiv 链接直接推 PDF（免一次网络请求）
    match = _ARXIV_ABS.search(url)
    if match:
        return f"https://arxiv.org/pdf/{match.group(1)}", "补链：arXiv 链接"

    # ② DOI → Unpaywall（最全的 OA 定位；需要邮箱）
    doi = _doi_of(paper)
    if doi:
        email = s.unpaywall_email or s.openalex_mailto
        if email:
            try:
                resp = await _request(
                    f"{UNPAYWALL_API}/{quote(doi, safe='/')}", {"email": email}, s, retries=1
                )
                found = pdf_url_from_unpaywall(resp.json() or {})
                if found:
                    return found, "补链：Unpaywall"
            except Exception as exc:  # noqa: BLE001 - 补链失败不阻断入库
                logger.info("Unpaywall 查询失败（%s）：%s", doi, exc)
            tried.append("Unpaywall")
        else:
            tried.append("Unpaywall(缺 UNPAYWALL_EMAIL)")

        # ③ DOI → OpenAlex best_oa_location
        try:
            work = await resolve_doi_openalex(doi, s)
            if work is not None and work.pdf_url:
                return work.pdf_url, "补链：OpenAlex"
        except Exception as exc:  # noqa: BLE001
            logger.info("OpenAlex 补链失败（%s）：%s", doi, exc)
        tried.append("OpenAlex")

    # ④ 落地页 HTML（citation_pdf_url meta / .pdf 链接）
    if _PDF_SUFFIX.search(url):
        return url, "补链：URL 疑似 PDF"
    if _HTTP_URL.match(url):
        try:
            final_url, html, is_pdf = await fetch_html(url, s)
            if is_pdf and final_url:
                return final_url, "补链：URL 重定向到 PDF"
            found = pdf_url_from_html(html, final_url or url)
            if found:
                return found, "补链：落地页"
        except Exception as exc:  # noqa: BLE001
            logger.info("落地页补链失败（%s）：%s", url, exc)
        tried.append("落地页")

    return "", f"补链未果（已试 {'、'.join(tried)}）" if tried else ""


async def fetch_page_text(paper: Paper, settings: Settings | None = None) -> tuple[str, str, str]:
    """把论文的落地页/网页抓成可入库正文（Wikipedia / 百科 / 新闻页等）。

    Returns:
        `(title, text, url)`；`text` 为空表示这个页面没有可入库的正文
        （不是网页、抓取失败、正文过短，或返回的其实是 PDF）。
    """
    s = settings or get_settings()
    url = paper.url or ""
    if not _HTTP_URL.match(url) or _PDF_SUFFIX.search(url):
        return "", "", ""
    try:
        final_url, html, is_pdf = await fetch_html(url, s)
    except Exception as exc:  # noqa: BLE001 - 网页兜底失败就退到题录
        logger.info("网页正文抓取失败（%s）：%s", url, exc)
        return "", "", ""
    if is_pdf:
        return "", "", ""

    title, text = html_to_text(html)
    if len(text) < max(1, s.web_text_min_chars):
        return "", "", ""
    return title or paper.title, text, final_url or url
