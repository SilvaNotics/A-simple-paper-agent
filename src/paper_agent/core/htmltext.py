# -*- coding: utf-8 -*-
"""HTML 纯函数工具：抽出 PDF 直链 + 抽出网页正文（不联网，便于离线单测）。

只依赖 beautifulsoup4（requirements 已含）；lxml 可用时优先，缺失则退回标准库解析器。
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

_INLINE_WS = re.compile(r"[ \t\u00a0\u3000]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
# 逐块取文本时会在行内元素边界留下空格，中文标点前的空格要收掉（「学科 。」→「学科。」）
_CJK_PUNCT_FIX = re.compile(r"\s+([，。；：、！？）》」』】…])")

# `citation_pdf_url` 是 Highwire/Google Scholar 事实标准，各家 OJS/出版商都在用
_PDF_META_NAMES = {
    "citation_pdf_url",
    "bepress_citation_pdf_url",
    "eprints.document_url",
    "dc.identifier.pdf",
}
_PDF_HREF = re.compile(r"\.pdf(?:$|[?#])", re.IGNORECASE)

# 正文容器候选选择器（按优先级；取文本最多的那个）
_CONTENT_SELECTORS = (
    "article",
    "main",
    "#mw-content-text",  # Wikipedia / MediaWiki
    ".mw-parser-output",
    ".J-lemma-content",  # 百度百科
    ".lemma-content",
    "#js_content",  # 微信公众号
    ".article-content",
    ".post-content",
    "#content",
    ".content",
)
_BLOCK_TAGS = (
    "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "blockquote", "pre", "dd", "dt", "figcaption", "td", "th",
)
_STRIP_TAGS = (
    "script", "style", "noscript", "template", "svg", "iframe", "form",
    "nav", "footer", "header", "aside", "button",
)
_NOISE_SELECTORS = (
    ".mw-editsection", ".reference", ".reflist", ".mw-references-wrap",
    ".navbox", ".infobox", ".toc", "#toc", ".sidebar", ".hatnote", ".metadata",
)
_MIN_CONTENT = 200  # 候选容器文本短于该长度就不算正文

# 网页标题里常见的站点后缀（`<title>` 会拼「词条 - 站点名」）
_TITLE_SUFFIX = re.compile(
    r"\s*[-–—_|]\s*(?:维基百科[^|]*|Wikipedia[^|]*|百度百科[^|]*|.*_百度百科|.*百科)\s*$",
    re.IGNORECASE,
)


def soup_of(markup: str) -> BeautifulSoup:
    """构造 BeautifulSoup；lxml 不可用时退回标准库解析器。"""
    try:
        return BeautifulSoup(markup or "", "lxml")
    except Exception:  # noqa: BLE001 - FeatureNotFound 等
        return BeautifulSoup(markup or "", "html.parser")


def pdf_url_from_html(html_text: str, base_url: str = "") -> str:
    """落地页 HTML → PDF 直链（纯函数）。

    优先级：`citation_pdf_url` meta > `link[type=application/pdf]` > 指向 `.pdf` 的 `<a href>`。
    相对地址按 `base_url` 补全；非 http(s) 的候选（javascript:/mailto: 等）忽略。
    """
    if not html_text:
        return ""
    soup = soup_of(html_text)
    candidates: list[str] = []

    for meta in soup.find_all("meta"):
        name = str(meta.get("name") or meta.get("property") or "").strip().lower()
        content = str(meta.get("content") or "").strip()
        if name in _PDF_META_NAMES and content:
            candidates.append(content)

    for link in soup.find_all("link", href=True):
        if "pdf" in str(link.get("type") or "").lower():
            candidates.append(str(link["href"]).strip())

    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if _PDF_HREF.search(href):
            candidates.append(href)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate.startswith(("javascript:", "mailto:", "#", "data:")):
            continue
        absolute = urljoin(base_url, candidate) if base_url else candidate
        if not absolute.lower().startswith(("http://", "https://")):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        return absolute
    return ""


def _clean_line(text: str) -> str:
    line = _INLINE_WS.sub(" ", (text or "").replace("\r", "\n")).strip()
    return _CJK_PUNCT_FIX.sub(r"\1", line)


def _node_text(node: Tag) -> str:
    """把容器里的块级元素文本按阅读顺序拼起来（跳过嵌套容器，避免重复）。"""
    parts: list[str] = []
    for el in node.find_all(_BLOCK_TAGS):
        if el.find(_BLOCK_TAGS):  # 容器（文本由其子块负责）
            continue
        line = _clean_line(el.get_text(" "))
        if line:
            parts.append(line)
    if not parts:
        raw = node.get_text("\n")
        return "\n".join(x for x in (_clean_line(l) for l in raw.split("\n")) if x)
    return _MULTI_NEWLINE.sub("\n\n", "\n\n".join(parts))


def page_title(soup: BeautifulSoup) -> str:
    """网页标题：优先 `<h1>`，退回 `<title>`（去掉「- 维基百科」这类站点后缀）。"""
    h1 = soup.find("h1")
    if h1:
        text = _clean_line(h1.get_text(" "))
        if text:
            return _TITLE_SUFFIX.sub("", text).strip() or text
    if soup.title:
        text = _clean_line(soup.title.get_text(" "))
        if text:
            return _TITLE_SUFFIX.sub("", text).strip() or text
    return ""


def _pick_content_node(soup: BeautifulSoup) -> Tag | None:
    """选择器都不匹配时的通用启发式（readability 简化版）。

    在所有含足够文本的块里，先只看文本量在前 10% 内的候选，再优先取**更深**的
    （外层 wrap 通常比正文 div 多带导航/页脚）、链接占比最低、文本最长的那个。
    """
    candidates: list[tuple[int, float, int, Tag]] = []
    for node in soup.find_all(("div", "td", "section", "article", "main")):
        if len(node.get_text()) < _MIN_CONTENT:  # 先用廉价长度过滤
            continue
        text = _node_text(node)
        length = len(text)
        if length < _MIN_CONTENT:
            continue
        link_len = sum(len(_clean_line(a.get_text(" "))) for a in node.find_all("a"))
        density = min(1.0, link_len / max(1, length))
        depth = sum(1 for _ in node.parents)
        candidates.append((length, density, depth, node))
    if not candidates:
        return None
    max_len = max(c[0] for c in candidates)
    near = [c for c in candidates if c[0] >= max_len * 0.9]
    near.sort(key=lambda c: (-c[2], c[1], -c[0]))
    return near[0][3]


def html_to_text(html_text: str) -> tuple[str, str]:
    """网页 HTML → (标题, 正文文本)（纯函数）。

    先剥掉脚本/导航/页眉页脚等噪声，再在候选正文容器里取文本最多的一个；
    都不够长时退回 `<body>`。返回文本为空表示这个页面没有可入库的正文。
    """
    if not html_text:
        return "", ""
    soup = soup_of(html_text)
    title = page_title(soup)

    for tag in list(soup.find_all(_STRIP_TAGS)):
        if not getattr(tag, "decomposed", False):
            tag.decompose()
    for selector in _NOISE_SELECTORS:
        for tag in list(soup.select(selector)):
            if not getattr(tag, "decomposed", False):
                tag.decompose()

    best_node: Tag | None = None
    best_len = 0
    for selector in _CONTENT_SELECTORS:
        for node in soup.select(selector):
            length = len(_node_text(node))
            if length > best_len:
                best_node, best_len = node, length
    if best_node is None or best_len < _MIN_CONTENT:
        # 选择器没命中（自建站/新闻页）：用「文本多 + 链接少」的块兜底，再去 body
        best_node = _pick_content_node(soup) or soup.body or soup

    text = _MULTI_NEWLINE.sub("\n\n", _node_text(best_node)).strip()
    return title, text
