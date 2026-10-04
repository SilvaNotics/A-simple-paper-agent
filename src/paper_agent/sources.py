# -*- coding: utf-8 -*-
"""内置联网抓取层（不依赖 MCP、不需要 API key）。

- **按 ID 直抓**：arXiv ID / DOI / 链接 → 元数据 + PDF 直链，无需先搜索；
- **MCP 回退**：直连公开 REST/Atom 接口检索（arXiv / OpenAlex / Crossref / …）。

只用 `httpx` + 标准库，内置重试与 arXiv 速率限制（官方要求 ≥3s 间隔）。
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Callable, Iterable
from urllib.parse import quote

import httpx

from .channels import domestic_first, is_domestic, spec_for
from .config import Settings, get_settings
from .schema import Paper
from .utils import clean_pasted, extract_json, first_list, normalize_paper_id

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
CROSSREF_API = "https://api.crossref.org/works"
ATOM_NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
# HTTP header 值不能有首尾空白，否则 httpx 抛 "Illegal header value"
USER_AGENT = "paper-agent/0.1 (+https://github.com/local/paper-agent; Python httpx)"

# arXiv 官方要求请求间隔 ≥3s；进程内节流器兜住
_ARXIV_LOCK = asyncio.Lock()
_ARXIV_LAST = 0.0

_JATS_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# HTTP 基础设施
# --------------------------------------------------------------------------


def _timeout(settings: Settings) -> httpx.Timeout:
    return httpx.Timeout(settings.http_timeout, connect=min(15.0, settings.http_timeout))


async def _request(
    url: str,
    params: dict[str, Any] | None,
    settings: Settings,
    retries: int = 2,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """带重试的 GET（429/5xx/超时重试，尊重 Retry-After）。"""
    last_exc: Exception | None = None
    request_headers = {"User-Agent": settings.http_user_agent or USER_AGENT}
    request_headers.update(headers or {})
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(
                timeout=_timeout(settings),
                follow_redirects=True,
                headers=request_headers,
            ) as client:
                resp = await client.get(url, params=params)
                if resp.status_code in {429, 500, 502, 503, 504}:
                    delay = float(resp.headers.get("Retry-After") or 0) or 2.0 * (attempt + 1)
                    if attempt < retries:
                        logger.warning("HTTP %s，%.1fs 后重试：%s", resp.status_code, delay, url)
                        await asyncio.sleep(delay)
                        continue
                resp.raise_for_status()
                return resp
        except httpx.HTTPStatusError as exc:
            # 4xx（除 429）是确定性错误，重试没意义（401/404 直接抛出，省时省配额）
            status = exc.response.status_code
            retryable = status == 429 or status >= 500
            last_exc = exc
            if retryable and attempt < retries:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise
        except Exception as exc:  # noqa: BLE001 - 连接/超时等网络异常可重试
            last_exc = exc
            if attempt < retries:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise

    # 理论不可达（循环内要么 return 要么 raise）；保留兜底以便类型检查与排错
    if last_exc is not None:  # pragma: no cover
        raise last_exc
    raise RuntimeError(f"请求重试耗尽：{url}")  # pragma: no cover


async def _post_json(
    url: str,
    payload: dict[str, Any],
    settings: Settings,
    headers: dict[str, str] | None = None,
    retries: int = 2,
    ensure_ascii: bool = False,
) -> httpx.Response:
    """带重试的 POST JSON（给需要 key 的搜索渠道用）。

    `ensure_ascii=True` 时手写 body（非 ASCII 转义）：国家图书馆 doSearch 对原始
    UTF-8 中文 body 会忽略检索词，转义后才正常。
    """
    request_headers = {
        "User-Agent": settings.http_user_agent or USER_AGENT,
        "Content-Type": "application/json",
    }
    request_headers.update(headers or {})
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(
                timeout=_timeout(settings), follow_redirects=True, headers=request_headers
            ) as client:
                if ensure_ascii:
                    resp = await client.post(
                        url, content=json.dumps(payload).encode("utf-8")
                    )
                else:
                    resp = await client.post(url, json=payload)
                if resp.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            last_exc = exc
            if (status == 429 or status >= 500) and attempt < retries:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise
    if last_exc is not None:  # pragma: no cover
        raise last_exc
    raise RuntimeError(f"请求重试耗尽：{url}")  # pragma: no cover


def _channel_creds(settings: Settings, kind: str, channel: dict[str, Any] | None = None) -> dict[str, str]:
    """取渠道凭据：显式传入的 channel 优先，其次 Settings 里的 `/channels` 配置。"""
    if channel:
        return {
            "name": str(channel.get("name", kind)),
            "api_key": str(channel.get("api_key") or ""),
            "email": str(channel.get("email") or ""),
            "base_url": str(channel.get("base_url") or ""),
        }
    return settings.channel_credentials(kind)


async def _throttle_arxiv(settings: Settings) -> None:
    global _ARXIV_LAST
    async with _ARXIV_LOCK:
        wait = settings.arxiv_min_interval - (time.monotonic() - _ARXIV_LAST)
        if wait > 0:
            await asyncio.sleep(wait)
        _ARXIV_LAST = time.monotonic()


def _clean_abstract(text: str | None, limit: int = 2000) -> str:
    if not text:
        return ""
    # Crossref 的摘要带 JATS 标签与 HTML 实体（&amp; 等），都要处理
    plain = _WS_RE.sub(" ", _JATS_RE.sub(" ", html.unescape(str(text)))).strip()
    return plain[:limit]


# --------------------------------------------------------------------------
# arXiv（Atom XML）
# --------------------------------------------------------------------------


def _atom_text(node: ET.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


def parse_arxiv_atom(xml_text: str) -> list[Paper]:
    """解析 arXiv Atom 响应（纯函数，便于离线单测）。"""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("arXiv Atom 解析失败：%s", exc)
        return []

    papers: list[Paper] = []
    for entry in root.findall("a:entry", ATOM_NS):
        raw_id = _atom_text(entry.find("a:id", ATOM_NS))
        if not raw_id:
            continue
        paper_id = normalize_paper_id(raw_id)
        if not paper_id.startswith("arxiv:"):
            continue
        authors = [(_atom_text(a.find("a:name", ATOM_NS))) for a in entry.findall("a:author", ATOM_NS)]
        published = _atom_text(entry.find("a:published", ATOM_NS)) or _atom_text(
            entry.find("a:updated", ATOM_NS)
        )
        primary = entry.find("arxiv:primary_category", ATOM_NS)
        categories = [c.attrib.get("term", "") for c in entry.findall("a:category", ATOM_NS)]
        arxiv_id = paper_id.split(":", 1)[1]

        papers.append(
            Paper(
                paper_id=paper_id,
                title=_WS_RE.sub(" ", _atom_text(entry.find("a:title", ATOM_NS))),
                authors="; ".join(a for a in authors if a),
                abstract=_clean_abstract(_atom_text(entry.find("a:summary", ATOM_NS))),
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                url=f"https://arxiv.org/abs/{arxiv_id}",
                published=published,
                source="arxiv-api",
                categories=", ".join(
                    x for x in [primary.attrib.get("term", "") if primary is not None else "", *categories] if x
                ),
            )
        )
    return papers


_PHRASE_RE = re.compile(r'"([^"]{3,80})"')


def build_arxiv_query(query: str, max_terms: int = 4) -> str:
    """把自然语言/布尔查询转成 arXiv API 的 search_query。

    LLM 常给出 `("cross-chunk" OR "graph augmentation") AND ...` 这类布尔串，
    直接丢给 arXiv 会命中很差。这里做两步：
      1. 先取引号里的短语 → `all:"graph augmentation"`（短语更精确）；
      2. 再补最多 `max_terms` 个关键词 → `all:crosschunk`（连字符会被 arXiv 当成
         词内连接，命中率反而低，故拆成纯词）。
    """
    text = clean_pasted(query)
    phrases = [m.group(1).strip() for m in _PHRASE_RE.finditer(text)][:2]
    remainder = _PHRASE_RE.sub(" ", text)
    keywords = [
        w
        for w in re.findall(r"[A-Za-z0-9]+", remainder)
        if len(w) > 2 and w.lower() not in {"and", "or", "not", "the", "for", "with", "from", "into"}
    ]
    parts = [f'all:"{ph}"' for ph in phrases]
    for word in keywords[:max_terms]:
        if all(word.lower() not in p.lower() for p in parts):
            parts.append(f"all:{word}")
    return " AND ".join(parts[: max(1, max_terms + len(phrases))])


async def _arxiv_query(query_string: str, limit: int, settings: Settings) -> list[Paper]:
    await _throttle_arxiv(settings)
    resp = await _request(
        ARXIV_API,
        {
            "search_query": query_string,
            "start": 0,
            "max_results": max(1, min(limit, 50)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
        settings,
    )
    return parse_arxiv_atom(resp.text)[:limit]


async def search_arxiv(query: str, limit: int = 10, settings: Settings | None = None) -> list[Paper]:
    """arXiv 关键词检索（无需 key）。首轮太窄导致 0 命中时会自动放宽。"""
    s = settings or get_settings()
    strict = build_arxiv_query(query, max_terms=4)
    if not strict:
        return []
    papers = await _arxiv_query(strict, limit, s)
    if papers:
        return papers

    relaxed = build_arxiv_query(query, max_terms=2)
    if relaxed and relaxed != strict:
        logger.info("arXiv 首轮 0 命中，放宽查询重试：%s", relaxed)
        papers = await _arxiv_query(relaxed, limit, s)
    return papers


async def resolve_arxiv(arxiv_id: str, settings: Settings | None = None) -> Paper | None:
    """按 arXiv ID 精确取元数据 + PDF 直链。"""
    s = settings or get_settings()
    ident = clean_pasted(arxiv_id)
    match = re.search(r"(\d{4}\.\d{4,5})", ident)
    if not match:
        return None
    await _throttle_arxiv(s)
    resp = await _request(ARXIV_API, {"id_list": match.group(1)}, s)
    papers = parse_arxiv_atom(resp.text)
    return papers[0] if papers else None


# --------------------------------------------------------------------------
# OpenAlex / Crossref（JSON）
# --------------------------------------------------------------------------


def _openalex_pdf_url(work: dict[str, Any]) -> str:
    best = work.get("best_oa_location") or {}
    if best.get("pdf_url"):
        return str(best["pdf_url"])
    for loc in work.get("locations") or []:
        if loc.get("pdf_url"):
            return str(loc["pdf_url"])
    return ""


def _openalex_abstract(work: dict[str, Any]) -> str:
    """把 OpenAlex 的 abstract_inverted_index 还原成文本。"""
    inverted = work.get("abstract_inverted_index")
    if not isinstance(inverted, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, spots in inverted.items():
        for spot in spots or []:
            if isinstance(spot, int):
                positions.append((spot, str(word)))
    positions.sort()
    return _clean_abstract(" ".join(word for _, word in positions))


def openalex_work_to_paper(work: dict[str, Any]) -> Paper | None:
    """OpenAlex work → Paper（纯函数，便于离线单测）。"""
    if not isinstance(work, dict) or not work.get("title"):
        return None
    doi = str(work.get("doi") or "").replace("https://doi.org/", "")
    authors: list[str] = []
    for item in work.get("authorships") or []:
        name = ((item or {}).get("author") or {}).get("display_name")
        if name:
            authors.append(str(name))
    primary = work.get("primary_location") or {}
    return Paper(
        paper_id=normalize_paper_id(doi or work.get("id") or work.get("title", "")),
        title=_WS_RE.sub(" ", str(work["title"])),
        authors="; ".join(authors),
        abstract=_openalex_abstract(work),
        pdf_url=_openalex_pdf_url(work),
        url=str(primary.get("landing_page_url") or work.get("doi") or work.get("id") or ""),
        doi=doi,
        published=str(work.get("publication_date") or work.get("publication_year") or ""),
        source="openalex",
        categories=", ".join(
            str(t.get("display_name"))
            for t in (work.get("topics") or [])[:3]
            if isinstance(t, dict) and t.get("display_name")
        ),
        citations=int(work.get("cited_by_count") or 0),
    )


async def search_openalex(
    query: str,
    limit: int = 10,
    settings: Settings | None = None,
    open_access_only: bool = False,
    channel: dict[str, Any] | None = None,
) -> list[Paper]:
    """OpenAlex 检索（免 key；带 email 进 polite pool，带 api_key 可提高配额）。"""
    s = settings or get_settings()
    creds = _channel_creds(s, "openalex", channel)
    params: dict[str, Any] = {
        "search": clean_pasted(query),
        "per-page": max(1, min(limit, 50)),
        "select": (
            "id,doi,title,authorships,primary_location,best_oa_location,locations,"
            "abstract_inverted_index,publication_date,publication_year,cited_by_count,topics"
        ),
    }
    if open_access_only:
        params["filter"] = "is_oa:true"
    mailto = creds.get("email") or s.openalex_mailto
    if mailto:
        params["mailto"] = mailto  # polite pool
    if creds.get("api_key"):
        params["api_key"] = creds["api_key"]
    resp = await _request(creds.get("base_url") or OPENALEX_API, params, s)
    payload = resp.json() or {}
    papers = [openalex_work_to_paper(w) for w in (payload.get("results") or [])]
    return [p for p in papers if p is not None][:limit]


async def resolve_doi_openalex(doi: str, settings: Settings | None = None) -> Paper | None:
    s = settings or get_settings()
    creds = s.channel_credentials("openalex")
    ident = clean_pasted(doi).replace("https://doi.org/", "")
    params: dict[str, Any] = {}
    mailto = creds.get("email") or s.openalex_mailto
    if mailto:
        params["mailto"] = mailto
    if creds.get("api_key"):
        params["api_key"] = creds["api_key"]
    resp = await _request(f"{OPENALEX_API}/https://doi.org/{ident}", params or None, s)
    return openalex_work_to_paper(resp.json() or {})


def crossref_work_to_paper(work: dict[str, Any]) -> Paper | None:
    """Crossref work → Paper（纯函数）。"""
    if not isinstance(work, dict):
        return None
    titles = work.get("title") or []
    title = _WS_RE.sub(" ", str(titles[0])) if titles else ""
    if not title:
        return None
    doi = str(work.get("DOI") or "")
    authors: list[str] = []
    for a in work.get("author") or []:
        name = " ".join(x for x in [a.get("given"), a.get("family")] if x) or a.get("name")
        if name:
            authors.append(str(name))
    issued = ((work.get("issued") or {}).get("date-parts") or [[None]])[0]
    pdf_url = ""
    for link in work.get("link") or []:
        if str(link.get("content-type", "")).startswith("application/pdf") and link.get("URL"):
            pdf_url = str(link["URL"])
            break
    return Paper(
        paper_id=normalize_paper_id(doi),
        title=title,
        authors="; ".join(authors),
        abstract=_clean_abstract(work.get("abstract")),
        pdf_url=pdf_url,
        url=str(work.get("URL") or (f"https://doi.org/{doi}" if doi else "")),
        doi=doi,
        published="-".join(str(x).zfill(2) for x in issued if x) if any(issued) else "",
        source="crossref",
        categories=", ".join(
            str(x) for x in ((work.get("container-title") or []) + (work.get("subject") or []))[:2]
        ),
        citations=int(work.get("is-referenced-by-count") or 0),
    )


async def search_crossref(query: str, limit: int = 10, settings: Settings | None = None) -> list[Paper]:
    """Crossref 检索（免 key；多数条目没有 PDF 直链，主要用于补元数据/DOI）。"""
    s = settings or get_settings()
    params: dict[str, Any] = {
        "query.bibliographic": clean_pasted(query),
        "rows": max(1, min(limit, 50)),
        "select": "DOI,title,author,issued,abstract,URL,link,container-title,subject,is-referenced-by-count",
    }
    creds = _channel_creds(s, "crossref")
    mailto = creds.get("email") or s.openalex_mailto
    if mailto:
        params["mailto"] = mailto
    resp = await _request(creds.get("base_url") or CROSSREF_API, params, s)
    items = ((resp.json() or {}).get("message") or {}).get("items") or []
    papers = [crossref_work_to_paper(w) for w in items]
    return [p for p in papers if p is not None][:limit]


async def resolve_doi_crossref(doi: str, settings: Settings | None = None) -> Paper | None:
    s = settings or get_settings()
    ident = clean_pasted(doi).replace("https://doi.org/", "")
    creds = _channel_creds(s, "crossref")
    mailto = creds.get("email") or s.openalex_mailto
    params = {"mailto": mailto} if mailto else None
    resp = await _request(f"{CROSSREF_API}/{ident}", params, s)
    return crossref_work_to_paper((resp.json() or {}).get("message") or {})


# --------------------------------------------------------------------------
# 可配置搜索渠道（`/channels`）：Europe PMC / PubMed / Semantic Scholar /
# DBLP / DOAJ / CORE / Tavily / Exa / SerpAPI
# 每个渠道统一签名 `search_x(query, limit, settings, channel=None)`，
# 便于 `builtin_search` 按 kind 分发；解析函数拆成纯函数以便离线单测。
# --------------------------------------------------------------------------

EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PUBMED_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
DOAJ_API = "https://doaj.org/api/search/articles"
CORE_API = "https://api.core.ac.uk/v3/search/works"
TAVILY_API = "https://api.tavily.com/search"
EXA_API = "https://api.exa.ai/search"
SERPAPI_API = "https://serpapi.com/search.json"
# 国内数据库
CHINAXIV_API = "https://chinarxiv.org/api/v1/papers"
BAIDU_SCHOLAR_API = "https://qianfan.baidubce.com/v2/tools/baidu_scholar/search"
WANFANG_API = "https://api.wanfangdata.com.cn/search"
NLC_API = "https://meta.nlc.cn/v2/doSearch"


def _first(value: Any, default: str = "") -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else default
    return str(value) if value not in (None, "") else default


def parse_europepmc(data: dict[str, Any]) -> list[Paper]:
    """Europe PMC 响应解析（纯函数）。"""
    results = ((data or {}).get("resultList") or {}).get("result") or []
    papers: list[Paper] = []
    for item in results:
        if not isinstance(item, dict) or not item.get("title"):
            continue
        doi = str(item.get("doi") or "")
        pmid = str(item.get("pmid") or "")
        ident = doi or (f"pmid:{pmid}" if pmid else "") or str(item.get("id") or "")
        pdf_url = ""
        for link in (item.get("fullTextUrlList") or {}).get("fullTextUrl") or []:
            if str(link.get("documentStyle", "")).lower() == "pdf" and link.get("url"):
                pdf_url = str(link["url"])
                break
        papers.append(
            Paper(
                paper_id=normalize_paper_id(ident) or f"europepmc:{item.get('id', '')}",
                title=_WS_RE.sub(" ", html.unescape(str(item["title"]))),
                authors=str(item.get("authorString") or ""),
                abstract=_clean_abstract(item.get("abstractText")),
                pdf_url=pdf_url,
                url=pdf_url or (f"https://doi.org/{doi}" if doi else ""),
                doi=doi,
                published=str(item.get("pubYear") or item.get("firstPublicationDate") or ""),
                source="europepmc",
                categories=str(item.get("journalTitle") or ""),
                citations=int(item.get("citedByCount") or 0),
            )
        )
    return papers


async def search_europepmc(
    query: str, limit: int = 10, settings: Settings | None = None, channel: dict[str, Any] | None = None
) -> list[Paper]:
    """Europe PMC（免 key；可填邮箱）。"""
    s = settings or get_settings()
    creds = _channel_creds(s, "europepmc", channel)
    params: dict[str, Any] = {
        "query": clean_pasted(query),
        "format": "json",
        "pageSize": max(1, min(limit, 100)),
        "resultType": "core",
    }
    if creds.get("email"):
        params["email"] = creds["email"]
    resp = await _request(creds.get("base_url") or EUROPEPMC_API, params, s)
    return parse_europepmc(resp.json() or {})[:limit]


_PUBMED_ABSTRACT_NS = "{http://www.ncbi.nlm.nih.gov/entrez/eutils}"


def parse_pubmed_abstracts(xml_text: str) -> dict[str, str]:
    """从 efetch XML 里提取 `pmid -> abstract`（纯函数）。

    同时兼容带命名空间（真实 efetch 响应）与不带命名空间的 XML。
    """
    out: dict[str, str] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    ns = _PUBMED_ABSTRACT_NS if str(root.tag).startswith(_PUBMED_ABSTRACT_NS) else ""
    for article in root.iter(f"{ns}PubmedArticle"):
        pmid = article.findtext(f"{ns}MedlineCitation/{ns}PMID") or ""
        chunks = [(node.text or "") for node in article.iter(f"{ns}AbstractText")]
        text = _clean_abstract(" ".join(x for x in chunks if x))
        if pmid and text:
            out[pmid] = text
    return out


def parse_pubmed_summaries(
    result: dict[str, Any], ids: list[str], abstracts: dict[str, str] | None = None
) -> list[Paper]:
    """esummary 响应 → Paper（纯函数）。"""
    abstracts = abstracts or {}
    papers: list[Paper] = []
    for pmid in ids:
        item = result.get(pmid)
        if not isinstance(item, dict) or not item.get("title"):
            continue
        doi = ""
        for ident in item.get("articleids") or []:
            if isinstance(ident, dict) and str(ident.get("idtype")) == "doi":
                doi = str(ident.get("value") or "")
                break
        authors = [
            str((a or {}).get("name") or "")
            for a in item.get("authors") or []
            if isinstance(a, dict)
        ]
        papers.append(
            Paper(
                paper_id=normalize_paper_id(doi) or f"pmid:{pmid}",
                title=_WS_RE.sub(" ", html.unescape(str(item["title"]))).rstrip("."),
                authors="; ".join(a for a in authors if a),
                abstract=abstracts.get(pmid, ""),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                doi=doi,
                published=str(item.get("pubdate") or ""),
                source="pubmed",
                categories=str(item.get("fulljournalname") or item.get("source") or ""),
            )
        )
    return papers


async def search_pubmed(
    query: str, limit: int = 10, settings: Settings | None = None, channel: dict[str, Any] | None = None
) -> list[Paper]:
    """PubMed（免 key；有 key 可提高配额；用 esearch→esummary→efetch 取摘要）。"""
    s = settings or get_settings()
    creds = _channel_creds(s, "pubmed", channel)
    base = creds.get("base_url") or PUBMED_EUTILS
    common: dict[str, Any] = {}
    if creds.get("api_key"):
        common["api_key"] = creds["api_key"]
    if creds.get("email"):
        common["email"] = creds["email"]
    cap = max(1, min(limit, 50))
    es = await _request(
        f"{base}/esearch.fcgi",
        {"db": "pubmed", "term": clean_pasted(query), "retmode": "json", "retmax": cap, "sort": "relevance", **common},
        s,
    )
    ids = [str(x) for x in ((es.json() or {}).get("esearchresult") or {}).get("idlist") or []][:cap]
    if not ids:
        return []
    eu = await _request(
        f"{base}/esummary.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "json", **common}, s
    )
    summaries = (eu.json() or {}).get("result") or {}
    abstracts: dict[str, str] = {}
    try:
        ef = await _request(
            f"{base}/efetch.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "xml", **common}, s
        )
        abstracts = parse_pubmed_abstracts(ef.text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PubMed efetch 摘要失败（仅保留题录）：%s", exc)
    return parse_pubmed_summaries(summaries, ids, abstracts)[:limit]


def parse_semantic_scholar(data: dict[str, Any]) -> list[Paper]:
    """Semantic Scholar 响应 → Paper（纯函数）。"""
    papers: list[Paper] = []
    for item in (data or {}).get("data") or []:
        if not isinstance(item, dict) or not item.get("title"):
            continue
        external = item.get("externalIds") or {}
        doi = str(external.get("DOI") or "")
        arxiv_id = str(external.get("ArXiv") or "")
        ident = arxiv_id or doi or item.get("paperId") or ""
        authors = [
            str((a or {}).get("name") or "") for a in item.get("authors") or [] if isinstance(a, dict)
        ]
        oa = item.get("openAccessPdf") or {}
        papers.append(
            Paper(
                paper_id=normalize_paper_id(ident),
                title=_WS_RE.sub(" ", str(item["title"])),
                authors="; ".join(a for a in authors if a),
                abstract=_clean_abstract(item.get("abstract")),
                pdf_url=str(oa.get("url") or ""),
                url=str(item.get("url") or ""),
                doi=doi,
                published=str(item.get("year") or ""),
                source="semanticscholar",
                categories=str(item.get("venue") or ""),
                citations=int(item.get("citationCount") or 0),
            )
        )
    return papers


async def search_semanticscholar(
    query: str, limit: int = 10, settings: Settings | None = None, channel: dict[str, Any] | None = None
) -> list[Paper]:
    """Semantic Scholar Graph API（免 key 限流严格，配 key 更稳）。"""
    s = settings or get_settings()
    creds = _channel_creds(s, "semanticscholar", channel)
    key = creds.get("api_key") or (
        s.semantic_scholar_api_key.get_secret_value() if s.semantic_scholar_api_key else ""
    )
    headers = {"x-api-key": key} if key else {}
    resp = await _request(
        creds.get("base_url") or SEMANTIC_SCHOLAR_API,
        {
            "query": clean_pasted(query),
            "limit": max(1, min(limit, 100)),
            "fields": "title,abstract,year,authors,externalIds,openAccessPdf,url,citationCount,venue",
        },
        s,
        headers=headers,
    )
    return parse_semantic_scholar(resp.json() or {})[:limit]


def parse_doaj(data: dict[str, Any]) -> list[Paper]:
    """DOAJ 响应 → Paper（纯函数）。"""
    papers: list[Paper] = []
    for item in (data or {}).get("results") or []:
        bib = (item or {}).get("bibjson") or {}
        title = str(bib.get("title") or "")
        if not title:
            continue
        identifiers = bib.get("identifier") or []
        doi = ""
        for ident in identifiers:
            if isinstance(ident, dict) and str(ident.get("type")) == "doi":
                doi = str(ident.get("id") or "")
                break
        authors = [
            str((a or {}).get("name") or "") for a in bib.get("author") or [] if isinstance(a, dict)
        ]
        links = bib.get("link") or []
        url = str((links[0] or {}).get("url") if links and isinstance(links[0], dict) else "")
        journal = bib.get("journal") or {}
        papers.append(
            Paper(
                paper_id=normalize_paper_id(doi) or f"doaj:{item.get('id', '')}",
                title=_WS_RE.sub(" ", html.unescape(title)),
                authors="; ".join(a for a in authors if a),
                abstract=_clean_abstract(bib.get("abstract")),
                url=url,
                doi=doi,
                published=str(bib.get("year") or ""),
                source="doaj",
                categories=str((journal or {}).get("title") or ""),
            )
        )
    return papers


async def search_doaj(
    query: str, limit: int = 10, settings: Settings | None = None, channel: dict[str, Any] | None = None
) -> list[Paper]:
    """DOAJ 开放获取期刊检索（免 key）。"""
    s = settings or get_settings()
    creds = _channel_creds(s, "doaj", channel)
    base = (creds.get("base_url") or DOAJ_API).rstrip("/")
    resp = await _request(
        f"{base}/{quote(clean_pasted(query), safe='')}",
        {"pageSize": max(1, min(limit, 100))},
        s,
    )
    return parse_doaj(resp.json() or {})[:limit]


def parse_core(data: dict[str, Any]) -> list[Paper]:
    """CORE 响应 → Paper（纯函数）。"""
    papers: list[Paper] = []
    for item in (data or {}).get("results") or []:
        if not isinstance(item, dict) or not item.get("title"):
            continue
        authors = [
            str((a or {}).get("name") or "") for a in item.get("authors") or [] if isinstance(a, dict)
        ]
        doi = str(item.get("doi") or "")
        papers.append(
            Paper(
                paper_id=normalize_paper_id(doi) or f"core:{item.get('id', '')}",
                title=_WS_RE.sub(" ", str(item["title"])),
                authors="; ".join(a for a in authors if a),
                abstract=_clean_abstract(item.get("abstract")),
                pdf_url=str(item.get("downloadUrl") or ""),
                url=str(item.get("downloadUrl") or ""),
                doi=doi,
                published=str(item.get("yearPublished") or ""),
                source="core",
                categories=str(item.get("publisher") or ""),
            )
        )
    return papers


async def search_core(
    query: str, limit: int = 10, settings: Settings | None = None, channel: dict[str, Any] | None = None
) -> list[Paper]:
    """CORE v3 检索（需要 API key）。"""
    s = settings or get_settings()
    creds = _channel_creds(s, "core", channel)
    key = creds.get("api_key")
    if not key:
        raise RuntimeError("CORE 需要 API key：/channels add core --key <KEY>")
    resp = await _post_json(
        creds.get("base_url") or CORE_API,
        {"q": clean_pasted(query), "limit": max(1, min(limit, 100))},
        s,
        headers={"Authorization": f"Bearer {key}"},
    )
    return parse_core(resp.json() or {})[:limit]


def parse_web_results(items: list[Any], source: str) -> list[Paper]:
    """通用网页搜索（Tavily/Exa/SerpAPI）结果 → Paper（纯函数）。"""
    papers: list[Paper] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("link") or "")
        title = str(item.get("title") or "")
        if not title and not url:
            continue
        snippet = (
            item.get("content")
            or item.get("text")
            or item.get("snippet")
            or ((item.get("publication_info") or {}).get("summary") if isinstance(item.get("publication_info"), dict) else "")
            or ""
        )
        authors = item.get("author") or ""
        if isinstance(authors, list):
            authors = "; ".join(str(a) for a in authors)
        papers.append(
            Paper(
                paper_id=normalize_paper_id(item.get("doi") or url or title),
                title=_WS_RE.sub(" ", html.unescape(title))[:300],
                authors=str(authors),
                abstract=_clean_abstract(snippet),
                pdf_url="" if source != "exa" else url,
                url=url,
                doi=str(item.get("doi") or ""),
                published=str(item.get("publishedDate") or item.get("published_date") or ""),
                source=source,
            )
        )
    return [p for p in papers if p.paper_id]


async def search_tavily(
    query: str, limit: int = 10, settings: Settings | None = None, channel: dict[str, Any] | None = None
) -> list[Paper]:
    """Tavily 网页搜索（需要 API key）。"""
    s = settings or get_settings()
    creds = _channel_creds(s, "tavily", channel)
    key = creds.get("api_key")
    if not key:
        raise RuntimeError("Tavily 需要 API key：/channels add tavily --key <KEY>")
    resp = await _post_json(
        creds.get("base_url") or TAVILY_API,
        {"api_key": key, "query": clean_pasted(query), "max_results": max(1, min(limit, 20))},
        s,
    )
    return parse_web_results((resp.json() or {}).get("results") or [], "tavily")[:limit]


async def search_exa(
    query: str, limit: int = 10, settings: Settings | None = None, channel: dict[str, Any] | None = None
) -> list[Paper]:
    """Exa 语义网页搜索（需要 API key）。"""
    s = settings or get_settings()
    creds = _channel_creds(s, "exa", channel)
    key = creds.get("api_key")
    if not key:
        raise RuntimeError("Exa 需要 API key：/channels add exa --key <KEY>")
    resp = await _post_json(
        creds.get("base_url") or EXA_API,
        {
            "query": clean_pasted(query),
            "numResults": max(1, min(limit, 20)),
            "contents": {"text": {"maxCharacters": 1200}},
        },
        s,
        headers={"x-api-key": key},
    )
    return parse_web_results((resp.json() or {}).get("results") or [], "exa")[:limit]


async def search_serpapi(
    query: str, limit: int = 10, settings: Settings | None = None, channel: dict[str, Any] | None = None
) -> list[Paper]:
    """SerpAPI Google Scholar（需要 API key）。"""
    s = settings or get_settings()
    creds = _channel_creds(s, "serpapi", channel)
    key = creds.get("api_key")
    if not key:
        raise RuntimeError("SerpAPI 需要 API key：/channels add serpapi --key <KEY>")
    resp = await _request(
        creds.get("base_url") or SERPAPI_API,
        {"engine": "google_scholar", "q": clean_pasted(query), "num": max(1, min(limit, 20)), "api_key": key},
        s,
    )
    return parse_web_results((resp.json() or {}).get("organic_results") or [], "serpapi")[:limit]


# --------------------------------------------------------------------------
# 国内数据库：ChinaXiv / 百度学术 / 万方
# --------------------------------------------------------------------------


def parse_chinaxiv(data: dict[str, Any]) -> list[Paper]:
    """ChinaXiv 预印本响应 → Paper（纯函数）。

    chinarxiv.org 提供 ChinaXiv 语料的公开 API：`data[]` 为论文列表，
    `_links.pdf` 是相对路径（英文 PDF），`source_url` 指回官方 chinaxiv.org 页面。
    """
    papers: list[Paper] = []
    for item in first_list(data, "data", "papers", "results"):
        if not isinstance(item, dict) or not item.get("title"):
            continue
        raw_id = str(item.get("id") or "")
        # "chinaxiv-202609.00204" → "chinaxiv:202609.00204"（对齐官方 ID）
        ident = raw_id.split("-", 1)[1] if raw_id.startswith("chinaxiv-") else raw_id
        links = item.get("_links") or {}
        pdf = str((links or {}).get("pdf") or "")
        if pdf.startswith("/"):
            pdf = "https://chinarxiv.org" + pdf
        source_url = str(item.get("source_url") or "")
        subjects = item.get("subjects") or []
        papers.append(
            Paper(
                paper_id=f"chinaxiv:{ident}" if ident else normalize_paper_id(item.get("title")),
                title=_WS_RE.sub(" ", html.unescape(str(item["title"]))),
                authors="; ".join(str(a) for a in (item.get("authors") or []) if a),
                abstract=_clean_abstract(item.get("abstract")),
                pdf_url=pdf,
                url=source_url or pdf,
                published=str(item.get("date") or ""),
                source="chinaxiv",
                categories="; ".join(str(s) for s in subjects if s),
            )
        )
    return papers


async def search_chinaxiv(
    query: str, limit: int = 10, settings: Settings | None = None, channel: dict[str, Any] | None = None
) -> list[Paper]:
    """ChinaXiv 中文预印本（公开 API，免 key；填 email 进 polite pool 提高配额）。"""
    s = settings or get_settings()
    creds = _channel_creds(s, "chinaxiv", channel)
    headers = {"X-API-Email": creds["email"]} if creds.get("email") else {}
    resp = await _request(
        creds.get("base_url") or CHINAXIV_API,
        {
            "q": clean_pasted(query),
            "limit": max(1, min(limit, 100)),
            "source": "chinaxiv",  # 只要中文语料，不要 SovietRxiv
        },
        s,
        headers=headers,
    )
    return parse_chinaxiv(resp.json() or {})[:limit]


def parse_baidu_scholar(data: dict[str, Any]) -> list[Paper]:
    """百度学术（千帆 API）响应 → Paper（纯函数）。

    响应形如 `{code, data: [{title, abstract, aiAbstract, keyword, doi, paperId, publishYear}]}`。
    """
    papers: list[Paper] = []
    for item in first_list(data, "data", "list", "results", "papers"):
        if not isinstance(item, dict):
            continue
        title = _WS_RE.sub(" ", html.unescape(str(item.get("title") or "")))
        if not title:
            continue
        doi = str(item.get("doi") or "")
        pid = str(item.get("paperId") or item.get("paper_id") or "")
        authors = item.get("authors") or item.get("author") or ""
        if isinstance(authors, list):
            authors = "; ".join(
                str(a.get("name") if isinstance(a, dict) else a) for a in authors if a
            )
        url = str(item.get("url") or item.get("link") or "")
        if not url:
            url = f"https://xueshu.baidu.com/s?wd={quote(title)}"
        year = item.get("publishYear") or item.get("year") or ""
        papers.append(
            Paper(
                # 用 DOI/paperId 做稳定 key，便于与其它源去重
                paper_id=normalize_paper_id(doi)
                or (f"baidu:{pid}" if pid else normalize_paper_id(title)),
                title=title,
                authors=str(authors),
                abstract=_clean_abstract(
                    item.get("abstract") or item.get("aiAbstract") or item.get("summary")
                ),
                url=url,
                doi=doi,
                published=str(year),
                source="baidu_scholar",
                categories=str(item.get("keyword") or ""),
            )
        )
    return [p for p in papers if p.paper_id]


async def search_baidu_scholar(
    query: str, limit: int = 10, settings: Settings | None = None, channel: dict[str, Any] | None = None
) -> list[Paper]:
    """百度学术检索（百度千帆官方 API，需要 Bearer key）。"""
    s = settings or get_settings()
    creds = _channel_creds(s, "baidu_scholar", channel)
    key = creds.get("api_key")
    if not key:
        raise RuntimeError("百度学术需要千帆 API key：/channels add baidu_scholar --key <KEY>")
    resp = await _request(
        creds.get("base_url") or BAIDU_SCHOLAR_API,
        {
            "wd": clean_pasted(query),
            "pageNum": 0,
            "enable_abstract": "true",
        },
        s,
        headers={"Authorization": f"Bearer {key}"},
    )
    payload = resp.json() or {}
    code = str(payload.get("code", "0"))
    if code not in {"0", "200"} and not payload.get("data"):
        raise RuntimeError(f"百度学术返回错误：{payload.get('message') or code}")
    return parse_baidu_scholar(payload)[:limit]


def _wanfang_items(data: dict[str, Any]) -> list[Any]:
    """万方响应里"结果列表"的位置不太统一，这里做兼容提取（纯函数）。"""
    payload = data or {}
    for key in ("data", "list", "rows", "records", "results", "papers", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for inner in ("list", "rows", "records", "results", "papers", "items"):
                if isinstance(value.get(inner), list):
                    return value[inner]
    return []


def _pick(item: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """按候选字段名取第一个非空值（兼容中英文字段）。"""
    for key in keys:
        value = item.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def parse_wanfang(data: dict[str, Any]) -> list[Paper]:
    """万方开放平台响应 → Paper（纯函数，字段名做中英文兼容）。"""
    papers: list[Paper] = []
    for item in _wanfang_items(data):
        if not isinstance(item, dict):
            continue
        title = _WS_RE.sub(
            " ", html.unescape(str(_pick(item, "title", "题名", "paperTitle", "name")))
        )
        if not title:
            continue
        doi = str(_pick(item, "doi", "DOI"))
        pid = str(_pick(item, "id", "paperId", "论文ID", "docId"))
        authors = _pick(item, "authors", "author", "作者", "creator")
        if isinstance(authors, list):
            authors = "; ".join(
                str(a.get("name") if isinstance(a, dict) else a) for a in authors if a
            )
        journal = _pick(item, "source", "journal", "期刊", "periodical", "venue")
        url = str(_pick(item, "url", "link", "detailUrl", "原文链接"))
        if not url and pid:
            url = f"https://d.wanfangdata.com.cn/periodical/{pid}"
        papers.append(
            Paper(
                paper_id=normalize_paper_id(doi)
                or (f"wanfang:{pid}" if pid else normalize_paper_id(title)),
                title=title,
                authors=str(authors),
                abstract=_clean_abstract(_pick(item, "abstract", "摘要", "summary")),
                pdf_url=str(_pick(item, "pdfUrl", "pdf_url", "fullTextUrl")),
                url=url,
                doi=doi,
                published=str(_pick(item, "year", "publishYear", "年份", "date")),
                source="wanfang",
                categories=str(journal),
                citations=int(_pick(item, "citedCount", "被引", "citations", default=0) or 0),
            )
        )
    return [p for p in papers if p.paper_id]


async def search_wanfang(
    query: str, limit: int = 10, settings: Settings | None = None, channel: dict[str, Any] | None = None
) -> list[Paper]:
    """万方数据检索（开放平台 API，需要 APPCODE 或 `AppKey:APPCODE`）。

    不同订阅的网关路径/参数可能不同：`/channels add wanfang --base-url <你的接口地址>`
    可覆盖 `base_url`。鉴权头按官方文档：`X-Ca-AppKey` + `Authorization: APPCODE <appcode>`。
    """
    s = settings or get_settings()
    creds = _channel_creds(s, "wanfang", channel)
    key = creds.get("api_key") or ""
    if not key:
        raise RuntimeError(
            "万方数据需要开放平台 key：/channels add wanfang --key <APPCODE 或 AppKey:APPCODE>"
        )
    headers = {"Content-Type": "application/json"}
    app_key, sep, appcode = key.partition(":")
    if sep and appcode.strip():  # “AppKey:APPCODE”
        headers["X-Ca-AppKey"] = app_key.strip()
        headers["Authorization"] = f"APPCODE {appcode.strip()}"
    else:
        headers["Authorization"] = f"APPCODE {key.strip()}"
    resp = await _post_json(
        creds.get("base_url") or WANFANG_API,
        {"query": clean_pasted(query), "page": 1, "pageSize": max(1, min(limit, 50))},
        s,
        headers=headers,
    )
    return parse_wanfang(resp.json() or {})[:limit]


def parse_nlc(data: dict[str, Any]) -> list[Paper]:
    """国家图书馆「图书馆检索」联合目录响应 → Paper（纯函数）。

    响应 `result[]._source` 是 MARC 风格字段：`TIT`(题名) / `AUT`(著者) / `PUB`(出版社)
    / `YEA`(出版年) / `ISB`(ISBN) / `ISS`(ISSN) / `CLC`(中图分类号) / `SUB`(主题词)
    / `LAN`(语言) / `SET`(丛编) / `SYS`(书目系统号) / `holdings`(馆藏)；书目无摘要。
    """
    papers: list[Paper] = []
    for item in (data or {}).get("result") or []:
        if not isinstance(item, dict):
            continue
        src = item.get("_source") or {}
        if not isinstance(src, dict):
            continue
        titles = src.get("TIT") or []
        title = _WS_RE.sub(" ", html.unescape(_first(titles))).strip()
        if not title:  # TAI 形如“红楼梦-(清)曹雪芹著”
            title = _WS_RE.sub(" ", html.unescape(str(src.get("TAI") or ""))).strip()
        if not title:
            continue
        isbn = _first(src.get("ISB") or []) or _first(src.get("ISS") or [])
        sys_id = str(src.get("SYS") or "")
        raw_id = str(item.get("_id") or "")
        holdings = src.get("holdings") or []
        papers.append(
            Paper(
                # ISBN 做稳定 key（便于同书跨馆藏去重）；否则用书目系统号
                paper_id=f"isbn:{isbn.replace('-', '').strip()}"
                if isbn
                else (f"nlc:{sys_id}" if sys_id else normalize_paper_id(raw_id or title)),
                title=title,
                authors="; ".join(str(a) for a in (src.get("AUT") or []) if a),
                abstract="",  # 书目数据没有摘要
                url=f"https://meta.nlc.cn/v2/detail/{raw_id}" if raw_id else "",
                published=_first(src.get("YEA") or []),
                source="nlc",
                categories="; ".join(
                    str(x) for x in [*(src.get("CLC") or []), *(src.get("SUB") or [])] if x
                ),
                extra={
                    "isbn": isbn,
                    "publisher": str(src.get("PUB") or ""),
                    "place": str(src.get("PUL") or ""),
                    "series": [str(s) for s in (src.get("SET") or []) if s],
                    "language": _first(src.get("LAN") or []),
                    "holding_count": int(src.get("holding_count") or len(holdings) or 0),
                    "holdings": holdings[:5],
                },
            )
        )
    return [p for p in papers if p.paper_id]


async def search_nlc(
    query: str, limit: int = 10, settings: Settings | None = None, channel: dict[str, Any] | None = None
) -> list[Paper]:
    """国家图书馆「图书馆检索」联合目录（公开接口，免 key；图书/古籍/学位论文书目）。

    该后台只支持**按字段**检索（`searchWay=TIT/SUB/AUT/...`，`ANY` 无效）：
    这里并发跑 `TIT`（书名，含出版社/年份）与 `SUB`（主题/关键词，多为学位论文）
    两路再合并去重，兼顾「找某本书」与「按主题找资料」。
    """
    s = settings or get_settings()
    creds = _channel_creds(s, "nlc", channel)
    url = creds.get("base_url") or NLC_API
    if "hight=" not in url:  # 门户固定带 hight=false（普通检索）
        url = f"{url}{'&' if '?' in url else '?'}hight=false"
    q = clean_pasted(query)
    size = max(1, min(limit, 50))

    async def _run(way: str) -> list[Paper]:
        payload = {
            "q": q,
            "searchWay": way,
            "query": f"{way}={q}",
            "sort": [{"_score": "DESC"}],
            "page_num": "1",
            "size": size,
            "facetVal": [],
            "exactMatch": False,
        }
        # ensure_ascii：绕开国图后台对原始 UTF-8 body 的解析 bug（否则忽略检索词）
        resp = await _post_json(url, payload, s, ensure_ascii=True)
        return parse_nlc(resp.json() or {})

    outcomes = await asyncio.gather(_run("TIT"), _run("SUB"), return_exceptions=True)
    papers: list[Paper] = []
    seen: set[str] = set()
    errors: list[BaseException] = []
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            errors.append(outcome)
            continue
        for paper in outcome:
            if paper.paper_id in seen:
                continue
            seen.add(paper.paper_id)
            papers.append(paper)
    if not papers and errors:
        raise errors[0]
    return papers[:limit]


# kind -> 渠道检索函数（builtin_search 按此分发）
# DBLP 有 bot 防护（无浏览器环境会被拦）故不登记；知网/维普/超星无公开检索 API。
CHANNEL_SEARCHERS: dict[str, Any] = {
    "europepmc": search_europepmc,
    "pubmed": search_pubmed,
    "semanticscholar": search_semanticscholar,
    "doaj": search_doaj,
    "core": search_core,
    "chinaxiv": search_chinaxiv,
    "baidu_scholar": search_baidu_scholar,
    "wanfang": search_wanfang,
    "nlc": search_nlc,
    "tavily": search_tavily,
    "exa": search_exa,
    "serpapi": search_serpapi,
}


# --------------------------------------------------------------------------
# 统一入口：按 ID 解析 / 关键词检索
# --------------------------------------------------------------------------

def registered_sources() -> list[str]:
    """全部「已注册」的可检索渠道 kind（运行时读 `CHANNEL_SEARCHERS`，便于测试）。"""
    return ["arxiv", "openalex", "crossref", *CHANNEL_SEARCHERS.keys()]


def papers_domestic_first(papers: list[Paper]) -> list[Paper]:
    """稳定排序：国内渠道的结果排在前面（其余保持原顺序）。"""
    dom = [p for p in papers if is_domestic(p.source)]
    rest = [p for p in papers if not is_domestic(p.source)]
    return [*dom, *rest]


def classify_identifier(raw: str) -> tuple[str, str]:
    """判断标识类型：("arxiv"|"doi"|"url"|"unknown", 归一化值)。"""
    text = clean_pasted(raw)
    if not text:
        return "unknown", ""
    normalized = normalize_paper_id(text)
    if normalized.startswith("arxiv:"):
        return "arxiv", normalized.split(":", 1)[1]
    if normalized.startswith("doi:"):
        return "doi", normalized.split(":", 1)[1]
    if re.match(r"^https?://", text):
        return "url", text
    return "unknown", text


async def resolve_identifier(raw: str, settings: Settings | None = None) -> Paper | None:
    """把一个标识（arXiv ID / DOI / 链接）解析成 Paper。"""
    s = settings or get_settings()
    kind, value = classify_identifier(raw)
    try:
        if kind == "arxiv":
            return await resolve_arxiv(value, s)
        if kind == "doi":
            paper = await resolve_doi_openalex(value, s)
            if paper is not None and paper.title:
                return paper
            return await resolve_doi_crossref(value, s)
        if kind == "url":
            if "arxiv.org" in value:
                return await resolve_arxiv(value, s)
            doi_match = re.search(r"10\.\d{4,9}/[^\s?#]+", value)
            if doi_match:
                return await resolve_identifier(f"doi:{doi_match.group(0)}", s)
    except Exception as exc:  # noqa: BLE001
        logger.warning("解析 %s 失败：%s", raw, exc)
    return None


async def resolve_ids(
    ids: Iterable[str],
    settings: Settings | None = None,
    limit: int | None = None,
) -> tuple[list[Paper], list[str]]:
    """按 ID 列表直接抓元数据（并发，保序去重）。

    Returns:
        (papers, failures)：failures 是解析失败的原始标识。
    """
    s = settings or get_settings()
    wanted = [clean_pasted(i) for i in ids if clean_pasted(i)]
    if limit:
        wanted = wanted[:limit]
    if not wanted:
        return [], []

    results = await asyncio.gather(
        *(resolve_identifier(item, s) for item in wanted), return_exceptions=True
    )
    papers: list[Paper] = []
    failures: list[str] = []
    seen: set[str] = set()
    for raw, item in zip(wanted, results):
        if isinstance(item, BaseException) or item is None:
            failures.append(raw)
            continue
        if item.paper_id in seen:
            continue
        seen.add(item.paper_id)
        papers.append(item)
    return papers, failures


# 每个渠道的检索进度事件（供 REPL 逐渠道渲染；headless 时不传回调）
# 事件字段：name(展示名) status(queued/running/done/failed/skipped) count error reason elapsed
ProgressCallback = Callable[[dict[str, Any]], None]


async def builtin_search(
    query: str,
    limit: int = 8,
    settings: Settings | None = None,
    sources: str = "",
    on_event: ProgressCallback | None = None,
    by_channel: dict[str, list[Paper]] | None = None,
) -> tuple[list[Paper], str]:
    """内置检索（MCP 回退）：每个渠道各自独立请求、并发执行。

    源解析优先级：显式 `sources` > `all` / `search_all_channels` > `builtin_sources`，
    已启用的渠道总会追加。全渠道模式下「需 key 但未配 key」的自动跳过并标 `⊘`。
    并发上限 `channel_concurrency`，单渠道超时 `channel_timeout`。
    """
    s = settings or get_settings()

    # 已配置并启用的渠道：登记凭据（name/kind/api_key/…）
    channels: dict[str, dict[str, Any]] = {}
    for item in s.enabled_channels():
        kind = str(item.get("kind") or "").lower()
        if kind and kind not in channels:
            channels[kind] = item

    disabled = {
        str(item.get("kind") or "").lower()
        for item in (s.search_channels or {}).values()
        if isinstance(item, dict) and not item.get("enabled", True)
    }

    explicit = [x.strip().lower() for x in (sources or "").split(",") if x.strip()]
    all_mode = "all" in explicit or (not explicit and s.search_all_channels)
    if all_mode:
        wanted = [k for k in registered_sources() if k not in disabled]
    elif explicit:
        wanted = [k for k in explicit if k not in disabled]
    else:
        wanted = [x.strip().lower() for x in s.builtin_sources.split(",") if x.strip()]
    # 已启用的渠道一定参与（即使不在显式列表/默认列表里）
    for kind in channels:
        if kind not in wanted and kind not in disabled:
            wanted.append(kind)
    # 优先国内渠道：默认/all 模式下把国内库排到最前（显式 --sources 时尊重用户顺序）
    if s.prefer_domestic and (all_mode or not explicit):
        wanted = domestic_first(wanted)

    def _emit(name: str, status: str, **extra: Any) -> None:
        """上报单渠道进度；回调出错绝不能影响检索本身。"""
        if on_event is None:
            return
        try:
            on_event({"name": name, "status": status, **extra})
        except Exception:  # noqa: BLE001
            logger.debug("检索进度回调异常", exc_info=True)

    per_source = max(1, limit)
    tasks: list[tuple[str, str, Any]] = []  # (kind, 展示名, 协程)
    skipped: list[tuple[str, str]] = []  # (展示名, 原因)
    handled: set[str] = set()
    for kind in wanted:
        if kind in handled:
            continue
        handled.add(kind)
        entry = channels.get(kind) or {}
        label = str(entry.get("name") or kind)
        spec = spec_for(kind)
        needs_key = bool(spec and spec.needs_key)
        if needs_key and not entry.get("api_key"):
            # 显式点名就问一下（报“需key✗”）；全渠道/自动纳入的静静跳过
            if all_mode or kind not in explicit:
                skipped.append((label, "需key"))
                _emit(label, "skipped", kind=kind, reason="需key")
                continue
        if kind == "arxiv":
            tasks.append((kind, label, search_arxiv(query, per_source, s)))
        elif kind == "openalex":
            tasks.append((kind, label, search_openalex(query, per_source, s)))
        elif kind == "crossref":
            tasks.append((kind, label, search_crossref(query, per_source, s)))
        elif kind in CHANNEL_SEARCHERS:
            tasks.append((kind, label, CHANNEL_SEARCHERS[kind](query, per_source, s, entry or None)))
    skipped_txt = [f"{label}({reason})⊘" for label, reason in skipped]
    if not tasks:
        detail = ", ".join(skipped_txt)
        return [], (
            "未启用任何渠道：用 /channels add <编号|kind> 添加（/channels 查看可选）"
            + (f"；已跳过：{detail}" if detail else "")
        )

    # 每个渠道独立请求；用信号量限流 + 单渠道超时，个别慢源不拖垬整次检索
    sem = asyncio.Semaphore(max(1, s.channel_concurrency))
    channel_timeout = max(1.0, s.channel_timeout)

    async def _run(kind: str, name: str, coro: Any) -> list[Paper]:
        started = time.monotonic()
        _emit(name, "running", kind=kind)
        try:
            async with sem:
                result = await asyncio.wait_for(coro, timeout=channel_timeout)
        except BaseException as exc:
            _emit(name, "failed", kind=kind, error=_short_error(exc), elapsed=time.monotonic() - started)
            raise
        _emit(name, "done", kind=kind, count=len(result), elapsed=time.monotonic() - started)
        return result

    for kind, name, _coro in tasks:  # 先登记 queued，UI 立即列出所有渠道
        _emit(name, "queued", kind=kind)
    results = await asyncio.gather(
        *(_run(kind, name, coro) for kind, name, coro in tasks), return_exceptions=True
    )
    papers: list[Paper] = []
    used: list[str] = []
    failed: list[str] = []
    for (kind, name, _coro), outcome in zip(tasks, results):
        # `by_channel`：按渠道分别收集结果（key=kind），供 REPL 分渠道展示
        bucket = by_channel.setdefault(kind, []) if by_channel is not None else None
        if isinstance(outcome, BaseException):
            short = _short_error(outcome)
            logger.warning("检索源 %s 失败：%s", name, outcome)
            failed.append(f"{name}({short})✗")
            continue
        if bucket is not None:
            bucket.extend(outcome)
        papers.extend(outcome)
        used.append(f"{name}({len(outcome)})")
    if not used:
        detail = ", ".join([*failed, *skipped_txt])
        head = "渠道全部失败" if failed else "未启用任何渠道"
        return [], f"{head}{('：' + detail) if detail else ''}"
    if s.prefer_domestic:
        papers = papers_domestic_first(papers)
    route = "内置: " + ",".join([*used, *failed, *skipped_txt])
    return papers, route


def _short_error(exc: BaseException) -> str:
    """把异常压成短标签（给路由展示用，如 429 / Timeout / RuntimeError）。"""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status:
        return str(status)
    if isinstance(exc, asyncio.TimeoutError):
        return "Timeout"
    if isinstance(exc, RuntimeError):
        text = str(exc)
        if "API key" in text or "需要" in text:
            return "需key"
        return "RuntimeError"
    return type(exc).__name__
