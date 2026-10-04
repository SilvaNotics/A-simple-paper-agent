# -*- coding: utf-8 -*-
"""数据模型与图状态定义。

约定：
- 所有 LLM 结构化输出都用这里的 Pydantic 模型，`create_agent(response_format=...)`；
- `ResearchState` 是 LangGraph 的图状态，列表字段用 `operator.add` 归并（配合 `Send` 并行）。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from .utils import normalize_paper_id

# --------------------------------------------------------------------------
# 论文
# --------------------------------------------------------------------------


class Paper(BaseModel):
    """统一后的论文元数据（融合 arXiv MCP 与 paper-search MCP 两种返回格式）。"""

    paper_id: str = Field(default="", description="归一化 ID，如 arxiv:2405.16506 或 doi:10.1145/xxx")
    title: str = ""
    authors: str = ""
    abstract: str = ""
    pdf_url: str = ""
    url: str = ""
    doi: str = ""
    published: str = ""
    source: str = ""
    categories: str = ""
    citations: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_open_access_hint(self) -> bool:
        url = (self.pdf_url or self.url or "").lower()
        return any(k in url for k in ("arxiv.org", "ncbi.nlm.nih.gov", "europepmc.org", ".pdf"))


class PaperList(BaseModel):
    """search_agent 的结构化输出。"""

    papers: list[Paper] = Field(default_factory=list, description="检索到的候选论文")


def coerce_paper(raw: dict[str, Any], source: str = "") -> Paper | None:
    """把 MCP 返回的论文 dict 归一化成 `Paper`（兼容 arXiv/paper-search 字段差异）。"""
    if not isinstance(raw, dict):
        return None

    authors = raw.get("authors")
    if isinstance(authors, (list, tuple)):
        authors = "; ".join(str(a) for a in authors if a)
    elif authors is None:
        authors = ""

    candidates = [
        raw.get("paper_id"),
        raw.get("versioned_id"),
        raw.get("id"),
        raw.get("doi"),
        raw.get("url"),
    ]
    normalized = [normalize_paper_id(c) for c in candidates if c]
    # OpenAlex 的 W-id 不利于跨源去重，若有 DOI/arXiv 形态的标识则优先用
    better = next((n for n in normalized if not n.startswith("openalex:")), "")
    paper_id = better or (normalized[0] if normalized else "")
    doi_value = raw.get("doi") or (
        paper_id.split(":", 1)[1] if paper_id.startswith("doi:") else ""
    )

    categories = raw.get("categories") or raw.get("primary_category") or ""
    if isinstance(categories, (list, tuple)):
        categories = ", ".join(str(c) for c in categories)

    abstract = raw.get("abstract") or raw.get("summary") or ""
    published = raw.get("published") or raw.get("published_date") or raw.get("update_date") or ""

    try:
        citations = int(raw.get("citations") or 0)
    except (TypeError, ValueError):
        citations = 0

    raw_url = raw.get("url") or raw.get("resource_uri") or ""
    pdf_url = raw.get("pdf_url") or ""
    if not pdf_url and isinstance(raw_url, str) and ("/pdf/" in raw_url or raw_url.lower().endswith(".pdf")):
        pdf_url = raw_url

    paper = Paper(
        paper_id=paper_id,
        title=(raw.get("title") or "").strip(),
        authors=str(authors).strip(),
        abstract=str(abstract).strip(),
        pdf_url=pdf_url,
        url=raw_url,
        doi=doi_value,
        published=str(published),
        source=raw.get("source") or source,
        categories=str(categories),
        citations=citations,
    )
    if not paper.title and not paper.abstract:
        return None
    return paper


# --------------------------------------------------------------------------
# 摘要 / 回答 / 引用
# --------------------------------------------------------------------------


class KeyQuote(BaseModel):
    text: str = Field(description="原文关键句（尽量逐字摘录）")
    locator: str = Field(default="", description="定位信息，如 p.7 或 chunk:12")


class PaperSummary(BaseModel):
    """单篇论文的结构化精读结果。"""

    paper_id: str = ""
    title: str = ""
    problem: str = Field(default="", description="研究问题/动机")
    method: str = Field(default="", description="核心方法/技术路线")
    data: str = Field(default="", description="数据集/实验设置")
    findings: str = Field(default="", description="主要结论与量化结果")
    limitations: str = Field(default="", description="局限与未来工作")
    reusable_ideas: str = Field(default="", description="可被复用的思路")
    key_quotes: list[KeyQuote] = Field(default_factory=list)
    confidence: float = Field(default=0.5, description="0~1，证据充分度")
    retrieved_chunks: int = 0


class Citation(BaseModel):
    """检索片段引用锚点。"""

    id: str
    paper_id: str
    page: int | None = None
    snippet: str = ""

    def label(self) -> str:
        return f"{self.paper_id} p.{self.page}" if self.page else self.paper_id


class Answer(BaseModel):
    """带引用的问答结果。"""

    question: str = ""
    text: str = ""
    citation_ids: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(
        default_factory=list, description="校验发现缺少原文支撑的引用 id（由代码回填）"
    )


class SelectionItem(BaseModel):
    """筛选节点对单篇论文的打分。"""

    paper_id: str
    score: float = 0.5
    reason: str = ""


class SelectionOutput(BaseModel):
    """筛选节点的结构化输出。"""

    selected: list[SelectionItem] = Field(default_factory=list)


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """图状态里 dict 字段的 reducer（Send 并行写入时合并）。"""
    out: dict = dict(left or {})
    out.update(right or {})
    return out


# --------------------------------------------------------------------------
# 入库状态
# --------------------------------------------------------------------------

# 入库结果状态 → 是否已进入 RAG 索引（chunks > 0）。
# - indexed：PDF 全文；web：网页正文；abstract/metadata：摘要/题录（非全文，已标注）
# - cached：索引中已存在；no_pdf/parse_error/empty/embed_error：未能入库
INDEXED_STATUSES = frozenset({"indexed", "cached", "web", "abstract", "metadata"})

# 终端（rich）打印用的状态配色
INGEST_STATUS_STYLES = {
    "indexed": "green",
    "web": "green",
    "abstract": "green",
    "metadata": "green",
    "cached": "cyan",
}


# --------------------------------------------------------------------------
# 图状态
# --------------------------------------------------------------------------


class ResearchState(TypedDict, total=False):
    """学术调研图的完整状态。"""

    query: str
    sub_questions: list[str]
    search_queries: list[str]
    candidates: Annotated[list[Paper], operator.add]
    selected: list[Paper]
    ingest_results: Annotated[list[dict[str, Any]], operator.add]
    summaries: Annotated[list[PaperSummary], operator.add]
    answers: Annotated[dict[str, Answer], merge_dicts]
    citations: Annotated[dict[str, Citation], merge_dicts]
    report_md: str
    bibtex: str
    verify_issues: list[str]
    # worker 节点异常（并发写入，必须用 reducer）
    worker_errors: Annotated[list[str], operator.add]
    retries: int
    retry: bool
    notes: Annotated[list[str], operator.add]
    simple: bool
    # Send 扇出时传给 worker 的“当前条目”
    current_paper: Paper | None
    current_question: str
    branch: str
