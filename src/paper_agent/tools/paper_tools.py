# -*- coding: utf-8 -*-
"""本地工具：补链/下载 → 解析 → 切分 → 入索引。

这些工具是**确定性**的（不经过 LLM），既可以给 agent 调用，也可以被图节点直接调用。

没有 PDF 直链时按「补链 → 网页正文 → 题录/摘要」逐级降级（见 `sources/oa.py`）：
- `indexed`：PDF 全文入库；
- `web`：抓落地页/网页正文入库（Wikipedia、百科、新闻页等）；
- `abstract`：只有摘要，入库时标注「非全文」；
- `metadata`：只有书目/题录，同样标注「非全文」；
- `no_pdf / parse_error / empty / embed_error`：确实无法入库。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

from ..core.config import Settings, get_settings
from ..core.schema import INDEXED_STATUSES, Paper
from ..core.utils import truncate as truncate_text
from ..rag.fetch import cached_pdf, download_pdf
from ..rag.parse import parse_pdf
from ..rag.split import build_record_text, split_paper, split_text_document
from ..rag.store import PaperIndex
from ..sources.oa import discover_pdf_url, fetch_page_text

logger = logging.getLogger(__name__)


def _join(*parts: str) -> str:
    """把若干说明用「；」拼起来（跳过空串）。"""
    return "；".join(p for p in parts if p)


async def _fallback_documents(
    paper: Paper,
    settings: Settings,
    message: str,
    result: dict[str, Any],
    reason: str = "无 PDF",
) -> tuple[Paper, list[Document], str, str]:
    """没有可入库全文时的降级链：网页正文 → 题录/摘要。

    `reason` 用于拼降级说明（无 PDF / PDF 无可用文本）。

    Returns:
        `(paper, documents, kind, message)`；`documents` 为空表示确实没有可入库内容。
    """
    if settings.web_fallback:
        page_title, text, page_url = await fetch_page_text(paper, settings)
        if text:
            if not paper.title and page_title:
                # 网页标题可以补空缺的元数据（如只抓到百度百科/维基词条时）
                paper = paper.model_copy(update={"title": page_title})
                result["title"] = page_title
            docs = split_text_document(paper, text, settings, kind="web", url=page_url)
            if docs:
                return paper, docs, "web", _join(message, f"{reason}，改抓网页正文（{len(text)} 字）")

    if settings.record_fallback:
        kind = "abstract" if paper.abstract.strip() else "metadata"
        docs = split_text_document(
            paper, build_record_text(paper, kind), settings, kind=kind, url=paper.url
        )
        if docs:
            label = "摘要" if kind == "abstract" else "题录"
            return paper, docs, kind, _join(message, f"{reason}，仅入库{label}（已标注非全文）")

    return paper, [], "", message


async def ingest_paper(
    paper: Paper,
    index: PaperIndex,
    settings: Settings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """完整入库一篇论文，返回可放进图状态的统计信息。"""
    s = settings or get_settings()
    result: dict[str, Any] = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "status": "skipped",
        "chunks": 0,
        "pages": 0,
        "engine": "",
        "message": "",
    }

    if not force and index.has_paper(paper.paper_id):
        result.update(status="cached", chunks=index.manifest["papers"][paper.paper_id].get("n_chunks", 0))
        result["message"] = "索引中已存在"
        return result

    # ① 补链：没有直链且本地也没有缓存时，用 Unpaywall/OpenAlex/落地页再找一次
    note = ""
    if s.pdf_lookup and not paper.pdf_url and cached_pdf(paper, s) is None:
        found, note = await discover_pdf_url(paper, s)
        if found:
            paper = paper.model_copy(update={"pdf_url": found})

    # ② PDF 全文：下载 → 解析 → 切分
    pdf_path, message = await download_pdf(paper, s)
    message = _join(note, message)

    documents: list[Document] = []
    kind = "pdf"
    pages = 0
    engine = ""
    if pdf_path is not None:
        try:
            parsed = parse_pdf(pdf_path)
        except Exception as exc:  # noqa: BLE001
            result.update(status="parse_error", message=f"解析失败 {type(exc).__name__}: {exc}")
            return result
        if not paper.title and parsed.title:
            # 元数据缺失（如 arXiv 元数据 API 限流，只凭 PDF 直链抓到）：用首页文本补个标题，
            # 否则 /papers、报告、引用里就只能看到裸的 paper_id。
            paper = paper.model_copy(update={"title": parsed.title})
            result["title"] = parsed.title
            logger.info("论文 %s 无标题元数据，改用 PDF 首页文本：%s", paper.paper_id, truncate_text(parsed.title, 80))
        documents = split_paper(paper, parsed, s)
        pages, engine = parsed.n_pages, parsed.engine
        if not documents:
            # 扫描版/无文字层：也走降级（网页版正文或摘要可能仍在）
            paper, documents, kind, message = await _fallback_documents(
                paper, s, "", result, reason="PDF 无可用文本"
            )
            if not documents:
                result.update(status="empty", message="解析成功但没有可用文本（可能是扫描版）")
                return result
    else:
        # ③ 降级：网页正文 → 题录/摘要（都带 `kind` 标注，避免被当成全文证据）
        # 去掉「没有直链」这句占位文案：降级成功的说明（或最终兜底）会替代它
        message = message.replace("没有 PDF 直链（仅有元数据）", "").strip("；")
        paper, documents, kind, message = await _fallback_documents(paper, s, message, result)

    if not documents:
        result.update(status="no_pdf", message=message or "没有 PDF 直链（仅有元数据）")
        return result

    if force:
        index.clear_paper(paper.paper_id)
    try:
        n = index.add_documents(documents, paper=paper, pdf=pdf_path if kind == "pdf" else None, kind=kind)
    except Exception as exc:  # noqa: BLE001 - 向量化失败时保留已成功的部分并落盘
        try:
            index.save()
        except Exception:  # noqa: BLE001
            logger.exception("保存部分索引失败")
        result.update(
            status="embed_error",
            pages=pages,
            engine=engine,
            message=f"{message}；解析成功但向量化失败：{type(exc).__name__}: {truncate_text(str(exc), 160)}",
        )
        return result

    if kind == "pdf":
        result["message"] = f"{message}；{pages} 页 / {n} chunks（{engine}）"
    else:
        label = {"web": "网页正文", "abstract": "仅摘要", "metadata": "仅题录"}.get(kind, kind)
        result["message"] = f"{message}；{n} chunks（{label}）"
    result.update(status=kind if kind != "pdf" else "indexed", chunks=n, pages=pages, engine=engine)
    return result


def make_paper_tools(index: PaperIndex, settings: Settings | None = None) -> list[BaseTool]:
    """构造「下载/入库/查看索引」工具集。"""
    s = settings or get_settings()

    @tool
    async def download_and_index_paper(paper_id: str, pdf_url: str = "", title: str = "") -> str:
        """下载指定论文的 PDF、解析并写入 RAG 索引。

        没有 PDF 直链时也会尝试补链 / 抓网页正文 / 记录题录，返回里会说明用了哪一级。

        Args:
            paper_id: 论文标识（arXiv id、DOI 或归一化 id）。
            pdf_url: PDF 直链；没有则留空（会尝试自动补链）。
            title: 论文标题（仅用于记录）。
        """
        paper = Paper(paper_id=paper_id, title=title, pdf_url=pdf_url)
        info = await ingest_paper(paper, index, s)
        return f"{info['paper_id']} [{info['status']}] {info['message']}"

    @tool
    def list_indexed_papers() -> str:
        """列出当前 RAG 索引里已入库的论文及其 chunk 数。"""
        papers = index.list_papers()
        if not papers:
            return "索引为空"
        return "\n".join(
            f"- {p['paper_id']} | {p.get('n_chunks', 0)} chunks | {p.get('kind', 'pdf')} | {p.get('title', '')}"
            for p in papers
        )

    return [download_and_index_paper, list_indexed_papers]
