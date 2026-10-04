# -*- coding: utf-8 -*-
"""本地工具：下载 PDF → 解析 → 切分 → 入索引。

这些工具是**确定性**的（不经过 LLM），既可以给 agent 调用，也可以被图节点直接调用。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool, tool

from ..config import Settings, get_settings
from ..rag.fetch import download_pdf
from ..rag.parse import parse_pdf
from ..rag.split import split_paper
from ..rag.store import PaperIndex
from ..schema import Paper
from ..utils import truncate as truncate_text

logger = logging.getLogger(__name__)


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

    pdf_path, message = await download_pdf(paper, s)
    result["message"] = message
    if pdf_path is None:
        result["status"] = "no_pdf"
        return result

    try:
        parsed = parse_pdf(pdf_path)
    except Exception as exc:  # noqa: BLE001
        result.update(status="parse_error", message=f"解析失败 {type(exc).__name__}: {exc}")
        return result

    docs = split_paper(paper, parsed, s)
    if not docs:
        result.update(status="empty", message="解析成功但没有可用文本（可能是扫描版）")
        return result

    if force:
        index.clear_paper(paper.paper_id)
    try:
        n = index.add_documents(docs, paper=paper, pdf=pdf_path)
    except Exception as exc:  # noqa: BLE001 - 向量化失败时保留已成功的部分并落盘
        try:
            index.save()
        except Exception:  # noqa: BLE001
            logger.exception("保存部分索引失败")
        result.update(
            status="embed_error",
            pages=parsed.n_pages,
            engine=parsed.engine,
            message=f"{message}；解析成功但向量化失败：{type(exc).__name__}: {truncate_text(str(exc), 160)}",
        )
        return result

    result.update(status="indexed", chunks=n, pages=parsed.n_pages, engine=parsed.engine)
    result["message"] = f"{message}；{parsed.n_pages} 页 / {n} chunks（{parsed.engine}）"
    return result


def make_paper_tools(index: PaperIndex, settings: Settings | None = None) -> list[BaseTool]:
    """构造「下载/入库/查看索引」工具集。"""
    s = settings or get_settings()

    @tool
    async def download_and_index_paper(paper_id: str, pdf_url: str = "", title: str = "") -> str:
        """下载指定论文的 PDF、解析并写入 RAG 索引。

        Args:
            paper_id: 论文标识（arXiv id、DOI 或归一化 id）。
            pdf_url: PDF 直链；没有则跳过下载。
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
        return "\n".join(f"- {p['paper_id']} | {p.get('n_chunks', 0)} chunks | {p.get('title', '')}" for p in papers)

    return [download_and_index_paper, list_indexed_papers]
