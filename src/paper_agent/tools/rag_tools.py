# -*- coding: utf-8 -*-
"""RAG 工具：检索本地语料并返回带 `[C#]` 锚点的上下文。"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from ..core.config import Settings, get_settings
from ..rag.retriever import CitationCollector, format_context, retrieve
from ..rag.store import PaperIndex


def make_rag_tools(
    index: PaperIndex,
    collector: CitationCollector,
    settings: Settings | None = None,
) -> list[BaseTool]:
    """构造「检索语料 / 读取片段」工具集（引用锚点写入 collector）。"""
    s = settings or get_settings()

    @tool
    def search_corpus(query: str, paper_ids: str = "", k: int = 0) -> str:
        """在已入库的论文全文中做语义检索，返回带 [C#] 锚点的原文片段。

        Args:
            query: 检索问题/关键词（建议用英文以获得更好召回）。
            paper_ids: 逗号分隔的论文 id 列表，用于限定检索范围；留空表示全库。
            k: 返回片段数，0 表示使用默认值。
        """
        ids = [p.strip() for p in paper_ids.split(",") if p.strip()] or None
        docs = retrieve(
            index,
            query,
            k=k or s.top_k,
            paper_ids=ids,
            hybrid=s.hybrid_retrieval,
        )
        if not docs:
            return "检索结果为空（索引可能还没有论文，或论文 id 过滤过严）"
        return format_context(docs, collector)

    @tool
    def read_chunk(citation_id: str) -> str:
        """按引用编号读取更完整的原文片段（当上下文被截断时使用）。

        Args:
            citation_id: 形如 C3 的引用编号。
        """
        cite = collector.citations.get(citation_id)
        if not cite:
            return f"未知引用编号 {citation_id}"
        # 用 snippet 所在位置无法精确定位，这里直接返回已缓存的片段
        return f"[{citation_id}] {cite.paper_id} p.{cite.page}\n{cite.snippet}"

    return [search_corpus, read_chunk]
