# -*- coding: utf-8 -*-
"""精读 agent：把单篇论文（全文片段 + 摘要）压成结构化摘要。"""

from __future__ import annotations

import asyncio
import logging

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from ..core.schema import Paper, PaperSummary
from ..rag.retriever import anchors_in
from ..core.utils import extract_json, truncate
from . import prompts
from .common import result_text, structured_response

logger = logging.getLogger(__name__)


def make_summarize_agent(model: BaseChatModel):
    return create_agent(
        model=model,
        tools=[],
        system_prompt=prompts.SUMMARIZE_PROMPT_ZH,
        response_format=PaperSummary,
        name="summarize_agent",
    )


def fallback_summary(paper: Paper) -> PaperSummary:
    """无全文/模型失败时的兜底摘要（只基于摘要，置信度低）。"""
    return PaperSummary(
        paper_id=paper.paper_id,
        title=paper.title,
        problem=truncate(paper.abstract, 320) or "原文未提及",
        method="原文未提及（未能获取全文）",
        data="原文未提及",
        findings="原文未提及",
        limitations="原文未提及",
        reusable_ideas="原文未提及",
        key_quotes=[],
        confidence=0.3 if paper.abstract else 0.1,
        retrieved_chunks=0,
    )


async def run_summarize_agent(
    agent,
    paper: Paper,
    context: str,
    has_fulltext: bool,
    timeout: float = 180.0,   # 单次等待响应上限：180s
) -> PaperSummary:
    """生成单篇结构化摘要（失败时退化为「仅摘要」版本）。"""
    body = (
        f"论文：{paper.title}\n"
        f"作者：{paper.authors}\n"
        f"发表：{paper.published}  来源：{paper.source}  ID：{paper.paper_id}\n"
        f"链接：{paper.url or paper.pdf_url}\n\n"
        f"摘要：{paper.abstract or '（无摘要）'}\n\n"
        + (f"全文片段：\n{context}" if has_fulltext else "（未能获取全文 PDF，只能基于摘要分析）")
    )
    task = (
        f"{body}\n\n请输出结构化中文摘要。"
        f"paper_id 字段填 `{paper.paper_id}`，title 填原文标题。"
    )

    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": task}]}, {"recursion_limit": 8}),
            timeout=timeout,
        )
        summary = structured_response(result)
        if isinstance(summary, PaperSummary) and (summary.method or summary.findings or summary.problem):
            summary.paper_id = summary.paper_id or paper.paper_id
            summary.title = summary.title or paper.title
            summary.retrieved_chunks = len(anchors_in(context))
            if not has_fulltext and summary.confidence > 0.6:
                summary.confidence = 0.5
            return summary

        payload = extract_json(result_text(result))
        if isinstance(payload, dict) and (payload.get("method") or payload.get("findings")):
            summary = PaperSummary(**{k: v for k, v in payload.items() if k in PaperSummary.model_fields})
            summary.paper_id = summary.paper_id or paper.paper_id
            summary.title = summary.title or paper.title
            return summary
    except Exception as exc:  # noqa: BLE001
        logger.warning("精读 agent 失败（%s）：%s", paper.paper_id, exc)

    return fallback_summary(paper)
