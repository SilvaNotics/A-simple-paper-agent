# -*- coding: utf-8 -*-
"""写作 agent：把逐篇摘要与问答结论组织成报告正文。"""

from __future__ import annotations

import asyncio
import logging

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from ..core.utils import extract_json, truncate
from . import prompts
from .common import result_text, structured_response

logger = logging.getLogger(__name__)


class WriterOutput(BaseModel):
    """报告正文各部分。"""

    executive_summary: str = Field(default="", description="核心结论 3~5 句")
    comparison: str = Field(default="", description="横向对比（Markdown）")
    gaps: str = Field(default="", description="局限与研究空白")
    conclusion: str = Field(default="", description="实践建议")


def make_writer_agent(model: BaseChatModel):
    return create_agent(
        model=model,
        tools=[],
        system_prompt=prompts.WRITER_PROMPT_ZH,
        response_format=WriterOutput,
        name="writer_agent",
    )


async def run_writer_agent(
    agent,
    topic: str,
    material: str,
    timeout: float = 180.0,   # 单次等待响应上限：180s
) -> WriterOutput:
    """写报告正文（失败时把材料原样放入 comparison，保证报告仍有内容）。"""
    task = f"研究主题：{topic}\n\n材料：\n{material}\n\n请按要求输出报告各部分。"
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": task}]}, {"recursion_limit": 8}),
            timeout=timeout,
        )
        out = structured_response(result)
        if isinstance(out, WriterOutput) and (out.executive_summary or out.comparison):
            return out

        payload = extract_json(result_text(result))
        if isinstance(payload, dict):
            candidate = WriterOutput(**{k: v for k, v in payload.items() if k in WriterOutput.model_fields})
            if candidate.executive_summary or candidate.comparison:
                return candidate
    except Exception as exc:  # noqa: BLE001
        logger.warning("写作 agent 失败：%s", exc)

    return WriterOutput(comparison=truncate(material, 6000))


def material_from_state(summaries: list, answers: list, limit: int = 12000) -> str:
    """把图状态里的摘要与问答拼成写作用的材料（有长度上限）。"""
    parts: list[str] = []
    for s in summaries:
        parts.append(
            f"### {s.paper_id} — {s.title}\n"
            f"- 问题：{s.problem}\n- 方法：{s.method}\n- 数据/实验：{s.data}\n"
            f"- 结论：{s.findings}\n- 局限：{s.limitations}\n- 可复用点：{s.reusable_ideas}\n"
            f"- 置信度：{s.confidence:.2f}"
        )
    for a in answers:
        parts.append(f"### 问答：{a.question}\n{a.text}")
    return truncate("\n\n".join(parts), limit)
