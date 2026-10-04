# -*- coding: utf-8 -*-
"""检索 agent：通过 MCP 工具在多个学术数据库里找论文。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ..core.schema import Paper, PaperList, coerce_paper
from ..core.utils import extract_json, first_list, mcp_result_to_text
from . import prompts
from .common import result_text, structured_response

logger = logging.getLogger(__name__)


def make_search_agent(model: BaseChatModel, tools: list[BaseTool]):
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=prompts.SEARCH_PROMPT_ZH,
        response_format=PaperList,
        name="search_agent",
    )


async def run_search_agent(
    agent,
    query: str,
    tools: list[BaseTool],
    limit: int = 6,
    source_hint: str = "",
    timeout: float = 180.0,   # 单次等待响应上限：180s
) -> tuple[list[Paper], str]:
    """跑一次检索 agent。

    Returns:
        (papers, route)：route 说明走了哪条路径（agent / agent-text / direct / failed）。
    """
    task = (
        f"研究主题：{query}\n"
        f"请检索 {limit} 篇左右最相关的学术论文"
        f"{('（优先使用 ' + source_hint + ' 源）') if source_hint else ''}。"
        "先少量检索确认召回质量，再补充检索，最后返回结构化候选列表。"
    )

    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": task}]}, {"recursion_limit": 30}),
            timeout=timeout,
        )
        structured = structured_response(result)
        if isinstance(structured, PaperList) and structured.papers:
            return structured.papers, "agent"

        payload = extract_json(result_text(result))
        raw = first_list(payload, "papers", "results", "items")
        if raw:
            papers = [p for p in (coerce_paper(item) for item in raw) if p]
            if papers:
                return papers, "agent-text"
    except Exception as exc:  # noqa: BLE001
        logger.warning("检索 agent 失败，改用直接调用 MCP 工具：%s", exc)

    papers, note = await direct_search(tools, query, limit)
    return papers, f"direct({note})"


# --------------------------------------------------------------------------
# 通用工具参数适配（不同 MCP server 的检索工具签名不一致）
# --------------------------------------------------------------------------


def _tool_arg_names(tool: BaseTool) -> set[str]:
    """取工具可接受的参数名（兼容 pydantic schema 与 MCP 原始 JSON schema）。"""
    args = getattr(tool, "args", None)
    if isinstance(args, dict) and args:
        return set(args)
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return set()
    if hasattr(schema, "model_json_schema"):
        try:
            return set((schema.model_json_schema().get("properties") or {}).keys())
        except Exception:  # noqa: BLE001
            return set()
    if isinstance(schema, dict):
        return set((schema.get("properties") or {}).keys())
    return set()


def _fit_search_args(tool: BaseTool, query: str, limit: int, sources: str = "") -> dict[str, Any]:
    """把通用检索参数适配到具体工具的参数名上。"""
    allowed = _tool_arg_names(tool)
    if not allowed:
        return {"query": query}
    out: dict[str, Any] = {}
    for key in ("query", "q", "search_query", "keywords", "keyword", "term"):
        if key in allowed:
            out[key] = query
            break
    for key in ("max_results", "max_results_per_source", "limit", "top_k", "num_results"):
        if key in allowed:
            out[key] = limit
            break
    if sources and "sources" in allowed:
        out["sources"] = sources
    if "sort_by" in allowed:
        out["sort_by"] = "relevance"
    return out or {"query": query}


def _pick_search_tools(tools: list[BaseTool], max_tools: int = 4) -> list[BaseTool]:
    """挑选直接检索用的工具：优先聚合工具 `search_papers`（可按 `sources` 精确限定）。"""
    aggregators = [t for t in tools if t.name.endswith("search_papers")]
    if aggregators:
        aggregators.sort(key=lambda t: 0 if "paper-search" in t.name else 1)
        return aggregators[:1]
    return [t for t in tools if "search" in t.name.lower()][:max_tools]


async def direct_search(
    tools: list[BaseTool], query: str, limit: int = 6, sources: str = ""
) -> tuple[list[Paper], str]:
    """不经过 LLM，直接调用 MCP 检索工具（降级路径 / `--offline` 快速路径）。"""
    picked = _pick_search_tools(tools)
    if not picked:
        return [], "没有可用的 search 工具"

    async def _one(tool: BaseTool):
        args = _fit_search_args(tool, query, max(2, limit), sources)
        try:
            out = await asyncio.wait_for(tool.ainvoke(args), timeout=120)
            return tool.name, out
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP 检索工具 %s 失败（%s）：%s", tool.name, args, exc)
            return tool.name, None

    results = await asyncio.gather(*(_one(t) for t in picked))

    papers: list[Paper] = []
    used: list[str] = []
    for name, out in results:
        if out is None:
            continue
        payload = extract_json(mcp_result_to_text(out))
        for item in first_list(payload, "papers", "results", "items", "data"):
            paper = coerce_paper(item, source=name.split("_", 1)[0])
            if paper:
                papers.append(paper)
        used.append(name)

    return papers, ",".join(used) if used else "全部失败"
