# -*- coding: utf-8 -*-
"""检索过程中的 LLM 增强：查询扩展 + 相关性重排。

需求「在搜索过程中使用 llm」的落点。两件事都不改变检索层本身：
- `expand_queries`：把中文/口语化主题扩展成多条英文检索式（含同义词），提高召回；
- `rank_papers`：对多源合并后的候选做相关性排序/裁剪，减少明显不相关的噪声。

两条路径都**严格受超时约束**（默认 180s，见 `Settings.llm_timeout`），
模型不可用、超时或输出解析失败时一律**静默降级**为原始行为，不阻塞检索。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..core.config import Settings, get_settings
from ..core.schema import Paper
from ..core.utils import extract_json, first_list, truncate

logger = logging.getLogger(__name__)

EXPAND_PROMPT = """你是学术检索助手。把用户的研究主题扩展成 {n} 条**英文**检索式，用于 arXiv / OpenAlex / Semantic Scholar 等数据库。

要求：
- 每条只包含检索关键词/短语（可用 AND/OR），不要写成句子，不要加引号包裹整条；
- 覆盖：核心方法名、同义/近义表达、关键任务或评测（如 survey / benchmark）；
- 若主题本身是英文，也补 1-2 条同义扩展；
- 只输出 JSON：{{"queries": ["...", "..."]}}，不要输出多余文字。

研究主题：{query}"""

RANK_PROMPT = """你是学术相关性评审。下面是候选论文列表（id | 年份 | 标题 | 摘要片段）。
请按与主题的相关性从高到低排序，最多保留 {limit} 篇，剔除明显不相关的条目。

主题：{query}

候选：
{candidates}

只输出 JSON：{{"paper_ids": ["id1", "id2", ...]}}（按相关性降序，必须是上面出现过的 id）。"""


def _text_of(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # 多模态 content blocks
        return " ".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


async def _ainvoke_with_timeout(model: Any, prompt: str, timeout: float) -> str:
    result = await asyncio.wait_for(model.ainvoke(prompt), timeout=timeout)
    return _text_of(result)


async def expand_queries(
    model: Any,
    query: str,
    n: int = 3,
    settings: Settings | None = None,
) -> list[str]:
    """LLM 查询扩展；失败/超时则只返回原始 query。"""
    s = settings or get_settings()
    query = (query or "").strip()
    if not query or model is None:
        return [query] if query else []
    prompt = EXPAND_PROMPT.format(n=max(1, n), query=query)
    try:
        raw = await _ainvoke_with_timeout(model, prompt, s.llm_timeout)
        payload = extract_json(raw)
        queries = [str(x).strip() for x in first_list(payload, "queries", "search_queries") if str(x).strip()]
    except asyncio.TimeoutError:
        logger.warning("查询扩展超时（%.0fs），使用原始检索式", s.llm_timeout)
        return [query]
    except Exception as exc:  # noqa: BLE001
        logger.warning("查询扩展失败（%s），使用原始检索式", exc)
        return [query]

    out: list[str] = []
    for item in [query, *queries]:
        if item and item not in out:
            out.append(item)
    return out[: max(1, n) + 1]


async def rank_papers(
    model: Any,
    query: str,
    papers: list[Paper],
    limit: int,
    settings: Settings | None = None,
) -> list[Paper]:
    """LLM 相关性重排；失败/超时则保留原顺序（截断到 limit）。"""
    s = settings or get_settings()
    if not papers:
        return []
    if model is None or len(papers) <= 1:
        return papers[:limit]

    lines = []
    for paper in papers[:40]:
        abstract = truncate(paper.abstract or "", 240)
        lines.append(f"{paper.paper_id} | {paper.published[:4]} | {truncate(paper.title, 140)} | {abstract}")
    prompt = RANK_PROMPT.format(query=query, limit=max(1, limit), candidates="\n".join(lines))
    try:
        raw = await _ainvoke_with_timeout(model, prompt, s.llm_timeout)
        payload = extract_json(raw)
        order = [str(x) for x in first_list(payload, "paper_ids", "papers", "ids") if str(x)]
    except asyncio.TimeoutError:
        logger.warning("相关性重排超时（%.0fs），保留原始顺序", s.llm_timeout)
        return papers[:limit]
    except Exception as exc:  # noqa: BLE001
        logger.warning("相关性重排失败（%s），保留原始顺序", exc)
        return papers[:limit]

    by_id = {paper.paper_id: paper for paper in papers}
    ranked = [by_id[pid] for pid in order if pid in by_id]
    # 补上模型没提到但确实检索到的（保持原有相对顺序），确保不会因为模型漏项而丢论文
    ranked_ids = {p.paper_id for p in ranked}
    ranked.extend(p for p in papers if p.paper_id not in ranked_ids)
    return ranked[:limit]
