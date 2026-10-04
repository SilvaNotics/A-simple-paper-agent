# -*- coding: utf-8 -*-
"""`/quick`：检索 → **内存**抓全文 → 临时 RAG → 带引用回答（PDF 不落盘）。

与 `/ask` 的区别：`/ask` 只在**已入库**语料上问答；`/quick` 是「即问即用」——
现场从已启用渠道检索候选，把 PDF / 网页正文抓到**内存**、切分、向量化进**临时索引**，
立刻让 LLM 带引用回答，命令结束数据即释放。

因此它**不写** `data/papers/*.pdf`，也**不写** `data/index/`：
- PDF 字节：`rag.fetch.fetch_pdf_bytes()`（内存）；
- 解析：`rag.parse.parse_pdf_bytes()`（pymupdf `stream=`，不落临时文件）；
- 索引：`PaperIndex(..., persist=False)`，`save()` 变成空操作。

想把这些论文「留下来」，用 `/ingest --ids ...` 或 `/search ... --ingest`。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from ..core.config import Settings, get_settings
from ..core.schema import Answer, Paper
from ..core.utils import dedupe_papers
from ..llm.factory import get_embeddings
from ..rag.fetch import cached_pdf
from ..rag.store import PaperIndex
from ..tools.paper_tools import collect_documents
from .session import Session, ask, run_search

logger = logging.getLogger(__name__)

# `/quick` 默认规模：候选检索上限 / 真正抓全文的篇数
DEFAULT_QUICK_LIMIT = 8
DEFAULT_QUICK_PAPERS = 3


@dataclass
class QuickItem:
    """一篇被临时抓进内存索引的论文（命令结束即释放）。"""

    paper: Paper
    kind: str = "pdf"        # pdf / web / abstract / metadata / 失败原因
    pages: int = 0
    chunks: int = 0
    message: str = ""

    @property
    def full_text(self) -> bool:
        """是否拿到了 PDF 全文（否则只是网页正文 / 摘要 / 题录）。"""
        return self.kind == "pdf"


@dataclass
class QuickResult:
    question: str
    answer: Answer
    citations: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    items: list[QuickItem] = field(default_factory=list)
    route: str = ""
    context: str = ""
    message: str = ""        # 没能形成答案时的说明（如「没抓到可读内容」）

    @property
    def chunks(self) -> int:
        return sum(i.chunks for i in self.items)

    @property
    def papers(self) -> list[Paper]:
        return [i.paper for i in self.items]


def _is_id_like(text: str) -> bool:
    """问题串本身就是一个论文标识（arXiv id / DOI / URL）？"""
    from ..sources.fetchers import classify_identifier

    kind, _ = classify_identifier(text)
    return kind != "unknown"


def _candidate_score(paper: Paper, settings: Settings) -> int:
    """越小越优先：本地已有缓存 → 有 PDF 直链 → 只有元数据（大概率抓不到全文）。"""
    if cached_pdf(paper, settings) is not None:
        return 0
    return 1 if paper.pdf_url else 2


def pick_candidates(papers: list[Paper], n: int, settings: Settings) -> list[Paper]:
    """挑候选：**优先能拿到全文的**（本地缓存 > PDF 直链 > 仅元数据），同分保持原顺序。"""
    ordered = sorted(dedupe_papers(list(papers)), key=lambda p: _candidate_score(p, settings))
    return ordered[: max(1, n)]


async def _candidates(
    question: str, settings: Settings, limit: int, papers: int, session: Session | None, source: str
) -> tuple[list[Paper], str]:
    """候选来源：像 ID 就按 ID 直解析，否则走渠道检索（MCP → 内置回退）。"""
    if _is_id_like(question):
        from ..sources.fetchers import resolve_ids

        wanted = [x for x in question.replace(",", " ").split() if x]
        resolved, failures = await resolve_ids(wanted, settings, limit=max(1, papers))
        if failures:
            logger.warning("以下标识解析失败：%s", ", ".join(failures))
        return list(resolved), f"按 ID 解析 {len(resolved)}/{len(wanted)}"

    found, route = await run_search(
        question, settings, limit=limit, session=session, source=source
    )
    return list(found), route


async def run_quick(
    question: str,
    settings: Settings | None = None,
    session: Session | None = None,
    limit: int = DEFAULT_QUICK_LIMIT,
    papers: int = DEFAULT_QUICK_PAPERS,
    k: int = 0,
    source: str = "",
    stream_callback: Callable[[str], None] | None = None,
    on_fetched: Callable[[list["QuickItem"]], None] | None = None,
) -> QuickResult:
    """即问即用：检索 → 内存抓全文 → 临时索引 → 带引用回答（不落盘）。

    Args:
        question: 问题（也可以是 arXiv id / DOI 这类标识）。
        limit: 候选检索上限（每渠道）。
        papers: 真正抓全文的论文数（默认 3，越大越慢）。
        k: 检索片段数（0 = 用配置默认 `top_k`）。
        source: 检索层 `auto` / `mcp` / `builtin` / `all`（空 = 用配置）。
        stream_callback: 流式回调（交互式传，脚本留空）。
        on_fetched: 抓完全文、开始生成前的回调（交互式拿它先打印抓取明细）。
    """
    s = session.settings if session is not None else (settings or get_settings())
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")

    # ① 候选
    candidates, route = await _candidates(question, s, limit, papers, session, source)
    picked = pick_candidates(candidates, papers, s)
    if not picked:
        return QuickResult(
            question=question,
            answer=Answer(question=question, text="", citation_ids=[]),
            route=route,
            message="没有检索到候选（先 /channels add 启用渠道，或换个关键词）",
        )

    # ② 抓全文（内存）→ 切分 → **临时**索引（persist=False：不写 data/index/）
    embeddings = session.index.embeddings if session is not None else get_embeddings(s)
    index = PaperIndex(embeddings, s, persist=False)
    items: list[QuickItem] = []
    for paper in picked:
        collected = await collect_documents(paper, s, save_pdf=False)   # 不落盘
        n = 0
        message = collected.message
        if collected.documents:
            try:
                n = index.add_documents(collected.documents, paper=collected.paper, kind=collected.kind)
            except Exception as exc:  # noqa: BLE001 - 单篇向量化失败不影响其他篇
                logger.warning("quick 向量化失败（%s）：%s", collected.paper.paper_id, exc)
                message = f"{message}；向量化失败 {type(exc).__name__}"
        items.append(QuickItem(collected.paper, collected.kind, collected.pages, n, message))
        logger.info(
            "quick 抓取 %s [%s] %d chunks（%d 页）— %s",
            collected.paper.paper_id, collected.kind, n, collected.pages, message,
        )

    if index.chunk_count == 0:
        if on_fetched is not None:
            on_fetched(items)
        return QuickResult(
            question=question,
            answer=Answer(question=question, text="", citation_ids=[]),
            items=items,
            route=route,
            message="抓到的候选都没有可用内容（无 PDF 直链，网页/摘要也不可用）",
        )

    if on_fetched is not None:
        on_fetched(items)

    # ③ 问答：复用 `ask()` 的检索 / 引用校验 / 流式 / 离线路径，把索引用临时会话顶掉
    quick_session = Session(
        settings=s, index=index, model=session.model if session is not None else None
    )
    result = await ask(question, session=quick_session, k=k, stream_callback=stream_callback)
    return QuickResult(
        question=question,
        answer=result.answer,
        citations=result.citations,
        problems=result.problems,
        items=items,
        route=route,
        context=result.context,
    )
