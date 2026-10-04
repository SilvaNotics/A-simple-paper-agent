# -*- coding: utf-8 -*-
"""可复用的业务流水线：CLI（`cli.py`）与交互式入口（`main.py`）都调用这里。

把「检索 / 入库 / 问答 / 报告」四件事从命令行解析中剥离出来，好处：
- 交互式 REPL 与脚本式 CLI 行为完全一致，不会出现两套逻辑；
- 单元测试可以直接调用这些函数，不必经过 typer。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from langchain_core.tools import BaseTool

from .agents.search_agent import direct_search
from .config import Settings, get_settings
from .llm import get_chat_model, get_embeddings
from .rag.retriever import (
    CitationCollector,
    anchors_in,
    format_context,
    normalize_answer_citations,
    retrieve_across_papers,
    verify_answer,
)
from .rag.store import IndexSignatureError, PaperIndex
from .report import build_bibtex, write_outputs
from .schema import Answer, Paper
from .utils import dedupe_papers, normalize_paper_id, truncate

logger = logging.getLogger(__name__)

# MCP 工具按 (配置文件, server 集合) 缓存：MultiServerMCPClient 是无状态的，
# 每次工具调用会新建 session，因此工具对象可以跨多次命令复用。
_TOOL_CACHE: dict[str, list[BaseTool]] = {}


# --------------------------------------------------------------------------
# 会话组件
# --------------------------------------------------------------------------


@dataclass
class Session:
    """一次交互会话里可复用的重组件（索引 / 模型 / MCP 工具）。"""

    settings: Settings
    index: PaperIndex
    model: Any = None
    search_tools: list[BaseTool] = field(default_factory=list)
    tools_loaded: bool = False

    @property
    def offline(self) -> bool:
        return bool(self.settings.fake_llm)

    def stats(self) -> dict[str, Any]:
        return {
            "papers": len(self.index.paper_ids()),
            "chunks": self.index.chunk_count,
            "tools": len(self.search_tools),
            "tools_loaded": self.tools_loaded,
            "model": getattr(self.model, "model_name", None) or getattr(self.model, "model", None),
            "offline": self.offline,
            "data_dir": str(self.settings.data_path),
        }


def build_session(settings: Settings | None = None) -> Session:
    """构造会话（不加载 MCP 工具，首次检索时懒加载）。

    若当前 embedding 与已有索引的签名不一致（例如 `/connect` 换了供应商），
    自动改用 `data/by-embedding/<签名>/` 作为该 embedding 的独立索引目录，
    避免不同维度互相污染，也避免用户还要手动清理索引。
    """
    from .llm import ConfigError

    s = settings or get_settings()
    s.ensure_dirs()
    try:
        embeddings = get_embeddings(s)
    except ConfigError as exc:
        raise RuntimeError(str(exc)) from exc

    try:
        index = PaperIndex.load_or_create(embeddings, s)
    except IndexSignatureError as exc:
        from .rag.store import embedding_signature
        from .utils import slugify

        sig = slugify(embedding_signature(embeddings, s).replace(":", "-").replace("/", "-"), 40)
        s = s.model_copy(update={"data_dir": Path(s.data_dir) / "by-embedding" / sig})
        s.ensure_dirs()
        logger.info("embedding 签名变化，改用独立索引目录 %s（%s）", s.data_path, exc)
        index = PaperIndex.load_or_create(embeddings, s)

    model = None if s.fake_llm else get_chat_model("default", s)
    return Session(settings=s, index=index, model=model)


def _session_settings(session: "Session | None", settings: Settings | None) -> Settings:
    """session 优先：build_session 可能把索引目录重定向到 data/by-embedding/<sig>/。"""
    return session.settings if session is not None else (settings or get_settings())


def _session_index(session: "Session | None", settings: Settings) -> PaperIndex:
    return session.index if session else PaperIndex.load_or_create(get_embeddings(settings), settings)


async def ensure_tools(session: Session) -> list[BaseTool]:
    """懒加载并缓存 MCP 检索工具。"""
    if session.tools_loaded:
        return session.search_tools
    key = f"{session.settings.servers_file}|{session.settings.fake_llm}"
    if key not in _TOOL_CACHE:
        from .mcp_client import load_mcp_tools

        try:
            _TOOL_CACHE[key] = await load_mcp_tools(session.settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP 工具加载失败：%s", exc)
            _TOOL_CACHE[key] = []
    session.search_tools = _TOOL_CACHE[key]
    session.tools_loaded = True
    return session.search_tools


# --------------------------------------------------------------------------
# 检索 / 入库
# --------------------------------------------------------------------------


async def run_search(
    query: str,
    settings: Settings | None = None,
    limit: int = 8,
    sources: str = "",
    session: Session | None = None,
    source: str = "",
    use_llm: bool | None = None,
    on_event: Any | None = None,
    per_source_limit: bool = False,
    by_channel: dict[str, list[Paper]] | None = None,
) -> tuple[list[Paper], str]:
    """联网检索论文。

    `source`：`auto`（MCP → 内置回退，默认）/ `mcp` / `builtin`（内置免 key 源 + 已启用渠道）
    / `all`（MCP 与内置合并）。

    `limit`：每个渠道（每个检索式）的上限。`per_source_limit=True`（`/search --limit`）时不在
    汇总前截断，总上限为 `max_total_results`；否则汇总后截断到 `limit`。

    `use_llm`：是否用 LLM 做查询扩展 + 重排；`None` 跟随 `search_use_llm`，离线/无模型时跳过。
    """
    s = _session_settings(session, settings)
    try:
        return await asyncio.wait_for(
            _run_search_impl(
                query, s, limit, sources, session, source, use_llm, on_event, per_source_limit, by_channel
            ),
            timeout=max(1.0, s.search_timeout),
        )
    except asyncio.TimeoutError:
        logger.warning("检索总耗时超过 %.0fs，返回已获得的部分结果", s.search_timeout)
        return [], f"检索超时（>{s.search_timeout:.0f}s）"


async def _run_search_impl(
    query: str,
    s: Settings,
    limit: int,
    sources: str,
    session: Session | None,
    source: str,
    use_llm: bool | None,
    on_event: Any | None = None,
    per_source_limit: bool = False,
    by_channel: dict[str, list[Paper]] | None = None,
) -> tuple[list[Paper], str]:
    if sources:
        s = s.model_copy(update={"search_sources": sources})
    mode = (source or s.search_source or "auto").lower()
    # ① 检索过程中使用 LLM：查询扩展 → 多检索式并发 → 相关性重排
    model = _search_model(s, session, use_llm)
    queries = [query]
    if model is not None:
        from .search_llm import expand_queries

        queries = await expand_queries(model, query, n=3, settings=s)
        logger.info("LLM 查询扩展：%s → %s", truncate(query, 40), queries)

    merged: list[Paper] = []
    routes: list[str] = []
    # `--limit` 是每渠道上限：per_source 时不在汇总前截断（总上限 max_total_results）；
    # 否则（ingest/report）LLM 开启时每渠道多取一些，最后统一截断到 limit。
    if per_source_limit:
        per_query = max(1, limit)
        cap = max(1, s.max_total_results)
    else:
        per_query = max(limit, min(limit * 3, 20)) if model is not None else limit
        cap = max(1, limit)
    outcomes = await asyncio.gather(
        *(
            _search_once(
                q, s, mode, per_query, sources, session, on_event, not per_source_limit, by_channel
            )
            for q in queries
        ),
        return_exceptions=True,
    )
    for q, outcome in zip(queries, outcomes):
        if isinstance(outcome, BaseException):
            logger.warning("检索式 %r 失败：%s", q, outcome)
            routes.append(f"{truncate(q, 20)}:{type(outcome).__name__}")
            continue
        papers, route = outcome
        merged.extend(papers)
        routes.append(route)
    merged = dedupe_papers(merged)

    # `/search --limit N`：N 是**每个渠道的总上限**，不随 LLM 检索式数量放大。
    # （builtin_search 已按 kind 把结果写入 by_channel；这里对每个渠道再截断一次。）
    if per_source_limit and by_channel is not None:
        for kind in list(by_channel):
            by_channel[kind] = dedupe_papers(by_channel[kind])[:limit]
        capped = [p for items in by_channel.values() for p in items]
        if capped:
            merged = dedupe_papers(capped)
            cap = len(merged)  # 已完成按渠道截断，不再做全局截断

    if model is not None and len(merged) > cap:
        from .search_llm import rank_papers

        merged = await rank_papers(model, query, merged, cap, settings=s)

    # 国内渠道优先：稳定排序把国内库结果提到前面，截断到 cap 时优先保留
    if getattr(s, "prefer_domestic", True):
        from .sources import papers_domestic_first

        merged = papers_domestic_first(merged)

    route = " ⟂ ".join(dict.fromkeys(r for r in routes if r))
    if model is not None and len(queries) > 1:
        route = f"LLM扩展×{len(queries)} | {route}"
    return merged[:cap], route


def _search_model(s: Settings, session: Session | None, use_llm: bool | None) -> Any:
    """决定检索时是否用 LLM；不可用则返回 None（静默降级）。"""
    enabled = s.search_use_llm if use_llm is None else use_llm
    if not enabled or s.fake_llm:
        return None
    try:
        model = (session.model if session else None) or get_chat_model("search", s)
    except Exception as exc:  # noqa: BLE001 - 没配模型时检索仍可用
        logger.info("检索用 LLM 不可用（%s），跳过查询扩展/重排", exc)
        return None
    if s.fake_llm:  # session.model 可能是假模型
        return None
    return model


async def _search_once(
    query: str,
    s: Settings,
    mode: str,
    limit: int,
    sources: str,
    session: Session | None,
    on_event: Any | None = None,
    truncate: bool = True,
    by_channel: dict[str, list[Paper]] | None = None,
) -> tuple[list[Paper], str]:
    """单条检索式的检索层：builtin / MCP / auto 回退。

    `truncate=False`：不在这里按 `limit` 截断（`/search --limit` = 每渠道上限时用，
    汇总总上限由上层 `_run_search_impl` 的 `max_total_results` 负责）。
    """

    def _trim(papers: list[Paper]) -> list[Paper]:
        papers = dedupe_papers(papers)
        return papers[:limit] if truncate else papers

    def _builtin():
        from .sources import builtin_search

        # 只透传“有值”的参数：显式 `--sources`、以及启用了逐渠道进度时的 on_event。
        # （不传 None，兼容只接受 (query, limit, settings) 的旧调用/测试替身。）
        kwargs: dict[str, Any] = {}
        if sources:
            kwargs["sources"] = sources
        if on_event is not None:
            kwargs["on_event"] = on_event
        if by_channel is not None:
            kwargs["by_channel"] = by_channel
        return builtin_search(query, limit, s, **kwargs)

    if mode == "builtin":
        papers, route = await _builtin()
        return _trim(papers), route

    # 未启用任何渠道：不做 MCP 检索（避免 MCP 工具默认拉全源），让内置层给出引导提示
    if not s.mcp_sources:
        if mode == "mcp":
            return [], "未启用任何渠道：用 /channels add <编号|kind> 添加"
        papers, route = await _builtin()
        return _trim(papers), route

    if mode == "all":
        # 同时用 MCP 与内置源/已启用渠道，确保配置的渠道一定参与检索
        mcp_papers, mcp_route = await _search_once(
            query, s, "mcp", limit, sources, session, on_event, truncate, by_channel
        )
        bi_papers, bi_route = await _builtin()
        return _trim([*mcp_papers, *bi_papers]), f"{mcp_route} + {bi_route}"

    tools = await (ensure_tools(session) if session else _load_tools_once(s))
    if not tools:
        if mode == "mcp":
            return [], "没有可用的 MCP 检索工具"
        papers, route = await _builtin()
        return _trim(papers), f"{route}（MCP 不可用，已回退）"

    papers, route = await direct_search(tools, query, max(limit, 3), sources=s.mcp_sources)
    papers = dedupe_papers(papers)
    if by_channel is not None and papers:
        by_channel.setdefault("mcp", []).extend(papers)
    if papers or mode == "mcp":
        return _trim(papers), route

    fb_papers, fb_route = await _builtin()
    if fb_papers:
        return _trim([*papers, *fb_papers]), f"{route} → {fb_route}（MCP 无结果，已回退）"
    return _trim(papers), route


async def _load_tools_once(settings: Settings) -> list[BaseTool]:
    key = f"{settings.servers_file}|{settings.fake_llm}"
    if key not in _TOOL_CACHE:
        from .mcp_client import load_mcp_tools

        try:
            _TOOL_CACHE[key] = await load_mcp_tools(settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP 工具加载失败：%s", exc)
            _TOOL_CACHE[key] = []
    return _TOOL_CACHE[key]


async def run_ingest(
    query: str,
    settings: Settings | None = None,
    limit: int = 5,
    ids: str | Iterable[str] = "",
    force: bool = False,
    session: Session | None = None,
) -> list[dict[str, Any]]:
    """检索 → 下载 → 解析 → 切分 → 入库（不调用 LLM）。"""
    from .tools.paper_tools import ingest_paper

    s = _session_settings(session, settings)
    index = _session_index(session, s)

    wanted = [i.strip() for i in (ids.split(",") if isinstance(ids, str) else ids) if i.strip()]

    # ① 给了 ID：直接用内置解析器按 ID 抓元数据（不需要先搜到、也不依赖 MCP）
    papers: list[Paper] = []
    route = ""
    if wanted:
        from .sources import resolve_ids

        resolved, failures = await resolve_ids(wanted, s, limit=limit)
        papers.extend(resolved)
        route = f"按 ID 解析 {len(resolved)}/{len(wanted)}"
        if failures:
            logger.warning("以下标识解析失败：%s", ", ".join(failures))

    # ② 有查询词：再走常规联网检索补候选（MCP → 内置回退）
    if query and query.strip():
        found, search_route = await run_search(query, s, max(limit, 3), session=session)
        papers.extend(found)
        route = f"{route} + {search_route}" if route else search_route

    papers = dedupe_papers(papers)
    if wanted:
        # 指定 ID 时只入库这些 ID（避免把搜索捎带的结果也塞进去）
        wanted_ids = {normalize_paper_id(w) for w in wanted}
        selected = [p for p in papers if p.paper_id in wanted_ids][:limit]
        if not selected:
            logger.warning("指定的 ID 一个都没解析成功：%s", ", ".join(wanted))
    else:
        selected = papers[:limit]

    return await ingest_papers(selected, session=session, settings=s, force=force)


async def ingest_papers(
    papers: Iterable[Paper],
    settings: Settings | None = None,
    session: Session | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """把已经拿到的 `Paper` 列表直接下载/解析/入库。

    供 `run_ingest` 与「`/search ... --ingest`」复用：前者不再需要二次检索，
    后者能把刚搜到的候选（含非 arXiv/DOI 来源）直接落库。
    """
    from .tools.paper_tools import ingest_paper

    s = _session_settings(session, settings)
    index = _session_index(session, s)

    results: list[dict[str, Any]] = []
    for paper in dedupe_papers(list(papers)):
        if force and index.has_paper(paper.paper_id):
            index.clear_paper(paper.paper_id)
        results.append(await ingest_paper(paper, index, s))
    return results


def remove_papers(
    ids: Iterable[str],
    settings: Settings | None = None,
    session: Session | None = None,
    remove_pdf: bool = True,
) -> list[str]:
    """从 RAG 索引（并可选地从磁盘）删除论文；返回真正删掉的 paper_id。"""
    s = _session_settings(session, settings)
    index = _session_index(session, s)
    removed: list[str] = []
    for raw in ids:
        paper_id = normalize_paper_id(raw) or str(raw).strip()
        if not paper_id:
            continue
        if index.delete_paper(paper_id, remove_pdf=remove_pdf):
            removed.append(paper_id)
    return removed


# --------------------------------------------------------------------------
# 问答
# --------------------------------------------------------------------------


def _rag_task(question: str, context: str) -> str:
    available = anchors_in(context)
    return (
        f"问题：{question}\n\n可用上下文：\n{context}\n\n"
        f"可用的引用锚点（必须原样复制，不要改写）：{', '.join(available) if available else '（无）'}\n"
        "请基于上下文回答，并在每个结论句后标注锚点。"
    )


@dataclass
class AskResult:
    answer: Answer
    citations: dict
    problems: list[str]
    context: str = ""

    @property
    def ok(self) -> bool:
        return not self.problems


async def ask(
    question: str,
    settings: Settings | None = None,
    session: Session | None = None,
    paper_ids: list[str] | None = None,
    k: int = 0,
    stream_callback: Callable[[str], None] | None = None,
) -> AskResult:
    """在已入库语料上带引用问答。

    - 有模型且给了 `stream_callback`：边生成边回调（交互式体验）；
    - 有模型未给回调：走结构化输出的 rag_agent（脚本/测试更稳）；
    - 离线（假模型）：不回退到 LLM，直接返回检索到的原文证据。
    """
    s = _session_settings(session, settings)
    index = _session_index(session, s)
    if index.chunk_count == 0:
        raise RuntimeError("索引为空：请先执行 /ingest <主题>（或 CLI: ingest）")

    collector = CitationCollector(prefix="A-")
    ask_settings = s.model_copy(update={"top_k": k}) if k else s
    docs = retrieve_across_papers(index, question, paper_ids, ask_settings)
    if not docs and paper_ids:
        # 指定论文里没检索到，放宽到全库（并提示调用方）
        docs = retrieve_across_papers(index, question, None, ask_settings)
    context = format_context(docs, collector) if docs else "（没有检索到相关片段）"

    if s.fake_llm:
        from .agents.rag_agent import answer_without_llm

        answer = await answer_without_llm(question, context)
    elif stream_callback is not None:
        # 单次回答的等待上限：180s（超过则用已生成的内容打分，不无限等）
        try:
            answer = await asyncio.wait_for(
                _stream_answer(session, s, question, context, stream_callback),
                timeout=max(1.0, s.llm_timeout),
            )
        except asyncio.TimeoutError:
            logger.warning("流式回答超过 %.0fs，改用已生成内容", s.llm_timeout)
            raise RuntimeError(f"模型响应超过 {s.llm_timeout:.0f}s（可在 .env 调 LLM_TIMEOUT）") from None
    else:
        # 非流式：复用与流式相同的普通对话路径（无工具、无 response_format），
        # 只丢弃 token 回调；不用带工具的 agent（会强制 tool_choice，思考型模型会 400）。
        try:
            answer = await asyncio.wait_for(
                _stream_answer(session, s, question, context, lambda _token: None),
                timeout=max(1.0, s.llm_timeout),
            )
        except asyncio.TimeoutError:
            logger.warning("非流式回答超过 %.0fs", s.llm_timeout)
            raise RuntimeError(
                f"模型响应超过 {s.llm_timeout:.0f}s（可在 .env 调 LLM_TIMEOUT）"
            ) from None

    answer = normalize_answer_citations(answer, collector.citations)
    answer.text = collapse_repetition(answer.text)
    problems = verify_answer(
        answer,
        collector.citations,
        chunk_lookup=collector.chunks,
        min_ratio=s.min_support_ratio,
    )
    answer.unsupported = problems
    return AskResult(answer=answer, citations=collector.citations, problems=problems, context=context)


_JSON_TAIL_RE = re.compile(r'\n*\{[^{}]*"citation_ids"[^{}]*\}\s*$', re.DOTALL)
_DUP_SENT_RE = re.compile(r"(?<=[。．.!?！？])")  # 只按句末标点切，不动段落换行


def _period_repetition_start(
    text: str, min_block: int = 16, max_block: int = 800, min_copies: int = 3
) -> int | None:
    """相邻复读：末尾有同一个块连续出现 ≥ min_copies 次时，返回应截断到的下标。

    这是最常见的复读形式（同一句/段连续输出多次），对小块（≥ 16 字）也很敏感。
    """
    n = len(text)
    if n < min_block * min_copies:
        return None
    upper = min(max_block, n // 2)  # 允许 p 到 n/2（长段只重复 2 次也能检出）
    for p in range(min_block, upper + 1):
        if text[-p:] != text[-2 * p : -p]:
            continue
        count = 2
        start = n - 2 * p
        while start - p >= 0 and text[start - p : start] == text[start : start + p]:
            count += 1
            start -= p
        # 长块（≥ 40 字）出现 2 次就足够可疑；短块要求 3 次，避免误伤
        if count >= (2 if p >= 40 else min_copies):
            return start + p  # 保留第一份副本，丢掉后面的复读
    return None


def _rolling_repetition_start(text: str, block: int = 32, min_hits: int = 3) -> int | None:
    """跨句/带间隔的复读：同一个 `block` 长窗口出现 ≥ `min_hits` 次时截断。"""
    n = len(text)
    if n < block * min_hits:
        return None
    step = max(1, block // 2)
    first: dict[str, int] = {}
    hits: dict[str, int] = {}
    for end in range(block, n + 1, step):
        window = text[end - block : end]
        first.setdefault(window, end - block)
        hits[window] = hits.get(window, 0) + 1
        if hits[window] >= min_hits:
            return first[window] + block
    return None


def truncate_repetition(text: str, max_scan: int = 1500) -> str:
    """检测“同一段内容反复出现”并截断：返回干净的一段（正常文本原样返回）。

    只在末尾 `max_scan` 个字符里找（复读总是发生在生成末尾），成本可控。
    """
    if not text:
        return text
    tail = text[-max_scan:] if len(text) > max_scan else text
    base = len(text) - len(tail)
    starts = [s for s in (_period_repetition_start(tail), _rolling_repetition_start(tail)) if s is not None]
    if not starts:
        return text
    return text[: base + min(starts)].rstrip()


def dedupe_repeated_blocks(text: str, max_block: int = 6, min_chars: int = 20) -> str:
    """删除“重复出现过的段落块”（不要求相邻）。

    典型退化：模型把“小节标题+正文”整块重复若干遍、中间再接着写别的，形如
    `[A,B,A,B,A,B,C,D…]`。只去重“相邻完全相同”抳不住；这里把段落切成序列，
    从每个位置找出“之前在输出里出现过的**最长**连续段落块”并跳过。
    只处理总长 ≥ `min_chars` 的块，避免误删像“参考文献”这种合法的短标题重复。
    """
    paras = re.split(r"\n{2,}", text or "")
    n = len(paras)
    if n < 3:
        return text
    out: list[str] = []
    i = 0
    while i < n:
        matched = 0
        for length in range(min(max_block, n - i), 0, -1):
            block = paras[i : i + length]
            if sum(len(p) for p in block) < min_chars:
                continue
            if any(out[j : j + length] == block for j in range(max(0, len(out) - length + 1))):
                matched = length
                break
        if matched:
            i += matched
            continue
        out.append(paras[i])
        i += 1
    return "\n\n".join(out)


def collapse_repetition(text: str) -> str:
    """折叠模型退化式重复：相邻重复 + 任意位置的整块复读 + 末尾连续复读，三重处理。"""
    if not text:
        return text
    # 1) 段落级：连续相同段落去重
    paras = re.split(r"\n{2,}", text)
    deduped: list[str] = []
    prev = None
    for para in paras:
        key = para.strip()
        if key and key == prev:
            continue
        deduped.append(para)
        prev = key
    text = "\n\n".join(deduped)
    # 2) 句子级：连续重复句子去重
    out: list[str] = []
    prev = None
    for sent in _DUP_SENT_RE.split(text):
        key = sent.strip()
        if key and key == prev:
            continue
        out.append(sent)
        prev = key
    text = "".join(out)
    # 3) 任意位置的整块复读（[A,B,A,B,...]）
    text = dedupe_repeated_blocks(text)
    # 4) 末尾连续复读：截断到第一份副本
    return truncate_repetition(text)


class RepetitionGuard:
    """流式生成时的实时复读检测器（检测到复读就截断，并可据此停止生成）。

    每累积 ≥`check_every` 个字符做一次滚动窗口检测；成本可控。
    """

    def __init__(self, check_every: int = 24) -> None:
        self.text = ""
        self._check_every = check_every
        self._next = check_every
        self._block_check = 200

    def push(self, delta: str) -> bool:
        """追加增量；返回 True 表示检测到复读（`self.text` 已被截断到重复处）。"""
        self.text += delta
        if len(self.text) < self._next:
            return False
        self._next = len(self.text) + self._check_every
        cleaned = truncate_repetition(self.text)
        if len(cleaned) < len(self.text):
            self.text = cleaned
            return True
        # 块复读（[A,B,A,B]）成本略高：每 ~200 字才查一次
        if len(self.text) >= self._block_check:
            self._block_check = len(self.text) + 200
            collapsed = dedupe_repeated_blocks(self.text)
            if len(collapsed) < len(self.text):
                self.text = collapsed
                return True
        return False


def _chunk_text(chunk: Any) -> str:
    """从流式 chunk 里取文本（兼容 str / 内容块列表）。"""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "".join(parts)
    return str(content) if content else ""


def clean_stream_output(text: str) -> str:
    """去掉流式回答末尾可能泄漏的结构化 JSON 尾巴（有些模型会照抄字段名）。"""
    cleaned = _JSON_TAIL_RE.sub("", text or "").strip()
    if '"citation_ids"' in cleaned:
        # 更复杂的情况：截断到 JSON 开始处
        idx = cleaned.find('{"')
        if idx > 0 and '"citation_ids"' in cleaned[idx:]:
            cleaned = cleaned[:idx].strip()
    return cleaned


async def _stream_answer(
    session: Session | None,
    settings: Settings,
    question: str,
    context: str,
    on_token: Callable[[str], None],
) -> Answer:
    """流式生成回答（不强制结构化输出，改为生成后按锚点解析）。"""
    from .agents import prompts

    model = (session.model if session else None) or get_chat_model("synthesize", settings)
    system = (
        prompts.RAG_PROMPT_ZH
        + "\n\n【输出格式】只输出答案正文（Markdown），不要输出 JSON、不要出现 citation_ids 等字段名。"
    )
    messages = [
        ("system", system),
        ("human", _rag_task(question, context)),
    ]
    guard = RepetitionGuard()
    stream = model.astream(messages)
    try:
        async for chunk in stream:
            piece = _chunk_text(chunk)
            if not piece:
                continue
            # 兼容两种流：增量式（delta）与**累计全文式**（每次发完整文本）。
            # 累计式如果直接 append，就会变成“反复输出”，所以这里只取增量。
            if guard.text and piece.startswith(guard.text):
                delta = piece[len(guard.text) :]
            else:
                delta = piece
            if not delta:
                continue
            on_token(delta)
            if guard.push(delta):
                # 模型陷入复读：立刻停止消费，保留已生成的正常部分
                logger.warning("检测到重复生成（复读），已截断并停止继续生成")
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("流式生成失败：%s", exc)
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001
                pass

    full = guard.text
    text = clean_stream_output(collapse_repetition(full))
    if not text:
        return Answer(question=question, text="资料不足，未能形成可靠结论。", citation_ids=[])
    return Answer(question=question, text=text, citation_ids=anchors_in(text))


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------


@dataclass
class ReportResult:
    topic: str
    markdown: str
    bibtex: str
    paths: dict[str, Path]
    papers: list[Paper]
    state: dict[str, Any]
    flagged: list[str]
    indexed: list[dict[str, Any]]


async def run_report(
    topic: str,
    settings: Settings | None = None,
    session: Session | None = None,
    papers: int = 0,
    simple: bool = False,
    search_tools: list[BaseTool] | None = None,
) -> ReportResult:
    """端到端调研报告（监督图或 simple 单 agent），并落盘 md/bib/json。"""
    s = _session_settings(session, settings)
    if papers:
        s = s.model_copy(update={"max_papers": papers})
    s.ensure_dirs()

    if search_tools is None:
        if (s.search_source or "auto").lower() == "builtin":
            # 明确要求内置源时不必加载 MCP（没有装 MCP server 的机器也能跑通）
            search_tools = []
        else:
            search_tools = await (ensure_tools(session) if session else _load_tools_once(s))

    collector = None
    if simple:
        from .agents.supervisor import build_simple_app

        graph, index, collector = await build_simple_app(s, search_tools=search_tools or None)
    else:
        from .agents.supervisor import build_app

        graph, deps = await build_app(s, search_tools=search_tools)
        index = deps.index

    state = await graph.ainvoke(
        {"query": topic, "notes": [], "branch": ""},
        {
            "configurable": {"thread_id": f"report-{abs(hash(topic)) % 10**8}"},
            "max_concurrency": s.concurrency,
            "recursion_limit": 120,
        },
    )

    markdown = state.get("report_md", "")
    if not markdown.strip():
        raise RuntimeError("报告生成为空（可能是检索/入库全部失败）")

    selected: list[Paper] = list(state.get("selected", []))
    indexed = index.list_papers()
    if not selected:
        selected = [
            Paper(paper_id=item["paper_id"], title=item.get("title", "")) for item in indexed
        ]
    if not markdown.lstrip().startswith("#"):
        markdown = f"# 学术调研报告：{topic}\n\n{markdown.strip()}\n"

    citations = state.get("citations") or (collector.citations if collector else {})
    bibtex = state.get("bibtex") or build_bibtex(selected)
    payload = {
        "topic": topic,
        "mode": "simple" if simple else "graph",
        "search_queries": state.get("search_queries", []),
        "sub_questions": state.get("sub_questions", []),
        "papers": [p.model_dump() for p in selected],
        "summaries": [x.model_dump() for x in state.get("summaries", [])],
        "answers": [a.model_dump() for a in (state.get("answers") or {}).values()],
        "citations": {k: v.model_dump() for k, v in citations.items()},
        "verify_issues": state.get("verify_issues", []),
        "ingest_results": state.get("ingest_results", []),
        "notes": state.get("notes", []),
        "indexed_papers": indexed,
    }
    paths = write_outputs(topic, markdown, bibtex, payload, s)
    return ReportResult(
        topic=topic,
        markdown=markdown,
        bibtex=bibtex,
        paths=paths,
        papers=selected,
        state=state,
        flagged=list(state.get("verify_issues", [])),
        indexed=indexed,
    )
