# -*- coding: utf-8 -*-
"""多 agent 监督图（LangGraph StateGraph）。

    plan → search_one×N → merge → ingest_one×M → summarize_one×M
         → answer_one×K → write → verify ─┬→ END
                                          └→ answer_one（重试 ≤1 次）

（每级扇出后都经过一个 collect_* barrier 节点再触发下一级。）

- 检索/入库/精读/问答按条目 `Send` 扇出，并行度由 `config={"max_concurrency": N}` 控制；
- 引用编号按分支加前缀（`S1-C1` / `Q2-C3`），合并到图状态时不撞号；
- 每个 LLM 节点都有确定性兜底，模型抽风或离线时仍能产出结果。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import prompts
from .rag_agent import make_rag_agent, run_rag_agent
from .search_agent import direct_search, make_search_agent, run_search_agent
from .summarize_agent import make_summarize_agent, run_summarize_agent
from .writer_agent import make_writer_agent, material_from_state, run_writer_agent
from ..core.config import Settings, get_settings
from ..llm.factory import get_chat_model, get_embeddings
from ..rag.retriever import (
    CitationCollector,
    retrieve_across_papers,
    format_context,
    retrieve,
    verify_answer,
    verify_report_citations,
)
from ..rag.store import PaperIndex
from ..pipeline.report import build_bibtex, render_report
from ..core.schema import (
    Paper,
    ResearchState,
    SelectionOutput,
)
from ..tools.paper_tools import ingest_paper, make_paper_tools
from ..tools.rag_tools import make_rag_tools
from ..core.utils import dedupe_papers, extract_json, first_list, truncate

logger = logging.getLogger(__name__)

MAX_PLAN_QUESTIONS = 4
MAX_SEARCH_QUERIES = 5
MAX_RETRIES = 1


# --------------------------------------------------------------------------
# 依赖容器
# --------------------------------------------------------------------------


@dataclass
class Deps:
    """图运行所需的全部依赖（便于测试注入假实现）。"""

    settings: Settings
    index: PaperIndex
    search_tools: list[BaseTool] = field(default_factory=list)
    rag_tools_factory: Callable[[CitationCollector], list[BaseTool]] | None = None
    search_agent: Any = None
    summarize_agent: Any = None
    writer_agent: Any = None
    planner: Any = None          # 用于 plan / select 的普通模型
    offline: bool = False        # True 时不调用 LLM（纯确定性）


# --------------------------------------------------------------------------
# 节点：plan
# --------------------------------------------------------------------------


def _fallback_queries(topic: str) -> tuple[list[str], list[str]]:
    """没有模型时的计划：直接用主题生成几条检索式。"""
    base = topic.strip()
    en_hint = base if base.isascii() else base
    return (
        [f"{base} 的核心方法与代表性工作", f"{base} 的评测与局限"],
        [en_hint, f"{en_hint} survey", f"{en_hint} benchmark evaluation"],
    )


def make_plan_node(deps: Deps):
    async def plan(state: ResearchState) -> dict[str, Any]:
        topic = state.get("query", "")
        sub_questions: list[str] = []
        queries: list[str] = []
        if not deps.offline and deps.planner is not None:
            try:
                prompt = f"{prompts.PLAN_PROMPT_ZH}\n研究主题：{topic}"
                raw = await deps.planner.ainvoke(prompt)
                payload = extract_json(getattr(raw, "content", str(raw)))
                if isinstance(payload, dict):
                    sub_questions = [str(x) for x in first_list(payload, "sub_questions")][:MAX_PLAN_QUESTIONS]
                    queries = [str(x) for x in first_list(payload, "search_queries")][:MAX_SEARCH_QUERIES]
            except Exception as exc:  # noqa: BLE001
                logger.warning("plan 节点模型调用失败，使用模板计划：%s", exc)

        if not sub_questions or not queries:
            fb_sub, fb_q = _fallback_queries(topic)
            sub_questions = sub_questions or fb_sub
            queries = queries or fb_q

        return {
            "sub_questions": sub_questions,
            "search_queries": queries,
            "notes": [f"计划：{len(sub_questions)} 个子问题 / {len(queries)} 条检索式"],
        }

    return plan


# --------------------------------------------------------------------------
# 节点：search（扇出 + 单条 worker + 合并筛选）
# --------------------------------------------------------------------------


def fan_out_search(state: ResearchState) -> list[Send]:
    queries = state.get("search_queries") or [state.get("query", "")]
    queries = [q for q in queries if q]
    if not queries:
        return [Send("no_papers", {"notes": ["没有检索式"]})]
    return [
        Send("search_one", {"current_question": q, "branch": f"q{i}:"})
        for i, q in enumerate(queries[:MAX_SEARCH_QUERIES])
    ]


def make_search_one_node(deps: Deps):
    """检索节点：MCP 优先 → 直接调 MCP 工具 → 内置 HTTP 源（arXiv/OpenAlex/Crossref）。

    `settings.search_source`：
      - `builtin`：跳过 MCP，直接用内置源（无需装 MCP server）；
      - `mcp`    ：只用 MCP；
      - `auto`   ：默认，逐级回退。
    """
    limit = max(3, deps.settings.max_papers)

    async def _builtin(query: str) -> tuple[list[Paper], str]:
        from ..sources.fetchers import builtin_search  # noqa: PLC0415  (延迟导入，避免循环依赖)

        return await builtin_search(query, limit, deps.settings)

    async def search_one(state: ResearchState) -> dict[str, Any]:
        query = state.get("current_question", "")
        mode = (deps.settings.search_source or "auto").lower()
        papers: list[Paper] = []
        route = "offline"

        if mode == "builtin":
            papers, route = await _builtin(query)
            return {
                "candidates": papers,
                "notes": [f"检索「{truncate(query, 40)}」→ {len(papers)} 条（{route}）"],
            }

        if deps.search_tools and (deps.offline or deps.settings.mcp_sources):
            if deps.offline:
                # 离线自检：只用本地/注入的工具，绝不发起网络请求
                papers, route = await direct_search(
                    deps.search_tools, query, limit, sources=deps.settings.mcp_sources
                )
            else:
                papers, route = await run_search_agent(
                    deps.search_agent,
                    query,
                    deps.search_tools,
                    limit=max(3, deps.settings.max_papers // 2),
                )
                if not papers:
                    papers, route = await direct_search(
                        deps.search_tools, query, limit, sources=deps.settings.mcp_sources
                    )
                    route = f"fallback-{route}"
        elif deps.offline:
            route = "offline（未启用联网检索）"
        else:
            route = "未启用任何渠道（用 /channels add <编号|kind> 添加）"

        # MCP 不可用 / 无结果 → 内置源兜底（auto 模式；离线模式下不走网络）
        if not papers and mode == "auto" and not deps.offline:
            fb_papers, fb_route = await _builtin(query)
            route = fb_route if not route or route == "offline" else f"{route} → {fb_route}"
            papers = fb_papers

        return {
            "candidates": papers,
            "notes": [f"检索「{truncate(query, 40)}」→ {len(papers)} 条（{route}）"],
        }

    return search_one


def _heuristic_select(papers: list[Paper], topic: str, limit: int) -> list[Paper]:
    """无模型时的排序：词面重合 + 时效 + 开放获取。"""
    import re

    words = {w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", topic.lower())}

    def score(p: Paper) -> float:
        text = f"{p.title} {p.abstract}".lower()
        overlap = sum(1 for w in words if w in text) / max(1, len(words))
        year: float = 0.0
        if p.published[:4].isdigit():
            year = max(0.0, min(1.0, (int(p.published[:4]) - 2015) / 11))
        oa = 1.0 if p.is_open_access_hint else 0.0
        return 0.55 * overlap + 0.2 * year + 0.25 * oa

    return sorted(papers, key=score, reverse=True)[:limit]


def make_merge_node(deps: Deps):
    async def merge(state: ResearchState) -> dict[str, Any]:
        topic = state.get("query", "")
        limit = deps.settings.max_papers
        unique = dedupe_papers(state.get("candidates", []))
        if not unique:
            return {"selected": [], "notes": ["去重后没有候选论文"]}

        selected: list[Paper] = []
        if not deps.offline and deps.planner is not None:
            listing = "\n".join(
                f"- paper_id: {p.paper_id}\n  title: {p.title}\n  abstract: {truncate(p.abstract, 300)}"
                for p in unique[:40]
            )
            try:
                scorer = deps.planner.with_structured_output(SelectionOutput)
                out = await scorer.ainvoke(
                    f"{prompts.SELECT_PROMPT_ZH}\n研究主题：{topic}\n候选论文：\n{listing}"
                )
                if isinstance(out, SelectionOutput) and out.selected:
                    ranked = sorted(out.selected, key=lambda s: s.score, reverse=True)
                    keep = {s.paper_id for s in ranked if s.score >= 0.5} or {
                        s.paper_id for s in ranked[:limit]
                    }
                    selected = [p for p in unique if p.paper_id in keep][:limit]
                    if selected:
                        return {
                            "selected": selected,
                            "notes": [f"筛选：候选 {len(unique)} → 选中 {len(selected)}（LLM 打分）"],
                        }
            except Exception as exc:  # noqa: BLE001
                logger.warning("筛选节点模型调用失败，改用启发式排序：%s", exc)

        selected = _heuristic_select(unique, topic, limit)
        return {
            "selected": selected,
            "notes": [f"筛选：候选 {len(unique)} → 选中 {len(selected)}（启发式排序）"],
        }

    return merge


# --------------------------------------------------------------------------
# 节点：ingest（扇出 + worker）
# --------------------------------------------------------------------------


def fan_out_ingest(state: ResearchState) -> list[Send]:
    selected = state.get("selected", [])
    if not selected:
        return [Send("no_papers", {"notes": ["没有可入库的论文"]})]
    return [
        Send("ingest_one", {"current_paper": p, "branch": f"p{i}:"})
        for i, p in enumerate(selected)
    ]


def fan_out_summarize(state: ResearchState) -> list[Send]:
    selected = state.get("selected", [])
    if not selected:
        return [Send("no_papers", {"notes": ["没有可精读的论文"]})]
    return [
        Send("summarize_one", {"current_paper": p, "branch": f"S{i}-"})
        for i, p in enumerate(selected)
    ]


def fan_out_answer(state: ResearchState) -> list[Send]:
    questions = state.get("sub_questions") or [state.get("query", "")]
    if not questions:
        return [Send("no_papers", {"notes": ["没有问题需要回答"]})]
    return [
        Send("answer_one", {"current_question": q, "branch": f"Q{i}-"})
        for i, q in enumerate(questions[:MAX_PLAN_QUESTIONS])
    ]


def make_ingest_one_node(deps: Deps):
    @guard("入库")
    async def ingest_one(state: ResearchState) -> dict[str, Any]:
        paper = state.get("current_paper")
        if not isinstance(paper, Paper):
            return {"ingest_results": [], "notes": ["ingest：缺少论文信息"]}
        info = await ingest_paper(paper, deps.index, deps.settings)
        return {
            "ingest_results": [info],
            "notes": [
                f"入库 {info['paper_id']}：{info['status']}（{info['chunks']} chunks，{info['message']}）"
            ],
        }

    return ingest_one


# --------------------------------------------------------------------------
# 节点：summarize（扇出 + worker）
# --------------------------------------------------------------------------


def _summary_query(state: ResearchState, paper: Paper) -> str:
    parts = [state.get("query", ""), paper.title, *state.get("sub_questions", [])]
    return " ".join(p for p in parts if p)


def make_summarize_one_node(deps: Deps):
    @guard("精读")
    async def summarize_one(state: ResearchState) -> dict[str, Any]:
        paper = state.get("current_paper")
        if not isinstance(paper, Paper):
            return {"summaries": [], "notes": ["summarize：缺少论文信息"]}

        prefix = state.get("branch", "S-")
        collector = CitationCollector(prefix=prefix)
        docs = retrieve(
            deps.index,
            _summary_query(state, paper),
            k=deps.settings.top_k,
            paper_ids=[paper.paper_id],
            hybrid=deps.settings.hybrid_retrieval,
        )
        context = format_context(docs, collector) if docs else ""
        has_fulltext = bool(docs)

        if deps.offline or deps.summarize_agent is None:
            from .summarize_agent import fallback_summary

            summary = fallback_summary(paper)
            summary.retrieved_chunks = len(docs)  # 兜底路径直接记录检索到的片段数
            if has_fulltext and summary.confidence < 0.4:
                summary.confidence = 0.4
        else:
            summary = await run_summarize_agent(
                deps.summarize_agent, paper, context, has_fulltext=has_fulltext
            )
        return {
            "summaries": [summary],
            "citations": collector.citations,
            "notes": [
                f"精读 {paper.paper_id}：{'全文' if has_fulltext else '仅摘要'}"
                f"（{len(docs)} 段，置信度 {summary.confidence:.2f}）"
            ],
        }

    return summarize_one


# --------------------------------------------------------------------------
# 节点：synthesize（扇出 + worker）
# --------------------------------------------------------------------------


def guard(label: str):
    """节点级容错：单个条目失败不要让整张图崩掉，把错误写进 notes/verify_issues。"""

    def decorator(fn):
        async def wrapper(state: ResearchState) -> dict[str, Any]:
            try:
                return await fn(state)
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s 节点异常", label)
                message = f"{label} 失败：{type(exc).__name__}: {truncate(str(exc), 160)}"
                # 注意：worker 会并行写状态，这里只能写带 reducer 的字段
                return {"notes": [message], "worker_errors": [message]}

        return wrapper

    return decorator


def _pass_through(label: str):
    """barrier 节点：等所有并行 worker 完成，再触发下一次扇出。"""

    async def node(state: ResearchState) -> dict[str, Any]:
        return {}

    node.__name__ = f"collect_{label}"
    return node


def _no_papers(state: ResearchState) -> dict[str, Any]:
    return {"notes": ["没有可用论文，直接生成空报告"]}


def make_answer_one_node(deps: Deps):
    @guard("问答")
    async def answer_one(state: ResearchState) -> dict[str, Any]:
        question = state.get("current_question", "")
        prefix = state.get("branch", "Q-")
        collector = CitationCollector(prefix=prefix)
        paper_ids = [p.paper_id for p in state.get("selected", [])]
        docs = retrieve_across_papers(deps.index, question, paper_ids, deps.settings)
        context = format_context(docs, collector) if docs else "（索引为空，无法检索到全文证据）"

        if deps.offline:
            from .rag_agent import answer_without_llm

            answer = await answer_without_llm(question, context)
        else:
            rag_tools = deps.rag_tools_factory(collector) if deps.rag_tools_factory else []
            agent = make_rag_agent(deps.planner, rag_tools)
            # 传 model：思考型模型拒绝工具/tool_choice 时，run_rag_agent 会回退为纯文本作答
            answer = await run_rag_agent(agent, question, context, model=deps.planner)

        from ..rag.retriever import normalize_answer_citations

        answer = normalize_answer_citations(answer, collector.citations)
        answer.unsupported = verify_answer(
            answer,
            collector.citations,
            chunk_lookup=collector.chunks,
            min_ratio=deps.settings.min_support_ratio,
        )
        return {
            "answers": {question: answer},
            "citations": collector.citations,
            "notes": [
                f"问答「{truncate(question, 40)}」：引用 {len(answer.citation_ids)} 个，"
                f"问题 {len(answer.unsupported)} 个"
            ],
        }

    return answer_one


# --------------------------------------------------------------------------
# 节点：write / verify
# --------------------------------------------------------------------------


def make_write_node(deps: Deps):
    async def write(state: ResearchState) -> dict[str, Any]:
        topic = state.get("query", "")
        summaries = state.get("summaries", [])
        answers = list((state.get("answers") or {}).values())
        selected = state.get("selected", [])
        citations = state.get("citations") or {}

        narrative: dict[str, str] = {}
        if deps.offline or deps.writer_agent is None:
            narrative = {
                "comparison": material_from_state(summaries, answers),
            }
        else:
            material = material_from_state(summaries, answers)
            if material.strip():
                out = await run_writer_agent(deps.writer_agent, topic, material)
                narrative = out.model_dump()

        issues = state.get("verify_issues") or []
        markdown = render_report(
            topic=topic,
            papers=selected,
            summaries=summaries,
            answers=answers,
            citations=citations,
            narrative=narrative,
            search_queries=state.get("search_queries", []),
            notes=state.get("notes", []),
            issues=issues,
        )
        return {
            "report_md": markdown,
            "bibtex": build_bibtex(selected),
            "notes": [f"报告已生成（{len(markdown)} 字符）"],
        }

    return write


def make_verify_node():
    async def verify(state: ResearchState) -> dict[str, Any]:
        problems: list[str] = []
        answers = list((state.get("answers") or {}).values())
        citations = state.get("citations") or {}
        summaries = state.get("summaries", [])
        selected = state.get("selected", [])
        markdown = state.get("report_md", "")

        if not summaries:
            problems.append("没有生成任何逐篇摘要（可能全部论文都没能获取全文）")
        if len(summaries) < len(selected):
            missing = {p.paper_id for p in selected} - {s.paper_id for s in summaries}
            if missing:
                problems.append(f"缺少摘要的论文：{', '.join(sorted(missing))}")

        problems.extend(state.get("worker_errors", []))

        for info in state.get("ingest_results", []):
            if info.get("status") in {"indexed", "cached"}:
                continue
            problems.append(
                f"论文 {info.get('paper_id', '?')} 未能入库（{info.get('status')}）："
                f"{truncate(str(info.get('message', '')), 100)}"
            )

        for answer in answers:
            for issue in answer.unsupported:
                problems.append(f"问答「{truncate(answer.question, 30)}」：{issue}")

        problems.extend(verify_report_citations(markdown, citations))

        retries = int(state.get("retries", 0))
        can_retry = bool(problems) and retries < MAX_RETRIES and bool(answers)
        return {
            "verify_issues": problems,
            "retry": can_retry,
            "retries": retries + 1 if can_retry else retries,
            "notes": [f"校验：发现 {len(problems)} 个问题；{'重试' if can_retry else '结束'}"],
        }

    return verify


def route_after_verify(state: ResearchState):
    """verify 之后：需要重试 → 重新扇出问答；否则结束。

    只看 verify 写下的 `retry` 标志；重试必须重新 Send 扇出（直接回到 answer_one
    会共用一份脏状态，出现空问题、无前缀引用等问题）。
    """
    if not state.get("retry"):
        return "end"
    questions = state.get("sub_questions") or [state.get("query", "")]
    questions = [q for q in questions if q][:MAX_PLAN_QUESTIONS]
    if not questions:
        return "end"
    return [
        Send("answer_one", {"current_question": q, "branch": f"R{i}-"})
        for i, q in enumerate(questions)
    ]


# --------------------------------------------------------------------------
# 组装
# --------------------------------------------------------------------------


def build_graph(
    deps: Deps,
    checkpointer: Any | None = None,
):
    """构建并编译监督图。

    注意：Send 扇出之后必须先经过一个 **barrier 节点**（worker 用普通边指向它），
    再由 barrier 的 condition 边发起下一次扇出；否则条件函数会对每个 worker
    实例各求值一次，导致下游任务成倍重复。
    """
    g = StateGraph(ResearchState)

    g.add_node("plan", make_plan_node(deps))
    g.add_node("search_one", make_search_one_node(deps))
    g.add_node("merge", make_merge_node(deps))          # barrier + 筛选
    g.add_node("ingest_one", make_ingest_one_node(deps))
    g.add_node("collect_ingest", _pass_through("入库完成"))
    g.add_node("summarize_one", make_summarize_one_node(deps))
    g.add_node("collect_summarize", _pass_through("精读完成"))
    g.add_node("answer_one", make_answer_one_node(deps))
    g.add_node("collect_answers", _pass_through("问答完成"))
    g.add_node("no_papers", _no_papers)
    g.add_node("write", make_write_node(deps))
    g.add_node("verify", make_verify_node())

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out_search, ["search_one"])
    g.add_edge("search_one", "merge")
    g.add_conditional_edges("merge", fan_out_ingest, ["ingest_one", "no_papers"])
    g.add_edge("ingest_one", "collect_ingest")
    g.add_conditional_edges("collect_ingest", fan_out_summarize, ["summarize_one", "no_papers"])
    g.add_edge("summarize_one", "collect_summarize")
    g.add_conditional_edges("collect_summarize", fan_out_answer, ["answer_one", "no_papers"])
    g.add_edge("answer_one", "collect_answers")
    g.add_edge("collect_answers", "write")
    g.add_edge("no_papers", "write")
    g.add_edge("write", "verify")
    # verify 的 condition 边在需要重试时直接再次扇出到 answer_one
    g.add_conditional_edges(
        "verify", route_after_verify, {"synthesize": "answer_one", "end": END}
    )

    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()

    return g.compile(checkpointer=checkpointer)


def build_deps(
    settings: Settings | None = None,
    search_tools: list[BaseTool] | None = None,
    embeddings: Any | None = None,
    index: PaperIndex | None = None,
    model: Any | None = None,
) -> Deps:
    """组装依赖（同步部分）。MCP 工具可由调用方传入或另行异步加载。"""
    s = settings or get_settings()
    emb = embeddings if embeddings is not None else get_embeddings(s)
    idx = index if index is not None else PaperIndex.load_or_create(emb, s)
    tools = list(search_tools or [])

    offline = s.fake_llm
    planner: Any = None
    if not offline:
        planner = model or get_chat_model("default", s)

    return Deps(
        settings=s,
        index=idx,
        search_tools=tools,
        search_agent=(make_search_agent(planner, tools) if tools and not offline else None),
        summarize_agent=(make_summarize_agent(planner) if not offline else None),
        rag_tools_factory=(lambda collector: make_rag_tools(idx, collector, s)) if not offline else None,
        writer_agent=(make_writer_agent(planner) if not offline else None),
        planner=planner,
        offline=offline,
    )


async def build_app(
    settings: Settings | None = None,
    search_tools: list[BaseTool] | None = None,
    embeddings: Any | None = None,
    index: PaperIndex | None = None,
    model: Any | None = None,
):
    """一步到位：加载 MCP 工具 → 组装依赖 → 编译图。"""
    s = settings or get_settings()
    if search_tools is None:
        try:
            from ..sources.mcp import load_mcp_tools

            search_tools = await load_mcp_tools(s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP 工具加载失败，将只用本地能力：%s", exc)
            search_tools = []
    deps = build_deps(s, search_tools=search_tools, embeddings=embeddings, index=index, model=model)
    return build_graph(deps), deps


# --------------------------------------------------------------------------
# 最简模式：单 agent + 子能力作为工具
# --------------------------------------------------------------------------


async def build_simple_app(
    settings: Settings | None = None,
    search_tools: list[BaseTool] | None = None,
    embeddings: Any | None = None,
    index: PaperIndex | None = None,
    model: Any | None = None,
):
    """`--simple`：一个 agent 拿到全部工具，自行决定检索/入库/写作顺序。

    图只有两个节点：agent（ReAct 循环）→ finalize（落盘）。
    """
    from langchain.agents import create_agent
    from langgraph.graph import END, START, StateGraph

    s = settings or get_settings()
    emb = embeddings if embeddings is not None else get_embeddings(s)
    idx = index if index is not None else PaperIndex.load_or_create(emb, s)
    tools = list(search_tools or [])
    if tools is None or not tools:
        from ..sources.mcp import load_mcp_tools

        tools = await load_mcp_tools(s)

    collector = CitationCollector(prefix="A-")
    tools = tools + make_paper_tools(idx, s) + make_rag_tools(idx, collector, s)
    llm = model or get_chat_model("default", s)

    system = (
        "你是学术调研助手，可以调用 MCP 检索工具找论文、下载并入库 PDF、检索本地语料、写报告。\n"
        "流程建议：检索（每次少量）→ 下载入库 → 检索语料逐篇分析 → 输出带 [C#] 引用的中文报告。\n"
        "最终消息请直接给出完整 Markdown 报告（含参考文献列表）。"
    )
    agent = create_agent(model=llm, tools=tools, system_prompt=system, name="simple_agent")

    async def run_agent(state: ResearchState) -> dict[str, Any]:
        task = (
            f"研究主题：{state.get('query', '')}\n"
            f"最多关注 {s.max_papers} 篇论文。请完成检索、入库、分析并给出报告。"
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": task}]},
            {"recursion_limit": 60, "max_concurrency": s.concurrency},
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []
        text = ""
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                text = content
                break
        return {
            "report_md": text or "（agent 未产出内容）",
            "citations": collector.citations,
            "notes": [f"simple 模式：agent 调用 {len(messages)} 条消息"],
        }

    async def finalize(state: ResearchState) -> dict[str, Any]:
        return {"notes": ["simple 模式报告已生成"]}

    g = StateGraph(ResearchState)
    g.add_node("agent", run_agent)
    g.add_node("finalize", finalize)
    g.add_edge(START, "agent")
    g.add_edge("agent", "finalize")
    g.add_edge("finalize", END)

    from langgraph.checkpoint.memory import InMemorySaver

    return g.compile(checkpointer=InMemorySaver()), idx, collector
