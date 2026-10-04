# -*- coding: utf-8 -*-
"""RAG 问答 agent：只依据检索到的全文片段回答，并强制标注引用锚点。"""

from __future__ import annotations

import asyncio
import logging

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ..schema import Answer
from ..utils import extract_json
from . import prompts
from .common import result_text, structured_response

logger = logging.getLogger(__name__)


def _output_format() -> str:
    return prompts.RAG_PROMPT_ZH + "\n\n【输出格式】只输出答案正文（Markdown），不要输出 JSON、不要出现 citation_ids 等字段名。"


def make_rag_agent(model: BaseChatModel, tools: list[BaseTool]):
    # 不用 `response_format`：它会强制 tool_choice，思考型模型会报 400。
    # 改为输出带锚点的 Markdown，再从正文解析引用（与流式路径一致）。
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=_output_format(),
        name="rag_agent",
    )


async def _plain_rag_answer(model, question: str, context: str, timeout: float) -> Answer:
    """普通对话作答（无工具、无 response_format），再从正文解析引用锚点。"""
    from ..rag.retriever import anchors_in

    available = anchors_in(context)
    task = (
        f"问题：{question}\n\n可用上下文：\n{context}\n\n"
        f"可用的引用锚点（必须原样复制，不要改写）：{', '.join(available) if available else '（无）'}\n"
        "请基于上下文回答，并在每个结论句后标注锚点。"
    )
    try:
        resp = await asyncio.wait_for(
            model.ainvoke([("system", _output_format()), ("human", task)]), timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("纯文本作答失败：%s", exc)
        return Answer(question=question, text="资料不足，未能形成可靠结论。", citation_ids=[])
    content = getattr(resp, "content", "")
    if isinstance(content, list):
        content = "".join(
            str(b.get("text") or "") if isinstance(b, dict) else str(b) for b in content
        )
    text = str(content or "").strip()
    if not text:
        return Answer(question=question, text="资料不足，未能形成可靠结论。", citation_ids=[])
    return Answer(question=question, text=text, citation_ids=anchors_in(text))


async def run_rag_agent(
    agent,
    question: str,
    context: str,
    model=None,
    timeout: float = 180.0,   # 单次等待响应上限：180s（与服务端 timeout 对齐）
) -> Answer:
    """带引用的问答（失败时先回退到纯文本作答，再不行才返回「资料不足」）。"""
    from ..rag.retriever import anchors_in

    available = anchors_in(context)
    task = (
        f"问题：{question}\n\n可用上下文：\n{context}\n\n"
        f"可用的引用锚点（必须原样复制，不要改写）：{', '.join(available) if available else '（无）'}\n"
        "请基于上下文回答，并在每个结论句后标注锚点；citation_ids 填实际用到的锚点。"
    )
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": task}]}, {"recursion_limit": 14}),
            timeout=timeout,
        )
        answer = structured_response(result)
        if isinstance(answer, Answer) and answer.text:
            answer.question = answer.question or question
            return answer

        payload = extract_json(result_text(result))
        if isinstance(payload, dict) and payload.get("text"):
            return Answer(
                question=question,
                text=str(payload.get("text", "")),
                citation_ids=list(payload.get("citation_ids", []) or []),
            )

        # 无结构化输出（已不用 response_format）：取最后一条 AI 消息正文，再解析锚点
        from .common import last_ai_text
        from ..pipeline import clean_stream_output, collapse_repetition

        text = clean_stream_output(collapse_repetition(last_ai_text(result))).strip()
        if text:
            return Answer(question=question, text=text, citation_ids=anchors_in(text))
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG agent 失败：%s", exc)

    # 回退：结构化/工具调用被拒时用普通对话作答
    if model is not None:
        fallback = await _plain_rag_answer(model, question, context, timeout)
        if fallback.text and fallback.text != "资料不足，未能形成可靠结论。":
            return fallback

    return Answer(question=question, text="资料不足，未能形成可靠结论。", citation_ids=[])


async def answer_without_llm(question: str, context: str) -> Answer:
    """离线/降级路径：不调用模型，直接把检索到的证据作为回答（保留锚点）。"""
    from ..rag.retriever import anchors_in

    ids = anchors_in(context)
    if not ids:
        return Answer(question=question, text="资料不足，未能形成可靠结论。", citation_ids=[])

    # 每条证据取前 200 字，保证回答里有可校验的句子 + 锚点
    blocks: list[str] = []
    for block in context.split("\n\n---\n\n"):
        anchor = anchors_in(block)
        if not anchor:
            continue
        body = block.split("\n", 1)[1] if "\n" in block else block
        blocks.append(f"{body[:200].strip()} [{anchor[0]}]")
    text = "（离线模式：未调用模型，以下为检索到的原文证据）\n\n" + "\n\n".join(blocks)
    return Answer(question=question, text=text, citation_ids=ids)
