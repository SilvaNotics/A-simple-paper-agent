# -*- coding: utf-8 -*-
"""agent 公共辅助：结构化结果提取与文本兜底。"""

from __future__ import annotations

from pydantic import BaseModel

from ..utils import mcp_result_to_text


def structured_response(result: object) -> BaseModel | None:
    """从 `create_agent(response_format=...)` 的返回值里取结构化结果。"""
    if isinstance(result, dict):
        for key in ("structured_response", "response"):
            value = result.get(key)
            if isinstance(value, BaseModel):
                return value
    return None


def last_ai_text(result: object) -> str:
    """取最后一条 AI 消息的文本（agent 的最终回答正文）。

    不拼全部消息，否则会把 system / 人类提问 / 工具输出都带进来。
    """
    if not isinstance(result, dict):
        return str(result)
    for msg in reversed(result.get("messages", []) or []):
        name = type(msg).__name__
        if "AI" not in name and getattr(msg, "type", "") != "ai":
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text = mcp_result_to_text(content)
            if text.strip():
                return text
    return result_text(result)


def result_text(result: object) -> str:
    """把 agent 返回的消息内容拼成文本（用于兜底 JSON 解析）。"""
    if not isinstance(result, dict):
        return str(result)
    parts: list[str] = []
    for msg in result.get("messages", []) or []:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.append(mcp_result_to_text(content))
    return "\n".join(parts)
