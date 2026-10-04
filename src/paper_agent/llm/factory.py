# -*- coding: utf-8 -*-
"""模型层：统一产出 ChatModel 与 Embeddings。

- 默认走 `LLM_*` / `/connect` 配置的 OpenAI 兼容供应商；无配置时回退 .env 的 DashScope / DeepSeek。
- DeepSeek 不提供 embedding，embedding 可与对话供应商不同。
- `PAPER_AGENT_FAKE_LLM=1` 时返回确定性假模型，供离线测试。
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from ..core.config import Settings, get_settings

# 角色 -> temperature。检索/抽取类任务要稳定，写作类允许一点多样性。
_ROLE_TEMPERATURE: dict[str, float] = {
    "default": 0.2,
    "search": 0.0,
    "summarize": 0.2,
    "synthesize": 0.2,
    "write": 0.4,
}


class ConfigError(RuntimeError):
    """配置缺失（如未设置任何 API Key）。"""


def _temperature_for(role: str) -> float:
    return _ROLE_TEMPERATURE.get(role, _ROLE_TEMPERATURE["default"])


def _thinking_body(s: Settings) -> dict[str, Any] | None:
    """按配置决定是否传 `enable_thinking`（默认不传，交给服务端）。"""
    if s.enable_thinking:
        return {"enable_thinking": True}
    if s.disable_thinking:
        return {"enable_thinking": False}
    return None


def get_chat_model(role: str = "default", settings: Settings | None = None, **kwargs: Any):
    """返回可用的 ChatModel。

    Args:
        role: 角色名，用于选择 temperature（见 `_ROLE_TEMPERATURE`）。
        settings: 显式传入配置；默认读进程单例。
        **kwargs: 覆盖 ChatOpenAI 参数（如 `model`、`max_tokens`）。
    """
    s = settings or get_settings()

    if s.fake_llm:
        from .fake import FakeToolCallingModel

        return FakeToolCallingModel()

    temperature = kwargs.pop("temperature", _temperature_for(role))

    # ① 通用 OpenAI 兼容供应商（/connect 配置的 JSON 或 LLM_* 环境变量）
    if s.llm_base_url and s.llm_api_key:
        model = kwargs.pop("model", None) or s.llm_model
        if not model:
            raise ConfigError(
                f"供应商「{s.provider_label}」还没有选定对话模型。\n"
                "  处理方式：/models 选择，或 /model <模型名> 直接指定。"
            )
        extra_body = kwargs.pop("extra_body", None)
        if extra_body is None and s.is_dashscope:
            extra_body = _thinking_body(s)
        return ChatOpenAI(
            model=model,
            api_key=s.llm_api_key,
            base_url=s.llm_base_url,
            temperature=temperature,
            extra_body=extra_body,
            timeout=kwargs.pop("timeout", s.llm_timeout),
            max_retries=kwargs.pop("max_retries", s.llm_max_retries),
            **kwargs,
        )

    # ② .env 里的 DashScope / DeepSeek 专用配置
    if s.dashscope_api_key:
        extra_body = kwargs.pop("extra_body", None)
        if extra_body is None:
            extra_body = _thinking_body(s)
        return ChatOpenAI(
            model=kwargs.pop("model", s.qwen_model),
            api_key=s.dashscope_api_key,
            base_url=s.dashscope_base_url,
            temperature=temperature,
            extra_body=extra_body,
            timeout=kwargs.pop("timeout", s.llm_timeout),
            max_retries=kwargs.pop("max_retries", s.llm_max_retries),
            **kwargs,
        )

    if s.deepseek_api_key:
        return ChatOpenAI(
            model=kwargs.pop("model", s.deepseek_model),
            api_key=s.deepseek_api_key,
            base_url=s.deepseek_base_url,
            temperature=temperature,
            timeout=kwargs.pop("timeout", s.llm_timeout),
            max_retries=kwargs.pop("max_retries", s.llm_max_retries),
            **kwargs,
        )

    raise ConfigError(
        "未配置任何模型密钥：请在 .env 中设置 DASHSCOPE_API_KEY（推荐，自带 embedding）"
        " 或 DEEPSEEK_API_KEY。"
    )


def get_embeddings(settings: Settings | None = None, **kwargs: Any):
    """返回 Embeddings 实例。

    `check_embedding_ctx_length=False`：否则 langchain 会发 token 数组，兼容接口不支持。
    """
    s = settings or get_settings()

    if s.fake_llm:
        from ..core.utils import DeterministicFakeEmbeddings

        return DeterministicFakeEmbeddings(dim=64)

    from ..rag.embeddings import RetryingEmbeddings

    def _generic_embeddings(base_url: str, api_key: SecretStr, model: str):
        base = OpenAIEmbeddings(
            model=model,
            api_key=SecretStr(api_key.get_secret_value()),
            base_url=base_url,
            check_embedding_ctx_length=False,
            chunk_size=kwargs.pop("chunk_size", s.embed_batch_size),
            request_timeout=kwargs.pop("request_timeout", s.llm_timeout),
            max_retries=kwargs.pop("max_retries", s.llm_max_retries),
            **kwargs,
        )
        return RetryingEmbeddings(base)

    # ① 专用 embedding 供应商（可与对话供应商不同，例如 chat=DeepSeek + embed=DashScope）
    if s.embed_base_url and s.embed_api_key and s.embed_model:
        return _generic_embeddings(s.embed_base_url, s.embed_api_key, s.embed_model)

    # ② 当前供应商自带 embedding
    if s.llm_base_url and s.llm_api_key and s.llm_embedding_model:
        return _generic_embeddings(s.llm_base_url, s.llm_api_key, s.llm_embedding_model)

    # ③ .env 的 DashScope 兜底 embedding
    if s.dashscope_api_key:
        return _generic_embeddings(s.dashscope_base_url, s.dashscope_api_key, s.embedding_model)

    # ④ 都不行：给出可执行的建议
    if s.llm_base_url and s.llm_api_key:
        raise ConfigError(
            f"供应商「{s.provider_label}」没有可用的 embedding 模型，RAG 无法建索引。\n"
            "  解决办法（任选）：\n"
            "    1) /connect 添加一个带 embedding 的供应商（如 DashScope 的 text-embedding-v4）；\n"
            "    2) /models --embedding 为该供应商指定 embedding 模型；\n"
            "    3) 在 .env 配置 DASHSCOPE_API_KEY 作为兜底 embedding。"
        )
    raise ConfigError(
        "缺少 embedding 配置：请执行 /connect 配置一个 OpenAI 兼容供应商，"
        "或在 .env 中设置 DASHSCOPE_API_KEY。"
    )
