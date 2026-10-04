# -*- coding: utf-8 -*-
"""MCP 接入层：把搜索引擎 MCP server 的工具挂到 LangChain agent 上。

- `mcp_servers.json` 只存模板（命令/传输），`${VAR}` 由环境变量展开，未配置的项会被丢弃；
- 工具数很多（arxiv 19 + paper-search 57），用白名单收敛，避免上下文被 schema 占满；
- 工具结果统一「文本化」成 JSON 字符串，方便模型阅读与离线单测。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langchain_core.tools import BaseTool

from .config import MCP_SOURCE_KINDS, Settings, get_settings
from .utils import mcp_result_to_text

if TYPE_CHECKING:  # 仅用于类型标注，运行时不导入
    from langchain_mcp_adapters.sessions import (
        SSEConnection,
        StdioConnection,
        StreamableHttpConnection,
        WebsocketConnection,
    )

    Connection = StdioConnection | SSEConnection | StreamableHttpConnection | WebsocketConnection

logger = logging.getLogger(__name__)

# 归一化后的工具名（去掉 `<server>_` 前缀）白名单：只放开「检索 + 取全文/分节」。
ALLOWED_TOOLS: set[str] = {
    # 检索
    "search_papers",
    "search_arxiv",
    "search_openalex",
    "search_crossref",
    "search_semantic",
    "search_pubmed",
    "search_europepmc",
    "search_pmc",
    "search_dblp",
    "search_doaj",
    "search_zenodo",
    "search_hal",
    # 元数据/全文
    "get_abstract",
    "download_paper",
    "download_arxiv",
    "download_openalex",
    "read_paper",
    "read_arxiv_paper",
    "list_papers",
    "get_paper_outline",
    "list_paper_latex_sections",
    "read_paper_section",
    "get_paper_latex_section",
    "search_paper_text",
    "citation_graph",
    "export_citations",
}

# 明确排除：合规风险 / 不稳定 / 与主流程无关。
BLOCKED_TOOLS: set[str] = {
    "download_scihub",
    "search_google_scholar",
    "watch_topic",
    "unwatch_topic",
    "check_alerts",
    "list_watches",
    "reindex",
}

# 未配 key 时必然 429 的工具（Semantic Scholar 匿名共享池限流），直接不暴露给模型。
KEYLESS_BLOCKED_TOOLS: set[str] = {"search_semantic"}

# `filter_tools` 在拿不到 Settings 时的纯默认：空 = 不强制收敛 sources（默认无启用渠道）
DEFAULT_MCP_SOURCES = ""

# 按工具名判断它查的是哪个源（聚合工具 `search_papers` 不在其中）。
_SEARCH_TOOL_KINDS: tuple[tuple[str, str], ...] = (
    ("openalex", "openalex"),
    ("crossref", "crossref"),
    ("europepmc", "europepmc"),
    ("pmc", "europepmc"),
    ("pubmed", "pubmed"),
    ("semantic", "semantic"),
    ("arxiv", "arxiv"),
    ("doaj", "doaj"),
    ("dblp", "dblp"),
    ("zenodo", "zenodo"),
    ("hal", "hal"),
)


def _search_tool_kind(base_name: str, server: str = "") -> str:
    """检索工具对应的源 kind；`search_papers` 聚合工具返回空（由 `sources` 参数限定）。"""
    low = base_name.lower()
    if low == "search_papers":
        # 只有「服务端名 = 某个源 kind」的单源工具（如 arxiv-mcp-server）需要按渠道限定；
        # paper-search（多源聚合）与其他自定义 server 都当作聚合工具保留。
        return server if server in MCP_SOURCE_KINDS and server != "paper-search" else ""
    if "search" not in low:
        return ""
    for hint, kind in _SEARCH_TOOL_KINDS:
        if hint in low:
            return kind
    return ""


def _server_of(tool_name: str, servers: list[str] | None) -> str:
    for server in servers or []:
        if tool_name.startswith(f"{server}_"):
            return server
    return ""

_BRACED_VAR_RE = re.compile(r"\$\{([^}]+)\}")


# --------------------------------------------------------------------------
# 服务器定义加载
# --------------------------------------------------------------------------


QUIET_DIR = Path(__file__).resolve().parent / "mcp_quiet"


def quiet_env() -> dict[str, str]:
    """让 MCP 子进程安静下来的环境变量（收敛第三方 logger 的 INFO/WARNING 噪声）。"""
    env: dict[str, str] = {
        "FASTMCP_LOG_LEVEL": "ERROR",
        "FASTMCP_SHOW_SERVER_BANNER": "false",
        "PYTHONWARNINGS": "ignore",
    }
    if QUIET_DIR.exists():
        existing = os.environ.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{QUIET_DIR}{os.pathsep}{existing}" if existing else str(QUIET_DIR)
    return env


def _expand(value: Any) -> Any:
    """展开字符串里的 ${VAR}；展开后仍含未解析变量的项由调用方过滤。"""
    if isinstance(value, str):
        return _BRACED_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _has_unresolved(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_BRACED_VAR_RE.search(value))
    if isinstance(value, dict):
        return any(_has_unresolved(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_unresolved(v) for v in value)
    return False


def default_bin(name: str, settings: Settings | None = None) -> str:
    """优先 settings 覆盖，其次当前解释器所在目录的 console script，最后 PATH。"""
    s = settings or get_settings()
    override = {
        "arxiv": s.arxiv_mcp_bin,
        "paper-search": s.paper_search_mcp_bin,
    }.get(name, "")
    if override:
        return override

    candidates = {
        "arxiv": "arxiv-mcp-server",
        "paper-search": "paper-search-mcp",
    }
    script = candidates.get(name)
    if not script:
        return ""

    local = Path(sys.executable).parent / script
    if local.exists():
        return str(local)
    found = shutil.which(script)
    return found or ""


def load_server_specs(settings: Settings | None = None) -> dict[str, dict[str, Any]]:
    """读取并展开 `mcp_servers.json`（已过滤 disabled / 缺二进制的 server）。"""
    s = settings or get_settings()
    path = s.servers_file
    if not path.exists():
        logger.warning("未找到 MCP 配置文件：%s", path)
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    env_overlay = {k: v for k, v in os.environ.items()}
    env_overlay.update({k: v for k, v in s.mcp_env().items()})

    specs: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        if not spec.get("enabled", True):
            logger.info("跳过已禁用 MCP server: %s", name)
            continue

        expanded = _expand(spec)

        # stdio：命令可能是 ${XXX_BIN} 占位符，未配置时退回自动探测
        if expanded.get("transport") == "stdio":
            command = expanded.get("command", "")
            if not command or _has_unresolved(command) or not Path(command).exists():
                command = default_bin(name, s)
            if not command or not Path(command).exists():
                logger.warning(
                    "跳过 MCP server %s：找不到可执行文件（%s）。"
                    "可在 .env 配置 %s_MCP_BIN，或 pip install 对应服务端。",
                    name,
                    expanded.get("command") or "<empty>",
                    name.upper().replace("-", "_"),
                )
                continue
            expanded["command"] = command
            env = {k: v for k, v in (expanded.get("env") or {}).items() if not _has_unresolved(v)}
            env.update({k: v for k, v in s.mcp_env().items()})
            env.update(quiet_env())
            if env:
                expanded["env"] = env
            else:
                expanded.pop("env", None)

        if expanded.get("transport") != "stdio" and _has_unresolved(expanded.get("url", "")):
            logger.warning("跳过 MCP server %s：远程 URL 尚未配置具体地址/key", name)
            continue

        expanded.pop("enabled", None)
        expanded.pop("description", None)
        specs[name] = expanded

    return specs


# --------------------------------------------------------------------------
# 客户端
# --------------------------------------------------------------------------


def build_client(connections: dict[str, dict[str, Any]] | None = None, settings: Settings | None = None):
    """构造 MultiServerMCPClient（需要 connections 非空）。"""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    conns = connections if connections is not None else load_server_specs(settings)
    if not conns:
        raise RuntimeError(
            "没有可用的 MCP server：请先 `pip install arxiv-mcp-server paper-search-mcp`，"
            "或在 .env 中配置 *_MCP_BIN / 远程 MCP URL。"
        )
    return MultiServerMCPClient(
        cast("dict[str, Connection]", conns),
        tool_name_prefix=True,      # 多 server 同名工具加前缀，避免覆盖
        handle_tool_errors=True,    # 工具异常变成错误文本，不炸掉整个图
        tool_interceptors=[logging_interceptor],
    )


def logging_interceptor(request, handler):  # noqa: ANN001 - 由 adapters 传入
    """工具调用日志（异步拦截器）。"""
    start = time.perf_counter()

    async def _run() -> Any:
        result = await handler(request)
        logger.info(
            "MCP tool=%s耗时=%.1fs args=%s",
            getattr(request, "name", "?"),
            time.perf_counter() - start,
            str(getattr(request, "args", ""))[:160],
        )
        return result

    return _run()


def strip_prefix(tool_name: str, servers: list[str] | None = None) -> str:
    """去掉 `<server>_` 前缀，得到原始 MCP 工具名。

    只在名字确实以某个已知 server 名为前缀时才裁剪；否则原样返回
    （工具名本身可能含下划线，例如 `download_scihub`/`search_papers`，
    盲目按下划线切分会破坏白名单匹配）。
    """
    for server in servers or []:
        prefix = f"{server}_"
        if tool_name.startswith(prefix):
            return tool_name[len(prefix) :]
    return tool_name


def textify_tool(tool: BaseTool) -> BaseTool:
    """把 MCP content blocks 结果转成纯 JSON 文本（原地包装 coroutine）。"""
    orig = getattr(tool, "coroutine", None)
    if orig is None:  # 同步工具（理论上 MCP 工具都是异步的）
        return tool

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = await orig(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            content, artifact = result
            return mcp_result_to_text(content), artifact
        return mcp_result_to_text(result)

    tool.coroutine = wrapper  # type: ignore[assignment]
    return tool


def _accepts_arg(tool: BaseTool, name: str) -> bool:
    """工具签名是否接受某个参数（兼容 pydantic schema 与 MCP 原始 JSON schema）。"""
    args = getattr(tool, "args", None)
    if isinstance(args, dict) and args:
        return name in args
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return False
    if hasattr(schema, "model_json_schema"):
        try:
            return name in (schema.model_json_schema().get("properties") or {})
        except Exception:  # noqa: BLE001
            return False
    return False


def guard_search_sources(tool: BaseTool, settings: Settings | None = None) -> BaseTool:
    """给 paper-search 的 `search_papers` 加参数护栏：收敛 `sources`。

    模型常省略 `sources`（默认 `all`），会连带触发 Semantic Scholar（无 key 必 429）
    与 SSRN/BASE 等慢源。调用前改写参数；工具签名没有 `sources` 时原样返回。
    """
    orig = getattr(tool, "coroutine", None)
    if orig is None or not _accepts_arg(tool, "sources"):
        return tool

    has_key = bool(settings is not None and settings.semantic_scholar_key)
    default_sources = settings.mcp_sources if settings is not None else DEFAULT_MCP_SOURCES

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        requested = str(kwargs.get("sources") or "").strip()
        if not requested or requested.lower() == "all":
            if default_sources:  # 未启用任何渠道时不动，交给工具自身默认
                kwargs["sources"] = default_sources
        else:
            parts = [p.strip() for p in requested.split(",") if p.strip()]
            if not has_key:
                parts = [p for p in parts if p not in ("semantic", "semanticscholar")]
            joined = ",".join(parts) or default_sources
            if joined:
                kwargs["sources"] = joined
            else:
                kwargs.pop("sources", None)
        return await orig(*args, **kwargs)

    tool.coroutine = wrapper  # type: ignore[assignment]
    return tool


def filter_tools(
    tools: list[BaseTool],
    servers: list[str] | None = None,
    allowlist: set[str] | None = None,
    blocklist: set[str] | None = None,
    settings: Settings | None = None,
    only_enabled_sources: bool = False,
) -> list[BaseTool]:
    """按白名单/黑名单过滤，并做文本化 / 参数护栏包装。

    传入 `settings` 时额外收敛 keyless 工具（无 S2 key 时不暴露 `search_semantic`）。
    `only_enabled_sources=True` 时进一步**只保留已启用渠道对应的检索工具**
    （聚合工具 `search_papers` 保留，靠 `sources` 参数限定）——避免用户只启用
    Tavily 时，MCP 的 `search_arxiv` 还把 arXiv 结果拉回来。
    """
    allow = ALLOWED_TOOLS if allowlist is None else allowlist
    block = set(BLOCKED_TOOLS if blocklist is None else blocklist)
    if settings is not None and not settings.semantic_scholar_key:
        block |= KEYLESS_BLOCKED_TOOLS
    enabled = (
        {x.strip().lower() for x in settings.mcp_sources.split(",") if x.strip()}
        if (only_enabled_sources and settings is not None)
        else None
    )
    kept: list[BaseTool] = []
    for t in tools:
        base = strip_prefix(t.name, servers)
        if base in block:
            continue
        if allow and base not in allow:
            continue
        if enabled is not None:
            kind = _search_tool_kind(base, _server_of(t.name, servers))
            if kind and kind not in enabled:
                continue
        t = textify_tool(t)
        if base == "search_papers":
            t = guard_search_sources(t, settings)
        kept.append(t)
    return kept


async def load_mcp_tools(
    settings: Settings | None = None,
    connections: dict[str, dict[str, Any]] | None = None,
    allowlist: set[str] | None = None,
    blocklist: set[str] | None = None,
    server_names: list[str] | None = None,
) -> list[BaseTool]:
    """加载并过滤 MCP 工具（默认全部启用的 server）。"""
    conns = connections if connections is not None else load_server_specs(settings)
    if server_names:
        conns = {k: v for k, v in conns.items() if k in server_names}
    s = settings or get_settings()
    client = build_client(conns, s)
    tools: list[BaseTool] = []
    for name in conns:
        tools.extend(await client.get_tools(server_name=name))
    return filter_tools(
        tools,
        servers=list(conns),
        allowlist=allowlist,
        blocklist=blocklist,
        settings=s,
        only_enabled_sources=True,
    )


async def describe_mcp_tools(settings: Settings | None = None) -> dict[str, Any]:
    """诊断信息：哪些 server 可用、各暴露了多少工具、白名单后保留多少。"""
    s = settings or get_settings()
    specs = load_server_specs(s)
    report: dict[str, Any] = {"servers": {}, "kept": [], "total_raw": 0}
    if not specs:
        report["error"] = "没有可用的 MCP server"
        return report

    client = build_client(specs, s)
    for name in specs:
        try:
            raw = await client.get_tools(server_name=name)
        except Exception as exc:  # noqa: BLE001
            report["servers"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        kept = filter_tools(raw, servers=list(specs), settings=s)
        report["total_raw"] += len(raw)
        report["servers"][name] = {
            "raw_tools": len(raw),
            "kept_tools": len(kept),
            "kept_names": sorted(t.name for t in kept),
        }
        report["kept"].extend(t.name for t in kept)
    return report
