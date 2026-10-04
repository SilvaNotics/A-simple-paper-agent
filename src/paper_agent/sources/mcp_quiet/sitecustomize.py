# -*- coding: utf-8 -*-
"""静音 MCP 子进程的日志（通过 PYTHONPATH 注入的 sitecustomize 自动生效）。

MCP server 默认会把「收到 ListToolsRequest」之类的 INFO/WARNING 直接写到 stderr，
在交互式会话里会刷屏。这里在子进程启动阶段把第三方 logger 收敛到 ERROR，
只保留真正的错误信息。`sources/mcp.py` 的 `load_server_specs()` 会把本目录加进子进程的
PYTHONPATH，因此不影响我们自己的进程日志。
"""

from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore")

for _name in (
    "mcp",
    "fastmcp",
    "httpx",
    "httpx2",
    "httpcore",
    "urllib3",
    "arxiv",
    "arxiv_mcp_server",
    "paper_search_mcp",
    "unpaywall",
    "core",
    "doaj",
    "semantic_scholar",
):
    logging.getLogger(_name).setLevel(logging.ERROR)

logging.getLogger().setLevel(logging.ERROR)
