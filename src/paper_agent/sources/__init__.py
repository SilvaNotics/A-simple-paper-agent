# -*- coding: utf-8 -*-
"""来源层：搜索渠道注册表、内置联网抓取、MCP 接入与供应商凭证配置。

- `channels.py`：渠道元数据（kind / base_url / 默认值）；
- `fetchers.py`：内置 HTTP 抓取（arXiv / OpenAlex / Crossref / …）；
- `oa.py`：全文获取兜底（Unpaywall / OpenAlex / 落地页补链 + 网页正文抽取）；
- `mcp.py`：MCP server 接入（`mcp_servers.json` + `mcp_quiet/`）；
- `userconfig.py`：`/connect` 写入的供应商 JSON（密钥 + 模型选择）。
"""
