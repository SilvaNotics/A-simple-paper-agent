# 学术论文检索与概括分析 Agent（纯 Python + LangChain）设计方案

> ⚠️ **本文是历史设计记录（设计期的方案 + 当时的实测回填），不是当前行为规范。**
> 当前实现以 [`../ARCHITECTURE.md`](../ARCHITECTURE.md)（模块与机制）和 [`../../README.md`](../../README.md)（用法与配置）为准：
> 其中提到的路径可能已变动（如供应商配置/命令行历史现在都在项目内 `.paper-agent/`，日志在 `logs/`，工作目录也已不是文中的路径；扁平模块后来也按层归入了 `core/ sources/ llm/ pipeline/ pdf/ repl/` 子包）。
> 需要「当时为什么这么设计」时看本文；需要「现在怎么用/怎么改」时看 ARCHITECTURE + README。

> 工作目录：`/home/silva/projects/pypj/llmTest`（conda env `agent`，Python 3.14.7）
> 状态：**已按本方案实现并验证**（见 §14 实现状态与实测回填）

---

## 1. 目标与范围

做一个**专用于学术论文查找 + 概括分析**的 Agent，约束与能力要求：

| 编号 | 要求 | 落点 |
|---|---|---|
| R1 | 纯 Python 实现 | 不引入 Node 运行时依赖（`npx` 型 MCP 仅作可选） |
| R2 | 基于 LangChain | `langchain` 1.x + LangGraph（已装） |
| R3 | 能接入搜索引擎的 MCP | `langchain-mcp-adapters` + arXiv / paper-search MCP（可扩展 Exa/Tavily 等 web 搜索 MCP） |
| R4 | 能用 RAG 分析文档 | PDF 抓取 → 解析 → 切分 → 向量化 → 检索 → 带引用的归纳 |
| R5 | 支持多 agent 分工 | LangGraph `StateGraph` 监督者 + 4 个专家 agent（含并行 map-reduce） |
| R6 | 其他方面相对简洁 | 单向量库（内存 + 本地持久化）、无中间件服务、CLI 单一入口、可选组件全部标「可选」 |

**交付物**：本方案文档（依赖 / 项目结构 / 各功能实现形式），以及按第 10 节步骤可实现出的代码骨架。

---

## 2. 现状盘点（已实测，非推测）

### 2.1 已有代码

| 文件 | 内容 | 备注 |
|---|---|---|
| ~~`src/useqwen.py`~~ / ~~`src/useds.py`~~ | 早期单独调 Qwen / DeepSeek 的示例脚本 | 已删除；功能由 `src/paper_agent/llm.py` 覆盖 |
| ~~`tests/_test1.py`~~ | 手工测试入口 | 已删除（有 pytest 测试集） |
| `output/` | 输出约定（`.gitignore` 已忽略） | 报告落盘位置 |
| `environment.yml` | name=`agent`，含 langchain 1.4.2 / langgraph 1.2.11 | 依赖追加处 |
| `.env` | `DEEPSEEK_*`、`DASHSCOPE_*`（key 已配置） | 无需新增 LLM 密钥 |

### 2.2 环境实测结论（本方案的关键依据）

```
langchain 1.4.2 | langchain-core 1.6.6 | langgraph 1.2.11 | langgraph-prebuilt 1.1.0
langchain-openai 1.6.7 | langchain-deepseek 1.1.1 | langchain-community 0.4.2 | langchain-classic 1.0.8
langchain-text-splitters 1.1.2 | dashscope 1.27.7 | pdfplumber 0.11.10 | pdfminer.six | pypdfium2
numpy 2.5.3 | rich | typer | SQLAlchemy | tenacity | pydantic 2.13.5 | pydantic-settings 2.15.0
```

- ✅ `from langchain.agents import create_agent` 可用（参数含 `model/tools/system_prompt/middleware/response_format/state_schema/checkpointer`）。
- ✅ `langgraph.graph.StateGraph`、`langgraph.types.Send`、`Command`、`InMemorySaver` 可用。
- ✅ `langchain_core.vectorstores.InMemoryVectorStore` 可用（纯 Python + numpy，无额外依赖）。
- ✅ `langchain_classic.retrievers.EnsembleRetriever`、`langchain_community.retrievers.BM25Retriever` 可用（混合检索用）。
- ✅ **DashScope 兼容接口实测通过**：`chat`（`qwen3.8-max`/`qwen-max`/`qwen-plus`）与 `embeddings`（`text-embedding-v4`，**dim=1024**）。RAG 无需本地模型。
- ❌ 未安装：`langchain-mcp-adapters`、`mcp`、`faiss`、`chromadb`、`rank_bm25`、`arxiv`、`pymupdf`、`pytest`。
- ✅ PyPI 有 Python 3.14 可用 wheel：`langchain-mcp-adapters 0.3.2`、`arxiv-mcp-server 0.8.0`、`paper-search-mcp 0.1.4`、`faiss-cpu 1.15.1`、`chromadb 1.5.9`、`pymupdf 1.28.2`、`rank-bm25`、`qdrant-client`。
- ⚠️ **版本约束**：`langchain-mcp-adapters 0.3.2` 依赖 `mcp>=1.24,<2.0`（`pip --dry-run` 实测解析为 `mcp 1.30.0`）。**不要**把 `mcp` 升到 2.x。
- ⚠️ `unstructured` 要求 `python<3.14` → 本环境不可用，PDF 解析走 `pdfplumber`/`pymupdf`。
- ⚠️ 本机未装 `uv/uvx`（`node/npm` 存在）。MCP server 用 **pip console script 绝对路径** 启动，不依赖 uvx。
- ⚠️ `raw.githubusercontent.com` 在本机 DNS 被解析到 127.0.0.1（不可访问）；依赖信息均通过 PyPI wheel 元数据核实。

### 2.3 MCP 服务端实测细节（读 wheel 源码得到，可直接照抄）

**`arxiv-mcp-server` 0.8.0**（`requires-python>=3.11`，`mcp>=1.27,<2`）
- console script：`arxiv-mcp-server = arxiv_mcp_server:main`；也有 `arxiv_mcp_server/__main__.py` → `python -m arxiv_mcp_server` 可用。
- 暴露工具（部分）：`search_papers`、`download_paper`、`list_papers`、`read_paper`、`get_paper_outline`、`read_paper_section`、`get_paper_latex`、`get_paper_latex_section`、`semantic_search`、`search_paper_text`、`citation_graph`、`export_citations`、`watch_topic`/`check_alerts`、`reindex`。
- 可选环境变量（源码字段）：`SEMANTIC_SCHOLAR_API_KEY`、`MAX_RESULTS`、`TRANSPORT`、`HOST`、`PORT`、`ARXIV_*`（超时/重试/退避）。默认 stdio。
- 自带 prompts（`literature_review_prompt`、`compare_papers_prompt`、`summarize_paper_prompt`、`deep_research_analysis_prompt`）→ 可直接借鉴其提示词思路。

**`paper-search-mcp` 0.1.4**（`requires-python>=3.10`，FastMCP）
- console script：`paper-search-mcp = paper_search_mcp.server:main`；另有 `paper-search` CLI。
- 关键工具：**`search_papers(query, max_results_per_source, sources="all", year=None)`** —— 一个工具聚合多源，最适合搜索 agent；以及 `search_arxiv/pubmed/openalex/semantic/crossref/dblp/hal/iacr/...`、`download_*`、`read_*_paper`、`download_with_fallback`。
- 环境变量前缀 `PAPER_SEARCH_MCP_`（源码中读取：`UNPAYWALL_EMAIL`、`SEMANTIC_SCHOLAR_API_KEY`、`CORE_API_KEY`、`DOAJ_API_KEY`、`IEEE_API_KEY`、`ACM_API_KEY`、`CITESEERX_API_KEY`、`OPENAIRE_API_KEY`、`ZENODO_ACCESS_TOKEN`、`GOOGLE_SCHOLAR_PROXY_URL`）。
- ⚠️ 工具数 **60+**（含 `download_scihub`、`search_google_scholar`）→ 必须做 **工具白名单过滤**，否则上下文爆炸。

---

## 3. 总体架构

```mermaid
flowchart TD
    U[CLI: paper-agent report "研究主题"] --> SUP

    subgraph LG[LangGraph 监督图 supervisor.py]
        SUP[plan 节点<br/>拆解子问题 + 生成中英检索式] --> SEARCH
        SEARCH[search 节点<br/>search_agent + MCP 工具<br/>Send 并行 fan-out] --> MERGE[merge/dedupe/rank<br/>按 DOI/arXiv ID 去重]
        MERGE --> SELECT[select<br/>LLM 相关性筛选 Top-K]
        SELECT --> INGEST[ingest 节点<br/>下载 PDF → 解析 → 切分 → 向量化<br/>Send 每篇并行]
        INGEST --> MAP[summarize 节点 map<br/>每篇结构化摘要 + 引用]
        MAP --> RAGNODE[rag_agent 节点<br/>跨论文检索归纳 synthesis]
        RAGNODE --> WRITE[writer_agent<br/>报告 + 参考文献]
        WRITE --> VERIFY{verify<br/>引用/覆盖率校验}
        VERIFY -->|不通过且重试<1| RAGNODE
        VERIFY -->|通过| ENDN[END]
    end

    SEARCH -.MCP stdio/HTTP.-> M1[arxiv-mcp-server]
    SEARCH -.MCP stdio.-> M2[paper-search-mcp]
    SEARCH -.MCP HTTP 可选.-> M3[Exa / Tavily web 搜索]
    INGEST --> RAG[(RAG 存储<br/>InMemoryVectorStore + 本地持久化)]
    RAGNODE --> RAG
    ENDN --> OUT[output/*.md, *.bib, *.json]
```

**分层**：配置/模型层 → 能力层（MCP 工具、本地 RAG 工具）→ 专家 agent 层 → 监督图编排层 → CLI/报告层。

---

## 4. 依赖方案

### 4.1 必须新增（核心，装完即可跑通 R3）

```bash
conda activate agent
pip install "langchain-mcp-adapters>=0.3.2,<0.4" "mcp>=1.24,<2.0"
pip install "arxiv-mcp-server>=0.8.0" "paper-search-mcp>=0.1.4"   # 本机 stdio 启动的搜索 MCP
pip install "pymupdf>=1.28.2"                                     # 实测：PDF 解析质量明显优于 pdfplumber
```

| 包 | 版本 | 作用 | 备注 |
|---|---|---|---|
| `langchain-mcp-adapters` | 0.3.2 | 把 MCP 工具转成 LangChain `BaseTool`（`MultiServerMCPClient`） | 依赖 `langchain-core>=1.3.3`，与已装 1.6.6 兼容 |
| `mcp` | 1.30.0（自动解析） | MCP 协议实现 | **pin <2.0** |
| `arxiv-mcp-server` | 0.8.0 | arXiv 检索/PDF/LaTeX/引用图/告警 | 提供 `search_papers`、`download_paper`、`read_paper`、`export_citations` |
| `paper-search-mcp` | 0.1.4 | 多源论文检索（arXiv/PubMed/OpenAlex/Crossref/Semantic/…） | 工具多，需白名单 |
| `pymupdf` | 1.28.2 | PDF 解析 | 实测同一篇 25 页论文：`pymupdf` 干净可读，`pdfplumber` 出现乱码/串行；保留 `pdfplumber` 作回退 |

### 4.2 复用已装（无需新增）

`langchain`、`langchain-core`、`langgraph`、`langgraph-prebuilt`、`langgraph-checkpoint`、`langchain-openai`（Chat + Embeddings）、`langchain-deepseek`、`langchain-text-splitters`、`pdfplumber`、`pdfminer.six`、`pypdfium2`、`numpy`、`pydantic`/`pydantic-settings`、`python-dotenv`、`httpx`/`requests`、`tenacity`、`rich`、`typer`、`SQLAlchemy`。

### 4.3 可选（按需，保持"简洁"默认不装）

| 包 | 用途 | 触发条件 |
|---|---|---|
| `pymupdf` | PDF 解析更快更准 | **已改为默认安装**（见 4.1）：实测 pdfplumber 对部分论文会输出乱码 |
| `faiss-cpu` | 向量检索加速 | 语料 > 数万 chunk |
| `chromadb` 或 `qdrant-client`+`langchain-qdrant` | 持久化向量库 | 需要跨进程/多用户共享索引 |
| `rank-bm25` | BM25 关键词检索，做 hybrid（`EnsembleRetriever`） | 检索召回不足时 |
| `arxiv` | 不依赖 MCP 的元数据回退通道 | MCP 不可用时的降级 |
| `pytest`、`pytest-asyncio` | 测试（已安装：63 条用例） | 开发期 |
| `langsmith` | 链路追踪（已装） | 需 `LANGSMITH_API_KEY` |
| `langchain-huggingface` + `sentence-transformers`(+torch) | 本地离线 embedding | 不能调云端 embedding 时（torch 体积大，慎重） |
| MCP 侧：`uv` | 用 `uvx` 跑第三方 MCP | 想跑 `npx`/`uvx` 型 MCP 时（`node` 已存在） |

### 4.4 `environment.yml` 追加（在 `pip:` 段）

```yaml
  - pip:
      # ... 现有条目保留
      - langchain-mcp-adapters==0.3.2
      - mcp>=1.24,<2.0
      - arxiv-mcp-server>=0.8.0
      - paper-search-mcp>=0.1.4
      # 可选
      # - pymupdf
      # - rank-bm25
      # - faiss-cpu
      # - pytest
      # - pytest-asyncio
```

---

## 5. 推荐项目结构

与现有 `src/` 布局保持一致（不推翻已有文件）：

```
llmTest/
├─ docs/PLAN.md                  # 本方案
├─ environment.yml               # 追加 4.4 依赖
├─ .env                          # 已有；新增项见 6.1（不入库）
├─ requirements.txt              # 可选：与 environment.yml pip 段保持一致，便于 pip 安装
├─ main.py                       # 交互式入口（pi 风格 REPL：斜杠命令/补全/历史/流式 Markdown）
├─ src/
│  └─ paper_agent/               # Agent 包（python -m src.paper_agent）
│     ├─ __init__.py
│     ├─ __main__.py             # asyncio.run(cli.main())
│     ├─ pipeline.py             # 可复用流水线（CLI 与 main.py REPL 共用同一套逻辑）
│     ├─ cli.py                  # typer 命令：search / ingest / ask / report / mcp-tools
│     ├─ config.py               # pydantic-settings Settings + .env 加载
│     ├─ llm.py                  # get_chat_model() / get_embeddings()（DashScope 优先，DeepSeek 备选）
│     ├─ schema.py               # Paper / Chunk / PaperSummary / Answer / Citation / 图状态
│     ├─ mcp_client.py           # MultiServerMCPClient 封装 + 工具白名单 + 名称前缀
│     ├─ mcp_servers.json        # MCP server 定义（stdio/http，支持 ${VAR} 展开）
│     ├─ rag/
│     │  ├─ __init__.py
│     │  ├─ fetch.py             # 下载/缓存 PDF（httpx + 重试），OA 链接优先
│     │  ├─ parse.py             # pdfplumber → 文本/分页；可选 pymupdf 加速
│     │  ├─ split.py             # RecursiveCharacterTextSplitter + 元数据装配
│     │  ├─ store.py             # InMemoryVectorStore + JSON/NPY 本地持久化
│     │  └─ retriever.py         # 相似度/MMR/hybrid 检索 + 引用锚点格式化
│     ├─ tools/
│     │  ├─ __init__.py
│     │  ├─ paper_tools.py       # @tool: download_paper / parse_and_index_paper / list_indexed
│     │  └─ rag_tools.py         # @tool: search_corpus / read_chunk / summarize_corpus
│     ├─ agents/
│     │  ├─ __init__.py
│     │  ├─ prompts.py           # 各角色 system prompt（中英）
│     │  ├─ search_agent.py      # create_agent + MCP 工具，结构化输出论文候选列表
│     │  ├─ summarize_agent.py   # 单篇精读：问题/方法/数据/结论/局限/可复用点
│     │  ├─ rag_agent.py         # 跨论文检索问答（带引用）
│     │  ├─ writer_agent.py      # 报告撰写 + 对比表 + 参考文献
│     │  └─ supervisor.py        # StateGraph：plan→search→select→ingest→map→synthesize→write→verify
│     ├─ report.py               # Markdown / BibTeX / JSON 渲染
│     └─ utils.py                # id 归一化、去重、slug、日志
├─ tests/
│  ├─ fixtures/fake_mcp_server.py  # stdio 假 MCP server（离线集成测试）
│  ├─ test_ids_dedupe.py
│  ├─ test_split_store.py
│  ├─ test_retriever_citations.py
│  └─ test_graph_offline.py        # FakeChatModel + FakeEmbeddings 全链路
├─ data/                          # 运行时产物（建议加入 .gitignore）
│  ├─ papers/<paper_id>.pdf
│  └─ index/{store.json,manifest.json}
└─ output/                        # 报告产物（已 gitignore）
```

**可精简版**（若追求最小可行）：把 `rag/` 收敛成 `rag.py`、`tools/` 收敛成 `tools.py`、删除 `agents/rag_agent.py`（由 supervisor 直接检索），文件数从 ~20 降到 ~10，功能不减。

---

## 6. 各功能实现形式

### 6.1 配置与模型层（`config.py` / `llm.py`）

```python
# config.py
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # LLM
    dashscope_api_key: SecretStr | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = Field("qwen3.8-max", alias="DASHSCOPE_MODEL")
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    # 嵌入
    embedding_model: str = "text-embedding-v4"     # 实测 1024 维
    # 检索/规模
    top_k: int = 6
    max_papers: int = 8
    chunk_size: int = 1200
    chunk_overlap: int = 200
    # 路径
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    mcp_servers_file: Path = Path("src/paper_agent/mcp_servers.json")
    # MCP 侧可选密钥（透传给子进程）
    semantic_scholar_api_key: SecretStr | None = None
    unpaywall_email: str | None = None
```

- `.env` **无需改动即可跑通**（现有 `DASHSCOPE_*` / `DEEPSEEK_*` 已实测可用）；建议追加注释性可选项：
  `SEMANTIC_SCHOLAR_API_KEY`、`UNPAYWALL_EMAIL`、`PAPER_AGENT_DATA_DIR`、`LANGSMITH_API_KEY`。
- `llm.py` 使用 `ChatOpenAI` + `SecretStr` + `base_url` 写法，提供：
  - `get_chat_model(role="default" | "search" | "summarize" | "write")` —— 不同角色可给不同 temperature/模型（如检索角色 `temperature=0`，写作 `0.4`）。
  - `get_embeddings()` —— `OpenAIEmbeddings(model="text-embedding-v4", base_url=DASHSCOPE_BASE_URL, check_embedding_ctx_length=False)`；**已实测可返回 1024 维向量**。
  - 统一异常与重试（`tenacity` 指数退避），provider 不可用时报错信息明确（沿用现有 `_require_api_key` 风格）。

### 6.2 MCP 搜索引擎接入（`mcp_servers.json` + `mcp_client.py`）

`mcp_servers.json`（`langchain-mcp-adapters` 支持 `${VAR}` 展开，密钥不落库）：

```json
{
  "arxiv": {
    "transport": "stdio",
    "command": "/home/silva/miniconda3/envs/agent/bin/arxiv-mcp-server",
    "args": [],
    "env": { "SEMANTIC_SCHOLAR_API_KEY": "${SEMANTIC_SCHOLAR_API_KEY}" }
  },
  "paper-search": {
    "transport": "stdio",
    "command": "/home/silva/miniconda3/envs/agent/bin/paper-search-mcp",
    "args": [],
    "env": { "PAPER_SEARCH_MCP_UNPAYWALL_EMAIL": "${UNPAYWALL_EMAIL}" }
  },
  "web-search": {
    "transport": "streamable_http",
    "url": "<按服务商文档填写，例如 https://mcp.exa.ai/mcp?exaApiKey=${EXA_API_KEY}>",
    "enabled": false
  }
}
```

> 注：`${VAR}` 由 adapters 展开为**当前进程**的环境变量值；未设置的变量会**原样保留**字符串，所以 `env` 段只写确实已配置的键（在 `mcp_client.py` 里按需拼装，未配置就省略）。
>
> MCP 传输方式（已核实 0.3.2 支持）：`stdio`（`command`/`args`/`env`）、`streamable_http`（`url`/`headers`，超时 `timeout` + `sse_read_timeout`）、`sse`、`websocket`。推荐 `streamable_http`（`sse` 在 MCP 侧逐步弃用）。

`mcp_client.py` 实现要点（API 已按 0.3.2 核实）：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

ALLOWLIST = {                      # 只放开检索/取全文相关工具，挡住 60+ 工具与 scihub
    "search_papers", "search_arxiv", "search_openalex", "search_semantic",
    "search_crossref", "search_pubmed", "search_europepmc", "search_dblp",
    "download_paper", "download_arxiv", "read_paper", "read_arxiv_paper",
    "get_paper_outline", "read_paper_section", "export_citations",
}
BLOCKLIST = {"download_scihub", "search_google_scholar", "watch_topic", "reindex"}

async def load_search_tools(settings) -> list[BaseTool]:
    client = MultiServerMCPClient(
        connections,                 # 过滤掉 enabled=false 的 server
        tool_name_prefix=True,       # 多 server 同名工具不冲突
        handle_tool_errors=True,     # 工具异常返回错误文本而非中断图
        tool_interceptors=[log_and_timeout_interceptor],
    )
    tools = await client.get_tools() # 也可 get_tools(server_name="arxiv") 按需取
    return [t for t in tools if t.name.split("__")[-1] in ALLOWLIST
                             and t.name.split("__")[-1] not in BLOCKLIST]
```

补充实现形式：
- **异步**：MCP 全异步 → 图用 `await graph.ainvoke(...)`，CLI 用 `asyncio.run(main())`；同步地方用局部 `asyncio.run` 包装（避免嵌套事件循环，统一在 CLI 顶层建 loop）。
- **有状态会话**：批量下载/读取同一 server 时用 `async with client.session("arxiv") as s: await load_mcp_tools(s)`，避免每次工具调用重启子进程（默认 stateless：每次调用新起 session）。
- **中文检索**：`plan` 节点把中文主题翻译/扩展成英文检索式（+ 同义词），因为 arXiv/OpenAlex 对英文关键词更友好。
- **降级链**：MCP 不可用 → `arxiv` pip 包 / OpenAlex REST（可选通道）→ 至少保证 `search`、`ingest` 可跑（用于离线演示与测试）。
- **可观测**：interceptor 里打印 `tool_name`、参数摘要、耗时；`langsmith` 通过环境变量开启（可选）。
- **可用工具自检**：`paper-agent mcp-tools` 打印实际加载的工具名/描述，便于排错。

### 6.3 RAG 文档分析（`rag/*.py` + `tools/*.py`）

| 环节 | 实现形式 |
|---|---|
| 抓取 `fetch.py` | 优先 OA 链接（arXiv/PMC/OpenAlex OA/Unpaywall）；`httpx.AsyncClient` + `tenacity` 重试；落 `data/papers/<paper_id>.pdf` 并缓存（已存在则跳过，记录 sha256） |
| 解析 `parse.py` | 默认 `pdfplumber`（已装）→ 逐页文本 + 页码；若装了 `pymupdf` 则优先（更快）。清洗：去页眉页脚重复行、合并跨行连字符、去参考文献段落（可选） |
| 切分 `split.py` | `RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separators=["\n\n","\n","。","；",". "," ",""])`；每 chunk 元数据 `{paper_id,title,page,chunk_index,section?}` |
| 向量化 `store.py` | `OpenAIEmbeddings(text-embedding-v4)` 批量 `embed_documents`（分批 + 重试） |
| 存储 `store.py` | 默认 `InMemoryVectorStore`，**用内置 `dump(path)` / `InMemoryVectorStore.load(path, embedding)` 持久化**（源码已确认：`dump` 保存 text/metadata/vector 全量，`load` 直接恢复，无需自己写 JSONL/NPY）；`delete(ids=[...])` 做增量更新；`data/index/store.json` + `manifest.json`（记录 paper→chunk 数、embedding 模型名、时间戳、内容 sha256）；接口抽象 `VectorStoreBackend`，大语料可切 FAISS/Qdrant 而不改上层 |
| 检索 `retriever.py` | `similarity_search_with_score(query, k=settings.top_k, filter=lambda d: d.metadata["paper_id"] in paper_ids)`（**已核实：filter 接收 `Document` 可调用对象**，不是 metadata dict）；可选 `max_marginal_relevance_search(query, k, fetch_k, lambda_mult)`；可选 hybrid：`BM25Retriever` + `EnsembleRetriever([bm25, vector], weights=[0.4,0.6])` |
| 引用锚点 | chunk → `[P<idx>:p<page>]`；`Answer` 里保留 `citation_ids`；渲染时映射到参考文献序号（`[1][2]`）+ 文末 source 表 |
| 幻觉校验 | 回答后校验：① 引用编号必须存在于本次检索集合；② 每个引用块的关键片段需在对应 chunk 原文中出现（长度≥20 字符的子串/近似匹配）；不通过则要求模型重写（一次重试） |
| 文档分析输出 | 单篇：`PaperSummary{problem, method, data, findings, limitations, reusable_ideas, key_quotes[]}`（Pydantic，强制结构化）；跨篇：对比表（任务/方法/数据集/指标/结论）+ 共识/分歧/空白点 |

### 6.4 多 agent 分工（`agents/supervisor.py` + 四个专家）

**角色与契约**（`schema.py` 中定义 Pydantic I/O）：

| Agent | 工具 | 输入 → 输出 | 实现 |
|---|---|---|---|
| `search_agent` | MCP 白名单工具 | `query, keywords_en, year_range, max_results` → `PaperList{Paper[]}` | `create_agent(model, tools=mcp_tools, response_format=PaperList, system_prompt=SEARCH_PROMPT)` |
| `summarize_agent` | `search_corpus`（限定单篇）、`read_chunk` | `paper_id, question` → `PaperSummary` | `create_agent(..., response_format=PaperSummary)`，map 阶段每篇一次 |
| `rag_agent` | `search_corpus`、`read_chunk`、`list_indexed` | `sub_question` → `Answer{text, citation_ids}` | `create_agent(..., response_format=Answer)`，用于跨论文归纳 |
| `writer_agent` | 无（输入即上下文） | `summaries[], answers[], papers[]` → `Report{markdown, bibtex, citations}` | `create_agent(..., response_format=Report)` 或纯 prompt chain（更省 token） |

**监督图**（`StateGraph`，状态与并行）：

```python
class ResearchState(TypedDict):
    query: str
    language: str                      # zh / en
    sub_questions: list[str]
    search_queries: list[str]
    candidates: Annotated[list[Paper], operator.add]      # Send 扇出后归并
    selected: list[Paper]
    ingested: dict[str, str]                              # paper_id -> index 路径
    summaries: Annotated[list[PaperSummary], operator.add] # map 归并
    synth: dict[str, Answer]
    report_md: str
    bibtex: str
    retries: int
    notes: Annotated[list[str], operator.add]

graph = StateGraph(ResearchState)
graph.add_node("plan", plan_node)                 # LLM 拆解 + 生成中英检索式
graph.add_node("search", search_node)             # 每源/每 query 一个 Send
graph.add_node("select", select_node)             # 去重(DOI/arXiv ID) + LLM 相关性排序
graph.add_node("ingest", ingest_node)             # 每篇一个 Send（下载/解析/切分/入库）
graph.add_node("summarize", summarize_node)       # map：每篇摘要
graph.add_node("synthesize", synthesize_node)     # reduce：RAG 归纳 + 对比
graph.add_node("write", write_node)
graph.add_node("verify", verify_node)             # 引用/覆盖校验
# 条件边：verify 失败且 retries<1 → synthesize；否则 END
app = graph.compile(checkpointer=InMemorySaver())   # 可选 SqliteSaver 持久化到 data/
```

- **并行**：`search` 节点用 `Send("search", {"q": q, "source": s})` 扇出；`ingest`/`summarize` 用 `Send` 每篇一任务；归并字段用 `Annotated[list, operator.add]`。并发上限用 `asyncio.Semaphore`（默认 4）防限流。
- **检查点**：`InMemorySaver` 即可（简洁）；需要断点续跑时换 `langgraph.checkpoint.sqlite`（可选装 `langgraph-checkpoint-sqlite`）。
- **人工介入（可选）**：`interrupt_before=["ingest"]`，让用户先看候选列表再决定下载哪些（MCP 论文下载有速率限制，很实用）。
- **最简替代实现**（"其他设计相对简洁"的降级路径）：只用一个 `create_agent`，把四个专家用 `@tool` 包成 3 个工具（`delegate_search` / `delegate_rag` / `delegate_summarize`），由主 agent 自行决定调用顺序。代码量约 1/3，但并行与可控性弱 → **默认用监督图，`--simple` 开关切最简版**。

### 6.5 报告输出（`report.py`）

- `output/<YYYYmmdd-HHMM>-<slug>.md`：① 主题与检索式 ② 核心结论（带 `[n]` 引用）③ 论文对比表 ④ 逐篇小结 ⑤ 局限与后续问题 ⑥ 参考文献（arXiv ID/DOI/链接）；表头标注数据来源与生成时间。
- `output/<slug>.bib`：BibTeX（优先用 arxiv-mcp-server 的 `export_citations`，失败则由元数据本地拼装）。
- `output/<slug>.json`：`Paper[]`/`PaperSummary[]`/`Answer[]` 结构化落盘，便于二次分析。
- 复用现有 `output/` 约定；文件名做 slug 安全化（沿用现有 `safe_name` 思路）。

### 6.6 CLI（`cli.py`，typer + rich）

```bash
python -m src.paper_agent report "图检索增强生成在长文档问答中的应用" --papers 8 --out output/
python -m src.paper_agent search "graph RAG" --limit 10 --sources arxiv,openalex   # 仅检索，不出 LLM 摘要
python -m src.paper_agent ingest --ids 2401.00001,10.1145/xxx                     # 建/更新 RAG 索引
python -m src.paper_agent ask "这些方法的主要局限是什么？" --papers 2401.00001
python -m src.paper_agent mcp-tools                                                # 检查 MCP 工具加载
python -m src.paper_agent report ... --simple                                      # 最简单 agent 模式
python -m src.paper_agent report ... --offline                                     # 假 MCP + fake LLM，自检管线
```

统一 `rich` 进度/表格输出；日志分级（`--verbose`）；非零退出码区分「MCP 失败 / 无结果 / LLM 失败 / 校验失败」。

### 6.7 测试与离线可跑（`tests/`）

- `fixtures/fake_mcp_server.py`：用 `mcp` 包写一个 stdio MCP server，暴露 `search_papers` / `download_paper`（返回本地预置 PDF 路径）→ 无网络即可验证 MCP 接入链路（也顺带证明"我们确实走的是 MCP 协议"）。
- `FakeEmbeddings`（hash → 确定性向量）+ `langchain_core.language_models.fake_chat_models` 中的 `FakeListChatModel` / `GenericFakeChatModel`（已核实存在于 langchain-core 1.6.6）→ 图可端到端离线跑，断言状态字段与产物文件。
- 单测：ID 归一化/去重（`arXiv:2401.00001v2` 与 `2401.00001` 视为同一篇）、切分元数据、store 持久化往返、检索按 `paper_id` 过滤、引用校验函数、工具白名单过滤。

---

## 7. 复用清单（避免重复造轮子）

| 需求 | 复用 | 位置 |
|---|---|---|
| OpenAI 兼容模型调用 | `ChatOpenAI` + `SecretStr` + `base_url` 写法 | `src/paper_agent/llm.py` |
| `.env` 加载与 key 校验 | pydantic-settings + `SecretStr` | `src/paper_agent/config.py` |
| 文本切分 | `RecursiveCharacterTextSplitter` | `langchain-text-splitters`（已装） |
| 向量库（零新增依赖） | `InMemoryVectorStore` | `langchain-core`（已装） |
| PDF 解析 | `pdfplumber` | 已装 |
| 混合检索 | `BM25Retriever` + `EnsembleRetriever` | `langchain-community` / `langchain-classic`（已装） |
| Agent/图 | `create_agent`、`StateGraph`、`Send`、`InMemorySaver` | `langchain` 1.4.2 / `langgraph` 1.2.11（已装） |
| 输出落盘约定 | `output/` + 文件名安全化 | `src/paper_agent/report.py` |
| 论文工作流提示词 | `literature_review_prompt` / `compare_papers_prompt` / `summarize_paper_prompt` | `arxiv-mcp-server` wheel 内 `arxiv_mcp_server/prompts/` |

---

## 8. 关键实现骨架（示意，供实现时照此落地）

```python
# src/paper_agent/agents/supervisor.py
async def build_app(settings: Settings):
    mcp_tools = await load_search_tools(settings)
    search_agent = make_search_agent(settings, mcp_tools)
    ...
    g = StateGraph(ResearchState)
    g.add_node("search", make_search_node(search_agent))
    g.add_node("ingest", ingest_node)          # 确定性节点（非 LLM）
    g.add_node("summarize", make_summarize_node(summarize_agent))
    g.add_node("synthesize", make_synthesize_node(rag_agent))
    g.add_node("write", make_write_node(writer_agent))
    g.add_conditional_edges("search", fan_out_search, ["search", "select"])
    g.add_conditional_edges("verify", route_after_verify, {"synthesize": "synthesize", "end": END})
    return g.compile(checkpointer=InMemorySaver())
```

```python
# src/paper_agent/rag/retriever.py 关键片段
def format_context(docs: list[Document]) -> tuple[str, dict[str, Citation]]:
    ctx, cites = [], {}
    for i, d in enumerate(docs, 1):
        cid = f"C{i}"
        cites[cid] = Citation(id=cid, paper_id=d.metadata["paper_id"],
                              page=d.metadata.get("page"), snippet=d.page_content[:200])
        ctx.append(f"[{cid}] ({d.metadata['paper_id']} p.{d.metadata.get('page')})\n{d.page_content}")
    return "\n\n---\n\n".join(ctx), cites

def verify_citations(answer: Answer, retrieved: dict[str, Citation]) -> list[str]:
    """返回问题列表；空表示通过。"""
    problems = [f"引用 {c} 不在本次检索结果中" for c in answer.citation_ids if c not in retrieved]
    for c in answer.citation_ids:
        if c in retrieved and not answer.supports(c, retrieved[c]):   # 片段近似匹配
            problems.append(f"引用 {c} 缺少原文支撑")
    return problems
```

---

## 9. 端到端数据流（一次 `report` 调用）

1. `plan`：中文主题 → 2~4 个子问题 + 3~5 组英文检索式（含同义词）。
2. `search`（MCP，并行）：`arxiv.search_papers` × N + `paper-search.search_papers(sources="all")` → `Paper[]`（含 title/abstract/arxiv_id/doi/pdf_url）。
3. `select`：按 DOI/arXiv ID 归一化去重 → LLM 相关性打分 → 取 Top-K（默认 8）。
4. `ingest`（并行，确定性）：OA PDF 下载（缓存）→ 解析 → 切分 → 批量 embedding → 入库 + 持久化。
5. `summarize`（map，并行）：每篇基于其 chunk 生成 `PaperSummary`，关键结论带 `[C#]` 锚点。
6. `synthesize`（reduce）：`rag_agent` 针对每个子问题检索全语料并回答，产出带引用的 `Answer`。
7. `write`：组装 Markdown 报告 + 对比表 + BibTeX + JSON。
8. `verify`：引用存在性 + 原文支撑校验；失败则回到 6 重试一次，仍失败则在报告中标注"待核实"。

---

## 10. 实施步骤（checklist）

- [x] 1. 装依赖（4.1），`python -c "import langchain_mcp_adapters, mcp"` 自检；`pip check` 无冲突。
- [x] 2. 安装并单独验证两个 MCP server：`python -m arxiv_mcp_server`（能启动、stdio 正常）、`paper-search-mcp`（同上）。
- [x] 3. 建包骨架：`src/paper_agent/`（6 个顶层模块 + `rag/` + `agents/` + `tools/`）。
- [x] 4. `config.py` + `llm.py`，跑通 `get_chat_model().invoke("ping")` 与 `get_embeddings().embed_query("x")`（期望 1024 维）。
- [x] 5. `mcp_servers.json` + `mcp_client.py`：`paper-agent mcp-tools` 列出经过白名单的工具（预期含 `search_papers`）。
- [x] 6. `schema.py` + `utils.py`（ID 归一化/去重/slug）。
- [x] 7. `rag/`（fetch → parse → split → store → retriever）+ `tools/paper_tools.py`、`tools/rag_tools.py`。
- [x] 8. 四个 agent（`create_agent` + 结构化输出）与 `prompts.py`。
- [x] 9. `supervisor.py`：图 + Send 并行 + 条件边 + checkpointer。
- [x] 10. `report.py` + `cli.py`（`search/ingest/ask/report/mcp-tools`，含 `--simple/--offline`）。
- [x] 11. `tests/`：fake MCP server + FakeEmbeddings/FakeChatModel 离线全链路 + 单测。
- [x] 12. `environment.yml` 追加依赖；`data/` 写入 `.gitignore`；README 补使用说明。

---

## 11. 验证方案（每条要求对应可检查证据）

| 要求 | 验证方式 | 期望证据 |
|---|---|---|
| R1 纯 Python | 代码全部 `.py`；MCP server 用 pip console script 启动 | 无 `package.json`/`npx` 必需依赖 |
| R2 LangChain | `create_agent` / `ChatOpenAI` / `InMemoryVectorStore` 均来自 langchain 包 | `pip list` + import 自检 |
| R3 MCP 接入 | `paper-agent mcp-tools`；`paper-agent search "graph rag" --limit 5` | 打印来自 MCP 的真实论文（title/arxiv_id/url）；`tests/fixtures/fake_mcp_server.py` 离线证明协议链路 |
| R4 RAG | `paper-agent ingest --ids <arxiv id>` 后 `ask "主要方法是什么"` | `data/index/manifest.json` 有 chunk 数、`store.json` 可被 `InMemoryVectorStore.load` 恢复；回答含 `[n]` 引用且 `verify_citations` 通过 |
| R5 多 agent | `report` 全流程日志显示各节点执行；`--simple` 对照 | `state["summaries"]` 长度 == 选中论文数；图 trace（或 LangSmith）显示 4 个 agent 被调用 |
| R6 简洁性 | 依赖仅新增 4 个必需包；无外部服务（向量库/DB） | `pip install` 命令仅 2 行；`data/` 全本地文件 |
| 端到端 | `python -m src.paper_agent report "<中文主题>" --papers 5 --verbose` | `output/*.md`、`output/*.bib`、`output/*.json` 三份产物齐全且引用可追溯 |
| 离线可回归 | `pytest -q`（含离线图测试） | 全绿；无网络依赖 |

**冒烟验证已提前完成的部分**：DashScope chat（`qwen3.8-max`）与 `text-embedding-v4`（1024 维）已用现有 `.env` 实测通过，`langchain-mcp-adapters` 在 py3.14 下依赖可解析（`mcp 1.30.0`），两个 MCP server 的 wheel 与 console script 入口已核实。

---

## 12. 风险与备选

| 风险 | 影响 | 对策 |
|---|---|---|
| `mcp` 2.x 与 adapters 不兼容 | MCP 全挂 | 显式 pin `mcp>=1.24,<2.0`；升 adapters 前先跑 `pip --dry-run` |
| Python 3.14 生态新 | 个别包无 wheel | 已验证核心包 abi3/纯 py wheel 可用；避免 `unstructured`；本地 embedding 需 torch（已确认有 cp314 wheel，但体积大 → 默认不用） |
| 工具数量爆炸（paper-search 60+） | 上下文超限、模型乱选工具 | 白名单 + `tool_name_prefix` + 每 server 限工具数 |
| arXiv 限流（406/429）、下载慢 | 检索/入库失败 | 并发信号量（≤4）+ 指数退避 + PDF 本地缓存 + 失败跳过并记录 |
| 付费墙 | 无法解析全文 | 只用 OA（arXiv/PMC/OpenAlex OA/Unpaywall）；默认禁用 `download_scihub`、`search_google_scholar`（合规 + 稳定性） |
| 引用幻觉 | 结论不可信 | 结构化 `Answer` + 引用存在性/片段支撑双重校验 + 报告标注置信度 |
| 云端 embedding 不可用 | RAG 无法建索引 | 备选：`langchain-huggingface` 本地模型（可选依赖）；或退化为 BM25-only 检索 |
| DeepSeek 无 embedding | 备选 provider 不完整 | 模型与 embedding 解耦（`llm.py` 两个工厂函数），embedding 固定走 DashScope/OpenAI 兼容端点 |
| 中文 query 检索英文论文效果差 | 召回低 | `plan` 节点做中→英检索式扩展；检索结果再用中文归纳 |
| MCP server 子进程路径硬编码 | 换机器/换 env 即挂 | `mcp_servers.json` 里的 `command` 支持用 `${PYTHON_EXECUTABLE}` 之类的变量覆盖（默认写当前 env 的 console script 绝对路径）；启动时自检 + 友好报错，并提供 `paper-agent mcp-tools` 排错 |
| `langchain-community` 已进入 sunset | 可选 hybrid 检索的 `BM25Retriever` 未来可能移除 | 先用 `langchain_community.retrievers.BM25Retriever`；若被移除，用 `rank-bm25` 自行包一个 `Retriever`（接口极小） |
| 远程/第三方 MCP 需密钥或 URL 变动 | 接入失败 | 默认只启用无需密钥的 `arxiv` + `paper-search` 两个本地 stdio server；远程 MCP 默认 `enabled: false` |

---

## 13. 待确认的决策（已给默认值，若不同意请指出）

1. **编排方式**：默认「LangGraph 监督图 + 4 专家（含并行 map-reduce）」，并提供 `--simple` 单 agent 降级；是否直接用最简版即可？
   → **已实现两种**（`report` 默认监督图，`report --simple` 单 agent），可自行选择。
2. **向量库**：默认 `InMemoryVectorStore` + 本地 JSON/NPY 持久化（零新增依赖）；是否需要直接上 Chroma/FAISS/Qdrant？
   → 采用了**内置 `dump/load` 的 JSON 持久化**（`data/index/store.json`），换后端只需替换 `PaperIndex` 内部实现。
3. **Embedding**：默认 DashScope `text-embedding-v4`（已验证可用）；是否需要"完全离线"的本地 embedding（需装 torch）？
4. **MCP 服务端**：默认 `arxiv-mcp-server` + `paper-search-mcp`（本地 stdio，pip 安装）；是否需要再加通用 web 搜索 MCP（Exa/Tavily，需额外 API key）？
5. **PDF 来源**：默认只用开放获取（不做 Sci-Hub / Google Scholar 抓取）；是否接受？
   → 已按此实现：`download_scihub` / `search_google_scholar` 进黑名单。
6. **是否要 Web UI**：默认只做 CLI（typer）；如需界面可用 `streamlit`/`gradio` 另开一步（会增加依赖）。
7. **是否需要 `PLAN.md` 之外再产出可运行代码**：本文件目前只含设计方案与骨架示意；确认后可按第 10 节直接实现（预计新增 ~20 个文件）。
   → **已实现**：`src/paper_agent/` 24 个 Python 文件 + `tests/` 6 个测试文件（63 条用例），并已用真实
   MCP + 真实模型跑通端到端报告（见 §14）。

---

## 14. 实现状态与实测回填（执行后）

### 14.1 实施清单完成情况

- [x] 1. 依赖安装（`langchain-mcp-adapters 0.3.2` / `mcp 1.30.0` / `arxiv-mcp-server 0.8.0` / `paper-search-mcp 0.1.4` / `pymupdf 1.28.2`），`pip check` 无冲突
- [x] 2. 两个 MCP server 单独验证：arxiv 19 个工具、paper-search 57 个工具，真实检索调用成功
- [x] 3. 包骨架 `src/paper_agent/`（config / llm / schema / utils / mcp_client / report / cli / rag / tools / agents）
- [x] 4. `config.py` + `llm.py`：`chat OK`（qwen3.8-max）+ `embedding OK dim=1024`（text-embedding-v4）
- [x] 5. `mcp_servers.json` + `mcp_client.py`：`mcp-tools` 显示 76 个原始工具 → 白名单保留 27 个
- [x] 6. `schema.py` + `utils.py`：ID 归一化（arXiv/DOI/OpenAlex/arXiv-DOI）、去重、MCP 结果解析
- [x] 7. `rag/`（fetch → parse → split → store → retriever）+ `tools/paper_tools.py`、`tools/rag_tools.py`
- [x] 8. 四个 agent（`create_agent` + 结构化输出 + 兜底路径）与 `prompts.py`
- [x] 9. `supervisor.py`：StateGraph + `Send` 扇出 + barrier + 条件重试 + `InMemorySaver`
- [x] 10. `report.py` + `cli.py`（search / ingest / ask / report / mcp-tools / selftest，含 `--simple/--offline`）
- [x] 11. `tests/`：63 条用例（含假 MCP server 的真实协议往返、离线全链路图）
- [x] 12. `environment.yml` / `requirements.txt` / `.gitignore` / `README.md` 更新

### 14.2 实测中暴露的坑（已在代码中处理，方案据此修订）

| 现象 | 根因 | 处理 |
|---|---|---|
| `create_agent(response_format=...)` 报 400 `tool_choice ... in thinking mode` | Qwen3 思考模式不支持 `tool_choice=required/object` | `get_chat_model()` 默认注入 `extra_body={"enable_thinking": False}`，可用 `DASHSCOPE_ENABLE_THINKING=1` 打开 |
| embedding 400 `batch size is invalid, it should not be larger than 10/20` | DashScope 单批上限（随模型不同） | `embed_batch_size=10` + `RetryingEmbeddings` 自适应二分拆批 |
| embedding 400 `Receive batching backend response failed!` | 后端偶发错误 | `RetryingEmbeddings` 指数退避重试（1s/2s/4s） |
| 部分论文 `pdfplumber` 输出乱码/串行 | pdfplumber 对复杂版式不稳 | 解析优先 `pymupdf`，`pdfplumber` 回退（质量差异已实测对比） |
| 条件边直接从 worker 出发导致下游任务成倍重复 | `Send` 扇出后条件函数按 worker 实例求值 | 每次扇出后加 **barrier 节点**（`collect_ingest/collect_summarize/collect_answers`），由 barrier 再扇出 |
| `verify_issues` 并发写入报错 | 多个 worker 同时写普通通道 | worker 只写带 reducer 的字段（`worker_errors: Annotated[list, operator.add]`），`verify` 再汇总 |
| 模型把 `[Q0-C1]` 简写成 `[C1]` | 提示词只给了 `[C#]` 形式 | 提示词要求原样复制 + `normalize_answer_citations()` 在前缀唯一可推断时自动补回 |
| 中文回答引用英文原文时支撑度恒为 0 | 字面重合度对跨语言无效 | 增加“硬令牌”（术语/数字，如 `HotpotQA`、`7.1`）判定；无可核验令牌时跳过并记录，不误报 |
| 一句标注多个来源时误报 | 逐引用独立校验 | 同句中任一被引来源能支撑即算通过 |
| `10.1109/TKDE.2024.1234567` 被误判成 arXiv | arXiv 正则过宽 | DOI 前缀优先判定 + arXiv 自身 DOI（`10.48550/arXiv.*`）反向映射 |
| `--offline` 与真实索引混用导致维度错乱 | 假 embedding 维度 ≠ 真实 | 索引写入 `embedding_signature`，不匹配抛 `IndexSignatureError`；`--offline` 自动用 `data/offline/` |

### 14.2b 交互式入口（追加需求）

按「使用 main.py 封装以上功能，做成 pi 那样交互式命令行」的要求补充：

- `main.py`：REPL（`paper-agent › ` 提示符）+ 一次性模式（`--search/--ingest/--report`）；
  斜杠命令 `/search /ingest /ask /report /papers /index /mcp /model /offline /stream /history /save /clear /exit`；
  stdlib `readline` 提供 Tab 补全与历史（`~/.paper_agent_history`）；`rich.Live` + Markdown 做**流式渲染**；
  回答结束后立即输出引用列表与校验结论；单条命令异常/Ctrl-C 不退出会话；非 TTY 时逐行读 stdin（可脚本化）。
- `src/paper_agent/pipeline.py`：把 `run_search / run_ingest / ask / run_report` 从 CLI 抽成公共函数，
  `cli.py` 与 `main.py` 共用，避免两套逻辑漂移；MCP 工具在会话内缓存复用（client 无状态，可安全复用）。
- 验证：`python main.py --help`、管道模式 `/index` `/papers` `/help` `/search`、流式一次性提问（真实模型，
  引用校验通过）、`main.py --offline` 下的 `/report`、以及重构后的 CLI 全命令复跑（selftest/search/ingest/ask/report）。
  测试仍为 **63 passed**。

### 14.2c 供应商配置（追加需求：/connect + JSON + 多供应商 + /models）

要求「打开 main.py 后输入 `/connect` 设置配置，存 JSON，统一 OpenAI 兼容格式，输入 base URL + api key，
自动识别多家供应商，自动识别模型列表并可用 `/models` 切换，Ctrl+C 设默认模型」。实现：

| 组件 | 位置 | 说明 |
|---|---|---|
| 供应商配置 | `src/paper_agent/userconfig.py` | `~/.config/paper-agent/config.json`（0600，可用 `PAPER_AGENT_CONFIG` 覆盖）；`detect_provider()` 按 base URL 识别 dashscope/deepseek/openai/moonshot/siliconflow/zhipu/volcengine/openrouter/local/通用；`classify_models()` 把 `/models` 结果分为对话/embedding；`settings_overrides()/apply_to()` 注入 `Settings`（JSON 优先于 .env，可 `PAPER_AGENT_IGNORE_USER_CONFIG=1` 关闭） |
| 交互界面 | `src/paper_agent/tui.py` | `/connect` 预设+问答流程（key 用 getpass 不回显）；`/models` 选择器：↑/↓、输入即过滤、**Ctrl+C 设为默认**、Enter 本次会话、Esc 取消；非 TTY 退化为编号输入（`d3`=设默认） |
| Settings | `config.py` | 新增 `llm_base_url/llm_api_key/llm_model/llm_kind/llm_label` 与独立 embedding 供应商 `embed_base_url/embed_api_key/embed_model/embed_label`；`get_settings()` 自动叠加用户 JSON |
| 模型层 | `llm.py` | 通用分支优先于 .env 专用分支；DashScope 才注入 `enable_thinking=False`；embedding 支持「对话 A + embedding B」，全无 embedding 时给出可执行建议 |
| 索引隔离 | `pipeline.build_session()` | embedding 签名变化（换供应商/模型/维度）时自动改用 `data/by-embedding/<签名>/`，避免维度混用 |
| 子进程静音 | `mcp_quiet/sitecustomize.py` + `mcp_client.quiet_env()` | MCP server 的 INFO/WARNING 不再刷屏（交互式 `/search` 噪声从数十行降到 0） |

实测：`/connect 1 --key <dashscope>` 识别 DashScope → 262 模型（对话 190 / embedding 2）→ 探测 `qwen3.7-text-embedding` 维度 1024 → 入库 → 带引用问答通过；
`/connect 2 --key <deepseek>` 识别 DeepSeek → `/models` 编号列表 `d2` 设为默认 → 对话用 deepseek-v4-pro、embedding 自动回退 DashScope → 问答引用校验通过；
无任何密钥时 REPL 不崩、提示 `/connect`，且 `/search` 免密钥可用。测试总数 63 → **116 passed**。

### 14.2d 可移植性（追加需求：补齐 .gitignore 与 requirements.txt）

| 交付物 | 内容 | 验证方式 |
|---|---|---|
| `requirements.txt` | 在**干净 venv** 里从 PyPI 安装运行必需包后 `pip freeze` 生成：129 个运行时包 + 9 个开发包，全部精确 pin、无 `@ file://` 本地路径；含可选增强（注释形式）与迁移说明；显式列出 `numpy`（隐式依赖）与 `mcp==1.30.0`（`mcp<2` 约束） | 新建 venv → `pip install -r requirements.txt` → `pytest` **126 passed**、`mypy` 零 error、`selftest`（真实 chat/embedding）、`mcp-tools`（76→27）、`main.py` 交互式 `/ask` 引用校验通过 |
| `.gitignore` | 完整版：密钥与本地配置（`.env*`，保留 `!.env.example`）、**测试文件（`tests/` + `test_*.py` + `conftest.py`）**、Python 编译/打包产物、各类工具缓存（`.pytest_cache`/`.mypy_cache`/`.ruff_cache`/coverage/hypothesis）、虚拟环境、运行产物（`data/`/`output/`/`logs/`）、Notebook/编辑器/系统文件 | `git check-ignore -v` 逐项验证；源码/文档/模板仍为 tracked |
| `.env.example` | 环境变量模板（三种 provider 方案 + MCP 可选 key + 全部运行参数），不夹带任何密钥 | 被 `!.env.example` 例外规则保留在版本库内 |
| `environment.yml` | 加头部说明指向 requirements.txt（跨机器移植用它），补 `mypy`、显式 `numpy` 与 `mcp<2` 提示 | 保持与本机 conda 环境一致 |

> 实测收益：正是这次「全新环境安装 + 跑测试」的验证发现了 `numpy` 缺失（`langchain-core` 的
> `InMemoryVectorStore` 计算余弦相似度需要它，但未声明为硬依赖），随后补进 requirements.txt。

### 14.2e environment.yml 简化

原本是 `conda env export` 的原始快照（112 行，含 `_libgcc_mutex`/构建号/`prefix:` 等机器相关项，跨机器无法复用）。
现改为**只列 22 条直接依赖**的 conda 环境定义（49 行，含 `python>=3.11` + `pip` 分组注释），精确版本与全部传递依赖交给 `requirements.txt`。

验证：
- `conda env create -f environment.yml -n pa-verify -d` → 退出码 0，求解出 33 个 conda 包（`DryRunExit`），未创建任何环境；
- YAML 解析正确（`name=agent`，`channels=[defaults]`，22 条 pip 依赖）；
- 一致性校验：`environment.yml` 里每条 pip 依赖都能在 `requirements.txt` 找到满足其约束的精确版本（22/22 ✓），避免两份清单打架。

### 14.2f 修复：`/connect` 无法粘贴 API key

- 根因：`_ask_api_key` 用 `getpass.getpass()` —— 它关回显但**不走 readline**，终端为粘贴插入的
  括号粘贴标记（`\x1b[200~…\x1b[201~`）会被当作内容读入，key 里混入转义序列 → 401，且无回显无法察觉。
- 修复：
  1. `tui.read_secret()`：只关闭 `ECHO`、**保留 `ICANON`**（行缓冲，终端自己处理粘贴与行编辑），
     读完在 `finally` 里恢复终端属性；非 TTY 时退回 `input()`；
  2. `utils.clean_pasted()` / `clean_secret()`：剥离 CSI/OSC 转义（含括号粘贴标记）、零宽字符、
     控制字符、外层引号、`Bearer ` 前缀，多行粘贴只取首行；密钥额外去除内部空白；
  3. `utils.mask_secret()`：读入后立即回显脱敏结果（含长度），用户可当场确认粘贴是否完整；
  4. `normalize_base_url` / `_split_args`（`/connect <url> <key>`、`/model <name>` 等）一并清洗，
     避免 URL、模型名被标记污染；`/models` 失败时打印实际请求的 URL 与脱敏 key，并针对 401 给出粘贴提示。
- 验证：
  - **真实 PTY** 跑 `main.py`：在 `/connect` 提示符下用 `\x1b[200~<真实 key>\x1b[201~\r` 模拟粘贴 →
    终端回显 `已读取 key：sk-w…OssD（117 位）`，落盘 JSON 与 `.env` 中 key **完全一致**、无任何转义/空白杂质，
    随后成功拉取 262 个模型并探测 embedding 维度 1024；
  - `tests/test_paste.py`（17 条）：`clean_pasted/clean_secret/mask_secret` 各种粘贴形态、
    **PTY 下的括号粘贴**、读取后 `ECHO`/`ICANON` 属性复原、非 TTY 回退、`/connect` 端到端粘贴输入落盘校验。

### 14.2g 补齐：内置联网抓取（免 MCP / 免 key）+ 按 ID 直抓

问题（用户提出「这个项目没有网络论文抓取功能吗」）暴露出两个真实缺口：
① `--ids` 仍需先搜索才可能入库（`run_ingest` 无条件先 `run_search`）；
② 检索/抓取完全依赖 MCP server，未安装或报错时**没有任何 HTTP 回退**，也没有「按 DOI/arXiv ID 直接抓」的能力。

新增 `src/paper_agent/sources.py`：
- **arXiv Atom API**：`search_arxiv`（关键词）+ `resolve_arxiv`（id_list 精确查询）；自建查询构造 `build_arxiv_query`
  （把 LLM 的布尔串转成 `all:"phrase" AND all:kw`，首轮 0 命中自动放宽重试）；
- **OpenAlex**：`search_openalex` / `resolve_doi_openalex`，解析 `abstract_inverted_index`，
  取 `best_oa_location.pdf_url` 作为开放获取 PDF 直链；
- **Crossref**：`search_crossref` / `resolve_doi_crossref`（DOI 元数据兜底，JATS 标签与 HTML 实体清洗）；
- **统一入口**：`classify_identifier`（arXiv/DOI/链接/未知）、`resolve_identifier`、`resolve_ids`（并发+去重+失败清单）、`builtin_search`；
- HTTP 层：UA/超时/重试（429、5xx、超时重试并尊重 `Retry-After`；**4xx 不重试**）+ arXiv 3s 节流。

接入：
- `pipeline.run_search(source=auto|mcp|builtin)`：auto = MCP →（不可用/0 条）内置回退；
- `pipeline.run_ingest`：给 `--ids` 时**直接按 ID 解析**（不再需要关键词，也不依赖 MCP）；
- `cli`：`search --source`、`ingest [query] 可为空 + --source`、`report --source`；
- `main.py`：`/search --source builtin`、`/ingest --ids ...`（query 可省）；
- 图内 `search_one` 节点同样接入回退，且 `--source builtin` 时**不加载 MCP**（没装 MCP 的机器可跑通整条链路）；
  离线模式（`deps.offline`）严格不联网。

顺带修掉两个真 bug：
- `_request` 会把 4xx（如 404/401）也重试 3 次 → 改为只重试 429/5xx/网络异常；
- `pipeline` 四个入口用 `settings or session.settings`，而 `build_session` 可能因 embedding 签名变化把索引目录
  重定向到 `data/by-embedding/<sig>/`，导致 `report` 又拿旧目录建索引并抛 `IndexSignatureError`
  → 改为 **session 优先**（用户 `/connect` 换 embedding 模型后 `report` 才正常工作）。

验证：`tests/test_sources.py`（32 条，全部离线：Atom/OpenAlex/Crossref 解析、本地假服务上的检索与重试、
`resolve_ids` 去重与失败、`builtin_search` 部分失败、pipeline 三种 source 模式、按 ID 入库不触发搜索）；
真实网络实测：`search --source builtin`（arXiv+OpenAlex+Crossref 各 3 条）、
`ingest --ids arxiv:2405.16506`（直抓 879KB PDF → 13 页 56 chunks 入库）、
`report --papers 1 --source builtin`（**全程不加载 MCP**：5 条检索式走 openalex/crossref → 26 候选 → 1 篇 → 摘要 → 问答 → 报告三件套）。

### 14.2h 追加需求：检索直接入库 / 删除论文 / 可配置搜索渠道 / 删 key / 检索用 LLM / 180s 超时

本轮按使用反馈追加六项能力（全部在 `main.py` 交互界面 + `cli.py` 同步提供）：

| # | 需求 | 落点与用法 | 关键实现 |
|---|---|---|---|
| 1 | `/search` 加特定参数直接入库 | `/search <query> --ingest [N]`（别名 `--save`/`--index`；CLI：`search ... --ingest N`） | `pipeline.ingest_papers()` 抽成公共函数：把**刚检索到的 `Paper` 直接**下载/解析/入库，不再二次检索（原 `run_ingest` 也改用它） |
| 2 | 交互界面删除库中论文 | `/papers rm <id>[,<id>]` / `/papers rm --all` / `/papers rm`（弹选择器）；CLI：`rm <ids\|all>` | `PaperIndex.delete_paper()` 删 chunks + manifest + 本地 PDF 缓存；`pipeline.remove_papers()` 归一化 ID 后调用 |
| 3 | 增加可配置搜索渠道 | `/channels`（`add/rm/key-rm/on/off/list`）；CLI：`channels ...` | 新增 `channels.py` 注册表 + `userconfig.SearchChannel`；渠道写入 `~/.config/paper-agent/config.json` 的 `channels`，经 `settings_overrides` 注入 `Settings.search_channels`；`sources.py` 实现 Europe PMC / PubMed / Semantic Scholar / DOAJ / CORE / Tavily / Exa / SerpAPI |
| 4 | 删除供应商/渠道的 API key | `/keys rm provider:<name>\|channel:<name>`、`/providers key-rm <name>`、`/channels key-rm <name>`；CLI：`keys --rm ...` | `UserConfig.remove_provider_key()` / `remove_channel_key()`：只置空 key，**保留**供应商/渠道配置与已选模型 |
| 5 | 检索过程中使用 LLM | 默认开启；`/search ... --no-llm`（或 `--llm`）可关/强制；CLI：`search --no-llm` | 新增 `search_llm.py`：`expand_queries()` 查询扩展（中→英+同义）→ 多条检索式并发 → `rank_papers()` 相关性重排；失败/超时静默降级为原始行为 |
| 6 | 等待响应 ≤ 180s | `.env`：`LLM_TIMEOUT=180`、`SEARCH_TIMEOUT=180`（默认值） | `ChatOpenAI(timeout=180, max_retries=1)`、`OpenAIEmbeddings(request_timeout=180)`；`run_search` 整体 `asyncio.wait_for`、`ask`/流式回答/agent 调用（search/summarize/rag/writer）统一 180s 上限 |

**搜索渠道实测与修正（关键反馈：「好多渠道不能用」）**：以下为真实网络实测结论，并据此调整默认值：

| 渠道 | 实测 | 处理 |
|---|---|---|
| arXiv / OpenAlex / Crossref | ✅ 免 key 可用 | 默认启用 |
| Europe PMC | ✅ 免 key 可用（含 OA PDF 直链） | **默认启用**（`BUILTIN_SOURCES` 增加） |
| PubMed | ✅ 免 key 可用（esearch→esummary→efetch 取摘要） | **默认启用** |
| DOAJ | ✅ 免 key 可用 | **默认启用** |
| Semantic Scholar | ❌ 无 key 固定 429（共享池限流，实测连续 4 次均 429） | 改为 `needs_key=True`（`/channels add semanticscholar --key ...`）；不再默认启用 |
| DBLP | ❌ dblp.org 有 bot 防护（返回 “Making sure you're not a bot!”） | 从注册表与检索分发中**移除**（保留解析器备查） |
| CORE / Tavily / Exa / SerpAPI | 需 API key | 保持按需配置，未配 key 时给出明确提示而非静默失败 |

**MCP 侧同类问题也已收敛**（此前只在内置源层处理，MCP 仍会刷 429）：
- 未配 `SEMANTIC_SCHOLAR_API_KEY` 时，`filter_tools(settings=...)` 把 `search_semantic` 并入 keyless 黑名单，不再暴露给模型；
- `search_papers` 装上参数护栏 `guard_search_sources`：缺省/`all` → `Settings.mcp_sources`（默认 `MCP_DEFAULT_SOURCES`，配 key 后才追加 `semantic`），显式请求里也会在无 key 时剔除 `semantic`/`semanticscholar`；
- 工具签名无 `sources`（如 `arxiv_search_papers`）时护栏原样透传。

新增 `--source all`：MCP 与内置源/已启用渠道**合并**（不再因为 MCP 有结果而跳过用户配置的渠道）。
另外 `builtin_search` 现在会把失败源用短标签写进路由（如 `semanticscholar(429)✗`、`core(需key)✗`），
便于一眼看出「哪个渠道没生效、为什么」，而不是默默少结果。

**验证**：`tests/test_channels.py`（41 条，全离线）：渠道注册表、`SearchChannel` JSON 往返、key 只删不删配置、
`settings_overrides` 注入、Europe PMC/PubMed/Semantic Scholar/DOAJ/CORE/网页结果的解析器、
`builtin_search` 纳入已启用渠道、`search_llm` 扩展/重排与超时降级、`pipeline.run_search` 的 LLM 扩展+重排、
`ingest_papers`/`remove_papers`、REPL 的 `/papers rm`、`/channels add|key-rm`、`/providers key-rm`、`/keys rm`。
真实网络实测：`builtin_search` 默认 6 源并发返回 48 条（`内置: arxiv(8),openalex(8),crossref(8),europepmc(8),pubmed(8),doaj(8)`）。
全量回归：**229 passed**（另 2 条 `tests/test_paste.py` 失败为改动前既有问题，已用 `git stash` 对照确认与本次无关）。

### 14.2i 新增需求：国内数据库（万方 / 百度学术 / ChinaXiv）

目标：把中文文献纳入同一套 `Paper` → 下载/入库/问答链路，但**不假装有接口、不绕过平台限制**。

| 渠道 | 能力 | 鉴权 | 实现 |
|---|---|---|---|
| `chinaxiv` | 中文预印本（ChinaXiv 语料，官方 `source_url` 指回 chinaxiv.org） | 免 key；可选 `X-API-Email` 进 polite pool | `parse_chinaxiv` + `search_chinaxiv`，GET `chinarxiv.org/api/v1/papers`（`source=chinaxiv` 只取中文语料） |
| `baidu_scholar` | 中英文期刊/会议/学位论文 | 千帆 `Authorization: Bearer <key>` | `parse_baidu_scholar` + `search_baidu_scholar`，GET `qianfan.baidubce.com/v2/tools/baidu_scholar/search`（`wd`/`pageNum`/`enable_abstract`） |
| `wanfang` | 中文期刊/学位/会议论文 | 开放平台 `X-Ca-AppKey` + `Authorization: APPCODE` | `parse_wanfang` + `search_wanfang`，POST（`query`/`page`/`pageSize`），`api_key` 支持 `AppKey:APPCODE` 两段式；`base_url` 可按订阅覆盖 |
| `nlc` | 国家图书馆联合目录：图书/古籍/学位论文书目 + 馆藏 | 免 key | `parse_nlc` + `search_nlc`，POST `meta.nlc.cn/v2/doSearch`（MARC 风格字段 `TIT/AUT/PUB/YEA/ISB/CLC/SUB/holdings`）；后台只支持按字段检索（`ANY` 无效），故并发 `TIT` + `SUB` 再合并去重 |

- 新增渠道分组 `channels.GROUP_CN`（`list_specs("cn")`），并加入 `PRESET_ORDER`，`/channels` 交互可直接选。
- 万方响应字段做了中英文兼容（`title/题名`、`作者/authors`、`摘要/abstract`、`期刊/journal`、`被引/citations`），
  容器兼容 `data` 为 list 或 `{list/rows/records/results}`。
- **知网 / 维普 / 超星**：无公开检索 API，仅写入 README「已知限制」，不做验证码绕过/批量抓取。
- 真实网络实测：`search_chinaxiv("graph neural network", 3)` → 3 条，`paper_id` 形如 `chinaxiv:202609.00218`，
  `pdf_url` 与官方 `source_url` 均可用；`search_nlc("深度学习", 4)` → 4 条相关书目。
- **踩坑记录**：国家图书馆 `doSearch` 对**原始 UTF-8 中文请求体**会忽略检索词、直接返回全库默认结果
  （`json.dumps(..., ensure_ascii=False)` 复现），改用 ASCII 转义 body 即正常；故 `_post_json` 新增
  `ensure_ascii` 开关，并有回归测试 `test_nlc_search_uses_ascii_body` 钉住。
- 离线单测（`tests/test_channels.py` 新增）：注册表/分组、三个解析器 + `nlc` 解析器、无 key 渠道报错、
  国图 ASCII body 回归；全量 **240+ passed**。

### 14.2j 追加需求：检索时并发跑全部已注册渠道

目标：一次检索把**所有已注册渠道各自独立、同时**请求一遍，而不是只跑默认 `builtin_sources`。

- 新增 `Settings.search_all_channels`（`SEARCH_ALL_CHANNELS=1`）；REPL `/channels all on|off`、CLI `channels all on|off`，
  写入 `~/.config/paper-agent/config.json` 的 `search_all_channels`，经 `settings_overrides` 叠加（只在开启时覆盖，不盖掉 env）。
- `builtin_search` 源解析优先级：显式 `--sources a,b` > `all` / `search_all_channels` > `BUILTIN_SOURCES`；
  已通过 `/channels` 启用的渠道总会追加；`/channels off` 的渠道一律不跑。
- 全渠道模式下仅跑「可用」渠道：免 key 的一律跑；**需 key 但未配 key 的自动跳过**并在路由标 `(需key)⊘`
  （以前会逐条抛 `需key✗`，现在不再刷屏）。
- 并发控制：每个渠道独立 `_post_json`/`_request`，用信号量 `CHANNEL_CONCURRENCY`（默认 6）限流，
  单渠道 `asyncio.wait_for(..., CHANNEL_TIMEOUT=25s)`，个别慢源不再拖垬整次检索。
- `registered_sources()` 运行时读 `CHANNEL_SEARCHERS`（便于测试与后续新增渠道）。
- 真实网络实测：`builtin_search("graph neural network", 3, search_all_channels=True)` →
  `内置: arxiv(3),openalex(3),crossref(3),europepmc(3),pubmed(3),doaj(3),chinaxiv(3),nlc(3),
  semanticscholar(需key)⊘,core(需key)⊘,baidu_scholar(需key)⊘,wanfang(需key)⊘,tavily(需key)⊘,exa(需key)⊘,serpapi(需key)⊘`，24 条。
- 新增离线单测：`test_all_channels_concurrent_and_skips_keyless`、`test_registered_sources_covers_cn`、
  `test_search_all_channels_toggle_roundtrip`。

### 14.2k 修复：`/` 动态补全失效 + 供应商删除

**动态补全一直没有生效的根因**（前几轮只改了匹配算法，没找到真因）：
`create_session()` 同时设了 `complete_while_typing=True` 与 `enable_history_search=True`，
而 prompt_toolkit 在 `enable_history_search` 开启时会**强制关闭** `complete_while_typing`
（`prompt_toolkit/shortcuts/prompt.py` 里的 `Condition(...)`）——所以输入 `/`+字母永远不弹菜单。
定位方法：用 `pty` 跑真 `main.py` 复现无菜单，再用 pipeline/pipe-input 逐参数二分，
`enable_history_search=True` 是唯一让菜单消失的参数。修复：去掉该参数（方向键仍可翻历史）。
新增回归：`session.default_buffer.complete_while_typing() is True` 且 `enable_history_search() is False`。

**供应商删除**：
- REPL `/providers rm [name]`：不给名字时弹选择器，删除前二次确认（`y/N`），删完重建会话；
  别名 `rm/remove/del/delete`；usage 与补全同步。
- CLI 新增 `paper-agent providers [list|use|rm|key-rm|sync]`（之前只有 `keys --rm` 能删 key）。
- 测试：`test_rm_with_confirmation` / `test_rm_can_be_cancelled`（REPL）、`TestDynamicCompletion`。

### 14.2l 逐渠道检索进度（不合并显示）

目标：检索时把**每个渠道的进度单独列出**，而不是最后合计成一行 `内置: a(3),b(2),…`。

- `sources.builtin_search(..., on_event=None)` 新增进度回调：每个渠道在上报
  `queued → running → (done|failed|skipped)` 时回调一次，字段含 `name/status/count/elapsed/error/reason`；
  回调异常被吞掉，绝不影响检索。
- `pipeline.run_search/_run_search_impl/_search_once` 逐层透传 `on_event`；headless（report/监督图）不传则行为不变。
  `_builtin()` 只透传“有值”的参数（兼容只接受 `(query, limit, settings)` 的旧替身）。
- REPL `cmd_search` 用 `main._SearchProgressView` + Rich `Live` 渲染表格（一行一渠道，`transient=False` 保留）；
  无事件时退化为单行“联网检索中”；状态配色：检索中=cyan、完成=green、失败=red、跳过=yellow。
- 测试：`TestChannelProgress`（事件序列/失败事件/跳过）+ `TestSearchProgressView`（按渠道聚合渲染）。
- 真终端实测：`/search graph rag --source builtin --sources arxiv,openalex,crossref,nlc --limit 2 --no-llm`
  输出逐行 `arxiv/openalex/crossref/nlc/europepmc/tavily` 的 `完成 N 耗时`。

### 14.2m 优先使用国内渠道

- `channels.py`：`is_domestic(kind)` / `domestic_kinds()` / `domestic_first(kinds)`（稳定排序，国内在前）；
  `chinaxiv` / `nlc` 标 `builtin=True`（免 key，默认启用）。
- `Settings.builtin_sources` 默认改为 `nlc,chinaxiv,arxiv,openalex,crossref,europepmc,pubmed,doaj`（国内在前）；
  新增 `Settings.prefer_domestic=True`（`PREFER_DOMESTIC=0` 可关）。
- `sources.builtin_search`：默认/all 模式下 `wanted = domestic_first(wanted)`（显式 `--sources` 尊重用户顺序）；
  返回前 `papers_domestic_first(papers)`；进度事件带上 `kind`，UI 对国内渠道标 `·国内`。
- `pipeline._run_search_impl`：合并+重排后、截断前 `papers_domestic_first(merged)`，保证 `--limit` 优先保留国内结果。
- 开关：`/channels domestic on|off`（REPL）、`paper-agent channels domestic on|off`（CLI），
  持久化到 JSON 的 `prefer_domestic`（默认 True；只在关闭时覆盖，不盖掉 env）。
- 测试：`TestDomesticPriority`（helpers / 结果排序 / prefer_domestic=off / 持久化）；
  真终端实测进度表顺序：`nlc ·国内 → chinaxiv ·国内 → arxiv → openalex → …`。

### 14.2n `/search --limit` 改为「每渠道上限」

- 语义：`/search <q> --limit N` 表示**每个渠道最多 N 条**；结果为各渠道汇总去重，不再截断到 N 条。
- `pipeline.run_search(..., per_source_limit=False|True)`：
  - `True`（`/search` 用）：`per_query = limit`（每渠道每条检索式取 N），汇总后只受 `Settings.max_total_results`（默认 200）兜底；
  - `False`（`/ingest`、`/report`、监督图等）：保持原行为 —— 每渠道多取一些，最后截断到 `limit`。
- `_search_once(..., truncate=)`：`truncate=False` 时不在单条检索式层按 `limit` 截断（之前这里会把每渠道结果又砍回 N）。
- 新增 `Settings.max_total_results`（`MAX_TOTAL_RESULTS=200`）。
- **结果按渠道分别展示**（不是把各渠道合并/相乘成一张大表）：
  `builtin_search(..., by_channel={})` 按 `kind` 分别收集结果，`run_search` 透传；
  `cmd_search` 逐渠道渲染小表（国内在前、标题带 `·国内`，每渠道内去重、最多 `limit` 条）；
  `--ingest` 仍用合并去重后的 `papers`。
- 测试：`TestPerSourceLimit`（per_source=True 不被 limit 截断但受总上限兜底；默认仍按总量截断）、
  `TestByChannelResults`（按渠道收集 + pipeline 透传）。
- 真终端实测：`/search graph rag --source builtin --limit 2 --no-llm` →
  `国家图书馆（图书馆检索） ·国内（2 篇）` → `ChinaXiv 预印本 ·国内（2 篇）` → `arXiv（2 篇）` → …，
  进度表 8 渠道各「完成 2」。

### 14.2o 修复：问答答案「反复输出」

两个独立原因，都已处理：

1. **流式渲染往滚动区重复打印**：`main._stream_renderer` 原来用
   `Live(vertical_overflow="visible")`。长回答超出屏幕后，Rich 无法就地重绘，每次
   `live.update()` 都会在上方再写一份完整 Markdown → 看起来就是“反复输出”。
   改为 `vertical_overflow="crop"` + `transient=True`（只在可视区重绘，结束时清掉重绘区），
   `cmd_ask` 无论流式与否都只 `console.print(Markdown(...))` 一次。
2. **「累计全文式」流 + 模型退化复读**：
   - `_stream_answer` 现在区分 delta 与**累计全文**两种流（`piece.startswith(full)` → 只取增量），
     避免把累计快照直接 append 成复读；`_chunk_text` 还兼容 content 为字符串/内容块列表。
   - 新增 `collapse_repetition()`：折叠紧邻且完全相同的段落/句子（保守，不动正常并列句式），
     在 `_stream_answer` 与 `ask()` 里都应用。
3. **实时复读检测（关键）**：前面的修复只解决了“显示重复”与“相邻去重”，不能阻止模型真的反复生成。
   新增 `RepetitionGuard` + `truncate_repetition()`：
   - `_period_repetition_start`：末尾同一块**连续**出现 ≥3 次（≥40 字长块只需 2 次）即判为复读；
   - `_rolling_repetition_start`：同一 32 字窗口出现 ≥3 次（施住跨句/带间隔的复读）；
   - `dedupe_repeated_blocks`：把段落切成序列，删除“之前在输出里出现过的**最长连续段落块**”，
     专门治 `[A,B,A,B,A,B,…]` 这种“小节标题+正文”整块复读（只去相邻是抳不住的）；
   - 只扫描末尾 1500 字（复读总在末尾），成本可控；命中后截断到第一份副本。
   - `_stream_answer` 里每累积 ≥24 字检测一次，命中则 `break` 并 `aclose()` 流，**立即停止生成**；
     流结束后再跑 `collapse_repetition` 兼底。
- 测试：`TestStreamAnswer`（累计流不重复、增量流正常、**复读流被停止并截断**，
  `TestRepetition`（相邻复读/长段复读被截断、正常文本不变、guard 触发）。

### 14.2o2 修复（真因）：流式输出改为 append-only，不再用 Live 重绘

- 现象：回答较长时出现“相同内容重复 N 遍”，且与**终端滚动**强相关。
- 根因：`main._stream_renderer` 用 `Live.update(Markdown(全量文本))` 反复重绘整段；
  内容超过一屏后终端滚动，Rich `Live` 会把已显示过的内容再打一遍（`vertical_overflow` 设为
  `visible` 时尤其明显；`crop` 也无法在所有终端上完全避免）。
- 修复：流式改成 **append-only** —— `console.print(token, end="", markup=False, soft_wrap=True)`，
  只追加、不重绘，结构上不可能重复；`cmd_ask` 流式时**不再重打最终 Markdown**（只补一个换行），
  非流式（`/stream off`）才渲染 Markdown。代价：流式期间看到的是纯文本（`**` 等标记不渲染）。
- 用 pty 实测 40 段长文逐字输出：每段恰好出现 1 次（不再重复）；Live 重绘版则可能出现重复。
- 测试：`TestStreamRendererAppendOnly`（追加一次、引用锤子保留、不启动 Live、空 token 忽略）。

### 14.2p 修复：`Event loop is closed`

- 现象：第二条问答/检索开始时报 `流式生成失败：Event loop is closed`（流式内容为空）。
- 根因：`main.py` 每条命令都 `asyncio.run(...)`，它会**新建并关闭**一个事件循环；而 `Session.model`
  是复用对象，其内部 httpx `AsyncClient`/连接池仍绑定在第一次的已关闭循环上，第二次使用即报错。
- 修复：`Repl` 增加会话级持久事件循环（`_run_async` / `_close_loop`），7 处 `asyncio.run(...)`
  全部改为 `self._run_async(...)`，REPL 退出时 `_close_loop()`。CLI 一次性命令仍用 `asyncio.run`（单进程无复用，不受影响）。
- 测试：`TestPersistentEventLoop`（同一 Repl 两次 `_run_async` 拿到同一循环；`_close_loop` 后已关闭）。

### 14.2q 修复：思考型模型下的 `tool_choice` 400（非流式 /ask）

- 现象：`/stream off` 后提问报 `400 Thinking mode does not support this tool_choice`，回答变成
  “资料不足，未能形成可靠结论”。
- 根因：非流式 RAG 用 `create_agent(model, response_format=Answer)`，LangChain 会以
  `tool_choice=required` 调结构化输出工具；Qwen3-thinking / GLM 等思考模式直接拒绝。
  流式路径 `_stream_answer` 用的是普通 `astream`（无工具、无 response_format），所以只在关流后暴露。
- 修复：
  1. `make_rag_agent` **去掉 `response_format`**，system prompt 里明确“只输出正文 Markdown，不要 JSON”；
  2. `run_rag_agent` 新增 `last_ai_text()` 取最后一条 AI 消息正文，`clean_stream_output` +
     `collapse_repetition` 清洗后再用 `anchors_in` 解析 `citation_ids`；
  3. 保留纯对话兼底 `_plain_rag_answer`（agent 报错时用）；tool_choice/thinking 类错误降为 INFO 日志。
- 测试：`TestRagAgentTextOutput`（从文本解析答案 + 锦点）、`TestRagAgentToolChoiceFallback`
  （agent 400 → 纯文本回退；无模型时优雅提示）。
- 注：`search/summarize/writer` 仍用 `response_format`，思考型模型下会走各自的降级路径（直接检索 /
  仅摘要 / 材料原文）；如需同样改造成本较高，可按需再做。

### 14.2r 修复（真因）：`enable_thinking is restricted to True`

- 现象：关流式后 `/ask` 仍为“资料不足”；日志先是 `Thinking mode does not support this tool_choice`（agent 路径），
  换掉 `response_format` 后又变成 `400 InternalError.Algo.InvalidParameter: The value of the enable_thinking parameter is restricted to True`。
- 真因：`get_chat_model` 以前在 DashScope 分支**总是**注入 `extra_body={"enable_thinking": False}`；
  而 `glm-5.3` 这类模型**只能开启思考**，传 `false` 直接 400 → 流式/非流式都返回空 → “资料不足”。
- 修复：
  1. 新增 `Settings.disable_thinking`（`DISABLE_THINKING=1`）；`get_chat_model` 改为
     `_thinking_body(s)`：只有显式 `enable_thinking=True` 传 true、显式 `disable_thinking=True` 传 false、
     **默认什么都不传**（交给服务端，兼容只能开思考的模型）。
  2. 非流式 `/ask` 不再走带工具/`response_format` 的 RAG agent，直接复用 `_stream_answer`
     （普通对话，无 `tool_choice`），只把 token 回调丢弃 —— 已验证可用。
- 真机验证（用户配置的 dashscope / glm-5.3）：非流式 `ask` 返回 202 字带 `[A-C1][A-C2]` 的回答，
  `problems=[]`（之前为“资料不足”）。
- 测试：`TestThinkingParam`（默认不传 / 显式开 / 显式关；DashScope 模型默认无 thinking 参数）、
  `TestAskNonStreamUsesPlainChat`（非流式走普通对话路径）。

### 14.3 验证证据（可复现）

```bash
python -m src.paper_agent selftest        # chat OK / embedding OK dim=1024
python -m src.paper_agent mcp-tools       # 76 → 27；含 arxiv_search_papers、paper-search_search_papers
python -m pytest                          # 63 passed in ~8s（全离线）
python -m src.paper_agent report "<中文主题>" --papers 2
# → output/<ts>-<slug>.md（5 万+ 字符，含对比表/逐篇摘要/引用锚点附录）
#   output/<ts>-<slug>.bib、output/<ts>-<slug>.json（结构化，含 summaries/answers/citations/notes）
```

真实一次运行的关键指标：检索候选 11 篇 → 选中 2 篇 → 入库 101 chunks（17 页 + 15 页，`pymupdf`）
→ 2 份精读摘要（置信度 0.85~0.90）→ 4 个问题的带引用回答（引用校验仅 1 条待核）
→ 校验重试 1 次后收敛，报告 54KB + BibTeX。
