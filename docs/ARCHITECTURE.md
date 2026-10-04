# 实现原理与文件职责

本文说明各功能**怎么实现的**、以及**每个文件负责什么**。设计背景与演进记录见 [`history/PLAN.md`](history/PLAN.md)；
安装、配置、命令用法见根目录 [`README.md`](../README.md)。

---

## 1. 总览

```
入口      main.py（REPL / 一次性）        cli.py（typer 子命令）
              │                              │
编排          pipeline/session.py（可复用流水线：run_search / run_ingest / ask / run_report）
              │
     ┌────────┼───────────────┬───────────────────────┐
     ▼        ▼               ▼                       ▼
  检索层    RAG 层          问答层                  报告层
sources/*   rag/*        rag/retriever          agents/supervisor
llm/search  tools/*      + llm/factory          + 四个专家 agent
     │        │               │                       │
     └────────┴───────────────┴───────────────────────┘
              ▼
         core/*（config / logging / schema / utils / ui）
```

**分层原则**：底层只提供能力（配置、模型、检索、RAG），中间层是确定性流水线，上层（图/REPL/CLI）只做编排与呈现。
`main.py` 与 `cli.py` 不直接做检索或 RAG，全部经 `pipeline/session.py`，保证两种入口行为一致。

**一次 `report` 的数据流**：

```
plan → search_one×N → merge → ingest_one×M → summarize_one×M
     → answer_one×K → write → verify ─┬→ END
                                      └→ answer_one（重试 ≤1）
```

---

## 2. 配置与模型层

### `core/config.py` — 全局 Settings

- 用 `pydantic-settings` 的 `BaseSettings` 读**仓库根目录** `.env` 与环境变量；`_abs()` 把所有相对路径解析到仓库根，避免从任意 cwd 启动时产物乱跑。
- 承载全部可调参数（检索、切分、超时、并发、路径、MCP 等）。
- 关键派生属性：
  - `provider_label` / `active_base_url` / `active_embedding_model` / `is_dashscope`：当前生效的供应商信息。
  - `papers_dir` / `index_dir` / `manifest_file` / `output_path` / `servers_file`：统一路径，`ensure_dirs()` 建目录。
  - `enabled_channels()` / `channel_credentials()`：`/channels` 写入的渠道配置。
  - `enabled_source_kinds` = 已启用渠道 kind + `BUILTIN_SOURCES`；`active_sources` = `SEARCH_SOURCES` 覆盖或前者。
  - `mcp_sources`：把 `active_sources` 过滤成 **MCP 支持的源**（模块常量 `MCP_SOURCE_KINDS`）。非 MCP 源（Tavily、百度学术、万方、国图…）不进 MCP，因此"只启用 Tavily"时 MCP 整体被跳过。
  - `mcp_env()`：只透传已配置的 MCP 侧密钥/邮箱（避免 `${VAR}` 字面量残留）。
- `get_settings()`：进程级单例，先读 `.env`，再用 `userconfig.apply_to()` 叠加 JSON 配置（JSON 优先）。

### `llm/factory.py` — 模型工厂

- `get_chat_model(role)`：按角色给不同 temperature（检索 0.0、写作 0.4…），依次尝试
  ① 通用供应商（`/connect` 或 `LLM_*`）→ ② `.env` 的 DashScope → ③ DeepSeek；未配置时抛 `ConfigError`（带可执行的修复建议）。
- `_thinking_body()`：`enable_thinking` **默认不传**（兼容只能开思考的模型），只有显式 `ENABLE_THINKING`/`DISABLE_THINKING` 才传。
- `get_embeddings()`：① 独立 embedding 供应商（`EMBED_*`）→ ② 当前供应商自带 embedding → ③ `.env` DashScope 兜底；
  统一用 `check_embedding_ctx_length=False` 并包一层 `RetryingEmbeddings`。
- 所有请求都带 `timeout`（默认 180s）与有限重试。

### `sources/userconfig.py` — 供应商 JSON 配置

- 数据结构：`Provider`（base_url / key / kind / chat_model / embedding_model / embedding_dim / 模型列表）与 `SearchChannel`。
- 读写 `<仓库根>/.paper-agent/config.json`（项目内，原子写 + 0600，随项目移植；可用 `PAPER_AGENT_CONFIG` 改路径，相对路径相对仓库根）。
- `detect_provider()`：按 base URL 子串识别 dashscope / deepseek / openai / moonshot / siliconflow / zhipu / volcengine / openrouter / local / 通用兼容端点。
- `classify_models()`：把 `/models` 结果按名称特征分成对话 / embedding；`guess_chat_model()` / `guess_embedding_model()` 推断默认值。
- 网络：`fetch_models()` 拉 `/models`；`probe_embedding_dim()` 发一次极小 embedding 请求探测维度（用于索引签名）。
- **对话与 embedding 解耦**：`pick_embedding_provider()` 自动挑选（对话供应商 → 默认 → 第一个带 embedding 的）；
  `resolve_embedding()` 优先用户显式指定的 `embedding_provider`（`/embed` 设置）。
- `settings_overrides()` / `apply_to()`：把当前供应商、默认模型、embedding 供应商、搜索渠道翻译成 `Settings` 覆盖字段。

### `repl/tui.py` — `/connect` 交互

- `connect_flow()`：预设/URL → 识别供应商 → 读 key → 拉模型 → 可选探测 embedding 维度 → 落盘。
- `read_secret()`：只关 `ECHO` 保留行缓冲（`ICANON`），终端粘贴的括号标记不会混入 key；读完恢复终端属性。
- `_sync_models()`：拉取 + 分类 + 推断默认；`embed_models_for()` 给选择器用。
- 选择器统一委托 `repl/input.pick_value()`。

---

## 3. 检索层

### `sources/channels.py` — 渠道注册表（元数据）

- `ChannelSpec`：`kind / label / base_url / description / needs_key / needs_email / group / builtin(免 key)`。
- `REGISTRY` 登记全部渠道；`GROUP_ACADEMIC / GROUP_CN / GROUP_WEB` 分组；`PRESET_ORDER` 决定 `/channels add` 的编号顺序。
- 辅助函数：`spec_for()`、`list_specs()`、`channel_label()`、`is_domestic()`、`domestic_first()`、`default_base_url()`。
- 只放元数据，真正的请求实现在 `sources/fetchers.py`。**默认不启用任何渠道**。

### `sources/fetchers.py` — 内置 HTTP 抓取

- HTTP 基础设施：`_request()`（GET，429/5xx/网络异常重试并尊重 `Retry-After`，4xx 不重试）、`_post_json()`（POST，`ensure_ascii` 开关给国图用）。UA、超时、`mailto` 都从这里统一处理。
- 三个基础源：
  - **arXiv**：Atom XML 解析 `parse_arxiv_atom()`、查询构造 `build_arxiv_query()`（布尔串转 `all:"phrase"`）、`_throttle_arxiv()` 保证 ≥3s 间隔。
  - **OpenAlex**：`openalex_work_to_paper()` 还原 `abstract_inverted_index`、取 `best_oa_location.pdf_url`；带 `mailto`/`api_key` 进 polite pool / 提配额。
  - **Crossref**：`crossref_work_to_paper()`（JATS 标签/HTML 实体清洗）。
- 可配置渠道：`search_europepmc / search_pubmed / search_semanticscholar / search_doaj / search_core / search_chinaxiv / search_baidu_scholar / search_wanfang / search_nlc / search_tavily / search_exa / search_serpapi`，统一签名 `(query, limit, settings, channel=None)`；响应解析拆成纯函数 `parse_*()` 便于离线单测。`CHANNEL_SEARCHERS` 做分发；`registered_sources()` 列出全部 kind。
- 标识解析：`classify_identifier()`（arxiv/doi/url）、`resolve_identifier()`、`resolve_ids()`（并发 + 去重 + 失败清单），供 `--ids` 直抓。
  - **arXiv 降级**：元数据 API 429/超时时 `resolve_arxiv()` 改用 `arxiv_fallback_paper()`（只给 `arxiv.org/pdf/<id>` 直链，`source="arxiv-pdf"`）；标题等元数据由 `parse_pdf()` 的 `guess_title()` 从 PDF 首页补齐（`ingest_paper()` 写入索引）。API 正常但查无此文仍返回 `None`，不凭空造条目。
- `builtin_search()`：核心调度。
  - 源解析优先级：显式 `sources` > `all`/`SEARCH_ALL_CHANNELS` > 已启用渠道；
  - 每渠道独立请求，`asyncio.Semaphore(channel_concurrency)` 限流 + 单渠道 `channel_timeout`；
  - `on_event` 回调上报 `queued/running/done/failed/skipped`（REPL 逐渠道进度表）；
  - `by_channel` 按 kind 分别收集结果；缺 key 的渠道跳过并标 `(需key)⊘`；
  - `papers_domestic_first()` 让国内库靠前；`_short_error()` 把异常压成 `429`/`Timeout`/`需key` 等短标签。

### `llm/search.py` — 检索中的 LLM 增强

- `expand_queries()`：把主题扩成多条英文检索式（提升召回）；`rank_papers()`：对合并后的候选做相关性重排/裁剪。
- 两者都受 `llm_timeout` 约束，**失败/超时静默降级**为原始行为，绝不阻塞检索。

### `sources/mcp.py` + `sources/mcp_servers.json` — MCP 接入

- `sources/mcp_servers.json`：只存模板（stdio 命令 / streamable_http URL），`${VAR}` 由 `load_server_specs()` 展开；命令缺失或 URL 未配置的 server 自动跳过。
- `quiet_env()` + `sources/mcp_quiet/sitecustomize.py`：给 MCP 子进程注入 `PYTHONPATH`，收敛第三方 logger 刷屏。
- `build_client()`：`MultiServerMCPClient`，开启工具名前缀、工具异常转错误文本、调用日志拦截器。
- `filter_tools()`：白名单收敛（76→27）、黑名单排除（scihub / google_scholar / watch 等）、无 S2 key 时不暴露 `search_semantic`；
  `only_enabled_sources=True` 时**进一步按已启用渠道收敛检索工具**（聚合工具 `search_papers` 保留，由 `sources` 参数限定；`arxiv-mcp-server` 那种单源 `search_papers` 按 arxiv 是否启用决定去留）。
- `guard_search_sources()`：改写 `search_papers` 的 `sources` 为已启用渠道，剔除无 key 的 semantic；无可配置源时不擅自改写。
- `textify_tool()`：把 MCP content blocks 拍平成 JSON 文本，方便模型阅读与离线断言。
- `describe_mcp_tools()`：`mcp-tools` 命令的诊断数据（各 server 原始/保留工具数）。

---

## 4. RAG 层（`rag/`）

| 文件 | 原理 |
|---|---|
| `fetch.py` | 只抓开放获取 PDF：命中本地缓存直接返回；`httpx` 下载后校验 `%PDF` 魔数再落 `data/papers/<safe_filename(paper_id)>.pdf`（文件名规则在 `core.utils.safe_filename`）。返回 `(path, message)`，失败原因会写进入库结果。 |
| `parse.py` | 优先 `pymupdf`（更快更准），失败/未装回退 `pdfplumber`。清洗：合并跨行连字符、压缩空白、去多页重复的页眉页脚；`_cut_references()` 在文末 References 处截断（保守策略，只在确实像文献表时才切）。`ParsedDoc` 保留逐页文本、engine 与 `title`（`guess_title()`：优先 PDF 内嵌元数据，其次首页第一行像标题的文本；仅在元数据缺失时作兜底）。 |
| `split.py` | `RecursiveCharacterTextSplitter`（中英混排分隔符）+ 每 chunk 元数据 `paper_id/title/page/chunk_index/source`。 |
| `embeddings.py` | `RetryingEmbeddings`：遇到 `batch size` 类 400 自动**二分拆批**，瞬态错误（429/5xx/超时）指数退避重试。 |
| `store.py` | `PaperIndex` = `InMemoryVectorStore` + 本地 JSON 持久化。`dump/load` 直接复用 langchain 内置能力；另维护 `manifest.json`（论文 → chunk id 列表、PDF 路径、sha256、元数据），支持增量更新与删除。`embedding_signature()` 记录模型+维度，加载时不匹配抛 `IndexSignatureError`。检索用 `similarity_search_with_score(filter=...)` 按 `paper_id` 过滤。 |
| `retriever.py` | `CitationCollector` 给片段编锚点（`{prefix}C1..Cn`，并行分支带不同前缀避免撞号）；`format_context()` 渲染带锚点上下文；`retrieve()` / `retrieve_across_papers()`（每篇保底片段数，避免证据偏置）/ `_hybrid_rerank()`（可选 BM25 线性加权）。**引用校验**：`tokenize/support_ratio/has_anchor` 做字面重合，`hard_tokens/cross_lingual_support` 用术语/数字做跨语言判定；`verify_answer()` 综合判定，`verify_report_citations()` 校验最终报告锚点。 |

---

## 5. 工具层与 Agent 层

### `tools/`

- `paper_tools.py`：`ingest_paper()` 确定性入库（下载→解析→切分→向量化→落盘），失败分类返回 `no_pdf / parse_error / empty / embed_error`；`make_paper_tools()` 暴露给 agent 的 `download_and_index_paper` / `list_indexed_papers`。
- `rag_tools.py`：`make_rag_tools()` 给 agent 的 `search_corpus` / `read_chunk`（检索结果写进 `CitationCollector`）。

### `agents/`

- `common.py`：从 agent 返回值里取结构化结果 / 最后一条 AI 文本 / 拼接文本（`structured_response`、`last_ai_text`、`result_text`）。
- `prompts.py`：plan / search / select / summarize / rag / writer 各角色 system prompt。
- `search_agent.py`：`create_agent(..., response_format=PaperList)`；`run_search_agent()` 失败时解析文本 JSON，再失败退回 `direct_search()`（不经过 LLM 直接调用 MCP 检索工具）。`_fit_search_args()` 适配不同 server 的工具签名，`_pick_search_tools()` 优先聚合工具 `search_papers`。
- `summarize_agent.py`：单篇结构化精读（problem/method/data/findings/limitations/reusable_ideas + 原文引句）；`fallback_summary()` 只基于摘要兜底。
- `rag_agent.py`：**不使用 `response_format`**（思考型模型会拒绝强制 `tool_choice`），改为输出带锚点的 Markdown 再解析；`_plain_rag_answer()` 普通对话兜底，`answer_without_llm()` 离线直接给原文证据。
- `writer_agent.py`：`WriterOutput`（executive_summary/comparison/gaps/conclusion）；`material_from_state()` 把摘要与问答拼成写作材料。
- `supervisor.py`：`StateGraph` 编排（见下）。

---

## 6. 编排层

### `pipeline/session.py` — 可复用流水线（CLI 与 REPL 共用）

- `Session` / `build_session()`：持有 Settings、`PaperIndex`、ChatModel；当 embedding 签名与现有索引不符时，自动把 `data_dir` 重定向到 `data/by-embedding/<签名>/`。
- `ensure_tools()` / `_load_tools_once()`：懒加载并缓存 MCP 工具（`MultiServerMCPClient` 无状态，可跨命令复用）。
- `run_search()` → `_run_search_impl()` → `_search_once()`：
  - `source` 决定走 MCP / 内置 / 合并；**未启用任何渠道时跳过 MCP**，直接落到内置层给出提示；
  - LLM 查询扩展 + 多检索式并发 + 相关性重排（可关）；
  - `/search --limit N` 在**按渠道汇总后再截断**，所以 N 是每渠道总数，不随检索式数量放大；
  - `on_event` / `by_channel` 透传给 REPL 用于逐渠道进度与分渠道展示。
- `run_ingest()` / `ingest_papers()`：有 `--ids` 时直接用 `resolve_ids()` 抓元数据；否则检索后入库；`remove_papers()` 删除索引与 PDF。
- `ask()` → `AskResult`：检索 → 生成 → `normalize_answer_citations()` + `verify_answer()`；流式路径 `_stream_answer()`。
- **复读保护**：`RepetitionGuard`（流式实时检测并停止）、`truncate_repetition()`（相邻/滚动窗口复读）、`dedupe_repeated_blocks()`（非相邻整块复读）、`collapse_repetition()`（段落+句子+整块+末尾四重折叠）、`clean_stream_output()`（去掉泄漏的 JSON 尾巴）。
- `run_report()`：组装依赖 → 编译并运行监督图 → 落盘三件套。

### `agents/supervisor.py` — LangGraph 监督图

- `Deps`：把 settings / index / agents / 工具等注入节点，便于测试替换。
- 节点：
  - `plan`：LLM 拆子问题 + 生成检索式，失败用 `_fallback_queries()` 模板；`merge`：去重 + LLM 打分（失败用 `_heuristic_select()` 词面/时效/OA 启发式）。
  - `search_one` / `ingest_one` / `summarize_one` / `answer_one`：worker，每条目一个 `Send` 任务。
  - `write`：写作 agent + `render_report()`；`verify`：汇总引用/覆盖率/入库失败等问题。
- **并行**：`fan_out_*()` 返回 `Send` 列表；`Annotated[list, operator.add]` 归并；每个扇出后加 `collect_*` **barrier 节点**再触发下一次扇出（否则条件边按 worker 实例求值会让下游成倍重复）。
- **容错**：`guard(label)` 装饰 worker，单条目异常只写 `notes`/`worker_errors`（带 reducer 的字段），不崩整图；每个 LLM 节点都有确定性兜底。
- `route_after_verify()`：按 `retry` 标志重新扇出问答（≤1 次）。
- `build_simple_app()`：`--simple` 路径，单 agent 拿全部工具自行决定顺序。

---

## 7. 呈现层

| 文件 | 原理 |
|---|---|
| `pipeline/report.py` | `render_report()` 把论文/摘要/问答/引用/叙述拼成 Markdown（含对比表、逐篇摘要、参考文献、锚点附录）；`build_bibtex()` 生成 BibTeX；`write_outputs()` 落盘 `.md/.bib/.json`。 |
| `core/schema.py` | 全部数据模型：`Paper`/`PaperList`/`coerce_paper`（兼容两种 MCP 返回）、`PaperSummary`、`Citation`、`Answer`、`SelectionItem/Output`、`ResearchState`（图状态，列表字段用 reducer）。 |
| `core/utils.py` | 纯函数工具：`normalize_paper_id()`（arXiv/DOI/OpenAlex/链接统一成稳定 key）、`dedupe_papers()`（后到者补空缺字段）、`clean_pasted/clean_secret/mask_secret`（粘贴清洗与脱敏）、`slugify/truncate/safe_filename`（文件名安全化，供 `rag/fetch` 与 `pdf/server` 共用）、`mcp_result_to_text/extract_json/first_list`、`DeterministicFakeEmbeddings`（离线确定性向量）。 |
| `llm/fake.py` | `FakeToolCallingModel`：实现 `bind_tools`/`with_structured_output`/`_generate`，让 `create_agent` 在离线时也能构建并一步结束。 |
| `cli.py` | typer 薄封装：`search/ingest/ask/report/rm/channels/providers/keys/embed/mcp-tools/selftest` + `papers-open`/`papers-close`（本地 PDF 预览，可单独跑），只解析参数并调用 `pipeline.session`/`pdf.server`。 |
| `main.py` | **薄入口**：argparse 定义（`--search/--ingest/--report/--offline/--no-stream/-v`）→ 建 `Repl` → 一次性命令或进 REPL；REPL 本体在包里（见下）。 |
| `repl/app.py` | `Repl` 核心：`__init__`（settings/session/config/history）、`banner`、`dispatch`/`safe_dispatch`、`_print_help`、`repl()` 主循环、`status_line`/`palette_context`（补全上下文）、`_run_async`（会话级持久事件循环，避免 `Event loop is closed`）、`_stream_renderer`（append-only 流式）、`_live`/`_stop_live`。本身只继承三个 mixin（`class Repl(SearchCommands, PaperCommands, ProviderCommands)`）。 |
| `repl/base.py` | `ReplBase`：把混入之间共享的状态（`settings/session/config/history/stream/_live/_loop/_pdf_server`）与核心方法签名集中声明一次（`raise NotImplementedError` 占位），供 mixin 做类型检查——否则 mypy 会在每个 mixin 里各自推断出更窄的类型。 |
| `repl/commands/search.py` | `SearchCommands`：`/search` `/ingest` `/ask` `/report` `/mcp` `/channels`（含 `add`、渠道失败处理与引导）。 |
| `repl/commands/papers.py` | `PaperCommands`：`/papers`（列表 / `rm` / `open` / `close`）、`/index`、`/logs`、`/history`、`/save`；本地 PDF 预览服务的生命周期也在这里。 |
| `repl/commands/providers.py` | `ProviderCommands`：`/connect` `/providers` `/keys` `/models` `/model` `/embed` `/offline`；配置写盘后统一走 `self._rebuild()` 重建会话。 |
| `repl/ui.py` | 展示层：`COMMANDS`/`COMMAND_USAGE`/`HELP_GROUPS`/`HELP_EXAMPLES`（`/help` 与命令面板补全共用）、`SearchProgressView`（逐渠道进度表）、`setup_readline/save_readline_history`（未装 prompt_toolkit 时的退化输入）、`nullcontext`。 |
| `core/ui.py` | 共享 `console`（进程内单例）+ `PROMPT`：`repl/` 各模块与 `main.py` 都从这里取，替换这一个对象就能捕获全部输出（测试就是这么做的）。 |
| `repl/input.py` | prompt_toolkit 输入层：命令面板补全（含 `--flag`、供应商名、模型名、论文 ID 等上下文补全）、`pick_value()` 可滚动选择器（Enter 选中 / Ctrl+C 设默认 / Ctrl+P 换供应商 / Esc 取消）、主题 `_THEMES`、`read_line()`；未装 prompt_toolkit 时退化为 rich + 标准输入。 |
| `core/logging.py` | 日志装配：控制台 handler + **按天分文件** handler。`DailyFileHandler`  写 `logs/paper-agent-YYYY-MM-DD.log`（跨天自动换文件、单日超过 `PAPER_AGENT_LOG_MAX_MB` 续写 `-02`、按「最近 N 天」清理旧文件、重启接着当天最后一段写）；`PAPER_AGENT_LOG_FILE` 则退回固定单文件 + 按大小轮转；`_NoisyFilter` 按**前缀**挡掉 httpx/httpcore(x2)/mcp 等噪声（写死名单拦不住改名包）；`/logs [n] [--files]` 查看。 |
| `pdf/server.py` | `/papers open` 的本地预览服务：只绑回环地址的 `ThreadingHTTPServer`，`/` 给列表页（可过滤 + 内嵌阅读器 + 底部「停止预览服务」）、`/pdf/<id>` 按 `papers_dir` 文件名映射返回 PDF（支持单段 `Range`/206/416、`?download=1`），校验 `Host` 头防 DNS rebinding；**退出四件套**：页面 `POST /shutdown`（带启动时随机 token）、`/papers close`、`atexit`+`SIGTERM` 收尾、空闲超时（`PAPER_AGENT_PDF_IDLE_MIN`，默认 30 分钟）；启动时把 `{pid,port,url}` 写进 `.paper-agent/pdf-server.json`（陈旧记录会被 pid/端口探测清掉），因此**另一个进程**也能找到并 SIGTERM 停掉它（`stop_registered_server()`）。`start_viewer()` 供 REPL 与 CLI 共用；自动开浏览器用 `open_in_browser()`（逐个试 `$BROWSER`/`xdg-open`/`wslview`/`gio` 等启动器，丢弃输出、失败只提示手动打开，不再像 stdlib `webbrowser` 那样把 `gio: ... Operation not supported` 漏到终端）。 |

---

## 8. 关键机制（横向）

- **ID 归一化**：`normalize_paper_id()` 把所有写法（`arXiv:2405.16506v2`、DOI 链接、OpenAlex W-id）统一成 `arxiv:` / `doi:` / `openalex:` 前缀 key，去重、引用与索引都基于它。
- **引用锚点与抗幻觉**：每段检索结果分配 `[C#]`（并行分支加前缀）；回答必须原样引用；`normalize_answer_citations()` 修模型省略前缀的问题；`verify_answer()` 做存在性 + 字面/跨语言支撑度校验；报告级再校验一次。
- **约束下的降级**：LLM 节点失败→模板/启发式；MCP 失败→内置 HTTP；内置某源失败→其余源继续；检索 429→提示 polite pool；401/403/需 key→引导配置。任何一步都不让整条链路失败。
- **并发与超时**：`Send` 扇出受 `max_concurrency` 限制；检索多源受 `channel_concurrency` + 单源 `channel_timeout`；LLM/embedding 统一 `llm_timeout`；整次检索受 `search_timeout`。
- **索引隔离**：`embedding_signature()`（模型:维度）写进 manifest；加载不匹配抛 `IndexSignatureError`，`build_session()` 自动切到 `data/by-embedding/<签名>/`。
- **源收敛**：`/channels` 的启用集合同时决定内置层要查哪些源、MCP 聚合工具的 `sources`、以及哪些 MCP 检索工具会挂给模型。

---

## 9. 测试与离线

- **全部离线可跑**：`FakeToolCallingModel`（`llm/fake.py`）+ `DeterministicFakeEmbeddings`（`core/utils.py`）+ `tests/fixtures/fake_mcp_server.py`（真实 stdio MCP 协议往返）。
- **关闭联网**：`--offline` / `PAPER_AGENT_FAKE_LLM=1` → 假模型 + 假 embedding，并强制使用独立的 `data/offline/` 索引目录（避免维度混用）。
- 主要测试文件：
  - `test_utils.py` ID 归一化/去重；`test_rag.py` 解析/切分/索引持久化/引用校验；
  - `test_mcp.py` 白名单、源收敛、参数护栏 + 真 MCP 往返；`test_agents.py` 各 agent 兜底与报告渲染；
  - `test_userconfig.py` 供应商识别/模型分类/JSON 读写/chat-embedding 解耦；
  - `test_tui.py` / `test_repl.py` 选择器、命令分发、粘贴清洗、事件循环；
  - `test_channels.py` 渠道注册表/各源解析器/每渠道截断/进度事件；`test_sources.py` 内置源解析与按 ID 解析；
  - `test_graph_offline.py` 整图离线端到端（检索→入库→精读→问答→报告→校验）。

---

## 10. 文件职责速查

```
main.py                        薄入口：argparse + 一次性命令分发（REPL 实现在 src/paper_agent/repl/）
src/paper_agent/
  __main__.py                  `python -m src.paper_agent` 入口 → cli.py
  cli.py                       typer 子命令（薄封装 pipeline；+ papers-open/papers-close 独立起/停预览）
  core/                        基础设施层（无业务逻辑，被各层复用）
    config.py                  pydantic-settings 全局配置 + 源/路径派生属性
    logging.py                 日志落盘：按天分文件 logs/paper-agent-YYYY-MM-DD.log（保留 N 天）+ 控制台 handler
    schema.py                  数据模型与图状态
    utils.py                   ID 归一化、去重、粘贴清洗、文件名安全化、JSON 抽取、假 embedding
    ui.py                      共享 console + 提示符
  sources/                     来源层（渠道 / 抓取 / MCP / 供应商凭证）
    channels.py                渠道注册表（元数据、分组、国内优先、预设编号）
    fetchers.py                内置 HTTP 检索：arXiv/OpenAlex/Crossref + 可配置渠道 + 按 ID 直抓（429 降级直链）
    mcp.py                     MCP 接入：server 加载、工具白名单/源收敛/参数护栏/文本化
    userconfig.py              供应商 JSON 配置：识别/读写/模型分类/注入 Settings
    mcp_servers.json           MCP server 模板（stdio / streamable_http）
    mcp_quiet/sitecustomize.py 静音 MCP 子进程的第三方日志
  llm/                         模型层
    factory.py                 聊天/embedding 模型工厂（供应商优先级、thinking、重试）
    search.py                  检索用 LLM：查询扩展 + 相关性重排（可降级）
    fake.py                    离线假模型
  rag/                         RAG 层
    fetch.py / parse.py / split.py / embeddings.py / store.py / retriever.py
  tools/                       工具层
    paper_tools.py             确定性入库 + 索引查询工具
    rag_tools.py               search_corpus / read_chunk 工具
  agents/                      Agent 层
    common.py / prompts.py    结构化结果提取、各角色提示词
    search_agent.py            检索 agent + 直接调用 MCP 的降级路径
    summarize_agent.py         单篇精读 agent + 摘要兜底
    rag_agent.py               带引用问答 agent + 纯文本/离线兜底
    writer_agent.py            报告写作 agent
    supervisor.py              LangGraph 监督图与 simple 模式
  pipeline/                    编排层
    session.py                 可复用流水线：搜索/入库/问答/报告 + 复读保护
    report.py                  Markdown / BibTeX / JSON 渲染与落盘
  pdf/
    server.py                  本地 PDF 预览：回环 HTTP 服务 + 列表页（`/papers open`）
  repl/                        交互层
    app.py                     Repl 核心：命令分发、帮助、主循环、事件循环与流式渲染
    base.py                    mixin 共享状态 + 核心方法签名（供类型检查）
    commands/search.py         SearchCommands：/search /ingest /ask /report /mcp /channels
    commands/papers.py         PaperCommands：/papers /index /logs /history /save + PDF 预览服务
    commands/providers.py      ProviderCommands：/connect /providers /keys /models /model /embed /offline
    ui.py                      命令表/帮助分组、检索进度视图、readline 历史
    input.py                   prompt_toolkit 补全与选择器、主题、read_line
    tui.py                     /connect 流程、密钥读取、模型列表同步
docs/history/PLAN.md           设计与演进记录（历史文档）
docs/ARCHITECTURE.md           本文
```
