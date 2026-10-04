# 学术论文检索与概括分析 Agent

纯 Python + LangChain：**联网检索论文 → 下载 OA 全文 → RAG 带引用问答 → 多 agent 出调研报告**。

- 实现原理与各文件职责：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- 设计与演进记录：[`docs/PLAN.md`](docs/PLAN.md)
- 依赖：[`requirements.txt`](requirements.txt)（精确全量）· [`environment.yml`](environment.yml)（直接依赖）
- 环境变量模板：[`.env.example`](.env.example)（可直接复制为 `.env`）

---

## 快速开始

```bash
# 任意 Python >= 3.11
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# 或 conda env create -f environment.yml && conda activate agent

cp .env.example .env        # 填一个 API key（也可启动后用 /connect 交互配置）
python main.py              # 进入交互式 REPL
```

自检：

```bash
python -m src.paper_agent selftest     # 模型 / embedding 是否可用
python -m src.paper_agent mcp-tools    # MCP server 与工具白名单（可选）
```

## 配置

两种方式，**JSON 配置优先于 `.env`**：

1. **交互（推荐）**：REPL 里 `/connect` —— 输入 base URL + API key，自动识别供应商、拉取 `/models`、分类对话/embedding 模型并写入 `~/.config/paper-agent/config.json`（0600）。
2. **`.env`**：直接填 `DASHSCOPE_*` / `DEEPSEEK_*`，或用通用变量 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`。
   embedding 可单独指定：`EMBED_BASE_URL` / `EMBED_API_KEY` / `EMBED_MODEL`（对话用 A 家、embedding 用 B 家）。

> JSON 配置 > `.env`；想只用 `.env` 设 `PAPER_AGENT_IGNORE_USER_CONFIG=1`。
> 换 embedding 模型/维度时索引自动切到 `data/by-embedding/<签名>/`，不会维度混用。

## 使用

### 交互式（`main.py`）

```bash
python main.py                         # 进入 REPL（输入 / 弹命令面板，/help 看全部命令）
python main.py "跨块图增强解决了什么问题？"   # 一次性问答（流式 + 引用校验）
python main.py --search "graph rag"    # 一次性检索
python main.py --ingest "graph rag"    # 一次性入库
python main.py --report "主题" --papers 3
python main.py --offline               # 假模型 + 独立索引目录（无密钥自检）
```

进入 REPL 后**直接输入自然语言 = `/ask`**。常用命令：

| 命令 | 作用 |
|---|---|
| `/search <词> [--limit N] [--ingest [N]] [--source auto\|mcp\|builtin\|all] [--no-llm]` | 联网检索；`--limit N` = **每个渠道**最多 N 条（不随 LLM 扩展的检索式数量放大） |
| `/ingest <词> [--limit N] [--ids arxiv:xxx] [--force]` | 下载入库，建立/更新索引 |
| `/ask <问题> [--papers a,b] [--k N]` | 带引用问答（默认流式） |
| `/report <主题> [--papers N] [--simple]` | 端到端报告 → `output/<时间戳>-<slug>.{md,bib,json}` |
| `/papers [rm <id>\|--all]` · `/index` · `/mcp` | 论文 / 索引 / MCP 工具 |
| `/channels [add\|rm\|key-rm\|on\|off\|all on\|domestic on]` | **搜索渠道配置**（见下） |
| `/connect` · `/models` · `/providers` · `/model` · `/embed` · `/keys` | 供应商与模型（`/embed` 单独指定 RAG embedding） |
| `/offline [on\|off]` · `/stream [on\|off]` · `/history` · `/save` · `/clear` · `/exit` | 会话管理 |

交互特性：Tab 补全 + 历史；`/models` 输入即筛选（Enter 本次使用、Ctrl+C 设为默认、Ctrl+P 换供应商）；流式输出 append-only（不重绘）；单条命令报错/`Ctrl-C` 不退出会话；支持管道 `printf '/index\n/exit\n' | python main.py`。

### 脚本式 CLI（`python -m src.paper_agent`）

```bash
python -m src.paper_agent search "graph retrieval augmented generation" --limit 8
python -m src.paper_agent search "..." --source builtin          # 不走 MCP
python -m src.paper_agent ingest "graph rag survey" --limit 3
python -m src.paper_agent ingest --ids arxiv:2405.16506,10.1145/3626772.3657775
python -m src.paper_agent ask "跨块图增强解决了什么问题？" --papers arxiv:2605.28004
python -m src.paper_agent report "..." --papers 3 [--simple|--offline]
python -m src.paper_agent channels add arxiv
python -m src.paper_agent embed dashscope --model text-embedding-v4
```

## 检索渠道（默认全部禁用）

**所有渠道默认禁用，必须手动添加后才参与检索**：

```
/channels                 # 查看可添加列表与当前状态
/channels add arxiv       # 按 kind 添加（也可用编号：/channels add 1）
/channels add tavily --key <KEY>
/channels all on          # 可选：每次检索并发跑全部已注册渠道
/channels domestic on     # 国内库（国家图书馆 / ChinaXiv / 百度学术 / 万方）排在前面（默认开）
```

- 免 key：`arxiv` `openalex` `crossref` `europepmc` `pubmed` `doaj` `chinaxiv` `nlc`；
  需 key：`semanticscholar` `core` `baidu_scholar` `wanfang` `tavily` `exa` `serpapi`。
- 未启用任何渠道时 `/search` 会直接提示；只启用某渠道时**不会**再去查别的源（MCP 工具也按已启用渠道收敛）。
- 渠道返回 `429` 会提示进 polite pool（`/channels add openalex --email you@example.com`）；返回 `401/403` 或需要 key 时会**直接引导配置**。
- `/ingest --ids ...` 按 ID 直抓，不经过搜索与渠道启用。

抓取顺序：`--source auto`（默认）= MCP 优先，不可用/无结果回退内置 HTTP；`mcp` / `builtin` / `all`（两者合并）。

## 常用参数（环境变量 / `config.py`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `SEARCH_SOURCE` | `auto` | `auto` / `mcp` / `builtin` / `all` |
| `BUILTIN_SOURCES` | 空 | 直接指定内置层要查的源；**默认空，渠道需 `/channels add`** |
| `SEARCH_ALL_CHANNELS` | `0` | 并发跑全部已注册渠道 |
| `TOP_K` / `MAX_PAPERS` / `CONCURRENCY` | 6 / 8 / 4 | 检索片段数 / 报告论文数 / 扇出并发 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1200 / 200 | 切分粒度 |
| `EMBED_BATCH_SIZE` | 10 | embedding 单批上限（超限自动二分） |
| `MIN_SUPPORT_RATIO` | 0.3 | 引用支撑度阈值 |
| `LLM_TIMEOUT` / `SEARCH_TIMEOUT` | 180 | 单次等待上限 / 一次检索总超时（秒） |
| `SEARCH_USE_LLM` | `true` | 检索时用 LLM 做查询扩展 + 重排（`--no-llm` 可关） |
| `PAPER_AGENT_THEME` | `dark` | 界面配色 `dark` / `light` / `none` |
| `OPENALEX_MAILTO` | 空 | 进 OpenAlex/Crossref polite pool（更稳） |

完整清单见 `.env.example`。

## 测试

```bash
python -m pytest          # 300 passed，全部离线（假模型 / 假 embedding / 假 MCP server）
python -m mypy src main.py
```

`tests/` 默认被 `.gitignore` 忽略（本地保留即可跑；要入库请从 `.gitignore` 删除 `tests/` 段）。

## 排错

| 现象 | 处理 |
|---|---|
| `/connect` 粘贴 key 后 401 | key 读入会回显脱敏结果确认真假；重新 `/connect` 覆盖更新 |
| 检索「只启用了 X 却返回别的源」 | 已按 `/channels` 严格收敛；若仍如此，请**重启 REPL**（长驻进程不会热加载源码） |
| `429` / `403` | 面向 OpenAlex/Crossref 配 `OPENALEX_MAILTO`；需要登录的渠道按提示 `/channels add` 配置 |
| embedding `batch size is invalid` | 调小 `EMBED_BATCH_SIZE`（已内置自动降批） |
| `IndexSignatureError` | 换了 embedding 模型/维度 → 自动用新目录；要用旧索引就换回原模型 |
| 引用校验误报 | 校验是启发式的，可调低 `MIN_SUPPORT_RATIO` |
| MCP 日志刷 Semantic Scholar 429 | 未配 `SEMANTIC_SCHOLAR_API_KEY` 时该源不暴露；配 key 后自动启用 |

## 已知限制

1. **引用校验是启发式的**：跨语言靠术语/数字"硬令牌"匹配，无法核验时跳过并标注，不静默丢弃。
2. **只用开放获取**：已屏蔽 `download_scihub` 与 `search_google_scholar`；部分出版社（如 MDPI）会 403。
3. **默认内存向量库 + 本地 JSON**：适合单次调研（数百篇内），更大语料建议换 FAISS/Qdrant（替换 `PaperIndex` 内部实现即可）。
4. **国内库边界**：ChinaXiv、国家图书馆有公开免 key 接口；百度学术需千帆 key、万方需 APPCODE；知网/维普/超星**无公开检索 API**，不做绕过抓取（可用官方题录导出后走 `/ingest --ids`）。
5. **MCP 工具数量**：`paper-search-mcp` 暴露 57 个工具，靠白名单过滤，勿关闭。
6. **离线模式索引独立**：`--offline` 用假 embedding，固定 `data/offline/`，与真实索引不混用。
