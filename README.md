# 学术论文检索与概括分析 Agent

纯 Python + LangChain：**联网检索论文 → 下载 OA 全文 → RAG 带引用问答 → 多 agent 出调研报告**。

- 实现原理与各文件职责：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- 设计与演进记录：[`docs/history/PLAN.md`](docs/history/PLAN.md)
- **依赖与构建配置（唯一来源）**：[`pyproject.toml`](pyproject.toml)
- conda 环境（只列直接依赖）：[`environment.yml`](environment.yml)
- 环境变量模板：[`.env.example`](.env.example)（可直接复制为 `.env`）

---

## 目录

- [1. 项目是什么](#1-项目是什么)
- [2. 环境要求](#2-环境要求)
- [3. 快速开始](#3-快速开始)
- [4. 构建与安装](#4-构建与安装)
- [5. 配置（供应商 / 密钥 / 渠道）](#5-配置供应商--密钥--渠道)
- [6. 移植到另一台机器](#6-移植到另一台机器)
- [7. 使用](#7-使用)
- [8. 检索渠道（默认全部禁用）](#8-检索渠道默认全部禁用)
- [9. 项目结构](#9-项目结构)
- [10. 常用环境变量](#10-常用环境变量)
- [11. 测试与类型检查](#11-测试与类型检查)
- [12. 排错](#12-排错)
- [13. 已知限制](#13-已知限制)

## 1. 项目是什么

一条从「问题/主题」到「带引用报告」的流水线：

| 能力 | 说明 |
|---|---|
| 联网检索 | MCP（arXiv / paper-search / crossref / pubmed / fetch）优先，不可用自动回退内置 HTTP 源；渠道可插拔、默认全禁用 |
| 全文获取 | 无 PDF 直链时四级降级：PDF 全文 → 网页正文 → 仅摘要 → 仅题录（后两级显式标注，不伪装成原文） |
| RAG 问答 | 分块 + embedding + 向量检索，回答带引用，启发式引用支撑度校验 |
| 报告生成 | 多 agent（LangGraph 监督图）产出 Markdown + BibTeX + JSON 三件套到 `output/` |
| 只读不落盘 | `/quick` 即抓即答：PDF 与索引只在内存，磁盘上不多一个文件 |
| 本地阅读 | `/papers open` 起本地服务，浏览器内嵌看已抓 PDF（支持 Range 拖动） |
| 可离线 | `--offline` / `PAPER_AGENT_FAKE_LLM=1` 用假模型 + 假 embedding，无密钥也能自检、跑测试 |

技术栈：Python ≥ 3.11、LangChain/LangGraph、pydantic-settings、typer + rich、prompt_toolkit、pymupdf/pdfplumber、MCP stdio 服务端。

## 2. 环境要求

| 项 | 要求 |
|---|---|
| Python | ≥ 3.11（验证环境 **3.14.7**；`arxiv-mcp-server` 要求 ≥ 3.11，`environment.yml` 与 `pyproject.toml` 约束一致） |
| 网络 | 安装依赖、检索与下载 PDF 需要；检索默认走 MCP，失败自动回退内置 HTTP |
| 密钥 | **可选**：`python main.py --offline` 无密钥可跑；真实问答需一个 OpenAI 兼容端点（`/connect` 或 `.env`） |
| node/npm | **可选**：`mcp-server-fetch` 看到 node 会在 site-packages 里 `npm install`（污染环境）；环境无 node 时自动退回纯 Python 抽取 |
| 磁盘 | `data/papers/` 存 PDF、`data/index/` 存向量；语料大时用 `PAPER_AGENT_DATA_DIR` 指到大磁盘 |

## 3. 快速开始

```bash
# 1) 环境
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"                          # 运行 + 开发依赖（构建方式见第 4 节）

# 2) 配置密钥（二选一；也可启动后用 /connect）
cp .env.example .env                             # 填 DASHSCOPE_API_KEY（或 DEEPSEEK_API_KEY / 通用 LLM_*）

# 3) 跑起来
python main.py                                   # 交互式 REPL：/help 看全部命令
python main.py --offline                         # 无密钥自检（假模型 + 独立索引目录）
```

自检命令：

```bash
python -m src.paper_agent selftest     # 模型 / embedding 是否可用（需已配置密钥）
python -m src.paper_agent mcp-tools    # MCP server 与工具白名单（可选）
python -m pytest                       # 483 passed，全部离线
```

## 4. 构建与安装

### 4.1 依赖与配置的唯一来源：`pyproject.toml`

本项目不靠 `setup.py`/`requirements.txt`/`pytest.ini`/`mypy.ini`，所有构建、依赖、工具配置集中在 `pyproject.toml`：

| 段 | 内容 |
|---|---|
| `[build-system]` | `setuptools>=68` + `setuptools.build_meta` |
| `[project]` | 名称 `paper-agent`、版本 `0.1.0`、`requires-python = ">=3.11"`、**138 个精确 pin 的运行依赖** |
| `[project.optional-dependencies].dev` | 9 个开发依赖（pytest / pytest-asyncio / mypy 等），`pip install -e ".[dev]"` 安装 |
| `[project.optional-dependencies]`（注释块） | 可选增强（BM25 混合检索、FAISS/Qdrant、本地 embedding、LangSmith…），需要时取消注释 |
| `[tool.setuptools.packages.find]` | `include = ["src.paper_agent*"]`，打包后导入路径仍是 `src.paper_agent` |
| `[tool.pytest.ini_options]` | `testpaths` / `pythonpath` / `asyncio_mode=auto` / `addopts=-q` / `filterwarnings` |
| `[tool.mypy]` | `python_version` / `explicit_package_bases` / `mypy_path` / `ignore_missing_imports` / `exclude` |

依赖策略：

- **运行依赖是完整闭包（精确 `==`）**：由干净 venv 里 `pip freeze` 生成，含全部传递依赖、无 `@ file://` 本地路径 → 换机器 `pip install` 即得同一套版本；
- `numpy` 被显式列出：`langchain-core` 的 `InMemoryVectorStore` 算余弦相似度需要它，但上游未声明为硬依赖；
- `mcp==1.30.0`：`langchain-mcp-adapters 0.3.2` 要求 `mcp<2`，**请勿单独升级 mcp 到 2.x**；
- `environment.yml` 只列 26 条直接依赖（conda 求解用），其每条约束都能被 `pyproject.toml` 的精确 pin 满足，两份清单不会打架。

> 不要另外新建 `pytest.ini` / `mypy.ini` / `requirements.txt`：pytest 与 mypy 会优先读旧式文件，导致 `pyproject.toml` 里的配置不生效。

### 4.2 方式一：pip + venv（推荐）

```bash
git clone <repo-url> paper_agent && cd paper_agent
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e .                                  # 运行依赖（可编辑安装）
pip install -e ".[dev]"                           # 需要跑 pytest / mypy 时
```

- `-e`（editable）把仓库根挂进环境，改代码立即生效，适合开发；
- `pip install .` 是普通安装（拷贝进 site-packages），改源码后需重装；
- 安装后 `python -m src.paper_agent ...` 在**任意目录**可用（包导入路径保持 `src.paper_agent`）。

### 4.3 方式二：conda

```bash
conda env create -f environment.yml     # 环境名 agent
conda activate agent
python -m src.paper_agent selftest      # 自检
```

`environment.yml` 只负责「把直接依赖装齐、由 conda 求解版本」；要复刻精确版本用 4.2 的 pip 方式，或在 conda 环境内 `pip install -e .`（两者混用时注意不要重复安装冲突包）。

### 4.4 方式三：不安装，直接在仓库根运行

项目就是这么设计的（包路径 `src.paper_agent`，`src/` 是 PEP 420 命名空间目录），仓库根在 `sys.path` 上即可导入：

```bash
cd paper_agent
python main.py --offline                      # main.py 会自己把仓库根塞进 sys.path
python -m src.paper_agent --help              # 需要 cwd = 仓库根（或已 pip install -e .）
```

适合「先试一下」「不想污染环境」；但 IDE 补全、`mypy`/`pytest` 之外的跨目录调用仍建议装成 editable。

### 4.5 方式四：构建 wheel / 内网离线安装

```bash
# 有网机器：项目 + 全部依赖的 wheel 一起放进 dist/（体积较大，但目标机完全离线可装）
python -m pip wheel -w dist .
# 只构建项目自身的 wheel（不含依赖）
python -m pip wheel --no-deps -w dist .       # → dist/paper_agent-0.1.0-py3-none-any.whl

# 目标机器（内网/无网）
python -m venv .venv && . .venv/bin/activate
pip install --no-index --find-links dist paper-agent
```

wheel 内只含 `src/paper_agent/**`（`build/`、`*.egg-info/` 等构建产物已被 `.gitignore` 忽略）。

### 4.6 依赖升级与版本策略

```bash
# 改 pyproject.toml 里的 pin 后重装
pip install -e ".[dev]"

# 临时试装某个新版本，确认无误后再回写 pyproject.toml 的 pin
pip install -U <包名>
```

升级后请至少跑 `python -m pytest` 与 `python -m src.paper_agent selftest`；涉及 MCP 相关包时复核 `mcp<2` 约束。

### 4.7 安装自检清单

| 检查 | 命令 | 期望 |
|---|---|---|
| 环境可导入 | `python -c "import src.paper_agent, langchain, langgraph; print('ok')"` | `ok` |
| 无密钥可跑 | `printf '/index\n/exit\n' \| python main.py --offline` | 正常进入 REPL 并退出 |
| 模型可用 | `python -m src.paper_agent selftest` | chat / embedding 均通过 |
| MCP 可用 | `python -m src.paper_agent mcp-tools` | 列出白名单工具；缺 server 会给出提示 |
| 测试 | `python -m pytest` | 483 passed（全离线） |
| 类型检查 | `python -m mypy src main.py` | 见 [11. 测试与类型检查](#11-测试与类型检查) |

## 5. 配置（供应商 / 密钥 / 渠道）

两种方式，**JSON 配置优先于 `.env`**：

1. **交互（推荐）**：REPL 里 `/connect` —— 输入 base URL + API key，自动识别供应商、拉取 `/models`、分类对话/embedding 模型并写入项目内 `.paper-agent/config.json`（0600）。配置跟着项目目录走，换系统直接拷贝即可；旧版 `~/.config/paper-agent/config.json` 会自动迁移过来。渠道（`/channels`）配置也落在同一个文件里。
2. **`.env`**：复制 `.env.example` 后直接填。三种方案：`DASHSCOPE_*`（对话 + embedding）、`DEEPSEEK_*`（仅对话）、通用 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`；embedding 可单独指定 `EMBED_BASE_URL` / `EMBED_API_KEY` / `EMBED_MODEL`（对话用 A 家、embedding 用 B 家）。

> JSON 配置 > `.env`；想只用 `.env` 设 `PAPER_AGENT_IGNORE_USER_CONFIG=1`。
> 换 embedding 模型/维度时索引自动切到 `data/by-embedding/<签名>/`，不会维度混用。
> `.env` 与 `.paper-agent/` 都已被 gitignore，**不要提交密钥**。

## 6. 移植到另一台机器

一句话：**项目把全部运行状态都放在项目目录内**（配置、历史、论文库、索引、报告、日志），所以移植 = **拷贝项目目录 + 在新机器重建 Python 环境 +（按需）复用 `data/`**。唯一的例外是 MCP 可执行文件——它们跟着 Python 环境走，必须在新机器上重新安装依赖（或用 `*_MCP_BIN` 指过去）。

### 6.1 目录与状态文件：该拷 / 不该拷

| 路径 | 内容 | 移植时 |
|---|---|---|
| 源码、`pyproject.toml`、`environment.yml`、`README.md`、`docs/`、`.env.example` | 项目本体 | **必须拷** |
| `.env` | `.env` 方式的密钥/参数 | 拷（含密钥：拷贝后 `chmod 600 .env`） |
| `.paper-agent/config.json` | `/connect` 的供应商 + 渠道配置（0600） | 拷（含密钥：同上；没有就用 `/connect` 重配） |
| `.paper-agent/history*` | 命令行历史 | 可选 |
| `data/papers/` | 论文库：下载的 OA 全文 PDF | 建议拷（省去重新抓取） |
| `data/index/`、`data/by-embedding/<签名>/` | 向量索引（按 embedding 签名隔离） | 建议拷，**但见 6.4 的签名约束** |
| `data/mcp/<server>/` | stdio MCP server 共享缓存（arXiv PDF/LaTeX、paper-search 下载） | 可选（可重建；`MCP_STORAGE_DIR` 可改到大磁盘） |
| `output/` | 历史报告（`.md` + `.bib` + `.json`） | 可选 |
| `.paper-agent/pdf-server.json` | 上一次 PDF 预览服务的注册信息（pid/端口） | **不必拷**：陈旧记录启动时会自动清理，删掉更干净 |
| `logs/` | 运行日志（按天分文件，默认 DEBUG） | 不必拷 |
| `.venv/`、`__pycache__/`、`*.pyc`、`.pytest_cache/`、`.mypy_cache/`、`build/`、`*.egg-info/` | 环境与缓存 | **不要拷**（目标机重建） |

> `tests/` 默认被 `.gitignore` 忽略：迁移时若需要测试，用 `rsync` 整体拷目录，或从 `.gitignore` 删除 `tests/` 段后纳入版本管理。

### 6.2 逐步操作

```bash
# 1) 拷贝目录（示例：保留环境相关产物不拷）
rsync -a --exclude '.venv' --exclude '__pycache__' --exclude '.mypy_cache' \
         --exclude '.pytest_cache' --exclude 'build' --exclude '*.egg-info' \
         paper_agent/ user@new-host:~/projects/paper_agent/

# 2) 在新机器重建环境（任选一种；内网用 4.5 的 dist/ 轮子目录）
cd paper_agent
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# 3) 恢复密钥 / 渠道：.env 与 .paper-agent/config.json 已随目录拷来；
#    没有就 python main.py 后 /connect（渠道用 /channels add）

# 4) 自检 + 验收（见 6.5）
```

### 6.3 数据目录可以搬走

`PAPER_AGENT_DATA_DIR`（默认 `data`）、`PAPER_AGENT_OUTPUT_DIR`（默认 `output`）、`PAPER_AGENT_LOG_DIR`（默认 `logs`）、`PAPER_AGENT_CONFIG`（默认 `.paper-agent/config.json`）、`MCP_STORAGE_DIR`（默认 `data/mcp`）都支持绝对路径，可把论文库/索引放到大磁盘：

```bash
PAPER_AGENT_DATA_DIR=/mnt/bigdisk/paper-agent-data python main.py
```

### 6.4 跨平台与常见坑

| 事项 | 说明 |
|---|---|
| **embedding 签名** | 索引 manifest 里记着 `模型:维度`。新机器用**相同** embedding 配置 → 旧索引直接可用；换了模型/维度 → 自动改用 `data/by-embedding/<新签名>/`（旧索引不会被破坏）。要用回旧索引就换回原模型，或重新 `/ingest`（`data/papers/` 里的 PDF 会被复用，不会重新下载） |
| **MCP 可执行文件** | 按「当前 Python 环境的 `bin`/`Scripts` → `PATH`」探测（`arxiv-mcp-server`、`paper-search-mcp`、`crossref-mcp`、`pubmedmcp`、`mcp-server-fetch`），所以必须在新机器上装依赖；特殊部署可用 `ARXIV_MCP_BIN` / `PAPER_SEARCH_MCP_BIN` / `FETCH_MCP_BIN` / `CROSSREF_MCP_BIN` / `PUBMED_MCP_BIN` 指定绝对路径 |
| **node/npm** | 若 PATH 里有 node，`mcp-server-fetch` 会在 site-packages 里 `npm install`（污染环境）；无 node 时自动退回纯 Python 抽取，行为一致 |
| **相对路径基准** | 以上所有相对目录都相对**仓库根**解析，与 cwd 无关（配置了绝对路径则原样使用） |
| **Windows** | 激活脚本为 `.venv\Scripts\activate`；包内路径全部用 `pathlib`，无 shell 依赖；命令行历史由 prompt_toolkit 管理 |
| **无网/内网** | 在能上网的机器 `python -m pip wheel -w dist .`，把 `dist/` 一起带走，目标机 `pip install --no-index --find-links dist paper-agent` |
| **端口** | PDF 预览默认 `127.0.0.1:8765`，被占用自动换空闲端口；远程机器用 `ssh -L <port>:127.0.0.1:<port>` 转发 |
| **conda** | 环境名固定 `agent`（`environment.yml` 的 `name`），与其他项目隔离 |
| **密钥安全** | `.env`、`.paper-agent/config.json` 均为明文密钥，权限设 0600，勿入 git（`.gitignore` 已忽略） |

### 6.5 移植后验收

```bash
python -m src.paper_agent selftest        # 模型 / embedding
python -m src.paper_agent mcp-tools       # MCP server 是否齐全
python -m pytest                          # 483 passed（离线，无需密钥）
printf '/index\n/exit\n' | python main.py --offline   # 无密钥冒烟
python main.py                            # 交互试用 /search、/ask
```

## 7. 使用

### 7.1 交互式（`main.py`）

```bash
python main.py                         # 进入 REPL（输入 / 弹命令面板，/help 看全部命令）
python main.py "跨块图增强解决了什么问题？"   # 一次性问答（流式 + 引用校验）
python main.py --search "graph rag"    # 一次性检索
python main.py --ingest "graph rag"    # 一次性入库
python main.py --report "主题" --papers 3
python main.py --quick "问题"           # 即抓即答（全文只在内存，PDF 不落盘）
python main.py --offline               # 假模型 + 独立索引目录（无密钥自检）
```

进入 REPL 后**直接输入自然语言 = `/ask`**。常用命令：

| 命令 | 作用 |
|---|---|
| `/search <词> [--limit N] [--ingest [N]] [--source auto\|mcp\|builtin\|all] [--no-llm]` | 联网检索；`--limit N` = **每个渠道**最多 N 条（不随 LLM 扩展的检索式数量放大） |
| `/ingest <词> [--limit N] [--ids arxiv:xxx] [--force]` | 下载入库，建立/更新索引；没有 PDF 直链时逐级降级：补链 → 网页正文 → 仅摘要/仅题录（标注为非全文） |
| `/ask <问题> [--papers a,b] [--k N]` | 带引用问答（默认流式） |
| `/quick <问题> [--papers N] [--limit N] [--k N]` | **即问即答**：现场从已启用渠道抓全文 → 内存 RAG → 带引用回答，**PDF 与索引都不落盘**（见 7.2） |
| `/report <主题> [--papers N] [--simple]` | 端到端报告 → `output/<时间戳>-<slug>.{md,bib,json}` |
| `/papers [rm <id>\|--all]` · `/papers open` · `/index` · `/mcp` | 论文 / 索引 / MCP 工具；`open` 起本地预览，用浏览器看抓到的 PDF（见 7.3）；`close` 停服务 |
| `/channels [add\|rm\|key-rm\|on\|off\|all on\|domestic on]` | **搜索渠道配置**（见第 8 节） |
| `/connect` · `/models` · `/providers` · `/model` · `/embed` · `/keys` | 供应商与模型（`/embed` 单独指定 RAG embedding） |
| `/offline [on\|off]` · `/stream [on\|off]` · `/history` · `/save` · `/logs [n] [--files]` · `/clear` · `/exit` | 会话管理（`/logs` 看今天日志末尾，`/logs --files` 列历史日志文件） |

交互特性：Tab 补全 + 历史；`/models` 输入即筛选（Enter 本次使用、Ctrl+C 设为默认、Ctrl+P 换供应商）；流式输出 append-only（不重绘）；单条命令报错/`Ctrl-C` 不退出会话；支持管道 `printf '/index\n/exit\n' | python main.py`。

### 7.2 即问即答，PDF 不落盘（`/quick`）

```bash
/quick 图 RAG 在企业知识库里的主要做法      # 现场抓 3 篇全文 → RAG → 带引用回答
/quick arxiv:2405.16506 --papers 1        # 也可以直接给 ID（不走渠道检索）
/quick 某个主题 --papers 5 --limit 12     # 多抓几篇 / 放宽每渠道候选上限
```

- **与 `/ask` 的区别**：`/ask` 只问**已入库**语料；`/quick` 即问即用——现场从已启用渠道（`/channels`）检索候选，把 PDF 抓到**内存**、`pymupdf` 从字节流解析、切分后写进**临时向量索引**，回答完即释放。
- **磁盘上不多一个文件**：不写 `data/papers/*.pdf`、不写 `data/index/`、也不建临时文件（`fetch_pdf_bytes()` + `parse_pdf_bytes()` + `PaperIndex(persist=False)`）。
- 抓不到 PDF 时同样逐级降级（网页正文 → 仅摘要 → 仅题录），命令输出里的表格会标出每篇用的是哪一级。
- 想留档：`/ingest --ids <id>`（或 `/search <词> --ingest`）。CLI 等价：`python -m src.paper_agent quick "问题" --papers 3`。

### 7.3 用浏览器看抓到的 PDF（`/papers open`）

```bash
/papers open                  # 终端打印 http://127.0.0.1:8765/，并尝试自动开浏览器
/papers open --port 9000      # 指定端口（被占用时自动换空闲端口）
/papers open --idle-timeout 0 # 关闭「空闲自动退出」（默认 30 分钟）
/papers open --no-browser     # 只打印地址，不自动开浏览器
/papers close                 # 停服务（包括别的进程起的那个）
```

也可以不经 REPL，直接当命令行工具用（适合放到后台/开机脚本）：

```bash
python -m src.paper_agent papers-open            # 起服务，Ctrl+C 退出
python -m src.paper_agent papers-open -p 9000 --idle 0 --no-browser
python -m src.paper_agent papers-close           # 停掉正在跑的服务（含其它进程）
```

服务只监听 `127.0.0.1`，页面左侧列出 `data/papers/` 里的全部 PDF（标题 / 年份 / chunks / 大小，可按关键词过滤），点击即在右侧内嵌阅读；支持 `Range` 请求，大文件拖动不卡。**在远程机器上**用 SSH 端口转发即可在本地浏览器打开：

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<远程主机>
# 然后本地浏览器打开 http://127.0.0.1:8765/
```

新入库的论文刷新页面即可看到（列表每次请求都重新扫描），不需要重启服务。

**退出机制（四层，总有一条能用）**：

| 方式 | 说明 |
|---|---|
| 页面右下角「停止预览服务」 | POST `/shutdown`，带启动时随机生成的 token（防其它本地页面误关）；关掉页面**不会**退服务 |
| `/papers close` · `papers-close` | REPL 内/命令行都行；本进程起的直接停，注册在案的**别的进程**用 SIGTERM 停 |
| 进程退出 | `atexit` + `SIGTERM` 处理，Ctrl+C / `kill` / REPL 退出都会收尾并清注册信息 |
| 空闲自动退出 | 默认 30 分钟无请求自己停（`--idle-timeout MIN` 或 `PAPER_AGENT_PDF_IDLE_MIN`，0 = 关闭） |

### 7.4 脚本式 CLI（`python -m src.paper_agent`）

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

## 8. 检索渠道（默认全部禁用）

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

**没有 PDF 直链时**不会直接放弃，入库按四级降级（结果里的 `status` 会说明用了哪级）：
`indexed`（PDF 全文）→ `web`（落地页/百科/维基/新闻等网页正文）→ `abstract`（仅摘要）→ `metadata`（仅书目题录）。
后两级会在 chunk 里写入「[仅摘要/仅题录（未获取全文）]」标注，引用上下文也会显式提醒模型，避免被当成论文原文证据。
补链依次尝试 `arxiv.org/pdf/<id>` → Unpaywall → OpenAlex → 落地页 `citation_pdf_url` / `.pdf` 链接（配 `UNPAYWALL_EMAIL` 覆盖更全）。
用 `PDF_LOOKUP=0` / `WEB_FALLBACK=0` / `RECORD_FALLBACK=0` 可分别关掉补链、网页正文、题录兜底。

抓取顺序：`--source auto`（默认）= MCP 优先，不可用/无结果回退内置 HTTP；`mcp` / `builtin` / `all`（两者合并）。

## 9. 项目结构

代码按「基础设施 → 领域 → 编排 → 入口」分层，`src/paper_agent/` 下每层一个子包（各文件职责见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)）：

```
main.py                  交互式入口（REPL / 一次性命令）
pyproject.toml           项目元数据 + 依赖 + pytest/mypy 配置（唯一配置源）
environment.yml          conda 环境（只列直接依赖）
src/paper_agent/
  __main__.py            `python -m src.paper_agent` 入口
  cli.py                 typer 子命令入口
  core/                  基础设施：config / logging / schema / utils / ui
  sources/               来源层：channels（渠道）/ fetchers（内置抓取）/ mcp / userconfig（凭证）
  llm/                   模型层：factory（ChatModel·Embeddings）/ search（查询扩展·重排）/ fake
  rag/                   RAG：fetch / parse / split / embeddings / store / retriever
  tools/                 agent 工具：paper_tools / rag_tools
  agents/                多 agent：search / summarize / rag / writer / supervisor（LangGraph）
  pipeline/              编排层：session（可复用流水线）/ report（Markdown·BibTeX·JSON）
  pdf/                   本地 PDF 预览服务
  repl/                  交互层：app（Repl）/ commands/* / input / ui / tui / base
```

- 只有 `main.py`、`cli.py`、`__main__.py` 在包根，其余按层归入子包；跨层引用一律写全路径（如 `from ..pipeline.session import ask`）。
- `pipeline/session.py` 是唯一业务入口：REPL 与 CLI 都只调它，保证行为一致。

## 10. 常用环境变量

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
| `PAPER_AGENT_DATA_DIR` / `PAPER_AGENT_OUTPUT_DIR` | `data` / `output` | 论文库·索引 / 报告输出目录（可绝对路径） |
| `PAPER_AGENT_CONFIG` / `PAPER_AGENT_IGNORE_USER_CONFIG` | `.paper-agent/config.json` / 空 | `/connect` 配置文件位置 / 只用 `.env` |
| `PAPER_AGENT_THEME` | `dark` | 界面配色 `dark` / `light` / `none` |
| `PAPER_AGENT_LOG_DIR` / `PAPER_AGENT_LOG_LEVEL` | `logs` / `DEBUG` | 日志目录（项目内，按天分文件）与文件级别；`PAPER_AGENT_LOG_KEEP_DAYS`（默认 14）、`PAPER_AGENT_LOG_MAX_MB`（默认 8，超出续写 `-02`）；`PAPER_AGENT_LOG_DISABLE=1` 关文件日志；`PAPER_AGENT_LOG_FILE` 改成固定单文件 |
| `MCP_STORAGE_DIR` | `data/mcp` | stdio MCP server 的共享存放目录（每个 server 一个同名子目录）；`ARXIV_MCP_BIN` / `PAPER_SEARCH_MCP_BIN` / `FETCH_MCP_BIN` / `CROSSREF_MCP_BIN` / `PUBMED_MCP_BIN` 可覆盖各自的可执行文件 |
| `PAPER_AGENT_PDF_IDLE_MIN` / `PAPER_AGENT_PDF_REGISTRY` | 30 / `.paper-agent/pdf-server.json` | 预览服务空闲退出分钟数（0 = 不退）/ 注册文件位置 |
| `OPENALEX_MAILTO` | 空 | 进 OpenAlex/Crossref polite pool（更稳；也会透传给 `crossref-mcp` 作 `CROSSREF_MAILTO`） |
| `UNPAYWALL_EMAIL` | 空 | 无直链时用 Unpaywall 补链查 OA 全文（留空则退回 `OPENALEX_MAILTO`） |
| `PDF_LOOKUP` / `WEB_FALLBACK` / `RECORD_FALLBACK` | `true` | 无 PDF 直链时的三级兜底：补链 / 抓网页正文 / 入库题录·摘要（`WEB_TEXT_MIN_CHARS` 默认 400，网页正文短于此判为无效） |
| `HYBRID_RETRIEVAL` | `false` | 混合检索（BM25 + 向量）；需按 `pyproject.toml` 注释块安装 `rank-bm25` + `langchain-community` |

完整清单见 [`.env.example`](.env.example)。

## 11. 测试与类型检查

```bash
python -m pytest          # 483 passed，全部离线（假模型 / 假 embedding / 假 MCP server）
python -m mypy src main.py
```

配置全部来自 `pyproject.toml`（`[tool.pytest.ini_options]` / `[tool.mypy]`），**不要**再建 `pytest.ini` / `mypy.ini`（旧式文件优先级更高，会把这里的配置压掉）。

- `python -m pytest` 不需要任何密钥：测试用假模型、假 embedding、假 MCP server；
- `python -m mypy src main.py`：当前依赖版本下会报 6 条第三方类型漂移错误（`llm/factory.py`、`repl/input.py`、`core/logging.py`、`cli.py`），与本项目的构建/移植无关，升级 `langchain-openai` / `prompt_toolkit` / `readline` 相关依赖后需复核；
- `tests/` 默认被 `.gitignore` 忽略（本地保留即可跑；要入库请从 `.gitignore` 删除 `tests/` 段）。

## 12. 排错

| 现象 | 处理 |
|---|---|
| `pip install -e .` 失败/装不干净 | 确认 Python ≥ 3.11、`pip install -U pip` 后重试；内网环境见 4.5 的轮子目录 |
| 安装后 `import src.paper_agent` 失败 | 确认在同一个虚拟环境里执行；未安装时需 `cd` 到仓库根再跑 |
| `/connect` 粘贴 key 后 401 | key 读入会回显脱敏结果确认真假；重新 `/connect` 覆盖更新 |
| `/ingest --ids arxiv:xxxx` 报「解析失败」 | 先看日志：arXiv **元数据** API 限流（429）时会自动降级用 `arxiv.org/pdf/<id>` 直链抓 PDF，标题从 PDF 首页补；若仍失败才是 ID 写错/无 OA 全文 |
| 检索「只启用了 X 却返回别的源」 | 已按 `/channels` 严格收敛；若仍如此，请**重启 REPL**（长驻进程不会热加载源码） |
| `429` / `403` | 面向 OpenAlex/Crossref 配 `OPENALEX_MAILTO`；需要登录的渠道按提示 `/channels add` 配置 |
| embedding `batch size is invalid` | 调小 `EMBED_BATCH_SIZE`（已内置自动降批） |
| `IndexSignatureError` | 索引与当前 embedding 不匹配：`build_session()` 通常已自动切到 `data/by-embedding/<签名>/`；也可换回原模型，或删掉索引重新 ingest |
| MCP 工具缺失 / server 起不来 | `python -m src.paper_agent mcp-tools` 看探测结果；确认依赖已装在**当前**环境，或用 `*_MCP_BIN` 指绝对路径 |
| 引用校验误报 | 校验是启发式的，可调低 `MIN_SUPPORT_RATIO` |
| MCP 日志刷 Semantic Scholar 429 | 未配 `SEMANTIC_SCHOLAR_API_KEY` 时该源不暴露；配 key 后自动启用 |
| 交互追问（`/connect`、y/N 确认）时按了 `Ctrl+D` | 视为「取消当前操作」并回到提示符（不会打 traceback、不会写入配置）；整行留空亦然 |
| 想知道为什么回退/失败但终端没显示 | 日志全量写在项目内 `logs/paper-agent-YYYY-MM-DD.log`（**按天分文件**，默认 DEBUG，保留 14 天）：REPL 里 `/logs 50` 看今天末尾、`/logs --files` 找更早的文件 |
| `/papers open` 打印的地址在本地浏览器打不开 | 服务只绑 `127.0.0.1`：远程机器先 `ssh -L <port>:127.0.0.1:<port> <host>` 转发；确实要在内网直连再加 `--host 0.0.0.0` |
| 预览服务忘了关 / 端口被占用 | `/papers close`（或 `python -m src.paper_agent papers-close`）会停掉**任何进程**注册在案的服务；默认空闲 30 分钟也会自动退，`PAPER_AGENT_PDF_IDLE_MIN=0` 才关掉这个保护 |

## 13. 已知限制

1. **引用校验是启发式的**：跨语言靠术语/数字"硬令牌"匹配，无法核验时跳过并标注，不静默丢弃。
2. **只用开放获取**：已屏蔽 `download_scihub` 与 `search_google_scholar`；部分出版社（如 MDPI）会 403。
3. **默认内存向量库 + 本地 JSON**：适合单次调研（数百篇内），更大语料建议换 FAISS/Qdrant（替换 `PaperIndex` 内部实现即可）。
4. **国内库边界**：ChinaXiv、国家图书馆有公开免 key 接口；百度学术需千帆 key、万方需 APPCODE；知网/维普/超星**无公开检索 API**，不做绕过抓取（可用官方题录导出后走 `/ingest --ids`）。
5. **MCP 工具数量**：`paper-search-mcp` 暴露 57 个工具、`crossref-mcp` 18 个，靠白名单过滤，勿关闭。
6. **离线模式索引独立**：`--offline` 用假 embedding，固定 `data/offline/`，与真实索引不混用。
