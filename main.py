#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学术论文检索与概括分析 Agent —— 交互式命令行入口。

用法：
    python main.py [问题]                  # 进入 REPL；带问题则一次性问答
    python main.py --search "graph rag"    # 一次性检索
    python main.py --ingest "graph rag"    # 一次性入库
    python main.py --report "主题" --papers 3
    python main.py --offline                # 假模型 + 独立索引目录（无密钥自检）

进入 REPL 后输入 `/` 弹出命令面板（Tab 补全、Enter 确认），`/help` 查看全部命令；
直接输入自然语言等价于 `/ask`（在当前 RAG 索引上带引用问答）。

检索默认走 MCP，不可用或没结果时回退内置公开接口（arXiv/OpenAlex/Crossref/…，免 key）；
**所有渠道默认禁用**，用 `/channels add <编号|kind>` 添加后才参与检索。
`/channels` 可添加 Semantic Scholar/CORE/Tavily 及国内库（ChinaXiv/国家图书馆免 key，百度学术/万方需 key）。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Callable, Literal

# 允许 `python main.py` 直接运行（仓库根目录入 path）
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console, Group  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from src.paper_agent.cli import cli_settings, papers_table  # noqa: E402
from src.paper_agent.channels import (  # noqa: E402
    REGISTRY,
    channel_label,
    is_domestic,
    list_specs,
    presets as channel_presets,
    spec_for,
)
from src.paper_agent.config import get_settings  # noqa: E402
from src.paper_agent.tui import (  # noqa: E402
    connect_flow,
    embed_models_for,
    pick_from_list,
    print_presets,
    read_secret,
)
from src.paper_agent.repl_input import PaletteContext, create_session, read_line  # noqa: E402
from src.paper_agent.utils import clean_pasted, mask_secret  # noqa: E402
from src.paper_agent.userconfig import (  # noqa: E402
    UserConfig,
    detect_provider,
    normalize_base_url,
    probe_embedding_dim,
    resolve_embedding,
    settings_overrides,
)
from src.paper_agent.pipeline import (  # noqa: E402
    Session,
    ask as pipeline_ask,
    build_session,
    collapse_repetition,
    ingest_papers,
    remove_papers as pipeline_remove_papers,
    run_ingest,
    run_report,
    run_search,
)

console = Console()
HISTORY_FILE = Path.home() / ".paper_agent_history"

PROMPT = "[bold cyan]paper-agent[/bold cyan] [dim]›[/dim] "

# 命令名 → 一句话说明（同时用于 Tab 补全与 /help；键只放命令名，别塞参数）
COMMANDS: dict[str, str] = {
    "/search": "联网检索论文（--limit = 每渠道上限；可用 LLM 扩展检索式并重排）",
    "/ingest": "下载 PDF 并入库，建立/更新向量索引",
    "/ask": "基于已入库语料带引用问答（直接输入自然语言同效）",
    "/report": "端到端生成调研报告（md / bib / json）",
    "/papers": "查看或删除已入库论文",
    "/channels": "配置搜索渠道；国内优先 / 全渠道并发开关",
    "/index": "查看索引统计（chunks / 论文数）",
    "/mcp": "查看 MCP server 与白名单工具",
    "/connect": "配置模型供应商（自动识别 + 拉模型 + 写 JSON）",
    "/models": "模型选择器（本次使用 / 设为默认 / 换供应商）",
    "/providers": "查看、切换、删除供应商，或只删除其 API key",
    "/keys": "查看并删除已保存的 API key（供应商 + 搜索渠道）",
    "/model": "查看或切换对话模型",
    "/embed": "单独指定负责 RAG embedding 的供应商与模型",
    "/offline": "切换离线模式（假模型 + 独立索引目录，不联网）",
    "/stream": "切换流式输出",
    "/history": "查看最近问答记录",
    "/save": "把本次会话问答保存为 Markdown",
    "/clear": "清屏",
    "/exit": "退出",
    "/help": "显示本帮助（可跟命令名，如 /help search）",
}

# 用法签名（/help 里单独一行展示）
COMMAND_USAGE: dict[str, str] = {
    "/search": "/search <关键词> [--limit N（每渠道最多 N 条）] [--ingest [N]] [--source auto|mcp|builtin|all] [--no-llm]",
    "/ingest": "/ingest <关键词> [--limit N] [--force]　或　/ingest --ids arxiv:2405.16506,10.1145/xxx",
    "/ask": "/ask <问题> [--papers a,b] [--k N]",
    "/report": "/report <主题> [--papers N] [--simple]",
    "/papers": "/papers [rm <id>|--all]",
    "/channels": "/channels [list|add|rm|key-rm|on|off|all on|off|domestic on|off]",
    "/index": "/index",
    "/mcp": "/mcp",
    "/connect": "/connect [base_url] [api_key] [--name N] [--kind K]",
    "/models": "/models [--all] [--embedding] [--provider NAME] [--refresh]",
    "/providers": "/providers [use [name]|rm [name]|key-rm <name>|sync [name]]",
    "/keys": "/keys [rm provider:<name>|channel:<name>|<name>]",
    "/model": "/model [name] [--default]",
    "/embed": "/embed [供应商] [模型] 　或　/embed --model <模型> 　或　/embed auto",
    "/offline": "/offline [on|off]",
    "/stream": "/stream [on|off]",
    "/history": "/history [n]",
    "/save": "/save [path]",
    "/clear": "/clear",
    "/exit": "/exit",
    "/help": "/help [命令名]",
}

# /help 的分组展示顺序（只列命令名）
HELP_GROUPS: list[tuple[str, list[str]]] = [
    ("检索与抓取", ["/search", "/ingest", "/channels", "/index", "/mcp"]),
    ("问答与报告", ["/ask", "/report", "/papers", "/history", "/save"]),
    ("模型与供应商", ["/connect", "/providers", "/keys", "/models", "/model", "/embed"]),
    ("会话与调试", ["/offline", "/stream", "/clear", "/help", "/exit"]),
]

# /help 末尾的常用示例
HELP_EXAMPLES: list[tuple[str, str]] = [
    ("/search graph rag --limit 5", "检索 5 篇 Graph RAG 论文"),
    ("/search graph rag --ingest", "检索并直接下载入库"),
    ("/ingest --ids arxiv:2405.16506", "已知 arXiv ID 直接抓取入库"),
    ("/channels all on", "检索时并发跑全部已注册渠道"),
    ("/ask 图 RAG 的主要方法有哪些", "基于本地语料带引用回答"),
    ("/report 图检索增强生成 --papers 3", "生成一份中文调研报告"),
]


# --------------------------------------------------------------------------
# 交互基础设施
# --------------------------------------------------------------------------


def _setup_readline() -> None:
    """启用行编辑 / 历史 / Tab 补全（stdlib readline，无需额外依赖）。"""
    try:
        import readline
    except ImportError:  # pragma: no cover - Windows 无 readline
        return

    def completer(text: str, state: int):
        options = [c for c in COMMANDS if c.startswith(text)] if text.startswith("/") else []
        return options[state] if state < len(options) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")
    try:
        readline.read_history_file(HISTORY_FILE)
    except OSError:
        pass
    readline.set_history_length(1000)


def _save_readline_history() -> None:
    try:
        import readline

        readline.write_history_file(HISTORY_FILE)
    except Exception:  # noqa: BLE001
        pass


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "httpcore", "mcp", "urllib3", "openai", "arxiv", "langchain"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _split_args(raw: str) -> tuple[str, dict[str, str]]:
    """把 `query --limit 3 --force` 拆成 (query, flags)。

    先做粘贴清洗：终端粘贴常带括号粘贴标记/控制字符，会污染 URL、key、模型名。
    """
    raw = clean_pasted(raw)
    try:
        parts = shlex.split(raw)
    except ValueError:
        parts = raw.split()

    positional: list[str] = []
    flags: dict[str, str] = {}
    i = 0
    while i < len(parts):
        token = parts[i]
        if token.startswith("--"):
            key = token[2:]
            if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                flags[key] = parts[i + 1]
                i += 2
                continue
            flags[key] = "true"
        else:
            positional.append(token)
        i += 1
    return " ".join(positional).strip(), flags


def _flag_bool(flags: dict[str, str], key: str, default: bool = False) -> bool:
    if key not in flags:
        return default
    return flags[key].lower() not in {"false", "0", "no", "off"}


class _SearchProgressView:
    """逐渠道显示检索进度：每行一个渠道，**不合并成一行**。

    `builtin_search` 在每个渠道 queued/running/done/failed/skipped 时回调 `on_event`，
    这里聚合成一张表用 Rich `Live` 实时刷新（无事件时退化为单行“检索中”提示）。
    """

    _STATUS: dict[str, tuple[str, str]] = {
        "queued": ("等待", "dim"),
        "running": ("检索中…", "cyan"),
        "done": ("完成", "green"),
        "failed": ("失败", "red"),
        "skipped": ("跳过", "yellow"),
    }

    def __init__(self, label: str, query: str) -> None:
        self.label = label
        self.query = query
        self.rows: dict[str, dict[str, Any]] = {}
        self._live: Any = None

    def bind(self, live: Any) -> None:
        self._live = live

    def on_event(self, event: dict) -> None:
        """处理单渠道进度事件（由 builtin_search 在每个渠道状态变化时调用）。"""
        name = str(event.get("name") or event.get("kind") or "?")
        row = self.rows.get(name)
        if row is None:
            row = {"status": "queued", "count": 0, "elapsed": 0.0, "detail": "", "kind": ""}
            self.rows[name] = row
        if event.get("kind"):
            row["kind"] = str(event["kind"])
        status = str(event.get("status") or row["status"])
        row["status"] = status
        count = event.get("count")
        if count is not None:
            # 多检索式时同一渠道只取最大值（结果会按渠道去重并截断到 --limit）
            row["count"] = max(int(row["count"]), int(count))
        detail = str(event.get("error") or event.get("reason") or "")
        if detail:
            row["detail"] = detail
        elapsed = event.get("elapsed")
        if elapsed is not None:
            row["elapsed"] = float(elapsed)
        if self._live is not None:
            self._live.update(self.render())

    def render(self) -> Any:
        if not self.rows:
            return Text.from_markup(
                f"[cyan]联网检索中（{self.label}）：{self.query}[/cyan]"
            )
        table = Table(
            title=f"检索进度 · {self.label}",
            title_style="bold",
            header_style="bold dim",
            box=None,
            pad_edge=False,
        )
        table.add_column("渠道", no_wrap=True)
        table.add_column("状态", no_wrap=True)
        table.add_column("条数", justify="right")
        table.add_column("耗时", justify="right", style="dim")
        for name, row in self.rows.items():
            status = str(row["status"])
            text, style = self._STATUS.get(status, (status, "white"))
            detail = str(row.get("detail") or "")
            cell = f"{text} · {detail}" if detail else text
            count = str(row["count"]) if status == "done" else "-"
            elapsed = f"{float(row['elapsed']):.1f}s" if row.get("elapsed") else "-"
            name_cell = Text(name, style="cyan")
            if is_domestic(str(row.get("kind") or "")):
                name_cell.append(" ·国内", style="bold green")
            table.add_row(name_cell, Text(cell, style=style), count, elapsed)
        return table


# --------------------------------------------------------------------------
# 会话
# --------------------------------------------------------------------------


class Repl:
    """交互式会话状态机。"""

    def __init__(self, settings=None, offline: bool = False, stream: bool = True) -> None:
        self.settings = settings or cli_settings(offline=offline)
        self.stream = stream
        self.config: UserConfig = UserConfig.load()
        self.session: Session | None = None
        self.session_error: str = ""
        try:
            self.session = build_session(self.settings)
        except RuntimeError as exc:  # 首次使用还没配置供应商/密钥
            self.session = None
            self.session_error = str(exc)
        self.history: list[dict] = []
        self._live: Live | None = None
        # 会话级持久事件循环（见 _run_async）
        self._loop: asyncio.AbstractEventLoop | None = None

    # ---------------- 基础展示 ----------------
    def require_session(self) -> Session | None:
        """需要 RAG/模型的命令统一入口：未配置时给出 /connect 引导。"""
        if self.session is not None:
            return self.session
        console.print(f"[yellow]尚未完成配置[/yellow]：{self.session_error or '缺少模型 / embedding 配置'}")
        console.print(
            "[bold]请先执行 [cyan]/connect[/cyan][/bold]：输入 base URL 与 API key 即可"
            "（自动识别供应商与模型列表；支持直接粘贴，key 不回显）。"
            "也可用 [cyan]/connect 2[/cyan] 之类预设。"
        )
        console.print("[dim]非交互方式：/connect https://api.deepseek.com sk-xxx\n[/dim]")
        return None

    def banner(self) -> None:
        if self.session is None:
            console.print(
                Panel(
                    Group(
                        Text.from_markup("[bold]学术论文检索与概括分析 Agent[/bold]  [dim](LangChain + MCP + RAG + 多 agent)[/dim]"),
                        Text.from_markup("[yellow]尚未配置模型供应商[/yellow]"),
                        Text.from_markup(
                            "执行 [bold cyan]/connect[/bold cyan] 输入 base URL + API key（OpenAI 兼容，自动识别供应商与模型列表）；"
                            "或直接在 .env 里配置 DASHSCOPE_API_KEY。"
                        ),
                        Text.from_markup("[dim]临时可用：/offline 进入离线自检模式；/help 查看所有命令[/dim]"),
                    ),
                    border_style="yellow",
                    padding=(1, 2),
                )
            )
            return
        stats = self.session.stats()
        info = Table.grid(padding=(0, 2))
        info.add_column(style="bold")
        info.add_column()
        s = self.settings
        info.add_row(
            "供应商",
            f"{s.provider_label}  [dim]{s.active_base_url or '（未配置，用 /connect 添加）'}[/dim]",
        )
        info.add_row("模型", str(stats["model"] or ("fake-llm（离线）" if stats["offline"] else "?")))
        info.add_row(
            "embedding",
            f"{s.active_embedding_model or '（无 → RAG 不可用）'}"
            + (f"  [dim]dim={s.embedding_dim}[/dim]" if s.embedding_dim else ""),
        )
        info.add_row("索引", f"{stats['papers']} 篇论文 / {stats['chunks']} chunks  [dim]{stats['data_dir']}[/dim]")
        info.add_row("MCP", f"{stats['tools']} 个工具" + ("" if stats["tools_loaded"] else " [dim]（首次检索时加载）[/dim]"))
        if self.config.providers:
            info.add_row("配置文件", f"[dim]{self.config.path}[/dim]（[cyan]/connect[/cyan] 修改）")
        info.add_row("模式", "离线自检" if stats["offline"] else "在线（真实模型 + 云端 embedding）")
        console.print(
            Panel(
                Group(
                    Text.from_markup("[bold]学术论文检索与概括分析 Agent[/bold]  [dim](LangChain + MCP + RAG + 多 agent)[/dim]"),
                    info,
                    Text.from_markup("[dim]直接输入问题即开始 RAG 问答；/help 查看命令；/exit 退出[/dim]"),
                ),
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def show_index(self) -> None:
        session = self.require_session()
        if session is None:
            return
        stats = session.stats()
        console.print(
            f"[bold]索引[/bold] {stats['papers']} 篇 / {stats['chunks']} chunks   "
            f"[dim]{stats['data_dir']}[/dim]"
        )

    def show_papers(self) -> None:
        session = self.require_session()
        if session is None:
            return
        papers = session.index.list_papers()
        if not papers:
            console.print("[yellow]索引为空，先 /ingest <主题> 或用 /search 看看能检索到什么[/yellow]")
            return
        table = Table(title=f"已入库论文（{len(papers)} 篇）")
        table.add_column("paper_id", style="cyan", no_wrap=True)
        table.add_column("chunks", justify="right")
        table.add_column("年份", justify="right")
        table.add_column("标题")
        for item in papers:
            table.add_row(
                item["paper_id"],
                str(item.get("n_chunks", 0)),
                str(item.get("published", ""))[:4],
                str(item.get("title", ""))[:80],
            )
        console.print(table)

    def cmd_papers(self, args: str = "") -> None:
        """`/papers` 列表；`/papers rm <id>|--all` 删除已入库论文。"""
        action, _, rest = args.strip().partition(" ")
        if action.lower() in {"rm", "remove", "del", "delete"}:
            self.remove_papers(rest)
            return
        if action:
            console.print("[yellow]用法：/papers | /papers rm <paper_id>[,<id>] | /papers rm --all[/yellow]")
            return
        self.show_papers()

    def remove_papers(self, arg: str) -> None:
        """从索引（与本地 PDF 缓存）中删除论文；不给 id 时弹选择器。"""
        session = self.require_session()
        if session is None:
            return
        papers = session.index.list_papers()
        if not papers:
            console.print("[yellow]索引为空，没有可删除的论文[/yellow]")
            return

        token = clean_pasted(arg).strip()
        targets: list[str] = []
        if token in {"--all", "all", "*"}:
            answer = clean_pasted(console.input(f"确认删除全部 {len(papers)} 篇论文？[y/N] › "))
            if answer.lower() not in {"y", "yes"}:
                console.print("[dim]已取消[/dim]")
                return
            targets = [p["paper_id"] for p in papers]
        elif token:
            targets = [x.strip() for x in token.split(",") if x.strip()]
        else:
            ids = [p["paper_id"] for p in papers]
            display = {
                p["paper_id"]: f"{p['paper_id']}  [dim]{str(p.get('title', ''))[:60]} · {p.get('n_chunks', 0)} chunks[/dim]"
                for p in papers
            }
            action, value = pick_from_list(
                console,
                ids,
                title="选择要删除的论文",
                display=display,
                footer="Enter 删除 / Esc 取消",
                display_class="pick.paper",
            )
            if action == "cancel" or not value:
                console.print("[dim]已取消[/dim]")
                return
            targets = [value]

        removed = pipeline_remove_papers(targets, session=session)
        for paper_id in targets:
            if paper_id in removed:
                console.print(f"[green]✓[/green] 已删除 {paper_id}（chunks + 本地 PDF）")
            else:
                console.print(f"[yellow]未找到 {paper_id}[/yellow]")
        if removed:
            self.show_index()

    def show_history(self, n: int = 5) -> None:
        if not self.history:
            console.print("[dim]还没有问答记录[/dim]")
            return
        for item in self.history[-n:]:
            console.print(f"[bold cyan]Q[/bold cyan] {item['question']}")
            console.print(Markdown(item["answer"][:1200] or "（空）"))
            cited = ", ".join(item["citations"]) or "（无）"
            status = "[green]引用校验通过[/green]" if not item["problems"] else f"[yellow]{'; '.join(item['problems'])}[/yellow]"
            console.print(f"[dim]引用：{cited}[/dim]  {status}\n")

    def save_history(self, path: str = "") -> None:
        if not self.history:
            console.print("[yellow]没有可保存的内容[/yellow]")
            return
        target = Path(path).expanduser() if path else (
            self.settings.output_path / f"session-{time.strftime('%Y%m%d-%H%M')}.md"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# 会话记录（{time.strftime('%Y-%m-%d %H:%M')}）", ""]
        for item in self.history:
            # 兼底：历史里可能存有较早（未去重）的答案，保存时再去一次复读
            answer = collapse_repetition(item["answer"])
            lines += [f"## Q: {item['question']}", "", answer, ""]
            if item["citations"]:
                lines.append(f"> 引用：{', '.join(item['citations'])}")
            if item["problems"]:
                lines.append(f"> 校验提示：{'; '.join(item['problems'])}")
            lines.append("")
        target.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]✓[/green] 已保存：{target}")

    # ---------------- 命令实现 ----------------
    def _search_channel_issues(self, progress: Any, requested: str) -> None:
        """检索后处理渠道失败：429 提示进 polite pool，401/403/需 key 则引导配置。"""
        rows = getattr(progress, "rows", None) or {}
        requested_kinds = {x.strip().lower() for x in (requested or "").split(",") if x.strip()}
        for row in rows.values():
            kind = str(row.get("kind") or "").strip().lower()
            if not kind:
                continue
            status = str(row.get("status") or "")
            detail = str(row.get("detail") or "")
            spec = spec_for(kind)
            label = spec.label if spec else kind
            if status == "failed" and detail == "429":
                console.print(
                    f"[yellow]「{label}」返回 429（请求过多）。[/yellow]"
                    f"[dim]可 /channels add {kind} --email you@example.com 进 polite pool，或稍后重试。[/dim]"
                )
            elif status == "failed" and detail in {"401", "403", "需key"}:
                self._offer_channel_setup(kind, label, detail)
            elif status == "skipped" and "需key" in detail and kind in requested_kinds:
                self._offer_channel_setup(kind, label, "需key")

    def _offer_channel_setup(self, kind: str, label: str, reason: str) -> None:
        """渠道要求登录/key 时，直接在 REPL 里引导配置。"""
        console.print(f"[yellow]「{label}」需要登录 / API key（{reason}）。[/yellow]")
        if not sys.stdin.isatty():
            console.print(f"[dim]稍后可执行：/channels add {kind}[/dim]")
            return
        answer = clean_pasted(console.input(f"现在配置 {label}？[y/N] › ")).lower()
        if answer in {"y", "yes"}:
            self.add_channel(kind)
        else:
            console.print(f"[dim]稍后可执行：/channels add {kind}[/dim]")

    def cmd_search(self, args: str) -> None:
        query, flags = _split_args(args)
        if not query:
            console.print(
                "[yellow]用法：/search <query> [--limit N]（每渠道最多 N 条）[--ingest [N]] "
                "[--sources arxiv,openalex] [--source auto|mcp|builtin] [--no-llm][/yellow]"
            )
            return
        limit = int(flags.get("limit", 8))
        source = flags.get("source", "")
        if flags.get("source"):
            self.settings = self.settings.model_copy(update={"search_source": source})
        # 检索过程中使用 LLM（默认开；--no-llm 关闭）
        use_llm: bool | None = None
        if _flag_bool(flags, "no-llm"):
            use_llm = False
        elif _flag_bool(flags, "llm"):
            use_llm = True
        label = {"builtin": "内置源+渠道", "mcp": "MCP", "all": "MCP + 内置源/渠道", "": "MCP → 内置回退"}.get(source, source)
        llm_note = "" if use_llm is False else "（LLM 扩展/重排）"
        progress = _SearchProgressView(f"{label}{llm_note}", query)
        by_channel: dict[str, list] = {}
        # 逐渠道进度：builtin_search 每当某个渠道 queued/running/done/failed/skipped 就回调 → 刷新表格
        with Live(progress.render(), console=console, refresh_per_second=10, transient=False) as live:
            progress.bind(live)
            papers, route = self._run_async(
                run_search(
                    query,
                    session=self.session,
                    limit=limit,
                    sources=flags.get("sources", ""),
                    source=source,
                    use_llm=use_llm,
                    on_event=progress.on_event,
                    per_source_limit=True,  # `/search --limit N` = 每个渠道最多 N 条
                    by_channel=by_channel,
                )
            )
        self._search_channel_issues(progress, flags.get("sources", ""))
        if not papers:
            console.print(f"[yellow]没有检索到结果（route={route}）[/yellow]")
            return
        # 分渠道显示结果（不合并成一张大表；国内渠道排最前，标题带 `·国内`）
        if by_channel:
            ordered = [k for k in by_channel if is_domestic(k)] + [
                k for k in by_channel if not is_domestic(k)
            ]
            for kind in ordered:
                seen: set[str] = set()
                items: list = []
                for p in by_channel[kind]:  # 同一渠道内去重（不跨渠道去重，保留各渠道原始命中）
                    if p.paper_id not in seen:
                        seen.add(p.paper_id)
                        items.append(p)
                tag = " ·国内" if is_domestic(kind) else ""
                title = f"{channel_label(kind, kind)}{tag}"
                if not items:
                    console.print(f"[dim]{title}：无结果[/dim]")
                    continue
                console.print(papers_table(items, title=title))
        else:
            console.print(papers_table(papers))
        console.print(f"[dim]来源：{route}[/dim]")

        # 特定参数 --ingest [N]：直接把检索结果入库，无需再跑一次 /ingest
        ingest_flag = flags.get("ingest", "") or flags.get("save", "") or flags.get("index", "")
        if not ingest_flag:
            console.print("[dim]提示：加 --ingest 可直接把这些结果入库（/search <query> --ingest）[/dim]")
            return
        self._ingest_found(papers, ingest_flag, force=_flag_bool(flags, "force"))

    def _ingest_found(self, papers: list, ingest_flag: str, force: bool = False) -> None:
        """把刚检索到的 `Paper` 直接入库（`/search ... --ingest [N]`）。"""
        session = self.require_session()
        if session is None:
            console.print("[yellow]需要配置 embedding（/connect）后才能入库[/yellow]")
            return
        count = int(ingest_flag) if str(ingest_flag).isdigit() else len(papers)
        selected = papers[: max(1, count)]
        with console.status(f"[cyan]直接入库 {len(selected)} 篇（下载 → 解析 → 向量化）…[/cyan]"):
            results = self._run_async(ingest_papers(selected, session=session, force=force))
        if not results:
            console.print("[yellow]没有可入库的论文[/yellow]")
            return
        for info in results:
            style = {"indexed": "green", "cached": "cyan"}.get(info["status"], "yellow")
            console.print(
                f"[{style}]{info['status']:>11}[/{style}] {info['paper_id']}  {info['chunks']} chunks  {info['message']}"
            )
        self.show_index()

    def cmd_ingest(self, args: str) -> None:
        query, flags = _split_args(args)
        if not query and not flags.get("ids"):
            console.print(
                "[yellow]用法：/ingest <query> [--limit N] | /ingest --ids arxiv:2405.16506,10.1145/xxx "
                "[--source builtin] [--force][/yellow]"
            )
            return
        session = self.require_session()
        if session is None:
            return
        limit = int(flags.get("limit", 3))
        if flags.get("source"):
            session.settings = session.settings.model_copy(update={"search_source": flags["source"]})
        with console.status("[cyan]抓取 → 下载 → 解析 → 向量化…[/cyan]"):
            results = self._run_async(
                run_ingest(
                    query or "",
                    session=session,
                    limit=limit,
                    ids=flags.get("ids", ""),
                    force=_flag_bool(flags, "force"),
                )
            )
        if not results:
            console.print("[yellow]没有可入库的论文[/yellow]")
            return
        for info in results:
            style = {"indexed": "green", "cached": "cyan"}.get(info["status"], "yellow")
            console.print(
                f"[{style}]{info['status']:>11}[/{style}] {info['paper_id']}  {info['chunks']} chunks  {info['message']}"
            )
        self.show_index()

    def cmd_ask(self, question: str) -> None:
        question = question.strip()
        if not question:
            console.print("[yellow]用法：/ask <question>[/yellow]")
            return
        session = self.require_session()
        if session is None:
            return
        if session.index.chunk_count == 0:
            console.print(
                "[yellow]索引为空。[/yellow]先执行 [cyan]/ingest <主题>[/cyan] 抓几篇论文，"
                "或 [cyan]/search <主题>[/cyan] 先看看检索结果。"
            )
            return

        on_token = self._stream_renderer() if self.stream and not session.offline else None
        deadline = max(1.0, self.settings.llm_timeout) + 30.0
        try:
            with console.status("[cyan]检索 + 生成中…[/cyan]") if on_token is None else _nullcontext():
                result = self._run_async(
                    asyncio.wait_for(
                        pipeline_ask(question, session=session, stream_callback=on_token),
                        timeout=deadline,
                    )
                )
        except asyncio.TimeoutError:
            console.print(f"\n[red]等待模型响应超过 {deadline:.0f}s，已中止[/red]")
            return
        except KeyboardInterrupt:
            console.print("\n[yellow]已中断[/yellow]")
            return
        except RuntimeError as exc:
            console.print(f"\n[red]{exc}[/red]")
            return
        finally:
            self._stop_live()

        if on_token is None:
            # 非流式：渲染后的 Markdown，打印一次
            console.print(Markdown(result.answer.text))
        else:
            # 流式：token 已 append-only 打印过，这里只补一个换行，**绝不重打**
            console.print()

        cited = ", ".join(result.answer.citation_ids) or "（无）"
        if result.problems:
            console.print(f"\n[dim]引用：{cited}[/dim]  [yellow]校验：{'; '.join(result.problems)}[/yellow]")
        else:
            console.print(f"\n[dim]引用：{cited}[/dim]  [green]引用校验通过[/green]")

        self.history.append(
            {
                "question": question,
                "answer": result.answer.text,
                "citations": result.answer.citation_ids,
                "problems": result.problems,
            }
        )

    def cmd_report(self, args: str) -> None:
        topic, flags = _split_args(args)
        if not topic:
            console.print("[yellow]用法：/report <topic> [--papers N] [--simple][/yellow]")
            return
        session = self.require_session()
        if session is None:
            return
        papers = int(flags.get("papers", self.settings.max_papers))
        simple = _flag_bool(flags, "simple")
        if flags.get("source"):
            self.settings = self.settings.model_copy(update={"search_source": flags["source"]})
        console.print(f"[cyan]开始生成报告：{topic}[/cyan] [dim]（{'simple 单 agent' if simple else '多 agent 监督图'}，"
                      f"最多 {papers} 篇；过程日志见终端）[/dim]")
        try:
            with console.status("[cyan]检索 → 入库 → 精读 → 归纳 → 写作…（可能需要几分钟）[/cyan]"):
                result = self._run_async(
                    run_report(
                        topic,
                        settings=self.settings,
                        session=session,
                        papers=papers,
                        simple=simple,
                        search_tools=session.search_tools if session.tools_loaded else None,
                    )
                )
        except KeyboardInterrupt:
            console.print("\n[yellow]已中断[/yellow]")
            return
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            return

        console.print(papers_table(result.papers, "本次分析的论文"))
        for name, path in result.paths.items():
            console.print(f"[green]✓[/green] {name}: {path}")
        if result.flagged:
            console.print("[yellow]校验提示：[/yellow]" + "；".join(result.flagged[:5]))
        summaries = result.state.get("summaries", []) or []
        console.print(
            f"[dim]摘要 {len(summaries)} 篇 / 索引 {len(result.indexed)} 篇论文[/dim]"
        )
        console.print("[dim]用 /save 保存会话，或直接用编辑器查看生成的 md[/dim]")

    def cmd_mcp(self) -> None:
        from src.paper_agent.mcp_client import describe_mcp_tools, load_server_specs

        specs = load_server_specs(self.settings)
        if not specs:
            console.print("[yellow]没有可用的 MCP server（pip install arxiv-mcp-server paper-search-mcp）[/yellow]")
            return
        with console.status("[cyan]读取 MCP 工具列表…[/cyan]"):
            data = self._run_async(describe_mcp_tools(self.settings))
        table = Table(title="MCP servers")
        table.add_column("server")
        table.add_column("原始", justify="right")
        table.add_column("白名单", justify="right")
        for name, info in data["servers"].items():
            if "error" in info:
                table.add_row(name, "-", "-", )
            else:
                table.add_row(name, str(info["raw_tools"]), str(info["kept_tools"]))
        console.print(table)
        console.print(f"[dim]白名单工具（{len(data['kept'])}）：" + ", ".join(sorted(data["kept"])) + "[/dim]")

    def cmd_model(self, args: str) -> None:
        """`/model` 查看；`/model <name>` 本次会话使用；`/model <name> --default` 写入默认。"""
        name, flags = _split_args(args)
        make_default = _flag_bool(flags, "default")
        session = self.require_session()
        if session is None:
            return

        if not name:
            model = session.model
            current = getattr(model, "model_name", None) or getattr(model, "model", None)
            provider = self.config.active_provider()
            console.print(
                f"当前模型：[cyan]{current or '（离线假模型）'}[/cyan]  "
                f"[dim]供应商={self.settings.provider_label} 默认={self.config.default_model or '-'}"
                f" 候选={len(provider.chat_models) if provider is not None else 0}[/dim]"
            )
            console.print("[dim]用法：/model <name> [--default]；或 /models 打开选择器[/dim]")
            return

        if session.offline:
            console.print("[yellow]当前离线模式（假模型），/offline off 后再切模型[/yellow]")
            return

        base = self.settings.model_copy(update={"llm_model": name})
        self._rebuild(base, f"模型已切换为 {name}")

        provider = self.config.active_provider()
        if provider is None:
            return
        provider.chat_model = name
        if make_default:
            self.config.set_default(provider.name, name)
            console.print(f"[green]★[/green] 已写入默认模型（{self.config.path}）")
        self.config.save()

    # ---------------- embedding 单独设置（/embed） ----------------
    def _show_embedding(self) -> None:
        chat = self.config.active_provider()
        embedder, model = resolve_embedding(self.config, chat)
        source = (
            f"显式指定（{self.config.embedding_provider}）"
            if self.config.embedding_provider
            else "自动挑选（对话供应商 → 默认 → 第一个带 embedding 的）"
        )
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("对话供应商", chat.name if chat else "-")
        table.add_row(
            "embedding 供应商",
            f"{embedder.name} ({embedder.kind})" if embedder else "（无）",
        )
        table.add_row("embedding 模型", model or "（无 → RAG 不可用）")
        table.add_row(
            "向量维度",
            str(embedder.embedding_dim) if embedder and embedder.embedding_dim else "（未探测）",
        )
        table.add_row("来源", source)
        table.add_row("索引目录", str(self.settings.data_dir))
        console.print(Panel(table, title="embedding 设置（RAG 建索引用）", border_style="cyan", padding=(0, 1)))

    def _pick_embedding_model(self, provider) -> str:
        items = embed_models_for(provider)
        if not items:
            return clean_pasted(
                console.input(f"{provider.name} 的 embedding 模型名（留空取消）› ")
            ).strip()
        action, value = pick_from_list(
            console,
            items,
            current=provider.embedding_model,
            default=provider.embedding_model,
            title=f"{provider.name} 的 embedding 模型",
            footer="Enter 确定",
            display_class="pick.embedding",
        )
        return "" if action == "cancel" else value

    def cmd_embed(self, args: str = "") -> None:
        """`/embed` 查看；`/embed [供应商] [模型]` 指定；`/embed auto` 改回自动。"""
        positional, flags = _split_args(args)
        tokens = positional.split()
        action = tokens[0].lower() if tokens else ""

        if not self.config.providers:
            console.print(
                "[yellow]还没有通过 /connect 配置供应商。[/yellow]"
                "[dim]用 .env 时直接设置 EMBED_MODEL / EMBED_BASE_URL / EMBED_API_KEY 即可分开指定 embedding。[/dim]"
            )
            return
        if not action and not flags:
            self._show_embedding()
            console.print(
                "[dim]用法：/embed <供应商> [模型] 指定；/embed --model <模型> 只换模型；"
                "/embed auto 改回自动；/embed set 交互选择[/dim]"
            )
            return
        if action in {"auto", "off", "reset"}:
            self.config.set_embedding_provider("")
            self.config.save()
            self._refresh_settings("embedding 来源已改回自动挑选")
            self._show_embedding()
            return
        if action in {"set", "pick"}:
            chosen = self.pick_provider("选择负责 embedding 的供应商")
            if not chosen:
                return
            action, tokens = chosen, [chosen]

        name = action if action in self.config.providers else flags.get("provider", "")
        if name and name not in self.config.providers:
            console.print(f"[red]没有供应商 {name}[/red]（用 /providers 查看）")
            return
        provider = self.config.providers.get(name) if name else None
        if provider is None:  # 未指定：作用于当前生效的 embedding 供应商
            provider, _ = resolve_embedding(self.config, self.config.active_provider())
        if provider is None:
            console.print("[yellow]没有可用的 embedding 供应商[/yellow]")
            return

        model = tokens[1] if len(tokens) > 1 else flags.get("model", "")
        if not model:
            model = self._pick_embedding_model(provider)
        if not model:
            console.print("[dim]已取消[/dim]")
            return

        self.config.set_embedding_provider(provider.name)
        provider.embedding_model = model
        try:
            with console.status(f"[cyan]探测 {model} 的向量维度…[/cyan]"):
                dim = self._run_async(
                    probe_embedding_dim(provider.base_url, provider.api_key, model)
                )
            if dim:
                provider.embedding_dim = dim
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]维度探测失败（{type(exc).__name__}）：{str(exc)[:100]}[/yellow]")
        self.config.save()
        self._rebuild(get_settings(refresh=True), f"embedding 已指定为 {provider.name} / {model}")

    def cmd_offline(self, args: str) -> None:
        current = self.settings.fake_llm
        token = args.strip().lower()
        if not token or token == "toggle":
            target = not current
        else:
            target = token in {"on", "true", "1", "yes"}
        if target == current:
            console.print(f"[dim]离线模式已是 {'开' if current else '关'}[/dim]")
            return
        self.settings = cli_settings(offline=target)
        self._rebuild(self.settings, "离线模式已切换")
        console.print(
            f"[green]✓[/green] 离线模式：{'开（假模型 + data/offline 索引）' if target else '关'}"
        )

    # ---------------- 供应商配置 ----------------
    def _rebuild(self, settings, note: str = "") -> bool:
        """重建会话（换供应商/模型/embedding 后调用）；失败时保留原会话。"""
        try:
            session = build_session(settings)
        except RuntimeError as exc:
            console.print(f"[red]会话刷新失败：{exc}[/red]")
            console.print("[dim]配置已保存；修好后可用 /providers use <name> 或 /connect 重试[/dim]")
            return False
        self.settings = settings
        self.session = session
        model = getattr(self.session.model, "model_name", None) or getattr(self.session.model, "model", None)
        console.print(
            f"[green]✓[/green] {note or '会话已刷新'}  "
            f"供应商=[cyan]{settings.provider_label}[/cyan] 模型=[cyan]{model or '-'}[/cyan] "
            f"[dim]({settings.active_base_url})[/dim]"
        )
        console.print(
            f"[dim]索引 {len(self.session.index.paper_ids())} 篇 / {self.session.index.chunk_count} chunks"
            f"（embedding={settings.active_embedding_model or '无'}"
            f"{' @ ' + settings.active_embedding_label if settings.embed_base_url else ''}）[/dim]"
        )
        return True

    def cmd_connect(self, args: str) -> None:
        positional, flags = _split_args(args)
        tokens = positional.split()
        base_url = ""
        preset = ""
        api_key = flags.get("key", "") or flags.get("api-key", "")

        if tokens:
            if tokens[0].isdigit():          # `/connect 2 --key sk-...`
                preset = tokens[0]
            else:                             # `/connect https://api.deepseek.com sk-...`
                base_url = tokens[0]
            if len(tokens) > 1:
                api_key = api_key or tokens[1]

        provider = connect_flow(
            console,
            self.config,
            base_url=base_url,
            api_key=api_key,
            name=flags.get("name", ""),
            kind=flags.get("kind", ""),
            preset=preset,
            do_fetch=not _flag_bool(flags, "no-fetch"),
            allow_empty_key=_flag_bool(flags, "allow-empty-key"),
        )
        if provider is None:
            return
        self.config = UserConfig.load()  # 重新读一行，确保与磁盘一致
        self._rebuild(get_settings(refresh=True), f"已连接供应商 {provider.name}")

    def _provider_display(self) -> dict[str, str]:
        """供应商选择器里的展示文案：类型 + 脱敏 key + 模型数。"""
        out: dict[str, str] = {}
        for name, prov in self.config.providers.items():
            star = "★ " if name == self.config.default_provider else ""
            embed = prov.embedding_model or "无 embedding"
            out[name] = (
                f"{star}{name}  [dim]({prov.kind} · {mask_secret(prov.api_key)} · "
                f"{len(prov.chat_models)} 个对话模型 · embed={embed})[/dim]"
            )
        return out

    def pick_provider(self, title: str = "选择供应商") -> str:
        """弹出供应商选择器；返回选中的供应商名（取消返回空串）。"""
        names = list(self.config.providers)
        if not names:
            console.print("[yellow]还没有配置供应商，先 /connect[/yellow]")
            return ""
        if len(names) == 1:
            return names[0]
        action, name = pick_from_list(
            console,
            names,
            current=self.config.default_provider,
            default=self.config.default_provider,
            title=title,
            display=self._provider_display(),
            footer="Enter 切换",
            display_class="pick.provider",
        )
        if action == "cancel" or not name:
            console.print("[dim]已取消[/dim]")
            return ""
        return name

    def _switch_provider(self, name: str) -> None:
        """把默认供应商切到 name 并重建会话。"""
        provider = self.config.providers.get(name)
        if provider is None:
            console.print(f"[red]没有供应商 {name}[/red]")
            return
        self.config.set_default(name, provider.chat_model)
        self.config.save()
        base = get_settings(refresh=True).model_copy(update=settings_overrides(self.config, provider))
        self._rebuild(base, f"供应商已切到 {name}")

    def cmd_models(self, args: str) -> None:
        _, flags = _split_args(args)
        provider_arg = flags.get("provider")

        # `/models --provider`（不带值）→ 先弹供应商选择器
        if "provider" in flags and provider_arg in {"", "true"}:
            chosen = self.pick_provider("选择要浏览模型的供应商")
            if not chosen:
                return
            self._switch_provider(chosen)
            provider_arg = chosen

        provider = (
            self.config.providers.get(provider_arg) if provider_arg and provider_arg != "true" else None
        ) or self.config.active_provider()
        if provider is None:
            console.print("[yellow]还没有配置供应商，先执行 [cyan]/connect[/cyan][/yellow]")
            return

        want_embedding = _flag_bool(flags, "embedding")
        if _flag_bool(flags, "refresh") or (not provider.models and not provider.chat_models):
            from src.paper_agent.tui import _sync_models

            _sync_models(console, self.config, provider, with_embedding_probe=want_embedding)
            self.config.save()

        while True:  # Ctrl+P 可在列表里直接换供应商，换完继续选模型
            if want_embedding:
                items = embed_models_for(provider)
                title = f"{provider.name} 的 embedding 模型（RAG 建索引用）"
                if not items:
                    console.print(
                        "[yellow]该供应商的 /models 里没有 embedding 模型。[/yellow]"
                        "可以让另一个供应商（如 DashScope）负责 embedding，或手动输入模型名。"
                    )
                    manual = clean_pasted(console.input("embedding 模型名（留空取消）› "))
                    if not manual:
                        return
                    items = [manual]
                current, default = provider.embedding_model, provider.embedding_model
            else:
                items = provider.models or provider.chat_models
                if _flag_bool(flags, "all"):
                    items = provider.models
                elif provider.chat_models:
                    items = provider.chat_models
                if not items:
                    console.print(
                        "[yellow]没有拿到模型列表：可用 /models --refresh 重试，或 /model <名字> 直接指定[/yellow]"
                    )
                    return
                current, default = provider.chat_model, self.config.default_model
                title = f"{provider.name} 的对话模型"

            action, value = pick_from_list(
                console,
                items,
                current=current,
                default=default,
                title=title,
                cursor_start=current,
                footer=f"Ctrl+P 切换供应商（当前 {provider.name}）",
                display_class="pick.embedding" if want_embedding else "pick.model",
            )
            if action == "provider":
                chosen = self.pick_provider("选择要浏览模型的供应商")
                if not chosen:
                    continue
                self._switch_provider(chosen)
                provider = self.config.providers[chosen]
                continue
            break

        if action == "cancel" or not value:
            console.print("[dim]已取消[/dim]")
            return

        if want_embedding:
            provider.embedding_model = value
            try:
                with console.status(f"[cyan]探测 {value} 的向量维度…[/cyan]"):
                    dim = self._run_async(probe_embedding_dim(provider.base_url, provider.api_key, value))
                provider.embedding_dim = dim or provider.embedding_dim
                console.print(f"[green]✓[/green] embedding 已切换为 {value}" + (f"（dim={dim}）" if dim else ""))
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]维度探测失败（{type(exc).__name__}）：{str(exc)[:100]}[/yellow]")
        else:
            provider.chat_model = value
            console.print(f"[green]✓[/green] 对话模型：{value}")

        if action == "default":
            self.config.set_default(provider.name, value if not want_embedding else provider.chat_model)
            console.print(f"[green]★[/green] 已写入默认（{self.config.path}）")
        self.config.save()

        overrides = settings_overrides(self.config, provider)
        if action == "select" and not want_embedding:
            # 仅本次会话使用：不覆盖默认模型
            overrides["llm_model"] = value
        base = get_settings(refresh=True).model_copy(update=overrides)
        self._rebuild(base, "模型已切换")

    def cmd_providers(self, args: str) -> None:
        action, _, name = args.strip().partition(" ")
        name = clean_pasted(name)
        if not action:
            print_presets(console, self.config)
            if self.config.providers:
                # 直接给选择器，省得记名字
                chosen = self.pick_provider("切换默认供应商（Enter 确认 / Esc 只看不改）")
                if chosen:
                    self._switch_provider(chosen)
            return
        if action == "use" and not name:
            chosen = self.pick_provider("切换默认供应商")
            if chosen:
                self._switch_provider(chosen)
            return
        if action == "use":
            if name not in self.config.providers:
                hint = ""
                for key, kind, label in __import__(
                    "src.paper_agent.tui", fromlist=["PRESETS"]
                ).PRESETS:
                    if kind == name:
                        hint = f"（可用 [cyan]/connect {key}[/cyan] 添加 {label}）"
                        break
                console.print(f"[red]没有供应商 {name}[/red]{hint}")
                return
            self.config.set_default(name, self.config.providers[name].chat_model)
            self.config.save()
            base = get_settings(refresh=True).model_copy(
                update=settings_overrides(self.config, self.config.providers[name])
            )
            self._rebuild(base, f"默认供应商已切到 {name}")
        elif action in {"rm", "remove", "del", "delete"}:
            target = name
            if not target:
                # `/providers rm`（不带名字）→ 直接弹选择器，省得记名字
                names = list(self.config.providers)
                if not names:
                    console.print("[yellow]还没有配置供应商[/yellow]")
                    return
                if len(names) == 1:
                    target = names[0]
                else:
                    _act, picked = pick_from_list(
                        console,
                        names,
                        current=self.config.default_provider,
                        default=self.config.default_provider,
                        title="选择要删除的供应商",
                        display=self._provider_display(),
                        footer="Enter 选中共确认删除 / Esc 取消",
                        display_class="pick.provider",
                    )
                    if _act == "cancel" or not picked:
                        console.print("[dim]已取消[/dim]")
                        return
                    target = picked
            if target not in self.config.providers:
                console.print(f"[red]没有供应商 {target}[/red]")
                return
            mark = "（当前默认供应商）" if target == self.config.default_provider else ""
            answer = clean_pasted(
                console.input(f"确认删除供应商 [cyan]{target}[/cyan]{mark}（含其 API key）？[y/N] › ")
            ).strip().lower()
            if answer not in {"y", "yes"}:
                console.print("[dim]已取消[/dim]")
                return
            self.config.remove_provider(target)
            self.config.save()
            console.print(f"[green]✓[/green] 已删除供应商 [cyan]{target}[/cyan]")
            self._rebuild(get_settings(refresh=True), f"已移除 {target}")
        elif action == "sync":
            provider = self.config.providers.get(name) or self.config.active_provider()
            if provider is None:
                console.print("[red]没有可同步的供应商[/red]")
                return
            from src.paper_agent.tui import _sync_models

            _sync_models(console, self.config, provider)
            self.config.save()
        elif action in {"key-rm", "rm-key", "clear-key", "key-off"}:
            # 只删 key，保留供应商与已选模型（原 key 不再可用）
            if self.config.remove_provider_key(name):
                self.config.save()
                console.print(f"[green]✓[/green] 已删除供应商 [cyan]{name}[/cyan] 的 API key（供应商保留）")
                self._rebuild(get_settings(refresh=True), f"已删除 {name} 的 key")
            else:
                console.print(f"[red]没有供应商 {name}[/red]")
        else:
            console.print(
                "[yellow]用法：/providers | /providers use [name] | /providers rm [name]（不给名字弹选择器） | "
                "/providers key-rm <name>（只删 key） | /providers sync [name][/yellow]"
            )

    # ---------------- 搜索渠道（/channels） ----------------
    def _print_help(self, topic: str = "") -> None:
        """`/help`：分组展示命令与用法；`/help search` 只看某条命令的细节。"""
        key = (topic or "").strip().lstrip("/").split(" ")[0].lower()
        if key:
            hits = [c for c in COMMANDS if c.lstrip("/").lower() == key]
            if not hits:
                hits = [c for c in COMMANDS if c.lstrip("/").lower().startswith(key)]
            if not hits:
                console.print(f"[yellow]没有命令 /{key}[/yellow]（输入 /help 查看全部）")
                return
            for cmd in hits:
                # 用法里含 [--flag] 这类方括号，必须用 Text 输出，否则会被 rich 当成标记吞掉
                line = Text()
                line.append(cmd, style="bold cyan")
                line.append("  " + COMMANDS[cmd], style="dim")
                console.print(line)
                console.print(Text("  " + COMMAND_USAGE.get(cmd, cmd), style="cyan"))
                console.print()
            return

        console.print(
            Panel(
                "[bold]paper-agent 命令帮助[/bold]\n"
                "[dim]直接输入自然语言 = [/dim][cyan]/ask[/cyan][dim] · Tab/Enter 补全 · Ctrl+C 退出[/dim]",
                border_style="cyan",
                padding=(0, 2),
            )
        )
        for title, cmds in HELP_GROUPS:
            table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
            table.add_column("cmd", no_wrap=True, vertical="top")
            table.add_column("desc", overflow="fold")
            for cmd in cmds:
                if cmd not in COMMANDS:
                    continue
                cell = Text(COMMANDS[cmd])
                usage = COMMAND_USAGE.get(cmd)
                if usage and usage != cmd:  # 用法与命令名相同的（如 /index）不重复一行
                    cell.append("\n" + usage, style="dim")
                table.add_row(Text(cmd, style="bold cyan"), cell)
            console.print(f"[bold]{title}[/bold]")
            console.print(table)
            console.print()

        table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
        table.add_column("ex", no_wrap=True)
        table.add_column("note", overflow="fold")
        for cmd, note in HELP_EXAMPLES:
            table.add_row(Text(cmd, style="cyan"), Text(note, style="dim"))
        console.print("[bold]常用示例[/bold]")
        console.print(table)
        console.print("[dim]看某条命令细节：[/dim] [cyan]/help search[/cyan]")

    def cmd_channels(self, args: str = "") -> None:
        """查看 / 添加 / 删陈搜索渠道；密钥与模型供应商一样由用户自配。"""
        action, _, rest = args.strip().partition(" ")
        action = action.lower()
        name = clean_pasted(rest).strip()
        if not action or action == "list":
            self.show_channels()
        elif action in {"add", "new"}:
            self.add_channel(rest)
        elif action in {"rm", "remove", "del", "delete"}:
            if not name:
                console.print("[yellow]用法：/channels rm <name>[/yellow]")
                return
            if self.config.remove_channel(name):
                self.config.save()
                self._refresh_settings("已删除搜索渠道")
            else:
                console.print(f"[red]没有渠道 {name}[/red]")
        elif action in {"key-rm", "rm-key", "clear-key"}:
            if self.config.remove_channel_key(name):
                self.config.save()
                console.print(f"[green]✓[/green] 已删除渠道 [cyan]{name}[/cyan] 的 API key（渠道保留）")
                self._refresh_settings("已删除渠道 key")
            else:
                console.print(f"[red]没有渠道 {name}[/red]")
        elif action in {"domestic", "cn", "国内"} or (
            action in {"on", "off"} and name.lower() in {"domestic", "cn", "国内"}
        ):
            # 支持 `/channels domestic on|off` 与 `/channels on|off domestic`
            word = name.lower() if action in {"domestic", "cn", "国内"} else action
            if word not in {"on", "off", "1", "0", "true", "false", "yes", "no"}:
                console.print("[yellow]用法：/channels domestic on|off[/yellow]")
                return
            enabled = word in {"on", "1", "true", "yes"}
            self.config.set_prefer_domestic(enabled)
            self.config.save()
            self._refresh_settings(f"国内渠道优先已{'开启' if enabled else '关闭'}")
            console.print(
                "[green]✓[/green] 国内渠道优先已开启（nlc / chinaxiv / 百度学术 / 万方 排在前面）"
                if enabled
                else "[dim]已关闭国内优先：源与结果按原顺序排列[/dim]"
            )
            return
        elif action in {"all", "全渠道"} or (action in {"on", "off"} and name.lower() in {"all", "*"}):
            # 支持两种写法：`/channels all on|off` 与 `/channels on|off all`
            word = name.lower() if action in {"all", "全渠道"} else action
            if word not in {"on", "off", "1", "0", "true", "false", "yes", "no"}:
                console.print("[yellow]用法：/channels all on|off[/yellow]")
                return
            enabled = word in {"on", "1", "true", "yes"}
            self.config.set_search_all_channels(enabled)
            self.config.save()
            self._refresh_settings(f"全渠道检索已{'开启' if enabled else '关闭'}")
            if enabled:
                console.print(
                    "[green]✓[/green] 检索时将并发跑全部已注册渠道（免 key 的 + 已配置 key 的；"
                    "缺 key 的自动跳过）"
                )
            else:
                console.print("[dim]已恢复：只跑 BUILTIN_SOURCES 里的默认源[/dim]")
            return
        elif action in {"on", "off"}:
            if not name or name not in self.config.channels:
                console.print(f"[red]没有渠道 {name}[/red]")
                return
            self.config.set_channel_enabled(name, action == "on")
            self.config.save()
            self._refresh_settings(f"渠道 {name} 已{'启用' if action == 'on' else '停用'}")
        else:
            console.print(
                "[yellow]用法：/channels | /channels add [kind|编号] [--key K] [--email E] | "
                "/channels rm <name> | /channels key-rm <name> | /channels on|off <name> | "
                "/channels all on|off | /channels domestic on|off[/yellow]"
            )

    def show_channels(self) -> None:
        table = Table(title="可添加的搜索渠道（/channels add <编号|kind>）", header_style="bold")
        table.add_column("#", justify="right", style="dim")
        table.add_column("kind", style="cyan")
        table.add_column("名称")
        table.add_column("凭据", style="dim")
        table.add_column("说明", style="dim")
        for num, spec in channel_presets():
            if spec.needs_key:
                creds = "key"
            elif spec.builtin:
                creds = "免 key"
            else:
                creds = "email(可选)"
            table.add_row(num, spec.kind, spec.label, creds, spec.description)
        console.print(table)
        mode = "[green]已开启[/green]" if self.config.search_all_channels else "[dim]关闭[/dim]"
        dom = "[green]已开启[/green]" if self.config.prefer_domestic else "[dim]关闭[/dim]"
        console.print(
            "[dim]所有渠道默认禁用：用 `/channels add <编号|kind>` 逐个添加后才参与检索"
            "（免 key 的可选填 email 进 polite pool；需 key 的会提示配置）。[/dim]"
        )
        console.print(
            f"国内渠道优先：{dom}（用 `/channels domestic on|off` 切换）——开启后国内库排在前面、结果优先保留。"
        )
        console.print(
            f"全渠道并发检索：{mode}（用 `/channels all on|off` 切换）——开启后每次检索会"
            "并发跑全部已注册渠道，缺 key 的自动跳过。"
        )

        if self.config.channels:
            existing = Table(title="已配置渠道", header_style="bold")
            existing.add_column("name", style="cyan")
            existing.add_column("kind")
            existing.add_column("状态")
            existing.add_column("key")
            existing.add_column("email", style="dim")
            for name, ch in self.config.channels.items():
                existing.add_row(
                    name,
                    ch.kind,
                    "[green]启用[/green]" if ch.enabled else "[yellow]停用[/yellow]",
                    ch.masked_key(),
                    ch.email or "-",
                )
            console.print(existing)
            console.print("[dim]搜索引擎实际使用的渠道：/channels on|off <name>；删 key：/channels key-rm <name>[/dim]")
        else:
            console.print(
                "[yellow]当前没有启用任何渠道[/yellow]：用 `/channels add <编号|kind>` 添加"
                "（如 `/channels add arxiv`，或 `/channels add 1`）"
            )

    def add_channel(self, args: str) -> None:
        positional, flags = _split_args(args)
        token = positional.split()[0] if positional.split() else ""
        preset_map = {num: spec for num, spec in channel_presets()}
        spec = preset_map.get(token) if token else None
        if spec is None and token:
            spec = spec_for(token)
        if spec is None:
            self.show_channels()
            answer = clean_pasted(console.input("[bold]渠道编号或 kind[/bold]（回车取消）› ")).strip()
            if not answer:
                console.print("[dim]已取消[/dim]")
                return
            spec = preset_map.get(answer) or spec_for(answer)
        if spec is None:
            console.print("[red]未知渠道类型[/red]（用 /channels 查看可选项）")
            return

        api_key = flags.get("key", "")
        email = flags.get("email", "")
        name = flags.get("name", "") or spec.kind
        if spec.needs_key and not api_key:
            api_key = read_secret(console, f"{spec.label} API key  › ")
            if not api_key:
                console.print(f"[yellow]{spec.label} 需要 API key，已取消[/yellow]")
                return
        if not email and (spec.needs_email or spec.group == "academic"):
            hint = "（可选，进 polite pool；留空跳过）"
            entered = clean_pasted(console.input(f"联系邮箱{hint} › ")).strip()
            email = entered

        if spec.builtin:
            console.print(f"[dim]{spec.label} 免 key；可选补充 email 进 polite pool。[/dim]")

        channel = self.config.upsert_channel(
            spec.kind, name=name, api_key=api_key, email=email, base_url=flags.get("base-url", "")
        )
        self.config.save()
        console.print(
            f"[green]✓[/green] 已添加渠道 [cyan]{channel.name}[/cyan]（{channel.label}）"
            f"  key={channel.masked_key()} → {self.config.path}"
        )
        self._refresh_settings("搜索渠道已更新")
        console.print("[dim]提示：检索时自动生效（/search ... --source builtin）；删 key 用 /channels key-rm <name>[/dim]")

    def _refresh_settings(self, note: str) -> None:
        """配置变更后刷新 Settings（无需重建会话：搜索渠道不影响索引/模型）。"""
        self.config = UserConfig.load()
        self.settings = get_settings(refresh=True)
        if self.session is not None:
            self.session.settings = self.settings
        if note:
            console.print(f"[dim]{note}（{self.config.path}）[/dim]")

    # ---------------- API key 总览 / 删除（/keys） ----------------
    def cmd_keys(self, args: str = "") -> None:
        """查看并删除模型供应商与搜索渠道的 API key。"""
        action, _, rest = args.strip().partition(" ")
        action = action.lower()
        rest = rest.strip()
        if action in {"rm", "remove", "del", "delete"}:
            self._remove_key(rest)
            return
        if action:
            console.print(
                "[yellow]用法：/keys | /keys rm provider:<name> | /keys rm channel:<name> | /keys rm <name>[/yellow]"
            )
            return

        table = Table(title="已保存的 API key", header_style="bold")
        table.add_column("类型")
        table.add_column("name", style="cyan")
        table.add_column("key")
        table.add_column("说明", style="dim")
        for name, prov in self.config.providers.items():
            table.add_row("provider", name, mask_secret(prov.api_key) if prov.api_key else "（无）", prov.base_url)
        for name, ch in self.config.channels.items():
            table.add_row("channel", name, ch.masked_key(), ch.kind)
        console.print(table)
        console.print("[dim]删除：/keys rm provider:<name> 或 /keys rm channel:<name>（也支持 /providers key-rm、/channels key-rm）[/dim]")

    def _remove_key(self, target: str) -> None:
        target = clean_pasted(target).strip()
        if not target:
            console.print("[yellow]请给出要删除的 name：/keys rm provider:<name> | channel:<name>[/yellow]")
            return
        kind, _, name = target.partition(":")
        if not name:
            name, kind = kind, ""
        if kind == "provider" or (not kind and name in self.config.providers):
            if self.config.remove_provider_key(name):
                self.config.save()
                console.print(f"[green]✓[/green] 已删除供应商 [cyan]{name}[/cyan] 的 API key（供应商保留）")
                self._rebuild(get_settings(refresh=True), f"已删除 {name} 的 key")
                return
        if kind == "channel" or (not kind and name in self.config.channels):
            if self.config.remove_channel_key(name):
                self.config.save()
                console.print(f"[green]✓[/green] 已删除渠道 [cyan]{name}[/cyan] 的 API key（渠道保留）")
                self._refresh_settings("已删除渠道 key")
                return
        console.print(f"[red]未找到 {target}（用 /keys 查看）[/red]")

    # ---------------- 流式渲染 ----------------
    def _stream_renderer(self) -> Callable[[str], None]:
        """边生成边直接追加打印（append-only，不做整段重绘）。

        Rich `Live` 重绘整段 Markdown 时，内容超过一屏后随终端滚动会重复打印内容；
        append-only 则只会向下滚。代价：流式期间不做 Markdown 渲染。
        """

        def on_token(token: str) -> None:
            if token:
                console.print(token, end="", markup=False, highlight=False, soft_wrap=True)

        return on_token

    def _stop_live(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            finally:
                self._live = None

    def _run_async(self, coro: Any) -> Any:
        """在持久事件循环上运行协程（替代每次 `asyncio.run`）。

        复用的 ChatModel 内部 httpx `AsyncClient` 绑定在首次的事件循环上；
        若每条命令都 `asyncio.run`（新建并关闭循环），第二次请求会报
        `Event loop is closed`。
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            self._loop = loop
        return loop.run_until_complete(coro)

    def _close_loop(self) -> None:
        loop = self._loop
        self._loop = None
        if loop is not None and not loop.is_closed():
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass

    # ---------------- 主循环 ----------------
    def dispatch(self, line: str) -> bool:
        """执行一行输入；返回 False 表示退出。"""
        line = line.strip()
        if not line:
            return True

        if not line.startswith("/"):
            self.cmd_ask(line)
            return True

        command, _, rest = line.partition(" ")
        rest = rest.strip()
        if command in {"/exit", "/quit", "/q"}:
            return False
        if command == "/help":
            self._print_help(rest)
        elif command == "/search":
            self.cmd_search(rest)
        elif command == "/ingest":
            self.cmd_ingest(rest)
        elif command == "/ask":
            self.cmd_ask(rest)
        elif command == "/report":
            self.cmd_report(rest)
        elif command == "/papers":
            self.cmd_papers(rest)
        elif command == "/index":
            self.show_index()
        elif command == "/mcp":
            self.cmd_mcp()
        elif command == "/channels":
            self.cmd_channels(rest)
        elif command == "/keys":
            self.cmd_keys(rest)
        elif command == "/connect":
            self.cmd_connect(rest)
        elif command == "/models":
            self.cmd_models(rest)
        elif command == "/providers":
            self.cmd_providers(rest)
        elif command == "/model":
            self.cmd_model(rest)
        elif command == "/embed":
            self.cmd_embed(rest)
        elif command == "/offline":
            self.cmd_offline(rest)
        elif command == "/stream":
            token = rest.lower()
            self.stream = not self.stream if token in {"", "toggle"} else token in {"on", "true", "1", "yes"}
            console.print(f"[green]✓[/green] 流式输出：{'开' if self.stream else '关'}")
        elif command == "/history":
            self.show_history(int(rest) if rest.isdigit() else 5)
        elif command == "/save":
            self.save_history(rest)
        elif command == "/clear":
            console.clear()
            self.banner()
        else:
            console.print(f"[yellow]未知命令 {command}，输入 /help 查看可用命令[/yellow]")
        return True

    def safe_dispatch(self, line: str) -> bool:
        """执行一行输入并吞掉异常（REPL 不应因单条命令崩溃）；返回 False 表示退出。"""
        try:
            return self.dispatch(line)
        except KeyboardInterrupt:
            self._stop_live()
            console.print("\n[yellow]已中断当前操作[/yellow]")
            return True
        except Exception as exc:  # noqa: BLE001
            self._stop_live()
            logging.getLogger(__name__).exception("命令执行失败")
            console.print(f"[red]执行失败：{type(exc).__name__}: {exc}[/red]")
            return True

    # ---------------- 命令面板（输入 / 即出现可滚动补全列表） ----------------
    def status_line(self) -> str:
        """底部状态栏内容（prompt_toolkit bottom_toolbar）。"""
        s = self.settings
        if self.session is None:
            return " 未配置供应商 · /connect 添加（支持粘贴）· /help 查看命令 "
        model = getattr(self.session.model, "model_name", None) or getattr(self.session.model, "model", None)
        return (
            f" {s.provider_label} · {model or '（离线假模型）'} · "
            f"索引 {len(self.session.index.paper_ids())} 篇/{self.session.index.chunk_count} chunks · "
            f"流式{'开' if self.stream else '关'} · /help 命令 · Tab/Enter 补全 "
        )

    def palette_context(self) -> PaletteContext:
        def providers() -> list[str]:
            return sorted(self.config.providers)

        def models() -> list[str]:
            provider = self.config.active_provider()
            if provider is None:
                return []
            return list(provider.chat_models or provider.models)[:200]

        def indexed() -> list[str]:
            if self.session is None:
                return []
            return [item["paper_id"] for item in self.session.index.list_papers()]

        def channels() -> list[str]:
            return sorted(self.config.channels)

        return PaletteContext(
            commands=COMMANDS,
            providers=providers,
            models=models,
            indexed=indexed,
            channels=channels,
            status=self.status_line,
        )

    def repl(self) -> None:
        self.banner()
        session = create_session(self.palette_context())
        if session is None:
            _setup_readline()  # 没装 prompt_toolkit → 退回 readline + 普通输入
        else:
            console.print(
                "[dim]提示：输入 [bold]/[/bold] 会弹出命令面板（Tab 补全 / Enter 确认），"
                "↑↓·PgUp/PgDn 滚动，底部状态栏常显供应商与索引规模。[/dim]"
            )
        while True:
            try:
                line = read_line(session) if session is not None else console.input(PROMPT)
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]再见 👋[/dim]")
                break
            except Exception as exc:  # noqa: BLE001 - prompt_toolkit 环境异常时兜底
                logging.getLogger(__name__).warning("输入层异常，改用普通输入：%s", exc)
                session = None
                continue
            if not self.safe_dispatch(line):
                console.print("[dim]再见 👋[/dim]")
                break
        _save_readline_history()
        self._close_loop()


class _nullcontext:
    """stdin 非 TTY / 流式输出时替代 console.status。"""

    def __enter__(self):
        return None

    def __exit__(self, *exc) -> Literal[False]:
        return False


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="学术论文检索与概括分析 Agent（交互式 REPL / 一次性命令）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("question", nargs="*", help="一次性提问（等价于 /ask）")
    parser.add_argument("--search", metavar="QUERY", help="一次性检索")
    parser.add_argument("--ingest", metavar="QUERY", help="一次性入库（关键词）")
    parser.add_argument("--ids", metavar="IDS", help="一次性入库：按 ID 直抓（逗号分隔，可省略关键词）")
    parser.add_argument("--search-ingest", action="store_true", help="--search 时把结果直接入库")
    parser.add_argument("--no-llm", action="store_true", help="检索时不调用 LLM（关闭查询扩展/重排）")
    parser.add_argument("--report", metavar="TOPIC", help="一次性生成报告")
    parser.add_argument("--papers", type=int, default=0, help="报告/入库的最大论文数")
    parser.add_argument("--simple", action="store_true", help="报告使用 simple 单 agent 模式")
    parser.add_argument("--offline", action="store_true", help="假模型 + 独立索引目录")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    parser.add_argument("--verbose", "-v", action="store_true", help="打开详细日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    repl = Repl(offline=args.offline, stream=not args.no_stream)

    if args.search:
        extra = ""
        if args.search_ingest:
            extra += " --ingest"
        if args.no_llm:
            extra += " --no-llm"
        repl.cmd_search(f"{args.search} --limit {args.papers or 8}{extra}")
        return 0
    if args.ingest or args.ids:
        parts = [args.ingest or ""]
        parts.append(f"--limit {args.papers or 3}")
        if args.ids:
            parts.append(f"--ids {args.ids}")
        repl.cmd_ingest(" ".join(p for p in parts if p).strip())
        return 0
    if args.report:
        flags = f" --papers {args.papers}" if args.papers else ""
        flags += " --simple" if args.simple else ""
        repl.cmd_report(f"{args.report}{flags}")
        return 0
    if args.question:
        repl.cmd_ask(" ".join(args.question))
        return 0

    if not sys.stdin.isatty():
        # 管道输入：逐行执行（便于脚本化 / 测试）
        repl.banner()
        for line in sys.stdin:
            if not repl.safe_dispatch(line.rstrip("\n")):
                break
        _save_readline_history()
        return 0

    repl.repl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
