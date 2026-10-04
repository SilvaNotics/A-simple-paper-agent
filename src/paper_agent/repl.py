# -*- coding: utf-8 -*-
"""Repl 核心：会话状态、命令分发与帮助、REPL 主循环、事件循环与流式渲染。

命令实现按域拆到三个 mixin（`repl_search` / `repl_papers` / `repl_providers`），
与本类合成同一个 `Repl`：方法之间可以互相调用（`self.cmd_*` / `self._helper`）。
共享状态都在 `__init__` 里建立：`settings` / `session` / `config` / `history` /
`stream` / `_live` / `_loop` / `_pdf_server`；输出统一走 `ui.console`。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import ui
from .cli import cli_settings
from .pdf_server import PdfServer
from .pipeline import Session, build_session
from .repl_input import PaletteContext, create_session, read_line
from .repl_ui import (
    COMMANDS,
    COMMAND_USAGE,
    HELP_EXAMPLES,
    HELP_GROUPS,
    save_readline_history,
    setup_readline,
)
from .userconfig import UserConfig

from .repl_papers import PaperCommands
from .repl_providers import ProviderCommands
from .repl_search import SearchCommands


class Repl(SearchCommands, PaperCommands, ProviderCommands):
    """Repl 核心：会话状态、命令分发与帮助、REPL 主循环、事件循环与流式渲染。"""

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
        # `/papers open` 起的本地 PDF 预览服务（REPL 退出时停止）
        self._pdf_server: PdfServer | None = None

    def require_session(self) -> Session | None:
        """需要 RAG/模型的命令统一入口：未配置时给出 /connect 引导。"""
        if self.session is not None:
            return self.session
        ui.console.print(f"[yellow]尚未完成配置[/yellow]：{self.session_error or '缺少模型 / embedding 配置'}")
        ui.console.print(
            "[bold]请先执行 [cyan]/connect[/cyan][/bold]：输入 base URL 与 API key 即可"
            "（自动识别供应商与模型列表；支持直接粘贴，key 不回显）。"
            "也可用 [cyan]/connect 2[/cyan] 之类预设。"
        )
        ui.console.print("[dim]非交互方式：/connect https://api.deepseek.com sk-xxx\n[/dim]")
        return None

    def banner(self) -> None:
        if self.session is None:
            ui.console.print(
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
        ui.console.print(
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
            ui.console.print(f"[green]✓[/green] 流式输出：{'开' if self.stream else '关'}")
        elif command == "/history":
            self.show_history(int(rest) if rest.isdigit() else 5)
        elif command == "/logs":
            self.cmd_logs(rest)
        elif command == "/save":
            self.save_history(rest)
        elif command == "/clear":
            ui.console.clear()
            self.banner()
        else:
            ui.console.print(f"[yellow]未知命令 {command}，输入 /help 查看可用命令[/yellow]")
        return True

    def safe_dispatch(self, line: str) -> bool:
        """执行一行输入并吞掉异常（REPL 不应因单条命令崩溃）；返回 False 表示退出。"""
        try:
            return self.dispatch(line)
        except KeyboardInterrupt:
            self._stop_live()
            ui.console.print("\n[yellow]已中断当前操作[/yellow]")
            return True
        except EOFError:
            # stdin 结束（Ctrl+D / 管道关闭）：视为取消当前命令，不打 traceback
            self._stop_live()
            ui.console.print("\n[yellow]输入已结束（EOF），当前操作已取消[/yellow]")
            return True
        except Exception as exc:  # noqa: BLE001
            self._stop_live()
            logging.getLogger(__name__).exception("命令执行失败")
            ui.console.print(f"[red]执行失败：{type(exc).__name__}: {exc}[/red]")
            return True

    def _print_help(self, topic: str = "") -> None:
        """`/help`：分组展示命令与用法；`/help search` 只看某条命令的细节。"""
        key = (topic or "").strip().lstrip("/").split(" ")[0].lower()
        if key:
            hits = [c for c in COMMANDS if c.lstrip("/").lower() == key]
            if not hits:
                hits = [c for c in COMMANDS if c.lstrip("/").lower().startswith(key)]
            if not hits:
                ui.console.print(f"[yellow]没有命令 /{key}[/yellow]（输入 /help 查看全部）")
                return
            for cmd in hits:
                # 用法里含 [--flag] 这类方括号，必须用 Text 输出，否则会被 rich 当成标记吞掉
                line = Text()
                line.append(cmd, style="bold cyan")
                line.append("  " + COMMANDS[cmd], style="dim")
                ui.console.print(line)
                ui.console.print(Text("  " + COMMAND_USAGE.get(cmd, cmd), style="cyan"))
                ui.console.print()
            return

        ui.console.print(
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
            ui.console.print(f"[bold]{title}[/bold]")
            ui.console.print(table)
            ui.console.print()

        table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
        table.add_column("ex", no_wrap=True)
        table.add_column("note", overflow="fold")
        for cmd, note in HELP_EXAMPLES:
            table.add_row(Text(cmd, style="cyan"), Text(note, style="dim"))
        ui.console.print("[bold]常用示例[/bold]")
        ui.console.print(table)
        ui.console.print("[dim]看某条命令细节：[/dim] [cyan]/help search[/cyan]")

    def repl(self) -> None:
        self.banner()
        session = create_session(self.palette_context())
        if session is None:
            setup_readline()  # 没装 prompt_toolkit → 退回 readline + 普通输入
        else:
            ui.console.print(
                "[dim]提示：输入 [bold]/[/bold] 会弹出命令面板（Tab 补全 / Enter 确认），"
                "↑↓·PgUp/PgDn 滚动，底部状态栏常显供应商与索引规模。[/dim]"
            )
        while True:
            try:
                line = read_line(session) if session is not None else ui.console.input(ui.PROMPT)
            except (EOFError, KeyboardInterrupt):
                ui.console.print("\n[dim]再见 👋[/dim]")
                break
            except Exception as exc:  # noqa: BLE001 - prompt_toolkit 环境异常时兜底
                logging.getLogger(__name__).warning("输入层异常，改用普通输入：%s", exc)
                session = None
                continue
            if not self.safe_dispatch(line):
                ui.console.print("[dim]再见 👋[/dim]")
                break
        save_readline_history()
        self.close_pdf_viewer(quiet=True)
        self._close_loop()

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

    def _stop_live(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            finally:
                self._live = None

    def _stream_renderer(self) -> Callable[[str], None]:
        """边生成边直接追加打印（append-only，不做整段重绘）。

        Rich `Live` 重绘整段 Markdown 时，内容超过一屏后随终端滚动会重复打印内容；
        append-only 则只会向下滚。代价：流式期间不做 Markdown 渲染。
        """

        def on_token(token: str) -> None:
            if token:
                ui.console.print(token, end="", markup=False, highlight=False, soft_wrap=True)

        return on_token
