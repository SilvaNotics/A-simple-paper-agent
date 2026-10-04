# -*- coding: utf-8 -*-
"""论文库 / 索引 / 本地 PDF 预览 / 日志 / 历史相关命令（`Repl` 的 mixin）。

- `/papers`（列表 / `rm` / `open` / `close`）、`/index`、`/logs`、`/history`、`/save`；
- PDF 预览的本地 HTTP 服务生命周期也在这里（起服务、停止、退出时收尾）。
"""

from __future__ import annotations

import logging
import sys
import time

from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import ui
from .config import resolve_path
from .logging_setup import (
    backup_count,
    keep_days,
    list_log_files,
    log_mode,
    log_size,
    max_bytes,
    resolve_log_dir,
    resolve_log_file,
    tail_log,
)
from .pdf_server import (
    DEFAULT_PORT as PDF_DEFAULT_PORT,
    collect_pdf_entries,
    registered_server,
    start_viewer,
    stop_registered_server,
)
from .pipeline import collapse_repetition, remove_papers as pipeline_remove_papers
from .tui import ask_line, pick_from_list
from .utils import clean_pasted, flag_bool, split_args

from .repl_base import ReplBase


class PaperCommands(ReplBase):
    """论文库 / 索引 / 本地 PDF 预览 / 日志 / 历史相关命令（`Repl` 的 mixin）。"""

    def show_index(self) -> None:
        session = self.require_session()
        if session is None:
            return
        stats = session.stats()
        ui.console.print(
            f"[bold]索引[/bold] {stats['papers']} 篇 / {stats['chunks']} chunks   "
            f"[dim]{stats['data_dir']}[/dim]"
        )

    def show_papers(self) -> None:
        session = self.require_session()
        if session is None:
            return
        papers = session.index.list_papers()
        if not papers:
            ui.console.print("[yellow]索引为空，先 /ingest <主题> 或用 /search 看看能检索到什么[/yellow]")
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
        ui.console.print(table)

    def cmd_papers(self, args: str = "") -> None:
        """`/papers` 列表；`/papers rm <id>|--all` 删除；`/papers open|close` 本地 PDF 预览。"""
        action, _, rest = args.strip().partition(" ")
        key = action.lower()
        if key in {"rm", "remove", "del", "delete"}:
            self.remove_papers(rest)
            return
        if key in {"open", "view", "serve"}:
            self.open_pdf_viewer(rest)
            return
        if key in {"close", "stop", "off"}:
            self.close_pdf_viewer()
            return
        if action:
            ui.console.print(
                "[yellow]用法：/papers | /papers rm <paper_id>[,<id>] | /papers rm --all | "
                "/papers open [--port N] [--host H] [--idle-timeout MIN] [--no-browser] | /papers close[/yellow]"
            )
            return
        self.show_papers()

    def open_pdf_viewer(self, args: str = "") -> None:
        """`/papers open [--port N] [--host H] [--idle-timeout MIN] [--no-browser]`。

        只监听回环地址（默认 `127.0.0.1:8765`，被占用自动换空闲端口），终端打印完整 URL；
        远程机器用 `ssh -L 8765:127.0.0.1:8765 <host>` 把端口转发到本地浏览器即可。

        退出机制（见 `pdf_server` 模块说明）：本进程起的用 `/papers close` 停；注册表里
        **别的进程**的服务也一并停；默认空闲 30 分钟自动退出（`--idle-timeout 0` 关闭）。
        """
        _positional, flags = split_args(args)
        raw_port = flags.get("port", "")
        port = int(raw_port) if raw_port.isdigit() and 0 <= int(raw_port) <= 65535 else PDF_DEFAULT_PORT
        host = flags.get("host", "") or "127.0.0.1"
        raw_idle = flags.get("idle-timeout", "")
        idle = float(raw_idle) * 60 if raw_idle.replace(".", "", 1).isdigit() else None

        if self._pdf_server is not None and self._pdf_server.running:
            ui.console.print(f"[dim]PDF 预览服务已在运行：[/dim][bold cyan]{self._pdf_server.url}[/bold cyan]")
            return

        external = registered_server()
        if external is not None:
            ui.console.print(
                f"[dim]已有另一个进程的预览服务在跑：[/dim][bold cyan]{external.get('url')}[/bold cyan]"
                "[dim]（想换端口先 `/papers close`）[/dim]"
            )
            return

        papers_dir = self.settings.papers_dir
        try:
            self._pdf_server = start_viewer(
                lambda: collect_pdf_entries(papers_dir, self._index_rows()),
                papers_dir=papers_dir,
                host=host,
                port=port,
                idle_seconds=idle,
                open_browser=not flag_bool(flags, "no-browser") and sys.stdin.isatty(),
            )
        except OSError as exc:
            ui.console.print(f"[red]无法启动本地 PDF 预览服务（{host}:{port}）：{exc}[/red]")
            return

    def close_pdf_viewer(self, quiet: bool = False) -> None:
        """停掉预览服务：先停本进程的，再停注册表里其它进程的（`/papers close`）。"""
        server, self._pdf_server = self._pdf_server, None
        stopped_own = server is not None and server.running
        if server is not None:
            server.stop()
        external = stop_registered_server()
        if not stopped_own and external is None:
            if not quiet:
                ui.console.print("[dim]没有正在运行的 PDF 预览服务[/dim]")
            return
        if not quiet:
            where = "本进程" if stopped_own else f"另一个进程 pid={external.get('pid') if external else '?'}"
            ui.console.print(f"[green]✓[/green] 已停止本地 PDF 预览服务（{where}）")

    def _index_rows(self) -> list[dict]:
        """当前索引里的论文行（没配供应商/没 session 时返回空——看 PDF 不需要模型）。"""
        if self.session is None:
            return []
        try:
            return self.session.index.list_papers()
        except Exception as exc:  # noqa: BLE001 - 索引坏了不该 block 住预览
            logging.getLogger(__name__).debug("读取索引列表失败，预览仍可用：%s", exc)
            return []

    def remove_papers(self, arg: str) -> None:
        """从索引（与本地 PDF 缓存）中删除论文；不给 id 时弹选择器。"""
        session = self.require_session()
        if session is None:
            return
        papers = session.index.list_papers()
        if not papers:
            ui.console.print("[yellow]索引为空，没有可删除的论文[/yellow]")
            return

        token = clean_pasted(arg).strip()
        targets: list[str] = []
        if token in {"--all", "all", "*"}:
            answer = ask_line(ui.console, f"确认删除全部 {len(papers)} 篇论文？[y/N] › ")
            if answer.lower() not in {"y", "yes"}:
                ui.console.print("[dim]已取消[/dim]")
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
                ui.console,
                ids,
                title="选择要删除的论文",
                display=display,
                footer="Enter 删除 / Esc 取消",
                display_class="pick.paper",
            )
            if action == "cancel" or not value:
                ui.console.print("[dim]已取消[/dim]")
                return
            targets = [value]

        removed = pipeline_remove_papers(targets, session=session)
        for paper_id in targets:
            if paper_id in removed:
                ui.console.print(f"[green]✓[/green] 已删除 {paper_id}（chunks + 本地 PDF）")
            else:
                ui.console.print(f"[yellow]未找到 {paper_id}[/yellow]")
        if removed:
            self.show_index()

    def cmd_logs(self, args: str = "") -> None:
        """`/logs [n]`：日志文件路径 + 末尾 n 行；`/logs --files`：列出按天分的历史文件。

        文件默认按天写 `<仓库根>/logs/paper-agent-YYYY-MM-DD.log`（保留 14 天，
        单日超过 8 MiB 续写 `-02`）；日志行里的 `[` 等字符一律当纯文本打印，不被 rich 当标记。
        """
        positional, flags = split_args(args)
        if flag_bool(flags, "files"):
            self._list_log_files()
            return
        token = positional.split()[0] if positional.split() else ""
        n = int(token) if token.isdigit() else 20

        path = resolve_log_file()
        mode = log_mode()
        ui.console.print(f"日志文件：[cyan]{path}[/cyan]")
        if mode == "daily":
            ui.console.print(
                f"[dim]按天分文件 · 目录 {resolve_log_dir()} · 保留 {keep_days()} 天 · "
                f"单文件上限 {max_bytes() // 1024 // 1024} MiB（超出续写 -02）[/dim]"
            )
        else:
            ui.console.print(
                f"[dim]固定单文件（PAPER_AGENT_LOG_FILE）· "
                f"轮转 {max_bytes() // 1024 // 1024} MiB × {backup_count()} 份[/dim]"
            )
        size = log_size(path)
        if not size:
            ui.console.print(
                "[dim]（还没有日志内容；置 PAPER_AGENT_LOG_DISABLE=1 可关闭文件日志）[/dim]"
            )
            return
        ui.console.print(f"[dim]大小 {size / 1024:.1f} KiB[/dim]")
        lines = tail_log(n, path)
        body = Text("\n".join(lines) or "（空）", style="dim")
        ui.console.print(
            Panel(
                body,
                title=f"最近 {len(lines)} 行",
                border_style="cyan",
                padding=(0, 1),
            )
        )

    def _list_log_files(self) -> None:
        """`/logs --files`：列出按天分文件的日志（从新到旧，含大小）。"""
        if log_mode() != "daily":
            ui.console.print(f"[dim]固定单文件模式，只有一份：{resolve_log_file()}[/dim]")
            return
        files = list_log_files()
        if not files:
            ui.console.print(f"[dim]还没有日志文件（目录 {resolve_log_dir()}）[/dim]")
            return
        table = Table(title=f"日志文件（{len(files)} 份 · 新 → 旧）", show_header=True, header_style="bold")
        table.add_column("#", justify="right", style="dim")
        table.add_column("文件", style="cyan", no_wrap=True)
        table.add_column("大小", justify="right")
        table.add_column("修改时间", style="dim")
        for index, path in enumerate(files, 1):
            try:
                stat = path.stat()
                size, mtime = f"{stat.st_size / 1024:.1f} KiB", time.strftime("%m-%d %H:%M", time.localtime(stat.st_mtime))
            except OSError:
                size, mtime = "-", "-"
            table.add_row(str(index), path.name, size, mtime)
        ui.console.print(table)
        ui.console.print(
            f"[dim]保留最近 {keep_days()} 天（PAPER_AGENT_LOG_KEEP_DAYS 可改）；默认看最新一份：/logs[/dim]"
        )

    def show_history(self, n: int = 5) -> None:
        if not self.history:
            ui.console.print("[dim]还没有问答记录[/dim]")
            return
        for item in self.history[-n:]:
            ui.console.print(f"[bold cyan]Q[/bold cyan] {item['question']}")
            ui.console.print(Markdown(item["answer"][:1200] or "（空）"))
            cited = ", ".join(item["citations"]) or "（无）"
            status = "[green]引用校验通过[/green]" if not item["problems"] else f"[yellow]{'; '.join(item['problems'])}[/yellow]"
            ui.console.print(f"[dim]引用：{cited}[/dim]  {status}\n")

    def save_history(self, path: str = "") -> None:
        if not self.history:
            ui.console.print("[yellow]没有可保存的内容[/yellow]")
            return
        target = resolve_path(path) if path else (
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
        ui.console.print(f"[green]✓[/green] 已保存：{target}")
