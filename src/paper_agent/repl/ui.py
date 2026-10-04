# -*- coding: utf-8 -*-
"""REPL 的展示层：命令表（`/help` 与补全共用）、检索进度视图、readline 历史。

这里只放**不依赖会话状态**的东西：`Repl`（`repl/app.py`）与 `main.py` 都从这里取命令表，
`repl/input.py` 的 `create_session()` 用 `COMMANDS` 做命令面板补全。
"""

from __future__ import annotations

from typing import Any, Literal

from rich.table import Table
from rich.text import Text

from ..sources.channels import is_domestic
from ..core.config import HISTORY_FILE

# --------------------------------------------------------------------------
# 命令表（命令名 → 一句话说明 + 用法签名）
# --------------------------------------------------------------------------

COMMANDS: dict[str, str] = {
    "/search": "联网检索论文（--limit = 每渠道上限；可用 LLM 扩展检索式并重排）",
    "/ingest": "下载 PDF 并入库，建立/更新向量索引",
    "/ask": "基于已入库语料带引用问答（直接输入自然语言同效）",
    "/report": "端到端生成调研报告（md / bib / json）",
    "/papers": "查看或删除已入库论文；`open` 起本地预览（浏览器看抓到的 PDF）",
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
    "/logs": "查看日志文件路径与末尾内容（默认按天分文件，/logs --files 看历史）",
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
    "/papers": "/papers [rm <id>|--all] | /papers open [--port N] [--host H] [--idle-timeout MIN] [--no-browser] | /papers close",
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
    "/logs": "/logs [n] [--files]",
    "/clear": "/clear",
    "/exit": "/exit",
    "/help": "/help [命令名]",
}

# /help 的分组展示顺序（只列命令名）
HELP_GROUPS: list[tuple[str, list[str]]] = [
    ("检索与抓取", ["/search", "/ingest", "/channels", "/index", "/mcp"]),
    ("问答与报告", ["/ask", "/report", "/papers", "/history", "/save"]),
    ("模型与供应商", ["/connect", "/providers", "/keys", "/models", "/model", "/embed"]),
    ("会话与调试", ["/offline", "/stream", "/history", "/logs", "/clear", "/help", "/exit"]),
]

# /help 末尾的常用示例
HELP_EXAMPLES: list[tuple[str, str]] = [
    ("/search graph rag --limit 5", "检索 5 篇 Graph RAG 论文"),
    ("/search graph rag --ingest", "检索并直接下载入库"),
    ("/ingest --ids arxiv:2405.16506", "已知 arXiv ID 直接抓取入库"),
    ("/channels all on", "检索时并发跑全部已注册渠道"),
    ("/ask 图 RAG 的主要方法有哪些", "基于本地语料带引用回答"),
    ("/report 图检索增强生成 --papers 3", "生成一份中文调研报告"),
    ("/logs 50", "看最近 50 行运行日志（定位报错/回退原因）"),
    ("/logs --files", "列出按天分的历史日志文件（默认保留 14 天）"),
    ("/papers open", "浏览器查看已抓到的 PDF（终端打印本地 HTTP 端口与退出方式）"),
    ("/papers close", "停掉预览服务（含其它进程起的那个）"),
]


# --------------------------------------------------------------------------
# readline（未装 prompt_toolkit 时的退化输入层：行编辑 / 历史 / Tab 补全）
# --------------------------------------------------------------------------


def setup_readline() -> None:
    """启用行编辑 / 历史 / Tab 补全（stdlib readline，无需额外依赖）。

    补全与 prompt_toolkit 命令面板对齐：第一个词补 `/命令`，之后补该命令支持的
    子命令 / `--flag`（例如 `/papers ` → `open` / `close` / `--port` …）。
    """
    try:
        import readline
    except ImportError:  # pragma: no cover - Windows 无 readline
        return

    def completer(text: str, state: int):
        options = _readline_options(readline.get_line_buffer(), text)
        return options[state] if state < len(options) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        readline.read_history_file(HISTORY_FILE)
    except OSError:
        pass
    readline.set_history_length(1000)


def _readline_options(line: str, text: str) -> list[str]:
    """readline Tab 补全的候选项（与 prompt_toolkit 命令面板同一份命令/参数表）。"""
    from .input import COMMAND_FLAGS

    head, _, _tail = line.partition(" ")
    if " " in line:
        return [flag for flag in COMMAND_FLAGS.get(head, ()) if flag.startswith(text)]
    if not text.startswith("/"):
        return []
    return [command for command in COMMANDS if command.startswith(text)]


def save_readline_history() -> None:
    """退出时落盘 readline 历史（失败无所谓）。"""
    try:
        import readline

        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(HISTORY_FILE)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------
# 检索进度视图
# --------------------------------------------------------------------------


class SearchProgressView:
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
            return Text.from_markup(f"[cyan]联网检索中（{self.label}）：{self.query}[/cyan]")
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


class nullcontext:
    """`contextlib.nullcontext` 的极简替代：stdin 非 TTY / 流式输出时替代 `console.status`。

    只实现 `with` 协议，避免为了一个「什么都不做」的上下文管理器再引标准库。
    """

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> Literal[False]:
        return False


__all__ = [
    "COMMANDS",
    "COMMAND_USAGE",
    "HELP_EXAMPLES",
    "HELP_GROUPS",
    "SearchProgressView",
    "nullcontext",
    "save_readline_history",
    "setup_readline",
]
