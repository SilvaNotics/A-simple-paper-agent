# -*- coding: utf-8 -*-
"""交互式输入层：命令面板（输入 `/` 即出现可滚动补全列表）。

用 `prompt_toolkit` 实现：
- 边输入边弹列表（含说明），可滚动；`Tab` 补全、`Enter` 确认并执行；
- 支持参数级补全：`--flag`、供应商名、模型名、已入库论文 ID；
- 底部状态栏常驻显示「供应商 · 模型 · 索引规模」；历史存项目内 `.paper-agent/history.ptk`。

未安装 `prompt_toolkit` 时自动退化为 `rich` + 标准输入。
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable

from ..core.config import PTK_HISTORY_FILE

logger = logging.getLogger(__name__)

# 渠道类型（给 `/channels add` / `/keys rm channel:` 补全用）
try:  # 避免任何导入异常影响输入层
    from ..sources.channels import PRESET_ORDER as CHANNEL_KINDS
    from ..sources.channels import REGISTRY as _CHANNEL_REGISTRY

    CHANNEL_KINDS = [*CHANNEL_KINDS, *_CHANNEL_REGISTRY.keys()]
except Exception:  # noqa: BLE001  # pragma: no cover
    CHANNEL_KINDS = []

HISTORY_PATH = PTK_HISTORY_FILE

# ---------------------------------------------------------------------------
# 界面主题（补全菜单 / 命令面板 / 底部状态栏）
#   深色底 + 蓝字为默认；设 PAPER_AGENT_THEME=light 可换浅色，none 用库默认
# ---------------------------------------------------------------------------
THEME_ENV = "PAPER_AGENT_THEME"

_THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "prompt": "#4da3ff bold",
        # 弹出列表：深色底 + 蓝色字
        "completion-menu": "bg:#0b1116 #4da3ff",
        "completion-menu.completion": "bg:#0b1116 #4da3ff",
        "completion-menu.completion.current": "bg:#1d4ed8 #e8f2ff bold",
        "completion-menu.meta.completion": "bg:#0b1116 #93a4bd",
        "completion-menu.meta.completion.current": "bg:#1d4ed8 #dbeafe",
        # 模糊匹配高亮：命中的字符更亮
        "completion-menu.completion fuzzymatch.inside": "#e8f2ff bold",
        "completion-menu.completion fuzzymatch.outside": "#3b82f6",
        # 滚动条
        "scrollbar.background": "bg:#0b1116",
        "scrollbar.button": "bg:#2563eb",
        "scrollbar.arrow": "#60a5fa",
        # 底部状态栏
        "bottom-toolbar": "bg:#0b1116 #60a5fa",
        "bottom-toolbar.text": "#60a5fa",
        "frame.border": "#1d4ed8",
        # ---- 语义化配色（给列表项与标记上色）----
        "mark.current": "#22c55e bold",      # ●当前  绿色
        "mark.default": "#f59e0b bold",      # ★默认  琥珀
        "pick.model": "#4da3ff",             # 模型名
        "pick.provider": "#22d3ee bold",     # 供应商名
        "pick.paper": "#a78bfa",             # 论文 ID
        "pick.embedding": "#34d399",         # embedding 模型
        "pick.dim": "#64748b",               # 次要说明
        "cmd.name": "#4da3ff bold",          # 命令
        "cmd.flag": "#f59e0b",               # 参数
        "cmd.meta": "#93a4bd",               # 命令说明
        "hint.key": "#f59e0b bold",          # 提示里的按键
        "hint.text": "#93a4bd",
        "hint.title": "#4da3ff bold",
    },
    "light": {
        "prompt": "#1d4ed8 bold",
        "completion-menu": "bg:#f1f5f9 #1d4ed8",
        "completion-menu.completion": "bg:#f1f5f9 #1d4ed8",
        "completion-menu.completion.current": "bg:#1d4ed8 #f8fafc bold",
        "completion-menu.meta.completion": "bg:#f1f5f9 #2563eb",
        "completion-menu.meta.completion.current": "bg:#1d4ed8 #e0e7ff",
        "scrollbar.background": "bg:#e2e8f0",
        "scrollbar.button": "bg:#1d4ed8",
        "bottom-toolbar": "bg:#e2e8f0 #1d4ed8",
        "bottom-toolbar.text": "#1d4ed8",
        "frame.border": "#1d4ed8",
        "mark.current": "#16a34a bold",
        "mark.default": "#d97706 bold",
        "pick.model": "#1d4ed8",
        "pick.provider": "#0891b2 bold",
        "pick.paper": "#7c3aed",
        "pick.embedding": "#059669",
        "pick.dim": "#64748b",
        "cmd.name": "#1d4ed8 bold",
        "cmd.flag": "#d97706",
        "cmd.meta": "#64748b",
        "hint.key": "#d97706 bold",
        "hint.text": "#475569",
        "hint.title": "#1d4ed8 bold",
    },
}


def theme_name() -> str:
    value = (os.getenv(THEME_ENV) or "dark").strip().lower()
    return value if value in _THEMES else ("none" if value in {"none", "off", "default"} else "dark")


def build_style():
    """构造 prompt_toolkit 样式；主题为 none 时返回 None（用库默认配色）。"""
    name = theme_name()
    if name not in _THEMES:
        return None
    try:
        from prompt_toolkit.styles import Style

        return Style.from_dict(_THEMES[name])
    except Exception:  # noqa: BLE001 - 没装 prompt_toolkit 时无所谓
        return None

# 每个命令支持的参数（用于参数级补全）
COMMAND_FLAGS: dict[str, tuple[str, ...]] = {
    "/search": ("--limit", "--sources", "--source", "--ingest", "--no-llm", "--llm", "--force"),
    "/ingest": ("--limit", "--ids", "--force", "--source"),
    "/ask": ("--papers", "--k"),
    "/report": ("--papers", "--simple", "--source"),
    "/papers": ("rm", "--all", "open", "close", "--port", "--host", "--idle-timeout", "--no-browser"),
    "/channels": ("add", "rm", "key-rm", "on", "off", "all", "domestic", "list"),
    "/keys": ("rm",),
    "/connect": ("--key", "--name", "--kind", "--no-fetch", "--allow-empty-key"),
    "/models": ("--all", "--embedding", "--provider", "--refresh"),
    "/providers": ("use", "rm", "key-rm", "sync"),
    "/model": ("--default",),
    "/embed": ("--model", "--provider", "auto", "set"),
    "/offline": ("on", "off"),
    "/stream": ("on", "off"),
    "/history": (),
    "/save": (),
}

FLAG_DOCS: dict[str, str] = {
    "--limit": "条数上限",
    "--sources": "检索源子集，如 arxiv,openalex",
    "--source": "检索层：auto / mcp / builtin / all",
    "--ingest": "检索完直接入库（可跟一个数字限定篇数）",
    "--no-llm": "检索时不调用 LLM（关闭查询扩展/重排）",
    "--llm": "强制在检索中使用 LLM",
    "--ids": "按 ID 直抓（arXiv ID / DOI / 链接，逗号分隔）",
    "--force": "已存在的论文重新解析入库",
    "--papers": "限定论文 ID（逗号分隔）或报告篇数",
    "--k": "检索片段数",
    "--simple": "报告用单 agent 模式",
    "--key": "API key（也可交互输入，支持粘贴）",
    "--name": "名称标识",
    "--kind": "强制指定类型",
    "--no-fetch": "跳过 /models 拉取",
    "--allow-empty-key": "允许空 key（本地服务）",
    "--all": "包含被过滤的模型 / 删除全部论文",
    "--embedding": "选择 embedding 模型",
    "--provider": "指定供应商（不给值则弹出选择器）",
    "--refresh": "重新拉取模型列表",
    "--default": "同时写入默认",
    "--port": "预览服务端口（默认 8765，被占用自动换空闲端口）",
    "--host": "预览服务监听地址（默认仅本机 127.0.0.1）",
    "--idle-timeout": "预览服务空闲多少分钟后自动退出（0 = 不自动退出）",
    "--no-browser": "起服务但不自动打开浏览器",
    "open": "起本地 PDF 预览服务（浏览器看抓到的 PDF）",
    "close": "停掉本地 PDF 预览服务",
    "auto": "自动挑选 / 恢复自动",
    "set": "交互选择",
    "use": "切换默认供应商",
    "rm": "删除（供应商/渠道/论文/key）",
    "key-rm": "只删除 API key，保留配置",
    "add": "添加搜索渠道",
    "list": "列出",
    "on": "开启",
    "off": "关闭",
    "sync": "重新拉取该供应商的模型",
}


@dataclass
class PaletteContext:
    """补全所需的动态上下文（由 REPL 提供）。"""

    commands: dict[str, str] = field(default_factory=dict)
    providers: Callable[[], list[str]] = lambda: []
    models: Callable[[], list[str]] = lambda: []
    indexed: Callable[[], list[str]] = lambda: []
    channels: Callable[[], list[str]] = lambda: []
    status: Callable[[], str] = lambda: ""


def _available() -> bool:
    try:
        import prompt_toolkit  # noqa: F401

        return True
    except ImportError:  # pragma: no cover
        return False


def build_completer(ctx: PaletteContext):
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import FormattedText

    def _item(
        text: str,
        style: str,
        meta: str = "",
        start_position: int = 0,
        meta_style: str = "class:cmd.meta",
        typed: str = "",
    ) -> Completion:
        # 注意：Completion 是 frozen dataclass，start_position 必须在构造时传入
        return Completion(
            text,
            start_position=start_position,
            display=highlight_text(typed, text, style),
            display_meta=FormattedText([(meta_style, meta)]) if meta else "",
        )

    class SlashCompleter(Completer):
        """`/命令` + 参数 + 动态值（供应商/模型/论文 ID）补全；输入即显示 + 模糊打分排序。"""

        def _ranked(self, items: list[str], typed: str):
            scored = [(match_score(typed, item), item) for item in items]
            return sorted(
                ((score, item) for score, item in scored if score is not None),
                key=lambda pair: (pair[0], pair[1].lower()),
            )

        def _ranked_commands(self, typed: str):
            # 命令名只用前缀/子串（不做子序列模糊，否则 `/p` 会带出 `/mcp`/`/report` 等噪声）
            scored = [(command_match_score(typed, cmd), cmd) for cmd in ctx.commands]
            return sorted(
                ((score, cmd) for score, cmd in scored if score is not None),
                key=lambda pair: (pair[0], pair[1].lower()),
            )

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            head, _, tail = text.partition(" ")
            if not tail and " " not in text:  # 还在敲命令名
                for _score, cmd in self._ranked_commands(head):
                    # 补全后自动带一个空格：Tab 选完命令就能直接接参数，并触发 flag 补全。
                    # display 不带空格（菜单里看起来仍然是 `/search`）。
                    yield Completion(
                        cmd + " ",
                        start_position=-len(head),
                        display=highlight_text(head, cmd, "class:cmd.name"),
                        display_meta=(
                            FormattedText([("class:cmd.meta", ctx.commands[cmd])])
                            if ctx.commands.get(cmd)
                            else ""
                        ),
                    )
                return

            partial = tail.split()[-1] if tail and not tail.endswith(" ") else ""
            prefix = "" if tail.endswith(" ") else partial
            if head == "/model" and not prefix.startswith("-"):
                for _score, model in self._ranked(ctx.models(), prefix):
                    yield _item(model, "class:pick.model", "模型", start_position=-len(prefix), typed=prefix)
                return
            if head == "/embed" and not prefix.startswith("-") and len(tail.split()) <= 1:
                for _score, name in self._ranked(ctx.providers(), prefix):
                    yield _item(name, "class:pick.provider", "供应商", start_position=-len(prefix), typed=prefix)
                return
            if head in {"/providers", "/models"} and (
                tail.startswith("use") or tail.startswith("rm") or tail.startswith("sync") or "--provider" in tail
            ):
                for _score, name in self._ranked(ctx.providers(), prefix):
                    yield _item(name, "class:pick.provider", "供应商", start_position=-len(prefix), typed=prefix)
                return
            if head == "/channels" and (
                tail.startswith("rm")
                or tail.startswith("key-rm")
                or tail.startswith("on")
                or tail.startswith("off")
            ):
                for _score, name in self._ranked(ctx.channels(), prefix):
                    yield _item(name, "class:pick.provider", "搜索渠道", start_position=-len(prefix), typed=prefix)
                return
            if head in {"/keys", "/channels"} and tail.startswith("add"):
                for _score, kind in self._ranked(list(CHANNEL_KINDS), prefix):
                    yield _item(kind, "class:pick.provider", "渠道类型", start_position=-len(prefix), typed=prefix)
                return
            if head in {"/papers", "/ask", "/report"} and (
                tail.startswith("rm") or "--papers" in tail
            ):
                for _score, pid in self._ranked(ctx.indexed(), prefix):
                    yield _item(pid, "class:pick.paper", "已入库", start_position=-len(prefix), typed=prefix)
                return
            if head == "/ingest" and "--ids" in tail:
                for _score, pid in self._ranked(ctx.indexed(), prefix):
                    yield _item(pid, "class:pick.paper", "已入库", start_position=-len(prefix), typed=prefix)
            for _score, flag in self._ranked(list(COMMAND_FLAGS.get(head, ())), prefix):
                yield _item(flag, "class:cmd.flag", FLAG_DOCS.get(flag, ""), start_position=-len(prefix), typed=prefix)

    return SlashCompleter()


def create_session(ctx: PaletteContext):
    """创建 prompt_toolkit 会话；不可用时返回 None（调用方退回 console.input）。"""
    if not _available():
        return None
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style

        kb = confirm_key_bindings()   # Tab/Enter 都是确定（Tab 先补全、已是该项则执行）

        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

        style = build_style() or Style.from_dict(
            {"prompt": "ansicyan bold", "bottom-toolbar": "bg:#0b1116 #60a5fa"}
        )

        return PromptSession(
            completer=build_completer(ctx),
            complete_while_typing=True,
            complete_style="COLUMN",                # 单列 + 说明，长列表自动滚动
            # 注意：**不能**开 enable_history_search —— prompt_toolkit 在它开启时会强制关掉
            # complete_while_typing（见 shortcuts/prompt.py），导致输入 `/`+字母不再弹补全菜单。
            # 上下方向键仍可翻阅历史。
            history=FileHistory(str(HISTORY_PATH)),
            key_bindings=kb,
            style=style,
            bottom_toolbar=lambda: ctx.status(),
            mouse_support=False,                   # 保留终端自身的复制/粘贴
        )
    except Exception as exc:  # noqa: BLE001 - 环境异常时退回普通输入
        logger.warning("prompt_toolkit 初始化失败，退回普通输入：%s", exc)
        return None


# --------------------------------------------------------------------------
# 模糊匹配：子串优先 + 全串子序列（prompt_toolkit 自带算法按“词”匹配，
# `q38` 匹配不到 `qwen3.8-max`，因为名字被 `.`/`-` 切成了多个词）
# --------------------------------------------------------------------------


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", (text or "").lower())


def command_match_score(typed: str, command: str) -> tuple[int, int] | None:
    """命令名专用打分：只认「前缀 → 子串」，故意不做子序列模糊。

    模糊子序列对模型名很好用（`q38` → `qwen3.8-max`），但对命令名是噪声：
    输入 `/p` 不应该冒出 `/mcp`、`/report`、`/help`。同时支持省略斜杠输入
    （`se` 也能命中 `/search`）。
    """
    t = (typed or "").strip().lower()
    c = (command or "").lower()
    if not t:
        return (0, 0)
    if t.startswith("/"):
        if c.startswith(t):
            return (0, -len(t))
        if t in c:
            return (2, c.find(t))
        return None
    bare = c.lstrip("/")
    if bare.startswith(t):
        return (1, -len(t))
    if t in c:
        return (2, c.find(t))
    return None


def match_score(needle: str, value: str) -> tuple[int, int] | None:
    """打分：返回 (score, first_index)，score 越小越靠前；不匹配返回 None。

    - 子串命中：最优先（位置越靠前越好）；
    - 全串子序列命中：次之（跨度越紧凑越好），因此 `q38` 能命中 `qwen3.8-max`。
    """
    n = (needle or "").strip().lower()
    if not n:
        return (0, 0)
    v = (value or "").lower()
    hit = v.find(n)
    if hit >= 0:
        return (hit, -len(n))

    ns, vs = _normalize_for_match(needle), _normalize_for_match(value)
    if not ns or not vs:
        return None
    pos, first, last = 0, -1, -1
    for ch in ns:
        found = vs.find(ch, pos)
        if found < 0:
            return None
        if first < 0:
            first = found
        last = found
        pos = found + 1
    span = (last - first + 1) - len(ns)      # 越紧凑分数越好
    return (1000 + span, first)


def matched_positions(needle: str, value: str) -> set[int]:
    """返回 value 中命中 needle 的字符下标（用于高亮）。"""
    n = (needle or "").strip().lower()
    if not n:
        return set()
    v = (value or "").lower()
    hit = v.find(n)
    if hit >= 0:
        return set(range(hit, hit + len(n)))
    # 子序列：在“去掉分隔符”的视图上匹配，再映射回原下标
    keep = [i for i, ch in enumerate(v) if re.match(r"[0-9a-z\u4e00-\u9fff]", ch)]
    compact = "".join(v[i] for i in keep)
    out: set[int] = set()
    pos = 0
    for ch in _normalize_for_match(needle):
        found = compact.find(ch, pos)
        if found < 0:
            return set()          # 没完全匹配上 → 不高亮任何字符
        out.add(keep[found])
        pos = found + 1
    return out


def highlight_text(needle: str, value: str, base_class: str, inside: str = "class:fuzzymatch.inside", outside: str = "class:fuzzymatch.outside"):
    """把 value 切成「命中/未命中」两种样式，返回 FormattedText。"""
    from prompt_toolkit.formatted_text import FormattedText

    hits = matched_positions(needle, value)
    if not hits:
        return FormattedText([(f"class:{base_class}" if not base_class.startswith("class:") else base_class, value)])
    base = f"class:{base_class}" if not base_class.startswith("class:") else base_class
    parts: list[tuple[str, str]] = []
    buf, buf_hit = "", None
    for index, char in enumerate(value):
        is_hit = index in hits
        if buf_hit is None:
            buf_hit = is_hit
        if is_hit != buf_hit:
            parts.append((inside if buf_hit else outside or base, buf))
            buf, buf_hit = char, is_hit
        else:
            buf += char
    if buf:
        parts.append((inside if buf_hit else outside or base, buf))
    return FormattedText(parts)


# --------------------------------------------------------------------------
# 通用「可滚动单选」：用 prompt_toolkit 的补全菜单实现
# --------------------------------------------------------------------------

ACTION_SELECT = "select"
ACTION_DEFAULT = "default"
ACTION_PROVIDER = "provider"
ACTION_CANCEL = "cancel"


class _DefaultRequested(Exception):
    def __init__(self, value: str) -> None:
        super().__init__(value)
        self.value = value


class _ProviderRequested(Exception):
    pass


class _Cancelled(Exception):
    pass


def _numbered_fallback(
    console,
    values: list[str],
    labels: dict[str, str],
    current: str,
    default: str,
    title: str,
    footer: str = "",
) -> tuple[str, str]:
    """非交互兜底：**一次性列出全部选项**（不分页、不重复打印），读取编号/关键词。"""
    from rich.table import Table
    from rich.text import Text

    from ..core.utils import clean_pasted

    matches = list(values)
    while True:
        table = Table(title=f"{title}（{len(matches)} 项）", show_header=False, box=None, pad_edge=False)
        table.add_column(justify="right", style="dim", width=5, no_wrap=True)
        table.add_column(overflow="ellipsis", no_wrap=True)
        for index, value in enumerate(matches, 1):
            marks = ("[cyan]●当前[/cyan] " if value == current else "") + (
                "[green]★默认[/green] " if value == default else ""
            )
            table.add_row(str(index), Text.from_markup(f"{marks}{labels.get(value, value)}"))
        console.print(table)
        console.print(
            "[dim]输入编号选择 · d<编号> 设为默认 · /关键词 过滤"
            + (" · c 切换供应商" if footer else "")
            + " · 回车取消[/dim]"
        )
        try:
            raw = clean_pasted(console.input("[bold]选择[/bold] › "))
        except (EOFError, KeyboardInterrupt):
            return ACTION_CANCEL, ""
        low = raw.lower()
        if low in {"c", "provider"} and footer:
            return ACTION_PROVIDER, ""
        if not raw:
            return ACTION_CANCEL, ""
        if raw.startswith("/"):
            needle = raw[1:].lower()
            filtered = [v for v in values if needle in v.lower()]
            if not filtered:
                console.print(f"[yellow]没有匹配 {raw[1:]} 的项[/yellow]")
                return ACTION_CANCEL, ""
            matches = filtered
            continue
        action = ACTION_SELECT
        if low.startswith("d"):
            raw = raw[1:].strip()
            action = ACTION_DEFAULT
        if not raw.isdigit():
            return ACTION_CANCEL, ""
        index = int(raw)
        if not 1 <= index <= len(matches):
            return ACTION_CANCEL, ""
        return action, matches[index - 1]


def pick_value(
    console,
    values: list[str],
    labels: dict[str, str] | None = None,
    current: str = "",
    default: str = "",
    title: str = "选择",
    footer: str = "",
    extra_hint: str = "",
    display_class: str = "pick.model",     # 列表项配色（pick.model / pick.provider / pick.embedding / pick.paper）
) -> tuple[str, str]:
    """可滚动单选：返回 (action, value)，action ∈ {select, default, provider, cancel}。

    复用 prompt_toolkit 补全菜单（与命令面板同一引擎）：输入即过滤、↑↓/PgUp/PgDn 滚动、
    鼠标滚轮可用、CJK 宽度由库处理。

    按键：`Enter` 选中（本次会话使用）· `Ctrl+C` 设为默认 · `Ctrl+P` 切换供应商
    （需 footer 提示）· `Esc` 取消。非交互环境退回「一次列全 + 编号输入」。
    """
    labels = labels or {}
    if not values:
        return ACTION_CANCEL, ""

    if not _available() or not sys.stdin.isatty():
        return _numbered_fallback(console, values, labels, current, default, title, footer)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import FormattedText

    class ValueCompleter(Completer):
        def get_completions(self, document, complete_event):
            typed = document.text_before_cursor
            if not typed.strip():
                # 空输入：保持原顺序，把「当前项」提到最前（回车/↓ 第一个就是它）
                ordered = ([current] if current in values else []) + [
                    v for v in values if v != current
                ]
            else:
                ranked = [
                    (score, value)
                    for value in values
                    if (score := match_score(typed, value)) is not None
                ]
                ranked.sort(key=lambda item: (item[0], item[1].lower()))
                ordered = [value for _score, value in ranked]

            for value in ordered:
                meta_parts: list[tuple[str, str]] = []
                if value == current:
                    meta_parts.append(("class:mark.current", "●当前 "))
                if value == default:
                    meta_parts.append(("class:mark.default", "★默认 "))
                label = labels.get(value, "")
                if label:
                    meta_parts.append(("class:pick.dim", label))
                yield Completion(
                    value,
                    start_position=-len(typed),
                    display=highlight_text(typed, value, display_class),
                    display_meta=FormattedText(meta_parts) if meta_parts else "",
                )

    kb = confirm_key_bindings()

    @kb.add("c-c")
    def _default(event):  # Ctrl+C = 设为默认（带上当前高亮项）
        buffer = event.current_buffer
        state = buffer.complete_state
        value = state.current_completion.text if (state and state.current_completion) else buffer.text
        event.app.exit(exception=_DefaultRequested(value.strip()))

    @kb.add("c-p")
    def _provider(event):
        event.app.exit(exception=_ProviderRequested())

    @kb.add("escape")
    def _cancel(event):
        event.app.exit(exception=_Cancelled())

    _cols, rows = _terminal_rows()

    def _open_menu() -> None:
        """打开时立刻弹出菜单：prompt_toolkit 在「空输入」时不会自动显示补全列表，
        否则用户看到的只有一行提示符（以为“什么都不显示”）。"""
        try:
            from prompt_toolkit.application import get_app

            buffer = get_app().current_buffer
            buffer.start_completion(select_first=False)
        except Exception:  # noqa: BLE001
            pass

    session = PromptSession(
        # 输入即筛选；打分器支持子串与全串子序列（`q38` → `qwen3.8-max`）
        completer=ValueCompleter(),
        complete_while_typing=True,
        complete_style="COLUMN",
        # 菜单尽量占满整屏，但要给「提示符 1 行 + 底部状态栏 1 行」留位置，
        # 否则 prompt_toolkit 会把状态栏挤掉（看不到操作提示）
        reserve_space_for_menu=max(4, rows - 4),
        key_bindings=kb,
        mouse_support=False,                        # 不劫持鼠标，保留终端复制/粘贴
        style=build_style(),                        # 深色底 + 蓝字（见 _THEMES）
        bottom_toolbar=lambda: _toolbar(current, default, footer, extra_hint),
    )
    # 标题与提示只打印一行（菜单浮层会盖住底部状态栏，提示符不再重复标题）
    hints = ["输入即过滤", "↑↓/PgUp/PgDn 滚动", "Enter 选中", "Ctrl+C 设为默认"]
    if footer:
        hints.insert(0, footer)
    hints.append("Esc 取消")
    try:
        console.print(f"[#4da3ff]{title}[/#4da3ff] [dim]· " + " · ".join(hints) + "[/dim]")
    except Exception:  # noqa: BLE001 - 老终端不支持 hex 颜色时退回
        console.print(f"{title} · " + " · ".join(hints))
    try:
        # pre_run：打开后立刻弹出菜单（空输入时 prompt_toolkit 不会自己显示补全列表）
        text = session.prompt("› ", pre_run=_open_menu)
        text = text.strip()
        if not text:
            return ACTION_CANCEL, ""
        if text in values:
            return ACTION_SELECT, text

        # 输入不是完整项时：按「子串 → 模糊打分」兜底解析（回车/直接输入也能确定）
        substring = [v for v in values if text.lower() in v.lower()]
        if len(substring) == 1:
            return ACTION_SELECT, substring[0]

        scored = sorted(
            ((match_score(text, v), v) for v in values),
            key=lambda pair: (pair[0] is None, pair[0] or (0, 0), pair[1].lower()),
        )
        scored = [(score, value) for score, value in scored if score is not None]
        if scored:
            best_score = scored[0][0]
            best = [value for score, value in scored if score == best_score]
            if len(best) == 1:
                return ACTION_SELECT, best[0]
        if len(substring) >= 1:
            return ACTION_SELECT, substring[0]
        return ACTION_CANCEL, ""
    except _DefaultRequested as exc:
        value = exc.value if exc.value in values else current
        return (ACTION_DEFAULT, value) if value else (ACTION_CANCEL, "")
    except _ProviderRequested:
        return ACTION_PROVIDER, ""
    except (_Cancelled, EOFError):
        return ACTION_CANCEL, ""
    except KeyboardInterrupt:
        return ACTION_CANCEL, ""


def confirm_key_bindings():  # noqa: ANN201
    """构造「Tab/Enter 都是确定」的按键绑定（供选择器与命令面板共用）。

    语义：
      - `Enter`：采纳高亮项并执行；
      - `Tab` ：**已是该项就直接执行**；否则先补全（便于继续输入参数，如 `/models --embedding`）；
                菜单未打开时按 Tab 先打开菜单。
    """
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    def _best_completion(buffer):
        """取「应被采纳」的候选：优先高亮项，否则第一条（模糊匹配的首选）。

        仅当输入里没有空格（即正在选一个值，而不是在编辑带参数的命令行）时才自动采纳，
        避免把用户敲好的 `--flag` 覆盖掉。
        """
        state = buffer.complete_state
        if not state or " " in buffer.text.strip():
            return None
        if state.current_completion:
            return state.current_completion
        return state.completions[0] if state.completions else None

    def _submit(buffer) -> None:
        completion = _best_completion(buffer)
        if completion is not None:
            buffer.apply_completion(completion)
        buffer.validate_and_handle()

    @kb.add("enter")
    def _enter(event):  # noqa: ANN001
        _submit(event.current_buffer)

    @kb.add("tab")
    def _tab(event):  # noqa: ANN001
        buffer = event.current_buffer
        typed = buffer.text.strip()
        state = buffer.complete_state
        # 1) 有候选且与当前输入不同 → 先补全（可继续补参数，如 `/models --embedding`）
        if state and " " not in typed:
            best = state.current_completion or (state.completions[0] if state.completions else None)
            if best is not None and typed != best.text.strip():
                buffer.apply_completion(best)
                return
        # 2) 输入已经是一个候选项（或已补全）→ Tab 即确定
        if typed:
            buffer.validate_and_handle()
            return
        # 3) 空输入 → 打开菜单并高亮第一项
        buffer.start_completion(select_first=True)

    return kb


def _terminal_rows() -> tuple[int, int]:
    import shutil

    try:
        size = os.get_terminal_size(sys.stdout.fileno())
        if size.lines > 5:
            return size.columns, size.lines
    except Exception:  # noqa: BLE001
        pass
    fallback = shutil.get_terminal_size(fallback=(80, 24))
    return fallback.columns, fallback.lines


def _toolbar(current: str, default: str, footer: str, extra_hint: str = "") -> str:
    """底部状态栏：不再重复标题（标题已在选择器上方打印一次）。"""
    parts: list[str] = []
    if current:
        parts.append(f"● 当前={current}")
    if default:
        parts.append(f"★ 默认={default}")
    parts.append("输入即过滤 · ↑↓/PgUp/PgDn 滚动 · Enter 选中 · Ctrl+C 设为默认")
    if footer:
        parts.append(footer)
    if extra_hint:
        parts.append(extra_hint)
    return " · ".join(parts) + " "


def read_line(session, prompt: str = "paper-agent › ") -> str:
    """读一行输入；优先用 prompt_toolkit 会话，否则由调用方兜底。"""
    if session is None:
        raise RuntimeError("no prompt session")
    return session.prompt(prompt)
