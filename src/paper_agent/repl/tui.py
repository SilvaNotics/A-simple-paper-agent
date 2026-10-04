# -*- coding: utf-8 -*-
"""`/connect` 相关交互与统一的选择器入口。

**选择器已完全重写**：不再手写原始终端渲染（在复杂终端下会错位/重复打印），
统一委托给 `repl/input.py` 的 `pick_value()` —— 它用 prompt_toolkit 的补全菜单实现：
输入即过滤、↑↓/PgUp/PgDn 滚动、CJK 宽度由库正确处理、鼠标滚轮可用，
与命令面板（输入 `/` 弹命令列表）共用同一套引擎。

本模块只保留：
- 供应商预设/识别结果的展示（`print_presets`）
- 支持粘贴的密钥读取（`read_secret` / `_ask_api_key`）
- `/connect` 问答流程与模型列表同步（`connect_flow` / `_sync_models`）
"""

from __future__ import annotations

import asyncio
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.utils import clean_pasted, clean_secret, mask_secret
from ..sources.userconfig import (
    PROVIDER_HINTS,
    Provider,
    UserConfig,
    detect_provider,
    fetch_models,
    normalize_base_url,
    probe_embedding_dim,
    provider_name,
)

logger = logging.getLogger(__name__)

PRESETS: list[tuple[str, str, str]] = [
    ("1", "dashscope", "阿里云百炼 / DashScope（Qwen，含 text-embedding-v4）"),
    ("2", "deepseek", "DeepSeek"),
    ("3", "openai", "OpenAI"),
    ("4", "moonshot", "Moonshot / Kimi"),
    ("5", "siliconflow", "SiliconFlow（含 BAAI/bge-m3 embedding）"),
    ("6", "zhipu", "智谱 GLM"),
    ("7", "local", "本地服务（vLLM / Ollama / LM Studio）"),
]

__all__ = [
    "PRESETS",
    "ask_line",
    "connect_flow",
    "embed_models_for",
    "pick_from_list",
    "preset_base_url",
    "print_presets",
    "read_secret",
]


# --------------------------------------------------------------------------
# 安全的行输入（EOF 视为取消）
# --------------------------------------------------------------------------


def ask_line(console: Console, prompt: str) -> str:
    """`console.input` 的安全包装：标准输入结束（EOF）时返回空串 = 取消。

    交互命令里散布着多个追问（预设/base URL、y-N 确认、手动模型名…）。当 stdin 结束
    —— Ctrl+D、或 stdin 被管道/子进程提前关闭 —— `console.input` 会抛 `EOFError`；
    它在各问答步骤之间冒泡，只会让 REPL 打出一大段 traceback（用户什么都没做错）。
    这里统一把它降级成空串，因为所有调用点本就把「空输入」当作取消。

    返回值同样经 `clean_pasted` 清洗（终端粘贴会带括号粘贴标记）。
    """
    try:
        return clean_pasted(console.input(prompt))
    except EOFError:
        return ""


# --------------------------------------------------------------------------
# 统一选择器入口（委托给 prompt_toolkit 补全菜单）
# --------------------------------------------------------------------------


def pick_from_list(
    console: Console,
    items: list[str],
    current: str = "",
    default: str = "",
    title: str = "选择",
    display: dict[str, str] | None = None,   # 值 → 展示标签（例如供应商行的说明）
    footer: str = "",
    display_class: str = "pick.model",       # 列表项配色类
) -> tuple[str, str]:
    """可滚动单选；返回 (action, value)，action ∈ {"select", "default", "provider", "cancel"}。

    交互（TTY）：prompt_toolkit 补全菜单 —— 输入即过滤、↑↓/PgUp/PgDn 滚动、
    `Enter` 选中、**`Ctrl+C` 设为默认**、`Ctrl+P` 切供应商（footer 非空时）、`Esc` 取消。
    非交互：一次性列出全部选项 + 编号输入（`d<编号>` 设默认、`/关键词` 过滤）。
    初始高亮位置由 `current` 决定。
    """
    from .input import pick_value

    return pick_value(
        console,
        list(items),
        labels=display or {},
        current=current,
        default=default,
        title=title,
        footer=footer,
        display_class=display_class,
    )


# --------------------------------------------------------------------------
# 供应商预设与配置展示
# --------------------------------------------------------------------------


def print_presets(console: Console, config: UserConfig) -> None:
    """打印供应商预设表 + 已配置供应商表。"""
    table = Table(title="供应商预设（统一 OpenAI 兼容格式）", show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("供应商")
    table.add_column("base URL", style="dim")
    for key, kind, label in PRESETS:
        base = PROVIDER_HINTS.get(kind, {}).get("base_url", "（自定义）")
        table.add_row(key, label, base)
    table.add_row("0", "手动输入 base URL", "https://your-endpoint/v1")
    console.print(table)

    if config.providers:
        existing = Table(title="已配置供应商", show_header=True, header_style="bold")
        existing.add_column("名称", style="cyan")
        existing.add_column("类型")
        existing.add_column("base URL", style="dim")
        existing.add_column("key")
        existing.add_column("对话模型")
        existing.add_column("embedding")
        for name, prov in config.providers.items():
            marker = " ★" if name == config.default_provider else ""
            existing.add_row(
                f"{name}{marker}",
                prov.kind,
                prov.base_url,
                mask_secret(prov.api_key),
                prov.chat_model or "-",
                prov.embedding_model or "-",
            )
        console.print(existing)
        console.print("[dim]切换供应商：/providers（选择器）或 /models --provider[/dim]")


# --------------------------------------------------------------------------
# 密钥读取（支持粘贴；不回显）
# --------------------------------------------------------------------------


def read_secret(console: Console, prompt: str = "api key  › ") -> str:
    """读取密钥：**支持粘贴**、不回显。

    不用 `getpass.getpass`：它关回显却不走 readline，终端为粘贴插入的括号粘贴标记
    （`\\x1b[200~…\\x1b[201~`）会被当成内容读入，导致 key 混入转义序列而认证失败。
    这里只关闭 ECHO、保留行缓冲（ICANON），读完再清洗。
    """
    console.print(f"[bold]{prompt}[/bold][dim]（支持粘贴；输入不回显）[/dim]", end="")
    raw = ""
    fd = None
    old_attrs = None
    try:
        fd = sys.stdin.fileno()
        if not sys.stdin.isatty():
            raise OSError("stdin 不是终端")
        import termios

        old_attrs = termios.tcgetattr(fd)
        new_attrs = termios.tcgetattr(fd)
        new_attrs[3] = new_attrs[3] & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, new_attrs)
        raw = sys.stdin.readline()
    except Exception:  # noqa: BLE001
        try:
            raw = input()
        except EOFError:
            raw = ""
    finally:
        if fd is not None and old_attrs is not None:
            try:
                import termios

                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:  # noqa: BLE001
                pass
    console.print()

    key = clean_secret(raw)
    if key != clean_pasted(raw):
        console.print("[dim]（已自动移除 key 中的空白字符）[/dim]")
    return key


def _ask_base_url(console: Console, preset: str, current: str = "") -> str:
    """询问 base URL。

    没有默认值就不给默认（否则「什么都不到直接回车」会静默连到 localhost）：
    只有重新连接已有供应商时才有默认（current），否则留空 = 取消。
    """
    if preset and preset in {k for k, _, _ in PRESETS}:
        kind = next(kind for key, kind, _ in PRESETS if key == preset)
        return PROVIDER_HINTS.get(kind, {}).get("base_url", "")

    hint = f"[dim]（回车沿用当前值 {current}）[/dim]" if current else ""
    console.print(
        f"[bold]请输入 base URL[/bold]（OpenAI 兼容端点，如 `https://api.deepseek.com`）{hint}"
    )
    console.print("[dim]（支持粘贴：Ctrl+Shift+V / 右键粘贴，然后回车；留空=取消）[/dim]")
    raw = ask_line(console, "base url › ")
    return raw or current


def _ask_api_key(console: Console, allow_empty: bool = False) -> str:
    key = read_secret(console)
    if key:
        console.print(f"[dim]已读取 key：[/dim][cyan]{mask_secret(key)}[/cyan]")
    elif not allow_empty:
        console.print("[yellow]未输入 key（本地服务通常不需要；如需空 key 请用 --allow-empty-key）[/yellow]")
    return key


def preset_base_url(key: str) -> str:
    """预设编号 → base URL（非预设编号返回空串）。"""
    for preset_key, kind, _label in PRESETS:
        if preset_key == str(key).strip():
            return PROVIDER_HINTS.get(kind, {}).get("base_url", "")
    return ""


# --------------------------------------------------------------------------
# /connect 流程
# --------------------------------------------------------------------------


def connect_flow(
    console: Console,
    config: UserConfig,
    base_url: str = "",
    api_key: str = "",
    name: str = "",
    kind: str = "",
    preset: str = "",
    do_fetch: bool = True,
    allow_empty_key: bool = False,
) -> Provider | None:
    """执行一次 `/connect`：收集 base URL + key → 识别供应商 → 拉模型 → 落盘。"""
    preset = preset or (base_url if preset_base_url(base_url) else "")
    if preset:
        base_url = preset_base_url(preset) or base_url

    if not base_url:
        print_presets(console, config)
        console.print(
            "[dim]输入预设编号（1-7）、0 手动输入 base URL，或直接粘贴 base URL；回车取消[/dim]"
        )
        answer = ask_line(console, "[bold]预设 / base URL[/bold] › ") or preset
        if not answer:
            console.print("[yellow]未输入任何内容，已取消（不会连接任何端点）[/yellow]")
            return None
        if answer.isdigit():
            base_url = preset_base_url(answer)          # 1-7；0/其它数字 → 手动输入
            if not base_url:
                base_url = _ask_base_url(console, "")
        elif "://" in answer or "." in answer:
            base_url = answer                           # 直接粘贴的 URL
        else:
            base_url = _ask_base_url(console, "")
        if not base_url:
            console.print("[yellow]未提供 base URL，已取消[/yellow]")
            return None

    base_url = normalize_base_url(base_url)
    if not base_url:
        console.print("[red]base URL 不能为空[/red]")
        return None
    api_key = clean_secret(api_key)

    detected_kind, detected_label = detect_provider(base_url)
    resolved_kind = kind or detected_kind
    console.print(f"识别为：[bold]{detected_label}[/bold] [dim]({resolved_kind})[/dim]")

    if not api_key:
        api_key = _ask_api_key(console, allow_empty=allow_empty_key)
    if api_key:
        console.print(f"[dim]将使用 key：[/dim][cyan]{mask_secret(api_key)}[/cyan]  [dim]→ {base_url}[/dim]")
    elif not allow_empty_key and resolved_kind != "local":
        console.print("[yellow]没有 key 也可以保存，但调用会失败。要继续吗？[/yellow]")
        if ask_line(console, "继续? [y/N] › ").lower() not in {"y", "yes"}:
            return None

    resolved_name = name or provider_name(base_url, resolved_kind)
    provider = config.upsert_provider(
        base_url=base_url,
        api_key=api_key,
        name=resolved_name,
        kind=resolved_kind,
        label=detected_label,
    )
    console.print(f"已保存供应商：[cyan]{provider.name}[/cyan] → {config.path}")

    if do_fetch:
        _sync_models(console, config, provider, with_embedding_probe=True)

    if not provider.chat_model:
        console.print(
            "[yellow]该供应商还没有可用的对话模型[/yellow]"
            "（/models 未返回列表或未匹配到候选）。"
        )
        manual = ask_line(console, "对话模型名（留空则稍后用 /models 选择）› ")
        if manual:
            provider.chat_model = manual
            console.print(f"[green]✓[/green] 对话模型：{manual}")

    config.set_default(provider.name, provider.chat_model)
    config.save()
    _print_provider_summary(console, provider, config)
    return provider


def _sync_models(
    console: Console,
    config: UserConfig,
    provider: Provider,
    with_embedding_probe: bool = False,
) -> None:
    """拉取 /models、分类、推断默认，并按需探测 embedding 维度。"""
    try:
        with console.status("[cyan]读取模型列表 /models …[/cyan]"):
            models = asyncio.run(fetch_models(provider.base_url, provider.api_key))
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        console.print(f"[yellow]无法读取 /models（{type(exc).__name__}: {text[:140]}）[/yellow]")
        console.print(
            f"[dim]请求：{provider.base_url}/models  鉴权：{mask_secret(provider.api_key)}[/dim]"
        )
        if any(code in text for code in ("401", "403", "Unauthorized", "invalid_api_key")):
            console.print(
                "[yellow]像是密钥无效[/yellow]：确认粘贴时没有多余空格/引号（本工具会自动清理），"
                "key 是否过期或属于别的区域/账号；重新执行 [cyan]/connect[/cyan] 可覆盖更新。"
            )
        else:
            console.print("[dim]本地服务未实现 /models 属正常现象，可手动 /model <模型名> 指定。[/dim]")
        return

    config.apply_sync(provider, models)
    console.print(
        f"发现 [bold]{len(provider.models)}[/bold] 个模型 → 对话候选 [cyan]{len(provider.chat_models)}[/cyan] / "
        f"embedding [cyan]{len(provider.embedding_models)}[/cyan]"
    )

    if with_embedding_probe and provider.embedding_model:
        try:
            with console.status("[cyan]探测 embedding 维度…[/cyan]"):
                dim = asyncio.run(
                    probe_embedding_dim(provider.base_url, provider.api_key, provider.embedding_model)
                )
            if dim:
                provider.embedding_dim = dim
                console.print(f"embedding `{provider.embedding_model}` 维度：{dim}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]embedding 探测失败：{type(exc).__name__}: {str(exc)[:100]}[/yellow]")
            console.print("[dim]RAG 建索引时仍会真实报错，可先用 /models 指定别的 embedding 模型。[/dim]")


def _print_provider_summary(console: Console, provider: Provider, config: UserConfig) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("供应商", f"{provider.name} ({provider.kind})")
    table.add_row("base URL", provider.base_url)
    table.add_row("对话模型", provider.chat_model or "-")
    table.add_row(
        "embedding",
        f"{provider.embedding_model or '（无，RAG 不可用）'}"
        + (f" dim={provider.embedding_dim}" if provider.embedding_dim else ""),
    )
    table.add_row("默认", f"{config.default_provider} / {config.default_model or '-'}")
    console.print(Panel(table, title="配置已生效", border_style="green", padding=(0, 1)))


def embed_models_for(provider: Provider) -> list[str]:
    """给 UI 用的 embedding 候选（含当前值）。"""
    items = list(provider.embedding_models)
    if provider.embedding_model and provider.embedding_model not in items:
        items.insert(0, provider.embedding_model)
    return items
