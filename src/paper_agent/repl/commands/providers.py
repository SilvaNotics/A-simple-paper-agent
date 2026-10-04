# -*- coding: utf-8 -*-
"""供应商与模型相关命令（`Repl` 的 mixin）。

- `/connect` `/providers` `/keys` `/models` `/model` `/embed` `/offline`；
- 用户配置写盘后统一用 `self._rebuild()` 重建会话。
"""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from ...core import ui
from ...cli import cli_settings
from ...core.config import get_settings
from ...pipeline.session import build_session
from ..tui import ask_line, connect_flow, embed_models_for, pick_from_list, print_presets
from ...sources.userconfig import UserConfig, probe_embedding_dim, resolve_embedding, settings_overrides
from ...core.utils import clean_pasted, flag_bool, mask_secret, split_args

from ..base import ReplBase


class ProviderCommands(ReplBase):
    """供应商与模型相关命令（`Repl` 的 mixin）。"""

    def _rebuild(self, settings, note: str = "") -> bool:
        """重建会话（换供应商/模型/embedding 后调用）；失败时保留原会话。"""
        try:
            session = build_session(settings)
        except RuntimeError as exc:
            ui.console.print(f"[red]会话刷新失败：{exc}[/red]")
            ui.console.print("[dim]配置已保存；修好后可用 /providers use <name> 或 /connect 重试[/dim]")
            return False
        self.settings = settings
        self.session = session
        model = getattr(self.session.model, "model_name", None) or getattr(self.session.model, "model", None)
        ui.console.print(
            f"[green]✓[/green] {note or '会话已刷新'}  "
            f"供应商=[cyan]{settings.provider_label}[/cyan] 模型=[cyan]{model or '-'}[/cyan] "
            f"[dim]({settings.active_base_url})[/dim]"
        )
        ui.console.print(
            f"[dim]索引 {len(self.session.index.paper_ids())} 篇 / {self.session.index.chunk_count} chunks"
            f"（embedding={settings.active_embedding_model or '无'}"
            f"{' @ ' + settings.active_embedding_label if settings.embed_base_url else ''}）[/dim]"
        )
        return True

    def cmd_connect(self, args: str) -> None:
        positional, flags = split_args(args)
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
            ui.console,
            self.config,
            base_url=base_url,
            api_key=api_key,
            name=flags.get("name", ""),
            kind=flags.get("kind", ""),
            preset=preset,
            do_fetch=not flag_bool(flags, "no-fetch"),
            allow_empty_key=flag_bool(flags, "allow-empty-key"),
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
            ui.console.print("[yellow]还没有配置供应商，先 /connect[/yellow]")
            return ""
        if len(names) == 1:
            return names[0]
        action, name = pick_from_list(
            ui.console,
            names,
            current=self.config.default_provider,
            default=self.config.default_provider,
            title=title,
            display=self._provider_display(),
            footer="Enter 切换",
            display_class="pick.provider",
        )
        if action == "cancel" or not name:
            ui.console.print("[dim]已取消[/dim]")
            return ""
        return name

    def _switch_provider(self, name: str) -> None:
        """把默认供应商切到 name 并重建会话。"""
        provider = self.config.providers.get(name)
        if provider is None:
            ui.console.print(f"[red]没有供应商 {name}[/red]")
            return
        self.config.set_default(name, provider.chat_model)
        self.config.save()
        base = get_settings(refresh=True).model_copy(update=settings_overrides(self.config, provider))
        self._rebuild(base, f"供应商已切到 {name}")

    def cmd_models(self, args: str) -> None:
        _, flags = split_args(args)
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
            ui.console.print("[yellow]还没有配置供应商，先执行 [cyan]/connect[/cyan][/yellow]")
            return

        want_embedding = flag_bool(flags, "embedding")
        if flag_bool(flags, "refresh") or (not provider.models and not provider.chat_models):
            from src.paper_agent.repl.tui import _sync_models

            _sync_models(ui.console, self.config, provider, with_embedding_probe=want_embedding)
            self.config.save()

        while True:  # Ctrl+P 可在列表里直接换供应商，换完继续选模型
            if want_embedding:
                items = embed_models_for(provider)
                title = f"{provider.name} 的 embedding 模型（RAG 建索引用）"
                if not items:
                    ui.console.print(
                        "[yellow]该供应商的 /models 里没有 embedding 模型。[/yellow]"
                        "可以让另一个供应商（如 DashScope）负责 embedding，或手动输入模型名。"
                    )
                    manual = ask_line(ui.console, "embedding 模型名（留空取消）› ")
                    if not manual:
                        return
                    items = [manual]
                current, default = provider.embedding_model, provider.embedding_model
            else:
                items = provider.models or provider.chat_models
                if flag_bool(flags, "all"):
                    items = provider.models
                elif provider.chat_models:
                    items = provider.chat_models
                if not items:
                    ui.console.print(
                        "[yellow]没有拿到模型列表：可用 /models --refresh 重试，或 /model <名字> 直接指定[/yellow]"
                    )
                    return
                current, default = provider.chat_model, self.config.default_model
                title = f"{provider.name} 的对话模型"

            action, value = pick_from_list(
                ui.console,
                items,
                current=current,
                default=default,
                title=title,
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
            ui.console.print("[dim]已取消[/dim]")
            return

        if want_embedding:
            provider.embedding_model = value
            try:
                with ui.console.status(f"[cyan]探测 {value} 的向量维度…[/cyan]"):
                    dim = self._run_async(probe_embedding_dim(provider.base_url, provider.api_key, value))
                provider.embedding_dim = dim or provider.embedding_dim
                ui.console.print(f"[green]✓[/green] embedding 已切换为 {value}" + (f"（dim={dim}）" if dim else ""))
            except Exception as exc:  # noqa: BLE001
                ui.console.print(f"[yellow]维度探测失败（{type(exc).__name__}）：{str(exc)[:100]}[/yellow]")
        else:
            provider.chat_model = value
            ui.console.print(f"[green]✓[/green] 对话模型：{value}")

        if action == "default":
            self.config.set_default(provider.name, value if not want_embedding else provider.chat_model)
            ui.console.print(f"[green]★[/green] 已写入默认（{self.config.path}）")
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
            print_presets(ui.console, self.config)
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
                    "src.paper_agent.repl.tui", fromlist=["PRESETS"]
                ).PRESETS:
                    if kind == name:
                        hint = f"（可用 [cyan]/connect {key}[/cyan] 添加 {label}）"
                        break
                ui.console.print(f"[red]没有供应商 {name}[/red]{hint}")
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
                    ui.console.print("[yellow]还没有配置供应商[/yellow]")
                    return
                if len(names) == 1:
                    target = names[0]
                else:
                    _act, picked = pick_from_list(
                        ui.console,
                        names,
                        current=self.config.default_provider,
                        default=self.config.default_provider,
                        title="选择要删除的供应商",
                        display=self._provider_display(),
                        footer="Enter 选中共确认删除 / Esc 取消",
                        display_class="pick.provider",
                    )
                    if _act == "cancel" or not picked:
                        ui.console.print("[dim]已取消[/dim]")
                        return
                    target = picked
            if target not in self.config.providers:
                ui.console.print(f"[red]没有供应商 {target}[/red]")
                return
            mark = "（当前默认供应商）" if target == self.config.default_provider else ""
            answer = ask_line(
                ui.console, f"确认删除供应商 [cyan]{target}[/cyan]{mark}（含其 API key）？[y/N] › "
            ).strip().lower()
            if answer not in {"y", "yes"}:
                ui.console.print("[dim]已取消[/dim]")
                return
            self.config.remove_provider(target)
            self.config.save()
            ui.console.print(f"[green]✓[/green] 已删除供应商 [cyan]{target}[/cyan]")
            self._rebuild(get_settings(refresh=True), f"已移除 {target}")
        elif action == "sync":
            provider = self.config.providers.get(name) or self.config.active_provider()
            if provider is None:
                ui.console.print("[red]没有可同步的供应商[/red]")
                return
            from src.paper_agent.repl.tui import _sync_models

            _sync_models(ui.console, self.config, provider)
            self.config.save()
        elif action in {"key-rm", "rm-key", "clear-key", "key-off"}:
            # 只删 key，保留供应商与已选模型（原 key 不再可用）
            if self.config.remove_provider_key(name):
                self.config.save()
                ui.console.print(f"[green]✓[/green] 已删除供应商 [cyan]{name}[/cyan] 的 API key（供应商保留）")
                self._rebuild(get_settings(refresh=True), f"已删除 {name} 的 key")
            else:
                ui.console.print(f"[red]没有供应商 {name}[/red]")
        else:
            ui.console.print(
                "[yellow]用法：/providers | /providers use [name] | /providers rm [name]（不给名字弹选择器） | "
                "/providers key-rm <name>（只删 key） | /providers sync [name][/yellow]"
            )

    def cmd_keys(self, args: str = "") -> None:
        """查看并删除模型供应商与搜索渠道的 API key。"""
        action, _, rest = args.strip().partition(" ")
        action = action.lower()
        rest = rest.strip()
        if action in {"rm", "remove", "del", "delete"}:
            self._remove_key(rest)
            return
        if action:
            ui.console.print(
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
        ui.console.print(table)
        ui.console.print("[dim]删除：/keys rm provider:<name> 或 /keys rm channel:<name>（也支持 /providers key-rm、/channels key-rm）[/dim]")

    def _remove_key(self, target: str) -> None:
        target = clean_pasted(target).strip()
        if not target:
            ui.console.print("[yellow]请给出要删除的 name：/keys rm provider:<name> | channel:<name>[/yellow]")
            return
        kind, _, name = target.partition(":")
        if not name:
            name, kind = kind, ""
        if kind == "provider" or (not kind and name in self.config.providers):
            if self.config.remove_provider_key(name):
                self.config.save()
                ui.console.print(f"[green]✓[/green] 已删除供应商 [cyan]{name}[/cyan] 的 API key（供应商保留）")
                self._rebuild(get_settings(refresh=True), f"已删除 {name} 的 key")
                return
        if kind == "channel" or (not kind and name in self.config.channels):
            if self.config.remove_channel_key(name):
                self.config.save()
                ui.console.print(f"[green]✓[/green] 已删除渠道 [cyan]{name}[/cyan] 的 API key（渠道保留）")
                self._refresh_settings("已删除渠道 key")
                return
        ui.console.print(f"[red]未找到 {target}（用 /keys 查看）[/red]")

    def cmd_model(self, args: str) -> None:
        """`/model` 查看；`/model <name>` 本次会话使用；`/model <name> --default` 写入默认。"""
        name, flags = split_args(args)
        make_default = flag_bool(flags, "default")
        session = self.require_session()
        if session is None:
            return

        if not name:
            model = session.model
            current = getattr(model, "model_name", None) or getattr(model, "model", None)
            provider = self.config.active_provider()
            ui.console.print(
                f"当前模型：[cyan]{current or '（离线假模型）'}[/cyan]  "
                f"[dim]供应商={self.settings.provider_label} 默认={self.config.default_model or '-'}"
                f" 候选={len(provider.chat_models) if provider is not None else 0}[/dim]"
            )
            ui.console.print("[dim]用法：/model <name> [--default]；或 /models 打开选择器[/dim]")
            return

        if session.offline:
            ui.console.print("[yellow]当前离线模式（假模型），/offline off 后再切模型[/yellow]")
            return

        base = self.settings.model_copy(update={"llm_model": name})
        self._rebuild(base, f"模型已切换为 {name}")

        provider = self.config.active_provider()
        if provider is None:
            return
        provider.chat_model = name
        if make_default:
            self.config.set_default(provider.name, name)
            ui.console.print(f"[green]★[/green] 已写入默认模型（{self.config.path}）")
        self.config.save()

    def cmd_embed(self, args: str = "") -> None:
        """`/embed` 查看；`/embed [供应商] [模型]` 指定；`/embed auto` 改回自动。"""
        positional, flags = split_args(args)
        tokens = positional.split()
        action = tokens[0].lower() if tokens else ""

        if not self.config.providers:
            ui.console.print(
                "[yellow]还没有通过 /connect 配置供应商。[/yellow]"
                "[dim]用 .env 时直接设置 EMBED_MODEL / EMBED_BASE_URL / EMBED_API_KEY 即可分开指定 embedding。[/dim]"
            )
            return
        if not action and not flags:
            self._show_embedding()
            ui.console.print(
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
            ui.console.print(f"[red]没有供应商 {name}[/red]（用 /providers 查看）")
            return
        provider = self.config.providers.get(name) if name else None
        if provider is None:  # 未指定：作用于当前生效的 embedding 供应商
            provider, _ = resolve_embedding(self.config, self.config.active_provider())
        if provider is None:
            ui.console.print("[yellow]没有可用的 embedding 供应商[/yellow]")
            return

        model = tokens[1] if len(tokens) > 1 else flags.get("model", "")
        if not model:
            model = self._pick_embedding_model(provider)
        if not model:
            ui.console.print("[dim]已取消[/dim]")
            return

        self.config.set_embedding_provider(provider.name)
        provider.embedding_model = model
        try:
            with ui.console.status(f"[cyan]探测 {model} 的向量维度…[/cyan]"):
                dim = self._run_async(
                    probe_embedding_dim(provider.base_url, provider.api_key, model)
                )
            if dim:
                provider.embedding_dim = dim
        except Exception as exc:  # noqa: BLE001
            ui.console.print(f"[yellow]维度探测失败（{type(exc).__name__}）：{str(exc)[:100]}[/yellow]")
        self.config.save()
        self._rebuild(get_settings(refresh=True), f"embedding 已指定为 {provider.name} / {model}")

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
        table.add_row("索引目录", str(self.settings.data_path))
        ui.console.print(Panel(table, title="embedding 设置（RAG 建索引用）", border_style="cyan", padding=(0, 1)))

    def _pick_embedding_model(self, provider) -> str:
        items = embed_models_for(provider)
        if not items:
            return ask_line(ui.console, f"{provider.name} 的 embedding 模型名（留空取消）› ").strip()
        action, value = pick_from_list(
            ui.console,
            items,
            current=provider.embedding_model,
            default=provider.embedding_model,
            title=f"{provider.name} 的 embedding 模型",
            footer="Enter 确定",
            display_class="pick.embedding",
        )
        return "" if action == "cancel" else value

    def cmd_offline(self, args: str) -> None:
        current = self.settings.fake_llm
        token = args.strip().lower()
        if not token or token == "toggle":
            target = not current
        else:
            target = token in {"on", "true", "1", "yes"}
        if target == current:
            ui.console.print(f"[dim]离线模式已是 {'开' if current else '关'}[/dim]")
            return
        self.settings = cli_settings(offline=target)
        self._rebuild(self.settings, "离线模式已切换")
        ui.console.print(
            f"[green]✓[/green] 离线模式：{'开（假模型 + data/offline 索引）' if target else '关'}"
        )
