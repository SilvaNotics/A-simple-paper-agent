# -*- coding: utf-8 -*-
"""检索 / 入库 / 问答 / 报告 / 渠道相关命令（`Repl` 的 mixin）。

- `/search` `/ingest` `/ask` `/report` `/mcp`：走 pipeline 与 MCP 工具；
- `/channels`（含 `add`）：搜索渠道配置。
依赖 `self.session` / `self.settings` / `self.config`，输出走 `ui.console`。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table

from ...core import ui
from ...sources.channels import channel_label, is_domestic, presets as channel_presets, spec_for
from ...cli import papers_table
from ...core.config import get_settings
from ...pipeline.session import ask as pipeline_ask, ingest_papers, run_ingest, run_report, run_search
from ..ui import SearchProgressView, nullcontext
from ..tui import ask_line, read_secret
from ...sources.userconfig import UserConfig
from ...core.schema import INGEST_STATUS_STYLES
from ...core.utils import clean_pasted, flag_bool, split_args

from ..base import ReplBase


class SearchCommands(ReplBase):
    """检索 / 入库 / 问答 / 报告 / 渠道相关命令（`Repl` 的 mixin）。"""

    def cmd_search(self, args: str) -> None:
        query, flags = split_args(args)
        if not query:
            ui.console.print(
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
        if flag_bool(flags, "no-llm"):
            use_llm = False
        elif flag_bool(flags, "llm"):
            use_llm = True
        label = {"builtin": "内置源+渠道", "mcp": "MCP", "all": "MCP + 内置源/渠道", "": "MCP → 内置回退"}.get(source, source)
        llm_note = "" if use_llm is False else "（LLM 扩展/重排）"
        progress = SearchProgressView(f"{label}{llm_note}", query)
        by_channel: dict[str, list] = {}
        # 逐渠道进度：builtin_search 每当某个渠道 queued/running/done/failed/skipped 就回调 → 刷新表格
        with Live(progress.render(), console=ui.console, refresh_per_second=10, transient=False) as live:
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
            ui.console.print(f"[yellow]没有检索到结果（route={route}）[/yellow]")
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
                    ui.console.print(f"[dim]{title}：无结果[/dim]")
                    continue
                ui.console.print(papers_table(items, title=title))
        else:
            ui.console.print(papers_table(papers))
        ui.console.print(f"[dim]来源：{route}[/dim]")

        # 特定参数 --ingest [N]：直接把检索结果入库，无需再跑一次 /ingest
        ingest_flag = flags.get("ingest", "") or flags.get("save", "") or flags.get("index", "")
        if not ingest_flag:
            ui.console.print("[dim]提示：加 --ingest 可直接把这些结果入库（/search <query> --ingest）[/dim]")
            return
        self._ingest_found(papers, ingest_flag, force=flag_bool(flags, "force"))

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
                ui.console.print(
                    f"[yellow]「{label}」返回 429（请求过多）。[/yellow]"
                    f"[dim]可 /channels add {kind} --email you@example.com 进 polite pool，或稍后重试。[/dim]"
                )
            elif status == "failed" and detail in {"401", "403", "需key"}:
                self._offer_channel_setup(kind, label, detail)
            elif status == "skipped" and "需key" in detail and kind in requested_kinds:
                self._offer_channel_setup(kind, label, "需key")

    def _offer_channel_setup(self, kind: str, label: str, reason: str) -> None:
        """渠道要求登录/key 时，直接在 REPL 里引导配置。"""
        ui.console.print(f"[yellow]「{label}」需要登录 / API key（{reason}）。[/yellow]")
        if not sys.stdin.isatty():
            ui.console.print(f"[dim]稍后可执行：/channels add {kind}[/dim]")
            return
        answer = ask_line(ui.console, f"现在配置 {label}？[y/N] › ").lower()
        if answer in {"y", "yes"}:
            self.add_channel(kind)
        else:
            ui.console.print(f"[dim]稍后可执行：/channels add {kind}[/dim]")

    def _ingest_found(self, papers: list, ingest_flag: str, force: bool = False) -> None:
        """把刚检索到的 `Paper` 直接入库（`/search ... --ingest [N]`）。"""
        session = self.require_session()
        if session is None:
            ui.console.print("[yellow]需要配置 embedding（/connect）后才能入库[/yellow]")
            return
        count = int(ingest_flag) if str(ingest_flag).isdigit() else len(papers)
        selected = papers[: max(1, count)]
        with ui.console.status(f"[cyan]直接入库 {len(selected)} 篇（下载 → 解析 → 向量化）…[/cyan]"):
            results = self._run_async(ingest_papers(selected, session=session, force=force))
        if not results:
            ui.console.print("[yellow]没有可入库的论文[/yellow]")
            return
        for info in results:
            style = INGEST_STATUS_STYLES.get(info["status"], "yellow")
            ui.console.print(
                f"[{style}]{info['status']:>11}[/{style}] {info['paper_id']}  {info['chunks']} chunks  {info['message']}"
            )
        self.show_index()

    def cmd_ingest(self, args: str) -> None:
        query, flags = split_args(args)
        if not query and not flags.get("ids"):
            ui.console.print(
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
        with ui.console.status("[cyan]抓取 → 下载 → 解析 → 向量化…[/cyan]"):
            results = self._run_async(
                run_ingest(
                    query or "",
                    session=session,
                    limit=limit,
                    ids=flags.get("ids", ""),
                    force=flag_bool(flags, "force"),
                )
            )
        if not results:
            ui.console.print("[yellow]没有可入库的论文[/yellow]")
            return
        for info in results:
            style = INGEST_STATUS_STYLES.get(info["status"], "yellow")
            ui.console.print(
                f"[{style}]{info['status']:>11}[/{style}] {info['paper_id']}  {info['chunks']} chunks  {info['message']}"
            )
        self.show_index()

    def cmd_ask(self, question: str) -> None:
        question = question.strip()
        if not question:
            ui.console.print("[yellow]用法：/ask <question>[/yellow]")
            return
        session = self.require_session()
        if session is None:
            return
        if session.index.chunk_count == 0:
            ui.console.print(
                "[yellow]索引为空。[/yellow]先执行 [cyan]/ingest <主题>[/cyan] 抓几篇论文，"
                "或 [cyan]/search <主题>[/cyan] 先看看检索结果。"
            )
            return

        on_token = self._stream_renderer() if self.stream and not session.offline else None
        deadline = max(1.0, self.settings.llm_timeout) + 30.0
        try:
            with ui.console.status("[cyan]检索 + 生成中…[/cyan]") if on_token is None else nullcontext():
                result = self._run_async(
                    asyncio.wait_for(
                        pipeline_ask(question, session=session, stream_callback=on_token),
                        timeout=deadline,
                    )
                )
        except asyncio.TimeoutError:
            ui.console.print(f"\n[red]等待模型响应超过 {deadline:.0f}s，已中止[/red]")
            return
        except KeyboardInterrupt:
            ui.console.print("\n[yellow]已中断[/yellow]")
            return
        except RuntimeError as exc:
            ui.console.print(f"\n[red]{exc}[/red]")
            return
        finally:
            self._stop_live()

        if on_token is None:
            # 非流式：渲染后的 Markdown，打印一次
            ui.console.print(Markdown(result.answer.text))
        else:
            # 流式：token 已 append-only 打印过，这里只补一个换行，**绝不重打**
            ui.console.print()

        cited = ", ".join(result.answer.citation_ids) or "（无）"
        if result.problems:
            ui.console.print(f"\n[dim]引用：{cited}[/dim]  [yellow]校验：{'; '.join(result.problems)}[/yellow]")
        else:
            ui.console.print(f"\n[dim]引用：{cited}[/dim]  [green]引用校验通过[/green]")

        self.history.append(
            {
                "question": question,
                "answer": result.answer.text,
                "citations": result.answer.citation_ids,
                "problems": result.problems,
            }
        )

    def _quick_table(self, items: list) -> Table:
        """`/quick` 的抓取明细表（全部在内存，磁盘上什么都没多）。"""
        from ...core.utils import truncate

        table = Table(title="本次抓取（全部在内存，未落盘）")
        table.add_column("paper_id")
        table.add_column("标题")
        table.add_column("内容")
        table.add_column("页", justify="right")
        table.add_column("chunks", justify="right")
        for item in items:
            label = {
                "pdf": "PDF 全文",
                "web": "网页正文",
                "abstract": "仅摘要",
                "metadata": "仅题录",
            }.get(item.kind, item.kind)
            text = f"[yellow]{label}[/yellow]" if not item.full_text else label
            table.add_row(
                item.paper.paper_id,
                truncate(item.paper.title or "-", 42),
                text,
                str(item.pages or "-"),
                str(item.chunks),
            )
        return table

    def cmd_quick(self, args: str) -> None:
        """`/quick <问题>`：现场从已启用渠道抓取 → RAG → LLM 回答，**PDF 不落盘**。"""
        question, flags = split_args(args)
        if not question:
            ui.console.print(
                "[yellow]用法：/quick <问题> [--papers N] [--limit N] [--k N] "
                "[--source auto|mcp|builtin|all][/yellow]"
            )
            return
        session = self.require_session()
        if session is None:
            return

        from ...pipeline.quick import DEFAULT_QUICK_LIMIT, DEFAULT_QUICK_PAPERS, run_quick

        papers = max(1, int(flags.get("papers", DEFAULT_QUICK_PAPERS)))
        limit = max(papers, int(flags.get("limit", max(papers * 3, DEFAULT_QUICK_LIMIT))))
        k = int(flags.get("k", 0))
        on_token = self._stream_renderer() if self.stream and not session.offline else None
        deadline = max(1.0, self.settings.search_timeout) + max(1.0, self.settings.llm_timeout) + 30.0

        def show_fetched(items: list) -> None:
            ui.console.print(self._quick_table(items))

        ui.console.print(
            f"[dim]检索 → 内存抓全文（最多 {papers} 篇）→ 向量化 → 生成：{question}[/dim]"
        )
        try:
            result = self._run_async(
                asyncio.wait_for(
                    run_quick(
                        question,
                        session=session,
                        limit=limit,
                        papers=papers,
                        k=k,
                        source=flags.get("source", ""),
                        stream_callback=on_token,
                        on_fetched=show_fetched,
                    ),
                    timeout=deadline,
                )
            )
        except asyncio.TimeoutError:
            ui.console.print(f"\n[red]等待超过 {deadline:.0f}s，已中止[/red]")
            return
        except KeyboardInterrupt:
            ui.console.print("\n[yellow]已中断[/yellow]")
            return
        except RuntimeError as exc:
            ui.console.print(f"\n[red]{exc}[/red]")
            return
        finally:
            self._stop_live()

        if result.message:
            ui.console.print(f"[yellow]{result.message}[/yellow]")
            ui.console.print(f"[dim]来源：{result.route}[/dim]")
            return

        if on_token is None:
            ui.console.print(Markdown(result.answer.text))
        else:
            # 流式：token 已 append-only 打印过，这里只补一个换行，**绝不重打**
            ui.console.print()

        cited = ", ".join(result.answer.citation_ids) or "（无）"
        if result.problems:
            ui.console.print(f"\n[dim]引用：{cited}[/dim]  [yellow]校验：{'; '.join(result.problems)}[/yellow]")
        else:
            ui.console.print(f"\n[dim]引用：{cited}[/dim]  [green]引用校验通过[/green]")
        ui.console.print(
            f"[dim]来源：{result.route} ｜ 临时索引 {result.chunks} chunks"
            "（PDF 与索引都没落盘，要留档用 /ingest --ids <id>）[/dim]"
        )
        self.history.append(
            {
                "question": question,
                "answer": result.answer.text,
                "citations": result.answer.citation_ids,
                "problems": result.problems,
                "mode": "quick",
            }
        )

    def cmd_report(self, args: str) -> None:
        topic, flags = split_args(args)
        if not topic:
            ui.console.print("[yellow]用法：/report <topic> [--papers N] [--simple][/yellow]")
            return
        session = self.require_session()
        if session is None:
            return
        papers = int(flags.get("papers", self.settings.max_papers))
        simple = flag_bool(flags, "simple")
        if flags.get("source"):
            self.settings = self.settings.model_copy(update={"search_source": flags["source"]})
        ui.console.print(f"[cyan]开始生成报告：{topic}[/cyan] [dim]（{'simple 单 agent' if simple else '多 agent 监督图'}，"
                      f"最多 {papers} 篇；过程日志见终端）[/dim]")
        try:
            with ui.console.status("[cyan]检索 → 入库 → 精读 → 归纳 → 写作…（可能需要几分钟）[/cyan]"):
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
            ui.console.print("\n[yellow]已中断[/yellow]")
            return
        except RuntimeError as exc:
            ui.console.print(f"[red]{exc}[/red]")
            return

        ui.console.print(papers_table(result.papers, "本次分析的论文"))
        for name, path in result.paths.items():
            ui.console.print(f"[green]✓[/green] {name}: {path}")
        if result.flagged:
            ui.console.print("[yellow]校验提示：[/yellow]" + "；".join(result.flagged[:5]))
        summaries = result.state.get("summaries", []) or []
        ui.console.print(
            f"[dim]摘要 {len(summaries)} 篇 / 索引 {len(result.indexed)} 篇论文[/dim]"
        )
        ui.console.print("[dim]用 /save 保存会话，或直接用编辑器查看生成的 md[/dim]")

    def cmd_mcp(self) -> None:
        from src.paper_agent.sources.mcp import describe_mcp_tools, load_server_specs

        specs = load_server_specs(self.settings)
        if not specs:
            ui.console.print("[yellow]没有可用的 MCP server（pip install arxiv-mcp-server paper-search-mcp）[/yellow]")
            return
        ui.console.print(f"[dim]stdio 存放目录：{self.settings.mcp_storage_path}[/dim]")
        with ui.console.status("[cyan]读取 MCP 工具列表…[/cyan]"):
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
        ui.console.print(table)
        ui.console.print(f"[dim]白名单工具（{len(data['kept'])}）：" + ", ".join(sorted(data["kept"])) + "[/dim]")

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
                ui.console.print("[yellow]用法：/channels rm <name>[/yellow]")
                return
            if self.config.remove_channel(name):
                self.config.save()
                self._refresh_settings("已删除搜索渠道")
            else:
                ui.console.print(f"[red]没有渠道 {name}[/red]")
        elif action in {"key-rm", "rm-key", "clear-key"}:
            if self.config.remove_channel_key(name):
                self.config.save()
                ui.console.print(f"[green]✓[/green] 已删除渠道 [cyan]{name}[/cyan] 的 API key（渠道保留）")
                self._refresh_settings("已删除渠道 key")
            else:
                ui.console.print(f"[red]没有渠道 {name}[/red]")
        elif action in {"domestic", "cn", "国内"} or (
            action in {"on", "off"} and name.lower() in {"domestic", "cn", "国内"}
        ):
            # 支持 `/channels domestic on|off` 与 `/channels on|off domestic`
            word = name.lower() if action in {"domestic", "cn", "国内"} else action
            if word not in {"on", "off", "1", "0", "true", "false", "yes", "no"}:
                ui.console.print("[yellow]用法：/channels domestic on|off[/yellow]")
                return
            enabled = word in {"on", "1", "true", "yes"}
            self.config.set_prefer_domestic(enabled)
            self.config.save()
            self._refresh_settings(f"国内渠道优先已{'开启' if enabled else '关闭'}")
            ui.console.print(
                "[green]✓[/green] 国内渠道优先已开启（nlc / chinaxiv / 百度学术 / 万方 排在前面）"
                if enabled
                else "[dim]已关闭国内优先：源与结果按原顺序排列[/dim]"
            )
            return
        elif action in {"all", "全渠道"} or (action in {"on", "off"} and name.lower() in {"all", "*"}):
            # 支持两种写法：`/channels all on|off` 与 `/channels on|off all`
            word = name.lower() if action in {"all", "全渠道"} else action
            if word not in {"on", "off", "1", "0", "true", "false", "yes", "no"}:
                ui.console.print("[yellow]用法：/channels all on|off[/yellow]")
                return
            enabled = word in {"on", "1", "true", "yes"}
            self.config.set_search_all_channels(enabled)
            self.config.save()
            self._refresh_settings(f"全渠道检索已{'开启' if enabled else '关闭'}")
            if enabled:
                ui.console.print(
                    "[green]✓[/green] 检索时将并发跑全部已注册渠道（免 key 的 + 已配置 key 的；"
                    "缺 key 的自动跳过）"
                )
            else:
                ui.console.print("[dim]已恢复：只跑 BUILTIN_SOURCES 里的默认源[/dim]")
            return
        elif action in {"on", "off"}:
            if not name or name not in self.config.channels:
                ui.console.print(f"[red]没有渠道 {name}[/red]")
                return
            self.config.set_channel_enabled(name, action == "on")
            self.config.save()
            self._refresh_settings(f"渠道 {name} 已{'启用' if action == 'on' else '停用'}")
        else:
            ui.console.print(
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
        ui.console.print(table)
        mode = "[green]已开启[/green]" if self.config.search_all_channels else "[dim]关闭[/dim]"
        dom = "[green]已开启[/green]" if self.config.prefer_domestic else "[dim]关闭[/dim]"
        ui.console.print(
            "[dim]所有渠道默认禁用：用 `/channels add <编号|kind>` 逐个添加后才参与检索"
            "（免 key 的可选填 email 进 polite pool；需 key 的会提示配置）。[/dim]"
        )
        ui.console.print(
            f"国内渠道优先：{dom}（用 `/channels domestic on|off` 切换）——开启后国内库排在前面、结果优先保留。"
        )
        ui.console.print(
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
            ui.console.print(existing)
            ui.console.print("[dim]搜索引擎实际使用的渠道：/channels on|off <name>；删 key：/channels key-rm <name>[/dim]")
        else:
            ui.console.print(
                "[yellow]当前没有启用任何渠道[/yellow]：用 `/channels add <编号|kind>` 添加"
                "（如 `/channels add arxiv`，或 `/channels add 1`）"
            )

    def add_channel(self, args: str) -> None:
        positional, flags = split_args(args)
        token = positional.split()[0] if positional.split() else ""
        preset_map = {num: spec for num, spec in channel_presets()}
        spec = preset_map.get(token) if token else None
        if spec is None and token:
            spec = spec_for(token)
        if spec is None:
            self.show_channels()
            answer = ask_line(ui.console, "[bold]渠道编号或 kind[/bold]（回车取消）› ").strip()
            if not answer:
                ui.console.print("[dim]已取消[/dim]")
                return
            spec = preset_map.get(answer) or spec_for(answer)
        if spec is None:
            ui.console.print("[red]未知渠道类型[/red]（用 /channels 查看可选项）")
            return

        api_key = flags.get("key", "")
        email = flags.get("email", "")
        name = flags.get("name", "") or spec.kind
        if spec.needs_key and not api_key:
            api_key = read_secret(ui.console, f"{spec.label} API key  › ")
            if not api_key:
                ui.console.print(f"[yellow]{spec.label} 需要 API key，已取消[/yellow]")
                return
        if not email and (spec.needs_email or spec.group == "academic"):
            hint = "（可选，进 polite pool；留空跳过）"
            entered = ask_line(ui.console, f"联系邮箱{hint} › ").strip()
            email = entered

        if spec.builtin:
            ui.console.print(f"[dim]{spec.label} 免 key；可选补充 email 进 polite pool。[/dim]")

        channel = self.config.upsert_channel(
            spec.kind, name=name, api_key=api_key, email=email, base_url=flags.get("base-url", "")
        )
        self.config.save()
        ui.console.print(
            f"[green]✓[/green] 已添加渠道 [cyan]{channel.name}[/cyan]（{channel.label}）"
            f"  key={channel.masked_key()} → {self.config.path}"
        )
        self._refresh_settings("搜索渠道已更新")
        ui.console.print("[dim]提示：检索时自动生效（/search ... --source builtin）；删 key 用 /channels key-rm <name>[/dim]")

    def _refresh_settings(self, note: str) -> None:
        """配置变更后刷新 Settings（无需重建会话：搜索渠道不影响索引/模型）。"""
        self.config = UserConfig.load()
        self.settings = get_settings(refresh=True)
        if self.session is not None:
            self.session.settings = self.settings
        if note:
            ui.console.print(f"[dim]{note}（{self.config.path}）[/dim]")
