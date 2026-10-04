# -*- coding: utf-8 -*-
"""命令行入口：`python -m src.paper_agent <command>`（typer + rich）。

命令见 `--help`；交互式使用请用仓库根目录的 `main.py`（同一套 pipeline，REPL 形式）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .config import Settings, get_settings
from .logging_setup import resolve_log_file, setup_logging
from .pdf_server import DEFAULT_PORT as PDF_DEFAULT_PORT
from .pdf_server import collect_pdf_entries, registered_server, start_viewer, stop_registered_server

app = typer.Typer(
    add_completion=False,
    help="学术论文检索与概括分析 Agent（LangChain + MCP + RAG + 多 agent）",
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    """配置日志：控制台 + **项目内**轮转文件 `logs/paper-agent.log`（见 logging_setup）。

    CLI 控制台默认 INFO（`-v` 为 DEBUG），文件始终按 DEBUG 记录，便于事后排查。
    """
    setup_logging(
        verbose=verbose,
        console_level=logging.DEBUG if verbose else logging.INFO,
    )


def cli_settings(offline: bool = False, out: Path | None = None, max_papers: int | None = None) -> Settings:
    """按命令行开关生成 Settings（offline 时使用假模型 + 独立索引目录）。"""
    base = get_settings()
    update: dict = {}
    if offline:
        update["fake_llm"] = True
        update["data_dir"] = Path(base.data_dir) / "offline"
    if out is not None:
        update["output_dir"] = out
    if max_papers is not None:
        update["max_papers"] = max_papers
    return base.model_copy(update=update) if update else base


def _run(coro):
    return asyncio.run(coro)


def papers_table(papers, title: str = "检索结果") -> Table:
    table = Table(title=f"{title}（{len(papers)} 篇）")
    table.add_column("#", justify="right", style="dim")
    table.add_column("paper_id", style="cyan", no_wrap=True)
    table.add_column("年份", justify="right")
    table.add_column("标题")
    table.add_column("PDF/链接", overflow="fold", style="dim")
    for i, p in enumerate(papers, 1):
        table.add_row(
            str(i),
            p.paper_id,
            (p.published or "")[:4],
            p.title[:90],
            (p.pdf_url or p.url or "")[:60],
        )
    return table


# --------------------------------------------------------------------------
# 检索 / 入库 / 问答
# --------------------------------------------------------------------------


@app.command("search")
def search(
    query: str = typer.Argument(..., help="检索词（建议英文）"),
    limit: int = typer.Option(8, "--limit", "-n", help="返回条数"),
    sources: str = typer.Option("", "--sources", help="检索源的子集，如 arxiv,openalex,pubmed"),
    source: str = typer.Option(
        "", "--source", help="检索层：auto（默认，MCP→内置回退）/ mcp / builtin（内置免 key 源+已启用渠道）/ all（MCP+内置合并）"
    ),
    ingest: int = typer.Option(
        0, "--ingest", "-i", help="检索完直接入库（给数字限定篇数；0 表示入库全部检索结果）"
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="检索时不调用 LLM（关闭查询扩展/重排）"),
    json_out: bool = typer.Option(False, "--json", help="以 JSON 输出"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """联网检索论文（默认用 LLM 扩展/重排；MCP 不可用自动回退内置源）。

    例：python -m src.paper_agent search "graph rag" --ingest 3
    """
    _setup_logging(verbose)
    from .pipeline import build_session, ingest_papers, run_search

    session = build_session(cli_settings())
    papers, route = _run(
        run_search(
            query,
            session=session,
            limit=limit,
            sources=sources,
            source=source,
            use_llm=False if no_llm else None,
        )
    )

    if json_out:
        console.print_json(json.dumps([p.model_dump() for p in papers], ensure_ascii=False))
        return
    if not papers:
        console.print(f"[yellow]没有检索到结果（route={route}）[/yellow]")
        raise typer.Exit(code=3)
    console.print(papers_table(papers))
    console.print(f"[dim]来源：{route}[/dim]")

    if ingest:
        selected = papers if ingest <= 0 else papers[:ingest]
        results = _run(ingest_papers(selected, session=session))
        for info in results:
            console.print(
                f"[green]{info['status']:>11}[/green] {info['paper_id']}  {info['chunks']} chunks  {info['message']}"
            )
        console.print(f"完成：{len(results)} 篇已处理，索引 {session.index.chunk_count} chunks")


@app.command("ingest")
def ingest(
    query: str = typer.Argument(None, help="检索词；用 --ids 按 ID 直接抓时可省略"),
    limit: int = typer.Option(5, "--limit", "-n", help="最多入库几篇"),
    ids: str = typer.Option(
        "", "--ids", help="按 ID 直接抓（arXiv ID / DOI / 链接，逗号分隔），不需要先搜索、也不依赖 MCP"
    ),
    source: str = typer.Option("", "--source", help="检索层：auto / mcp / builtin"),
    force: bool = typer.Option(False, "--force", help="已存在的论文重新解析入库"),
    offline: bool = typer.Option(False, "--offline", help="用假 embedding（独立索引目录）"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """联网抓取 → 下载 PDF → 解析 → 切分 → 写入 RAG 索引（不调用 LLM）。

    例：
      python -m src.paper_agent ingest "graph rag survey" --limit 3
      python -m src.paper_agent ingest --ids arxiv:2405.16506,10.1145/3626772.3657775
    """
    _setup_logging(verbose)
    from .pipeline import build_session, run_ingest

    if not (query or ids):
        console.print("[yellow]请给出检索词，或用 --ids 指定要抓取的论文[/yellow]")
        raise typer.Exit(code=2)
    try:
        settings = cli_settings(offline=offline)
        if source:
            settings = settings.model_copy(update={"search_source": source})
        session = build_session(settings)
        results = _run(
            run_ingest(query or "", session=session, limit=limit, ids=ids, force=force)
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=3) from exc

    if not results:
        console.print("[yellow]没有可入库的论文[/yellow]")
        raise typer.Exit(code=3)
    for info in results:
        style = {"indexed": "green", "cached": "cyan"}.get(info["status"], "yellow")
        console.print(
            f"[{style}]{info['status']:>11}[/{style}] {info['paper_id']}  "
            f"{info['chunks']} chunks  {info['message']}",
        )
    ok = sum(1 for r in results if r["status"] in {"indexed", "cached"})
    console.print(f"完成：{ok}/{len(results)} 篇可用于 RAG 分析")


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="问题（中文/英文均可）"),
    papers: str = typer.Option("", "--papers", help="限定论文 id（逗号分隔）；留空=全库"),
    k: int = typer.Option(0, "--k", help="检索片段数（0=用配置默认）"),
    offline: bool = typer.Option(False, "--offline", help="不调用 LLM，只回证据"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """在已入库语料上带引用问答。"""
    _setup_logging(verbose)
    from .pipeline import ask as ask_pipeline
    from .pipeline import build_session

    session = build_session(cli_settings(offline=offline))
    try:
        result = _run(
            ask_pipeline(
                question,
                session=session,
                paper_ids=[p.strip() for p in papers.split(",") if p.strip()] or None,
                k=k,
            )
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=3) from exc

    console.print(Markdown(result.answer.text))
    console.print(f"\n[dim]引用：{', '.join(result.answer.citation_ids) or '（无）'}[/dim]")
    if result.problems:
        console.print("[yellow]引用校验：[/yellow]" + "；".join(result.problems))
    else:
        console.print("[green]引用校验通过[/green]")


# --------------------------------------------------------------------------
# 端到端报告
# --------------------------------------------------------------------------


@app.command("report")
def report(
    topic: str = typer.Argument(..., help="研究主题（中文可）"),
    papers: int = typer.Option(0, "--papers", "-n", help="最多分析几篇（0=用配置默认）"),
    simple: bool = typer.Option(False, "--simple", help="最简模式：单个 agent 自行决定流程"),
    source: str = typer.Option("", "--source", help="检索层：auto / mcp / builtin"),
    offline: bool = typer.Option(False, "--offline", help="离线自检：假模型 + 不做网络检索"),
    out: Path = typer.Option(None, "--out", help="输出目录（默认 output/）"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """端到端：检索 → 入库 → 逐篇精读 → 跨篇归纳 → 出报告（md/bib/json）。"""
    _setup_logging(verbose)
    from .pipeline import build_session, run_report

    settings = cli_settings(offline=offline, out=out, max_papers=papers or None)
    if source:
        settings = settings.model_copy(update={"search_source": source})
    session = build_session(settings)
    tools: list | None = [] if settings.fake_llm else None
    try:
        result = _run(
            run_report(topic, settings=settings, session=session, papers=papers, simple=simple, search_tools=tools)
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=4) from exc

    console.print(papers_table(result.papers, "本次分析的论文"))
    for name, path in result.paths.items():
        console.print(f"[green]✓[/green] {name}: {path}")
    if result.flagged:
        console.print("[yellow]校验提示：[/yellow]" + "；".join(result.flagged[:5]))
    console.print(
        f"[dim]索引当前共 {session.index.chunk_count} chunks / {len(session.index.paper_ids())} 篇论文[/dim]"
    )


# --------------------------------------------------------------------------
# 删除论文 / 搜索渠道 / API key
# --------------------------------------------------------------------------


@app.command("rm")
def rm(
    ids: str = typer.Argument(..., help="要删除的 paper_id（逗号分隔），或 all 删除全部"),
    keep_pdf: bool = typer.Option(False, "--keep-pdf", help="保留本地 PDF 缓存"),
    offline: bool = typer.Option(False, "--offline", help="使用 data/offline 独立索引"),
) -> None:
    """从 RAG 索引中删除论文（默认同时删掉本地 PDF 缓存）。"""
    from .pipeline import build_session, remove_papers

    session = build_session(cli_settings(offline=offline))
    target = ids.strip()
    if target in {"all", "--all", "*"}:
        target_ids = [p["paper_id"] for p in session.index.list_papers()]
    else:
        target_ids = [x.strip() for x in target.split(",") if x.strip()]
    if not target_ids:
        console.print("[yellow]索引为空或没有指定 id[/yellow]")
        raise typer.Exit(code=2)
    removed = remove_papers(target_ids, session=session, remove_pdf=not keep_pdf)
    for paper_id in target_ids:
        ok = paper_id in removed
        console.print(f"{'✓' if ok else '×'} {paper_id}" + ("" if ok else "（未找到）"))
    console.print(f"[dim]剩余 {session.index.chunk_count} chunks / {len(session.index.paper_ids())} 篇[/dim]")


@app.command("channels")
def channels(
    action: str = typer.Argument("list", help="list / add / rm / key-rm / on / off（name=all 切全渠道）"),
    name: str = typer.Argument("", help="渠道 name（rm/key-rm/on/off）或 kind（add）；all/domestic=开关"),
    key: str = typer.Option("", "--key", help="API key（add 时用）"),
    email: str = typer.Option("", "--email", help="联系邮箱（OpenAlex/PubMed polite pool 等）"),
    channel_name: str = typer.Option("", "--name", help="自定义渠道名（默认用 kind）"),
) -> None:
    """搜索渠道配置（像 /connect 配置模型供应商一样，密钥只存本地 JSON）。"""
    from .channels import channel_label, presets, spec_for
    from .userconfig import UserConfig

    config = UserConfig.load()
    action = (action or "list").lower()

    if action == "list":
        table = Table(title="可添加的搜索渠道")
        table.add_column("kind", style="cyan")
        table.add_column("名称")
        table.add_column("凭据", style="dim")
        table.add_column("说明", style="dim")
        for _, spec in presets():
            if spec.needs_key:
                creds = "key"
            elif spec.builtin:
                creds = "免 key"
            else:
                creds = "email(可选)"
            table.add_row(spec.kind, spec.label, creds, spec.description)
        console.print(table)
        if config.channels:
            existing = Table(title="已配置渠道")
            existing.add_column("name", style="cyan")
            existing.add_column("kind")
            existing.add_column("状态")
            existing.add_column("key")
            for cname, ch in config.channels.items():
                existing.add_row(cname, ch.kind, "启用" if ch.enabled else "停用", ch.masked_key())
            console.print(existing)
        return

    if action in {"add", "new"}:
        spec = spec_for(name)
        if spec is None:
            console.print(f"[red]未知渠道 {name}[/red]（执行 channels 查看可选项）")
            raise typer.Exit(code=2)
        if spec.needs_key and not key:
            console.print(f"[red]{spec.label} 需要 --key[/red]")
            raise typer.Exit(code=2)
        channel = config.upsert_channel(spec.kind, name=channel_name, api_key=key, email=email)
        config.save()
        console.print(
            f"[green]✓[/green] 已添加 {channel.name}（{channel_label(channel.kind)}）"
            f" → {config.path}"
        )
        return

    # 开关：支持 `channels all|domestic on|off` 与 `channels on|off all|domestic`
    valid_toggles = {"on", "off", "1", "0", "true", "false", "yes", "no"}
    toggle = ""
    target = ""
    if action in {"all", "domestic", "cn"}:
        toggle, target = name.strip().lower(), action
    elif action in {"on", "off"} and name.strip().lower() in {"all", "*", "domestic", "cn"}:
        toggle, target = action, name.strip().lower()
    if target:
        if toggle not in valid_toggles:
            console.print("[yellow]用法：channels all on|off | channels domestic on|off[/yellow]")
            raise typer.Exit(code=2)
        enabled = toggle in {"on", "1", "true", "yes"}
        if target == "all":
            config.set_search_all_channels(enabled)
            label = "全渠道并发检索"
        else:
            config.set_prefer_domestic(enabled)
            label = "国内渠道优先"
        config.save()
        console.print(f"[green]✓[/green] {label}已{'开启' if enabled else '关闭'} → {config.path}")
        return

    if not name or name not in config.channels:
        console.print(f"[red]没有渠道 {name}[/red]")
        raise typer.Exit(code=2)
    if action in {"rm", "remove", "del"}:
        config.remove_channel(name)
    elif action in {"key-rm", "rm-key", "clear-key"}:
        config.remove_channel_key(name)
    elif action in {"on", "off"}:
        config.set_channel_enabled(name, action == "on")
    else:
        console.print("[yellow]用法：channels [list|add|rm|key-rm|on|off|all on|all off][/yellow]")
        raise typer.Exit(code=2)
    config.save()
    console.print(f"[green]✓[/green] {action} {name} → {config.path}")


@app.command("providers")
def providers(
    action: str = typer.Argument("list", help="list / use / rm / key-rm / sync"),
    name: str = typer.Argument("", help="供应商名（use/rm/key-rm/sync 用）"),
) -> None:
    """管理模型供应商：查看 / 切换默认 / 删除 / 只删 key / 重新拉模型。"""
    from .tui import _sync_models, print_presets
    from .userconfig import UserConfig

    config = UserConfig.load()
    act = (action or "list").lower()
    if act in {"list", "ls"}:
        print_presets(console, config)
        return
    if act in {"use", "switch"}:
        if name not in config.providers:
            console.print(f"[red]没有供应商 {name}[/red]")
            raise typer.Exit(code=2)
        config.set_default(name, config.providers[name].chat_model)
        config.save()
        console.print(f"[green]✓[/green] 默认供应商已切到 {name} → {config.path}")
        return
    if act in {"rm", "remove", "del", "delete"}:
        if not name:
            console.print("[red]用法：providers rm <name>[/red]（providers list 查看名字）")
            raise typer.Exit(code=2)
        if not config.remove_provider(name):
            console.print(f"[red]没有供应商 {name}[/red]")
            raise typer.Exit(code=2)
        config.save()
        console.print(f"[green]✓[/green] 已删除供应商 {name}（含其 API key）→ {config.path}")
        return
    if act in {"key-rm", "rm-key", "clear-key"}:
        if not config.remove_provider_key(name):
            console.print(f"[red]没有供应商 {name}[/red]")
            raise typer.Exit(code=2)
        config.save()
        console.print(f"[green]✓[/green] 已删除 {name} 的 API key（供应商保留）")
        return
    if act == "sync":
        provider = config.providers.get(name) or config.active_provider()
        if provider is None:
            console.print("[red]没有可同步的供应商[/red]")
            raise typer.Exit(code=2)
        _sync_models(console, config, provider)
        config.save()
        return
    console.print(
        "[yellow]用法：providers [list|use <name>|rm <name>|key-rm <name>|sync <name>][/yellow]"
    )
    raise typer.Exit(code=2)


@app.command("embed")
def embed(
    provider: str = typer.Argument("", help="负责 embedding 的供应商名；留空=只查看当前设置"),
    model: str = typer.Option("", "--model", help="embedding 模型名（留空=沿用该供应商已选模型）"),
    auto: bool = typer.Option(False, "--auto", help="改回自动挑选（对话供应商 → 默认 → 第一个带 embedding 的）"),
) -> None:
    """单独设置负责 RAG embedding（建索引用）的供应商与模型。"""
    from .userconfig import UserConfig, resolve_embedding

    config = UserConfig.load()
    if auto:
        config.set_embedding_provider("")
        config.save()
    elif provider:
        item = config.providers.get(provider)
        if item is None:
            console.print(f"[red]没有供应商 {provider}[/red]")
            raise typer.Exit(code=2)
        if model:
            item.embedding_model = model
        config.set_embedding_provider(provider)
        config.save()

    embedder, name = resolve_embedding(config, config.active_provider())
    source = f"显式（{config.embedding_provider}）" if config.embedding_provider else "自动"
    console.print(
        f"embedding：[cyan]{embedder.name if embedder else '（无）'}[/cyan] / "
        f"[cyan]{name or '（无 → RAG 不可用）'}[/cyan]  "
        f"[dim]{source}"
        + (f" · dim={embedder.embedding_dim}" if embedder and embedder.embedding_dim else "")
        + "[/dim]"
    )


@app.command("keys")
def keys(
    rm: str = typer.Option("", "--rm", help="删除指定 key：provider:<name> | channel:<name> | <name>"),
) -> None:
    """查看并删除模型供应商 / 搜索渠道的 API key（只删 key，保留配置）。"""
    from .userconfig import UserConfig

    config = UserConfig.load()
    if rm:
        target = rm.strip()
        kind, _, name = target.partition(":")
        if not name:
            name, kind = kind, ""
        if kind == "provider" or (not kind and name in config.providers):
            if not config.remove_provider_key(name):
                console.print(f"[red]没有供应商 {name}[/red]")
                raise typer.Exit(code=2)
        elif kind == "channel" or (not kind and name in config.channels):
            if not config.remove_channel_key(name):
                console.print(f"[red]没有渠道 {name}[/red]")
                raise typer.Exit(code=2)
        else:
            console.print(f"[red]未找到 {target}[/red]")
            raise typer.Exit(code=2)
        config.save()
        console.print(f"[green]✓[/green] 已删除 {target} 的 API key → {config.path}")
        return

    table = Table(title="已保存的 API key")
    table.add_column("类型")
    table.add_column("name", style="cyan")
    table.add_column("key")
    for name, prov in config.providers.items():
        table.add_row("provider", name, prov.masked_key() if prov.api_key else "（无）")
    for name, ch in config.channels.items():
        table.add_row("channel", name, ch.masked_key())
    console.print(table)
    console.print("[dim]删除：keys --rm provider:<name> | channel:<name>[/dim]")


# --------------------------------------------------------------------------
# 本地 PDF 预览（可独立起服务；退出机制见 pdf_server 模块说明）
# --------------------------------------------------------------------------


def _manifest_rows(index_dir: Path) -> list[dict]:
    """直接读索引 manifest 里的论文行（不加载 embedding、不联网）——供独立命令用。"""
    import json as _json

    try:
        manifest = _json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    papers = manifest.get("papers") if isinstance(manifest, dict) else None
    if not isinstance(papers, dict):
        return []
    return [{"paper_id": pid, **(info if isinstance(info, dict) else {})} for pid, info in papers.items()]


@app.command("papers-open")
def papers_open(
    port: int = typer.Option(PDF_DEFAULT_PORT, "--port", "-p", help="监听端口（被占用会自动换空闲端口）"),
    host: str = typer.Option("127.0.0.1", "--host", help="监听地址（默认只本机；0.0.0.0 = 同网段可访问）"),
    idle: float = typer.Option(-1.0, "--idle", help="空闲多少分钟自动退出（0 = 不自动退出；默认读 PAPER_AGENT_PDF_IDLE_MIN，30）"),
    no_browser: bool = typer.Option(False, "--no-browser", help="不要自动打开浏览器"),
) -> None:
    """起一个本地 HTTP 服务，用浏览器看 `data/papers/` 里抓到的 PDF（Ctrl+C 退出）。

    - 页面右下角「停止预览服务」、`papers-open --stop` 都能停；
    - 默认空闲 30 分钟自动退出；
    - 终端会打印地址与 `ssh -L` 端口转发命令。
    """
    _setup_logging(False)
    s = get_settings()
    existing = registered_server()
    if existing is not None:
        console.print(f"[yellow]已有预览服务在跑：[/yellow]{existing.get('url')}")
        console.print("[dim]停掉它：papers-open --stop（或页面上的「停止预览服务」）[/dim]")
        raise typer.Exit(code=0)
    rows = _manifest_rows(s.index_dir)
    try:
        server = start_viewer(
            lambda: collect_pdf_entries(s.papers_dir, rows),
            papers_dir=s.papers_dir,
            host=host,
            port=port,
            idle_seconds=None if idle < 0 else max(idle, 0.0) * 60,
            open_browser=not no_browser,
            console=console,
        )
    except OSError as exc:
        console.print(f"[red]无法启动预览服务（{host}:{port}）：{exc}[/red]")
        raise typer.Exit(code=4) from exc
    console.print("[dim]按 Ctrl+C 退出；服务日志会写进项目内 logs/[/dim]")
    try:
        while server.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[dim]已中断[/dim]")
    finally:
        server.stop()
    console.print("[green]✓[/green] 预览服务已停止")


@app.command("papers-close")
def papers_close() -> None:
    """停掉正在跑的 PDF 预览服务（含**其它进程**起的那个）。"""
    entry = stop_registered_server()
    if entry is None:
        console.print("[dim]没有注册在案的 PDF 预览服务[/dim]")
        return
    console.print(f"[green]✓[/green] 已停止预览服务（pid={entry.get('pid')} port={entry.get('port')}）")


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------


@app.command("mcp-tools")
def mcp_tools(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="打印详细日志"),
    raw: bool = typer.Option(False, "--raw", help="同时列出被白名单过滤掉的工具"),
) -> None:
    """列出可用的 MCP server 与（白名单过滤后的）工具。"""
    _setup_logging(verbose)
    from .mcp_client import build_client, describe_mcp_tools, filter_tools, load_server_specs

    specs = load_server_specs()
    if not specs:
        console.print("[red]没有可用的 MCP server。[/red] 先执行：")
        console.print("  pip install arxiv-mcp-server paper-search-mcp")
        raise typer.Exit(code=2)

    report_data = _run(describe_mcp_tools())

    table = Table(title="MCP servers")
    table.add_column("server")
    table.add_column("原始工具数", justify="right")
    table.add_column("白名单保留", justify="right")
    table.add_column("状态")
    for name, info in report_data["servers"].items():
        if "error" in info:
            table.add_row(name, "-", "-", f"[red]{info['error'][:80]}[/red]")
        else:
            table.add_row(name, str(info["raw_tools"]), str(info["kept_tools"]), "[green]ok[/green]")
    console.print(table)

    console.print(f"合计原始工具 {report_data['total_raw']} 个，白名单后 {len(report_data['kept'])} 个：")
    for tool_name in sorted(report_data["kept"]):
        console.print(f"  - {tool_name}")

    if raw:
        client = build_client(specs)
        console.print("\n[dim]（含被过滤的工具）[/dim]")
        for name in specs:
            tools = _run(client.get_tools(server_name=name))
            kept = {t.name for t in filter_tools(tools, servers=list(specs))}
            blocked = sorted(t.name for t in tools if t.name not in kept)
            console.print(f"  {name}: " + ", ".join(blocked))


@app.command("selftest")
def selftest(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """检查配置 / 模型 / embedding 是否就绪。"""
    _setup_logging(verbose)
    from .llm import get_chat_model, get_embeddings

    s = get_settings()
    console.print(f"[bold]配置[/bold] model={s.qwen_model} embedding={s.embedding_model} fake_llm={s.fake_llm}")
    console.print(f"[bold]数据[/bold] {s.data_path} | [bold]输出[/bold] {s.output_path}")
    console.print(f"[bold]日志[/bold] {resolve_log_file()}")

    if not s.fake_llm:
        try:
            reply = get_chat_model().invoke("ping（只回复 pong）")
            console.print(f"[green]chat OK[/green] {str(reply.content)[:60]}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]chat FAILED[/red] {type(exc).__name__}: {exc}")
    try:
        vec = get_embeddings().embed_query("academic paper search")
        console.print(f"[green]embedding OK[/green] dim={len(vec)}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]embedding FAILED[/red] {type(exc).__name__}: {exc}")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        console.print("[yellow]已中断[/yellow]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
