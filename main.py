#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学术论文检索与概括分析 Agent —— 交互式命令行入口。

用法：
    python main.py [问题]                  # 进入 REPL；带问题则一次性问答
    python main.py --search "graph rag"    # 一次性检索
    python main.py --ingest "graph rag"    # 一次性入库
    python main.py --report "主题" --papers 3
    python main.py --quick "问题"            # 即抓即答（全文只在内存，PDF 不落盘）
    python main.py --offline                # 假模型 + 独立索引目录（无密钥自检）

进入 REPL 后输入 `/` 弹出命令面板（Tab 补全、Enter 确认），`/help` 查看全部命令；
直接输入自然语言等价于 `/ask`（在当前 RAG 索引上带引用问答）。

检索默认走 MCP，不可用或没结果时回退内置公开接口（arXiv/OpenAlex/Crossref/…，免 key）；
**所有渠道默认禁用**，用 `/channels add <编号|kind>` 添加后才参与检索。
`/channels` 可添加 Semantic Scholar/CORE/Tavily 及国内库（ChinaXiv/国家图书馆免 key，百度学术/万方需 key）。

本文件只做「入口」：解析参数 → 交给 `src/paper_agent/repl/` 的 `Repl`。
REPL 的实现在包里按域拆开：`repl/app.py`（核心） / `repl/commands/*.py`（命令） /
`repl/ui.py`（命令表与进度视图） / `repl/input.py`（补全） / `core/ui.py`（共享 console）。
其他层次：`pipeline/`（检索→入库→报告）/ `sources/`（渠道·抓取·MCP）/ `llm/`（模型） /
`rag/`、`agents/`、`tools/`、`pdf/`、`core/`。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 允许 `python main.py` 直接运行（仓库根目录入 path）
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper_agent.core.logging import setup_logging  # noqa: E402
from src.paper_agent.repl import Repl  # noqa: E402
from src.paper_agent.repl.ui import save_readline_history  # noqa: E402


def _setup_logging(verbose: bool) -> None:
    """配置日志：控制台 + 项目内按天文件（见 `core/logging.py`）。

    REPL 控制台默认只显示 WARNING（`-v` 显示 DEBUG），但文件始终按 DEBUG 记录，
    方便事后用 `/logs` 或直接看文件排查。
    """
    setup_logging(verbose=verbose, console_level=logging.DEBUG if verbose else logging.WARNING)


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
    parser.add_argument(
        "--quick", metavar="QUERY", help="即问即用：现场抓全文→RAG→回答（PDF 不落盘）"
    )
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
    if args.quick:
        repl.cmd_quick(f"{args.quick} --papers {args.papers or 3}")
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
        save_readline_history()
        return 0

    repl.repl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
