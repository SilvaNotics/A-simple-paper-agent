# -*- coding: utf-8 -*-
"""共享的 Rich 控制台与交互提示符。

`console` 是**进程内单例**：`repl*` 各模块与 `main.py` 都从这里取，别再各自
`Console()`。测试或嵌入方要改输出目标，替换这一个对象即可：

    from src.paper_agent.core import ui
    monkeypatch.setattr(ui, "console", Console(file=StringIO()))

（`Console` 也在这里再导出，方便只 import 一个模块。）
"""

from __future__ import annotations

from rich.console import Console

console = Console()

#: REPL 提示符（`main.py` 与 `repl/input.py` 共用同一份）
PROMPT = "[bold cyan]paper-agent[/bold cyan] [dim]›[/dim] "

__all__ = ["PROMPT", "Console", "console"]
