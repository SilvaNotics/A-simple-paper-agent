# -*- coding: utf-8 -*-
"""交互层：REPL 入口与命令实现。

- `app.py`：`Repl`（组合命令 mixin、分发输入、兜底异常）；
- `commands/`：按域拆分的命令 mixin（papers / search / providers）；
- `input.py`：prompt_toolkit 补全与选择器；`tui.py`：`/connect` 问答流程；
- `ui.py`：命令表、帮助文本与进度视图；`base.py`：mixin 共享状态声明。
"""

from __future__ import annotations

from .app import Repl

__all__ = ["Repl"]
