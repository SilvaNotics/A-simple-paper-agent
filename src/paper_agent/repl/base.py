# -*- coding: utf-8 -*-
"""`Repl` 的共享状态与核心方法签名（只声明，不实现；供各 mixin 做类型检查）。

`Repl` = `SearchCommands` + `PaperCommands` + `ProviderCommands`，三者都继承本类：
运行时属性由 `Repl.__init__` 建立、核心方法由 `repl.Repl` 实现。把类型集中声明一次，
可以避免 mypy 在每个 mixin 里各自推断出更窄的类型（例如把 `session` 推成非 None）。
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from rich.live import Live

from ..core.config import Settings
from ..pdf.server import PdfServer
from ..pipeline.session import Session
from ..sources.userconfig import UserConfig


class ReplBase:
    """会话共享状态（mixin 之间可见）。
"""

    settings: Settings
    config: UserConfig
    session: Session | None = None
    session_error: str
    history: list[dict]
    stream: bool
    _live: Live | None
    _loop: asyncio.AbstractEventLoop | None
    _pdf_server: PdfServer | None

    # ---- 核心方法：实现在 `repl.Repl`，这里只给签名（含 `self` 交叉调用）----
    def _refresh_settings(self, note: str) -> None:
        raise NotImplementedError

    def _run_async(self, coro: Any) -> Any:
        raise NotImplementedError

    def _stop_live(self) -> None:
        raise NotImplementedError

    def _stream_renderer(self) -> Callable[[str], None]:
        raise NotImplementedError

    def close_pdf_viewer(self, quiet: bool = False) -> None:
        raise NotImplementedError

    def cmd_ask(self, question: str) -> None:
        raise NotImplementedError

    def cmd_channels(self, args: str = "") -> None:
        raise NotImplementedError

    def cmd_connect(self, args: str) -> None:
        raise NotImplementedError

    def cmd_embed(self, args: str = "") -> None:
        raise NotImplementedError

    def cmd_ingest(self, args: str) -> None:
        raise NotImplementedError

    def cmd_keys(self, args: str = "") -> None:
        raise NotImplementedError

    def cmd_logs(self, args: str = "") -> None:
        raise NotImplementedError

    def cmd_mcp(self) -> None:
        raise NotImplementedError

    def cmd_model(self, args: str) -> None:
        raise NotImplementedError

    def cmd_models(self, args: str) -> None:
        raise NotImplementedError

    def cmd_offline(self, args: str) -> None:
        raise NotImplementedError

    def cmd_papers(self, args: str = "") -> None:
        raise NotImplementedError

    def cmd_providers(self, args: str) -> None:
        raise NotImplementedError

    def cmd_report(self, args: str) -> None:
        raise NotImplementedError

    def cmd_search(self, args: str) -> None:
        raise NotImplementedError

    def require_session(self) -> Session | None:
        raise NotImplementedError

    def save_history(self, path: str = "") -> None:
        raise NotImplementedError

    def show_history(self, n: int = 5) -> None:
        raise NotImplementedError

    def show_index(self) -> None:
        raise NotImplementedError
