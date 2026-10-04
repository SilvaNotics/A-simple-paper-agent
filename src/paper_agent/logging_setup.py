# -*- coding: utf-8 -*-
"""日志落盘：控制台 + **项目内按天分文件**（默认 `<仓库根>/logs/paper-agent-YYYY-MM-DD.log`）。

设计要点：
- 交互式 REPL 的控制台默认只显示 WARNING（`-v/--verbose` 打开 DEBUG），但**文件始终记录**
  （默认 DEBUG），这样「终端里看不到的问题」事后仍可查（REPL 里用 `/logs` 看路径与末尾）；
- 默认**按天分文件**：同一天写满后不截断旧内容，而是续写 `-02`、`-03`…（一天一个日期，便于翻账）；
- 磁盘上只保留最近 `PAPER_AGENT_LOG_KEEP_DAYS`（默认 14）个日期的日志，换文件时顺手清理；
- 设了 `PAPER_AGENT_LOG_FILE` 就切成**固定单文件**模式（按大小轮转 `RotatingFileHandler`）；
- 路径与级别可用环境变量覆盖（写在 shell 或项目 `.env` 里都行）：
  `PAPER_AGENT_LOG_FILE`（完整路径，切换单文件模式）> `PAPER_AGENT_LOG_DIR`（目录，默认 `logs`）；
  `PAPER_AGENT_LOG_LEVEL`（文件级别，默认 DEBUG）；`PAPER_AGENT_LOG_DISABLE=1` 关闭文件日志；
- 旧版的固定文件 `paper-agent.log` 会在按天模式下改名成它最后修改那天（不覆盖已有目标）；
- 相对路径一律相对仓库根（与 `data/`、`output/` 一致），整个项目目录拷走即可带走日志；
- `setup_logging()` 可重复调用（REPL 与 CLI 共用），只清理自己装的 handler，不会叠加；
- 轮转不做跨进程锁：同一目录同时跑多个实例时，极少数情况下会各自分段（日志不丢，只是切片可能零散）。
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import ENV_FILE, resolve_path

# 默认位置：<仓库根>/logs/paper-agent-YYYY-MM-DD.log
DEFAULT_LOG_DIR = "logs"
LOG_FILE_PREFIX = "paper-agent"
LEGACY_FILE_NAME = "paper-agent.log"

# 日期 / 分段：paper-agent-2026-10-04.log、paper-agent-2026-10-04-02.log …
DAY_FORMAT = "%Y-%m-%d"
DAILY_FILE_RE = re.compile(rf"^{re.escape(LOG_FILE_PREFIX)}-(\d{{4}}-\d{{2}}-\d{{2}})(?:-(\d{{2}}))?\.log$")

# 默认保留策略
DEFAULT_KEEP_DAYS = 14     # 按天模式下保留最近 N 个日期
DEFAULT_MAX_MB = 8         # 单个文件上限（同一天超过就续写 -02、-03…）
DEFAULT_BACKUPS = 5        # 固定单文件模式下的历史份数

# 环境变量
LOG_FILE_ENV = "PAPER_AGENT_LOG_FILE"        # 完整文件路径（设了即固定单文件模式）
LOG_DIR_ENV = "PAPER_AGENT_LOG_DIR"          # 目录（默认 logs）
LOG_KEEP_DAYS_ENV = "PAPER_AGENT_LOG_KEEP_DAYS"  # 保留天数（按天模式）
LOG_MAX_MB_ENV = "PAPER_AGENT_LOG_MAX_MB"    # 单文件上限（MiB）
LOG_BACKUPS_ENV = "PAPER_AGENT_LOG_BACKUPS"  # 固定单文件模式的历史份数
LOG_LEVEL_ENV = "PAPER_AGENT_LOG_LEVEL"      # 文件日志级别（默认 DEBUG）
LOG_DISABLE_ENV = "PAPER_AGENT_LOG_DISABLE"  # 置 1 关闭文件日志（只留控制台）

# 文件里带日期（便于事后排查跨天会话），控制台沿用短时间戳
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"
CONSOLE_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
CONSOLE_DATEFMT = "%H:%M:%S"

# 这些第三方库在非 verbose 下一律压到 ERROR：它们默认的 INFO/DEBUG 会淹没日志文件。
# 用**前缀**而不是写死全名：这类库经常换包名（httpx→httpx2、httpcore→httpcore2），
# 写死名单会让新名字的 DEBUG 把日志刷满（见 `_NoisyFilter`）。
NOISY_PREFIXES = (
    "httpx",
    "httpcore",
    "mcp",
    "urllib3",
    "asyncio",
    "openai",
    "arxiv",
    "langchain",
    "langsmith",
)

_TAG_ATTR = "_paper_agent_handler"

# `.env` 只被 pydantic-settings 读取，不会进 os.environ；这里自己兜底读一次并缓存
_DOTENV: dict[str, str] | None = None


def _dotenv_values() -> dict[str, str]:
    """项目 `.env` 的键值（读不到就返回空 dict）——让日志开关也能写在 `.env` 里。"""
    global _DOTENV
    if _DOTENV is None:
        values: dict[str, str] = {}
        try:
            from dotenv import dotenv_values

            if ENV_FILE.exists():
                values = {
                    key: str(value) for key, value in dotenv_values(ENV_FILE).items() if value is not None
                }
        except Exception:  # noqa: BLE001 - 缺依赖/文件坏都不应影响启动
            values = {}
        _DOTENV = values
    return _DOTENV


def _env(name: str, default: str = "") -> str:
    """环境变量优先，其次项目 `.env`（与其它配置项的生效顺序一致）。"""
    value = os.getenv(name)
    return value if value is not None else _dotenv_values().get(name, default)


def _env_number(env: str, default: float, *, minimum: float = 1.0) -> float:
    """读数值型环境变量：非法/过小一律回落默认值。"""
    raw = _env(env).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


# --------------------------------------------------------------------------
# 路径 / 开关
# --------------------------------------------------------------------------


def resolve_log_dir() -> Path:
    """日志目录：`PAPER_AGENT_LOG_DIR`（默认 `logs/`），相对路径相对仓库根。"""
    return resolve_path(_env(LOG_DIR_ENV).strip() or DEFAULT_LOG_DIR)


def log_mode() -> str:
    """`"single"`（设了 `PAPER_AGENT_LOG_FILE`）或 `"daily"`（默认按天分文件）。"""
    return "single" if _env(LOG_FILE_ENV).strip() else "daily"


def daily_log_path(day: str | None = None, part: int = 1, directory: Path | None = None) -> Path:
    """某天第 `part` 段的日志路径（`day` 省略即今天）；`part <= 1` 是当天主文件。"""
    base = directory if directory is not None else resolve_log_dir()
    stamp = day or time.strftime(DAY_FORMAT)
    name = f"{LOG_FILE_PREFIX}-{stamp}.log" if part <= 1 else f"{LOG_FILE_PREFIX}-{stamp}-{part:02d}.log"
    return Path(base) / name


def _parse_log_name(name: str) -> tuple[str, int] | None:
    """从文件名解析出 `(日期, 段号)`；不是本项目的按天日志返回 `None`。"""
    match = DAILY_FILE_RE.match(name)
    if not match:
        return None
    return match.group(1), int(match.group(2) or 1)


def list_log_files(directory: Path | None = None) -> list[Path]:
    """按天日志文件列表，从新到旧（先比日期，再比段号）。目录不存在返回空列表。"""
    base = Path(directory) if directory is not None else resolve_log_dir()
    try:
        entries = list(base.iterdir())
    except OSError:
        return []
    parsed: list[tuple[tuple[str, int], Path]] = []
    for path in entries:
        info = _parse_log_name(path.name)
        if info and path.is_file():
            parsed.append((info, path))
    parsed.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in parsed]


def latest_log_file(directory: Path | None = None) -> Path:
    """最新一份按天日志；一份都没有时返回「今天的主文件」路径（可能还不存在）。"""
    files = list_log_files(directory)
    if files:
        return files[0]
    base = Path(directory) if directory is not None else resolve_log_dir()
    return daily_log_path(None, 1, base)


def resolve_log_file() -> Path:
    """当前生效的日志文件：单文件模式就是它，按天模式是磁盘上最新一份。"""
    explicit = _env(LOG_FILE_ENV).strip()
    if explicit:
        return resolve_path(explicit)
    return latest_log_file()


def keep_days() -> int:
    """按天模式保留的天数（`PAPER_AGENT_LOG_KEEP_DAYS`，默认 14）。"""
    return int(_env_number(LOG_KEEP_DAYS_ENV, DEFAULT_KEEP_DAYS, minimum=1))


def max_bytes() -> int:
    """单个日志文件的字节上限（`PAPER_AGENT_LOG_MAX_MB`，默认 8 MiB）。"""
    return int(_env_number(LOG_MAX_MB_ENV, DEFAULT_MAX_MB, minimum=0.1) * 1024 * 1024)


def backup_count() -> int:
    """固定单文件模式保留的历史份数（`PAPER_AGENT_LOG_BACKUPS`，默认 5）。"""
    return int(_env_number(LOG_BACKUPS_ENV, DEFAULT_BACKUPS, minimum=1))


def file_logging_enabled() -> bool:
    """文件日志是否启用（`PAPER_AGENT_LOG_DISABLE=1` 时关闭）。"""
    return _env(LOG_DISABLE_ENV).strip().lower() not in {"1", "true", "yes", "on"}


def _env_level(env: str, default: int) -> int:
    """读日志级别环境变量：支持 `DEBUG` / `debug` / `10` 三种写法。"""
    raw = _env(env).strip()
    if not raw:
        return default
    if raw.lstrip("+-").isdigit():
        return int(raw)
    return logging.getLevelNamesMapping().get(raw.upper(), default)


# --------------------------------------------------------------------------
# 按天 handler
# --------------------------------------------------------------------------


class DailyFileHandler(logging.Handler):
    """按天分文件的 handler：跨天自动换文件，单日写满续写 `-02`、`-03`…

    - `current_path`：当前正在写的文件；
    - 启动时若当天最后一段已写满，直接开新段；否则**接着写**（重启不丢历史，也不覆盖）；
    - 每次换文件顺手清理超过 `keep_days` 的旧日期（只删自己的文件，不动别人的）。
    """

    terminator = "\n"

    def __init__(
        self,
        directory: str | Path,
        *,
        now=None,
        max_bytes_: int | None = None,
        keep_days_: int | None = None,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        self._now = now if now is not None else time.time
        self.maxBytes = int(max_bytes_) if max_bytes_ is not None else max_bytes()
        self.keep_days = int(keep_days_) if keep_days_ is not None else keep_days()
        self.encoding = encoding
        self.current_path: Path | None = None
        self._stream = None
        self._size = 0
        self._roll(self._day(), part=None)

    # -- 内部工具 ---------------------------------------------------------

    def _day(self) -> str:
        return time.strftime(DAY_FORMAT, time.localtime(self._now()))

    def _pick_part(self, day: str) -> int:
        """当天应写入的段号：默认接着最后一段写，已写满就新开一段。"""
        parts = []
        for path in list_log_files(self.directory):
            info = _parse_log_name(path.name)
            if info and info[0] == day:
                parts.append(info[1])
        if not parts:
            return 1
        last = max(parts)
        path = daily_log_path(day, last, self.directory)
        try:
            full = path.stat().st_size >= self.maxBytes
        except OSError:
            full = False
        return last + 1 if full else last

    def _open(self, day: str, part: int) -> None:
        self.current_path = daily_log_path(day, part, self.directory)
        try:
            self.current_path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.current_path.open("a", encoding=self.encoding)
        except OSError:
            self._stream = None
            self._size = 0
            raise
        try:
            self._size = self.current_path.stat().st_size
        except OSError:
            self._size = 0

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:  # noqa: BLE001 - 关闭失败不该影响日志流程
                pass
        self._stream = None

    def _day_parts(self) -> dict[str, list[Path]]:
        """目录里本项目按天日志的 `日期 -> 文件列表` 映射。"""
        grouped: dict[str, list[Path]] = {}
        for path in list_log_files(self.directory):
            info = _parse_log_name(path.name)
            if info:
                grouped.setdefault(info[0], []).append(path)
        return grouped

    def _prune(self, current_day: str) -> None:
        """只保留最近 `keep_days` 个日期的自家文件（当前这天永远保留）。"""
        grouped = self._day_parts()
        keep = set(sorted(grouped, reverse=True)[: max(1, self.keep_days)])
        keep.add(current_day)
        for day, paths in grouped.items():
            if day in keep:
                continue
            for path in paths:
                try:
                    path.unlink()
                except OSError:
                    pass

    def _roll(self, day: str, part: int | None) -> None:
        self._close_stream()
        if part is None:
            part = self._pick_part(day)
        self._open(day, part)
        self._prune(day)

    # -- logging.Handler 接口 ---------------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record) + self.terminator
            size = len(line.encode(self.encoding, errors="replace"))
            day = self._day()
            if self._stream is None or self.current_path is None or day != self._current_day():
                self._roll(day, part=None)
            elif self._size and self._size + size > self.maxBytes:
                # 先换段再写，单段最多超出一个 record 的量
                self._roll(day, part=self._current_part() + 1)
            if self._stream is None:  # 目录不可写等极端情况
                return
            self._stream.write(line)
            self._stream.flush()
            self._size += size
        except Exception:  # noqa: BLE001 - 交给 logging 的默认错误处理
            self.handleError(record)

    def _current_day(self) -> str:
        info = _parse_log_name(self.current_path.name) if self.current_path else None
        return info[0] if info else ""

    def _current_part(self) -> int:
        info = _parse_log_name(self.current_path.name) if self.current_path else None
        return info[1] if info else 1

    def close(self) -> None:
        self._close_stream()
        super().close()


# --------------------------------------------------------------------------
# 装配
# --------------------------------------------------------------------------


def _tagged(handler: logging.Handler, name: str) -> logging.Handler:
    setattr(handler, _TAG_ATTR, name)
    return handler


class _NoisyFilter(logging.Filter):
    """非 verbose 时把第三方噪声 logger 挡在**我们的 handler** 之外。

    只装在文件/控制台 handler 上（不动 root 与别人的 handler，如 pytest 的 caplog）：
    `logger.setLevel()` 只能管到已经存在的 logger，而 `httpcore2.http11` 这类名字
    是在请求时才创建的，靠前缀过滤才拦得住。
    """

    def __init__(self) -> None:
        super().__init__(name="paper-agent-noisy")

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith(NOISY_PREFIXES)


_NOISY_FILTER = _NoisyFilter()


def _quiet_noisy_loggers() -> None:
    """把已存在的噪声 logger 压到 ERROR（少造 record，比只靠 filter 更省）。"""
    for name in list(logging.root.manager.loggerDict):
        if name.startswith(NOISY_PREFIXES):
            logging.getLogger(name).setLevel(logging.ERROR)
    for name in NOISY_PREFIXES:
        logging.getLogger(name).setLevel(logging.ERROR)


def _reset_handlers(root: logging.Logger) -> None:
    """移除本模块此前装的 handler（重复调用不叠加），其它 handler（pytest 等）保持不动。"""
    for handler in list(root.handlers):
        if getattr(handler, _TAG_ATTR, None):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响后续配置
                pass


def _migrate_legacy_file(directory: Path) -> None:
    """旧版固定 `paper-agent.log` → 按天文件名（否则用户会以为历史日志没了）。

    目标日期取旧文件的最后修改时间；目标已存在时不动旧文件（宁可不合并，也不覆盖）。
    """
    legacy = directory / LEGACY_FILE_NAME
    try:
        if not legacy.is_file():
            return
        stamp = time.strftime(DAY_FORMAT, time.localtime(legacy.stat().st_mtime))
        target = daily_log_path(stamp, 1, directory)
        if not target.exists():
            legacy.replace(target)
    except OSError:
        pass


def setup_logging(
    *,
    verbose: bool = False,
    console_level: int | None = None,
    console: bool = True,
    log_file: str | Path | None = None,
    file_level: int | None = None,
) -> Path | None:
    """装配根 logger：控制台（可选）+ 项目内日志文件（默认按天分文件）。

    - `log_file` 给了就是固定单文件模式（按大小轮转）；否则读 `PAPER_AGENT_LOG_FILE`，
      也没有就按天分文件（`PAPER_AGENT_LOG_DIR`）；
    - 返回实际生效的日志文件路径；文件日志被禁用或目录不可写时返回 `None`
      （此时只保留控制台输出，不抛异常——日志配置失败不该让程序起不来）。
    """
    root = logging.getLogger()
    _reset_handlers(root)

    effective_console = logging.DEBUG if verbose else (
        console_level if console_level is not None else logging.WARNING
    )
    effective_file = file_level if file_level is not None else _env_level(LOG_LEVEL_ENV, logging.DEBUG)

    handlers: list[logging.Handler] = []
    target: Path | None = None

    if file_logging_enabled():
        explicit = Path(log_file) if log_file is not None else None
        if explicit is None and log_mode() == "single":
            explicit = resolve_path(_env(LOG_FILE_ENV).strip())
        try:
            if explicit is not None:
                target = resolve_path(explicit)
                target.parent.mkdir(parents=True, exist_ok=True)
                file_handler: logging.Handler = RotatingFileHandler(
                    target,
                    maxBytes=max_bytes(),
                    backupCount=backup_count(),
                    encoding="utf-8",
                )
            else:
                directory = resolve_log_dir()
                directory.mkdir(parents=True, exist_ok=True)
                _migrate_legacy_file(directory)
                daily = DailyFileHandler(directory)
                target = daily.current_path
                file_handler = daily
            file_handler.setLevel(effective_file)
            file_handler.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATEFMT))
            handlers.append(_tagged(file_handler, "file"))
        except OSError as exc:  # 只读目录 / 权限不足 / 路径不是目录等
            logging.getLogger(__name__).warning("日志文件不可写（%s），仅输出到控制台：%s", target, exc)
            target = None

    if console:
        stream = logging.StreamHandler()
        stream.setLevel(effective_console)
        stream.setFormatter(logging.Formatter(CONSOLE_FORMAT, CONSOLE_DATEFMT))
        handlers.append(_tagged(stream, "console"))

    for handler in handlers:
        if not verbose:
            handler.addFilter(_NOISY_FILTER)
        root.addHandler(handler)
    # 根级别取所有 handler 中最低的，交给各 handler 自己过滤
    root.setLevel(min((h.level for h in handlers), default=effective_console))

    if not verbose:
        _quiet_noisy_loggers()

    return target


# --------------------------------------------------------------------------
# 查看（`/logs`）
# --------------------------------------------------------------------------


def tail_log(lines: int = 20, log_file: str | Path | None = None) -> list[str]:
    """读取日志文件末尾 `lines` 行（文件不存在返回空列表）。

    默认读「当前生效」的日志（按天模式下即磁盘上最新一份）；同一天的 `-02`… 历史段
    不在此列——`/logs --files` 可列出全部，需要翻旧账直接看对应文件。
    """
    target = resolve_path(log_file) if log_file else resolve_log_file()
    try:
        with target.open("r", encoding="utf-8", errors="replace") as fh:
            return [line.rstrip("\n") for line in deque(fh, maxlen=max(1, lines))]
    except FileNotFoundError:
        return []


def log_size(log_file: str | Path | None = None) -> int:
    """当前日志文件字节数（不存在返回 0）。"""
    target = resolve_path(log_file) if log_file else resolve_log_file()
    try:
        return target.stat().st_size
    except OSError:
        return 0


__all__ = [
    "DEFAULT_BACKUPS",
    "DEFAULT_KEEP_DAYS",
    "DEFAULT_LOG_DIR",
    "DEFAULT_MAX_MB",
    "DAY_FORMAT",
    "DailyFileHandler",
    "LOG_BACKUPS_ENV",
    "LOG_DIR_ENV",
    "LOG_DISABLE_ENV",
    "LOG_FILE_ENV",
    "LOG_KEEP_DAYS_ENV",
    "LOG_LEVEL_ENV",
    "LOG_MAX_MB_ENV",
    "backup_count",
    "daily_log_path",
    "file_logging_enabled",
    "keep_days",
    "latest_log_file",
    "list_log_files",
    "log_mode",
    "log_size",
    "max_bytes",
    "resolve_log_dir",
    "resolve_log_file",
    "setup_logging",
    "tail_log",
]
