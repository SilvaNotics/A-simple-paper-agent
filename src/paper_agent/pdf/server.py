# -*- coding: utf-8 -*-
"""本地 PDF 预览服务：用浏览器看 `data/papers/` 里抓到的 PDF。

`/papers open` 会在终端打印一个**只监听回环地址**的 HTTP 端口，浏览器打开后：
左侧列出已抓取的 PDF（标题 / 年份 / chunks / 大小，可按关键词过滤），
点击即在右侧内嵌查看（浏览器自带 PDF 阅读器），也可新标签打开或下载。

远程机器（SSH）用法：服务默认绑 `127.0.0.1`，在本地做端口转发即可——
`ssh -L 8765:127.0.0.1:8765 user@host`，然后本地浏览器打开 `http://127.0.0.1:8765/`。

**退出机制**（四层，总有一条能用）：
1. 页面上「停止预览服务」按钮（POST `/shutdown`，带随机 token，防被别的本地页面关掉）；
2. REPL 里 `/papers close`（本进程起的直接停，注册在案的别的进程用 SIGTERM 停）；
3. 进程退出：`atexit` + `SIGTERM` 处理，服务和注册信息都会收尾；
4. **空闲自动退出**：默认 30 分钟没请求就自己停（`PAPER_AGENT_PDF_IDLE_MIN`，0 = 关闭）。

安全边界（这是个随手起的小服务，刻意做得很窄）：
- 默认只 `bind 127.0.0.1`；`--host 0.0.0.0` 会打印告警但仍允许（自担风险）；
- 只服务 `papers_dir` 下、通过**文件名映射**出来的 PDF，路径不接受用户输入拼接；
- 校验 `Host` 头只认本机名（localhost / 127.0.0.1 / ::1），防 DNS rebinding；
- `/shutdown` 需要启动时生成的 token；索引页只读论文标题等元数据，不涉及 API key。
"""

from __future__ import annotations

import atexit
import html
import json
import logging
import os
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from ..core.config import STATE_DIR
from ..core.utils import safe_filename

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8765          # 终端里打印的默认端口（被占用时自动改用系统分配的空闲端口）
LOOPBACK = "127.0.0.1"
_CHUNK = 256 * 1024          # 传输分片（大 PDF 也不占内存）
_INDEX_TITLE = "paper-agent · 本地 PDF"

# 空闲自动退出：默认 30 分钟没请求就停（0 = 关闭）；可用 PAPER_AGENT_PDF_IDLE_MIN 改
DEFAULT_IDLE_MINUTES = 30.0
IDLE_ENV = "PAPER_AGENT_PDF_IDLE_MIN"
# 「服务注册表」：让**另一个进程**（新起的 REPL / 自己）也能找到并停掉服务
REGISTRY_ENV = "PAPER_AGENT_PDF_REGISTRY"
REGISTRY_FILE = STATE_DIR / "pdf-server.json"


def default_idle_seconds() -> float:
    """空闲退出秒数：`PAPER_AGENT_PDF_IDLE_MIN` 分钟（默认 30，0 = 不自动退出）。"""
    raw = os.getenv(IDLE_ENV, "").strip()
    if not raw:
        return DEFAULT_IDLE_MINUTES * 60
    try:
        minutes = float(raw)
    except ValueError:
        return DEFAULT_IDLE_MINUTES * 60
    return max(minutes, 0.0) * 60


# --------------------------------------------------------------------------
# 打开系统浏览器
#   stdlib `webbrowser` 在本机只装了 `gio` 时会挑 `gio open`；没有桌面环境
#   （服务器 / SSH / 容器）时它会失败并把 `gio: ... Operation not supported`
#   直接打到终端。这里改成自己按优先级试候选启动器，输出一律丢弃。
# --------------------------------------------------------------------------

# 已经成功拉起、需要回收的启动器进程（避免留僵尸）
_BROWSER_PROCS: list[subprocess.Popen] = []
_BROWSER_LOCK = threading.Lock()
_LAUNCH_PROBE_SECONDS = 1.0     # 等这么久还没退出，就当作「浏览器已拉起」


def browser_commands(url: str) -> list[list[str]]:
    """按优先级列出「打开这个 URL」的候选命令（`$BROWSER` 用户指定排最前）。"""
    commands: list[list[str]] = []
    env = (os.getenv("BROWSER") or "").strip()
    if env:
        try:
            commands.append([*shlex.split(env), url])
        except ValueError:
            logger.debug("$BROWSER 解析失败，忽略：%r", env)
    if sys.platform == "darwin":
        commands.append(["open", url])
    elif os.name == "nt":  # pragma: no cover - Windows
        commands.append(["cmd", "/c", "start", "", url])
    else:
        commands += [
            ["xdg-open", url],
            ["wslview", url],              # WSL：交给 Windows 侧默认浏览器
            ["gio", "open", "--", url],
            ["gvfs-open", url],
            ["x-www-browser", url],
            ["sensible-browser", url],
        ]
    return commands


def _reap_browsers() -> None:
    """回收已退出的启动器进程（成功拉起的浏览器会一直跑，不能 wait 到天荒地老）。"""
    with _BROWSER_LOCK:
        _BROWSER_PROCS[:] = [proc for proc in _BROWSER_PROCS if proc.poll() is None]


def _try_launch(cmd: list[str]) -> bool:
    """启动一个候选命令，判断它是否真的把浏览器拉起来了。

    像 `gio open` 在无桌面环境时会**立刻**以非 0 退出——这种情况算失败，继续试下一个。
    """
    try:
        proc: subprocess.Popen = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,   # 启动器的报错不再漏到 REPL 终端
            start_new_session=True,
        )
    except OSError as exc:
        logger.debug("启动 %s 失败：%s", cmd[0], exc)
        return False

    deadline = time.monotonic() + _LAUNCH_PROBE_SECONDS
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if proc.returncode == 0:
        return True
    if proc.returncode is None:          # 还活着 → 浏览器已拉起，交给后台回收
        with _BROWSER_LOCK:
            _BROWSER_PROCS.append(proc)
        return True
    logger.debug("启动器 %s 退出码 %s，继续试下一个", cmd[0], proc.returncode)
    return False


def open_in_browser(url: str) -> bool:
    """用系统默认浏览器打开 `url`，成功返回 True（失败静默，由调用方给提示）。

    逐个尝试 `browser_commands()`，跳过安装不存在的；某个启动器失败就换下一个。
    """
    _reap_browsers()
    for cmd in browser_commands(url):
        if shutil.which(cmd[0]) is None:
            continue
        if _try_launch(cmd):
            logger.debug("已用 %s 打开 %s", cmd[0], url)
            return True
    logger.debug("没有可用的浏览器启动器，未能自动打开 %s", url)
    return False


# --------------------------------------------------------------------------
# 服务注册表（跨进程找到/停掉正在跑的预览服务）
# --------------------------------------------------------------------------


def registry_path() -> Path:
    """注册文件路径（`PAPER_AGENT_PDF_REGISTRY` 可改，测试靠它隔离）。"""
    raw = os.getenv(REGISTRY_ENV, "").strip()
    return Path(raw).expanduser() if raw else REGISTRY_FILE


def read_registry() -> dict[str, Any] | None:
    """读注册信息；文件不存在/坏了返回 None。"""
    path = registry_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def clear_registry(pid: int | None = None) -> None:
    """删除注册文件；给了 `pid` 则只在它属于该进程时删（避免误删新服务）。"""
    entry = read_registry()
    if entry is None:
        return
    if pid is not None and int(entry.get("pid", -1)) != pid:
        return
    try:
        registry_path().unlink()
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:   # 别的用户的进程：当作活着，不用去动它
        return True
    except OSError:
        return False
    return True


def _port_reachable(host: str, port: int, timeout: float = 0.4) -> bool:
    target = LOOPBACK if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((target, port), timeout=timeout):
            return True
    except OSError:
        return False


def registered_server() -> dict[str, Any] | None:
    """当前**活着的**已注册服务（进程在 + 端口可连）；陈旧记录会被清掉。"""
    entry = read_registry()
    if entry is None:
        return None
    pid, port = int(entry.get("pid", -1)), int(entry.get("port", 0))
    if port and _pid_alive(pid) and _port_reachable(str(entry.get("host") or LOOPBACK), port):
        return entry
    logger.info("清理陈旧的 PDF 预览注册信息（pid=%s port=%s）", pid, port)
    clear_registry()
    return None


def stop_registered_server(timeout: float = 5.0) -> dict[str, Any] | None:
    """停掉注册表里那个**别的进程**的服务（SIGTERM + 等待退出）；没有则返回 None。

    绝不对自己发信号：本进程起的服务请用 `PdfServer.stop()`（否则会把自己杀掉）。
    """
    entry = registered_server()
    if entry is None:
        return None
    pid = int(entry.get("pid", -1))
    if pid == os.getpid():
        logger.debug("注册表指向本进程，交给调用方自行 stop()")
        return None
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        logger.warning("停止 PDF 预览服务失败（pid=%s）：%s", pid, exc)
        clear_registry(pid)
        return entry
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _pid_alive(pid):
        time.sleep(0.1)
    clear_registry(pid)
    logger.info("已停止外部 PDF 预览服务（pid=%s port=%s）", pid, entry.get("port"))
    return entry


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PdfEntry:
    """一个可预览的 PDF：索引元数据（可能缺失）+ 本地文件。"""

    paper_id: str
    path: Path
    title: str = ""
    published: str = ""
    n_chunks: int = 0
    indexed: bool = False

    @property
    def size(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0


def collect_pdf_entries(
    papers_dir: Path,
    rows: Sequence[Mapping[str, Any]] = (),
) -> list[PdfEntry]:
    """扫描 `papers_dir` 里的 PDF，并用索引行（`/papers` 的那些）补标题等元数据。

    文件名由 `safe_filename(paper_id)` 生成，这里按同样的规则反查：
    命中索引的算「已入库」，没命中的（例如索引被删但文件还在）也照样列出来。
    """
    by_stem = {safe_filename(str(row.get("paper_id") or "")): row for row in rows if row.get("paper_id")}
    entries: list[PdfEntry] = []
    if papers_dir.is_dir():
        for path in sorted(papers_dir.glob("*.pdf")):
            row = by_stem.get(path.stem)
            entries.append(
                PdfEntry(
                    paper_id=str(row.get("paper_id")) if row else path.stem,
                    path=path,
                    title=str(row.get("title") or "") if row else "",
                    published=str(row.get("published") or "") if row else "",
                    n_chunks=int(row.get("n_chunks") or 0) if row else 0,
                    indexed=row is not None,
                )
            )
    entries.sort(key=lambda e: (not e.indexed, e.paper_id))
    return entries


def pdf_url(base_url: str, paper_id: str, *, download: bool = False) -> str:
    """拼 `/pdf/<id>` 链接（id 里的 `:`、`/` 等必须转义）。"""
    url = f"{base_url.rstrip('/')}/pdf/{quote(paper_id, safe='')}"
    return f"{url}?download=1" if download else url


# --------------------------------------------------------------------------
# 页面
# --------------------------------------------------------------------------

_CSS = """
:root { color-scheme: dark; --bg:#0b1116; --panel:#111a22; --line:#1f2b36; --fg:#dbe7f3; --dim:#8fa3b8;
        --accent:#4da3ff; --ok:#22c55e; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); height:100vh;
       font:14px/1.5 -apple-system,"Segoe UI","Noto Sans CJK SC",Roboto,sans-serif; }
header { display:flex; gap:12px; align-items:baseline; padding:12px 16px; border-bottom:1px solid var(--line); }
h1 { font-size:15px; margin:0; white-space:nowrap; }
h1 small { color:var(--dim); font-weight:400; }
#q { flex:1; min-width:120px; padding:6px 10px; border-radius:6px; border:1px solid var(--line);
     background:var(--panel); color:var(--fg); }
main { display:grid; grid-template-columns:minmax(280px,380px) 1fr; height:calc(100vh - 92px); }
#list { list-style:none; margin:0; padding:6px; overflow:auto; border-right:1px solid var(--line); }
#list li { border-radius:8px; padding:8px 10px; margin-bottom:4px; background:var(--panel); }
#list li:hover { background:#16232e; }
#list a { color:var(--fg); text-decoration:none; }
#list a.t { color:var(--accent); font-weight:600; }
#list .meta { color:var(--dim); font-size:12px; margin-top:2px; display:flex; gap:8px; flex-wrap:wrap; }
#list .pid { font-family:ui-monospace,Menlo,Consolas,monospace; }
#list .ext { float:right; color:var(--dim); text-decoration:none; }
iframe { border:0; width:100%; height:100%; background:#20262c; }
.empty { padding:24px; color:var(--dim); }
footer { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:6px 16px;
         border-top:1px solid var(--line); color:var(--dim); font-size:12px; }
footer button { padding:4px 12px; border-radius:6px; border:1px solid var(--line); background:var(--panel);
                color:var(--fg); cursor:pointer; }
footer button:hover { background:#16232e; color:#f87171; }
@media (max-width:900px) { main { grid-template-columns:1fr; grid-template-rows:40vh 1fr; } }
"""

_JS = """
const q = document.getElementById('q'), list = document.getElementById('list');
const items = [...list.querySelectorAll('li')], count = document.getElementById('count');
function apply() {
  const v = q.value.trim().toLowerCase(); let n = 0;
  for (const li of items) { const hit = !v || li.dataset.k.includes(v); li.hidden = !hit; if (hit) n++; }
  count.textContent = n + ' / ' + items.length;
}
q.addEventListener('input', apply);
document.addEventListener('keydown', e => {
  if (e.key === '/' && document.activeElement !== q) { e.preventDefault(); q.focus(); }
});
apply();
"""


def _fmt_idle(idle_seconds: float) -> str:
    """空闲退出时长的中文描述（0 = 不自动退出）。"""
    if idle_seconds <= 0:
        return "不自动退出"
    if idle_seconds < 90:
        return f"空闲 {idle_seconds:.0f} 秒后自动退出"
    return f"空闲 {idle_seconds / 60:.0f} 分钟后自动退出"


def _fmt_size(size: int) -> str:
    size = max(size, 0)
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MiB"
    if size >= 1024:
        return f"{size / 1024:.0f} KiB"
    return f"{size} B"


def _render_index(
    entries: Sequence[PdfEntry],
    papers_dir: Path,
    base_url: str,
    *,
    token: str = "",
    idle_seconds: float = 0,
) -> str:
    """索引页：左边列表 + 右边内嵌阅读器 + 底部「停止服务」（全部转义，标题里的尖括号不会破坏页面）。"""
    rows: list[str] = []
    for entry in entries:
        badge = "" if entry.indexed else ' <span style="color:#f59e0b">· 未入库</span>'
        title = html.escape(entry.title or entry.paper_id)
        meta = " ".join(
            part
            for part in (
                f"<span class=pid>{html.escape(entry.paper_id)}</span>",
                html.escape(entry.published[:4]) if entry.published else "",
                f"{entry.n_chunks} chunks" if entry.n_chunks else "",
                _fmt_size(entry.size),
            )
            if part
        )
        link = html.escape(pdf_url(base_url, entry.paper_id), quote=True)
        key = html.escape(f"{entry.paper_id} {entry.title}".lower(), quote=True)
        rows.append(
            f'<li data-k="{key}">'
            f'<a class="ext" href="{link}" target="_blank" rel="noopener" title="新标签打开">↗</a>'
            f'<a class="t" href="{link}" target="view">{title}</a>{badge}'
            f'<div class="meta">{meta}</div></li>'
        )

    if rows:
        body = f'<ul id="list">{"".join(rows)}</ul><iframe id="view" name="view" title="PDF 预览"></iframe>'
    else:
        body = (
            '<div class="empty">还没有抓取到 PDF：先 <code>/ingest &lt;主题&gt;</code>，'
            "或 <code>/search … --ingest</code> 下载入库。<br>"
            f"（论文库目录：{html.escape(str(papers_dir))}）</div>"
        )

    stop_form = ""
    if token:
        stop_form = (
            '<form id="stop" method="post" action="/shutdown">'
            f'<input type="hidden" name="token" value="{html.escape(token, quote=True)}">'
            '<button type="submit" title="停掉这个本地预览服务（终端里的 /papers open 与本页面都会失效）">'
            "停止预览服务</button></form>"
        )
    footer = (
        '<footer><span>Ctrl+C 结束终端里的 REPL 也会停服务 · '
        f'{html.escape(_fmt_idle(idle_seconds))} · 关掉页面不会立即退出</span>{stop_form}</footer>'
    )

    return (
        "<!doctype html><html lang=zh-CN><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{_INDEX_TITLE}</title><style>{_CSS}</style></head><body>"
        "<header><h1>已抓取 PDF <small><span id=count>0 / 0</span> · "
        f"{html.escape(str(papers_dir))}</small></h1>"
        '<input id=q placeholder="过滤 paper_id / 标题…（按 / 聚焦）" autofocus></header>'
        f"<main>{body}</main>{footer}<script>{_JS}</script></body></html>"
    )


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def _allowed_hostnames(host: str) -> set[str]:
    return {"localhost", "127.0.0.1", "::1", "[::1]", host}


def _parse_range(header: str, size: int) -> tuple[int, int] | str | None:
    """解析单段 `Range` 请求头。

    返回 `(start, end)`（含两端）、`"unsatisfiable"`（越界，应回 416）或
    `None`（没带 / 语法不认识 → 忽略，按完整内容回 200）。只支持单段范围，
    够浏览器拖动进度条用（多段 `bytes=0-1,5-6` 会退化成完整传输）。
    """
    if not header.startswith("bytes=") or size <= 0:
        return None
    spec = header[len("bytes=") :].split(",")[0].strip()
    first, sep, last = spec.partition("-")
    if not sep:
        return None
    if first.isdigit():
        start = int(first)
        end = min(int(last), size - 1) if last.isdigit() else size - 1
    elif last.isdigit():                      # 后缀范围：bytes=-500（最后 500 字节）
        start, end = max(size - int(last), 0), size - 1
    else:
        return None
    if start >= size or start > end:
        return "unsatisfiable"
    return start, end


class _PdfHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        entries: Callable[[], list[PdfEntry]],
        papers_dir: Path,
        allowed_hosts: set[str],
        *,
        token: str = "",
        idle_seconds: float = 0,
        on_shutdown: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(address, _PdfHandler)
        self.entries = entries
        self.papers_dir_view = papers_dir
        self.allowed_hosts = allowed_hosts
        self.token = token
        self.idle_seconds = idle_seconds
        self.on_shutdown = on_shutdown
        #: 最近一次请求时间（空闲退出的依据）；由 handler 每次请求时刷新
        self.last_request = time.monotonic()


class _PdfHandler(BaseHTTPRequestHandler):
    server_version = "paper-agent-pdf/1.0"
    protocol_version = "HTTP/1.1"   # keep-alive：浏览器翻页/拖动更顺

    server: _PdfHTTPServer          # 由 ThreadingHTTPServer 注入（类型提示用）

    # ---- 基础输出 ----

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str = "text/html; charset=utf-8",
        extra: Mapping[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        body = f"<!doctype html><meta charset=utf-8><h3>{status}</h3><p>{html.escape(message)}</p>"
        self._send_bytes(status, body.encode("utf-8"))

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN401 - 父类签名
        logger.debug("pdf-server %s - %s", self.address_string(), fmt % args)

    # ---- 路由 ----

    def do_GET(self) -> None:  # noqa: N802 - 父类约定
        self._dispatch()

    def do_HEAD(self) -> None:  # noqa: N802 - 父类约定
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802 - 父类约定
        self._dispatch()

    def _dispatch(self) -> None:
        self.server.last_request = time.monotonic()
        try:
            self._route()
        except (BrokenPipeError, ConnectionResetError):  # 浏览器提前取消：不是错误
            logger.debug("pdf-server 连接被客户端中断：%s", self.path)
        except Exception as exc:  # noqa: BLE001 - 单次请求出错不该拖垮服务
            logger.warning("pdf-server 处理 %s 失败：%s", self.path, exc, exc_info=True)
            try:
                self._error(500, f"服务器内部错误：{type(exc).__name__}")
            except Exception:  # noqa: BLE001
                pass

    def _route(self) -> None:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip().lower()
        if host and host not in self.server.allowed_hosts:
            self._error(403, "只接受本机访问（Host 校验失败）")
            return

        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in {"/", "/index.html"}:
            base = f"http://{self.headers.get('Host') or LOOPBACK}/"
            page = _render_index(
                self.server.entries(),
                self.server.papers_dir_view,
                base,
                token=self.server.token,
                idle_seconds=self.server.idle_seconds,
            )
            self._send_bytes(200, page.encode("utf-8"))
            return
        if path == "/healthz":
            self._send_bytes(200, b"ok", "text/plain; charset=utf-8")
            return
        if path == "/shutdown":
            self._shutdown()
            return
        if path.startswith("/pdf/"):
            self._serve_pdf(path[len("/pdf/") :], download="download" in parse_qs(parsed.query))
            return
        self._error(404, f"没有这个路径：{path}")

    def _shutdown(self) -> None:
        """停止服务（页面上的「停止预览服务」按钮）。

        必须带启动时生成的 token：否则任何本地页面/脚本都能把你的预览关掉。
        先回一张「已停止」页，再在后台线程里真正 shutdown（免得把响应掐掉）。
        """
        sent = parse_qs(urlparse(self.path).query).get("token", [""])[0]
        if not sent:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                body = self.rfile.read(length).decode("utf-8", "replace")
                sent = parse_qs(body).get("token", [""])[0]
        if not self.server.token or not secrets.compare_digest(sent, self.server.token):
            self._error(403, "停止服务需要页面上的 token（或直接在终端里用 /papers close）")
            return
        page = (
            "<!doctype html><meta charset=utf-8><title>已停止</title>"
            '<body style="font:16px/1.6 system-ui;padding:40px;background:#0b1116;color:#dbe7f3">'
            "<h2>预览服务已停止</h2><p>可以关掉这个页面了。需要时重新执行 "
            "<code>/papers open</code>（或在终端里 <code>Ctrl+C</code>）。</p></body>"
        )
        self._send_bytes(200, page.encode("utf-8"))
        callback = self.server.on_shutdown
        if callback is not None:
            # 交给后台线程：不能在当前请求线程里等 serve_forever 退出来
            threading.Thread(target=callback, name="pdf-server-shutdown", daemon=True).start()

    def _serve_pdf(self, raw_id: str, *, download: bool) -> None:
        wanted = raw_id[:-4] if raw_id.lower().endswith(".pdf") else raw_id
        entry = next((e for e in self.server.entries() if e.paper_id == wanted), None)
        if entry is None:
            self._error(404, f"没有这篇 PDF：{wanted}（回首页看看可用列表）")
            return
        try:
            size = entry.path.stat().st_size
            handle = entry.path.open("rb")
        except OSError as exc:
            self._error(404, f"文件不可读：{exc}")
            return

        with handle:
            start, end = 0, size - 1
            status = 200
            parsed = _parse_range(self.headers.get("Range", ""), size)
            if parsed == "unsatisfiable":
                self._send_bytes(416, b"", extra={"Content-Range": f"bytes */{size}"})
                return
            if isinstance(parsed, tuple):
                start, end = parsed
                status = 206

            length = end - start + 1
            disposition = "attachment" if download else "inline"
            extra = {
                "Accept-Ranges": "bytes",
                "Content-Disposition": f'{disposition}; filename="{entry.path.name}"',
            }
            if status == 206:
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            self.send_response(status)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(length))
            for key, value in extra.items():
                self.send_header(key, value)
            self.end_headers()
            if self.command == "HEAD":
                return
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


# --------------------------------------------------------------------------
# 服务生命周期
# --------------------------------------------------------------------------


class PdfServer:
    """`/papers open` 背后的小服务：起一个线程跑 HTTP，退出时自动收尾。

    - `entries` 是**回调**而不是快照：每次刷新首页都重新扫描论文库，新入库的论文
      不用重启服务就能看到；
    - 启动时把 `{pid, port, url, ...}` 写进注册表（`.paper-agent/pdf-server.json`），
      其它进程（新的 REPL、`/papers close`）据此找到并停掉它；
    - 退出机制见模块 docstring：页面按钮 / `/papers close` / `atexit`+`SIGTERM` / 空闲超时。
    """

    def __init__(
        self,
        entries: Callable[[], list[PdfEntry]],
        *,
        papers_dir: Path,
        host: str = LOOPBACK,
        port: int = DEFAULT_PORT,
        idle_seconds: float | None = None,
        token: str | None = None,
        register: bool = True,
    ) -> None:
        self._entries = entries
        self._papers_dir = papers_dir
        self.host = host
        self.requested_port = port
        self.idle_seconds = default_idle_seconds() if idle_seconds is None else max(idle_seconds, 0.0)
        self.token = token if token is not None else secrets.token_urlsafe(9)
        self.register = register
        self._httpd: _PdfHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None
        self._closing = threading.Event()
        self._previous_sigterm: Any = None
        self.port = 0

    # ---- 状态 ----

    @property
    def running(self) -> bool:
        return self._httpd is not None

    @property
    def url(self) -> str:
        """给用户/浏览器用的地址（绑 0.0.0.0 时展示 127.0.0.1 才有意义）。"""
        host = LOOPBACK if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{host}:{self.port}/"

    @property
    def idle_text(self) -> str:
        return _fmt_idle(self.idle_seconds)

    # ---- 生命周期 ----

    def start(self) -> str:
        """启动服务并返回访问地址；端口被占用时自动改用系统分配的空闲端口。"""
        if self._httpd is not None:
            return self.url
        allowed = _allowed_hostnames(self.host)

        def _make(address: tuple[str, int]) -> _PdfHTTPServer:
            return _PdfHTTPServer(
                address,
                self._entries,
                self._papers_dir,
                allowed,
                token=self.token,
                idle_seconds=self.idle_seconds,
                on_shutdown=self.stop,
            )

        try:
            httpd = _make((self.host, self.requested_port))
        except OSError:
            if self.requested_port == 0:
                raise
            logger.info("端口 %s 被占用，改用系统分配的空闲端口", self.requested_port)
            httpd = _make((self.host, 0))
        self._httpd = httpd
        self._closing.clear()
        self.port = int(httpd.server_address[1])
        self._thread = threading.Thread(target=httpd.serve_forever, name="pdf-server", daemon=True)
        self._thread.start()
        if self.register:
            self._write_registry()
        atexit.register(self.stop)
        self._install_sigterm()
        if self.idle_seconds > 0:
            self._watchdog = threading.Thread(target=self._watch_idle, name="pdf-server-idle", daemon=True)
            self._watchdog.start()
        logger.info(
            "PDF 预览服务已启动：%s（目录 %s，%s）", self.url, self._papers_dir, self.idle_text
        )
        return self.url

    def stop(self) -> None:
        """停服务：关 HTTP、清注册表、还原信号处理（可重复调用）。"""
        httpd, thread = self._httpd, self._thread
        self._httpd, self._thread, self._watchdog = None, None, None
        self._closing.set()
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        if self.register:
            clear_registry(os.getpid())
        try:
            atexit.unregister(self.stop)
        except Exception:  # noqa: BLE001 - 取消失败不影响退出
            pass
        self._restore_sigterm()
        if httpd is not None:
            logger.info("PDF 预览服务已停止（端口 %s）", self.port)

    # ---- 注册表 / 信号 / 空闲看门狗 ----

    def _write_registry(self) -> None:
        if not self.register:
            return
        entry = {
            "pid": os.getpid(),
            "host": self.host,
            "port": self.port,
            "url": self.url,
            "papers_dir": str(self._papers_dir),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "idle_seconds": self.idle_seconds,
        }
        path = registry_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:   # 只读目录等：服务照跑，只是别人找不到它
            logger.warning("写入 PDF 预览注册信息失败（%s）：%s", path, exc)

    def _install_sigterm(self) -> None:
        """让 `kill`（SIGTERM）也能干净退出（清注册表）；只在主线程装。"""
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            self._previous_sigterm = signal.getsignal(signal.SIGTERM)

            def _handler(_signum: int, _frame: Any) -> None:
                logger.info("收到 SIGTERM，停止 PDF 预览服务")
                self.stop()
                raise SystemExit(0)

            signal.signal(signal.SIGTERM, _handler)
        except (ValueError, OSError):   # 非主线程 / 平台不支持
            self._previous_sigterm = None

    def _restore_sigterm(self) -> None:
        if self._previous_sigterm is None:
            return
        try:
            signal.signal(signal.SIGTERM, self._previous_sigterm)
        except (ValueError, OSError, TypeError):
            pass
        self._previous_sigterm = None

    def _watch_idle(self) -> None:
        """空闲超时自动退出（默认 30 分钟没请求），免得挂着一个没人用的端口。"""
        while not self._closing.wait(min(15.0, max(self.idle_seconds / 4, 1.0))):
            httpd = self._httpd
            if httpd is None:
                return
            idle = time.monotonic() - httpd.last_request
            if idle >= self.idle_seconds:
                logger.info(
                    "PDF 预览服务空闲 %.0f 分钟，自动停止（可用 PAPER_AGENT_PDF_IDLE_MIN=0 关闭）",
                    idle / 60,
                )
                self.stop()
                return


# --------------------------------------------------------------------------
# 起服务 + 打印提示（REPL 与 CLI 共用）
# --------------------------------------------------------------------------


def start_viewer(
    entries: Callable[[], list[PdfEntry]],
    *,
    papers_dir: Path,
    host: str = LOOPBACK,
    port: int = DEFAULT_PORT,
    idle_seconds: float | None = None,
    open_browser: bool = False,
    console: Any = None,
) -> PdfServer:
    """起本地预览服务并打印「地址 / 端口转发 / 退出方式」，返回 `PdfServer`。

    REPL（`/papers open`）与 CLI（`papers-open`）共用，保证两边的提示与退出行为一致。
    """
    from ..core import ui

    out = console or ui.console
    server = PdfServer(entries, papers_dir=papers_dir, host=host, port=port, idle_seconds=idle_seconds)
    url = server.start()
    count = len(entries())
    out.print(f"[green]✓[/green] 本地 PDF 预览已启动：[bold cyan]{url}[/bold cyan]")
    out.print(f"[dim]论文库：{papers_dir} · 共 {count} 篇 PDF（新入库的刷新页面即可看到）[/dim]")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        out.print(f"[yellow]⚠ 已监听 {host}：同一网络内均可访问，仅建议在可信内网使用[/yellow]")
    else:
        out.print(
            f"[dim]远程机器可在本地执行：[/dim][cyan]ssh -L {server.port}:127.0.0.1:{server.port} <user>@<host>[/cyan]"
            "[dim]，再打开上面的地址[/dim]"
        )
    out.print(
        f"[dim]退出方式：页面右下角「停止预览服务」· /papers close · 终端 Ctrl+C · {server.idle_text}[/dim]"
    )
    if open_browser:
        try:
            opened = open_in_browser(url)
        except Exception as exc:  # noqa: BLE001 - 开浏览器失败不该影响预览服务
            logger.debug("自动打开浏览器异常（不影响服务）：%s", exc)
            opened = False
        if not opened:
            # 服务器 / SSH / 容器里没有桌面环境是常态：不报错，只提示手动打开
            out.print("[dim]未能自动打开浏览器，请手动复制上面的地址打开（或加 --no-browser 关闭本提示）[/dim]")
    return server
