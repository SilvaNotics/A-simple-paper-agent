# -*- coding: utf-8 -*-
"""集中配置：读仓库根目录 .env / 环境变量（pydantic-settings）。

相对路径统一相对仓库根解析；MCP 子进程需要的密钥/邮箱经 `mcp_env()` 透传。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/paper_agent/core/config.py -> 包目录 -> 仓库根目录
# 用「包目录的上一级」而不是固定 parents[N]：以后在 core/ 下再分层也不会算错。
PACKAGE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PACKAGE_DIR.parents[1]
ENV_FILE = ROOT_DIR / ".env"
# 项目内本地状态目录（用户配置 + 命令行历史），随项目目录一起移植。
STATE_DIR = ROOT_DIR / ".paper-agent"
HISTORY_FILE = STATE_DIR / "history"           # stdlib readline 历史
PTK_HISTORY_FILE = STATE_DIR / "history.ptk"   # prompt_toolkit 历史


def resolve_path(path: str | Path) -> Path:
    """把相对路径统一解析到仓库根目录下（绝对路径原样返回）。"""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (ROOT_DIR / p)


# MCP（paper-search / arxiv-mcp-server）支持的源；用于把 `/channels` 的 kind 映射到 MCP sources。
# 不在其中的（如 tavily / baidu_scholar / nlc）只由内置 HTTP 层处理。
MCP_SOURCE_KINDS = frozenset(
    {
        "arxiv",
        "openalex",
        "crossref",
        "europepmc",
        "pmc",
        "pubmed",
        "doaj",
        "semantic",
        "semanticscholar",
        "dblp",
        "zenodo",
        "hal",
        "core",
        "base",
        "ssrn",
    }
)


class Settings(BaseSettings):
    """项目全部可调参数。环境变量名与字段同名（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- 通用 OpenAI 兼容供应商（由 /connect 写入 JSON 后注入） ----------------
    # 这几个字段若被设置，会**优先于**下面的 DashScope / DeepSeek 专用配置。
    llm_kind: str = ""              # dashscope / deepseek / openai / local / openai-compatible ...
    llm_label: str = ""             # 人类可读的供应商标识（界面展示用）
    llm_base_url: str = ""          # 如 https://api.deepseek.com 或 http://localhost:8000/v1
    llm_api_key: SecretStr | None = None
    llm_model: str = ""             # 对话模型
    llm_embedding_model: str = ""   # RAG 用的 embedding 模型（空=该供应商不支持）
    # embedding 可以来自**另一个**供应商（如对话用 DeepSeek、embedding 用 DashScope）
    embed_base_url: str = ""
    embed_api_key: SecretStr | None = None
    embed_model: str = ""
    embed_label: str = ""

    # ---------------- LLM（默认 Qwen / DashScope 兼容接口） ----------------
    dashscope_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("DASHSCOPE_API_KEY", "QWEN_API_KEY")
    )
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices("DASHSCOPE_BASE_URL", "QWEN_BASE_URL"),
    )
    qwen_model: str = Field(
        default="qwen3.8-max", validation_alias=AliasChoices("DASHSCOPE_MODEL", "QWEN_MODEL")
    )
    # 思考模式：默认两者都为 False → 不传 `enable_thinking`，交给服务端，
    # 兼容只能开启思考的模型；ENABLE_THINKING=1 显式开，DISABLE_THINKING=1 显式关。
    enable_thinking: bool = False
    disable_thinking: bool = False

    # ---------------- LLM（备用 DeepSeek） ----------------
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"

    # ---------------- 嵌入模型 ----------------
    embedding_model: str = "text-embedding-v4"   # 实测 1024 维
    embedding_dim: int = 1024
    # 单次 embedding 批次上限（DashScope 实测 10），超限由 store 二分降级。
    embed_batch_size: int = 10

    # ---------------- 检索 / 规模 ----------------
    top_k: int = 6
    max_papers: int = 8
    chunk_size: int = 1200
    chunk_overlap: int = 200
    concurrency: int = 4
    # 联网检索来源：auto（先 MCP，失败/为空则用内置 HTTP）/ mcp / builtin
    search_source: str = "auto"
    # 内置回退层源列表。**默认空**：所有渠道都需要用 `/channels add` 手动添加后才参与检索。
    builtin_sources: str = ""
    prefer_domestic: bool = True      # 源排序 / 结果顺序让国内库靠前
    # `/search --limit` 是「每渠道上限」，用总上限防止多检索式 × 多渠道时结果爆炸。
    max_total_results: int = 200
    # 是否并发跑全部已注册渠道（免 key + 已配置 key）；`/channels all on` 或 SEARCH_ALL_CHANNELS=1 打开。
    search_all_channels: bool = False
    channel_concurrency: int = 6      # 全渠道并发上限
    channel_timeout: float = 25.0     # 单渠道超时（秒），避免慢源拖垮整次检索
    openalex_mailto: str = ""         # 填邮箱进 OpenAlex/Crossref 的 polite pool
    # ---------------- 全文获取兜底（没有 PDF 直链时） ----------------
    # 入库顺序：补链（Unpaywall/OpenAlex/落地页）→ 网页正文 → 题录/摘要，逐级降级。
    pdf_lookup: bool = True           # 没直链时用 Unpaywall/OpenAlex/落地页再找一次
    web_fallback: bool = True         # 补不到 PDF 时抓网页正文入库（Wikipedia/百科/新闻页）
    record_fallback: bool = True      # 连网页都没有时把题录/摘要入库（明确标注「非全文」）
    web_text_min_chars: int = 400     # 网页正文短于该长度视为无效（导航页/占位页）
    http_timeout: float = 30.0
    http_user_agent: str = ""         # 留空用内置 UA
    arxiv_min_interval: float = 3.0   # arXiv 要求请求间隔 ≥3s
    # 显式覆盖检索源（逗号分隔）；留空则用 `/channels add` 启用的渠道。
    search_sources: str = ""
    # `builtin_sources` 之外额外启用的搜索渠道（`/channels` 写入，形如 name -> {kind, api_key, ...}）
    search_channels: dict[str, Any] = Field(default_factory=dict)
    search_use_llm: bool = True       # 检索时用 LLM 做查询扩展 + 相关性重排
    hybrid_retrieval: bool = False    # 需额外安装 rank-bm25
    min_support_ratio: float = 0.3    # 引用原文支撑度阈值

    # ---------------- 超时（单次等待响应上限，秒） ----------------
    llm_timeout: float = 180.0
    llm_max_retries: int = 1
    search_timeout: float = 180.0

    # ---------------- MCP ----------------
    # 默认用包内绝对路径（随包一起移动也不会失效）；仍可用 MCP_SERVERS_FILE 覆盖。
    mcp_servers_file: Path = Field(default=PACKAGE_DIR / "sources" / "mcp_servers.json")
    arxiv_mcp_bin: str = ""       # 留空则用当前解释器同目录下的 console script
    paper_search_mcp_bin: str = ""
    semantic_scholar_api_key: SecretStr | None = None
    unpaywall_email: str = ""        # Unpaywall 补链用的邮箱（留空则退回 openalex_mailto）
    # paper-search `search_papers` 的默认源。默认空：由 `/channels` 启用的渠道决定；
    # 仅当需要固定一组源时才设（显式列出可避免默认 `all` 触发 429 / 慢源）。
    mcp_default_sources: str = ""

    # ---------------- 路径 ----------------
    # 运行产物相对仓库根解析：论文库 data/papers/，报告（Markdown/BibTeX/JSON）output/。
    # 相对值一律相对仓库根，绝对路径原样使用；支持 PAPER_AGENT_DATA_DIR / PAPER_AGENT_OUTPUT_DIR。
    data_dir: Path = Field(
        default=Path("data"), validation_alias=AliasChoices("PAPER_AGENT_DATA_DIR", "DATA_DIR")
    )
    output_dir: Path = Field(
        default=Path("output"), validation_alias=AliasChoices("PAPER_AGENT_OUTPUT_DIR", "OUTPUT_DIR")
    )

    # ---------------- 调试 / 测试 ----------------
    fake_llm: bool = False   # 置 1 用假模型 + 假 embedding：无网络无密钥也能跑通全链路

    # ------------------------------------------------------------------
    # ---------------- 当前生效的模型信息 ----------------
    @property
    def provider_label(self) -> str:
        """界面展示用的供应商标识。"""
        if self.llm_label:
            return self.llm_label
        if self.llm_kind:
            return self.llm_kind
        if self.llm_base_url:
            return self.llm_base_url
        if self.dashscope_api_key:
            return "阿里云百炼 / DashScope（Qwen）"
        if self.deepseek_api_key:
            return "DeepSeek"
        return "未配置"

    @property
    def active_base_url(self) -> str:
        return self.llm_base_url or self.dashscope_base_url

    @property
    def active_embedding_model(self) -> str:
        return self.embed_model or self.llm_embedding_model or self.embedding_model

    @property
    def active_embedding_label(self) -> str:
        return self.embed_label or self.provider_label

    @property
    def is_dashscope(self) -> bool:
        if self.llm_kind:
            return self.llm_kind == "dashscope"
        return bool(self.dashscope_api_key) and not self.llm_base_url

    @property
    def data_path(self) -> Path:
        """数据根目录：`<仓库根>/data/`（PDF 缓存 + 向量索引）。"""
        return resolve_path(self.data_dir)

    @property
    def papers_dir(self) -> Path:
        """论文库：`<仓库根>/data/papers/`（下载的 PDF 缓存）。"""
        return self.data_path / "papers"

    @property
    def index_dir(self) -> Path:
        return self.data_path / "index"

    @property
    def index_file(self) -> Path:
        return self.index_dir / "store.json"

    @property
    def manifest_file(self) -> Path:
        return self.index_dir / "manifest.json"

    @property
    def output_path(self) -> Path:
        """报告输出目录：`<仓库根>/output/`（Markdown + BibTeX + JSON）。"""
        return resolve_path(self.output_dir)

    @property
    def servers_file(self) -> Path:
        return resolve_path(self.mcp_servers_file)

    def ensure_dirs(self) -> None:
        for p in (self.papers_dir, self.index_dir, self.output_path):
            p.mkdir(parents=True, exist_ok=True)

    # ---------------- 搜索渠道 -------------
    def enabled_channels(self) -> list[dict[str, Any]]:
        """返回已启用且带 kind 的搜索渠道配置（`/channels` 写入的 JSON）。"""
        out: list[dict[str, Any]] = []
        for name, cfg in (self.search_channels or {}).items():
            if not isinstance(cfg, dict):
                continue
            if not cfg.get("kind") or not cfg.get("enabled", True):
                continue
            out.append({"name": name, **cfg})
        return out

    def channel_credentials(self, kind: str) -> dict[str, str]:
        """取某个渠道的凭据（key/email/base_url）；没有配置返回空 dict。"""
        for item in self.enabled_channels():
            if item.get("kind") == kind:
                return {
                    "name": str(item.get("name", kind)),
                    "api_key": str(item.get("api_key") or ""),
                    "email": str(item.get("email") or ""),
                    "base_url": str(item.get("base_url") or ""),
                }
        return {}

    @property
    def semantic_scholar_key(self) -> str:
        """Semantic Scholar API key（去空白）；未配置时返回空串。"""
        if self.semantic_scholar_api_key:
            return self.semantic_scholar_api_key.get_secret_value().strip()
        return ""

    @property
    def enabled_source_kinds(self) -> list[str]:
        """用户启用的渠道 kind：`/channels add` 的 JSON 配置 + BUILTIN_SOURCES 显式列表。"""
        kinds = [str(c.get("kind")) for c in self.enabled_channels() if c.get("kind")]
        kinds += [p.strip().lower() for p in self.builtin_sources.split(",") if p.strip()]
        out: list[str] = []
        for kind in kinds:
            if kind and kind not in out:
                out.append(kind)
        return out

    @property
    def active_sources(self) -> str:
        """检索实际使用的源：SEARCH_SOURCES 显式覆盖 > 已启用渠道（默认无）。"""
        explicit = [p.strip().lower() for p in self.search_sources.split(",") if p.strip()]
        return ",".join(explicit or self.enabled_source_kinds)

    @property
    def mcp_sources(self) -> str:
        """MCP `search_papers` 应使用的源：只取已启用渠道里 MCP 支持的 kind。

        例如只启用 Tavily（非 MCP 源）时返回空 → 上层跳过 MCP，只跑内置层。
        无 key 时不追加 `semantic`（匿名共享池固定 429）。
        """
        parts: list[str] = []
        for raw in (self.active_sources or self.mcp_default_sources).split(","):
            kind = raw.strip().lower()
            if kind and kind in MCP_SOURCE_KINDS and kind not in parts:
                parts.append(kind)
        if self.semantic_scholar_key and "semantic" not in parts:
            parts.append("semantic")
        return ",".join(parts)

    def mcp_env(self) -> dict[str, str]:
        """只透传已配置的 MCP 侧变量（未配置的不传，避免字面量 ${VAR} 残留）。"""
        env: dict[str, str] = {}
        if self.semantic_scholar_key:
            env["SEMANTIC_SCHOLAR_API_KEY"] = self.semantic_scholar_key
            env["PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY"] = self.semantic_scholar_key
        if self.unpaywall_email:
            env["UNPAYWALL_EMAIL"] = self.unpaywall_email
            env["PAPER_SEARCH_MCP_UNPAYWALL_EMAIL"] = self.unpaywall_email
        return env


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """进程级单例：先读 .env / 环境变量，再叠加 `<仓库根>/.paper-agent/config.json`。

    叠加规则：`/connect` 写入的 JSON 配置优先级更高（可用
    `PAPER_AGENT_IGNORE_USER_CONFIG=1` 关闭叠加，退回纯 .env 行为）。
    """
    global _settings
    if _settings is None or refresh:
        base = Settings()
        try:
            from ..sources.userconfig import apply_to

            _settings = apply_to(base)
        except Exception as exc:  # noqa: BLE001 - 用户配置坏了不能影响启动
            import logging

            logging.getLogger(__name__).warning("用户配置加载失败，继续使用 .env 配置：%s", exc)
            _settings = base
    return _settings
