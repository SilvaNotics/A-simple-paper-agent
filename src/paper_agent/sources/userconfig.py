# -*- coding: utf-8 -*-
"""用户级供应商配置（`/connect` 写入的 JSON）。

所有供应商统一走 OpenAI 兼容格式（`{base_url}/chat/completions`、`/models`、`/embeddings`）。

- 配置文件默认落在**项目内** `<仓库根>/.paper-agent/config.json`（`PAPER_AGENT_CONFIG` 可覆盖），
  0600，随项目目录一起拷贝即可跨系统移植；该目录已被 `.gitignore` 忽略，密钥不会入库；
- 旧版 `~/.config/paper-agent/config.json` 自动迁移到项目内（见 `_migrate_legacy_config`）；
- 按 base URL 自动识别供应商类型，据此决定少量差异化行为与推荐模型；
- 自动拉取 `/models` 并分类（chat / embedding），供 `/models` 切换；
- `apply_to(settings)` 把当前供应商 + 默认模型注入 `Settings`，`llm/factory.py` 优先使用。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx
from pydantic import SecretStr

from .channels import channel_label, default_base_url, spec_for
from ..core.config import STATE_DIR, Settings, resolve_path
from ..core.utils import clean_pasted, clean_secret, mask_secret

logger = logging.getLogger(__name__)

CONFIG_ENV = "PAPER_AGENT_CONFIG"
IGNORE_ENV = "PAPER_AGENT_IGNORE_USER_CONFIG"
# 默认写进项目内：整个项目目录拷贝到别的机器/系统后配置与密钥一并带走。
DEFAULT_CONFIG_PATH = STATE_DIR / "config.json"
# 旧版位置（仅用于一次性迁移，不再作为默认读写路径）。
LEGACY_CONFIG_PATH = Path("~/.config/paper-agent/config.json")


def _migrate_legacy_config() -> None:
    """把旧版 `~/.config/paper-agent/config.json` 迁到项目内（幂等，只做一次）。"""
    legacy = LEGACY_CONFIG_PATH.expanduser()
    if DEFAULT_CONFIG_PATH.exists() or not legacy.exists():
        return
    try:
        DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, DEFAULT_CONFIG_PATH)
        DEFAULT_CONFIG_PATH.chmod(0o600)
        logger.info("已迁移旧用户配置：%s → %s", legacy, DEFAULT_CONFIG_PATH)
    except OSError as exc:  # pragma: no cover - 权限异常时退回默认位置
        logger.warning("旧用户配置迁移失败（%s）：%s", legacy, exc)

# --------------------------------------------------------------------------
# 供应商识别
# --------------------------------------------------------------------------

# 命中 key（base_url 中的子串）→ 供应商类型。顺序敏感：更具体的放前面。
PROVIDER_SIGNATURES: list[tuple[str, str, str]] = [
    ("dashscope.aliyuncs.com", "dashscope", "阿里云百炼 / DashScope（Qwen）"),
    ("api.deepseek.com", "deepseek", "DeepSeek"),
    ("api.openai.com", "openai", "OpenAI"),
    ("api.moonshot.cn", "moonshot", "Moonshot / Kimi"),
    ("api.siliconflow.cn", "siliconflow", "SiliconFlow"),
    ("open.bigmodel.cn", "zhipu", "智谱 GLM"),
    ("ark.cn-beijing.volces.com", "volcengine", "火山方舟 / 豆包"),
    ("openrouter.ai", "openrouter", "OpenRouter"),
    ("api.together.xyz", "together", "Together AI"),
    ("api.groq.com", "groq", "Groq"),
    ("api.mistral.ai", "mistral", "Mistral"),
    ("generativelanguage.googleapis.com", "gemini-openai", "Gemini（OpenAI 兼容端点）"),
    ("api.x.ai", "xai", "xAI Grok"),
    ("api.minimax.chat", "minimax", "MiniMax"),
    ("api.baichuan-ai.com", "baichuan", "百川"),
    ("api.lingyiwanwu.com", "yi", "零一万物"),
    ("localhost", "local", "本地服务（vLLM / Ollama / LM Studio）"),
    ("127.0.0.1", "local", "本地服务（vLLM / Ollama / LM Studio）"),
    ("0.0.0.0", "local", "本地服务（vLLM / Ollama / LM Studio）"),
    ("host.docker.internal", "local", "本地服务（Docker 内访问宿主）"),
]

# 各类型的推荐默认（用于 /connect 提示与模型兜底选择）
PROVIDER_HINTS: dict[str, dict[str, Any]] = {
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat": ["qwen3.8-max", "qwen-max", "qwen-plus", "qwen-turbo"],
        "embedding": "text-embedding-v4",
        "embedding_candidates": ["text-embedding-v4", "text-embedding-v3", "text-embedding-v2"],
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "chat": ["deepseek-chat", "deepseek-flash", "deepseek-v4-pro", "deepseek-reasoner"],
        "embedding": "",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "chat": ["gpt-4o", "gpt-4.1", "gpt-4o-mini", "o3-mini"],
        "embedding": "text-embedding-3-small",
        "embedding_candidates": ["text-embedding-3-small", "text-embedding-3-large"],
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "chat": ["kimi-k2-0905-preview", "moonshot-v1-128k", "moonshot-v1-32k"],
        "embedding": "",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "chat": ["Qwen/Qwen3-235B-A22B-Instruct-2507", "deepseek-ai/DeepSeek-V3"],
        "embedding": "BAAI/bge-m3",
        "embedding_candidates": ["BAAI/bge-m3", "BAAI/bge-large-zh-v1.5"],
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "chat": ["glm-4.6", "glm-4-plus", "glm-4-flash"],
        "embedding": "embedding-3",
        "embedding_candidates": ["embedding-3", "embedding-2"],
    },
    "local": {
        "base_url": "http://localhost:8000/v1",
        "chat": [],
        "embedding": "",
    },
}

# 明显不是对话模型的家族（用于从 /models 结果里过滤）
NON_CHAT_PATTERNS = (
    "embedding",
    "rerank",
    "image",
    "video",
    "audio",
    "tts",
    "asr",
    "speech",
    "voice",
    "realtime",
    "ocr",
    "moderation",
    "wanx",
    "cosyvoice",
    "paraformer",
    "sensevoice",
    "flux",
    "stable-diffusion",
    "sd3",
    "upscale",
    "background",
    "virtualtryon",
    "imageedit",
    "imagesynthesis",
    "wordart",
    "anime",
    "emoji",
    "livemotion",
    "videoretalk",
    "video-edit",
    "videoedit",
    "kling",
    "codegeex",
)
EMBEDDING_PATTERNS = ("embedding", "embed-", "bge-", "bge_", "gte-", "/bge", "jina-embed", "voyage")


def detect_provider(base_url: str) -> tuple[str, str]:
    """从 base URL 推断 (kind, 说明)。"""
    url = (base_url or "").strip().lower()
    for needle, kind, label in PROVIDER_SIGNATURES:
        if needle in url:
            return kind, label
    if url.endswith("/v1") or "/v1/" in url or "compatible-mode" in url:
        return "openai-compatible", "通用 OpenAI 兼容端点"
    return "openai-compatible", "通用 OpenAI 兼容端点"


def normalize_base_url(base_url: str) -> str:
    """补全协议、去掉结尾斜杠；对本地服务不瞎补 /v1（用户填什么用什么）。

    先做粘贴清洗，避免把终端注入的括号粘贴标记当成 URL 的一部分。
    """
    url = clean_pasted(base_url).rstrip("/")
    if not url:
        return ""
    if not re.match(r"^https?://", url):
        url = "http://" + url if re.match(r"^(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])", url) else "https://" + url
    return url


def provider_name(base_url: str, kind: str) -> str:
    """生成稳定的 short name（同名则视为同一个供应商，重复 /connect 会覆盖更新）。"""
    url = (base_url or "").lower()
    m = re.match(r"^https?://([^/:]+)(?::(\d+))?", url)
    host = m.group(1) if m else kind
    port = m.group(2) if m else None
    host = re.sub(r"^api\.|^open\.|^www\.", "", host)
    host = host.split(".")[0] if "." in host else host
    name = host or kind
    if port:
        name = f"{name}-{port}"
    return re.sub(r"[^a-zA-Z0-9._-]", "-", name)[:40] or kind


def classify_models(models: Iterable[str]) -> tuple[list[str], list[str]]:
    """把模型列表分成 (chat 候选, embedding 候选)。"""
    chat: list[str] = []
    embed: list[str] = []
    for mid in models:
        low = (mid or "").lower()
        if not low:
            continue
        if any(p in low for p in EMBEDDING_PATTERNS):
            embed.append(mid)
            continue
        if any(p in low for p in NON_CHAT_PATTERNS):
            continue
        chat.append(mid)
    return sorted(set(chat)), sorted(set(embed))


def guess_chat_model(kind: str, models: list[str], current: str = "") -> str:
    """在候选里挑一个合理的默认对话模型。"""
    if current and current in models:
        return current
    hints = PROVIDER_HINTS.get(kind, {}).get("chat", []) or []
    lowered = {m.lower(): m for m in models}
    for hint in hints:
        if hint.lower() in lowered:
            return lowered[hint.lower()]
    for m in models:
        low = m.lower()
        if any(k in low for k in ("max", "chat", "instruct", "plus", "pro")):
            return m
    return models[0] if models else ""


def guess_embedding_model(kind: str, embed_models: list[str], current: str = "") -> str:
    """挑默认 embedding 模型。

    顺序：保留已有选择 → 提示模型命中列表 → 列表里带 embedding 的条目 →
    已知可用但列表未列出的提示模型（很多供应商的 /models 不返回 embedding）
    → 列表第一项 → 空串（RAG 会提示不可用）。
    """
    if current and current in embed_models:
        return current
    hint_single = PROVIDER_HINTS.get(kind, {}).get("embedding") or ""
    hints = ([hint_single] if hint_single else []) + list(
        PROVIDER_HINTS.get(kind, {}).get("embedding_candidates") or []
    )
    lowered = {m.lower(): m for m in embed_models}
    for hint in hints:
        if hint.lower() in lowered:
            return lowered[hint.lower()]
    for m in embed_models:
        if "embedding" in m.lower():
            return m
    if hint_single:
        return hint_single
    return embed_models[0] if embed_models else ""


# --------------------------------------------------------------------------
# 网络：模型列表 / 探针
# --------------------------------------------------------------------------


async def fetch_models(base_url: str, api_key: str, timeout: float = 25.0) -> list[str]:
    """GET {base_url}/models（OpenAI 兼容）。失败抛异常，由调用方给出可读提示。"""
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    items = data.get("data") if isinstance(data, dict) else None
    if items is None and isinstance(data, list):
        items = data
    models: list[str] = []
    for item in items or []:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("name") or item.get("model")
        else:
            mid = str(item)
        if mid:
            models.append(str(mid))
    return sorted(set(models))


async def probe_embedding_dim(
    base_url: str, api_key: str, model: str, timeout: float = 30.0
) -> int:
    """用一次极小的 embedding 调用探测向量维度（用于索引签名，避免跨供应商混用）。"""
    if not model:
        return 0
    url = base_url.rstrip("/") + "/embeddings"
    payload = {"model": model, "input": ["dimension probe"], "encoding_format": "float"}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    try:
        return len(data["data"][0]["embedding"])
    except Exception:  # noqa: BLE001
        return 0


# --------------------------------------------------------------------------
# 配置读写
# --------------------------------------------------------------------------


@dataclass
class Provider:
    """一个 OpenAI 兼容供应商。"""

    name: str
    base_url: str
    api_key: str
    kind: str = "openai-compatible"
    label: str = ""
    chat_model: str = ""          # 该供应商上次使用的对话模型
    embedding_model: str = ""     # 用于 RAG 的 embedding 模型（空=不支持）
    embedding_dim: int = 0
    models: list[str] = field(default_factory=list)          # 全部模型 id
    chat_models: list[str] = field(default_factory=list)     # 过滤后的对话模型
    embedding_models: list[str] = field(default_factory=list)
    synced_at: str = ""

    def masked_key(self) -> str:
        return mask_secret(self.api_key)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Provider":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class SearchChannel:
    """一个搜索渠道（学术库 / 网页搜索）。密钥与模型供应商一样只存在本地 JSON。"""

    name: str
    kind: str
    label: str = ""
    base_url: str = ""
    api_key: str = ""
    email: str = ""
    enabled: bool = True
    notes: str = ""
    added_at: str = ""

    def masked_key(self) -> str:
        return mask_secret(self.api_key) if self.api_key else "（无 key）"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SearchChannel":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    @property
    def display(self) -> str:
        state = "启用" if self.enabled else "停用"
        return f"{self.label or self.kind} ({self.kind} · {state} · {self.masked_key()})"


class UserConfig:
    """`<仓库根>/.paper-agent/config.json`（项目内，随项目移植）的读写。"""

    def __init__(self, path: Path | None = None) -> None:
        explicit = path or os.getenv(CONFIG_ENV)
        if not explicit:
            _migrate_legacy_config()
        # 相对路径统一相对仓库根解析（与 data/output 等保持一致），而非当前工作目录。
        self.path = resolve_path(explicit) if explicit else DEFAULT_CONFIG_PATH
        self.default_provider: str = ""
        self.default_model: str = ""
        # 显式指定负责 embedding 的供应商（空 = 自动挑选：对话供应商 → 默认 → 第一个带 embedding 的）
        self.embedding_provider: str = ""
        # 检索时并发跑全部已注册渠道（`/channels all on` / `SEARCH_ALL_CHANNELS=1`）
        self.search_all_channels: bool = False
        # 优先国内渠道（源排序 + 结果排序；`/channels domestic off` / `PREFER_DOMESTIC=0`）
        self.prefer_domestic: bool = True
        self.providers: dict[str, Provider] = {}
        self.channels: dict[str, SearchChannel] = {}

    # ---------------- 读写 ----------------
    @classmethod
    def load(cls, path: Path | None = None) -> "UserConfig":
        cfg = cls(path)
        if not cfg.path.exists():
            return cfg
        try:
            data = json.loads(cfg.path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("配置文件解析失败（%s）：%s", cfg.path, exc)
            return cfg
        cfg.default_provider = data.get("default_provider", "")
        cfg.default_model = data.get("default_model", "")
        cfg.embedding_provider = data.get("embedding_provider", "")
        cfg.search_all_channels = bool(data.get("search_all_channels", False))
        cfg.prefer_domestic = bool(data.get("prefer_domestic", True))
        cfg.providers = {
            name: Provider.from_json(item) for name, item in (data.get("providers") or {}).items()
        }
        cfg.channels = {
            name: SearchChannel.from_json(item)
            for name, item in (data.get("channels") or {}).items()
        }
        return cfg

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "default_provider": self.default_provider,
            "default_model": self.default_model,
            "embedding_provider": self.embedding_provider,
            "search_all_channels": self.search_all_channels,
            "prefer_domestic": self.prefer_domestic,
            "providers": {name: p.to_json() for name, p in self.providers.items()},
            "channels": {name: c.to_json() for name, c in self.channels.items()},
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        try:  # 密钥文件收紧权限
            self.path.chmod(0o600)
        except OSError:  # pragma: no cover
            pass
        return self.path

    # ---------------- 供应商管理 ----------------
    def upsert_provider(
        self,
        base_url: str,
        api_key: str,
        name: str = "",
        kind: str = "",
        label: str = "",
    ) -> Provider:
        base_url = normalize_base_url(base_url)
        api_key = clean_secret(api_key)
        detected_kind, detected_label = detect_provider(base_url)
        kind = kind or detected_kind
        label = label or detected_label
        name = name or provider_name(base_url, kind)

        existing = self.providers.get(name)
        provider = existing or Provider(name=name, base_url=base_url, api_key=api_key)
        provider.base_url = base_url
        provider.api_key = api_key or provider.api_key
        provider.kind = kind
        provider.label = label
        if not provider.chat_model:
            provider.chat_model = guess_chat_model(kind, provider.chat_models)
        if not provider.embedding_model:
            provider.embedding_model = guess_embedding_model(kind, provider.embedding_models)
        self.providers[name] = provider
        if not self.default_provider:
            self.default_provider = name
        return provider

    def remove_provider(self, name: str) -> bool:
        if name not in self.providers:
            return False
        self.providers.pop(name)
        if self.default_provider == name:
            self.default_provider = next(iter(self.providers), "")
        if self.embedding_provider == name:
            self.embedding_provider = ""
        return True

    def remove_provider_key(self, name: str) -> bool:
        """只删除某个供应商的 API key，保留供应商与已选模型。"""
        provider = self.providers.get(name)
        if provider is None:
            return False
        provider.api_key = ""
        return True

    # ---------------- 搜索渠道管理 ----------------
    def set_search_all_channels(self, enabled: bool) -> None:
        """开关「检索时并发跑全部已注册渠道」（`/channels all on|off`）。"""
        self.search_all_channels = bool(enabled)

    def set_prefer_domestic(self, enabled: bool) -> None:
        """开关「优先国内渠道」（`/channels domestic on|off`）。"""
        self.prefer_domestic = bool(enabled)

    def upsert_channel(
        self,
        kind: str,
        name: str = "",
        api_key: str = "",
        email: str = "",
        base_url: str = "",
        label: str = "",
        enabled: bool = True,
    ) -> SearchChannel:
        import time

        kind = str(kind or "").strip().lower()
        spec = spec_for(kind)
        name = clean_pasted(name) or kind
        existing = self.channels.get(name)
        channel = existing or SearchChannel(name=name, kind=kind)
        channel.kind = kind
        channel.label = label or channel.label or channel_label(kind)
        channel.base_url = normalize_base_url(base_url) or channel.base_url or default_base_url(kind)
        cleaned_key = clean_secret(api_key)
        channel.api_key = cleaned_key or channel.api_key
        channel.email = clean_pasted(email) or channel.email
        channel.enabled = bool(enabled)
        if not channel.added_at:
            channel.added_at = time.strftime("%Y-%m-%d %H:%M")
        if spec and spec.description and not channel.notes:
            channel.notes = spec.description
        self.channels[name] = channel
        return channel

    def remove_channel(self, name: str) -> bool:
        if name not in self.channels:
            return False
        self.channels.pop(name)
        return True

    def remove_channel_key(self, name: str) -> bool:
        """只删除某个搜索渠道的 API key（保留渠道配置）。"""
        channel = self.channels.get(name)
        if channel is None:
            return False
        channel.api_key = ""
        return True

    def set_channel_enabled(self, name: str, enabled: bool) -> bool:
        channel = self.channels.get(name)
        if channel is None:
            return False
        channel.enabled = bool(enabled)
        return True

    def active_channels(self) -> list[SearchChannel]:
        return [c for c in self.channels.values() if c.enabled]

    def active_provider(self) -> Provider | None:
        if self.default_provider and self.default_provider in self.providers:
            return self.providers[self.default_provider]
        return next(iter(self.providers.values()), None)

    def set_default(self, provider_name: str, model: str = "") -> None:
        self.default_provider = provider_name
        provider = self.providers.get(provider_name)
        if provider and model:
            provider.chat_model = model
        self.default_model = model or (provider.chat_model if provider else "")

    def set_embedding_provider(self, name: str) -> bool:
        """显式指定负责 embedding 的供应商；`name` 为空则改回自动挑选。"""
        name = (name or "").strip()
        if name and name not in self.providers:
            return False
        self.embedding_provider = name
        return True

    def apply_sync(self, provider: Provider, models: list[str]) -> None:
        """把 /models 拉取结果写进 provider（含分类与默认模型推断）。"""
        import time

        chat, embed = classify_models(models)
        provider.models = models
        provider.chat_models = chat
        provider.embedding_models = embed
        provider.synced_at = time.strftime("%Y-%m-%d %H:%M")
        if not provider.chat_model or provider.chat_model not in chat:
            provider.chat_model = guess_chat_model(provider.kind, chat, provider.chat_model)
        guessed_embed = guess_embedding_model(provider.kind, embed, provider.embedding_model)
        if guessed_embed:
            provider.embedding_model = guessed_embed


# --------------------------------------------------------------------------
# 注入 Settings
# --------------------------------------------------------------------------


def pick_embedding_provider(config: UserConfig, chat_provider: Provider | None) -> Provider | None:
    """挑选负责 embedding 的供应商。

    对话供应商常常没有 embedding（如 DeepSeek），因此允许「对话 A + embedding B」：
    优先当前供应商，其次默认供应商，最后按配置顺序找第一个支持 embedding 的。
    """
    if chat_provider is not None and chat_provider.embedding_model:
        return chat_provider
    ordered: list[Provider] = []
    if config.default_provider in config.providers:
        ordered.append(config.providers[config.default_provider])
    ordered += [p for p in config.providers.values() if p not in ordered]
    if chat_provider is not None and chat_provider not in ordered:
        ordered.insert(0, chat_provider)
    for item in ordered:
        if item.embedding_model:
            return item
    return None


def resolve_embedding(
    config: UserConfig, chat_provider: Provider | None = None
) -> tuple[Provider | None, str]:
    """解析负责 embedding 的 (供应商, 模型)。

    显式 `config.embedding_provider` 优先（`/embed` 设置）；未设置时自动挑选，
    规则见 `pick_embedding_provider`。
    """
    if config.embedding_provider:
        pinned = config.providers.get(config.embedding_provider)
        if pinned is not None:
            return pinned, pinned.embedding_model
    provider = pick_embedding_provider(config, chat_provider)
    return provider, (provider.embedding_model if provider else "")


def settings_overrides(config: UserConfig, provider: Provider | None = None, model: str = "") -> dict[str, Any]:
    """把「当前供应商 + 默认模型 + embedding 供应商」翻译成 `Settings` 覆盖字段。"""
    provider = provider or config.active_provider()
    if provider is None:
        return {}
    # 关键：全局 default_model 只属于「默认供应商」。换到另一家时若它自己没选过模型，
    # 绝不能沿用别家的模型（否则会出现「连的是本地服务、模型却是 qwen」这种串台）。
    own_default = config.default_model if config.default_provider == provider.name else ""
    chosen = model or provider.chat_model or own_default
    overrides: dict[str, Any] = {
        "llm_kind": provider.kind,
        "llm_label": provider.label,
        "llm_base_url": provider.base_url,
        "llm_api_key": SecretStr(provider.api_key) if provider.api_key else None,
        "llm_model": chosen,
        "llm_embedding_model": provider.embedding_model,
    }
    if provider.embedding_dim:
        overrides["embedding_dim"] = provider.embedding_dim

    embedder, embed_model = resolve_embedding(config, provider)
    if embedder is not None and embed_model:
        overrides["embed_base_url"] = embedder.base_url
        overrides["embed_api_key"] = SecretStr(embedder.api_key) if embedder.api_key else None
        overrides["embed_model"] = embed_model
        overrides["embed_label"] = embedder.label or embedder.name
        if embedder.embedding_dim:
            overrides["embedding_dim"] = embedder.embedding_dim

    # 搜索渠道（与模型供应商同理：JSON 配置叠加到 Settings）
    if config.search_all_channels:  # 只在开启时覆盖，避免把 .env 的 SEARCH_ALL_CHANNELS=1 盖掉
        overrides["search_all_channels"] = True
    if not config.prefer_domestic:  # 默认开启；只在显式关闭时覆盖（不盖掉 .env）
        overrides["prefer_domestic"] = False
    if config.channels:
        overrides["search_channels"] = {name: c.to_json() for name, c in config.channels.items()}
        # 兼容既有字段：Semantic Scholar key 也透传给 MCP 子进程
        for channel in config.channels.values():
            if channel.kind == "semanticscholar" and channel.api_key:
                overrides["semantic_scholar_api_key"] = SecretStr(channel.api_key)
                break
    return overrides


def apply_to(settings: Settings, config: UserConfig | None = None) -> Settings:
    """在 env 配置之上叠加用户 JSON 配置（JSON 优先级更高）。"""
    if os.getenv(IGNORE_ENV, "").strip() in {"1", "true", "yes"}:
        return settings
    cfg = config or UserConfig.load()
    overrides = settings_overrides(cfg)
    if not overrides:
        return settings
    merged = settings.model_copy(update=overrides)
    logger.debug(
        "用户配置生效：provider=%s model=%s base_url=%s",
        merged.llm_label or merged.llm_kind,
        merged.llm_model,
        merged.llm_base_url,
    )
    return merged
