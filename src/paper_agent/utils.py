# -*- coding: utf-8 -*-
"""通用工具：论文 ID 归一化/去重、MCP 返回值解析、slug、离线假 embedding。

这些函数刻意保持纯函数、无 IO，方便单测。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Sequence

from langchain_core.embeddings import Embeddings

# --------------------------------------------------------------------------
# 论文 ID
# --------------------------------------------------------------------------

_ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
_ARXIV_INLINE_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?(?!\d)")
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+")
_DOI_START_RE = re.compile(r"^10\.\d{4,9}/")
_DOI_PREFIX_RE = re.compile(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)", re.IGNORECASE)
_OPENALEX_RE = re.compile(r"(?:openalex\.org/)?(W\d{6,})", re.IGNORECASE)


def normalize_paper_id(raw: str | None) -> str:
    """把各种写法的论文标识归一化成稳定 key。

    例：
        "2405.16506v3" / "arXiv:2405.16506" / "https://arxiv.org/abs/2405.16506v2"
            -> "arxiv:2405.16506"
        "10.1145/1234.5678" / "https://doi.org/10.1145/1234.5678"
            -> "doi:10.1145/1234.5678"
    """
    if not raw:
        return ""
    text = str(raw).strip()
    stripped = _DOI_PREFIX_RE.sub("", text)

    # 1) DOI（以 10.xxxx/ 开头最可靠，优先判定，避免与 arXiv 的 YYMM.NNNNN 混淆）
    if _DOI_START_RE.match(stripped):
        # arXiv 自身的 DOI（DataCite）统一归到 arxiv:，方便与 arXiv 结果去重
        m_arxiv_doi = re.match(
            r"^10\.48550/(?:arxiv\.)?(\d{4}\.\d{4,5})", stripped, re.IGNORECASE
        )
        if m_arxiv_doi:
            return "arxiv:" + m_arxiv_doi.group(1)
        m = _DOI_RE.match(stripped)
        if m:
            return "doi:" + m.group(0).rstrip(".").lower()

    # 2) arXiv：带 arxiv 前缀，或整体就是 YYMM.NNNNN(vN)
    if "arxiv" in text.lower():
        m = _ARXIV_INLINE_RE.search(text)
        if m:
            return "arxiv:" + m.group(1)
    m = _ARXIV_ID_RE.match(stripped)
    if m:
        return "arxiv:" + m.group(1)

    # 3) 其他位置的 DOI
    m = _DOI_RE.search(stripped)
    if m:
        return "doi:" + m.group(0).rstrip(".").lower()

    # 4) OpenAlex work id（如 W7171728473 / https://openalex.org/W7171728473）
    m = _OPENALEX_RE.search(text)
    if m:
        return "openalex:" + m.group(1).lower()

    # 5) 兜底：小写化去掉空白
    return re.sub(r"\s+", "", text).lower()


# --------------------------------------------------------------------------
# 去重
# --------------------------------------------------------------------------


def dedupe_papers(papers: Sequence[Any], key: str = "paper_id") -> list[Any]:
    """按归一化后的论文 ID 去重；后出现的条目用于补齐前者的空缺字段。

    支持传入 pydantic 模型或 dict。
    """
    out: dict[str, Any] = {}

    def _get(obj: Any, field: str) -> Any:
        return getattr(obj, field, None) if not isinstance(obj, dict) else obj.get(field)

    def _set(obj: Any, field: str, value: Any) -> None:
        if isinstance(obj, dict):
            obj[field] = value
        else:
            setattr(obj, field, value)

    for p in papers:
        pid = normalize_paper_id(_get(p, key) or _get(p, "doi") or _get(p, "url"))
        if not pid:
            continue
        if pid not in out:
            _set(p, key, pid)
            out[pid] = p
            continue
        prev = out[pid]
        # 用新条目补齐旧条目的空字段（如后一个来源提供了 pdf_url）
        for field in ("title", "authors", "abstract", "pdf_url", "url", "doi", "published", "source"):
            if hasattr(prev, "model_fields") or isinstance(prev, dict):
                old = _get(prev, field)
                new = _get(p, field)
                if (old in (None, "", [], {})) and (new not in (None, "", [], {})):
                    _set(prev, field, new)
    return list(out.values())


# --------------------------------------------------------------------------
# 粘贴清洗（终端会把粘贴内容包在括号粘贴标记里，需要剥掉）
# --------------------------------------------------------------------------

# CSI（含 \x1b[200~ / \x1b[201~ 括号粘贴标记、颜色、光标控制）、OSC、其他转义
_ESCAPE_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"       # CSI 序列
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC 序列
    r"|\x1b[()][A-Za-z0-9]"             # 字符集切换
    r"|\x1b[=>NOP]"                     # 其它单字符转义
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2028\u2029\ufeff]")
_QUOTES = "\"'\u201c\u201d\u2018\u2019"


def clean_pasted(text: str | None) -> str:
    """清理终端粘贴带来的杂质：括号粘贴标记/转义序列、控制字符、零宽字符、外层引号。

    典型场景：终端把 `sk-xxx` 粘贴成 `\x1b[200~sk-xxx\x1b[201~`，
    直接拿去请求就会 401；这里统一剥干净。多行粘贴会合并（只保留第一行内容）。
    """
    if not text:
        return ""
    value = _ESCAPE_RE.sub("", str(text))
    value = _ZERO_WIDTH_RE.sub("", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _CONTROL_RE.sub("", value)
    lines = [ln.strip() for ln in value.split("\n") if ln.strip()]
    value = lines[0] if lines else ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTES:
        value = value[1:-1].strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value


def clean_secret(text: str | None) -> str:
    """清洗密钥：在 `clean_pasted` 基础上去掉所有空白（粘贴常带首尾/中间空格）。"""
    return re.sub(r"\s+", "", clean_pasted(text))


def mask_secret(secret: str | None, head: int = 4, tail: int = 4) -> str:
    """脱敏展示密钥（用于确认是否粘贴正确）。"""
    value = secret or ""
    if not value:
        return "(空)"
    if len(value) <= head + tail:
        return "*" * len(value)
    return f"{value[:head]}…{value[-tail:]}（{len(value)} 位）"


# --------------------------------------------------------------------------
# 文本处理
# --------------------------------------------------------------------------

_SLUG_BAD = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")


def slugify(text: str, max_len: int = 60) -> str:
    """生成安全的文件名片段（保留中文）。"""
    slug = _SLUG_BAD.sub("-", (text or "").strip()).strip("-")
    return (slug or "untitled")[:max_len]


def truncate(text: str | None, limit: int = 400) -> str:
    if not text:
        return ""
    text = str(text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------
# MCP 返回值解析
# --------------------------------------------------------------------------


def mcp_result_to_text(result: Any) -> str:
    """把 MCP 工具返回值（content blocks / str / dict）拍平成文本。

    langchain-mcp-adapters 把 MCP content blocks 原样返回，形如：
        [{"type": "text", "text": "{...json...}"}]
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if isinstance(result.get("text"), str):
            return result["text"]
        return json.dumps(result, ensure_ascii=False)
    if isinstance(result, (list, tuple)):
        parts: list[str] = []
        for item in result:
            parts.append(mcp_result_to_text(item))
        return "\n".join(p for p in parts if p)
    return str(result)


def extract_json(text: str) -> Any | None:
    """从模型/MCP 输出里尽力抽取第一个 JSON 对象或数组。"""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    return None


def first_list(value: Any, *keys: str) -> list:
    """从 dict 里按候选 key 找第一个 list。"""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for k in keys:
            if isinstance(value.get(k), list):
                return value[k]
    return []


# --------------------------------------------------------------------------
# 离线假 embedding（测试用，确定性）
# --------------------------------------------------------------------------


class DeterministicFakeEmbeddings(Embeddings):
    """基于 sha256 的确定性 embedding：同样的文本永远得到同样的向量。

    仅用于离线测试；语义无关，但足以验证存储/检索/持久化链路。
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.model = f"fake-{dim}"   # 供索引签名使用

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        raw = (digest * (self.dim // len(digest) + 1))[: self.dim]
        vec = [(b - 127.5) / 127.5 for b in raw]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    # Embeddings 接口
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
