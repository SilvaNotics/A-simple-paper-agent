# -*- coding: utf-8 -*-
"""搜索渠道注册表（`/channels` 的元数据层）。

设计目标：**像 `/connect` 配置模型供应商一样配置搜索渠道**——
每个渠道就是一条记录（kind + base_url + api_key/email + enabled），
用户可自主添加/删除，密钥只写进 `~/.config/paper-agent/config.json`。

本模块只放「元数据与默认值」，真正的联网检索实现在 `sources.py`
（那里统一走带重试的 `_request`）。新增渠道的步骤：
1. 在这里加一条 `ChannelSpec`；
2. 在 `sources.py` 实现同名的 `search_<kind>()` 并登记进 `CHANNEL_SEARCHERS`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# 渠道分组：academic=学术库，cn=国内数据库，web=通用网页搜索（结果可能是网页而非论文）
GROUP_ACADEMIC = "academic"
GROUP_CN = "cn"
GROUP_WEB = "web"


@dataclass(frozen=True)
class ChannelSpec:
    """一个搜索渠道的元数据。"""

    kind: str
    label: str
    base_url: str = ""
    description: str = ""
    needs_key: bool = False
    needs_email: bool = False
    group: str = GROUP_ACADEMIC
    # 免 key 的公开源（仍需 `/channels add` 手动启用）
    builtin: bool = False


REGISTRY: dict[str, ChannelSpec] = {
    # ---------------- 免 key 的公开学术源（仍需 /channels add 手动启用） ----------------
    "arxiv": ChannelSpec(
        kind="arxiv",
        label="arXiv",
        base_url="https://export.arxiv.org/api/query",
        description="预印本检索（Atom API，免 key）",
        builtin=True,
    ),
    "openalex": ChannelSpec(
        kind="openalex",
        label="OpenAlex",
        base_url="https://api.openalex.org/works",
        description="开放学术图谱，含 OA PDF 直链（免 key；填邮箱进 polite pool）",
        needs_email=False,
        builtin=True,
    ),
    "crossref": ChannelSpec(
        kind="crossref",
        label="Crossref",
        base_url="https://api.crossref.org/works",
        description="DOI 元数据兜底（免 key）",
        builtin=True,
    ),
    "europepmc": ChannelSpec(
        kind="europepmc",
        label="Europe PMC",
        base_url="https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        description="生命科学/医学文献，含 OA 全文链接（免 key）",
        builtin=True,
    ),
    "pubmed": ChannelSpec(
        kind="pubmed",
        label="PubMed",
        base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        description="NCBI PubMed（免 key；有 key 可提高配额）",
        needs_key=False,
        needs_email=False,
        builtin=True,
    ),
    "doaj": ChannelSpec(
        kind="doaj",
        label="DOAJ",
        base_url="https://doaj.org/api/search/articles",
        description="开放获取期刊目录（免 key）",
        builtin=True,
    ),
    # ---------------- 需要 API key 的学术源 ----------------
    "semanticscholar": ChannelSpec(
        kind="semanticscholar",
        label="Semantic Scholar",
        base_url="https://api.semanticscholar.org/graph/v1/paper/search",
        description="语义检索 + 引用量（无 key 会被 429 限流，需配 key）",
        needs_key=True,
        builtin=False,
    ),
    "core": ChannelSpec(
        kind="core",
        label="CORE",
        base_url="https://api.core.ac.uk/v3/search/works",
        description="开放获取全文聚合（需要 CORE API key）",
        needs_key=True,
        builtin=False,
    ),
    # ---------------- 国内数据库（中文文献） ----------------
    # 能力边界（不要假装都有接口）：
    # - ChinaXiv：有公开 API（chinarxiv.org），免 key，可选 email 进 polite pool；
    # - 国家图书馆："图书馆检索"联合目录（meta.nlc.cn）有公开 JSON 接口，免 key，限图书/古籍/学位论文书目；
    # - 百度学术：有百度千帆官方 API，需 Bearer key（每日免费额度）；
    # - 万方：有开放平台 API，需 APPCODE/AppKey，按点计费；
    # - 知网/维普/超星：**没有公开检索 API**，本项目不做绕过验证码的抓取（见 README 已知限制）。
    "chinaxiv": ChannelSpec(
        kind="chinaxiv",
        label="ChinaXiv 预印本",
        base_url="https://chinarxiv.org/api/v1/papers",
        description="中文预印本（中国科学院 ChinaXiv 语料；公开 API，免 key）",
        group=GROUP_CN,
        builtin=True,
    ),
    "baidu_scholar": ChannelSpec(
        kind="baidu_scholar",
        label="百度学术",
        base_url="https://qianfan.baidubce.com/v2/tools/baidu_scholar/search",
        description="中英文期刊/会议/学位论文（百度千帆官方 API，需 Bearer key，每日有免费额度）",
        needs_key=True,
        group=GROUP_CN,
    ),
    "wanfang": ChannelSpec(
        kind="wanfang",
        label="万方数据",
        base_url="https://api.wanfangdata.com.cn/search",
        description="中文期刊/学位/会议论文（万方开放平台，需 APPCODE 或 AppKey:APPCODE，按点计费）",
        needs_key=True,
        group=GROUP_CN,
    ),
    "nlc": ChannelSpec(
        kind="nlc",
        label="国家图书馆（图书馆检索）",
        base_url="https://meta.nlc.cn/v2/doSearch",
        description="国家图书馆联合目录：图书/古籍/学位论文等书目+馆藏（公开接口，免 key）",
        group=GROUP_CN,
        builtin=True,
    ),
    # ---------------- 通用网页搜索（需要 API key） ----------------
    "tavily": ChannelSpec(
        kind="tavily",
        label="Tavily",
        base_url="https://api.tavily.com/search",
        description="网页搜索 API（需要 Tavily key）",
        needs_key=True,
        group=GROUP_WEB,
    ),
    "exa": ChannelSpec(
        kind="exa",
        label="Exa",
        base_url="https://api.exa.ai/search",
        description="语义网页搜索（需要 Exa key）",
        needs_key=True,
        group=GROUP_WEB,
    ),
    "serpapi": ChannelSpec(
        kind="serpapi",
        label="SerpAPI (Google Scholar)",
        base_url="https://serpapi.com/search.json",
        description="Google Scholar 结果（需要 SerpAPI key；合规风险自负）",
        needs_key=True,
        group=GROUP_WEB,
    ),
}

# `/channels add` 的预设编号（按注册表顺序，便于交互选择）
PRESET_ORDER: list[str] = [
    "arxiv",
    "openalex",
    "crossref",
    "europepmc",
    "pubmed",
    "doaj",
    "chinaxiv",
    "nlc",
    "semanticscholar",
    "core",
    "baidu_scholar",
    "wanfang",
    "tavily",
    "exa",
    "serpapi",
]


def spec_for(kind: str) -> ChannelSpec | None:
    return REGISTRY.get(str(kind or "").strip().lower())


def is_domestic(kind: str) -> bool:
    """是否国内数据库渠道（`GROUP_CN`）。"""
    spec = spec_for(kind)
    return bool(spec and spec.group == GROUP_CN)


def domestic_first(kinds: Iterable[str]) -> list[str]:
    """稳定排序：国内渠道在前，其余保持原顺序。"""
    items = [str(k or "").strip().lower() for k in kinds]
    return [k for k in items if is_domestic(k)] + [k for k in items if not is_domestic(k)]


def channel_label(kind: str, fallback: str = "") -> str:
    spec = spec_for(kind)
    return spec.label if spec else (fallback or kind)


def list_specs(group: str = "") -> list[ChannelSpec]:
    items = list(REGISTRY.values())
    if group:
        items = [s for s in items if s.group == group]
    return items


def presets() -> list[tuple[str, ChannelSpec]]:
    """返回 (编号, spec)，编号从 1 开始，用于交互式添加。"""
    out: list[tuple[str, ChannelSpec]] = []
    for i, kind in enumerate(PRESET_ORDER, 1):
        spec = REGISTRY.get(kind)
        if spec is not None:
            out.append((str(i), spec))
    return out


def default_base_url(kind: str) -> str:
    spec = spec_for(kind)
    return spec.base_url if spec else ""
