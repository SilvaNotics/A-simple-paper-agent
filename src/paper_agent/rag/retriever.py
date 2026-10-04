# -*- coding: utf-8 -*-
"""检索与引用：把 chunk 变成带锚点的上下文，并校验回答里的引用是否真有原文支撑。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from ..schema import Answer, Citation
from ..utils import truncate
from .store import PaperIndex

logger = logging.getLogger(__name__)

_CJK = re.compile(r"[\u4e00-\u9fff]")
_BRACKET = re.compile(r"\[[^\]]*\]")  # 引用锚点/编号，参与比对只会引入噪声
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{3,}")
_STOP = {
    "this", "that", "with", "from", "have", "were", "which", "there", "their", "these",
    "those", "than", "then", "them", "they", "been", "being", "also", "such", "using",
    "used", "when", "where", "what", "into", "more", "most", "some", "only", "over",
    "each", "both", "does", "about", "between", "however", "therefore", "thus",
}


@dataclass
class CitationCollector:
    """跨多次检索累积引用锚点；编号 `{prefix}C1..Cn` 全局唯一。

    并行分支（Send）传入不同 prefix（如 `S1-`、`Q2-`），合并到图状态时不会碰撞。
    """

    citations: dict[str, Citation] = field(default_factory=dict)
    # 完整 chunk 文本（引用校验用；citations[cid].snippet 是截断后的展示文本）
    chunks: dict[str, str] = field(default_factory=dict)
    prefix: str = ""
    _counter: int = 0

    def add_document(self, doc: Document) -> str:
        self._counter += 1
        cid = f"{self.prefix}C{self._counter}"
        self.citations[cid] = Citation(
            id=cid,
            paper_id=doc.metadata.get("paper_id", ""),
            page=doc.metadata.get("page"),
            snippet=truncate(doc.page_content, 240),
        )
        self.chunks[cid] = doc.page_content
        return cid

    def add_documents(self, docs: list[Document]) -> list[str]:
        return [self.add_document(d) for d in docs]


def format_context(docs: list[Document], collector: CitationCollector) -> str:
    """把检索片段渲染成带 `[C#]` 锚点的上下文。"""
    blocks: list[str] = []
    for doc, cid in zip(docs, collector.add_documents(docs)):
        meta = doc.metadata
        header = f"[{cid}] {meta.get('paper_id', '?')}"
        if meta.get("page"):
            header += f" p.{meta['page']}"
        if meta.get("title"):
            header += f" — {truncate(meta['title'], 80)}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------
# 引用校验
# --------------------------------------------------------------------------


def _normalize_word(word: str) -> str:
    """极轻量词形归一（只为提高引用校验的容错，不追求语言正确性）。"""
    word = word.lower()
    for suffix in ("ations", "ation", "ments", "ment", "ings", "ing", "ness", "ally", "ers", "er", "ly", "ed", "es", "al", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> set[str]:
    """英文词（≥4 字母，做轻量词形归一）+ 中文二元组，用于粗略的支撑度比较。"""
    text = _BRACKET.sub(" ", text or "")
    tokens = {_normalize_word(w) for w in _WORD.findall(text)}
    tokens -= _STOP
    cjk = "".join(ch for ch in text if _CJK.match(ch))
    tokens |= {cjk[i : i + 2] for i in range(len(cjk) - 1)}
    return tokens


def support_ratio(sentence: str, chunk_text: str) -> float:
    """句子里的实词有多大比例能在原文片段中找到。"""
    s_tokens = tokenize(sentence)
    if not s_tokens:
        return 1.0
    c_tokens = tokenize(chunk_text)
    return len(s_tokens & c_tokens) / len(s_tokens)


def has_anchor(sentence: str, chunk_text: str, min_len: int = 6) -> bool:
    """是否存在一个较长的实词命中（防“共享全是一堆通用词”式的假支撑）。"""
    s_tokens = {t for t in tokenize(sentence) if len(t) >= min_len and not _CJK.match(t)}
    if not s_tokens:
        return True  # 中文/短句不适用该规则
    return bool(s_tokens & tokenize(chunk_text))


_HARD_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{4,}|\d+(?:\.\d+)?")


def hard_tokens(text: str) -> set[str]:
    """“硬证据令牌”：拉丁术语（≥5 字母）、缩写、数字（含小数）。

    用途：中文回答引用英文原文时，字面重合度意义不大，但方法名 / 数据集名 /
    数字（如 `HotpotQA`、`GRAG`、`7.1`）通常会被原样保留，可用作跨语言支撑。
    """
    tokens: set[str] = set()
    for raw in _HARD_TOKEN.findall(_BRACKET.sub(" ", text or "")):
        tokens.add(raw if raw[0].isdigit() else _normalize_word(raw))
    return tokens - _STOP


def cross_lingual_support(sentence: str, chunk_text: str, min_hit: float = 0.34) -> bool | None:
    """跨语言支撑判定。

    Returns:
        True/False：可判定；None：句子没有任何硬证据令牌，无法自动核验。
    """
    hard = hard_tokens(sentence)
    if not hard:
        return None
    hits = len(hard & hard_tokens(chunk_text)) / len(hard)
    return hits >= min_hit


_ANCHOR_ONLY = re.compile(r"^(?:\[[^\]]+\]|[\s.,;:。，、；：（）()\-—])*$")


def sentences_with_citation(text: str, cid: str) -> list[str]:
    """找出包含指定引用锚点的句子。

    注意：模型习惯把锚点写在句号之后（`...提升 7.1 个点 [C1]。` 或 `... [C1]`），
    按标点切句会让锚点单独成段，所以要把“只有锚点”的片段并回上一句。
    """
    pattern = re.compile(rf"\[{re.escape(cid)}\]")
    parts = re.split(r"(?<=[。！？.!?])\s+|\n+", text or "")

    merged: list[str] = []
    for part in parts:
        stripped = part.strip()
        if merged and stripped and _ANCHOR_ONLY.match(stripped):
            merged[-1] = f"{merged[-1]} {stripped}"
        else:
            merged.append(part)

    return [p.strip() for p in merged if pattern.search(p)]


def normalize_answer_citations(answer: Answer, citations: dict[str, Citation]) -> Answer:
    """就地修复模型“省略前缀”的引用编号。

    实测：上下文里的锚点形如 `Q0-C1`，模型倾向于在正文里写成 `[C1]`。
    这里在编号唯一可推断时自动补回前缀，避免把正确引用误判为“不在检索结果中”。
    """
    if not citations:
        return answer

    reverse: dict[str, set[str]] = {}
    for cid in citations:
        reverse.setdefault(cid.split("-")[-1], set()).add(cid)

    def resolve(cid: str) -> str:
        if cid in citations:
            return cid
        candidates = reverse.get(cid)
        if candidates and len(candidates) == 1:
            return next(iter(candidates))
        return cid

    fixed_text = answer.text
    for raw in list(answer.citation_ids) + anchors_in(answer.text):
        fixed = resolve(raw)
        if fixed != raw:
            fixed_text = re.sub(rf"\[{re.escape(raw)}\]", f"[{fixed}]", fixed_text)

    fixed_ids: list[str] = []
    for cid in answer.citation_ids or anchors_in(fixed_text):
        fixed = resolve(cid)
        if fixed not in fixed_ids:
            fixed_ids.append(fixed)

    answer.text = fixed_text
    answer.citation_ids = fixed_ids
    return answer


def verify_answer(
    answer: Answer,
    citations: dict[str, Citation],
    chunk_lookup: dict[str, str] | None = None,
    min_ratio: float = 0.3,
) -> list[str]:
    """校验引用。

    规则：
    1. 引用的编号必须来自本次检索；
    2. 每个引用编号都要在正文里出现；
    3. 引用所在句子与原文片段的实词重合度需 >= min_ratio。

    Returns:
        问题列表，空列表表示通过。
    """
    problems: list[str] = []
    if not answer.text.strip():
        problems.append("回答为空")
        return problems

    if not answer.citation_ids:
        problems.append("没有任何引用")

    for cid in answer.citation_ids:
        if cid not in citations:
            problems.append(f"引用 {cid} 不在本次检索结果中")
            continue

        cited_sentences = sentences_with_citation(answer.text, cid)
        if not cited_sentences:
            problems.append(f"引用 {cid} 未在正文中标注")
            continue

        supported = False
        unverifiable = True
        best = 0.0

        for sentence in cited_sentences:
            # 一句里可能同时标注多个来源（[C3][C4]）；只要其中一个能支撑该句即算通过，
            # 这与人类引用多来源的习惯一致，也能减少误报。
            co_cited = [x for x in anchors_in(sentence) if x in citations] or [cid]
            sentence_supported = False
            sentence_unknown = False
            for other in co_cited:
                source = (chunk_lookup or {}).get(other, citations[other].snippet)
                ratio = support_ratio(sentence, source)
                if other == cid:
                    best = max(best, ratio)
                if ratio >= min_ratio and has_anchor(sentence, source):
                    sentence_supported = True
                    break
                cross = cross_lingual_support(sentence, source)
                if cross is True:
                    sentence_supported = True
                    break
                if cross is None:
                    sentence_unknown = True
            if sentence_supported:
                supported = True
                break
            if not sentence_unknown:
                unverifiable = False

        if supported:
            continue
        if unverifiable:
            # 纯中文表述引用英文原文且没有可核验的术语/数字：不报错，仅记录
            logger.debug("引用 %s 无法自动核验（缺少可对比的术语/数字）", cid)
            continue
        problems.append(f"引用 {cid} 缺少原文支撑（重合度 {best:.2f}）")

    return problems


# --------------------------------------------------------------------------
# 报告级引用校验
# --------------------------------------------------------------------------

_ANCHOR_RE = re.compile(r"\[([A-Za-z0-9\-]*C\d+)\]")


def anchors_in(text: str) -> list[str]:
    """找出文本里的所有 [C#] / [S1-C2] 形式锚点（去重、保序）。"""
    seen: list[str] = []
    for match in _ANCHOR_RE.finditer(text or ""):
        cid = match.group(1)
        if cid not in seen:
            seen.append(cid)
    return seen


def verify_report_citations(markdown: str, citations: dict[str, Citation]) -> list[str]:
    """校验最终报告：锚点必须存在；有检索证据时报告必须标注引用。"""
    problems: list[str] = []
    anchors = anchors_in(markdown)
    unknown = [a for a in anchors if a not in citations]
    if unknown:
        problems.append(f"报告引用了不存在的锚点：{', '.join(unknown[:5])}")
    if citations and not anchors:
        problems.append("检索到了全文证据，但报告正文没有任何引用锚点")
    return problems


# --------------------------------------------------------------------------
# 检索入口
# --------------------------------------------------------------------------


def retrieve_across_papers(
    index: PaperIndex,
    query: str,
    paper_ids: list[str] | None,
    settings=None,
    per_paper_min: int = 2,
) -> list[Document]:
    """跨论文检索：每篇论文保底给若干片段，避免证据全落在同一篇上。

    跨篇归纳若引用全来自同一篇论文，结论会严重偏置；这里按论文配额检索后去重合并，
    总片段数上限 = top_k + 论文数。
    """
    from ..config import get_settings

    s = settings or get_settings()
    if not paper_ids:
        return retrieve(index, query, k=s.top_k, hybrid=s.hybrid_retrieval)

    per_paper = max(per_paper_min, s.top_k // max(1, len(paper_ids)))
    merged: list[Document] = []
    seen: set[str] = set()
    for pid in paper_ids:
        for doc in retrieve(index, query, k=per_paper, paper_ids=[pid], hybrid=s.hybrid_retrieval):
            key = doc.id or (
                f"{doc.metadata.get('paper_id')}::{doc.metadata.get('page')}::{doc.page_content[:40]}"
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
    return merged[: max(s.top_k, per_paper_min * len(paper_ids))]


def retrieve(
    index: PaperIndex,
    query: str,
    k: int = 6,
    paper_ids: list[str] | None = None,
    hybrid: bool = False,
) -> list[Document]:
    """向量检索（可选 BM25 混合）。"""
    hits = index.search(query, k=k, paper_ids=paper_ids)
    docs = [d for d, _ in hits]

    if not hybrid or not docs:
        return docs

    try:
        return _hybrid_rerank(index, query, docs, k=k, paper_ids=paper_ids)
    except ImportError:
        logger.warning("未安装 rank-bm25，跳过混合检索（pip install rank-bm25 可启用）")
        return docs
    except Exception as exc:  # noqa: BLE001
        logger.warning("混合检索失败，退回纯向量检索：%s", exc)
        return docs


def _hybrid_rerank(
    index: PaperIndex,
    query: str,
    vector_docs: list[Document],
    k: int,
    paper_ids: list[str] | None,
    alpha: float = 0.6,
) -> list[Document]:
    """向量分数 + BM25 分数线性加权（不用 EnsembleRetriever，便于按 paper_id 过滤）。"""
    from langchain_community.retrievers import BM25Retriever

    pool = list((getattr(index.store, "store", {}) or {}).values())
    docs_pool = [
        Document(id=item["id"], page_content=item["text"], metadata=item["metadata"])
        for item in pool
        if not paper_ids or item["metadata"].get("paper_id") in set(paper_ids)
    ]
    if not docs_pool:
        return vector_docs

    bm25 = BM25Retriever.from_documents(docs_pool, k=min(len(docs_pool), max(k * 3, 10)))
    bm25_hits = bm25.invoke(query)
    bm25_rank = {d.page_content: 1.0 / (1 + i) for i, d in enumerate(bm25_hits)}
    vec_rank = {d.page_content: 1.0 / (1 + i) for i, d in enumerate(vector_docs)}

    merged: dict[str, Document] = {d.page_content: d for d in vector_docs}
    for d in bm25_hits:
        merged.setdefault(d.page_content, d)

    scored = sorted(
        merged.values(),
        key=lambda d: alpha * vec_rank.get(d.page_content, 0.0)
        + (1 - alpha) * bm25_rank.get(d.page_content, 0.0),
        reverse=True,
    )
    return scored[:k]
