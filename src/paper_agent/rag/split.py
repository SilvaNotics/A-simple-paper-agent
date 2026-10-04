# -*- coding: utf-8 -*-
"""切分：把逐页文本切成带元数据的 chunk。"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.config import Settings, get_settings
from ..core.schema import Paper
from .parse import ParsedDoc

# 中英混排论文的切分优先级
SEPARATORS = ["\n\n", "\n", "。", "；", ". ", "; ", "! ", "? ", " ", ""]


def build_splitter(settings: Settings | None = None) -> RecursiveCharacterTextSplitter:
    s = settings or get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        keep_separator=True,
    )


def split_text_document(
    paper: Paper,
    text: str,
    settings: Settings | None = None,
    kind: str = "web",
    url: str = "",
) -> list[Document]:
    """把整段纯文本（网页正文/摘要/题录）切成 chunk。

    与 `split_paper` 的区别：没有页码，改为用 `kind` 标记内容级别
    （`web` 网页正文 / `abstract` 仅摘要 / `metadata` 仅题录），
    供引用渲染与「这是不是全文证据」的判断使用。
    """
    splitter = build_splitter(settings)
    docs: list[Document] = []
    for chunk_index, chunk in enumerate(splitter.split_text(text or "")):
        if not chunk.strip():
            continue
        metadata: dict[str, object] = {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "chunk_index": chunk_index,
            "source": paper.source or "local",
            "kind": kind,
        }
        if url:
            metadata["url"] = url
        docs.append(Document(page_content=chunk.strip(), metadata=metadata))
    return docs


def build_record_text(paper: Paper, kind: str = "metadata") -> str:
    """只有题录/摘要时，构造一段明确标注来源级别的文本。

    开头的 `[仅摘要…]` / `[仅题录…]` 会进入 chunk 内容，提醒模型（与用户）
    这**不是全文证据**，避免把书目信息当成论文结论引用。
    """
    label = {
        "abstract": "仅摘要（未获取全文）",
        "metadata": "仅题录（书目/图书等没有可抓取全文）",
    }.get(kind, kind)
    lines = [f"[{label}]"]
    if paper.title:
        lines.append(f"题名：{paper.title}")
    if paper.authors:
        lines.append(f"作者：{paper.authors}")
    if paper.published:
        lines.append(f"出版年：{paper.published}")
    if paper.source:
        lines.append(f"来源渠道：{paper.source}")
    if paper.doi:
        lines.append(f"DOI：{paper.doi}")
    if paper.categories:
        lines.append(f"分类/主题：{paper.categories}")
    extra = paper.extra or {}
    for key, name in (
        ("publisher", "出版社"),
        ("place", "出版地"),
        ("isbn", "ISBN"),
        ("language", "语言"),
        ("series", "丛编"),
    ):
        value = extra.get(key)
        if value:
            rendered = "；".join(str(x) for x in value) if isinstance(value, (list, tuple)) else str(value)
            lines.append(f"{name}：{rendered}")
    if paper.url:
        lines.append(f"链接：{paper.url}")
    if paper.abstract:
        lines.append(f"\n摘要：{paper.abstract}")
    lines.append("\n说明：以上为书目/摘要级信息，本库未获取该文献全文，引用时请注明。")
    return "\n".join(lines)


def split_paper(
    paper: Paper,
    parsed: ParsedDoc,
    settings: Settings | None = None,
) -> list[Document]:
    """把一篇论文切分成 chunk，元数据里保留 paper_id / 页码。"""
    splitter = build_splitter(settings)
    docs: list[Document] = []

    for page_no, page_text in enumerate(parsed.pages, start=1):
        if not page_text.strip():
            continue
        for chunk_index, chunk in enumerate(splitter.split_text(page_text)):
            if not chunk.strip():
                continue
            docs.append(
                Document(
                    page_content=chunk.strip(),
                    metadata={
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "page": page_no,
                        "chunk_index": chunk_index,
                        "source": paper.source or "local",
                    },
                )
            )
    return docs
