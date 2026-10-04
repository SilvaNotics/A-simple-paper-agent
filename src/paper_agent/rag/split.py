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
