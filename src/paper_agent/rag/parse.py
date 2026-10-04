# -*- coding: utf-8 -*-
"""PDF 解析：优先 pymupdf（若安装），否则 pdfplumber。

`ParsedDoc.pages` 保留页码，便于引用定位。入口有两个：`parse_pdf()`（磁盘路径）与
`parse_pdf_bytes()`（内存字节）——后者供 `/quick` 使用，**不写任何临时文件**。
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t\u00a0]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_REFERENCES_HEAD = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|参考文献)\s*$", re.IGNORECASE | re.MULTILINE
)


@dataclass
class ParsedDoc:
    pages: list[str] = field(default_factory=list)
    engine: str = ""
    truncated_references: bool = False
    # 从 PDF 里猜出的标题（仅作「元数据缺失」时的兜底，可能为空）
    title: str = ""

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)

    @property
    def n_pages(self) -> int:
        return len(self.pages)


_ARXIV_STAMP_RE = re.compile(r"^arxiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE)
_TITLE_BAD_START_RE = re.compile(
    r"^(?:doi\b|https?://|www\.|\u00a9|copyright\b|preprint\b|proceedings\b|published\b)",
    re.IGNORECASE,
)


def _plausible_title(candidate: str) -> str:
    """判断一段文本能不能当标题（过滤水印行 / DOI 行 / 文件名 / 参考文献行）。"""
    text = _clean(candidate or "").strip(" \t\u00b7\u2022-\u2014")
    if not 8 <= len(text) <= 300:
        return ""
    if _ARXIV_STAMP_RE.match(text) or _TITLE_BAD_START_RE.match(text):
        return ""
    if text.lower().endswith(".pdf") or _looks_like_reference(text):
        return ""
    letters = sum(1 for ch in text if ch.isalpha())
    if letters < 6 or letters < len(text) * 0.5:
        return ""
    return text


def guess_title(first_page: str, meta_title: str = "") -> str:
    """尽力从 PDF 里猜标题：优先 PDF 内嵌元数据，其次首页第一行「像标题」的文本。

    只用于元数据缺失时的兜底（如 arXiv 元数据 API 限流、只能靠直链抓 PDF 的情况）：
    猜不出就返回空串——宁缺勿错，不能把正文首句或 arXiv 水印当成标题。
    """
    meta = _plausible_title(meta_title)
    if meta:
        return meta
    for line in (first_page or "").split("\n"):
        candidate = _plausible_title(line)
        if candidate:
            return candidate
    return ""


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # 修复跨行连字符断词
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _MULTI_NEWLINE.sub("\n\n", text).strip()


def _drop_repeated_margins(pages: list[str], min_ratio: float = 0.5) -> list[str]:
    """去掉在多页重复出现的首/尾行（页眉页脚）。"""
    if len(pages) < 3:
        return pages

    def first_lines(p: str) -> str:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        return lines[0] if lines else ""

    def last_lines(p: str) -> str:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        return lines[-1] if lines else ""

    firsts = [first_lines(p) for p in pages]
    lasts = [last_lines(p) for p in pages]
    threshold = max(3, int(len(pages) * min_ratio))

    def common(values: list[str]) -> set[str]:
        counts: dict[str, int] = {}
        for v in values:
            if v:
                counts[v] = counts.get(v, 0) + 1
        return {v for v, c in counts.items() if c >= threshold}

    drop_first = common(firsts)
    drop_last = common(lasts)

    out: list[str] = []
    for page in pages:
        lines = [ln for ln in page.split("\n")]
        if lines and lines[0].strip() in drop_first:
            lines = lines[1:]
        if lines and lines[-1].strip() in drop_last:
            lines = lines[:-1]
        out.append("\n".join(lines).strip())
    return out


_REF_LIKE = re.compile(
    r"(^\s*\[?\d{1,3}\]?[\.\)]?\s)|(\b(?:19|20)\d{2}\b)|" 
    r"(et\s+al\.)|(\bdoi:|\barXiv:|\bpp\.|\bvol\.|\bIn:\s)",
    re.IGNORECASE,
)


def _looks_like_reference(line: str) -> bool:
    line = line.strip()
    if len(line) < 12:
        return False
    return bool(_REF_LIKE.search(line))


def _cut_references(pages: list[str]) -> tuple[list[str], bool]:
    """从最后一次出现的「References」标题处截断（正文分析通常不需要参考文献列表）。

    保守策略（宁可少切，不可切错）：
    - 标题在最后一页：直接截断；
    - 标题在倒数几页：仅当标题之后的内容“看起来像文献表”（≥50% 行匹配
      编号/年份/et al./doi 等特征）时才截断；
    - 其他情况不动。
    """
    for idx in range(len(pages) - 1, -1, -1):
        matches = list(_REFERENCES_HEAD.finditer(pages[idx]))
        if not matches:
            continue
        start = matches[-1].start()
        tail_text = "\n".join([pages[idx][start:], *pages[idx + 1 :]])
        tail_lines = [ln for ln in tail_text.split("\n") if ln.strip()][1:]  # 去掉标题行

        if idx == len(pages) - 1:
            keep = True
        elif tail_lines:
            keep = sum(_looks_like_reference(ln) for ln in tail_lines) / len(tail_lines) >= 0.5
        else:
            keep = False

        if keep:
            return pages[:idx] + [pages[idx][:start].strip()], True
        return pages, False
    return pages, False


def _parse_with_pymupdf(source: "Path | bytes") -> tuple[list[str], str] | None:
    try:
        import pymupdf  # type: ignore  # 新版包名
    except Exception:  # noqa: BLE001
        try:
            import fitz as pymupdf  # type: ignore  # 旧别名，会打 deprecation 警告
        except Exception:  # noqa: BLE001 - 未安装则回退 pdfplumber
            return None
    try:
        # 字节流走 stream=（`/quick` 不落盘）；路径原样交给 pymupdf
        doc = (
            pymupdf.open(stream=bytes(source), filetype="pdf")
            if isinstance(source, (bytes, bytearray, memoryview))
            else pymupdf.open(source)
        )
        with doc:
            pages = [doc.load_page(i).get_text("text") for i in range(doc.page_count)]
            meta_title = str((doc.metadata or {}).get("title") or "")
            return pages, meta_title
    except Exception as exc:  # noqa: BLE001
        logger.warning("pymupdf 解析失败，回退 pdfplumber：%s", exc)
        return None


def _parse_with_pdfplumber(source: "Path | bytes") -> list[str]:
    import pdfplumber

    target = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray, memoryview)) else source
    pages: list[str] = []
    with pdfplumber.open(target) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def _finish_pages(pages: list[str], meta_title: str, engine: str, cut_references: bool) -> ParsedDoc:
    """共用收尾：清洗 → 去页眉页脚 → 猜标题 → 截参考文献。"""
    pages = [_clean(p) for p in pages]
    pages = [p for p in pages if p.strip()]
    pages = _drop_repeated_margins(pages)
    title = guess_title(pages[0] if pages else "", meta_title)

    truncated = False
    if cut_references:
        pages, truncated = _cut_references(pages)
        pages = [p for p in pages if p.strip()]

    return ParsedDoc(pages=pages, engine=engine, truncated_references=truncated, title=title)


def _parse_source(source: "Path | bytes", cut_references: bool = True) -> ParsedDoc:
    parsed = _parse_with_pymupdf(source)
    engine = "pymupdf"
    meta_title = ""
    if parsed is None:
        pages = _parse_with_pdfplumber(source)
        engine = "pdfplumber"
    else:
        pages, meta_title = parsed
    return _finish_pages(pages, meta_title, engine, cut_references)


def parse_pdf(path: str | Path, cut_references: bool = True) -> ParsedDoc:
    """解析磁盘上的 PDF 为逐页文本（并尽力猜一个标题，见 `guess_title`）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 不存在：{path}")
    return _parse_source(path, cut_references)


def parse_pdf_bytes(content: bytes, cut_references: bool = True) -> ParsedDoc:
    """解析**内存里**的 PDF 字节，不写临时文件（`/quick` 的“不落盘”就靠它）。

    与 `parse_pdf()` 共用同一套清洗/去页眉/截参考文献逻辑，因此两者输出一致。
    """
    if not content:
        raise ValueError("PDF 内容为空")
    return _parse_source(bytes(content), cut_references)
