# -*- coding: utf-8 -*-
"""报告渲染：Markdown 正文 + BibTeX + 结构化 JSON。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import Settings, get_settings
from .schema import Answer, Citation, Paper, PaperSummary
from .utils import slugify, truncate


def _authors_bibtex(authors: str) -> str:
    if not authors:
        return "Unknown"
    return authors.replace("; ", " and ")


def build_bibtex(papers: list[Paper]) -> str:
    """按论文元数据生成 BibTeX（arXiv 用 @misc + eprint，其余用 @article）。"""
    entries: list[str] = []
    for p in papers:
        key = (p.paper_id or p.title or "ref").replace("arxiv:", "").replace("doi:", "").replace("/", "_")
        key = "".join(ch for ch in key if ch.isalnum() or ch in "._-") or "ref"
        year = (p.published or "")[:4]
        if p.paper_id.startswith("arxiv:"):
            entries.append(
                "@misc{%s,\n  title = {%s},\n  author = {%s},\n  year = {%s},\n"
                "  eprint = {%s},\n  archivePrefix = {arXiv},\n  url = {%s}\n}"
                % (
                    key,
                    p.title,
                    _authors_bibtex(p.authors),
                    year,
                    p.paper_id.split(":", 1)[1],
                    p.url or p.pdf_url,
                )
            )
        else:
            entries.append(
                "@article{%s,\n  title = {%s},\n  author = {%s},\n  year = {%s},\n  doi = {%s},\n  url = {%s}\n}"
                % (
                    key,
                    p.title,
                    _authors_bibtex(p.authors),
                    year,
                    p.doi or p.paper_id.split(":", 1)[-1],
                    p.url or p.pdf_url,
                )
            )
    return "\n\n".join(entries) + ("\n" if entries else "")


def render_report(
    topic: str,
    papers: list[Paper],
    summaries: list[PaperSummary],
    answers: list[Answer],
    citations: dict[str, Citation],
    narrative: dict[str, str],
    search_queries: list[str] | None = None,
    notes: list[str] | None = None,
    issues: list[str] | None = None,
) -> str:
    """组装最终 Markdown 报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [f"# 学术调研报告：{topic}", ""]
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 论文数量：{len(summaries)} 篇（检索候选 {len(papers)} 篇）")
    if search_queries:
        lines.append("- 检索式：" + "；".join(f"`{q}`" for q in search_queries))
    if notes:
        lines.append("- 过程记录：" + "；".join(truncate(n, 100) for n in notes if n))

    if narrative.get("executive_summary"):
        lines += ["", "## 1. 核心结论", "", narrative["executive_summary"].strip()]

    lines += ["", "## 2. 论文对比", ""]
    if summaries:
        lines.append("| # | 论文 | 年份 | 方法要点 | 结论要点 | 局限 |")
        lines.append("|---|---|---|---|---|---|")
        for i, s in enumerate(summaries, 1):
            year = next((p.published[:4] for p in papers if p.paper_id == s.paper_id), "")
            lines.append(
                "| %d | %s | %s | %s | %s | %s |"
                % (
                    i,
                    f"`{s.paper_id}` {truncate(s.title, 60)}",
                    year,
                    truncate(s.method.replace("\n", " "), 110),
                    truncate(s.findings.replace("\n", " "), 110),
                    truncate(s.limitations.replace("\n", " "), 90),
                )
            )
    else:
        lines.append("（未能生成逐篇摘要）")

    if narrative.get("comparison"):
        lines += ["", "### 2.1 深入对比", "", narrative["comparison"].strip()]

    lines += ["", "## 3. 逐篇摘要", ""]
    for i, s in enumerate(summaries, 1):
        lines.append(f"### 3.{i} `{s.paper_id}` — {s.title}")
        lines.append(
            f"- **研究问题**：{s.problem}\n- **方法**：{s.method}\n- **数据/实验**：{s.data}"
            f"\n- **结论**：{s.findings}\n- **局限**：{s.limitations}"
            f"\n- **可复用点**：{s.reusable_ideas}\n- **置信度**：{s.confidence:.2f}"
            f"（全文片段 {s.retrieved_chunks} 段）"
        )
        if s.key_quotes:
            lines.append("- **原文关键句**：")
            for q in s.key_quotes:
                lines.append(f"  - “{truncate(q.text, 220)}”{('（' + q.locator + '）') if q.locator else ''}")
        lines.append("")

    if answers:
        lines += ["## 4. 关键问题与证据", ""]
        for a in answers:
            lines.append(f"### {a.question}")
            lines.append(a.text.strip() or "（无回答）")
            if a.unsupported:
                lines.append(f"\n> ⚠️ 引用校验提示：{'; '.join(a.unsupported)}")
            lines.append("")

    if narrative.get("gaps"):
        lines += ["## 5. 局限与研究空白", "", narrative["gaps"].strip()]
    if narrative.get("conclusion"):
        lines += ["", "## 6. 实践建议", "", narrative["conclusion"].strip()]

    lines += ["", "## 7. 参考文献", ""]
    for i, p in enumerate(sorted(papers, key=lambda x: x.published or ""), 1):
        link = p.url or p.pdf_url or ""
        lines.append(f"{i}. `{p.paper_id}` {p.title} — {p.authors or 'Unknown'} ({p.published[:10]}) {link}")

    if citations:
        lines += ["", "## 附录 A. 引用锚点", ""]
        for cid, cite in sorted(citations.items()):
            loc = f" p.{cite.page}" if cite.page else ""
            lines.append(f"- `{cid}` → `{cite.paper_id}`{loc}：{truncate(cite.snippet, 160)}")

    if issues:
        lines += ["", "## 附录 B. 校验与告警", ""]
        lines += [f"- {msg}" for msg in issues]

    lines.append("")
    return "\n".join(lines)


def write_outputs(
    topic: str,
    markdown: str,
    bibtex: str,
    payload: dict,
    settings: Settings | None = None,
) -> dict[str, Path]:
    """把报告写到 `output/<时间戳>-<slug>.{md,bib,json}`。"""
    s = settings or get_settings()
    s.ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    base = f"{stamp}-{slugify(topic)}"

    md_path = s.output_path / f"{base}.md"
    bib_path = s.output_path / f"{base}.bib"
    json_path = s.output_path / f"{base}.json"

    md_path.write_text(markdown, encoding="utf-8")
    bib_path.write_text(bibtex, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    return {"markdown": md_path, "bibtex": bib_path, "json": json_path}
