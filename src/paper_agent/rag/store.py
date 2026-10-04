# -*- coding: utf-8 -*-
"""向量存储：InMemoryVectorStore + 本地 JSON 持久化（零额外依赖）。

- `InMemoryVectorStore.dump/load` 是 langchain-core 自带能力，直接复用；
- 另外维护 `manifest.json`，记录每篇论文的 chunk id 列表，支持增量更新与删除；
- 需要在更大语料上换 FAISS/Qdrant 时，只要替换本类内部实现即可。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from ..core.config import Settings, get_settings
from ..core.schema import Paper

logger = logging.getLogger(__name__)

BATCH_SIZE = 10  # 默认批次；上限由 settings.embed_batch_size 覆盖（DashScope 实测 ≤ 10）


def _sha256(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


class IndexSignatureError(RuntimeError):
    """索引与当前 embedding 配置不匹配（维度/模型不同）。"""


def embedding_signature(embeddings: Embeddings, settings: Settings | None = None) -> str:
    """给 embedding 生成稳定签名，用于防止“用 A 模型建索引、用 B 模型检索”。"""
    s = settings or get_settings()
    base = getattr(embeddings, "base", embeddings)
    model = getattr(embeddings, "model", None) or getattr(base, "model", None)
    dim = getattr(embeddings, "dim", None) or getattr(base, "dim", None) or s.embedding_dim
    return f"{model or s.embedding_model}:{dim}"


class PaperIndex:
    """论文语料的向量索引（内存 + JSON 落盘）。"""

    def __init__(self, embeddings: Embeddings, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.batch_size = max(1, min(BATCH_SIZE, self.settings.embed_batch_size))
        self.embeddings = embeddings
        self.store = InMemoryVectorStore(embedding=embeddings)
        self.signature = embedding_signature(embeddings, self.settings)
        self.manifest: dict = {
            "embedding_model": self.settings.embedding_model,
            "embedding_signature": self.signature,
            "papers": {},
        }

    # ---------------- 生命周期 ----------------
    @classmethod
    def load_or_create(cls, embeddings: Embeddings, settings: Settings | None = None) -> "PaperIndex":
        s = settings or get_settings()
        s.ensure_dirs()
        index = cls(embeddings, s)
        if not s.index_file.exists():
            return index

        # 先校验签名，避免“用 A 模型建索引、用 B 模型检索”导致的维度错乱
        manifest: dict = {}
        if s.manifest_file.exists():
            try:
                manifest = json.loads(s.manifest_file.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("manifest 解析失败，将重建：%s", exc)
        stored = manifest.get("embedding_signature")
        if stored and stored != index.signature:
            raise IndexSignatureError(
                f"索引是用 {stored} 建的，当前是 {index.signature}。"
                "维度不同无法混用：请换 data_dir（例如 PAPER_AGENT_DATA_DIR=data/offline）"
                "或删除索引后重新 ingest。"
            )

        try:
            index.store = InMemoryVectorStore.load(str(s.index_file), embeddings)
            index.manifest = manifest or index.manifest
            logger.info(
                "已加载索引：%d 篇论文 / %d chunks", len(index.paper_ids()), index.chunk_count
            )
            return index
        except Exception as exc:  # noqa: BLE001 - 索引损坏则重建
            logger.warning("索引加载失败，将重建：%s", exc)
            index.store = InMemoryVectorStore(embedding=embeddings)
            index.manifest = {
                "embedding_model": s.embedding_model,
                "embedding_signature": index.signature,
                "papers": {},
            }
            return index

    def save(self) -> None:
        s = self.settings
        s.ensure_dirs()
        tmp = s.index_file.with_suffix(".json.tmp")
        self.store.dump(str(tmp))
        tmp.replace(s.index_file)
        s.manifest_file.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---------------- 写入 ----------------
    def add_documents(self, docs: Iterable[Document], paper: Paper | None = None, pdf: Path | None = None) -> int:
        docs = [d for d in docs if d.page_content.strip()]
        if not docs:
            return 0

        # 统一 id：paper_id::page::chunk_index（便于删除与引用定位）
        ids = [
            f"{d.metadata.get('paper_id', 'unknown')}::{d.metadata.get('page', 0)}::{d.metadata.get('chunk_index', i)}"
            for i, d in enumerate(docs)
        ]
        for batch_start in range(0, len(docs), self.batch_size):
            self._add_batch(
                docs[batch_start : batch_start + self.batch_size],
                ids[batch_start : batch_start + self.batch_size],
            )

        pid = (paper.paper_id if paper else docs[0].metadata.get("paper_id", "")) or "unknown"
        entry = self.manifest["papers"].get(pid, {})
        entry.update(
            {
                "title": paper.title if paper else docs[0].metadata.get("title", ""),
                "n_chunks": int(entry.get("n_chunks", 0)) + len(docs),
                "ids": sorted(set(entry.get("ids", [])) | set(ids)),
                "pdf": str(pdf) if pdf else entry.get("pdf", ""),
                "sha256": _sha256(pdf) or entry.get("sha256", ""),
            }
        )
        if paper is not None:
            # 供 simple 模式/报告直接生成 BibTeX 与参考文献
            entry.update(
                {
                    "authors": paper.authors,
                    "published": paper.published,
                    "doi": paper.doi,
                    "url": paper.url or paper.pdf_url,
                    "pdf_url": paper.pdf_url,
                    "source": paper.source,
                }
            )
        self.manifest["papers"][pid] = entry
        self.save()
        return len(docs)

    def _add_batch(self, docs: list[Document], ids: list[str]) -> None:
        """写入一批；遇到“批次过大”类 400 错误时二分重试（不同 embedding 服务上限不同）。"""
        try:
            self.store.add_documents(docs, ids=ids)
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            too_large = ("batch size" in message) or ("larger than" in message)
            if not too_large or len(docs) <= 1:
                raise
            mid = len(docs) // 2
            logger.warning("embedding 批次过大，自动二分重试（%d -> %d + %d）", len(docs), mid, len(docs) - mid)
            self._add_batch(docs[:mid], ids[:mid])
            self._add_batch(docs[mid:], ids[mid:])

    def clear_paper(self, paper_id: str) -> bool:
        """删除某篇论文的全部 chunk（增量重建用）。"""
        entry = self.manifest["papers"].get(paper_id)
        if not entry:
            return False
        self.store.delete(ids=list(entry.get("ids", [])))
        self.manifest["papers"].pop(paper_id, None)
        self.save()
        return True

    def delete_paper(self, paper_id: str, remove_pdf: bool = True) -> bool:
        """从索引中彻底删除一篇论文（交互式 `/papers rm` 用）。

        与 `clear_paper` 的区别：除了 chunks + manifest 条目，还会（默认）删掉
        `data/papers/<paper_id>.pdf` 缓存，避免下次 ingest 命中旧文件。
        """
        entry = self.manifest["papers"].get(paper_id)
        if not entry:
            return False

        pdf_path = entry.get("pdf") or ""
        if not pdf_path:
            from .fetch import local_pdf_path

            pdf_path = str(local_pdf_path(Paper(paper_id=paper_id), self.settings))

        self.store.delete(ids=list(entry.get("ids", [])))
        self.manifest["papers"].pop(paper_id, None)
        self.save()

        if remove_pdf and pdf_path:
            try:
                path = Path(pdf_path)
                if path.exists():
                    path.unlink()
            except OSError as exc:  # pragma: no cover - 删除失败不影响索引一致性
                logger.warning("删除 PDF 缓存失败（%s）：%s", pdf_path, exc)
        return True

    # ---------------- 查询 ----------------
    @property
    def chunk_count(self) -> int:
        return len(getattr(self.store, "store", {}) or {})

    def paper_ids(self) -> set[str]:
        return set(self.manifest.get("papers", {}).keys())

    def has_paper(self, paper_id: str) -> bool:
        return paper_id in self.manifest.get("papers", {})

    def list_papers(self) -> list[dict]:
        return [
            {"paper_id": pid, **{k: v for k, v in info.items() if k != "ids"}}
            for pid, info in sorted(self.manifest.get("papers", {}).items())
        ]

    def search(
        self,
        query: str,
        k: int = 6,
        paper_ids: Iterable[str] | None = None,
    ) -> list[tuple[Document, float]]:
        """相似度检索；`paper_ids` 用于把检索限制在指定论文内。"""
        allowed = set(paper_ids) if paper_ids else None
        flt = (lambda d: d.metadata.get("paper_id") in allowed) if allowed else None
        if self.chunk_count == 0:
            return []
        k = max(1, min(k, self.chunk_count))
        if flt is None:
            return self.store.similarity_search_with_score(query, k=k)
        # 先按过滤条件取全量候选再截断，避免 filter 后为空
        return self.store.similarity_search_with_score(query, k=k, filter=flt)

    def get_chunk(self, chunk_id: str) -> Document | None:
        try:
            docs = self.store.get_by_ids([chunk_id])
        except Exception:  # noqa: BLE001
            return None
        return docs[0] if docs else None
