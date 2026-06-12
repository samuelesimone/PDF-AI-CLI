"""
RAGIndexer — builds or loads a persisted VectorStoreIndex for a given PDF.

Responsibilities:
- Initialise the LLM and embedding model via llama-index Settings.
- Delegate PDF parsing to a BasePDFParser (Strategy pattern).
- Persist the index to disk so re-runs don't re-embed the same file.
- Provide a stable cache key (PDF filename + short MD5 hash).
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from ..config import Config
from ..parsers.base import BasePDFParser


class RAGIndexer:
    """Builds and caches a llama-index VectorStoreIndex for a PDF file."""

    def __init__(self, config: Config, parser: BasePDFParser) -> None:
        self.config = config
        self.parser = parser
        self._configure_settings()

    # ── llama-index global settings ──────────────────────────────────────────

    def _configure_settings(self) -> None:
        """Push LLM, embedding model, and splitter into llama-index Settings."""
        Settings.llm = Ollama(
            model=self.config.llm_model,
            base_url=self.config.ollama_base_url,
            request_timeout=self.config.request_timeout,
            temperature=self.config.temperature,
            # Keep the model loaded in Ollama for the whole session.
            # Without this Ollama unloads after 5 min, causing the next
            # query to wait for a cold-start (and often hit a ReadTimeout).
            keep_alive="30m",
            # Disable qwen3 extended thinking unless explicitly requested;
            # thinking adds latency and the <think> blocks are stripped anyway.
            thinking=self.config.enable_thinking,
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=self.config.embed_model,
            base_url=self.config.ollama_base_url,
        )
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    # ── cache helpers ────────────────────────────────────────────────────────

    def index_path(self, pdf_path: Path) -> Path:
        """Return the on-disk directory that stores this PDF's index."""
        digest = hashlib.md5(str(pdf_path.resolve()).encode()).hexdigest()[:8]
        return self.config.storage_dir / f"{pdf_path.stem}_{digest}"

    def is_indexed(self, pdf_path: Path) -> bool:
        """Return True if a persisted index already exists for *pdf_path*."""
        marker = self.index_path(pdf_path) / "docstore.json"
        return marker.exists()

    def drop_index(self, pdf_path: Path) -> None:
        """Delete a cached index so the next call re-indexes from scratch."""
        idx = self.index_path(pdf_path)
        if idx.exists():
            shutil.rmtree(idx)

    # ── main entry points ────────────────────────────────────────────────────

    def load_index(self, pdf_path: Path) -> VectorStoreIndex:
        """Load a previously persisted index from disk."""
        idx_dir = self.index_path(pdf_path)
        storage_context = StorageContext.from_defaults(persist_dir=str(idx_dir))
        return load_index_from_storage(storage_context)

    def index_documents(
        self,
        pdf_path: Path,
        parsed_docs: List,
    ) -> VectorStoreIndex:
        """Embed *parsed_docs* and persist a new vector index for *pdf_path*.

        Args:
            pdf_path:    Path to the source PDF (used to derive the storage key).
            parsed_docs: List of :class:`~src.parsers.base.ParsedDocument` objects
                to embed.  This can include text chunks, figure descriptions, or
                any other pre-parsed content.
        """
        idx_dir = self.index_path(pdf_path)
        documents: List[Document] = [
            Document(text=p.text, metadata=p.metadata) for p in parsed_docs
        ]
        index = VectorStoreIndex.from_documents(documents)
        idx_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(idx_dir))
        return index

    def build_or_load(
        self,
        pdf_path: Path,
        extra_docs: Optional[List] = None,
    ) -> VectorStoreIndex:
        """Return an index for *pdf_path*, building it first if necessary.

        This is a convenience wrapper used by callers that do not need per-step
        progress reporting.  For interactive use with progress feedback, call
        :meth:`load_index` or :meth:`index_documents` directly.

        Args:
            pdf_path:   Path to the PDF file.
            extra_docs: Optional additional :class:`~src.parsers.base.ParsedDocument`
                objects (e.g. figure descriptions from FigureAnalyzer) to
                include when building the index.  Ignored when loading a
                cached index.
        """
        if self.is_indexed(pdf_path):
            return self.load_index(pdf_path)

        parsed = list(self.parser.parse(pdf_path))
        if extra_docs:
            parsed.extend(extra_docs)
        return self.index_documents(pdf_path, parsed)
