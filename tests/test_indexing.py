"""
Tests for RAGIndexer.

llama-index Settings, Ollama, and OllamaEmbedding are all mocked so
these tests run without a real Ollama instance or GPU.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.parsers.base import ParsedDocument


# Helper: create a RAGIndexer with all heavy deps mocked
def _make_indexer(config: Config):
    with (
        patch("src.indexing.indexer.Ollama") as mock_llm_cls,
        patch("src.indexing.indexer.OllamaEmbedding") as mock_emb_cls,
        patch("src.indexing.indexer.Settings"),
        patch("src.indexing.indexer.SentenceSplitter"),
    ):
        from src.indexing.indexer import RAGIndexer
        from src.parsers.base import BasePDFParser

        mock_parser = MagicMock(spec=BasePDFParser)
        indexer = RAGIndexer(config=config, parser=mock_parser)
        return indexer, mock_parser


class TestRAGIndexerCacheKey:
    def test_index_path_contains_stem(self, config, tmp_path):
        indexer, _ = _make_indexer(config)
        pdf = tmp_path / "physics_textbook.pdf"
        path = indexer.index_path(pdf)
        assert "physics_textbook" in path.name

    def test_index_path_includes_storage_dir(self, config, tmp_path):
        indexer, _ = _make_indexer(config)
        pdf = tmp_path / "book.pdf"
        path = indexer.index_path(pdf)
        assert path.parent == config.storage_dir

    def test_different_files_get_different_paths(self, config, tmp_path):
        indexer, _ = _make_indexer(config)
        p1 = indexer.index_path(tmp_path / "a.pdf")
        p2 = indexer.index_path(tmp_path / "b.pdf")
        assert p1 != p2


class TestRAGIndexerIsIndexed:
    def test_returns_false_when_no_index_exists(self, config, tmp_path):
        indexer, _ = _make_indexer(config)
        pdf = tmp_path / "new.pdf"
        assert indexer.is_indexed(pdf) is False

    def test_returns_true_when_marker_file_exists(self, config, tmp_path):
        indexer, _ = _make_indexer(config)
        pdf = tmp_path / "cached.pdf"
        marker = indexer.index_path(pdf) / "docstore.json"
        marker.parent.mkdir(parents=True)
        marker.touch()
        assert indexer.is_indexed(pdf) is True


class TestRAGIndexerDropIndex:
    def test_drop_removes_directory(self, config, tmp_path):
        indexer, _ = _make_indexer(config)
        pdf = tmp_path / "book.pdf"
        idx_dir = indexer.index_path(pdf)
        idx_dir.mkdir(parents=True)
        (idx_dir / "docstore.json").touch()

        indexer.drop_index(pdf)

        assert not idx_dir.exists()

    def test_drop_is_idempotent(self, config, tmp_path):
        indexer, _ = _make_indexer(config)
        pdf = tmp_path / "book.pdf"
        # Should not raise even if there is nothing to drop
        indexer.drop_index(pdf)


class TestRAGIndexerLoadIndex:
    @patch("src.indexing.indexer.StorageContext")
    @patch("src.indexing.indexer.load_index_from_storage")
    def test_returns_loaded_index(self, mock_load, mock_sc, config, tmp_path):
        indexer, _ = _make_indexer(config)
        pdf = tmp_path / "book.pdf"
        mock_load.return_value = MagicMock()

        result = indexer.load_index(pdf)

        mock_load.assert_called_once()
        assert result is mock_load.return_value

    @patch("src.indexing.indexer.StorageContext")
    @patch("src.indexing.indexer.load_index_from_storage")
    def test_uses_correct_persist_dir(self, mock_load, mock_sc, config, tmp_path):
        indexer, _ = _make_indexer(config)
        pdf = tmp_path / "book.pdf"
        mock_load.return_value = MagicMock()

        indexer.load_index(pdf)

        expected_dir = str(indexer.index_path(pdf))
        mock_sc.from_defaults.assert_called_once_with(persist_dir=expected_dir)


class TestRAGIndexerIndexDocuments:
    @patch("src.indexing.indexer.VectorStoreIndex")
    def test_returns_index(self, mock_vi_cls, config, tmp_path):
        indexer, _ = _make_indexer(config)
        pdf = tmp_path / "book.pdf"
        pdf.touch()

        docs = [ParsedDocument(text="chunk", metadata={"doc_id": "book.pdf"})]
        mock_index = MagicMock()
        mock_vi_cls.from_documents.return_value = mock_index

        result = indexer.index_documents(pdf, docs)

        mock_vi_cls.from_documents.assert_called_once()
        assert result is mock_index

    @patch("src.indexing.indexer.VectorStoreIndex")
    def test_persists_to_correct_dir(self, mock_vi_cls, config, tmp_path):
        indexer, _ = _make_indexer(config)
        pdf = tmp_path / "book.pdf"
        pdf.touch()

        mock_index = MagicMock()
        mock_vi_cls.from_documents.return_value = mock_index

        indexer.index_documents(pdf, [ParsedDocument(text="x", metadata={})])

        expected_dir = str(indexer.index_path(pdf))
        mock_index.storage_context.persist.assert_called_once_with(
            persist_dir=expected_dir
        )

    @patch("src.indexing.indexer.VectorStoreIndex")
    def test_converts_parsed_docs_to_documents(self, mock_vi_cls, config, tmp_path):
        indexer, _ = _make_indexer(config)
        pdf = tmp_path / "book.pdf"
        pdf.touch()

        mock_index = MagicMock()
        mock_vi_cls.from_documents.return_value = mock_index

        docs = [
            ParsedDocument(text="first", metadata={"page": 1}),
            ParsedDocument(text="second", metadata={"page": 2}),
        ]
        indexer.index_documents(pdf, docs)

        call_args = mock_vi_cls.from_documents.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0].text == "first"
        assert call_args[1].text == "second"

    @patch("src.indexing.indexer.VectorStoreIndex")
    def test_creates_index_directory(self, mock_vi_cls, config, tmp_path):
        indexer, _ = _make_indexer(config)
        pdf = tmp_path / "book.pdf"
        pdf.touch()

        mock_index = MagicMock()
        mock_vi_cls.from_documents.return_value = mock_index

        indexer.index_documents(pdf, [ParsedDocument(text="x", metadata={})])

        assert indexer.index_path(pdf).exists()


class TestRAGIndexerBuildOrLoad:
    @patch("src.indexing.indexer.StorageContext")
    @patch("src.indexing.indexer.load_index_from_storage")
    def test_loads_from_storage_when_cached(
        self, mock_load, mock_sc, config, tmp_path
    ):
        indexer, _ = _make_indexer(config)
        pdf = tmp_path / "book.pdf"

        # Simulate existing index
        marker = indexer.index_path(pdf) / "docstore.json"
        marker.parent.mkdir(parents=True)
        marker.touch()

        mock_load.return_value = MagicMock()

        result = indexer.build_or_load(pdf)

        mock_load.assert_called_once()
        assert result is mock_load.return_value

    @patch("src.indexing.indexer.VectorStoreIndex")
    def test_builds_index_when_not_cached(self, mock_vi_cls, config, tmp_path):
        indexer, mock_parser = _make_indexer(config)
        pdf = tmp_path / "book.pdf"
        pdf.touch()

        mock_parser.parse.return_value = [
            ParsedDocument(text="Some text", metadata={"doc_id": "book.pdf"})
        ]

        mock_index = MagicMock()
        mock_vi_cls.from_documents.return_value = mock_index

        result = indexer.build_or_load(pdf)

        mock_vi_cls.from_documents.assert_called_once()
        mock_index.storage_context.persist.assert_called_once()
        assert result is mock_index
