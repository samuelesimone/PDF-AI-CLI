"""
Tests for the parser layer.

Docling is mocked so these tests run without the actual library installed
and without a real PDF.  We test:
  - The Strategy interface contract (ParsedDocument fields).
  - DoclingParser behaviour when Docling returns content / empty string.
  - That the parser correctly signals latex support.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.parsers.base import BasePDFParser, ParsedDocument
from src.parsers.docling_parser import DoclingParser


# ── ParsedDocument ─────────────────────────────────────────────────────────────

class TestParsedDocument:
    def test_required_text_field(self):
        doc = ParsedDocument(text="hello")
        assert doc.text == "hello"

    def test_metadata_defaults_to_empty_dict(self):
        doc = ParsedDocument(text="hello")
        assert doc.metadata == {}

    def test_metadata_is_set(self):
        doc = ParsedDocument(text="hi", metadata={"page": 3})
        assert doc.metadata["page"] == 3


# ── BasePDFParser (interface contract) ────────────────────────────────────────

class TestBasePDFParserContract:
    def test_subclass_without_name_raises(self):
        class BadParser(BasePDFParser):
            def parse(self, pdf_path: Path):
                return []

        with pytest.raises(TypeError):
            BadParser()  # type: ignore[abstract]

    def test_subclass_without_parse_raises(self):
        class BadParser(BasePDFParser):
            @property
            def name(self):
                return "bad"

        with pytest.raises(TypeError):
            BadParser()  # type: ignore[abstract]

    def test_supports_latex_defaults_false(self):
        class MinimalParser(BasePDFParser):
            @property
            def name(self):
                return "minimal"

            def parse(self, pdf_path: Path):
                return []

        assert MinimalParser().supports_latex is False


# ── DoclingParser ──────────────────────────────────────────────────────────────

class TestDoclingParser:
    def test_name(self):
        assert DoclingParser().name == "docling"

    def test_supports_latex(self):
        assert DoclingParser().supports_latex is True

    # Patch at the source module because DoclingParser uses a lazy import
    # (`from docling.document_converter import DocumentConverter` inside parse()).
    @patch("docling.document_converter.DocumentConverter")
    def test_parse_returns_one_document(self, mock_cls, tmp_path):
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "# Chapter 1\n\nSome text."
        mock_cls.return_value.convert.return_value = mock_result

        pdf = tmp_path / "book.pdf"
        pdf.touch()

        docs = DoclingParser().parse(pdf)

        assert len(docs) == 1
        assert docs[0].text == "# Chapter 1\n\nSome text."

    @patch("docling.document_converter.DocumentConverter")
    def test_metadata_contains_required_keys(self, mock_cls, tmp_path):
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "content"
        mock_cls.return_value.convert.return_value = mock_result

        pdf = tmp_path / "my_book.pdf"
        pdf.touch()

        docs = DoclingParser().parse(pdf)

        meta = docs[0].metadata
        assert meta["doc_id"] == "my_book.pdf"
        assert meta["parser"] == "docling"
        assert meta["has_latex"] is True
        assert "source" in meta

    @patch("docling.document_converter.DocumentConverter")
    def test_empty_document_returns_empty_list(self, mock_cls, tmp_path):
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "   "
        mock_cls.return_value.convert.return_value = mock_result

        pdf = tmp_path / "empty.pdf"
        pdf.touch()

        docs = DoclingParser().parse(pdf)
        assert docs == []

    @patch("docling.document_converter.DocumentConverter")
    def test_converter_called_with_string_path(self, mock_cls, tmp_path):
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "text"
        mock_instance = mock_cls.return_value
        mock_instance.convert.return_value = mock_result

        pdf = tmp_path / "test.pdf"
        pdf.touch()

        DoclingParser().parse(pdf)

        mock_instance.convert.assert_called_once_with(str(pdf))
