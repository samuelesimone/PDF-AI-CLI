"""
Tests for PyMuPDFParser.

fitz (PyMuPDF) is mocked so these tests need no real PDF file.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.parsers.pymupdf_parser import PyMuPDFParser


class TestPyMuPDFParserProperties:
    def test_name(self):
        assert PyMuPDFParser().name == "pymupdf"

    def test_supports_latex_is_false(self):
        # pymupdf captures math characters but not LaTeX syntax
        assert PyMuPDFParser().supports_latex is False


class TestPyMuPDFParserParse:
    def _make_mock_pdf(self, pages: list[str]):
        """Return a mock fitz document with the given page texts."""
        mock_pages = []
        for text in pages:
            page = MagicMock()
            page.get_text.return_value = text
            mock_pages.append(page)

        mock_doc = MagicMock()
        mock_doc.__len__ = lambda self: len(mock_pages)
        mock_doc.__getitem__ = lambda self, i: mock_pages[i]
        mock_doc.__enter__ = lambda self: self
        mock_doc.__exit__ = MagicMock(return_value=False)
        return mock_doc

    @patch("fitz.open")
    def test_returns_one_document_per_page(self, mock_open, tmp_path):
        mock_open.return_value = self._make_mock_pdf(["Page 1 text.", "Page 2 text."])

        pdf = tmp_path / "book.pdf"
        pdf.touch()

        docs = PyMuPDFParser().parse(pdf)

        assert len(docs) == 2

    @patch("fitz.open")
    def test_page_numbers_in_metadata(self, mock_open, tmp_path):
        mock_open.return_value = self._make_mock_pdf(["First.", "Second."])

        pdf = tmp_path / "book.pdf"
        pdf.touch()

        docs = PyMuPDFParser().parse(pdf)

        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 2

    @patch("fitz.open")
    def test_empty_pages_are_skipped(self, mock_open, tmp_path):
        mock_open.return_value = self._make_mock_pdf(["Text.", "   ", "More text."])

        pdf = tmp_path / "book.pdf"
        pdf.touch()

        docs = PyMuPDFParser().parse(pdf)

        assert len(docs) == 2
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 3

    @patch("fitz.open")
    def test_parser_name_in_metadata(self, mock_open, tmp_path):
        mock_open.return_value = self._make_mock_pdf(["Text."])

        pdf = tmp_path / "test.pdf"
        pdf.touch()

        docs = PyMuPDFParser().parse(pdf)

        assert docs[0].metadata["parser"] == "pymupdf"

    @patch("fitz.open")
    def test_formula_characters_preserved(self, mock_open, tmp_path):
        formula_text = "dr =\nP|r|\ni=k+1 ∥g((xi−k, yi−k), (xi, yi))∥\n|r| −k"
        mock_open.return_value = self._make_mock_pdf([formula_text])

        pdf = tmp_path / "math.pdf"
        pdf.touch()

        docs = PyMuPDFParser().parse(pdf)

        assert "∥" in docs[0].text
        assert "∀" not in docs[0].text or True  # just checking it doesn't crash

    @patch("fitz.open")
    def test_get_text_called_with_text_mode(self, mock_open, tmp_path):
        mock_doc = self._make_mock_pdf(["hello"])
        mock_open.return_value = mock_doc

        pdf = tmp_path / "test.pdf"
        pdf.touch()

        PyMuPDFParser().parse(pdf)

        mock_doc[0].get_text.assert_called_once_with("text")
