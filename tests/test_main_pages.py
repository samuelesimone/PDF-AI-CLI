"""
Tests for the --pages helpers in main.py:
  _parse_page_range  — pure string parser
  _extract_page_range — PyMuPDF extraction (fitz is mocked)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from main import _parse_page_range, _extract_page_range


# ── _parse_page_range ─────────────────────────────────────────────────────────

class TestParsePageRange:
    def test_simple_range(self):
        assert _parse_page_range("1-50") == (1, 50)

    def test_start_equals_end(self):
        assert _parse_page_range("10-10") == (10, 10)

    def test_large_range(self):
        assert _parse_page_range("100-800") == (100, 800)

    def test_missing_dash_raises(self):
        with pytest.raises(ValueError, match="START-END"):
            _parse_page_range("50")

    def test_too_many_parts_raises(self):
        with pytest.raises(ValueError, match="START-END"):
            _parse_page_range("1-10-20")

    def test_start_zero_raises(self):
        with pytest.raises(ValueError, match="START must be"):
            _parse_page_range("0-10")

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="END must be"):
            _parse_page_range("20-5")

    def test_non_numeric_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            _parse_page_range("a-b")


# ── _extract_page_range ───────────────────────────────────────────────────────

def _make_fitz_mock(total_pages: int = 100):
    """Return a mock fitz module with a plausible open() interface."""
    mock_doc = MagicMock()
    mock_doc.__len__ = lambda self: total_pages

    mock_extracted = MagicMock()

    mock_fitz = MagicMock()
    mock_fitz.open.side_effect = [mock_doc, mock_extracted]

    return mock_fitz, mock_doc, mock_extracted


class TestExtractPageRange:
    def test_creates_output_file_in_pages_subdir(self, tmp_path):
        mock_fitz, _, _ = _make_fitz_mock()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            out = _extract_page_range(
                Path("/data/book.pdf"), 1, 50, tmp_path
            )
        assert out == tmp_path / "pages" / "book_p1-50.pdf"

    def test_pages_dir_is_created(self, tmp_path):
        mock_fitz, _, _ = _make_fitz_mock()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            _extract_page_range(Path("/data/book.pdf"), 1, 50, tmp_path)
        assert (tmp_path / "pages").is_dir()

    def test_uses_cache_when_file_exists(self, tmp_path):
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        existing = pages_dir / "book_p1-50.pdf"
        existing.touch()

        mock_fitz = MagicMock()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            out = _extract_page_range(Path("/data/book.pdf"), 1, 50, tmp_path)

        # fitz.open should NOT have been called — we used the cached file
        mock_fitz.open.assert_not_called()
        assert out == existing

    def test_force_reextract_ignores_cache(self, tmp_path):
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        existing = pages_dir / "book_p1-50.pdf"
        existing.touch()

        mock_fitz, mock_doc, mock_extracted = _make_fitz_mock(total_pages=100)
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            _extract_page_range(
                Path("/data/book.pdf"), 1, 50, tmp_path, force=True
            )

        # fitz.open was called → re-extracted
        assert mock_fitz.open.called

    def test_insert_pdf_called_with_correct_pages(self, tmp_path):
        mock_fitz, _, mock_extracted = _make_fitz_mock(total_pages=200)
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            _extract_page_range(Path("/data/book.pdf"), 10, 60, tmp_path)

        # from_page/to_page are 0-based → 10-1=9, 60-1=59
        mock_extracted.insert_pdf.assert_called_once()
        _, kwargs = mock_extracted.insert_pdf.call_args
        assert kwargs["from_page"] == 9
        assert kwargs["to_page"] == 59

    def test_end_clamped_to_document_length(self, tmp_path):
        mock_fitz, _, mock_extracted = _make_fitz_mock(total_pages=30)
        with patch.dict("sys.modules", {"fitz": mock_fitz}), \
             patch("main.console"):
            _extract_page_range(Path("/data/book.pdf"), 1, 100, tmp_path)

        _, kwargs = mock_extracted.insert_pdf.call_args
        # clamped to 30 pages → to_page = 30-1 = 29
        assert kwargs["to_page"] == 29

    def test_output_filename_encodes_range(self, tmp_path):
        mock_fitz, _, _ = _make_fitz_mock()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            out = _extract_page_range(
                Path("/data/my_thesis.pdf"), 100, 200, tmp_path
            )
        assert out.name == "my_thesis_p100-200.pdf"
