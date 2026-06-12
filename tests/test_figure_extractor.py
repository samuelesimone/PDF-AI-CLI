"""
Tests for src/figures/extractor.py.

Pure-geometry helpers are tested directly.
extract_figures() is tested with a mocked fitz module so no real PDF or
PyMuPDF installation is required.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.figures.extractor import (
    ExtractedFigure,
    _is_valid_region,
    _mostly_contained,
    extract_figures,
)


# ── Minimal rect stand-in ─────────────────────────────────────────────────────

class _R:
    """Lightweight Rect-like object for testing pure-geometry functions."""
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = float(x0), float(y0)
        self.x1, self.y1 = float(x1), float(y1)
        self.width  = float(x1 - x0)
        self.height = float(y1 - y0)


# ── _is_valid_region ──────────────────────────────────────────────────────────

class TestIsValidRegion:
    def test_typical_figure_passes(self):
        assert _is_valid_region(_R(0, 0, 300, 200), page_area=500_000)

    def test_too_narrow_rejected(self):
        # width=30 < _MIN_SIDE=50
        assert not _is_valid_region(_R(0, 0, 30, 200), page_area=500_000)

    def test_too_short_rejected(self):
        # height=30 < _MIN_SIDE=50
        assert not _is_valid_region(_R(0, 0, 200, 30), page_area=500_000)

    def test_too_small_area_rejected(self):
        # area = 60*60 = 3600 < _MIN_AREA=7200
        assert not _is_valid_region(_R(0, 0, 60, 60), page_area=500_000)

    def test_exactly_min_area_passes(self):
        # area = 90*80 = 7200 == _MIN_AREA
        assert _is_valid_region(_R(0, 0, 90, 80), page_area=500_000)

    def test_full_page_rejected(self):
        # area=250_000 = 86% of 290_000 → > _MAX_PAGE_FRACTION (0.85)
        assert not _is_valid_region(_R(0, 0, 500, 500), page_area=290_000)

    def test_large_but_not_full_page_passes(self):
        # area=200_000 = 50% of 400_000
        assert _is_valid_region(_R(0, 0, 500, 400), page_area=400_000)

    def test_zero_page_area_does_not_crash(self):
        # Should not raise; page_area=0 skips the fraction check
        assert _is_valid_region(_R(0, 0, 200, 200), page_area=0)


# ── _mostly_contained ─────────────────────────────────────────────────────────

class TestMostlyContained:
    def test_fully_inside(self):
        inner = _R(10, 10, 90, 90)
        outer = _R(0, 0, 100, 100)
        assert _mostly_contained(inner, outer)

    def test_no_overlap(self):
        inner = _R(200, 200, 300, 300)
        outer = _R(0, 0, 100, 100)
        assert not _mostly_contained(inner, outer)

    def test_partial_overlap_below_threshold(self):
        # Intersection = 20×100 = 2000; inner area = 100×100 = 10000 → 20%
        inner = _R(80, 0, 180, 100)
        outer = _R(0, 0, 100, 100)
        assert not _mostly_contained(inner, outer)

    def test_partial_overlap_above_threshold(self):
        # inner fully covered by larger outer → 100%
        inner = _R(0, 0, 100, 100)
        outer = _R(0, 0, 200, 200)
        assert _mostly_contained(inner, outer)

    def test_exactly_threshold(self):
        # Intersection = 80×100 = 8000; inner = 100×100 = 10000 → 80% == threshold
        inner = _R(20, 0, 120, 100)
        outer = _R(0, 0, 100, 100)
        assert _mostly_contained(inner, outer, threshold=0.80)

    def test_zero_area_inner_does_not_crash(self):
        inner = _R(5, 5, 5, 5)  # zero area
        outer = _R(0, 0, 100, 100)
        assert not _mostly_contained(inner, outer)


# ── extract_figures integration (fitz mocked) ─────────────────────────────────

def _make_pixmap(w=400, h=300):
    pix = MagicMock()
    pix.width  = w
    pix.height = h
    pix.tobytes = MagicMock(return_value=b"\x89PNG\r\n\x1a\n")
    return pix


def _build_fitz_mock(clusters=None, raster_images=None,
                     page_w=595.0, page_h=842.0):
    """Return (mock_fitz, mock_doc, mock_page)."""
    mock_page = MagicMock()
    mock_page.rect = _R(0, 0, page_w, page_h)
    mock_page.cluster_drawings.return_value = clusters or []
    mock_page.get_images.return_value       = raster_images or []
    mock_page.get_image_rects.return_value  = []
    mock_page.get_pixmap.return_value       = _make_pixmap()

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(side_effect=lambda: iter([mock_page]))
    mock_doc.extract_image.return_value = {}

    mock_fitz = MagicMock()
    mock_fitz.open.return_value      = mock_doc
    mock_fitz.Matrix.return_value    = MagicMock()
    # Rect() passes the object through unchanged (our _R already has all attrs)
    mock_fitz.Rect.side_effect       = lambda r: r

    return mock_fitz, mock_doc, mock_page


class TestExtractFigures:
    def test_returns_list(self, tmp_path):
        pdf = tmp_path / "a.pdf"
        pdf.touch()
        mock_fitz, _, _ = _build_fitz_mock()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = extract_figures(pdf)
        assert isinstance(result, list)

    def test_no_figures_returns_empty(self, tmp_path):
        pdf = tmp_path / "empty.pdf"
        pdf.touch()
        mock_fitz, _, _ = _build_fitz_mock(clusters=[], raster_images=[])
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = extract_figures(pdf)
        assert result == []

    def test_vector_figure_extracted(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        large_rect = _R(50, 50, 350, 300)  # 300×250 pt → area 75 000
        mock_fitz, _, _ = _build_fitz_mock(clusters=[large_rect])
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = extract_figures(pdf)
        assert len(result) == 1
        assert result[0].source == "vector"

    def test_vector_figure_page_number(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        mock_fitz, _, _ = _build_fitz_mock(clusters=[_R(0, 0, 300, 200)])
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = extract_figures(pdf)
        assert result[0].page == 1  # first page, 1-based

    def test_small_cluster_filtered(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        tiny = _R(0, 0, 40, 40)  # area=1600 < _MIN_AREA
        mock_fitz, _, _ = _build_fitz_mock(clusters=[tiny])
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = extract_figures(pdf)
        assert result == []

    def test_full_page_cluster_filtered(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        # 595×842 ≈ full A4 page → exceeds _MAX_PAGE_FRACTION
        full_page = _R(0, 0, 595, 842)
        mock_fitz, _, _ = _build_fitz_mock(clusters=[full_page])
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = extract_figures(pdf)
        assert result == []

    def test_figure_image_bytes_populated(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        mock_fitz, _, _ = _build_fitz_mock(clusters=[_R(0, 0, 300, 200)])
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = extract_figures(pdf)
        assert result[0].image_bytes == b"\x89PNG\r\n\x1a\n"

    def test_raster_image_extracted(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        # img_info tuple: (xref, smask, w, h, bpc, cs, alt_cs, name, filter, ref)
        img_info = (5, 0, 400, 300, 8, "RGB", "", "Im5", "DCT", 0)
        large_rect = _R(50, 50, 450, 350)

        mock_fitz, mock_doc, mock_page = _build_fitz_mock(
            clusters=[], raster_images=[img_info]
        )
        mock_page.get_image_rects.return_value = [large_rect]
        mock_doc.extract_image.return_value = {
            "image": b"\xff\xd8\xff",  # JPEG magic bytes
            "ext": "jpeg",
        }
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = extract_figures(pdf)
        assert len(result) == 1
        assert result[0].source == "raster"

    def test_raster_in_vector_region_deduplicated(self, tmp_path):
        """A raster image mostly covered by a vector region should be skipped."""
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        vector_rect = _R(0, 0, 400, 300)
        img_info = (7, 0, 400, 300, 8, "RGB", "", "Im7", "DCT", 0)
        # Same rect as vector → 100% contained
        raster_rect = _R(0, 0, 400, 300)

        mock_fitz, mock_doc, mock_page = _build_fitz_mock(
            clusters=[vector_rect], raster_images=[img_info]
        )
        mock_page.get_image_rects.return_value = [raster_rect]
        mock_doc.extract_image.return_value = {"image": b"x", "ext": "png"}

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = extract_figures(pdf)

        # Only the vector figure should appear; raster is deduplicated
        assert len(result) == 1
        assert result[0].source == "vector"

    def test_sequential_index_across_figures(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        rects = [_R(0, 0, 300, 200), _R(0, 300, 300, 500)]
        mock_fitz, _, _ = _build_fitz_mock(clusters=rects)
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = extract_figures(pdf)
        assert result[0].index == 0
        assert result[1].index == 1
