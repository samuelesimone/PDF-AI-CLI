"""
Tests for MarkerParser.

marker-pdf is mocked so these tests need no real PDF file and no
downloaded models.  We patch at the source modules because all three
imports are lazy (inside parse()).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.parsers.marker_parser import MarkerParser


# ── Properties ─────────────────────────────────────────────────────────────────

class TestMarkerParserProperties:
    def test_name(self):
        assert MarkerParser().name == "marker"

    def test_supports_latex_is_true(self):
        assert MarkerParser().supports_latex is True


# ── Parse behaviour ───────────────────────────────────────────────────────────

class TestMarkerParserParse:
    """
    Patch order matches decorator order (innermost decorator → first argument):
      @patch("marker.output.text_from_rendered")   → mock_tfr    (3rd arg)
      @patch("marker.models.create_model_dict")     → mock_models (2nd arg)
      @patch("marker.converters.pdf.PdfConverter")  → mock_cls    (1st arg)
    """

    @patch("marker.output.text_from_rendered")
    @patch("marker.models.create_model_dict")
    @patch("marker.converters.pdf.PdfConverter")
    def test_returns_one_document(self, mock_cls, mock_models, mock_tfr, tmp_path):
        mock_tfr.return_value = ("# Title\n\nSome text.", "md", {})

        pdf = tmp_path / "paper.pdf"
        pdf.touch()

        docs = MarkerParser().parse(pdf)

        assert len(docs) == 1

    @patch("marker.output.text_from_rendered")
    @patch("marker.models.create_model_dict")
    @patch("marker.converters.pdf.PdfConverter")
    def test_text_is_preserved(self, mock_cls, mock_models, mock_tfr, tmp_path):
        content = (
            "# Results\n\n"
            r"$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_{\text{true}} - y_{\text{pred}})^2$$"
        )
        mock_tfr.return_value = (content, "md", {})

        pdf = tmp_path / "paper.pdf"
        pdf.touch()

        docs = MarkerParser().parse(pdf)

        assert docs[0].text == content

    @patch("marker.output.text_from_rendered")
    @patch("marker.models.create_model_dict")
    @patch("marker.converters.pdf.PdfConverter")
    def test_latex_formula_preserved(self, mock_cls, mock_models, mock_tfr, tmp_path):
        """Confirm proper LaTeX syntax reaches the ParsedDocument."""
        latex_content = r"$MSE = \frac{1}{n}\sum_{i=1}^{n}(y - \hat{y})^2$"
        mock_tfr.return_value = (latex_content, "md", {})

        pdf = tmp_path / "math.pdf"
        pdf.touch()

        docs = MarkerParser().parse(pdf)

        assert r"\frac" in docs[0].text
        assert "$" in docs[0].text

    @patch("marker.output.text_from_rendered")
    @patch("marker.models.create_model_dict")
    @patch("marker.converters.pdf.PdfConverter")
    def test_metadata_contains_required_keys(self, mock_cls, mock_models, mock_tfr, tmp_path):
        mock_tfr.return_value = ("content", "md", {})

        pdf = tmp_path / "my_paper.pdf"
        pdf.touch()

        docs = MarkerParser().parse(pdf)

        meta = docs[0].metadata
        assert meta["doc_id"] == "my_paper.pdf"
        assert meta["parser"] == "marker"
        assert meta["has_latex"] is True
        assert "source" in meta

    @patch("marker.output.text_from_rendered")
    @patch("marker.models.create_model_dict")
    @patch("marker.converters.pdf.PdfConverter")
    def test_empty_document_returns_empty_list(self, mock_cls, mock_models, mock_tfr, tmp_path):
        mock_tfr.return_value = ("   ", "md", {})

        pdf = tmp_path / "empty.pdf"
        pdf.touch()

        docs = MarkerParser().parse(pdf)

        assert docs == []

    @patch("marker.output.text_from_rendered")
    @patch("marker.models.create_model_dict")
    @patch("marker.converters.pdf.PdfConverter")
    def test_converter_called_with_string_path(self, mock_cls, mock_models, mock_tfr, tmp_path):
        mock_tfr.return_value = ("text", "md", {})

        pdf = tmp_path / "test.pdf"
        pdf.touch()

        MarkerParser().parse(pdf)

        # PdfConverter(...) returns a callable; that callable is called with str(pdf)
        mock_cls.return_value.assert_called_once_with(str(pdf))

    @patch("marker.output.text_from_rendered")
    @patch("marker.models.create_model_dict")
    @patch("marker.converters.pdf.PdfConverter")
    def test_create_model_dict_called(self, mock_cls, mock_models, mock_tfr, tmp_path):
        mock_tfr.return_value = ("text", "md", {})

        pdf = tmp_path / "test.pdf"
        pdf.touch()

        MarkerParser().parse(pdf)

        mock_models.assert_called_once()
