"""
Tests for src/figures/analyzer.py.

All HTTP calls to Ollama are mocked — no running server needed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.figures.analyzer import FigureAnalyzer, _strip_thinking
from src.figures.extractor import ExtractedFigure


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fig(page: int = 1, source: str = "vector") -> ExtractedFigure:
    return ExtractedFigure(
        page=page, index=0, source=source,
        image_bytes=b"\x89PNG\r\n\x1a\n", width=400, height=300,
    )


def _make_analyzer(**kwargs) -> FigureAnalyzer:
    return FigureAnalyzer(
        model="qwen3-vl:8b",
        base_url="http://localhost:11434",
        timeout=30.0,
        **kwargs,
    )


def _mock_response(content: str, status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"message": {"content": content}}
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return resp


# ── _strip_thinking ────────────────────────────────────────────────────────────

class TestStripThinking:
    def test_no_tags_unchanged(self):
        assert _strip_thinking("Hello world") == "Hello world"

    def test_single_think_block_removed(self):
        result = _strip_thinking("<think>some reasoning</think>Answer here.")
        assert result == "Answer here."
        assert "<think>" not in result

    def test_multiline_think_block_removed(self):
        text = "<think>\nline 1\nline 2\n</think>\nThe answer."
        assert _strip_thinking(text) == "The answer."

    def test_multiple_think_blocks_removed(self):
        text = "<think>first</think>Middle<think>second</think>End"
        assert _strip_thinking(text) == "MiddleEnd"

    def test_returns_stripped_string(self):
        assert _strip_thinking("  <think>x</think>  result  ") == "result"


# ── FigureAnalyzer.analyze ────────────────────────────────────────────────────

class TestAnalyze:
    def test_returns_description_on_success(self):
        analyzer = _make_analyzer()
        mock_resp = _mock_response("A line chart showing training loss over epochs.")
        with patch("requests.post", return_value=mock_resp):
            result = analyzer.analyze(_fig())
        assert result == "A line chart showing training loss over epochs."

    def test_strips_thinking_from_response(self):
        analyzer = _make_analyzer()
        content = "<think>Internal monologue</think>Bar chart showing accuracy."
        with patch("requests.post", return_value=_mock_response(content)):
            result = analyzer.analyze(_fig())
        assert result == "Bar chart showing accuracy."
        assert "<think>" not in result

    def test_returns_none_on_request_exception(self):
        analyzer = _make_analyzer()
        with patch("requests.post", side_effect=ConnectionError("refused")):
            result = analyzer.analyze(_fig())
        assert result is None

    def test_returns_none_on_http_error(self):
        analyzer = _make_analyzer()
        mock_resp = _mock_response("", status=500)
        with patch("requests.post", return_value=mock_resp):
            result = analyzer.analyze(_fig())
        assert result is None

    def test_returns_none_when_content_empty_after_strip(self):
        analyzer = _make_analyzer()
        # Only a thinking block, no actual answer
        with patch("requests.post", return_value=_mock_response("<think>x</think>")):
            result = analyzer.analyze(_fig())
        assert result is None

    def test_post_called_with_correct_url(self):
        analyzer = FigureAnalyzer(model="qwen3-vl:8b",
                                   base_url="http://localhost:11434")
        mock_resp = _mock_response("desc")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            analyzer.analyze(_fig())
        url = mock_post.call_args[0][0]
        assert url == "http://localhost:11434/api/chat"

    def test_post_payload_contains_model(self):
        analyzer = FigureAnalyzer(model="qwen3-vl:8b",
                                   base_url="http://localhost:11434")
        mock_resp = _mock_response("desc")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            analyzer.analyze(_fig())
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "qwen3-vl:8b"

    def test_post_payload_contains_base64_image(self):
        import base64
        analyzer = _make_analyzer()
        mock_resp = _mock_response("desc")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            analyzer.analyze(_fig())
        payload = mock_post.call_args[1]["json"]
        images = payload["messages"][0]["images"]
        assert len(images) == 1
        # Verify it is valid base64
        decoded = base64.b64decode(images[0])
        assert decoded == b"\x89PNG\r\n\x1a\n"


# ── FigureAnalyzer.analyze_all ────────────────────────────────────────────────

class TestAnalyzeAll:
    def test_returns_parsed_documents(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        analyzer = _make_analyzer()
        with patch("requests.post",
                   return_value=_mock_response("Line chart showing X vs Y.")):
            docs = analyzer.analyze_all([_fig(page=3)], pdf)
        assert len(docs) == 1
        assert "Line chart showing X vs Y." in docs[0].text

    def test_text_includes_page_number(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        analyzer = _make_analyzer()
        with patch("requests.post", return_value=_mock_response("Bar chart.")):
            docs = analyzer.analyze_all([_fig(page=7)], pdf)
        assert "page 7" in docs[0].text

    def test_metadata_fields(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.touch()
        analyzer = _make_analyzer()
        with patch("requests.post", return_value=_mock_response("desc")):
            docs = analyzer.analyze_all([_fig(page=2)], pdf)
        meta = docs[0].metadata
        assert meta["doc_id"] == "doc.pdf"
        assert meta["parser"] == "figure_analysis"
        assert meta["page"] == 2
        assert meta["vision_model"] == "qwen3-vl:8b"

    def test_failed_figures_skipped(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        analyzer = _make_analyzer()
        with patch("requests.post", side_effect=ConnectionError("fail")):
            docs = analyzer.analyze_all([_fig(), _fig()], pdf)
        assert docs == []

    def test_progress_callback_called(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        analyzer = _make_analyzer()
        calls = []
        with patch("requests.post", return_value=_mock_response("desc")):
            analyzer.analyze_all(
                [_fig(page=1), _fig(page=2)], pdf,
                progress_callback=lambda cur, tot, pg: calls.append((cur, tot, pg)),
            )
        assert len(calls) == 2
        assert calls[0] == (0, 2, 1)
        assert calls[1] == (1, 2, 2)

    def test_mixed_success_and_failure(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        analyzer = _make_analyzer()
        responses = [
            _mock_response("Good description."),
            MagicMock(side_effect=Exception("bad")),
        ]
        with patch("requests.post",
                   side_effect=[responses[0], Exception("bad")]):
            docs = analyzer.analyze_all([_fig(page=1), _fig(page=2)], pdf)
        assert len(docs) == 1
        assert "Good description." in docs[0].text
