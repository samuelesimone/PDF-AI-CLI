"""
Tests for IndexInspector.

Uses a synthetic docstore.json built in-memory — no real PDF or index needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.inspection.inspector import ChunkReport, IndexInspector, _detect_latex


# ── _detect_latex ──────────────────────────────────────────────────────────────

class TestDetectLatex:
    def test_display_dollars(self):
        hits = _detect_latex("The formula is $$E = mc^2$$ here.")
        labels = [l for l, _ in hits]
        assert "display_math_dollars" in labels

    def test_inline_dollar(self):
        hits = _detect_latex("Let $x = 5$ be given.")
        labels = [l for l, _ in hits]
        assert "inline_math_dollar" in labels

    def test_frac_command(self):
        hits = _detect_latex(r"We have \frac{a}{b} in the expression.")
        labels = [l for l, _ in hits]
        assert "math_commands" in labels

    def test_begin_equation(self):
        hits = _detect_latex(r"\begin{equation} f(x) = x^2 \end{equation}")
        labels = [l for l, _ in hits]
        assert "latex_env" in labels

    def test_markdown_math_fence(self):
        hits = _detect_latex("```math\nx^2\n```")
        labels = [l for l, _ in hits]
        assert "markdown_math_fence" in labels

    def test_plain_text_no_hits(self):
        hits = _detect_latex("This is a plain sentence with no math.")
        assert hits == []

    def test_escaped_underscore_not_counted(self):
        # Docling sometimes emits \_identifier — should not be flagged as math
        hits = _detect_latex(r"The variable \_length is 10.")
        # No real math pattern should match
        math_labels = {"display_math_dollars", "inline_math_dollar",
                       "display_math_bracket", "inline_math_paren",
                       "latex_env", "math_commands"}
        matched = {l for l, _ in hits}
        assert matched.isdisjoint(math_labels)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _write_docstore(tmp_path: Path, nodes: list[dict]) -> Path:
    """Write a minimal docstore.json and return its parent directory."""
    data: dict = {"docstore/data": {}, "docstore/metadata": {}, "docstore/ref_doc_info": {}}
    for i, node in enumerate(nodes):
        node_id = f"node-{i:04d}"
        data["docstore/data"][node_id] = {
            "__type__": "TextNode",
            "__data__": {
                "id_": node_id,
                "text": node.get("text", ""),
                "metadata": node.get("metadata", {}),
                "embedding": None,
                "relationships": {},
                "excluded_embed_metadata_keys": [],
                "excluded_llm_metadata_keys": [],
            },
        }
    idx_dir = tmp_path / "test_index"
    idx_dir.mkdir()
    (idx_dir / "docstore.json").write_text(json.dumps(data), encoding="utf-8")
    return idx_dir


@pytest.fixture
def plain_index(tmp_path):
    return _write_docstore(tmp_path, [
        {"text": "Chapter 1. Introduction to machine learning.", "metadata": {"page": 1}},
        {"text": "Chapter 2. Neural networks and backpropagation.", "metadata": {"page": 2}},
        {"text": "Chapter 3. Conclusion.", "metadata": {"page": 3}},
    ])


@pytest.fixture
def mixed_index(tmp_path):
    return _write_docstore(tmp_path, [
        {"text": "Plain text with no formulas.", "metadata": {"page": 1}},
        {"text": r"The loss function is $$L = -\sum_i y_i \log(\hat{y}_i)$$.", "metadata": {"page": 2}},
        {"text": r"Gradient: \frac{\partial L}{\partial w} = x^T(y - \hat{y}).", "metadata": {"page": 3}},
        {"text": "Another plain chunk.", "metadata": {"page": 4}},
    ])


# ── IndexInspector ─────────────────────────────────────────────────────────────

class TestIndexInspectorLoading:
    def test_loads_all_nodes(self, plain_index):
        inspector = IndexInspector(plain_index)
        assert len(inspector.chunks) == 3

    def test_raises_if_no_docstore(self, tmp_path):
        inspector = IndexInspector(tmp_path / "missing")
        with pytest.raises(FileNotFoundError):
            _ = inspector.chunks

    def test_chunk_report_fields(self, plain_index):
        chunk = IndexInspector(plain_index).chunk_at(0)
        assert isinstance(chunk, ChunkReport)
        assert chunk.index == 0
        assert chunk.char_count > 0
        assert "Introduction" in chunk.full_text
        assert "Introduction" in chunk.text_preview


class TestIndexInspectorSummary:
    def test_total_chunks(self, plain_index):
        s = IndexInspector(plain_index).summary()
        assert s["total_chunks"] == 3

    def test_zero_latex_in_plain_index(self, plain_index):
        s = IndexInspector(plain_index).summary()
        assert s["chunks_with_latex"] == 0
        assert s["latex_coverage_pct"] == 0.0

    def test_latex_coverage_in_mixed_index(self, mixed_index):
        s = IndexInspector(mixed_index).summary()
        assert s["chunks_with_latex"] == 2
        assert s["latex_coverage_pct"] == 50.0

    def test_avg_char_count(self, plain_index):
        s = IndexInspector(plain_index).summary()
        assert s["avg_chunk_chars"] > 0


class TestIndexInspectorQueries:
    def test_latex_chunks_returns_only_math_chunks(self, mixed_index):
        chunks = IndexInspector(mixed_index).latex_chunks()
        assert len(chunks) == 2
        assert all(c.has_latex for c in chunks)

    def test_latex_chunks_empty_for_plain(self, plain_index):
        assert IndexInspector(plain_index).latex_chunks() == []

    def test_search_finds_match(self, plain_index):
        results = IndexInspector(plain_index).search("neural")
        assert len(results) == 1
        assert "neural" in results[0].full_text.lower()

    def test_search_case_insensitive_by_default(self, plain_index):
        results = IndexInspector(plain_index).search("NEURAL")
        assert len(results) == 1

    def test_search_no_match(self, plain_index):
        assert IndexInspector(plain_index).search("quantum physics") == []

    def test_chunk_at_valid_index(self, plain_index):
        chunk = IndexInspector(plain_index).chunk_at(1)
        assert chunk.index == 1

    def test_chunk_at_out_of_range(self, plain_index):
        with pytest.raises(IndexError):
            IndexInspector(plain_index).chunk_at(99)

    def test_latex_types_named_correctly(self, mixed_index):
        chunks = IndexInspector(mixed_index).latex_chunks()
        all_types = {t for c in chunks for t in c.latex_types}
        assert all_types & {"display_math_dollars", "math_commands"}
