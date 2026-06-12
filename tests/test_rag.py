"""
Tests for the RAGEngine facade.

The llama-index VectorStoreIndex is mocked so no embedding or LLM calls
are made during tests.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.engine import QueryResult, RAGEngine, SourceNode, _extract_thinking, _strip_thinking


# ── _extract_thinking ──────────────────────────────────────────────────────────

class TestExtractThinking:
    def test_splits_single_block(self):
        text = "<think>internal reasoning</think>The answer is 42."
        answer, thinking = _extract_thinking(text)
        assert answer == "The answer is 42."
        assert thinking == "internal reasoning"

    def test_no_think_block_returns_none(self):
        text = "The answer is 42."
        answer, thinking = _extract_thinking(text)
        assert answer == "The answer is 42."
        assert thinking is None

    def test_multiple_blocks_joined(self):
        text = "<think>Part A</think>Answer<think>Part B</think> here."
        answer, thinking = _extract_thinking(text)
        assert answer == "Answer here."
        assert "Part A" in thinking
        assert "Part B" in thinking

    def test_multiline_block(self):
        text = "<think>\nLine one\nLine two\n</think>Answer."
        answer, thinking = _extract_thinking(text)
        assert answer == "Answer."
        assert "Line one" in thinking

    def test_only_think_block_empty_answer(self):
        text = "<think>nothing useful</think>"
        answer, thinking = _extract_thinking(text)
        assert answer == ""
        assert thinking == "nothing useful"

    def test_strips_surrounding_whitespace_from_answer(self):
        text = "  <think>x</think>   result   "
        answer, thinking = _extract_thinking(text)
        assert answer == "result"


# ── _strip_thinking (backwards-compat wrapper) ────────────────────────────────

class TestStripThinking:
    def test_removes_single_think_block(self):
        assert _strip_thinking("<think>reasoning</think>Answer.") == "Answer."

    def test_no_block_unchanged(self):
        assert _strip_thinking("The answer is 42.") == "The answer is 42."


# ── QueryResult / SourceNode ──────────────────────────────────────────────────

class TestDataClasses:
    def test_query_result_defaults(self):
        r = QueryResult(answer="hi")
        assert r.sources == []
        assert r.thinking is None
        assert r.raw_response is None
        assert r.retrieve_s == 0.0
        assert r.generate_s == 0.0
        assert r.mode == "strict"

    def test_query_result_with_thinking(self):
        r = QueryResult(answer="hi", thinking="some reasoning")
        assert r.thinking == "some reasoning"

    def test_query_result_timing_fields(self):
        r = QueryResult(answer="hi", retrieve_s=0.05, generate_s=12.3)
        assert abs(r.retrieve_s - 0.05) < 1e-9
        assert abs(r.generate_s - 12.3) < 1e-9

    def test_query_result_mode_field(self):
        assert QueryResult(answer="hi", mode="explain").mode == "explain"
        assert QueryResult(answer="hi", mode="strict").mode == "strict"

    def test_source_node_fields(self):
        s = SourceNode(text="excerpt", metadata={"page": 3}, score=0.95)
        assert s.score == 0.95
        assert s.metadata["page"] == 3


# ── RAGEngine ─────────────────────────────────────────────────────────────────

class TestRAGEngine:
    def _make_engine(self, config, answer_text="The answer.", source_nodes=None):
        mock_response = MagicMock()
        mock_response.__str__ = lambda self: answer_text

        mock_nodes = source_nodes or []

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_nodes

        mock_strict_synth = MagicMock()
        mock_strict_synth.synthesize.return_value = mock_response

        mock_explain_synth = MagicMock()
        mock_explain_synth.synthesize.return_value = mock_response

        mock_index = MagicMock()
        # get_response_synthesizer is called twice (strict + explain); return both mocks
        with patch("src.rag.engine.RetrieverQueryEngine"), \
             patch("src.rag.engine.get_response_synthesizer",
                   side_effect=[mock_strict_synth, mock_explain_synth]), \
             patch.object(mock_index, "as_retriever", return_value=mock_retriever):
            engine = RAGEngine(index=mock_index, config=config)

        engine._retriever = mock_retriever
        engine._strict_synthesizer = mock_strict_synth
        engine._explain_synthesizer = mock_explain_synth
        engine._synthesizer = mock_strict_synth
        return engine, mock_retriever, mock_strict_synth, mock_explain_synth

    def test_query_returns_query_result(self, config):
        engine, _, _, _ = self._make_engine(config)
        result = engine.query("What is gravity?")
        assert isinstance(result, QueryResult)

    def test_answer_text_is_preserved(self, config):
        engine, _, _, _ = self._make_engine(config, answer_text="Gravity is a force.")
        result = engine.query("What is gravity?")
        assert result.answer == "Gravity is a force."

    def test_thinking_separated_from_answer(self, config):
        engine, _, _, _ = self._make_engine(
            config, answer_text="<think>hmm</think>Gravity is a force."
        )
        result = engine.query("What is gravity?")
        assert result.answer == "Gravity is a force."
        assert "<think>" not in result.answer
        assert result.thinking == "hmm"

    def test_thinking_is_none_when_no_block(self, config):
        engine, _, _, _ = self._make_engine(config, answer_text="Gravity is a force.")
        result = engine.query("What is gravity?")
        assert result.thinking is None

    def test_empty_sources_when_no_nodes(self, config):
        engine, _, _, _ = self._make_engine(config, source_nodes=[])
        result = engine.query("test")
        assert result.sources == []

    def test_sources_populated_from_response(self, config):
        mock_node = MagicMock()
        mock_node.node.text = "Relevant passage."
        mock_node.node.metadata = {"page": 12}
        mock_node.score = 0.87

        engine, _, _, _ = self._make_engine(config, source_nodes=[mock_node])
        result = engine.query("test")

        assert len(result.sources) == 1
        src = result.sources[0]
        assert src.text == "Relevant passage."
        assert src.metadata["page"] == 12
        assert abs(src.score - 0.87) < 1e-6

    def test_retriever_called_with_question(self, config):
        engine, mock_retriever, _, _ = self._make_engine(config)
        engine.query("Explain entropy.")
        mock_retriever.retrieve.assert_called_once_with("Explain entropy.")

    def test_strict_mode_uses_strict_synthesizer(self, config):
        mock_node = MagicMock()
        engine, _, mock_strict, mock_explain = self._make_engine(
            config, source_nodes=[mock_node]
        )
        engine.query("What is the accuracy?", mode="strict")
        mock_strict.synthesize.assert_called_once()
        mock_explain.synthesize.assert_not_called()

    def test_explain_mode_uses_explain_synthesizer(self, config):
        mock_node = MagicMock()
        engine, _, mock_strict, mock_explain = self._make_engine(
            config, source_nodes=[mock_node]
        )
        engine.query("Explain the MSE formula.", mode="explain")
        mock_explain.synthesize.assert_called_once()
        mock_strict.synthesize.assert_not_called()

    def test_result_carries_mode(self, config):
        engine, _, _, _ = self._make_engine(config)
        assert engine.query("test", mode="strict").mode == "strict"
        assert engine.query("test", mode="explain").mode == "explain"

    def test_timing_fields_are_positive(self, config):
        engine, _, _, _ = self._make_engine(config)
        result = engine.query("test")
        assert result.retrieve_s >= 0.0
        assert result.generate_s >= 0.0
