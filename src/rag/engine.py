"""
RAGEngine — Facade over llama-index retrieval + generation.

Keeps the rest of the code agnostic of llama-index internals.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from ..config import Config

# ── Prompt templates ──────────────────────────────────────────────────────────
# {context_str} and {query_str} are required placeholders consumed by llama-index.

# Strict mode — factual retrieval only, no outside knowledge.
_QA_TEMPLATE_STR = """\
The following passages are extracted from the document you are studying.
Use them as the ONLY source of truth for your answer.

---------------------
{context_str}
---------------------

INSTRUCTIONS:
1. Answer ONLY from the passages above — never invent information.
2. Cite page numbers or section names when available: [Page X] or [Section: Y].
3. Preserve every mathematical formula exactly as written.
4. When explaining a formula, number each step of the derivation.
5. If the answer is not in the passages, reply:
   "I cannot find this information in the provided document."
6. Be concise but thorough.

Question: {query_str}
Answer: \
"""

# Explain mode — paper grounds the context; general math knowledge fills the explanation.
# Hallucination guard: paper-specific facts (results, claims) are still passage-only.
_EXPLAIN_TEMPLATE_STR = """\
You are an expert tutor helping a student understand content from a research document.

The following passages come directly from the document:
---------------------
{context_str}
---------------------

RULES — follow precisely:

[paper] sources — draw ONLY from the passages above:
  • What each variable or symbol means in THIS paper
  • Paper-specific claims, results, datasets, or conclusions
  • How a formula is motivated or used by the authors
  • Author-defined notation or naming conventions

[math] sources — you MAY use your general mathematical knowledge:
  • What a mathematical operator does (Σ, ∂, ‖·‖, etc.)
  • Step-by-step computation or evaluation of a formula
  • Worked numerical examples
  • Intuition, geometric interpretation, or analogies
  • Standard derivations and proofs

For EVERY claim in your answer, tag the source inline: [paper] or [math].

If the passages do not provide context for something asked, write:
  "The document does not specify this — general explanation:"
and continue.

NEVER invent paper-specific numbers, accuracy results, or conclusions.
Preserve every formula exactly as it appears in the passages.

Question: {query_str}
Answer: \
"""

# ── Data objects ──────────────────────────────────────────────────────────────


@dataclass
class SourceNode:
    """A single retrieved chunk with score and provenance."""
    text: str
    metadata: dict
    score: float


@dataclass
class QueryResult:
    """Structured result from a RAG query."""
    answer: str
    sources: List[SourceNode] = field(default_factory=list)
    thinking: Optional[str] = field(default=None)   # None when thinking was disabled
    raw_response: Optional[Any] = field(default=None, repr=False)
    retrieve_s: float = field(default=0.0)   # wall time for vector retrieval
    generate_s: float = field(default=0.0)   # wall time for LLM generation
    mode: str = field(default="strict")      # "strict" | "explain"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_thinking(text: str) -> tuple[str, Optional[str]]:
    """Split qwen3 output into (answer, thinking_or_None).

    Returns the thinking content as a separate string so callers can
    display, log, or discard it independently.
    """
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinking_blocks = pattern.findall(text)
    answer = pattern.sub("", text).strip()
    thinking = "\n\n".join(t.strip() for t in thinking_blocks) or None
    return answer, thinking


def _strip_thinking(text: str) -> str:
    """Convenience wrapper — returns only the answer (backwards-compat)."""
    answer, _ = _extract_thinking(text)
    return answer


# ── Engine ────────────────────────────────────────────────────────────────────


class RAGEngine:
    """Facade: wraps index + query engine into a single .query() call."""

    def __init__(self, index: VectorStoreIndex, config: Config) -> None:
        self.index = index
        self.config = config
        self._query_engine = self._build_query_engine()

    def _build_query_engine(self) -> RetrieverQueryEngine:
        self._retriever = self.index.as_retriever(
            similarity_top_k=self.config.similarity_top_k,
        )
        self._strict_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=PromptTemplate(_QA_TEMPLATE_STR),
        )
        self._explain_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=PromptTemplate(_EXPLAIN_TEMPLATE_STR),
        )
        # _synthesizer kept as alias for backwards-compat (points to strict)
        self._synthesizer = self._strict_synthesizer
        return RetrieverQueryEngine(
            retriever=self._retriever,
            response_synthesizer=self._strict_synthesizer,
        )

    def query(self, question: str, mode: str = "strict") -> QueryResult:
        """Run a RAG query and return a structured result with per-phase timing.

        Args:
            question: The user's question.
            mode: "strict" (answers only from retrieved passages) or
                  "explain" (passages anchor context; general math knowledge
                  allowed for derivations, intuition, and worked examples).
        """
        synthesizer = (
            self._explain_synthesizer if mode == "explain"
            else self._strict_synthesizer
        )

        # ── Phase 1: vector retrieval ──────────────────────────────────────
        t0 = time.perf_counter()
        nodes = self._retriever.retrieve(question)
        retrieve_s = time.perf_counter() - t0

        # ── Phase 2: LLM generation (thinking + answer) ────────────────────
        t1 = time.perf_counter()
        response = synthesizer.synthesize(question, nodes)
        generate_s = time.perf_counter() - t1

        answer, thinking = _extract_thinking(str(response))

        sources: List[SourceNode] = [
            SourceNode(
                text=nws.node.text,
                metadata=nws.node.metadata,
                score=nws.score or 0.0,
            )
            for nws in nodes
        ]

        return QueryResult(
            answer=answer,
            sources=sources,
            thinking=thinking,
            raw_response=response,
            retrieve_s=retrieve_s,
            generate_s=generate_s,
            mode=mode,
        )
