"""
RAGEngine — Facade over llama-index retrieval + generation.

Keeps the rest of the code agnostic of llama-index internals.

Conversation memory
-------------------
Each RAGEngine instance stores a session history (list of user/assistant turns).
The history is injected into every prompt so follow-up questions are answered
in context ("what about the other approach?" correctly refers to the previous answer).

Context compaction
------------------
When the turn count reaches _COMPACT_TRIGGER, the oldest turns are summarised by
the same local Ollama LLM and replaced with a compact anchor summary.  The last
_COMPACT_KEEP turns are always kept verbatim so recent context is never lost.
The compaction call is fully local — no external API is involved.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from ..config import Config

# ── Compaction settings ────────────────────────────────────────────────────────

# Compact when the session has this many full turns (user + assistant = 1 turn).
_COMPACT_TRIGGER: int = 10

# After compaction, keep this many recent turns verbatim.
_COMPACT_KEEP: int = 4

# Prompt used to summarise old turns into the anchor summary.
_COMPACT_PROMPT = """\
You are summarising a conversation between a student and an AI tutor about a PDF document.

Produce a concise summary (max 8 sentences) that preserves:
- The student's main questions and goals
- Specific concepts, terms, formulas, or page numbers that were established
- Answers that were confirmed correct
- Any misconceptions that were corrected
- Unresolved questions or topics that were left open

Be specific: keep exact names, formulas, and numbers. Drop pleasantries and redundant phrasing.

Conversation to summarise:
{history}

Summary:"""


# ── Prompt templates ───────────────────────────────────────────────────────────
# {context_str} — retrieved document passages (injected by llama-index)
# {query_str}   — the formatted query, which includes conversation history
#                 when the session has prior turns (see _build_query_str).

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

{query_str}
Answer: \
"""

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

{query_str}
Answer: \
"""

# Study mode — Socratic tutor applying evidence-based learning science.
_STUDY_TEMPLATE_STR = """\
You are a Socratic study tutor helping a student master this document's content.
Apply evidence-based learning science: active recall, elaborative interrogation, \
and the Feynman technique.

Passages retrieved from the document:
---------------------
{context_str}
---------------------

━━━ MANDATORY RULES ━━━

SOURCE TAGGING — tag every factual claim in your response:
  [doc]   — information that comes directly from the retrieved passages above
  [model] — your general knowledge, not found in this document
  Never omit tags. The learner must always know the origin of every claim.

━━━ TURN STRUCTURE — follow this exactly ━━━

STEP 1 — DETERMINE WHAT THE LEARNER SENT:

  A) They sent a QUESTION about a concept:
     • If the concept is NEW (not yet covered this session):
       → Answer it directly. Tag every claim [doc] or [model].
       → After your answer, optionally ask ONE follow-up question to deepen understanding.
     • If the concept was ALREADY EXPLAINED earlier in the conversation:
       → Ask exactly ONE short recall question (e.g. "Before I recap — in one sentence,
         what was the key point about X?") then STOP. Wait for the reply.
       → NEVER ask more than one question per turn.

  B) They sent an ANSWER to your previous question:
     → Step 1: Evaluate it in one sentence — correct, partially correct, or off.
       Cite the specific passage that confirms or contradicts their answer [doc].
     → Step 2: Immediately give the FULL explanation of the original topic they asked about.
       Do NOT ask another question before explaining. The explanation always comes now.
     → Step 3: After the explanation, you may OPTIONALLY add one deepening question
       (e.g. "Why does this hold true?" or "How would this apply to X?") — but only if
       the learner seemed confident. Skip it if they struggled.

  C) They say "just tell me", "I don't know", "skip", or show frustration:
     → Answer directly without any recall step. No questions, just the explanation.

CRITICAL CONSTRAINTS:
  • Ask AT MOST ONE question per turn — never a list of questions.
  • Never ask a question and then immediately answer it yourself in the same turn.
  • Never repeat a recall deflection if you already asked one in the previous turn.
  • The explanation must always arrive — it is never optional.

FEYNMAN CHECK — use at most once every 5 turns, only for complex concepts:
  "Feynman check — explain [concept] as if teaching it to a newcomer in 2-3 sentences."
  Evaluate the attempt [doc], then give the correct explanation regardless.

PRODUCTIVE ERRORS — wrong answers are learning events:
  Identify the specific misconception in one sentence, cite [doc], then explain correctly.
  Never just say "wrong" without immediately following with the right answer.

SESSION CLOSE — when the learner writes "done", "stop", "summary", "end session", or \
"recap", output exactly this structure:
    ✓ Strong: [concepts answered correctly and with confidence]
    ✗ Review soon (1–3 days): [concepts where gaps or errors appeared]
    📚 Next session — attempt these from memory without notes:
       1. [question targeting a weak area]
       2. [question targeting a weak area]
       3. [question targeting a weak area]

━━━ CURRENT INPUT ━━━
{query_str}

Respond now. Ask before telling. Tag every factual claim [doc] or [model].
"""


# ── Data objects ───────────────────────────────────────────────────────────────


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
    thinking: Optional[str] = field(default=None)
    raw_response: Optional[Any] = field(default=None, repr=False)
    retrieve_s: float = field(default=0.0)
    generate_s: float = field(default=0.0)
    mode: str = field(default="strict")       # "strict" | "explain" | "study"
    compacted: bool = field(default=False)    # True when compaction ran this turn


# ── Helpers ────────────────────────────────────────────────────────────────────


def _extract_thinking(text: str) -> tuple[str, Optional[str]]:
    """Split qwen3 output into (answer, thinking_or_None)."""
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinking_blocks = pattern.findall(text)
    answer = pattern.sub("", text).strip()
    thinking = "\n\n".join(t.strip() for t in thinking_blocks) or None
    return answer, thinking


def _strip_thinking(text: str) -> str:
    """Convenience wrapper — returns only the answer (backwards-compat)."""
    answer, _ = _extract_thinking(text)
    return answer


# ── Engine ─────────────────────────────────────────────────────────────────────


class RAGEngine:
    """Facade: wraps index + query engine into a single .query() call.

    Maintains a session conversation history.  Each turn is stored as a
    (user_question, assistant_answer) pair.  When the history grows past
    _COMPACT_TRIGGER turns, older turns are summarised locally by the same
    Ollama LLM and replaced with a compact anchor so the context window
    never overflows.
    """

    def __init__(self, index: VectorStoreIndex, config: Config) -> None:
        self.index = index
        self.config = config
        self._history: List[Tuple[str, str]] = []   # (user, assistant) pairs
        self._anchor: str = ""                       # summary of compacted turns
        self._query_engine = self._build_query_engine()

    # ── Setup ──────────────────────────────────────────────────────────────

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
        self._study_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=PromptTemplate(_STUDY_TEMPLATE_STR),
        )
        self._synthesizer = self._strict_synthesizer  # backwards-compat alias
        return RetrieverQueryEngine(
            retriever=self._retriever,
            response_synthesizer=self._strict_synthesizer,
        )

    # ── History helpers ────────────────────────────────────────────────────

    def _build_query_str(self, question: str) -> str:
        """Return the query string to pass to the synthesizer.

        When there is prior history (including an anchor summary from earlier
        compactions), the query includes the full conversation context so the
        LLM can answer follow-up questions correctly.
        """
        if not self._history and not self._anchor:
            return f"QUESTION:\n{question}"

        parts: List[str] = ["CONVERSATION HISTORY:"]

        if self._anchor:
            parts.append(f"[Summary of earlier conversation]\n{self._anchor}")

        for user_msg, assistant_msg in self._history:
            parts.append(f"User: {user_msg}")
            parts.append(f"Assistant: {assistant_msg}")

        parts.append(f"\nCURRENT QUESTION:\n{question}")
        return "\n".join(parts)

    def _should_compact(self) -> bool:
        return len(self._history) >= _COMPACT_TRIGGER

    def _compact(self) -> None:
        """Summarise the oldest turns and merge into the anchor summary.

        Only the newly evicted turns are sent to the LLM for summarisation;
        they are merged with the existing anchor.  This anchored-iterative
        strategy preserves detail across multiple compaction cycles.
        """
        keep = self._history[-_COMPACT_KEEP:]
        to_evict = self._history[:-_COMPACT_KEEP]

        history_text = "\n".join(
            f"User: {u}\nAssistant: {a}" for u, a in to_evict
        )
        if self._anchor:
            history_text = f"Previous summary:\n{self._anchor}\n\n{history_text}"

        response = Settings.llm.complete(
            _COMPACT_PROMPT.format(history=history_text)
        )
        self._anchor = response.text.strip()
        self._history = keep

    def reset_history(self) -> None:
        """Clear conversation history and the anchor summary."""
        self._history = []
        self._anchor = ""

    @property
    def history_turns(self) -> int:
        """Total turns tracked: recent verbatim + any compacted turns (≥0)."""
        return len(self._history)

    # ── Main entry point ───────────────────────────────────────────────────

    def query(self, question: str, mode: str = "strict") -> QueryResult:
        """Run a RAG query in the context of the current session history.

        Args:
            question: The user's question.
            mode: "strict" | "explain" | "study"
        """
        # Compact first if the history buffer is full
        compacted = False
        if self._should_compact():
            self._compact()
            compacted = True

        if mode == "explain":
            synthesizer = self._explain_synthesizer
        elif mode == "study":
            synthesizer = self._study_synthesizer
        else:
            synthesizer = self._strict_synthesizer

        # Build query string (includes history when available)
        query_str = self._build_query_str(question)

        # ── Phase 1: vector retrieval (use bare question for best recall) ──
        t0 = time.perf_counter()
        nodes = self._retriever.retrieve(question)
        retrieve_s = time.perf_counter() - t0

        # ── Phase 2: LLM generation ────────────────────────────────────────
        t1 = time.perf_counter()
        response = synthesizer.synthesize(query_str, nodes)
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

        # Store this turn in history
        self._history.append((question, answer))

        return QueryResult(
            answer=answer,
            sources=sources,
            thinking=thinking,
            raw_response=response,
            retrieve_s=retrieve_s,
            generate_s=generate_s,
            mode=mode,
            compacted=compacted,
        )
