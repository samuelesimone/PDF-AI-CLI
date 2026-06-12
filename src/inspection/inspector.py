"""
IndexInspector — reads a persisted llama-index docstore and analyses its chunks.

Answers questions like:
  - How many chunks were created?
  - Which chunks contain LaTeX / math?
  - What does a specific chunk look like in full?
  - Is the formula extraction working?

Reads directly from the JSON files on disk — no Ollama, no embeddings needed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── LaTeX pattern catalogue ────────────────────────────────────────────────────
# Each entry is (label, compiled_regex).
# We match ANY of these to consider a chunk as containing math/formula content.

_LATEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Display math: $$...$$ or \[...\]
    ("display_math_dollars", re.compile(r"\$\$.+?\$\$", re.DOTALL)),
    ("display_math_bracket", re.compile(r"\\\[.+?\\\]", re.DOTALL)),
    # Inline math: $...$ (not escaped underscore)
    ("inline_math_dollar",   re.compile(r"(?<!\$)\$(?!\$)[^\n$]{2,80}\$")),
    # Inline math: \(...\)
    ("inline_math_paren",    re.compile(r"\\\(.+?\\\)", re.DOTALL)),
    # LaTeX environments
    ("latex_env",            re.compile(r"\\begin\{(equation|align|matrix|pmatrix|bmatrix|cases|gather|multline)\*?\}")),
    # Common math commands
    ("math_commands",        re.compile(r"\\(frac|sqrt|sum|prod|int|lim|infty|partial|nabla|Delta|Sigma|alpha|beta|gamma|theta|lambda|mu|sigma|omega|pi)\b")),
    # Docling markdown math fence: ```math
    ("markdown_math_fence",  re.compile(r"```math")),
    # Docling failed to decode a formula region (visual model gave up).
    # The formula IS in the PDF — use PyMuPDFParser to extract its text.
    ("docling_formula_not_decoded", re.compile(r"<!--\s*formula-not-decoded\s*-->")),
]


def _detect_latex(text: str) -> list[tuple[str, str]]:
    """Return list of (pattern_label, matched_snippet) for every LaTeX hit."""
    hits: list[tuple[str, str]] = []
    for label, pat in _LATEX_PATTERNS:
        for m in pat.finditer(text):
            snippet = m.group(0).replace("\n", " ")[:80]
            hits.append((label, snippet))
    return hits


# ── Data objects ───────────────────────────────────────────────────────────────

@dataclass
class ChunkReport:
    index: int
    node_id: str
    char_count: int
    text_preview: str          # first 120 chars, newlines collapsed
    full_text: str
    metadata: dict
    latex_hits: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_latex(self) -> bool:
        return bool(self.latex_hits)

    @property
    def latex_types(self) -> list[str]:
        return list({label for label, _ in self.latex_hits})


# ── Inspector ──────────────────────────────────────────────────────────────────

class IndexInspector:
    """
    Reads a persisted llama-index docstore directory and exposes its chunks
    for analysis without requiring Ollama or any model.
    """

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self._chunks: Optional[List[ChunkReport]] = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> List[ChunkReport]:
        docstore_path = self.index_dir / "docstore.json"
        if not docstore_path.exists():
            raise FileNotFoundError(
                f"No docstore.json found in {self.index_dir}\n"
                "Run 'python main.py <pdf>' first to build the index."
            )

        with docstore_path.open(encoding="utf-8") as f:
            raw = json.load(f)

        nodes = raw.get("docstore/data", {})
        chunks: List[ChunkReport] = []

        for i, (node_id, node_wrapper) in enumerate(nodes.items()):
            data = node_wrapper.get("__data__", {})
            text: str = data.get("text", "")
            metadata: dict = data.get("metadata", {})
            latex_hits = _detect_latex(text)

            chunks.append(ChunkReport(
                index=i,
                node_id=node_id,
                char_count=len(text),
                text_preview=text.replace("\n", " ")[:120],
                full_text=text,
                metadata=metadata,
                latex_hits=latex_hits,
            ))

        return chunks

    @property
    def chunks(self) -> List[ChunkReport]:
        if self._chunks is None:
            self._chunks = self._load()
        return self._chunks

    # ── Queries ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """High-level statistics about the index."""
        total = len(self.chunks)
        with_latex = [c for c in self.chunks if c.has_latex]
        all_types: list[str] = []
        for c in with_latex:
            all_types.extend(c.latex_types)

        type_counts: dict[str, int] = {}
        for t in all_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        not_decoded = sum(
            1 for c in self.chunks
            if any(l == "docling_formula_not_decoded" for l, _ in c.latex_hits)
        )
        return {
            "index_dir": str(self.index_dir),
            "total_chunks": total,
            "chunks_with_latex": len(with_latex),
            "latex_coverage_pct": round(100 * len(with_latex) / total, 1) if total else 0,
            "avg_chunk_chars": round(sum(c.char_count for c in self.chunks) / total) if total else 0,
            "min_chunk_chars": min(c.char_count for c in self.chunks) if total else 0,
            "max_chunk_chars": max(c.char_count for c in self.chunks) if total else 0,
            "latex_pattern_breakdown": type_counts,
            "source_doc": self.chunks[0].metadata.get("doc_id", "unknown") if total else "—",
            "parser": self.chunks[0].metadata.get("parser", "unknown") if total else "—",
            "formula_not_decoded_count": not_decoded,
        }

    def latex_chunks(self) -> List[ChunkReport]:
        """Return only chunks that contain at least one LaTeX pattern."""
        return [c for c in self.chunks if c.has_latex]

    def search(self, query: str, case_sensitive: bool = False) -> List[ChunkReport]:
        """Return chunks whose text contains *query*."""
        flags = 0 if case_sensitive else re.IGNORECASE
        pat = re.compile(re.escape(query), flags)
        return [c for c in self.chunks if pat.search(c.full_text)]

    def chunk_at(self, index: int) -> ChunkReport:
        if index < 0 or index >= len(self.chunks):
            raise IndexError(f"Chunk index {index} out of range (0–{len(self.chunks)-1})")
        return self.chunks[index]
