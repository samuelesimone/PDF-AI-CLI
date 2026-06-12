"""
FigureAnalyzer — describes PDF figures using a local Ollama vision model.

Sends each figure image to Ollama's /api/chat endpoint with a structured
prompt designed to produce descriptions rich enough for semantic search.
<think>…</think> blocks (produced by qwen3-style models) are stripped so
only the final answer is stored in the index.
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Callable, List, Optional

from ..parsers.base import ParsedDocument
from .extractor import ExtractedFigure

logger = logging.getLogger(__name__)

_PROMPT = """\
This image is a figure extracted from a scientific or academic document.

Describe it for a question-answering system. Cover every point that applies:
1. Figure type (line chart, bar chart, scatter plot, histogram, architecture diagram, flowchart, table, equation, photograph, etc.)
2. Title or caption text visible in the image
3. Axes labels and units (for plots and charts)
4. Data series, legend entries, or categories shown
5. Key numerical values, ranges, or trends in the data
6. Annotations, callouts, or highlighted regions
7. The main scientific finding or insight this figure conveys

Be factual and specific. Omit filler phrases.\
"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks from qwen3-style model output."""
    return _THINK_RE.sub("", text).strip()


class FigureAnalyzer:
    """Describe extracted figures using a local Ollama vision model."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # ── Single figure ─────────────────────────────────────────────────────────

    def analyze(self, figure: ExtractedFigure) -> Optional[str]:
        """Return a text description of *figure*, or ``None`` on failure."""
        import requests

        b64 = base64.b64encode(figure.image_bytes).decode()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": _PROMPT,
                    "images": [b64],
                }
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            return _strip_thinking(content) or None
        except Exception as exc:
            logger.warning(
                "Figure analysis failed (page %d, figure %d): %s",
                figure.page, figure.index, exc,
            )
            return None

    # ── Batch ────────────────────────────────────────────────────────────────

    def analyze_all(
        self,
        figures: List[ExtractedFigure],
        pdf_path: Path,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> List[ParsedDocument]:
        """Analyze every figure and return :class:`ParsedDocument` objects.

        Args:
            figures:           Figures to analyse.
            pdf_path:          Original PDF path (written into metadata).
            progress_callback: Optional ``callback(current, total, page)``
                called *before* each figure is analysed (0-based current).
        """
        docs: List[ParsedDocument] = []
        total = len(figures)

        for i, fig in enumerate(figures):
            if progress_callback:
                progress_callback(i, total, fig.page)

            description = self.analyze(fig)
            if not description:
                continue

            docs.append(ParsedDocument(
                text=f"[Figure on page {fig.page}] {description}",
                metadata={
                    "source": str(pdf_path),
                    "doc_id": pdf_path.name,
                    "parser": "figure_analysis",
                    "page": fig.page,
                    "figure_index": fig.index,
                    "figure_source": fig.source,
                    "vision_model": self.model,
                },
            ))

        return docs
