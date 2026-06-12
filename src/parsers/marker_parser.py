"""
Concrete parser using marker-pdf (Datalab).

marker-pdf converts academic PDFs to Markdown with proper LaTeX:
- Mathematical formulas as inline/block LaTeX ($...$, $$...$$)
- Tables as Markdown tables
- Section hierarchy from visual layout analysis
- Reading order via Surya OCR

Import is deferred to parse() so the class can be imported and tested
without marker-pdf installed (the tests mock it).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .base import BasePDFParser, ParsedDocument


class MarkerParser(BasePDFParser):
    """Parse a PDF with marker-pdf, producing proper LaTeX for formulas."""

    @property
    def name(self) -> str:
        return "marker"

    @property
    def supports_latex(self) -> bool:
        return True

    def parse(self, pdf_path: Path) -> List[ParsedDocument]:
        """Convert *pdf_path* to Markdown with LaTeX formulas via marker-pdf.

        Strategy:
        1. Load marker models (downloaded on first run, then cached locally).
        2. Convert the PDF — marker uses Surya OCR + layout detection.
        3. Export as Markdown with inline/block LaTeX for all formulas.
        4. Return one ParsedDocument; llama-index handles chunking downstream.
        """
        from marker.converters.pdf import PdfConverter  # lazy import
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(pdf_path))
        full_markdown, _, _ = text_from_rendered(rendered)

        if not full_markdown.strip():
            return []

        return [
            ParsedDocument(
                text=full_markdown,
                metadata={
                    "source": str(pdf_path),
                    "doc_id": pdf_path.name,
                    "parser": self.name,
                    "has_latex": self.supports_latex,
                },
            )
        ]
