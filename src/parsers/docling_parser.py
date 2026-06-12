"""
Concrete parser using IBM Docling.

Docling converts PDFs to structured Markdown that preserves:
- LaTeX / math formulas
- Tables
- Section hierarchy
- Reading order

Import is deferred to parse() so the class can be imported and tested
without docling installed (the tests mock it).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .base import BasePDFParser, ParsedDocument


class DoclingParser(BasePDFParser):
    """Parse a PDF with Docling and export it as Markdown."""

    @property
    def name(self) -> str:
        return "docling"

    @property
    def supports_latex(self) -> bool:
        return True

    def parse(self, pdf_path: Path) -> List[ParsedDocument]:
        """Convert *pdf_path* to one or more ParsedDocument objects.

        Strategy:
        1. Convert the whole PDF via Docling (handles OCR, math, tables).
        2. Export the full document as Markdown — this preserves LaTeX.
        3. Return a single ParsedDocument containing the full Markdown text.
           llama-index's SentenceSplitter will handle chunking downstream.
        """
        from docling.document_converter import DocumentConverter  # lazy import

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))

        full_markdown: str = result.document.export_to_markdown()

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
