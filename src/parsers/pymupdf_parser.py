"""
PyMuPDFParser — direct text extraction via PyMuPDF (fitz).

When to use instead of DoclingParser:
  - PDFs compiled directly from LaTeX source.
  - When Docling outputs <!-- formula-not-decoded --> for your equations.

Why it works for LaTeX PDFs:
  LaTeX embeds all text — including math — as actual character streams using
  TeX fonts (Computer Modern, etc.).  PyMuPDF reads those streams directly,
  so every symbol (∑, ∥, subscripts, fractions) is captured as Unicode.
  The result is not perfect LaTeX notation, but the LLM can read and reason
  about it perfectly well.

Trade-off vs Docling:
  - Better for:  text-based PDFs, LaTeX-compiled papers, formulas as text.
  - Worse for:   scanned/OCR PDFs, complex table layouts, image-only math.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .base import BasePDFParser, ParsedDocument


class PyMuPDFParser(BasePDFParser):
    """
    Parse a PDF page-by-page using PyMuPDF text extraction.

    Each page becomes one ParsedDocument, preserving page-number metadata
    that the RAG engine can surface as citations.
    """

    @property
    def name(self) -> str:
        return "pymupdf"

    @property
    def supports_latex(self) -> bool:
        # Does not decode to LaTeX syntax, but preserves math characters.
        return False

    def parse(self, pdf_path: Path) -> List[ParsedDocument]:
        import fitz  # lazy import — only needed at parse time

        docs: List[ParsedDocument] = []

        with fitz.open(str(pdf_path)) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]

                # "text" mode: linearised reading order, merges all spans.
                text: str = page.get_text("text").strip()

                if not text:
                    continue

                docs.append(
                    ParsedDocument(
                        text=text,
                        metadata={
                            "page": page_num + 1,
                            "source": str(pdf_path),
                            "doc_id": pdf_path.name,
                            "parser": self.name,
                        },
                    )
                )

        return docs
