"""
Strategy pattern: abstract base for PDF parsers.

Adding a new parser is as simple as:
    class MyParser(BasePDFParser):
        ...
and passing it to RAGIndexer.  No other code needs to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ParsedDocument:
    """A single piece of text extracted from a PDF, with provenance metadata."""
    text: str
    metadata: dict = field(default_factory=dict)


class BasePDFParser(ABC):
    """Abstract parser — Strategy interface."""

    @abstractmethod
    def parse(self, pdf_path: Path) -> List[ParsedDocument]:
        """Parse *pdf_path* and return a list of document segments."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (used for logging and metadata tagging)."""
        ...

    @property
    def supports_latex(self) -> bool:
        """Whether this parser preserves LaTeX / math notation."""
        return False
