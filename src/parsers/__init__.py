from .base import BasePDFParser, ParsedDocument
from .docling_parser import DoclingParser
from .marker_parser import MarkerParser
from .mineru_parser import MinerUParser
from .pymupdf_parser import PyMuPDFParser

PARSERS = {
    "docling": DoclingParser,
    "pymupdf": PyMuPDFParser,
    "marker": MarkerParser,
    "mineru": MinerUParser,
}

__all__ = [
    "BasePDFParser", "ParsedDocument",
    "DoclingParser", "MarkerParser", "MinerUParser", "PyMuPDFParser",
    "PARSERS",
]
