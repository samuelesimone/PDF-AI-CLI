from .base import BasePDFParser, ParsedDocument
from .docling_parser import DoclingParser
from .marker_parser import MarkerParser
from .pymupdf_parser import PyMuPDFParser

PARSERS = {
    "docling": DoclingParser,
    "pymupdf": PyMuPDFParser,
    "marker": MarkerParser,
}

__all__ = ["BasePDFParser", "ParsedDocument", "DoclingParser", "MarkerParser", "PyMuPDFParser", "PARSERS"]
