"""
Shared pytest fixtures.

All fixtures that touch the filesystem use tmp_path (pytest built-in)
so tests never pollute the real storage directory.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """A Config instance pointing storage at a temp directory."""
    return Config(
        ollama_base_url="http://localhost:11434",
        llm_model="qwen3.5:9b",
        embed_model="bge-m3:latest",
        temperature=0.0,
        request_timeout=30.0,
        chunk_size=256,
        chunk_overlap=32,
        similarity_top_k=3,
        storage_dir=tmp_path / "storage",
    )


@pytest.fixture
def minimal_pdf(tmp_path: Path) -> Path:
    """A minimal, structurally valid PDF file for parser-level tests."""
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello, World!) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        b"endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000274 00000 n \n"
        b"0000000369 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n444\n%%EOF"
    )
    path = tmp_path / "test.pdf"
    path.write_bytes(pdf_bytes)
    return path
