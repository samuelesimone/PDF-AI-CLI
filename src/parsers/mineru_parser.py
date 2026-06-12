"""
Concrete parser using MinerU (OpenDataLab).

MinerU converts PDFs to Markdown with proper LaTeX formulas using either
a pipeline backend (stable, CPU/GPU) or a VLM backend (higher accuracy).

Backend is selected via the MINERU_BACKEND environment variable.
Valid values for MinerU ≥ 3.x:
    pipeline      stable, CPU/GPU, no server overhead (default)
    vlm-engine    local VLM, higher accuracy, needs more VRAM
    hybrid-engine best accuracy, combines pipeline + VLM

Import is deferred to parse() so the class can be imported and tested
without MinerU installed (subprocess is mocked in tests).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

from .base import BasePDFParser, ParsedDocument

# Environment variable that controls which MinerU backend is used.
# Override in .env or shell without changing code.
_BACKEND_ENV = "MINERU_BACKEND"
_DEFAULT_BACKEND = "pipeline"


class MinerUParser(BasePDFParser):
    """Parse a PDF with MinerU, producing proper LaTeX for formulas.

    Backend is selected via the MINERU_BACKEND environment variable:
        pipeline      stable, CPU/GPU, no server overhead (default)
        vlm-engine    local VLM, higher accuracy, needs more VRAM
        hybrid-engine best accuracy, combines pipeline + VLM
    """

    @property
    def name(self) -> str:
        return "mineru"

    @property
    def supports_latex(self) -> bool:
        return True

    def parse(self, pdf_path: Path) -> List[ParsedDocument]:
        """Convert *pdf_path* to Markdown with LaTeX formulas via MinerU CLI.

        Strategy:
        1. Run `mineru -p <pdf> -o <tmp_dir> -b <backend>`.
        2. MinerU writes <tmp_dir>/<stem>/auto_<stem>.md (pipeline) or
           similar depending on backend/version.
        3. Locate the primary .md file via glob; prefer names containing
           'auto' or 'full' (MinerU conventions), fall back to largest file.
        4. Return one ParsedDocument; llama-index handles chunking downstream.
        """
        backend = os.environ.get(_BACKEND_ENV, _DEFAULT_BACKEND)

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                subprocess.run(
                    ["mineru", "-p", str(pdf_path), "-o", tmp_dir, "-b", backend],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                # Surface the real error message rather than just the exit code
                raise RuntimeError(
                    f"mineru failed (exit {exc.returncode}, backend={backend})"
                    + (f":\n{stderr}" if stderr else "")
                ) from exc

            md_files = sorted(Path(tmp_dir).glob("**/*.md"))
            if not md_files:
                return []

            # MinerU names vary by version; prefer known conventions
            preferred = [
                f for f in md_files
                if any(k in f.stem for k in ("auto", "full"))
            ]
            md_file = (
                preferred[0]
                if preferred
                else max(md_files, key=lambda f: f.stat().st_size)
            )
            full_markdown = md_file.read_text(encoding="utf-8")

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
                    "backend": backend,
                },
            )
        ]
