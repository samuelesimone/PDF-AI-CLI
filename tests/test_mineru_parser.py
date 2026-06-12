"""
Tests for MinerUParser.

MinerU is invoked via subprocess, so all tests mock subprocess.run.
The mock side-effect writes a realistic output directory structure so the
parser can find and read the produced markdown file — no real binary needed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.parsers.mineru_parser import MinerUParser, _DEFAULT_BACKEND, _BACKEND_ENV

# ── Helpers ────────────────────────────────────────────────────────────────────

_LATEX_CONTENT = (
    "# Results\n\n"
    r"$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_{\text{true}} - y_{\text{pred}})^2$$"
    "\n\nSome paragraph text."
)


def _make_fake_run(content: str = _LATEX_CONTENT):
    """
    Return a side_effect for subprocess.run that creates MinerU-style output:
        <output_dir>/<pdf_stem>/auto_<pdf_stem>.md
    """
    def _fake(cmd, **kwargs):
        o_idx = cmd.index("-o")
        out_dir = Path(cmd[o_idx + 1])
        p_idx = cmd.index("-p")
        stem = Path(cmd[p_idx + 1]).stem
        sub = out_dir / stem
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"auto_{stem}.md").write_text(content, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return _fake


# ── Properties ────────────────────────────────────────────────────────────────

class TestMinerUParserProperties:
    def test_name(self):
        assert MinerUParser().name == "mineru"

    def test_supports_latex(self):
        assert MinerUParser().supports_latex is True


# ── Parse — happy path ────────────────────────────────────────────────────────

class TestMinerUParserParse:
    def test_returns_one_document(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        with patch("subprocess.run", side_effect=_make_fake_run()):
            docs = MinerUParser().parse(pdf)
        assert len(docs) == 1

    def test_text_is_preserved(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        with patch("subprocess.run", side_effect=_make_fake_run(_LATEX_CONTENT)):
            docs = MinerUParser().parse(pdf)
        assert docs[0].text == _LATEX_CONTENT

    def test_latex_formulas_preserved(self, tmp_path):
        pdf = tmp_path / "math.pdf"
        pdf.touch()
        with patch("subprocess.run", side_effect=_make_fake_run(_LATEX_CONTENT)):
            docs = MinerUParser().parse(pdf)
        assert r"\frac" in docs[0].text
        assert r"\sum" in docs[0].text
        assert "$$" in docs[0].text

    def test_metadata_required_keys(self, tmp_path):
        pdf = tmp_path / "my_paper.pdf"
        pdf.touch()
        with patch("subprocess.run", side_effect=_make_fake_run()):
            docs = MinerUParser().parse(pdf)
        meta = docs[0].metadata
        assert meta["doc_id"] == "my_paper.pdf"
        assert meta["parser"] == "mineru"
        assert meta["has_latex"] is True
        assert "source" in meta
        assert "backend" in meta

    def test_metadata_backend_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv(_BACKEND_ENV, raising=False)
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        with patch("subprocess.run", side_effect=_make_fake_run()):
            docs = MinerUParser().parse(pdf)
        assert docs[0].metadata["backend"] == _DEFAULT_BACKEND

    def test_metadata_backend_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_BACKEND_ENV, "vlm-mlx-engine")
        pdf = tmp_path / "paper.pdf"
        pdf.touch()
        with patch("subprocess.run", side_effect=_make_fake_run()):
            docs = MinerUParser().parse(pdf)
        assert docs[0].metadata["backend"] == "vlm-mlx-engine"


# ── Parse — edge cases ────────────────────────────────────────────────────────

class TestMinerUParserEdgeCases:
    def test_empty_markdown_returns_empty_list(self, tmp_path):
        pdf = tmp_path / "empty.pdf"
        pdf.touch()
        with patch("subprocess.run", side_effect=_make_fake_run("   ")):
            docs = MinerUParser().parse(pdf)
        assert docs == []

    def test_no_md_file_produced_returns_empty_list(self, tmp_path):
        """If MinerU writes no .md file (e.g. unsupported PDF), return []."""
        pdf = tmp_path / "broken.pdf"
        pdf.touch()

        def _fake_no_output(cmd, **kwargs):
            # Creates the output dir but writes no markdown file
            o_idx = cmd.index("-o")
            Path(cmd[o_idx + 1]).mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch("subprocess.run", side_effect=_fake_no_output):
            docs = MinerUParser().parse(pdf)
        assert docs == []

    def test_subprocess_error_raises_runtime_error(self, tmp_path):
        """CalledProcessError is converted to RuntimeError with the stderr included."""
        pdf = tmp_path / "paper.pdf"
        pdf.touch()

        def _fake_fail(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, stderr="model load error")

        with patch("subprocess.run", side_effect=_fake_fail):
            with pytest.raises(RuntimeError, match="mineru failed"):
                MinerUParser().parse(pdf)

    def test_subprocess_error_includes_stderr(self, tmp_path):
        """The RuntimeError message contains the original stderr output."""
        pdf = tmp_path / "paper.pdf"
        pdf.touch()

        def _fake_fail(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, stderr="CUDA out of memory")

        with patch("subprocess.run", side_effect=_fake_fail):
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                MinerUParser().parse(pdf)

    def test_subprocess_error_includes_backend_name(self, tmp_path, monkeypatch):
        """The RuntimeError message includes the backend that was used."""
        monkeypatch.setenv("MINERU_BACKEND", "hybrid-engine")
        pdf = tmp_path / "paper.pdf"
        pdf.touch()

        def _fake_fail(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, stderr="")

        with patch("subprocess.run", side_effect=_fake_fail):
            with pytest.raises(RuntimeError, match="hybrid-engine"):
                MinerUParser().parse(pdf)


# ── Subprocess command ─────────────────────────────────────────────────────────

class TestMinerUParserCommand:
    def test_command_uses_mineru(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        captured = {}

        def _capture(cmd, **kwargs):
            captured["cmd"] = cmd
            _make_fake_run()(cmd, **kwargs)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch("subprocess.run", side_effect=_capture):
            MinerUParser().parse(pdf)

        assert captured["cmd"][0] == "mineru"

    def test_command_passes_pdf_path(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        captured = {}

        def _capture(cmd, **kwargs):
            captured["cmd"] = cmd
            _make_fake_run()(cmd, **kwargs)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch("subprocess.run", side_effect=_capture):
            MinerUParser().parse(pdf)

        p_idx = captured["cmd"].index("-p")
        assert captured["cmd"][p_idx + 1] == str(pdf)

    def test_command_passes_backend(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_BACKEND_ENV, "hybrid-engine")
        pdf = tmp_path / "test.pdf"
        pdf.touch()
        captured = {}

        def _capture(cmd, **kwargs):
            captured["cmd"] = cmd
            _make_fake_run()(cmd, **kwargs)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch("subprocess.run", side_effect=_capture):
            MinerUParser().parse(pdf)

        b_idx = captured["cmd"].index("-b")
        assert captured["cmd"][b_idx + 1] == "hybrid-engine"
