"""
Tests for the _render_latex helper in terminal.py.

These tests verify formula conversion across all four delimiter styles and
check graceful fallback when pylatexenc is unavailable.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from src.chat.terminal import _render_latex


# ── No-op cases ────────────────────────────────────────────────────────────────

class TestRenderLatexNoop:
    def test_plain_text_unchanged(self):
        text = "The model converges after 10 epochs."
        assert _render_latex(text) == text

    def test_empty_string_unchanged(self):
        assert _render_latex("") == ""


# ── Inline math: $...$ ────────────────────────────────────────────────────────

class TestInlineDollar:
    def test_greek_letters(self):
        result = _render_latex(r"The parameter $\alpha$ controls learning rate.")
        assert "α" in result
        assert r"\alpha" not in result
        assert "$" not in result

    def test_fraction(self):
        result = _render_latex(r"The mean is $\frac{1}{n}$.")
        assert "1/n" in result
        assert "$" not in result

    def test_multiple_inline(self):
        result = _render_latex(r"Both $\alpha$ and $\beta$ matter.")
        assert "α" in result
        assert "β" in result

    def test_sigma_and_nabla(self):
        result = _render_latex(r"Update via $\nabla L(\theta)$.")
        assert "∇" in result
        assert "θ" in result


# ── Display math: $$...$$ ─────────────────────────────────────────────────────

class TestDisplayDoubleDollar:
    def test_gets_newlines(self):
        result = _render_latex(r"See $$\frac{1}{n}$$ above.")
        assert "\n" in result

    def test_sum_converted(self):
        result = _render_latex(
            r"$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_{true} - y_{pred})^2$$"
        )
        assert "∑" in result
        assert "MSE" in result
        assert "$$" not in result

    def test_does_not_bleed_into_single_dollar(self):
        # $$...$$ should be consumed before the $...$ pass runs
        result = _render_latex(r"$$\alpha$$")
        assert "$$" not in result
        assert "α" in result


# ── Display math: \[...\] ─────────────────────────────────────────────────────

class TestDisplayBrackets:
    def test_bracket_display_converted(self):
        result = _render_latex(r"\[\sigma(x) = \frac{1}{1+e^{-x}}\]")
        assert "σ" in result
        assert r"\[" not in result
        assert "\n" in result


# ── Inline math: \(...\) ──────────────────────────────────────────────────────

class TestInlineBrackets:
    def test_bracket_inline_converted(self):
        result = _render_latex(r"The activation \(\sigma(x)\) is applied.")
        assert "σ" in result
        assert r"\(" not in result

    def test_no_extra_newlines_for_inline(self):
        result = _render_latex(r"Value \(\alpha\) is small.")
        assert "\n" not in result


# ── Robustness ────────────────────────────────────────────────────────────────

class TestRenderLatexRobustness:
    def test_malformed_latex_does_not_raise(self):
        # Should not crash — worst case returns formula unchanged
        result = _render_latex(r"$\notacommand{x}$")
        assert result is not None

    def test_fallback_when_pylatexenc_missing(self):
        # Simulate pylatexenc not being installed
        with patch.dict(sys.modules, {
            "pylatexenc": None,
            "pylatexenc.latex2text": None,
        }):
            text = r"$\frac{1}{n}$"
            # Re-import is needed to trigger the ImportError path
            import importlib
            import src.chat.terminal as term_mod
            importlib.reload(term_mod)
            result = term_mod._render_latex(text)
            assert result == text  # returned unchanged

    def test_real_mse_formula(self):
        formula = (
            r"$$MSE = \frac{1}{n}\sum_{i=1}^{n}"
            r"(y_{\mathrm{true}} - y_{\mathrm{pred}})^2$$"
        )
        result = _render_latex(formula)
        assert "∑" in result
        assert "MSE" in result
        assert "1/n" in result
        assert "$$" not in result
