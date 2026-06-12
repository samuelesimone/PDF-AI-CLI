"""
Tests for TerminalChat session state and the :latex toggle command.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.chat.terminal import TerminalChat


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_chat() -> TerminalChat:
    """Return a TerminalChat with a mocked engine and pdf path."""
    engine = MagicMock()
    engine.config.llm_model = "test-model"
    engine.config.embed_model = "test-embed"
    return TerminalChat(engine=engine, pdf_path=Path("test.pdf"))


# ── Default state ──────────────────────────────────────────────────────────────

class TestLatexRenderDefault:
    def test_latex_render_is_true_by_default(self):
        chat = _make_chat()
        assert chat._latex_render is True

    def test_apply_latex_converts_when_enabled(self):
        chat = _make_chat()
        result = chat._apply_latex(r"The value $\alpha$ is small.")
        assert "α" in result
        assert r"\alpha" not in result

    def test_apply_latex_passthrough_when_disabled(self):
        chat = _make_chat()
        chat._latex_render = False
        raw = r"The value $\alpha$ is small."
        assert chat._apply_latex(raw) == raw


# ── :latex toggle command ──────────────────────────────────────────────────────

class TestLatexToggleCommand:
    def test_latex_command_turns_rendering_off(self):
        chat = _make_chat()
        with patch("src.chat.terminal.console"):
            chat._commands.handle(":latex")
        assert chat._latex_render is False

    def test_latex_command_turns_rendering_back_on(self):
        chat = _make_chat()
        chat._latex_render = False
        with patch("src.chat.terminal.console"):
            chat._commands.handle(":latex")
        assert chat._latex_render is True

    def test_latex_alias_l_works(self):
        chat = _make_chat()
        with patch("src.chat.terminal.console"):
            chat._commands.handle(":l")
        assert chat._latex_render is False

    def test_latex_toggle_is_idempotent_over_two_calls(self):
        chat = _make_chat()
        with patch("src.chat.terminal.console"):
            chat._commands.handle(":latex")
            chat._commands.handle(":latex")
        assert chat._latex_render is True

    def test_latex_command_prints_on_message(self):
        chat = _make_chat()
        chat._latex_render = False  # currently off → toggling ON
        with patch("src.chat.terminal.console") as mock_console:
            chat._commands.handle(":latex")
        printed = mock_console.print.call_args[0][0]
        assert "ON" in printed

    def test_latex_command_prints_off_message(self):
        chat = _make_chat()
        chat._latex_render = True  # currently on → toggling OFF
        with patch("src.chat.terminal.console") as mock_console:
            chat._commands.handle(":latex")
        printed = mock_console.print.call_args[0][0]
        assert "OFF" in printed

    def test_latex_command_is_recognised_by_handler(self):
        chat = _make_chat()
        with patch("src.chat.terminal.console"):
            recognised = chat._commands.handle(":latex")
        assert recognised is True
