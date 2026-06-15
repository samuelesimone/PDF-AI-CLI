"""
Terminal REPL for chatting with a PDF.

Command pattern: special inputs starting with ':' are dispatched as commands.
Everything else is treated as a question to the RAG engine.

Commands:
  :help / :h       — show help
  :sources / :s    — print retrieved sources from the last answer
  :latex / :l      — toggle LaTeX rendering (unicode ↔ raw LaTeX code)
  :study / :st     — toggle study mode (Socratic tutor with [doc]/[model] tagging)
  :clear / :c      — clear the terminal
  :quit / :q       — exit
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ..config import Config
from ..rag.engine import QueryResult, RAGEngine

console = Console()

_HELP_MD = """\
## PDF-AI Terminal Chat

**Commands**
| Command | Description |
|---------|-------------|
| `:help` `:h` | Show this help |
| `:sources` `:s` | Show retrieved passages from the last answer |
| `:latex` `:l` | Toggle LaTeX rendering: unicode symbols ↔ raw LaTeX code |
| `:study` `:st` | Toggle study mode (Socratic tutor — active recall + source tagging) |
| `:reset` `:r` | Start a new conversation (clears session history) |
| `:clear` `:c` | Clear the screen |
| `:quit` `:q` | Exit |

**Query modes**
| Mode | Activated by | Behaviour |
|------|-------------|-----------|
| `strict` | Auto-detected (factual questions) | Answers only from retrieved passages |
| `explain` | Auto-detected ("Explain…", "How does…", "Derive…") | Passages anchor context; general math knowledge fills derivations |
| `study` | `:study` toggle | Socratic tutor: asks before telling, tags every claim `[doc]` or `[model]`, applies Feynman checks, elaborative interrogation, and Bloom's taxonomy progression |

**Study mode tips**
- Type `:study` to enter/exit study mode.
- The tutor will ask you to attempt an answer before explaining — this is intentional (active recall).
- Every factual claim is tagged `[doc]` (from your PDF) or `[model]` (general knowledge).
- Type `summary` or `done` at the end of a session for a recap and spaced-review plan.

**Tips**
- Ask about any topic, formula, or concept in your PDF.
- Follow up with `:sources` to see exactly which passages were used.
- Use `--top-k N` when launching to retrieve more context.
- Use `:latex` to switch between rendered unicode (α β ∑) and raw LaTeX ($\\alpha$ $\\sum$).
"""

# Keywords that signal an explanation request → explain mode.
# Checked against the lowercased question (prefix or anywhere in the string).
_EXPLAIN_PREFIXES = (
    "explain", "describe", "derive", "derivation",
    "walk me", "what does", "what do", "what is the meaning",
    "what are", "how does", "how do", "how is",
    "give me an intuition", "intuition", "why does", "why is",
    "illustrate", "show me how", "demonstrate",
)


def _detect_mode(question: str) -> str:
    """Return "explain" if the question looks like a concept/formula explanation."""
    q = question.lower().strip()
    for prefix in _EXPLAIN_PREFIXES:
        if q.startswith(prefix):
            return "explain"
    return "strict"


def _render_latex(text: str) -> str:
    """Replace LaTeX math blocks with Unicode approximations via pylatexenc.

    Handles all four delimiter styles:
      $$...$$      display math  (given its own line)
      \\[...\\]    display math  (given its own line)
      \\(...\\)    inline math
      $...$        inline math

    Falls back silently to the original text if pylatexenc is not installed
    or if an individual formula fails to parse.
    """
    try:
        from pylatexenc.latex2text import LatexNodes2Text
    except ImportError:
        return text

    converter = LatexNodes2Text()

    def _convert(formula: str) -> str:
        try:
            return converter.latex_to_text(formula.strip())
        except Exception:
            return formula  # leave raw LaTeX on any conversion error

    # Process display-math delimiters first to avoid single-$ pattern consuming them.

    # 1. $$...$$ — display math
    text = re.sub(
        r'\$\$(.*?)\$\$',
        lambda m: f"\n{_convert(m.group(1))}\n",
        text,
        flags=re.DOTALL,
    )
    # 2. \[...\] — display math
    text = re.sub(
        r'\\\[(.*?)\\\]',
        lambda m: f"\n{_convert(m.group(1))}\n",
        text,
        flags=re.DOTALL,
    )
    # 3. \(...\) — inline math
    text = re.sub(
        r'\\\((.*?)\\\)',
        lambda m: _convert(m.group(1)),
        text,
        flags=re.DOTALL,
    )
    # 4. $...$ — inline math (single $, not preceded or followed by another $)
    text = re.sub(
        r'(?<!\$)\$([^\$\n]+)\$(?!\$)',
        lambda m: _convert(m.group(1)),
        text,
    )
    return text


# ── Command handler (Command pattern) ─────────────────────────────────────────


class _CommandHandler:
    """Dispatches ':command' strings to their handlers."""

    def __init__(self, chat: TerminalChat) -> None:
        self._chat = chat
        self._dispatch = {
            frozenset([":help", ":h"]): self._help,
            frozenset([":sources", ":s"]): self._sources,
            frozenset([":latex", ":l"]): self._latex,
            frozenset([":study", ":st"]): self._study,
            frozenset([":reset", ":r"]): self._reset,
            frozenset([":clear", ":c"]): self._clear,
            frozenset([":quit", ":q", "exit", "quit"]): self._quit,
        }

    def handle(self, cmd: str) -> bool:
        """Return True if *cmd* was a known command."""
        key = cmd.strip().lower()
        for keys, handler in self._dispatch.items():
            if key in keys:
                handler()
                return True
        return False

    def _help(self) -> None:
        console.print(Markdown(_HELP_MD))

    def _sources(self) -> None:
        self._chat._print_sources(self._chat._last_result)

    def _latex(self) -> None:
        self._chat._latex_render = not self._chat._latex_render
        if self._chat._latex_render:
            console.print("[green]LaTeX rendering ON[/green]  — formulas shown as unicode (α β ∑ ∫)")
        else:
            console.print("[yellow]LaTeX rendering OFF[/yellow] — formulas shown as raw LaTeX ($\\alpha$ $\\sum$)")

    def _study(self) -> None:
        self._chat._study_mode = not self._chat._study_mode
        if self._chat._study_mode:
            console.print(
                Panel(
                    "[bold green]Study mode ON[/bold green]\n\n"
                    "The tutor will ask you to attempt an answer before explaining.\n"
                    "Every claim is tagged [bold cyan][doc][/bold cyan] (your PDF) "
                    "or [bold yellow][model][/bold yellow] (general knowledge).\n\n"
                    "Type [yellow]summary[/yellow] or [yellow]done[/yellow] at any point "
                    "for a session recap and spaced-review plan.\n"
                    "Type [yellow]:study[/yellow] again to exit study mode.",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            console.print("[yellow]Study mode OFF[/yellow] — back to normal chat.")

    def _reset(self) -> None:
        self._chat.engine.reset_history()
        console.print("[yellow]Conversation reset.[/yellow] Starting a fresh session.")

    def _clear(self) -> None:
        console.clear()

    def _quit(self) -> None:
        console.print("[yellow]Goodbye![/yellow]")
        sys.exit(0)


# ── Main chat class ────────────────────────────────────────────────────────────


class TerminalChat:
    """Interactive terminal REPL backed by a RAGEngine."""

    def __init__(self, engine: RAGEngine, pdf_path: Path) -> None:
        self.engine = engine
        self.pdf_path = pdf_path
        self._last_result: Optional[QueryResult] = None
        self._latex_render: bool = True   # True = unicode, False = raw LaTeX
        self._study_mode: bool = False    # True = Socratic tutor mode
        self._commands = _CommandHandler(self)

    # ── Rendering helpers ──────────────────────────────────────────────────

    def _apply_latex(self, text: str) -> str:
        """Apply LaTeX → unicode conversion only when rendering is enabled."""
        return _render_latex(text) if self._latex_render else text

    def _print_welcome(self) -> None:
        console.print(
            Panel(
                f"[bold green]PDF-AI Terminal Chat[/bold green]\n\n"
                f"  [dim]Document :[/dim]  [cyan]{self.pdf_path.name}[/cyan]\n"
                f"  [dim]LLM      :[/dim]  [cyan]{self.engine.config.llm_model}[/cyan]\n"
                f"  [dim]Embedder :[/dim]  [cyan]{self.engine.config.embed_model}[/cyan]\n\n"
                f"  Type [yellow]:help[/yellow] for commands, [yellow]:quit[/yellow] to exit.",
                border_style="green",
                padding=(1, 2),
            )
        )

    def _print_answer(self, result: QueryResult) -> None:
        total_s = result.retrieve_s + result.generate_s
        timing = (
            f"[dim]retrieve {result.retrieve_s:.1f}s  ·  "
            f"generate {result.generate_s:.1f}s  ·  "
            f"total {total_s:.1f}s[/dim]"
        )
        if result.mode == "study":
            mode_badge = "  [bold green]study mode[/bold green]"
        elif result.mode == "explain":
            mode_badge = "  [bold yellow]explain mode[/bold yellow]"
        else:
            mode_badge = "  [dim]strict mode[/dim]"

        if result.thinking:
            console.print(
                Panel(
                    self._apply_latex(result.thinking),
                    title=f"[dim]Reasoning (qwen3 thinking mode)[/dim]  {timing}",
                    border_style="dim",
                    style="dim",
                    padding=(1, 2),
                )
            )
        console.print(
            Panel(
                Markdown(self._apply_latex(result.answer)),
                title=f"[bold blue]Answer[/bold blue]{mode_badge}  {timing}",
                border_style="blue",
                padding=(1, 2),
            )
        )
        if result.sources:
            console.print(
                f"[dim]  {len(result.sources)} passage(s) retrieved — "
                "type [yellow]:sources[/yellow] to inspect them.[/dim]"
            )

    def _print_sources(self, result: Optional[QueryResult]) -> None:
        if not result or not result.sources:
            console.print("[dim]No sources available yet.[/dim]")
            return

        table = Table(
            title="Retrieved Sources",
            show_header=True,
            header_style="bold magenta",
            expand=False,
        )
        table.add_column("Score", style="cyan", width=7, justify="right")
        table.add_column("Page", style="green", width=6, justify="center")
        table.add_column("Excerpt", style="white", max_width=90)

        for src in result.sources:
            score_str = f"{src.score:.3f}"
            page_str = str(src.metadata.get("page", "—"))
            excerpt = src.text.replace("\n", " ")
            if len(excerpt) > 120:
                excerpt = excerpt[:117] + "…"
            table.add_row(score_str, page_str, excerpt)

        console.print(table)

    # ── REPL loop ──────────────────────────────────────────────────────────

    def run(self) -> None:
        self._print_welcome()

        while True:
            try:
                user_input = console.input("\n[bold green]You[/bold green] > ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Goodbye![/yellow]")
                break

            if not user_input:
                continue

            # Delegate ':command' inputs
            if user_input.startswith(":") or user_input.lower() in ("exit", "quit"):
                if self._commands.handle(user_input):
                    continue
                console.print(
                    f"[yellow]Unknown command:[/yellow] {user_input!r}  "
                    "(type [yellow]:help[/yellow] for a list)"
                )
                continue

            # RAG query — study mode overrides auto-detection
            if self._study_mode:
                mode = "study"
                spinner_label = "[bold green]Studying…[/bold green]"
            else:
                mode = _detect_mode(user_input)
                spinner_label = (
                    "[bold yellow]Explaining…[/bold yellow]"
                    if mode == "explain"
                    else "[bold cyan]Thinking…[/bold cyan]"
                )
            with console.status(spinner_label, spinner="dots"):
                result = self.engine.query(user_input, mode=mode)

            self._last_result = result
            if result.compacted:
                console.print(
                    "[dim]  ↩ Context compacted — older turns summarised locally "
                    "to stay within the model's context window.[/dim]"
                )
            self._print_answer(result)
