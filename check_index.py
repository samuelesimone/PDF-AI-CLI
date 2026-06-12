#!/usr/bin/env python3
"""
inspect.py — inspect a stored index to verify parsing quality.

Usage:
    python inspect.py <pdf_path>               # summary + all chunks
    python inspect.py <pdf_path> --latex       # only chunks with LaTeX/math
    python inspect.py <pdf_path> --chunk 3     # full text of chunk #3
    python inspect.py <pdf_path> --search "sigmoid"  # grep across chunks
    python inspect.py <pdf_path> --all         # print ALL chunks in full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

load_dotenv()
console = Console()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _highlight_latex(text: str) -> Text:
    """Return a Rich Text object with LaTeX snippets highlighted in yellow."""
    import re
    rich_text = Text(text)
    patterns = [
        r"\$\$.+?\$\$",
        r"\$[^\n$]{2,80}\$",
        r"\\\(.+?\\\)",
        r"\\\[.+?\\\]",
        r"\\begin\{[^}]+\}",
        r"\\(?:frac|sqrt|sum|prod|int|lim|infty|partial|nabla|Delta|Sigma|alpha|beta|gamma|theta|lambda)\b",
        r"```math",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.DOTALL):
            try:
                rich_text.stylize("bold yellow", m.start(), m.end())
            except Exception:
                pass
    return rich_text


def _render_summary(summary: dict) -> None:
    console.print(Rule("[bold]Index Summary[/bold]"))

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", min_width=28)
    grid.add_column(style="bold cyan")

    grid.add_row("Document",        summary["source_doc"])
    grid.add_row("Parser used",     summary.get("parser", "—"))
    grid.add_row("Index directory", summary["index_dir"])
    grid.add_row("Total chunks",    str(summary["total_chunks"]))
    grid.add_row("Avg chunk size",  f"{summary['avg_chunk_chars']} chars")
    grid.add_row("Min / max chunk", f"{summary['min_chunk_chars']} / {summary['max_chunk_chars']} chars")
    grid.add_row("Chunks with math", f"{summary['chunks_with_latex']} "
                 f"({summary['latex_coverage_pct']}% of total)")

    not_decoded = summary.get("formula_not_decoded_count", 0)
    if not_decoded:
        grid.add_row(
            "[yellow]formula-not-decoded[/yellow]",
            f"[yellow]{not_decoded} chunk(s)[/yellow]  ← "
            "re-index with [bold]--parser pymupdf[/bold] to recover formula text",
        )

    breakdown = summary["latex_pattern_breakdown"]
    if breakdown:
        grid.add_row("Math pattern types", ", ".join(
            f"{k}: {v}" for k, v in sorted(breakdown.items(), key=lambda x: -x[1])
        ))
    else:
        grid.add_row("Math pattern types", "[dim]none detected[/dim]")

    console.print(grid)
    console.print()


def _render_chunk_table(chunks, title: str = "Chunks") -> None:
    table = Table(title=title, show_header=True, header_style="bold magenta",
                  expand=True, show_lines=False)
    table.add_column("#",       style="cyan",  width=4,  justify="right")
    table.add_column("Chars",   style="green", width=6,  justify="right")
    table.add_column("LaTeX?",  style="yellow",width=8,  justify="center")
    table.add_column("Types",   style="dim",   width=20)
    table.add_column("Preview", style="white", min_width=40)

    for c in chunks:
        latex_flag = "[bold yellow]✓[/bold yellow]" if c.has_latex else "[dim]—[/dim]"
        types = ", ".join(c.latex_types) if c.latex_types else "—"
        preview = c.text_preview[:100]
        table.add_row(str(c.index), str(c.char_count), latex_flag, types, preview)

    console.print(table)


def _render_full_chunk(chunk) -> None:
    console.print(Rule(f"[bold]Chunk #{chunk.index}[/bold]"))

    meta_grid = Table.grid(padding=(0, 2))
    meta_grid.add_column(style="dim", min_width=12)
    meta_grid.add_column(style="cyan")
    meta_grid.add_row("Node ID", chunk.node_id)
    meta_grid.add_row("Size",    f"{chunk.char_count} chars")
    meta_grid.add_row("LaTeX",   str(chunk.has_latex))
    if chunk.latex_hits:
        meta_grid.add_row("Patterns", ", ".join(chunk.latex_types))
    for k, v in chunk.metadata.items():
        meta_grid.add_row(k, str(v))
    console.print(meta_grid)
    console.print()

    # Show text with LaTeX highlighted
    console.print("[dim]── text ─────────────────────────────────────────────[/dim]")
    console.print(_highlight_latex(chunk.full_text))

    if chunk.latex_hits:
        console.print()
        console.print("[dim]── extracted formulas / math snippets ─────────────[/dim]")
        for label, snippet in chunk.latex_hits:
            console.print(f"  [yellow]{label}[/yellow]: {snippet}")


# ── Main ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="inspect",
        description="Inspect a stored PDF index — verify parsing and LaTeX extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("pdf", type=Path, help="PDF path that was previously indexed")
    p.add_argument("--latex",  action="store_true", help="Show only chunks containing LaTeX/math")
    p.add_argument("--chunk",  type=int, default=None, metavar="N", help="Print full text of chunk #N")
    p.add_argument("--search", type=str, default=None, metavar="QUERY", help="Search text across chunks")
    p.add_argument("--all",    action="store_true", help="Print all chunks in full (verbose)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from src.config import Config
    from src.indexing.indexer import RAGIndexer
    from src.parsers.docling_parser import DoclingParser

    config = Config.from_env()
    # We only need the index path — avoid initialising Ollama
    indexer = RAGIndexer.__new__(RAGIndexer)
    indexer.config = config
    indexer.parser = None

    idx_path = indexer.index_path(args.pdf.resolve())

    if not (idx_path / "docstore.json").exists():
        console.print(f"[red]No index found for:[/red] {args.pdf.name}")
        console.print(f"[dim]Expected at: {idx_path}[/dim]")
        console.print("[dim]Run 'python main.py <pdf>' first.[/dim]")
        sys.exit(1)

    from src.inspection import IndexInspector
    inspector = IndexInspector(idx_path)

    # ── Summary (always shown) ─────────────────────────────────────────────
    _render_summary(inspector.summary())

    # ── --chunk N ──────────────────────────────────────────────────────────
    if args.chunk is not None:
        try:
            _render_full_chunk(inspector.chunk_at(args.chunk))
        except IndexError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        return

    # ── --search QUERY ─────────────────────────────────────────────────────
    if args.search:
        results = inspector.search(args.search)
        console.print(f"[cyan]{len(results)} chunk(s)[/cyan] matching [yellow]{args.search!r}[/yellow]\n")
        if not results:
            return
        _render_chunk_table(results, title=f'Search: "{args.search}"')
        console.print("\n[dim]Use --chunk N to read a chunk in full.[/dim]")
        return

    # ── --latex ────────────────────────────────────────────────────────────
    if args.latex:
        chunks = inspector.latex_chunks()
        if not chunks:
            console.print("[yellow]No chunks with detected LaTeX patterns.[/yellow]")
            console.print("[dim]This may mean: the PDF has no math, or formulas are rasterised images.[/dim]")
            return
        _render_chunk_table(chunks, title="Chunks with LaTeX / math")
        console.print("\n[dim]Use --chunk N to read a chunk in full.[/dim]")
        return

    # ── --all ──────────────────────────────────────────────────────────────
    if args.all:
        for chunk in inspector.chunks:
            _render_full_chunk(chunk)
            console.print()
        return

    # ── Default: table of all chunks ───────────────────────────────────────
    _render_chunk_table(inspector.chunks, title="All chunks")
    console.print("\n[dim]Flags: --latex  --chunk N  --search TEXT  --all[/dim]")


if __name__ == "__main__":
    main()
