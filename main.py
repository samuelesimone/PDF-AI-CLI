#!/usr/bin/env python3
"""
PDF-AI — terminal MVP entry point.

Usage:
    python main.py path/to/book.pdf
    python main.py path/to/book.pdf --model qwen3.5:9b
    python main.py path/to/book.pdf --top-k 8
    python main.py path/to/book.pdf --reindex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Load .env before any library imports so HF_HUB_OFFLINE takes effect
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console

console = Console()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pdf-ai",
        description="Chat with a PDF using a local Ollama LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("pdf", type=Path, help="Path to the PDF file")
    p.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help="Ollama LLM model (default: qwen3.6:27b)",
    )
    p.add_argument(
        "--embed-model",
        default=None,
        metavar="NAME",
        help="Ollama embedding model (default: bge-m3:latest)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="N",
        help="Number of passages to retrieve per query (default: 5)",
    )
    p.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-parsing and re-indexing (ignores cache)",
    )
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Enable qwen3 extended reasoning (slower but deeper)",
    )
    p.add_argument(
        "--parser",
        choices=["docling", "pymupdf", "marker"],
        default="docling",
        help=(
            "PDF parser to use (default: docling). "
            "'pymupdf' extracts raw text page-by-page (fast, no LaTeX). "
            "'marker' uses ML layout analysis to output proper LaTeX formulas "
            "(best for math-heavy academic PDFs, slower on first run)."
        ),
    )
    return p.parse_args()


def _check_pdf(pdf_path: Path) -> None:
    if not pdf_path.exists():
        console.print(f"[red]Error:[/red] File not found: {pdf_path}")
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        console.print(f"[red]Error:[/red] Expected a .pdf file, got: {pdf_path.suffix}")
        sys.exit(1)


def main() -> None:
    args = _parse_args()
    pdf_path = args.pdf.resolve()
    _check_pdf(pdf_path)

    # ── Config ──────────────────────────────────────────────────────────────
    from src.config import Config

    config = Config.from_env()
    if args.model:
        config.llm_model = args.model
    if args.embed_model:
        config.embed_model = args.embed_model
    if args.top_k:
        config.similarity_top_k = args.top_k
    if args.thinking:
        config.enable_thinking = True

    # ── Build pipeline ───────────────────────────────────────────────────────
    from src.indexing import RAGIndexer
    from src.parsers import PARSERS
    from src.rag import RAGEngine
    from src.chat import TerminalChat

    parser = PARSERS[args.parser]()
    console.print(f"[dim]Parser:[/dim] [cyan]{args.parser}[/cyan]")
    indexer = RAGIndexer(config=config, parser=parser)

    if args.reindex:
        indexer.drop_index(pdf_path)
        console.print("[yellow]Existing index removed — re-indexing from scratch.[/yellow]")

    # ── Index ────────────────────────────────────────────────────────────────
    if indexer.is_indexed(pdf_path):
        console.print(
            f"[green]Loading cached index[/green] for [cyan]{pdf_path.name}[/cyan]…"
        )
    else:
        console.print(
            f"[green]Parsing & indexing[/green] [cyan]{pdf_path.name}[/cyan]…\n"
            "[dim]This may take a minute for large PDFs — result will be cached.[/dim]"
        )

    with console.status("[bold cyan]Working…[/bold cyan]", spinner="dots"):
        index = indexer.build_or_load(pdf_path)

    console.print("[bold green]✓ Index ready.[/bold green]")

    # ── Chat ─────────────────────────────────────────────────────────────────
    engine = RAGEngine(index=index, config=config)
    TerminalChat(engine=engine, pdf_path=pdf_path).run()


if __name__ == "__main__":
    main()
