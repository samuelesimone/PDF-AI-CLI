#!/usr/bin/env python3
"""
PDF-AI — terminal MVP entry point.

Usage:
    python main.py path/to/book.pdf
    python main.py path/to/book.pdf --model qwen3.5:9b
    python main.py path/to/book.pdf --top-k 8
    python main.py path/to/book.pdf --reindex
    python main.py path/to/book.pdf --pages 1-50
    python main.py path/to/book.pdf --pages 10-80 --parser marker
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
        choices=["docling", "pymupdf", "marker", "mineru"],
        default="docling",
        help=(
            "PDF parser to use (default: docling). "
            "'pymupdf' extracts raw text page-by-page (fast, no LaTeX). "
            "'marker' uses ML layout analysis to output proper LaTeX formulas "
            "(best for math-heavy academic PDFs, slower on first run). "
            "'mineru' uses MinerU VLM pipeline for high-accuracy LaTeX extraction "
            "(backend controlled by MINERU_BACKEND env var: pipeline / vlm-engine / hybrid-engine)."
        ),
    )
    p.add_argument(
        "--pages",
        default=None,
        metavar="START-END",
        help=(
            "Page range to extract before parsing, e.g. '1-50' or '10-80'. "
            "Both numbers are 1-based and inclusive. "
            "The extracted PDF is cached in storage/pages/ so subsequent runs "
            "with the same range reuse it without re-extracting."
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


def _parse_page_range(pages: str) -> tuple[int, int]:
    """Parse '1-50' into (1, 50). Raises ValueError on bad input."""
    parts = pages.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --pages value {pages!r}. "
            "Expected format: START-END (e.g. '1-50')."
        )
    start, end = int(parts[0]), int(parts[1])
    if start < 1 or end < start:
        raise ValueError(
            f"Invalid page range {pages!r}: "
            "START must be ≥ 1 and END must be ≥ START."
        )
    return start, end


def _extract_page_range(
    pdf_path: Path,
    start: int,
    end: int,
    storage_dir: Path,
    force: bool = False,
) -> Path:
    """Extract pages [start, end] (1-based, inclusive) to a stable cached file.

    The extracted PDF is saved to storage/pages/{stem}_p{start}-{end}.pdf so
    subsequent runs with the same range hit the cache instantly.

    Args:
        pdf_path:    Original PDF.
        start, end:  1-based, inclusive page range.
        storage_dir: Root storage directory (from config).
        force:       If True, re-extract even if the cached file exists.
    """
    import fitz  # PyMuPDF — already a hard dependency

    pages_dir = storage_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    out_path = pages_dir / f"{pdf_path.stem}_p{start}-{end}.pdf"

    if out_path.exists() and not force:
        return out_path

    doc = fitz.open(str(pdf_path))
    total = len(doc)

    # Clamp to actual page count (1-based)
    clamped_end = min(end, total)
    if clamped_end < end:
        console.print(
            f"[yellow]Warning:[/yellow] document has {total} pages; "
            f"clamping end from {end} to {clamped_end}."
        )

    extracted = fitz.open()
    extracted.insert_pdf(doc, from_page=start - 1, to_page=clamped_end - 1)
    extracted.save(str(out_path))
    extracted.close()
    doc.close()

    return out_path


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

    # ── Page range extraction ────────────────────────────────────────────────
    if args.pages:
        try:
            start, end = _parse_page_range(args.pages)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)

        console.print(
            f"[dim]Pages:[/dim] [cyan]{start}–{end}[/cyan]  "
            f"[dim](extracting from {pdf_path.name})[/dim]"
        )
        pdf_path = _extract_page_range(
            pdf_path, start, end, config.storage_dir,
            force=args.reindex,
        )
        console.print(
            f"[dim]Extracted:[/dim] [cyan]{pdf_path.name}[/cyan]"
        )

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
