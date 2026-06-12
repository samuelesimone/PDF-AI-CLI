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
import time
from pathlib import Path
from typing import Optional

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
    p.add_argument(
        "--vision-model",
        default=None,
        metavar="NAME",
        help=(
            "Ollama vision model for figure analysis (e.g. qwen3-vl:8b). "
            "Overrides VISION_MODEL from .env. "
            "Figures are extracted and described automatically when building a new index."
        ),
    )
    p.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure analysis even if VISION_MODEL is configured in .env.",
    )
    return p.parse_args()


def _get_page_count(pdf_path: Path) -> Optional[int]:
    """Return the PDF page count via PyMuPDF, or None if unavailable."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return None


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
    if args.vision_model:
        config.vision_model = args.vision_model
    if args.no_figures:
        config.vision_model = ""

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

    # ── Figure analysis (runs only when building a new index) ────────────────
    figure_docs = []
    if config.vision_model and not indexer.is_indexed(pdf_path):
        from src.figures import FigureAnalyzer, extract_figures
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn,
        )

        console.print(f"[dim]Scanning for figures…[/dim]")
        figures = extract_figures(pdf_path)

        if figures:
            console.print(
                f"[dim]Found [/dim][cyan]{len(figures)}[/cyan][dim] figure(s) — "
                f"analyzing with [/dim][cyan]{config.vision_model}[/cyan]"
            )
            analyzer = FigureAnalyzer(
                model=config.vision_model,
                base_url=config.ollama_base_url,
                timeout=config.request_timeout,
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                console=console,
                transient=True,
            ) as progress:
                task_id = progress.add_task("Analyzing figures", total=len(figures))

                def _on_progress(current: int, total: int, page: int) -> None:
                    progress.update(
                        task_id, completed=current,
                        description=f"Analyzing figures  [dim][page {page}][/dim]",
                    )

                figure_docs = analyzer.analyze_all(
                    figures, pdf_path, progress_callback=_on_progress,
                )

            n_ok = len(figure_docs)
            n_fail = len(figures) - n_ok
            summary = f"[green]✓[/green] [dim]{n_ok} figure description(s) added to index"
            if n_fail:
                summary += f" · {n_fail} skipped"
            console.print(summary + "[/dim]")
        else:
            console.print("[dim]No figures detected in this PDF.[/dim]")

    # ── Index ────────────────────────────────────────────────────────────────
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    _prog_cols = (
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        TimeElapsedColumn(),
    )

    if indexer.is_indexed(pdf_path):
        t0 = time.perf_counter()
        with console.status("[bold cyan]Loading cached index…[/bold cyan]", spinner="dots"):
            index = indexer.load_index(pdf_path)
        console.print(
            f"[bold green]✓ Cached index loaded[/bold green] "
            f"[dim]({time.perf_counter() - t0:.1f}s)[/dim]"
        )
    else:
        page_count = _get_page_count(pdf_path)
        if page_count:
            console.print(
                f"[dim]Building index for the first time — "
                f"{page_count} page(s), result will be cached.[/dim]"
            )

        # Step 1: Parse text
        parse_s: float = 0.0
        with Progress(*_prog_cols, console=console) as prog:
            prog.add_task(f"Parsing with {args.parser}…", total=None)
            t0 = time.perf_counter()
            parsed = parser.parse(pdf_path)
            parse_s = time.perf_counter() - t0

        all_docs = list(parsed) + figure_docs
        fig_note = (
            f" + {len(figure_docs)} figure description(s)" if figure_docs else ""
        )
        console.print(
            f"[green]✓[/green] [dim]Parsed {len(parsed)} text chunk(s)"
            f"{fig_note} · {parse_s:.1f}s[/dim]"
        )

        # Step 2: Embed chunks and build the vector index
        embed_s: float = 0.0
        with Progress(*_prog_cols, console=console) as prog:
            prog.add_task(f"Embedding {len(all_docs)} chunk(s)…", total=None)
            t1 = time.perf_counter()
            index = indexer.index_documents(pdf_path, all_docs)
            embed_s = time.perf_counter() - t1

        console.print(
            f"[green]✓[/green] [dim]Indexed · {embed_s:.1f}s "
            f"· total {parse_s + embed_s:.1f}s[/dim]"
        )

    console.print("[bold green]✓ Index ready.[/bold green]")

    # ── Chat ─────────────────────────────────────────────────────────────────
    engine = RAGEngine(index=index, config=config)
    TerminalChat(engine=engine, pdf_path=pdf_path).run()


if __name__ == "__main__":
    main()
