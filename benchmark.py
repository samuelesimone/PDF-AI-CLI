#!/usr/bin/env python3
"""
Benchmark PDF parsers on a single document.

Measures parsing speed, formula detection, LaTeX richness, and extrapolates
the estimated time for a target page count (useful before processing large books).

Usage:
    python benchmark.py path/to/document.pdf
    python benchmark.py doc.pdf --parsers docling pymupdf
    python benchmark.py doc.pdf --sample 5
    python benchmark.py doc.pdf --target-pages 800
    python benchmark.py doc.pdf --save results.json
    python benchmark.py doc.pdf --no-samples
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ── Formula detection regexes ─────────────────────────────────────────────────

# Order matters: display before inline so $$ is not consumed by $
_RE_DISPLAY_DOLLAR  = re.compile(r'\$\$(.*?)\$\$',    re.DOTALL)
_RE_INLINE_DOLLAR   = re.compile(r'(?<!\$)\$([^\$\n]+)\$(?!\$)')
_RE_DISPLAY_BRACKET = re.compile(r'\\\[(.*?)\\\]',    re.DOTALL)
_RE_INLINE_BRACKET  = re.compile(r'\\\((.*?)\\\)',    re.DOTALL)

# Counts how many rich LaTeX commands appear (proxy for formula quality)
_RE_RICH_CMDS = re.compile(
    r'\\(?:frac|sum|int|partial|sqrt|nabla|prod|lim|infty|'
    r'alpha|beta|gamma|delta|epsilon|theta|sigma|lambda|mu|'
    r'pi|omega|Sigma|Delta|Omega|times|cdot|leq|geq|neq|'
    r'in|subset|cup|cap|forall|exists|mathbb|mathrm|text)\b'
)

# Docling placeholder when formula extraction fails
_RE_NOT_DECODED = re.compile(r'<!--\s*formula-not-decoded\s*-->', re.IGNORECASE)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ParserResult:
    name:             str
    parse_time_s:     float
    error:            Optional[str]  = None
    total_chars:      int            = 0
    total_words:      int            = 0
    inline_formulas:  int            = 0   # $...$ and \(...\)
    display_formulas: int            = 0   # $$...$$ and \[...\]
    rich_latex_cmds:  int            = 0   # \frac, \sum, etc.
    not_decoded:      int            = 0   # <!-- formula-not-decoded -->
    sample_formulas:  List[str]      = field(default_factory=list)

    @property
    def total_formulas(self) -> int:
        return self.inline_formulas + self.display_formulas

    @property
    def ok(self) -> bool:
        return self.error is None


# ── Analysis helpers ──────────────────────────────────────────────────────────

def _collect_samples(formulas: List[str], n: int) -> List[str]:
    seen: List[str] = []
    for f in formulas:
        f = f.strip()
        if len(f) > 4 and len(seen) < n:
            seen.append(f[:300])
    return seen


def _analyse(text: str, n_samples: int) -> dict:
    display = _RE_DISPLAY_DOLLAR.findall(text) + _RE_DISPLAY_BRACKET.findall(text)
    inline  = _RE_INLINE_DOLLAR.findall(text)  + _RE_INLINE_BRACKET.findall(text)
    return dict(
        total_chars      = len(text),
        total_words      = len(text.split()),
        inline_formulas  = len(inline),
        display_formulas = len(display),
        rich_latex_cmds  = len(_RE_RICH_CMDS.findall(text)),
        not_decoded      = len(_RE_NOT_DECODED.findall(text)),
        sample_formulas  = _collect_samples(display + inline, n_samples),
    )


# ── Page count (reuses PyMuPDF which is already installed) ────────────────────

def _page_count(pdf_path: Path) -> Optional[int]:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return None


# ── Runner ────────────────────────────────────────────────────────────────────

def _run_one(name: str, pdf_path: Path, n_samples: int) -> ParserResult:
    from src.parsers import PARSERS
    parser = PARSERS[name]()
    t0 = time.perf_counter()
    try:
        with console.status(f"  [yellow]Parsing with {name}…[/yellow]", spinner="dots"):
            docs = parser.parse(pdf_path)
        elapsed = time.perf_counter() - t0
        full_text = "\n".join(d.text for d in docs)
        return ParserResult(name=name, parse_time_s=elapsed,
                            **_analyse(full_text, n_samples))
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return ParserResult(name=name, parse_time_s=elapsed, error=str(exc))


# ── Display ───────────────────────────────────────────────────────────────────

def _display(
    results:      List[ParserResult],
    pages:        Optional[int],
    target:       int,
    show_samples: bool,
) -> None:
    console.rule("[bold]Results[/bold]")

    # ── Summary table ──────────────────────────────────────────────────────────
    table = Table(box=box.ROUNDED, highlight=True)
    table.add_column("Parser",      style="bold cyan", no_wrap=True)
    table.add_column("Time (s)",    justify="right")
    table.add_column("Chars",       justify="right")
    table.add_column("Formulas",    justify="right", style="green")
    table.add_column("LaTeX cmds",  justify="right", style="green")
    table.add_column("Not decoded", justify="right")
    if pages:
        table.add_column(f"Est. {target}p (min)", justify="right", style="yellow")
    table.add_column("Status")

    for r in results:
        if not r.ok:
            row: List[str] = [r.name, f"{r.parse_time_s:.1f}",
                               "–", "–", "–", "–"]
            if pages:
                row.append("–")
            row.append("[red]ERROR[/red]")
            table.add_row(*row)
            continue

        not_dec = (f"[red]{r.not_decoded:,}[/red]"
                   if r.not_decoded else "[dim]0[/dim]")
        row = [
            r.name,
            f"{r.parse_time_s:.1f}",
            f"{r.total_chars:,}",
            f"{r.total_formulas:,}",
            f"{r.rich_latex_cmds:,}",
            not_dec,
        ]
        if pages:
            est_min = (r.parse_time_s / pages) * target / 60
            row.append(f"{est_min:.0f}")
        row.append("[green]OK[/green]")
        table.add_row(*row)

    console.print(table)

    if pages:
        console.print(
            f"  [dim]Detected {pages} pages in benchmark file · "
            f"extrapolated to {target} pages[/dim]\n"
        )

    # ── Sample formulas ────────────────────────────────────────────────────────
    if show_samples:
        for r in results:
            if r.ok and r.sample_formulas:
                body = "\n".join(
                    f"  {i+1}.  {f}" for i, f in enumerate(r.sample_formulas)
                )
                console.print(Panel(
                    body,
                    title=f"[cyan]{r.name}[/cyan] · sample formulas",
                    border_style="dim",
                ))
            elif r.ok and r.total_formulas == 0:
                console.print(
                    f"  [dim]{r.name}: no formulas detected "
                    f"(parser does not preserve LaTeX)[/dim]"
                )

    # ── Error details ──────────────────────────────────────────────────────────
    for r in results:
        if not r.ok:
            console.print(
                Panel(
                    r.error or "unknown error",
                    title=f"[red bold]{r.name} — error detail[/red bold]",
                    border_style="red",
                )
            )

    # ── Recommendation ─────────────────────────────────────────────────────────
    ok_results = [r for r in results if r.ok]
    if len(ok_results) > 1:
        best_formula = max(ok_results, key=lambda r: r.rich_latex_cmds)
        fastest      = min(ok_results, key=lambda r: r.parse_time_s)
        console.rule("[bold]Recommendation[/bold]")
        console.print(
            f"  Best formula quality : [cyan]{best_formula.name}[/cyan]"
            f"  ({best_formula.rich_latex_cmds:,} LaTeX commands)"
        )
        console.print(
            f"  Fastest              : [cyan]{fastest.name}[/cyan]"
            f"  ({fastest.parse_time_s:.1f}s)"
        )
        if best_formula.name != fastest.name:
            console.print(
                "\n  For a formula-heavy book use [bold cyan]"
                f"{best_formula.name}[/bold cyan]. "
                "If speed is critical, consider [bold cyan]"
                f"{fastest.name}[/bold cyan] with the trade-off "
                "of lower formula quality."
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark PDF parsers and compare formula extraction quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("pdf", type=Path, help="PDF file to benchmark")
    p.add_argument(
        "--parsers", nargs="+",
        choices=["docling", "pymupdf", "marker", "mineru"],
        default=["docling", "pymupdf", "marker", "mineru"],
        metavar="NAME",
        help="Parsers to run (default: all four)",
    )
    p.add_argument(
        "--sample", type=int, default=3, metavar="N",
        help="Sample formulas to show per parser (default: 3)",
    )
    p.add_argument(
        "--no-samples", action="store_true",
        help="Suppress sample formula panels",
    )
    p.add_argument(
        "--target-pages", type=int, default=800, metavar="N",
        help="Page count to extrapolate time to (default: 800)",
    )
    p.add_argument(
        "--save", type=Path, metavar="FILE",
        help="Save results as JSON",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    if not args.pdf.exists():
        console.print(f"[red]File not found:[/red] {args.pdf}")
        sys.exit(1)

    pages = _page_count(args.pdf)

    console.rule("[bold]PDF Parser Benchmark[/bold]")
    console.print(f"  File         : {args.pdf.name}")
    console.print(f"  Pages        : {pages or 'unknown'}")
    console.print(f"  Parsers      : {', '.join(args.parsers)}")
    console.print(f"  Target pages : {args.target_pages}\n")

    results: List[ParserResult] = []
    for name in args.parsers:
        console.print(f"[bold cyan]{name}[/bold cyan]")
        result = _run_one(name, args.pdf, args.sample)
        results.append(result)
        if result.ok:
            console.print(
                f"  [green]done[/green] — "
                f"{result.parse_time_s:.1f}s · "
                f"{result.total_formulas} formulas · "
                f"{result.rich_latex_cmds} LaTeX cmds\n"
            )
        else:
            console.print(f"  [red]failed[/red] ({result.parse_time_s:.1f}s)\n")

    _display(results, pages, args.target_pages, not args.no_samples)

    if args.save:
        data = []
        for r in results:
            d = dataclasses.asdict(r)
            d["total_formulas"] = r.total_formulas
            d["ok"] = r.ok
            data.append(d)
        args.save.write_text(json.dumps(data, indent=2))
        console.print(f"\n[dim]Results saved → {args.save}[/dim]")


if __name__ == "__main__":
    main()
