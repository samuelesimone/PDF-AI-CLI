#!/usr/bin/env python3
"""
PDF-AI — Web UI entry point.

Serves a browser-based chat interface backed by the same RAG pipeline as
the terminal UI.  LaTeX is rendered client-side with KaTeX; Markdown with
marked.js.

Usage:
    python web_ui.py path/to/document.pdf
    python web_ui.py doc.pdf --model qwen3.5:9b
    python web_ui.py doc.pdf --port 7860
    python web_ui.py doc.pdf --no-browser
    python web_ui.py doc.pdf --pages 1-50 --parser marker
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# ── Shared state (set in main() before uvicorn starts) ────────────────────────

_engine = None
_pdf_path: Optional[Path] = None
_HTML_PATH = Path(__file__).parent / "src" / "web" / "index.html"


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    text: str
    score: float
    page: str


class ChatResponse(BaseModel):
    answer: str
    thinking: Optional[str]
    mode: str
    retrieve_s: float
    generate_s: float
    total_s: float
    sources: List[SourceItem]


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="PDF-AI", docs_url=None, redoc_url=None)


@app.get("/")
async def index():
    return FileResponse(_HTML_PATH)


@app.get("/info")
async def info():
    return {
        "document": _pdf_path.name if _pdf_path else "",
        "llm_model": _engine.config.llm_model if _engine else "",
        "embed_model": _engine.config.embed_model if _engine else "",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    from src.chat.terminal import _detect_mode

    mode = _detect_mode(question)
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _engine.query, question, mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [
        SourceItem(
            text=s.text,
            score=round(s.score, 3),
            page=str(s.metadata.get("page", "—")),
        )
        for s in result.sources
    ]

    return ChatResponse(
        answer=result.answer,
        thinking=result.thinking,
        mode=result.mode,
        retrieve_s=round(result.retrieve_s, 2),
        generate_s=round(result.generate_s, 2),
        total_s=round(result.retrieve_s + result.generate_s, 2),
        sources=sources,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pdf-ai-web",
        description="Chat with a PDF via a local web UI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("pdf", type=Path, help="Path to the PDF file")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="Ollama LLM model (default: from .env)")
    p.add_argument("--embed-model", default=None, metavar="NAME",
                   help="Ollama embedding model (default: from .env)")
    p.add_argument("--top-k", type=int, default=None, metavar="N",
                   help="Number of passages to retrieve per query (default: 5)")
    p.add_argument("--reindex", action="store_true",
                   help="Force re-parsing and re-indexing (ignores cache)")
    p.add_argument("--thinking", action="store_true",
                   help="Enable qwen3 extended reasoning (slower but deeper)")
    p.add_argument("--parser",
                   choices=["docling", "pymupdf", "marker", "mineru"],
                   default="docling",
                   help="PDF parser to use (default: docling)")
    p.add_argument("--pages", default=None, metavar="START-END",
                   help="Page range to extract before parsing, e.g. '1-50'")
    p.add_argument("--port", type=int, default=7860, metavar="PORT",
                   help="Port to serve the web UI on (default: 7860)")
    p.add_argument("--no-browser", action="store_true",
                   help="Do not open a browser window automatically")
    return p.parse_args()


def main() -> None:
    global _engine, _pdf_path

    args = _parse_args()
    pdf_path = args.pdf.resolve()

    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print(f"Error: Expected a .pdf file, got: {pdf_path.suffix}", file=sys.stderr)
        sys.exit(1)

    # ── Config ─────────────────────────────────────────────────────────────────
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

    # ── Page range extraction ──────────────────────────────────────────────────
    if args.pages:
        from main import _parse_page_range, _extract_page_range

        try:
            start, end = _parse_page_range(args.pages)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        pdf_path = _extract_page_range(
            pdf_path, start, end, config.storage_dir, force=args.reindex
        )

    # ── Build pipeline ─────────────────────────────────────────────────────────
    from src.indexing import RAGIndexer
    from src.parsers import PARSERS
    from src.rag import RAGEngine

    print(f"Parser  : {args.parser}")
    parser = PARSERS[args.parser]()
    indexer = RAGIndexer(config=config, parser=parser)

    if args.reindex:
        indexer.drop_index(pdf_path)
        print("Existing index removed — re-indexing from scratch.")

    if indexer.is_indexed(pdf_path):
        print(f"Loading cached index for {pdf_path.name}…")
    else:
        print(f"Parsing & indexing {pdf_path.name}…")
        print("This may take a minute for large PDFs — result will be cached.")

    index = indexer.build_or_load(pdf_path)
    print("Index ready.\n")

    _engine = RAGEngine(index=index, config=config)
    _pdf_path = pdf_path

    url = f"http://localhost:{args.port}"
    print(f"PDF-AI Web UI  →  {url}")
    print(f"Document       →  {pdf_path.name}")
    print(f"LLM            →  {config.llm_model}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        webbrowser.open(url)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
