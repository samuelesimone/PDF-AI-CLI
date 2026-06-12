# PDF-AI

A fully local, privacy-first terminal chatbot for studying PDF documents.
Ask questions, get cited answers, and explore mathematical formulas — no data ever leaves your machine.

---

## How it works

```
PDF file
   │
   ▼  (optional)
--pages flag  →  extract page range  →  storage/pages/<name>_p{start}-{end}.pdf
   │
   ▼
Parser  (Docling / PyMuPDF / marker-pdf / MinerU)
   │  extracts text + LaTeX formulas
   ▼
Indexer  (llama-index + SentenceSplitter)
   │  embeds chunks via Ollama embedding model → vector store
   ▼
Vector Store  (persisted in ./storage/)
   │
   ▼  at query time
Retriever  →  top-k relevant chunks
   │
   ▼
LLM (Ollama)  →  synthesises answer  →  Terminal UI (Rich)
```

Everything runs on **localhost**. The only network calls are to Ollama on `http://localhost:11434`.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.12 recommended |
| [Ollama](https://ollama.com) | Must be running before you start |
| An LLM pulled in Ollama | e.g. `ollama pull qwen3.5:9b` |
| An embedding model pulled | e.g. `ollama pull bge-m3:latest` |

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd PDF-AI

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the environment template and edit it
cp .env.example .env              # if .env.example exists, otherwise create .env manually
```

### Minimum `.env` file

```ini
LLM_MODEL=qwen3.5:9b
EMBED_MODEL=bge-m3:latest
OLLAMA_BASE_URL=http://localhost:11434
REQUEST_TIMEOUT=600
HF_HUB_OFFLINE=1
```

---

## Configuration (`.env`)

All settings can be overridden via environment variables.
Copy the table below into your `.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `qwen3.6:27b` | Ollama LLM used for answering |
| `EMBED_MODEL` | `bge-m3:latest` | Ollama embedding model for indexing |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `REQUEST_TIMEOUT` | `600` | Seconds before an LLM call times out |
| `CHUNK_SIZE` | `512` | Characters per text chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `SIMILARITY_TOP_K` | `5` | Passages retrieved per query |
| `STORAGE_DIR` | `./storage` | Where vector indexes are cached |
| `HF_HUB_OFFLINE` | `1` | Block outbound HuggingFace requests |
| `ENABLE_THINKING` | `false` | Enable qwen3 extended reasoning by default |
| `MINERU_BACKEND` | `pipeline` | MinerU backend: `pipeline` (default), `vlm-engine`, `hybrid-engine` |

> **Important:** Changing `EMBED_MODEL` invalidates all cached indexes.
> Run with `--reindex` after switching embedding models.
> Changing `LLM_MODEL` requires no re-indexing.

---

## Usage

### Terminal chat

```bash
python main.py path/to/document.pdf
```

### Web UI

```bash
python web_ui.py path/to/document.pdf
```

Opens a browser-based chat interface at `http://localhost:7860`.
LaTeX equations are rendered with **KaTeX** directly in the browser.
All other options (`--model`, `--parser`, `--pages`, `--reindex`, etc.) work identically to the terminal version.

```bash
# Choose a port
python web_ui.py doc.pdf --port 8080

# Do not open browser automatically
python web_ui.py doc.pdf --no-browser

# Combine with any main.py option
python web_ui.py book.pdf --pages 1-50 --parser marker --model qwen3.5:9b
```

> **Note:** The web UI loads KaTeX and marked.js from a CDN on first use (then browser-cached). All PDF processing and LLM inference remain fully local.

---

### Chat with a PDF (terminal)

```bash
python main.py path/to/document.pdf
```

### Common options

```bash
# Choose a different LLM for this session only
python main.py doc.pdf --model qwen3.5:9b

# Retrieve more context passages per query (default: 5)
python main.py doc.pdf --top-k 8

# Force re-parsing and re-indexing (ignore cache)
python main.py doc.pdf --reindex

# Enable qwen3 extended thinking (deeper reasoning, slower)
python main.py doc.pdf --thinking

# Choose a PDF parser (see Parsers section below)
python main.py doc.pdf --parser marker
python main.py doc.pdf --parser pymupdf
python main.py doc.pdf --parser mineru
python main.py doc.pdf --parser docling   # default
```

### Page range — study a chapter without modifying the source

```bash
# Index only pages 1 to 50
python main.py book.pdf --pages 1-50

# A specific chapter with a better parser for formulas
python main.py book.pdf --pages 100-200 --parser marker

# Combine with --reindex to force re-extraction and re-indexing
python main.py book.pdf --pages 1-50 --reindex
```

**How caching works with `--pages`:**

| Run | What happens |
|---|---|
| First run with `--pages 1-50` | Extracts pages → `storage/pages/book_p1-50.pdf`, then parses and indexes |
| Same range again | Skips extraction, loads the cached index instantly |
| Different range `--pages 51-100` | New extracted file, new index — both coexist in cache |
| `--pages 1-50 --reindex` | Re-extracts the PDF and rebuilds the index from scratch |

The original PDF is never modified.

---

## Parsers

Four parsers are available via `--parser`. The choice affects formula quality.
**Re-indexing is required when switching parsers.**

| Parser | `--parser` flag | Formula quality | Speed | Best for |
|---|---|---|---|---|
| **Docling** | `docling` (default) | Medium — detects formulas but some are dropped as `<!-- formula-not-decoded -->` | Slow | General PDFs, tables, scanned docs |
| **PyMuPDF** | `pymupdf` | Low — extracts raw text, linearises 2D layout (fractions and subscripts may be garbled) | Fast | Simple text-heavy PDFs |
| **marker-pdf** | `marker` | High — ML-based layout analysis, outputs proper LaTeX (`\frac`, `\sum`, etc.) | Slow on first run (downloads ~3.5 GB of models, then cached) | Academic/LaTeX-compiled PDFs with maths |
| **MinerU** | `mineru` | High — VLM pipeline (MinerU2.5), Apache 2.0 licence | pipeline: moderate · vlm-engine / hybrid-engine: higher accuracy | Math-heavy books and papers; strong alternative to marker |

### Recommended for maths-heavy papers

```bash
python main.py paper.pdf --parser marker --reindex
# or
python main.py paper.pdf --parser mineru --reindex
```

---

## Terminal commands

Once inside the chat session:

| Command | Aliases | Description |
|---|---|---|
| `:help` | `:h` | Show the help panel |
| `:sources` | `:s` | Show the retrieved passages used for the last answer |
| `:latex` | `:l` | Toggle LaTeX rendering: unicode symbols (α β ∑) ↔ raw LaTeX code |
| `:clear` | `:c` | Clear the terminal screen |
| `:quit` | `:q`, `exit`, `quit` | Exit the chat |

---

## Query modes

The engine automatically selects a mode based on your question phrasing.

| Mode | Triggered by | Behaviour |
|---|---|---|
| **strict** | Factual questions ("What accuracy did the model achieve?") | Answers **only** from retrieved passages — never invents information |
| **explain** | Explanation questions ("Explain the MSE formula", "How does the LSTM gate work?", "Derive…") | Passages anchor paper-specific context; the LLM may use general mathematical knowledge for derivations and intuition |

Both modes display a badge in the answer panel:

```
╭── Answer  explain mode  retrieve 0.1s · generate 38.4s · total 38.5s ──╮
│  ...                                                                    │
╰─────────────────────────────────────────────────────────────────────────╯
```

---

## LaTeX rendering

Mathematical formulas in answers are automatically converted from LaTeX to Unicode for terminal display via `pylatexenc`:

```
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_{\text{true}} - y_{\text{pred}})^2$$
  →  MSE = 1/n∑_i=1^n(ytrue - ypred)^2
```

Greek letters (`α β γ θ σ ∇ ∈ ℝ`), operators (`∑ ∏ ∫`), and symbols (`× ÷ ≤ ≥`) render as Unicode. Subscripts and superscripts are approximated as `_` and `^`.

Toggle rendering on/off at any time with `:latex` (or `:l`) to switch between unicode display and raw LaTeX code.

---

## Parser benchmark tool

Before processing a large document, compare parsers on a representative sample to find the right speed/quality trade-off.

```bash
# Run all four parsers and compare
python benchmark.py path/to/sample.pdf

# Only specific parsers (skip slow ones)
python benchmark.py sample.pdf --parsers docling pymupdf

# Show 5 sample formulas per parser instead of 3
python benchmark.py sample.pdf --sample 5

# Extrapolate times to a 1200-page book instead of the default 800
python benchmark.py sample.pdf --target-pages 1200

# Save results as JSON for later comparison
python benchmark.py sample.pdf --save results.json

# Suppress the formula sample panels
python benchmark.py sample.pdf --no-samples
```

**Output columns explained:**

| Column | Meaning |
|---|---|
| Time (s) | Wall-clock seconds to parse the benchmark file |
| Chars | Total characters extracted |
| Formulas | Detected `$...$`, `$$...$$`, `\[...\]`, `\(...\)` environments |
| LaTeX cmds | Occurrences of `\frac`, `\sum`, `\int`, etc. — proxy for formula richness |
| Not decoded | Docling `<!-- formula-not-decoded -->` failures |
| Est. 800p (min) | Extrapolated minutes for an 800-page document |

The script ends with an automatic recommendation based on formula richness vs. speed.

---

## Index inspection tool

After indexing a PDF, you can inspect the stored chunks without starting a full chat session:

```bash
# Summary + table of all chunks
python check_index.py path/to/document.pdf

# Show only chunks that contain mathematical formulas
python check_index.py document.pdf --latex

# Print the full text of chunk number 12
python check_index.py document.pdf --chunk 12

# Search for a keyword across all chunks
python check_index.py document.pdf --search "LSTM"

# Print all chunks in full (verbose)
python check_index.py document.pdf --all
```

The summary shows total chunks, average chunk size, and how many contain detected LaTeX patterns. If `formula-not-decoded` entries appear (Docling limitation), the tool recommends switching to `--parser marker`.

---

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

All tests are fully mocked — no Ollama server, no real PDF files, and no model downloads are needed to run the test suite.

```
149 passed in ~6s
```

---

## Project structure

```
PDF-AI/
├── main.py                  # Terminal entry point — CLI + pipeline wiring
├── web_ui.py                # Web UI entry point — FastAPI + browser chat
├── benchmark.py             # Parser benchmark CLI
├── check_index.py           # Index inspection CLI
├── .env                     # Local configuration (not committed)
├── requirements.txt
├── requirements-dev.txt
│
├── src/
│   ├── config.py            # Config dataclass, loaded from .env
│   ├── parsers/
│   │   ├── base.py          # Strategy interface (BasePDFParser)
│   │   ├── docling_parser.py
│   │   ├── pymupdf_parser.py
│   │   ├── marker_parser.py
│   │   └── mineru_parser.py
│   ├── indexing/
│   │   └── indexer.py       # RAGIndexer — build/load vector store
│   ├── rag/
│   │   └── engine.py        # RAGEngine facade — retrieval + generation
│   ├── chat/
│   │   └── terminal.py      # Terminal REPL, LaTeX rendering, :latex toggle
│   ├── web/
│   │   └── index.html       # Single-page web UI (KaTeX + marked.js)
│   └── inspection/
│       └── inspector.py     # IndexInspector — reads docstore.json
│
├── tests/                   # pytest test suite (149 tests)
└── storage/                 # Auto-generated, git-ignored
    ├── pages/               # Extracted page-range PDFs (--pages cache)
    └── <name>_<hash>/       # Vector indexes per document
```

---

## Privacy

| Component | Where it runs |
|---|---|
| LLM inference | Ollama on `localhost` |
| Embeddings | Ollama on `localhost` |
| PDF parsing | Local process |
| Page extraction | Local process (PyMuPDF) |
| Vector storage | Local filesystem (`./storage/`) |
| Model downloads | One-time, then `HF_HUB_OFFLINE=1` blocks further requests |
| Web UI rendering | `localhost` (FastAPI + Uvicorn) |
| KaTeX / marked.js | CDN on first load only, then browser-cached — no document data sent |

No document content or queries are sent to any external API.
