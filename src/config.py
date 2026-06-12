"""
Configuration management for PDF-AI.

Uses a dataclass to keep settings in one place. Values can be overridden via
environment variables (useful for testing / CI) or by direct attribute mutation
in main.py after parsing CLI args.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.6:27b"
    embed_model: str = "bge-m3:latest"
    temperature: float = 0.1
    request_timeout: float = 600.0

    # ── Chunking ────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Retrieval ───────────────────────────────────────────────────────────
    similarity_top_k: int = 5

    # ── Storage ─────────────────────────────────────────────────────────────
    storage_dir: Path = field(default_factory=lambda: Path("./storage"))

    # ── qwen3 thinking mode (set True to enable extended reasoning) ──────────
    enable_thinking: bool = False

    # ── Figure analysis (optional — empty string disables it) ─────────────
    vision_model: str = ""   # e.g. "qwen3-vl:8b"; set via VISION_MODEL in .env

    @classmethod
    def from_env(cls) -> Config:
        """Create a Config whose defaults are overridden by env vars."""
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            llm_model=os.getenv("LLM_MODEL", "qwen3.6:27b"),
            embed_model=os.getenv("EMBED_MODEL", "bge-m3:latest"),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "600.0")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
            similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "5")),
            storage_dir=Path(os.getenv("STORAGE_DIR", "./storage")),
            enable_thinking=os.getenv("ENABLE_THINKING", "false").lower() == "true",
            vision_model=os.getenv("VISION_MODEL", ""),
        )
