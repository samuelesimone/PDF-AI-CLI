"""Tests for Config — no external dependencies required."""
from pathlib import Path

import pytest

from src.config import Config


class TestConfigDefaults:
    def test_default_llm_model(self):
        c = Config()
        assert c.llm_model == "qwen3.6:27b"

    def test_default_embed_model(self):
        c = Config()
        assert c.embed_model == "bge-m3:latest"

    def test_default_storage_dir_is_path(self):
        c = Config()
        assert isinstance(c.storage_dir, Path)

    def test_default_thinking_disabled(self):
        c = Config()
        assert c.enable_thinking is False


class TestConfigFromEnv:
    def test_reads_llm_model_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "llama3.2:latest")
        c = Config.from_env()
        assert c.llm_model == "llama3.2:latest"

    def test_reads_chunk_size_from_env(self, monkeypatch):
        monkeypatch.setenv("CHUNK_SIZE", "1024")
        c = Config.from_env()
        assert c.chunk_size == 1024

    def test_reads_thinking_flag_from_env(self, monkeypatch):
        monkeypatch.setenv("ENABLE_THINKING", "true")
        c = Config.from_env()
        assert c.enable_thinking is True

    def test_thinking_false_for_other_values(self, monkeypatch):
        monkeypatch.setenv("ENABLE_THINKING", "yes")
        c = Config.from_env()
        assert c.enable_thinking is False

    def test_storage_dir_is_path_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
        c = Config.from_env()
        assert c.storage_dir == tmp_path
