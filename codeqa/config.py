"""Shared settings. Override via environment variables."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = Path(os.environ.get("CODEQA_CHROMA_DIR", PROJECT_ROOT / ".chroma"))
TRACE_DIR = Path(os.environ.get("CODEQA_TRACE_DIR", PROJECT_ROOT / "traces"))
COLLECTION = "codebase"

OPENAI_MODEL = os.environ.get("CODEQA_MODEL", "gpt-4.1-mini")
JUDGE_MODEL = os.environ.get("CODEQA_JUDGE_MODEL", OPENAI_MODEL)

# Chunking
WINDOW_LINES = 60        # fallback window size for non-Python / module-level code
WINDOW_OVERLAP = 10
MAX_CHUNK_LINES = 150    # classes longer than this are split per-method
MAX_FILE_BYTES = 200_000

SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".chroma",
             "dist", "build", ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "traces"}
TEXT_EXTS = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json",
             ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".sh", ".sql", ".rst"}
