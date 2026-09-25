"""Shared settings. Override via environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # secrets live in .env (gitignored), never in code
CHROMA_DIR = Path(os.environ.get("CODEQA_CHROMA_DIR", PROJECT_ROOT / ".chroma"))
TRACE_DIR = Path(os.environ.get("CODEQA_TRACE_DIR", PROJECT_ROOT / "traces"))
COLLECTION = "codebase"

# Any OpenAI-compatible API works. Default: Google Gemini (free tier).
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
LLM_MODEL = os.environ.get("CODEQA_MODEL", "gemini-flash-latest")
JUDGE_MODEL = os.environ.get("CODEQA_JUDGE_MODEL", LLM_MODEL)
# Tried in order if the main model is overloaded (free tier returns 503/429 under load).
FALLBACK_MODELS = [m for m in os.environ.get(
    "CODEQA_FALLBACK_MODELS", "gemini-3.5-flash,gemini-flash-lite-latest").split(",") if m]

# Chunking
WINDOW_LINES = 60        # fallback window size for non-Python / module-level code
WINDOW_OVERLAP = 10
MAX_CHUNK_LINES = 150    # classes longer than this are split per-method
MAX_FILE_BYTES = 200_000

SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".chroma",
             "dist", "build", ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "traces"}
SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock"}
TEXT_EXTS = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json",
             ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".sh", ".sql", ".rst"}
