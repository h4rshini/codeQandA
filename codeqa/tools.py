"""Step 2: tools the LLM can call. Each returns a JSON-serializable dict."""
import threading
from pathlib import Path

from codeqa import config
from codeqa.indexer import get_collection

MAX_SNIPPET_LINES = 80


_col = None
_col_lock = threading.Lock()


def _collection():
    """Open the chromadb collection once. Locked: chromadb's client setup isn't thread-safe, and
    the web server's worker threads can hit this simultaneously on the first requests."""
    global _col
    if _col is None:
        with _col_lock:
            if _col is None:
                _col = get_collection()
    return _col


def repo_root() -> Path:
    return Path(_collection().metadata["repo_root"])


def search_code(query: str, k: int = 5) -> dict:
    """Vector search over indexed chunks."""
    res = _collection().query(query_texts=[query], n_results=k)
    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        body = doc.split("\n", 1)[1] if "\n" in doc else doc  # drop the "# path :: symbol" header
        body_lines = body.splitlines()
        if len(body_lines) > MAX_SNIPPET_LINES:
            body = "\n".join(body_lines[:MAX_SNIPPET_LINES]) + "\n... (truncated, use read_file)"
        results.append({
            "path": meta["path"],
            "lines": [meta["start_line"], meta["end_line"]],
            "symbol": meta["symbol"],
            "score": round(1 - dist, 3),  # cosine similarity
            "content": body,
        })
    return {"query": query, "results": results}


MAX_READ_LINES = 200
MAX_LIST_ENTRIES = 200


def _safe_path(rel: str) -> Path:
    """Resolve a repo-relative path; refuse anything that escapes the repo root."""
    root = repo_root().resolve()
    p = (root / rel.lstrip("/")).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"path '{rel}' is outside the repository")
    return p


def read_file(path: str, start_line: int, end_line: int) -> dict:
    """Return lines [start_line, end_line] (1-indexed, inclusive), prefixed with line numbers."""
    p = _safe_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if start_line > len(lines):
        raise ValueError(f"start_line {start_line} is past end of file ({len(lines)} lines)")
    start = max(1, start_line)
    end = min(len(lines), end_line, start + MAX_READ_LINES - 1)
    numbered = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
    return {"path": path, "lines": [start, end], "total_lines": len(lines), "content": numbered}


def list_files(directory: str) -> dict:
    """List files and subdirectories (non-recursive). Use "." for the repo root."""
    d = _safe_path(directory or ".")
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")
    root = repo_root().resolve()
    entries = []
    for p in sorted(d.iterdir()):
        if p.name in config.SKIP_DIRS or p.name.startswith("."):
            continue
        rel = p.relative_to(root).as_posix()
        entries.append(rel + "/" if p.is_dir() else rel)
    return {"directory": directory or ".", "entries": entries[:MAX_LIST_ENTRIES],
            "truncated": len(entries) > MAX_LIST_ENTRIES}


# ---------- schemas the LLM sees ----------

def _fn(name, description, props):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props,
                       "required": list(props), "additionalProperties": False}}}


TOOL_SCHEMAS = [
    _fn("search_code",
        "Semantic search over the codebase. Returns the most relevant code chunks with file path and "
        "line range. Start here when you don't know where something lives.",
        {"query": {"type": "string", "description": "Natural-language description or identifier to search for."}}),
    _fn("read_file",
        f"Read raw lines of a file (1-indexed, inclusive, max {MAX_READ_LINES} lines per call). "
        "Use to verify details or see the context around a search hit.",
        {"path": {"type": "string", "description": "Repo-relative file path, e.g. 'src/app.py'."},
         "start_line": {"type": "integer"},
         "end_line": {"type": "integer"}}),
    _fn("list_files",
        "List files and subdirectories in a directory (non-recursive). Use '.' for the repo root.",
        {"directory": {"type": "string", "description": "Repo-relative directory path."}}),
]

TOOL_FUNCS = {"search_code": search_code, "read_file": read_file, "list_files": list_files}


def execute_tool(name: str, args: dict) -> dict:
    """Run a tool by name. Errors are returned (not raised) so the model can recover."""
    if name not in TOOL_FUNCS:
        return {"error": f"unknown tool '{name}'"}
    try:
        return TOOL_FUNCS[name](**args)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
