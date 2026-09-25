"""Step 2: tools the LLM can call. Each returns a JSON-serializable dict."""
from functools import lru_cache
from pathlib import Path

from codeqa import config
from codeqa.indexer import get_collection

MAX_SNIPPET_LINES = 80


@lru_cache(maxsize=1)
def _collection():
    return get_collection()


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
