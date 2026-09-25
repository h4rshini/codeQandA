"""Step 1: walk a repo, chunk files, embed chunks, store them in chromadb.

Usage: python -m codeqa.indexer /path/to/repo
"""
import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

import chromadb

from codeqa import config


@dataclass
class Chunk:
    path: str        # relative to repo root, posix style
    start_line: int  # 1-indexed, inclusive
    end_line: int    # inclusive
    symbol: str      # e.g. "MyClass.method", "<module>", "<window>"
    text: str

    @property
    def id(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


# ---------- file discovery ----------

def iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in config.SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        if p.is_file() and p.suffix in config.TEXT_EXTS and p.stat().st_size <= config.MAX_FILE_BYTES:
            yield p


# ---------- chunking ----------

def window_chunks(path: str, lines: list[str], start: int, end: int, symbol="<window>"):
    """Fixed-size overlapping windows over lines[start-1:end] (1-indexed, inclusive)."""
    step = config.WINDOW_LINES - config.WINDOW_OVERLAP
    s = start
    while s <= end:
        e = min(s + config.WINDOW_LINES - 1, end)
        text = "\n".join(lines[s - 1:e])
        if text.strip():
            yield Chunk(path, s, e, symbol, text)
        if e == end:
            break
        s += step


def _node_start(node) -> int:
    # include decorators in the chunk
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])


def python_chunks(path: str, source: str):
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        yield from window_chunks(path, lines, 1, len(lines))
        return

    covered: list[tuple[int, int]] = []
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in tree.body:
        if not isinstance(node, defs):
            continue
        s, e = _node_start(node), node.end_lineno
        covered.append((s, e))
        big_class = isinstance(node, ast.ClassDef) and (e - s + 1) > config.MAX_CHUNK_LINES
        if not big_class:
            yield Chunk(path, s, e, node.name, "\n".join(lines[s - 1:e]))
            continue
        # Large class: one chunk per method, plus the class header up to the first method.
        methods = [n for n in node.body if isinstance(n, defs)]
        header_end = (_node_start(methods[0]) - 1) if methods else e
        yield Chunk(path, s, header_end, node.name, "\n".join(lines[s - 1:header_end]))
        for m in methods:
            ms, me = _node_start(m), m.end_lineno
            yield Chunk(path, ms, me, f"{node.name}.{m.name}", "\n".join(lines[ms - 1:me]))

    # Module-level code (imports, constants, `if __name__ == ...`) between defs.
    cursor = 1
    for s, e in sorted(covered) + [(len(lines) + 1, len(lines) + 1)]:
        if s > cursor:
            yield from window_chunks(path, lines, cursor, s - 1, symbol="<module>")
        cursor = max(cursor, e + 1)


def chunk_file(root: Path, file: Path):
    rel = file.relative_to(root).as_posix()
    try:
        source = file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    if file.suffix == ".py":
        yield from python_chunks(rel, source)
    else:
        lines = source.splitlines()
        yield from window_chunks(rel, lines, 1, len(lines))


# ---------- storage ----------

def get_client():
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection():
    return get_client().get_collection(config.COLLECTION)


def build_index(repo: Path) -> int:
    repo = repo.resolve()
    client = get_client()
    try:
        client.delete_collection(config.COLLECTION)  # full rebuild each time: simple and correct
    except Exception:
        pass
    # repo_root is stored on the collection so the tools know where files live.
    col = client.create_collection(
        config.COLLECTION,
        metadata={"repo_root": str(repo), "hnsw:space": "cosine"},
    )

    chunks = [c for f in iter_files(repo) for c in chunk_file(repo, f)]
    batch = 200
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.add(
            ids=[c.id for c in part],
            # Prefix path + symbol so the embedding "knows" where the code lives.
            documents=[f"# {c.path} :: {c.symbol}\n{c.text}" for c in part],
            metadatas=[{"path": c.path, "start_line": c.start_line,
                        "end_line": c.end_line, "symbol": c.symbol} for c in part],
        )
    return len(chunks)


def main():
    ap = argparse.ArgumentParser(description="Index a codebase into chromadb.")
    ap.add_argument("repo", type=Path)
    args = ap.parse_args()
    n = build_index(args.repo)
    print(f"Indexed {n} chunks from {args.repo.resolve()} into {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
