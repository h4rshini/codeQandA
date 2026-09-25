"""Step 4: per-query JSONL traces. One event per line, flushed as it happens.

View a trace: python -m codeqa.tracing traces/<file>.jsonl
"""
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from codeqa import config


def summarize_result(name: str, result: dict) -> str:
    """Short human-readable summary of a tool result (full results would bloat traces).
    Never raises: tracing must not be able to break the agent."""
    try:
        return _summarize(name, result)
    except (KeyError, TypeError, IndexError):
        return json.dumps(result, default=str)[:200]


def _summarize(name: str, result: dict) -> str:
    if "error" in result:
        return f"ERROR {result['error']}"
    if name == "search_code":
        hits = [f"{r['path']}:{r['lines'][0]}-{r['lines'][1]} ({r['score']})" for r in result.get("results", [])]
        return f"{len(hits)} hits: " + ", ".join(hits)
    if name == "read_file":
        return f"read {result['path']}:{result['lines'][0]}-{result['lines'][1]} of {result['total_lines']}"
    if name == "list_files":
        return f"{len(result['entries'])} entries" + (" (truncated)" if result["truncated"] else "")
    return json.dumps(result)[:200]


class Tracer:
    def __init__(self, question: str, trace_dir: Path | None = None, run_id: str | None = None):
        trace_dir = Path(trace_dir or config.TRACE_DIR)
        trace_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:40]
        name = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{slug}"
        self.path = trace_dir / f"{name}.jsonl"
        self._f = self.path.open("w", encoding="utf-8")
        self._t0 = time.perf_counter()

    def log(self, type: str, **fields):
        event = {"type": type, "t_ms": round((time.perf_counter() - self._t0) * 1000), **fields}
        self._f.write(json.dumps(event, default=str) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def print_trace(path: Path):
    for line in path.read_text().splitlines():
        e = json.loads(line)
        t = f"{e['t_ms'] / 1000:6.1f}s"
        if e["type"] == "start":
            print(f"{t}  QUESTION: {e['question']}  [model={e['model']}]")
        elif e["type"] == "llm_call":
            reasoning = f"  thinking: {e['reasoning']}" if e.get("reasoning") else ""
            print(f"{t}  llm step {e['step']} ({e['model']}, {e['latency_ms']}ms){reasoning}")
        elif e["type"] == "tool_call":
            print(f"{t}    -> {e['name']}({json.dumps(e['args'])})\n           {e['result_summary']}")
        elif e["type"] == "final":
            print(f"{t}  FINAL ({e['steps']} steps, {e['total_tokens']} tokens):")
            print(json.dumps(e["answer"], indent=2))
        elif e["type"] == "error":
            print(f"{t}  ERROR: {e['error']}")


if __name__ == "__main__":
    print_trace(Path(sys.argv[1]))
