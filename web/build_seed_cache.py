"""Build web/seed_cache.json from an eval run, so example questions replay instantly and free.

Usage: python -m web.build_seed_cache <run_id>
"""
import json
import re
import sys

from codeqa import config
from web.guard import normalize

EVALS = config.PROJECT_ROOT / "evals" / "results"
OUT = config.PROJECT_ROOT / "web" / "seed_cache.json"
KEEP = {"start", "llm_call", "tool_call"}  # what the UI renders; drop final/validation noise


def meta_from_summary(name: str, summary: str) -> dict | None:
    """Rebuild structured tool meta from the text summary (older traces only have the text)."""
    if name == "search_code":
        hits = re.findall(r"([^\s,]+):(\d+)-(\d+) \(([\d.]+)\)", summary)
        return {"hits": [{"path": p, "lines": [int(a), int(b)], "score": float(s)} for p, a, b, s in hits]}
    if name == "read_file" and (m := re.match(r"read (\S+):(\d+)-(\d+) of (\d+)", summary)):
        return {"path": m[1], "lines": [int(m[2]), int(m[3])], "total_lines": int(m[4])}
    if name == "list_files" and (m := re.match(r"(\d+) entries", summary)):
        return {"directory": "", "count": int(m[1])}
    return None


def main(run_id: str):
    rows = [json.loads(line) for line in (EVALS / f"{run_id}.jsonl").read_text().splitlines() if line.strip()]
    trace_dir = config.TRACE_DIR / f"eval-{run_id}"
    seed = {}
    for r in rows:
        trace = trace_dir / f"{r['id']}.jsonl"
        if "error" in r or not trace.exists():
            continue
        events = [e for e in map(json.loads, trace.read_text().splitlines()) if e["type"] in KEEP]
        for e in events:
            if e["type"] == "tool_call" and not e.get("meta") and not e.get("error"):
                e["meta"] = meta_from_summary(e["name"], e["result_summary"])
                if e["name"] == "list_files" and e["meta"]:
                    e["meta"]["directory"] = e["args"].get("directory", "")
        seed[normalize(r["question"])] = {"events": events, "answer": r["answer"]}
    OUT.write_text(json.dumps(seed, indent=1))
    print(f"Wrote {len(seed)} cached answers to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
