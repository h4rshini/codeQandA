"""Web app: a thin HTTP layer over the agent, tools and eval results.

Run locally: .venv/bin/uvicorn web.app:app --reload   then open http://127.0.0.1:8000
"""
import asyncio
import json
import logging
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from codeqa import config
from codeqa.agent import AllModelsExhausted, run_agent
from codeqa.tools import _collection, read_file, repo_root
from codeqa.tracing import Tracer

log = logging.getLogger("codeqa.web")
STATIC = Path(__file__).parent / "static"
EVALS = config.PROJECT_ROOT / "evals"
MAX_SOURCE_LINES = 120

app = FastAPI(title="codeQandA")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=300)


def _sse(kind: str, data) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/api/ask")
async def ask(body: AskRequest):
    """Run the agent and stream its trace events live, then the final answer (Server-Sent Events)."""
    question = body.question.strip()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(kind, data=None):  # called from the worker thread
        loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

    def work():
        try:
            tracer = Tracer(question, trace_dir=config.TRACE_DIR / "web", on_event=lambda e: emit("trace", e))
            emit("answer", run_agent(question, tracer=tracer).model_dump())
        except AllModelsExhausted:
            emit("error", {"message": "The free API quota is used up for now. Try again later, "
                                      "or pick one of the example questions (they're cached)."})
        except Exception:
            log.exception("agent failed for question %r", question)
            emit("error", {"message": "Something went wrong answering that question. Please try again."})
        finally:
            emit("done")

    loop.run_in_executor(None, work)

    async def stream():
        while True:
            kind, data = await queue.get()
            if kind == "done":
                break
            yield _sse(kind, data)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/source")
def source(path: str, start: int, end: int):
    """Code for a citation. Sandboxed to the repo via read_file."""
    end = min(end, start + MAX_SOURCE_LINES - 1)
    try:
        r = read_file(path, start, end)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))
    lines = [line.split(": ", 1)[1] if ": " in line else "" for line in r["content"].splitlines()]
    return {"path": r["path"], "start": r["lines"][0], "end": r["lines"][1],
            "total_lines": r["total_lines"], "lines": lines}


@app.get("/api/info")
def info():
    spec = yaml.safe_load((EVALS / "questions.yaml").read_text())
    return {"repo": repo_root().name, "chunks": _collection().count(), "model": config.LLM_MODEL,
            "examples": [q["question"] for q in spec["questions"]]}


@app.get("/api/evals")
def evals():
    """Saved eval runs, oldest first, with per-question rows (local file paths stripped)."""
    runs = []
    for summary_path in sorted((EVALS / "results").glob("*_summary.json")):
        run_id = summary_path.name.removesuffix("_summary.json")
        rows = [json.loads(line) for line in (EVALS / "results" / f"{run_id}.jsonl").read_text().splitlines()
                if line.strip()]
        for r in rows:
            r.pop("trace", None)
        runs.append({"summary": json.loads(summary_path.read_text()), "rows": rows})
    return runs


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
