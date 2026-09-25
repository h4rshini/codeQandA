"""Web app: a thin HTTP layer over the agent, tools and eval results.

Run locally: .venv/bin/uvicorn web.app:app --reload   then open http://127.0.0.1:8000
"""
import asyncio
import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from codeqa import config
from codeqa.agent import AllModelsExhausted, run_agent
from codeqa.tools import _collection, read_file, repo_root
from codeqa.tracing import Tracer
from web.guard import AnswerCache, DailyCap, RateLimiter

log = logging.getLogger("codeqa.web")
STATIC = Path(__file__).parent / "static"
EVALS = config.PROJECT_ROOT / "evals"
MAX_SOURCE_LINES = 120
SEED_CACHE = Path(__file__).parent / "seed_cache.json"

# Public-demo limits (override with env vars). Cached answers never count against them.
PER_HOUR = int(os.environ.get("CODEQA_WEB_PER_HOUR", "5"))      # new questions per visitor per hour
DAILY = int(os.environ.get("CODEQA_WEB_DAILY", "60"))           # new questions per day, all visitors
CONCURRENCY = int(os.environ.get("CODEQA_WEB_CONCURRENCY", "2"))  # live agent runs at once

# The web app defaults to the model the eval measured (and the cached answers came from):
# it's also the fastest on the free tier. CODEQA_MODEL / CODEQA_FALLBACK_MODELS still override.
if not os.environ.get("CODEQA_MODEL"):
    config.LLM_MODEL = config.JUDGE_MODEL = "gemini-3.5-flash-lite"
if not os.environ.get("CODEQA_FALLBACK_MODELS"):
    config.FALLBACK_MODELS = ["gemini-3.1-flash-lite"]

LIVE = bool(config.LLM_API_KEY)
if not LIVE:
    log.warning("No API key set (GEMINI_API_KEY / LLM_API_KEY): only cached questions will work.")

cache = AnswerCache()
cache.load(SEED_CACHE)
limiter = RateLimiter(PER_HOUR)
daily = DailyCap(DAILY)
running = threading.BoundedSemaphore(CONCURRENCY)

app = FastAPI(title="codeQandA")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=300)


def _sse(kind: str, data) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"


def _stream(events):
    return StreamingResponse(events, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _replay(hit: dict):
    """Replay a cached answer's recorded steps, sped up, so the UI animates as if live."""
    prev = 0
    for e in hit["events"]:
        await asyncio.sleep(min(max(e.get("t_ms", 0) - prev, 0), 900) / 1000 / 4 + 0.05)
        prev = e.get("t_ms", 0)
        yield _sse("trace", e)
    yield _sse("answer", hit["answer"] | {"cached": True})


def _reject(message: str):
    raise HTTPException(429, message)


@app.post("/api/ask")
async def ask(body: AskRequest, request: Request):
    """Stream the agent's trace events live, then the final answer (Server-Sent Events).
    Repeat questions are replayed from cache; new ones pass the concurrency, per-visitor
    and daily limits first."""
    question = body.question.strip()
    if hit := cache.get(question):
        return _stream(_replay(hit))
    if not LIVE:
        raise HTTPException(503, "New questions are switched off on this server right now. "
                                 "The example questions still work (they're instant).")

    if not running.acquire(blocking=False):
        _reject("Two questions are already being answered. Try again in a minute, "
                "or pick an example question (those are instant).")
    try:
        ok, wait = limiter.allow(request.client.host if request.client else "unknown")
        if not ok:
            _reject(f"You've asked {PER_HOUR} new questions this hour. Try again in "
                    f"{max(1, round(wait / 60))} min, or pick an example question (those are instant).")
        if not daily.take():
            _reject("The demo has reached today's limit for new questions. Example questions still "
                    "work (they're instant), and new questions reopen tomorrow.")
    except HTTPException:
        running.release()
        raise

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(kind, data=None):  # called from the worker thread
        loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

    def work():
        events = []
        def on_event(e):
            events.append(e)
            emit("trace", e)
        try:
            tracer = Tracer(question, trace_dir=config.TRACE_DIR / "web", on_event=on_event)
            answer = run_agent(question, tracer=tracer).model_dump()
            cache.put(question, events, answer)
            emit("answer", answer)
        except AllModelsExhausted:
            emit("error", {"message": "The free API quota is used up for now. Try again later, "
                                      "or pick an example question (those are instant)."})
        except Exception:
            log.exception("agent failed for question %r", question)
            emit("error", {"message": "Something went wrong answering that question. Please try again."})
        finally:
            running.release()
            emit("done")

    loop.run_in_executor(None, work)

    async def stream():
        while True:
            kind, data = await queue.get()
            if kind == "done":
                break
            yield _sse(kind, data)

    return _stream(stream())


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
    repo = repo_root().name
    # The eval questions are about one specific repo; only offer them when that repo is loaded.
    examples = [q["question"] for q in spec["questions"]] if spec.get("repo") == repo else []
    return {"repo": repo, "repo_url": _collection().metadata.get("repo_url"),
            "description": os.environ.get("CODEQA_REPO_DESCRIPTION", ""),
            "chunks": _collection().count(), "model": config.LLM_MODEL, "examples": examples,
            "live": LIVE, "live_left_today": daily.remaining() if LIVE else 0, "per_hour": PER_HOUR}


@lru_cache(maxsize=1)
def _indexed_files() -> list[dict]:
    root = repo_root()
    paths = sorted({m["path"] for m in _collection().get(include=["metadatas"])["metadatas"]})
    return [{"path": p, "lines": len((root / p).read_text(encoding="utf-8", errors="replace").splitlines())}
            for p in paths]


@app.get("/api/files")
def files():
    """Every indexed file with its line count (the UI draws the repo from this)."""
    return _indexed_files()


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
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})
