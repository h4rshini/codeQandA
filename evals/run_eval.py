"""Step 5: run every ground-truth question through the agent and score it.

Usage:
  python -m evals.run_eval                      # new run, with LLM judge
  python -m evals.run_eval --no-judge           # deterministic scoring only (fewer API calls)
  python -m evals.run_eval --resume <run_id>    # continue a run that stopped (e.g. quota)
  python -m evals.run_eval --only q01,q07       # subset
  python -m evals.run_eval --model gemini-3.5-flash-lite   # pin one model, no fallbacks
  python -m evals.run_eval --rescore <run_id>   # re-grade saved answers (no agent calls)
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from codeqa import config
from codeqa.agent import AllModelsExhausted, _client, complete, run_agent
from codeqa.tools import _collection, repo_root
from codeqa.tracing import Tracer
from evals.scoring import render_report, score_answer, summarize

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


class Verdict(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"]
    reason: str


JUDGE_PROMPT = """You are grading an AI assistant's answer to a question about a codebase.

Question: {question}

Reference answer (ground truth, written by the codebase author):
{reference}

Assistant's answer:
{answer}

Grade the assistant's answer against the reference:
- "correct": states every key fact of the reference and contradicts none of them. Extra correct
  detail is fine.
- "partially_correct": gets the main idea right but misses or garbles a key fact. In particular, if
  the reference gives a specific value (a number, threshold, symbol, status code, file or function
  name) that answers part of the question, and the assistant omits it or only says it is "configured"
  or "set somewhere", that is partially_correct, not correct.
- "incorrect": misses the main point, or contradicts the reference.
Give a one-sentence reason naming any missing or wrong fact."""


def judge(client, q: dict, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=q["question"], reference=q["reference_answer"].strip(), answer=answer)
    resp = complete(client, model=config.JUDGE_MODEL, messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "verdict", "schema": Verdict.model_json_schema()}})
    return Verdict.model_validate_json(resp.choices[0].message.content).model_dump() | {"model": resp.model}


def grade(q: dict, answer: dict, root: Path, client, use_judge: bool) -> dict:
    """All scoring for one answer. Separate from answering, so saved answers can be re-graded."""
    scores = score_answer(q, answer, root)
    if use_judge:
        scores["judge"] = judge(client, q, answer["answer"])
    return scores


def write_report(run_id: str, rows: list[dict], use_judge: bool, meta: dict):
    if use_judge and any("error" not in r and "judge" not in r for r in rows):
        use_judge = False  # mixed run (e.g. resumed with --no-judge): fall back to strict-only report
    summary = summarize(rows, use_judge)
    report = render_report(summary, rows, meta)
    (RESULTS_DIR / f"{run_id}_report.md").write_text(report)
    (RESULTS_DIR / f"{run_id}_summary.json").write_text(json.dumps(summary | meta, indent=2))
    print("\n" + report)
    print(f"Saved: {RESULTS_DIR / f'{run_id}_report.md'}")


def rescore(run_id: str, questions: list[dict], root: Path, use_judge: bool):
    """Re-grade a finished run's saved answers with the current scoring code and judge prompt."""
    path = RESULTS_DIR / f"{run_id}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {q["id"]: q for q in questions}
    client = _client()
    for r in rows:
        if "error" in r or r["id"] not in by_id:
            continue
        r.pop("judge", None)
        r |= grade(by_id[r["id"]], r["answer"], root, client, use_judge)
        print(f"  {r['id']}: strict={'✓' if r['strict_correct'] else '✗'}"
              + (f" | judge={r['judge']['verdict']}" if use_judge else ""), flush=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    # Keep the original run's metadata (model, index size): the answers were produced under it.
    old = RESULTS_DIR / f"{run_id}_summary.json"
    meta = {k: v for k, v in json.loads(old.read_text()).items()
            if k in ("run_id", "repo", "model", "fallbacks", "chunks")} if old.exists() else {
        "run_id": run_id, "repo": root.name, "model": config.LLM_MODEL, "fallbacks": [], "chunks": "?"}
    meta["judge_model"] = config.JUDGE_MODEL
    write_report(run_id, [r for r in rows if r["id"] in by_id], use_judge, meta)


def load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows if "error" not in r}  # errored questions get retried on resume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.yaml")
    ap.add_argument("--resume", metavar="RUN_ID")
    ap.add_argument("--rescore", metavar="RUN_ID", help="re-grade saved answers without re-running the agent")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--only", help="comma-separated question ids")
    ap.add_argument("--model", help="pin agent + judge to this model and disable fallbacks "
                                    "(needed for fair before/after comparisons)")
    ap.add_argument("--delay", type=float, default=0, help="seconds to wait between questions (rate limits)")
    args = ap.parse_args()

    if args.model:
        config.LLM_MODEL = config.JUDGE_MODEL = args.model
        config.FALLBACK_MODELS = []

    spec = yaml.safe_load(args.questions.read_text())
    questions = spec["questions"]
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]

    root = repo_root()
    if spec.get("repo") and spec["repo"] != root.name:
        sys.exit(f"Questions are for '{spec['repo']}' but the index is of '{root.name}'. Re-index first.")

    use_judge = not args.no_judge
    if args.rescore:
        return rescore(args.rescore, questions, root, use_judge)

    run_id = args.resume or datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / f"{run_id}.jsonl"
    trace_dir = config.TRACE_DIR / f"eval-{run_id}"
    done = load_done(results_path)
    client = _client()

    # Rewrite the file with only the successful rows, then append as we go.
    with results_path.open("w") as f:
        for r in done.values():
            f.write(json.dumps(r) + "\n")

    stopped = False
    for i, q in enumerate(questions, 1):
        if q["id"] in done:
            continue
        print(f"[{i}/{len(questions)}] {q['id']}: {q['question']}", flush=True)
        t0 = time.perf_counter()
        tracer = Tracer(q["question"], trace_dir=trace_dir, run_id=q["id"])
        row = {"id": q["id"], "type": q["type"], "question": q["question"], "trace": str(tracer.path)}
        try:
            answer = run_agent(q["question"], client=client, tracer=tracer).model_dump()
            row |= {"answer": answer, **grade(q, answer, root, client, use_judge)}
        except AllModelsExhausted as e:
            print(f"   stopped: {e}")
            stopped = True
            break
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        row["seconds"] = round(time.perf_counter() - t0, 1)
        done[q["id"]] = row
        with results_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if "error" in row:
            print(f"   ERROR {row['error'][:120]}")
        else:
            verdict = f" | judge={row['judge']['verdict']}" if use_judge else ""
            print(f"   strict={'✓' if row['strict_correct'] else '✗'}{verdict} | "
                  f"confidence={row['answer']['confidence']} | hallucinations={len(row['hallucinations'])} | "
                  f"{row['seconds']}s")
        if args.delay:
            time.sleep(args.delay)

    rows = [done[q["id"]] for q in questions if q["id"] in done]
    if stopped or len(rows) < len(questions):
        print(f"\n{len(rows)}/{len(questions)} questions done. Resume later with:\n"
              f"  python -m evals.run_eval --resume {run_id}" + (" --no-judge" if args.no_judge else "")
              + (f" --model {args.model}" if args.model else "") + (f" --delay {args.delay:g}" if args.delay else ""))
    if not rows:
        return
    meta = {"run_id": run_id, "repo": root.name, "model": config.LLM_MODEL,
            "fallbacks": config.FALLBACK_MODELS, "chunks": _collection().count(),
            "judge_model": config.JUDGE_MODEL}
    write_report(run_id, rows, use_judge, meta)

if __name__ == "__main__":
    main()
