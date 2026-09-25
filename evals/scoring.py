"""Step 5: deterministic scoring. Pure functions, no API calls."""
import re
from collections import defaultdict
from pathlib import Path


def _norm(path: str) -> str:
    return path.strip().removeprefix("./").lstrip("/")


def check_citation(path: str, lines: list[int], repo_root: Path) -> str | None:
    """Return a problem string if the citation points at something that doesn't exist, else None."""
    root = repo_root.resolve()
    p = (root / _norm(path)).resolve()
    if root not in p.parents or not p.is_file():
        return f"{path}: file does not exist"
    start, end = lines
    n = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
    if start < 1 or end < start or end > n:
        return f"{path}:{start}-{end}: invalid range (file has {n} lines)"
    return None


def citation_match(cited: list[dict], expected: list[str]) -> dict:
    cited_paths = {_norm(c["path"]) for c in cited}
    expected = {_norm(e) for e in expected}
    hits = cited_paths & expected
    return {"hit": bool(hits), "recall": len(hits) / len(expected) if expected else None}


def _contains(text: str, keyword: str) -> bool:
    # Whole-word, case-insensitive. Boundaries are letters/digits only (not \b), so "/seen" and "3.0"
    # work and "price" matches inside the identifier "price_move", but "no" doesn't match "note".
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])", text, re.IGNORECASE) is not None


def keyword_check(answer: str, must_mention: list[str]) -> dict:
    missing = [group for group in must_mention
               if not any(_contains(answer, alt.strip()) for alt in group.split("|"))]
    return {"ok": not missing, "missing": missing}


def score_answer(q: dict, answer: dict, repo_root: Path) -> dict:
    """Deterministic scores for one question. `answer` is an AgentAnswer dict."""
    problems = [p for c in answer["cited_files"]
                if (p := check_citation(c["path"], c["lines"], repo_root))]
    kw = keyword_check(answer["answer"], q.get("must_mention", []))
    if q["expected_files"]:
        cm = citation_match(answer["cited_files"], q["expected_files"])
    else:
        cm = {"hit": None, "recall": None}  # negative question: nothing to cite
    return {
        "citation_hit": cm["hit"],
        "citation_recall": cm["recall"],
        "hallucinations": problems,
        "keywords_ok": kw["ok"],
        "missing_keywords": kw["missing"],
        "strict_correct": kw["ok"] and cm["hit"] is not False,
    }


# ---------- aggregate report ----------

def _pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "n/a"


def summarize(rows: list[dict], use_judge: bool) -> dict:
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    cited = [r for r in ok if r["citation_hit"] is not None]
    s = {
        "questions": len(rows),
        "errors": len(rows) - n,
        "strict_correct": sum(r["strict_correct"] for r in ok),
        "citation_hits": sum(r["citation_hit"] for r in cited),
        "citation_questions": len(cited),
        "keywords_ok": sum(r["keywords_ok"] for r in ok),
        "hallucinated_citations": sum(len(r["hallucinations"]) for r in ok),
        "total_citations": sum(len(r["answer"]["cited_files"]) for r in ok),
        "questions_with_hallucination": sum(bool(r["hallucinations"]) for r in ok),
    }
    if use_judge:
        s["judge_correct"] = sum(r["judge"]["verdict"] == "correct" for r in ok)
        s["judge_partial"] = sum(r["judge"]["verdict"] == "partially_correct" for r in ok)
        s["judge_strict_agree"] = sum((r["judge"]["verdict"] == "correct") == r["strict_correct"] for r in ok)

    # Calibration: is "high" confidence actually right more often than "low"?
    correct_key = (lambda r: r["judge"]["verdict"] == "correct") if use_judge else (lambda r: r["strict_correct"])
    calib = defaultdict(lambda: [0, 0])
    for r in ok:
        c = calib[r["answer"]["confidence"]]
        c[0] += 1
        c[1] += correct_key(r)
    s["calibration"] = {k: {"n": v[0], "correct": v[1]} for k, v in calib.items()}
    s["calibration_basis"] = "judge" if use_judge else "strict"
    return s


def render_report(s: dict, rows: list[dict], meta: dict) -> str:
    n = s["questions"] - s["errors"]
    out = [f"# Eval report: {meta['run_id']}", "",
           f"Repo: `{meta['repo']}` ({meta['chunks']} chunks indexed) | Model: `{meta['model']}`"
           + (f" (fallbacks: {', '.join(meta['fallbacks'])})" if meta["fallbacks"] else " (pinned)")
           + (f" | Judge: `{meta['judge_model']}`" if meta.get("judge_model") else "")
           + f" | Questions: {s['questions']}"
           + (f" ({s['errors']} errored)" if s["errors"] else ""), "",
           "## Headline", "",
           "| Metric | Value | Meaning |", "|---|---|---|"]
    if "judge_correct" in s:
        out.append(f"| Accuracy (LLM judge) | **{_pct(s['judge_correct'], n)}** ({s['judge_correct']}/{n}) "
                   f"| Judge rated the answer fully correct vs the reference answer |")
    out += [
        f"| Accuracy (strict) | **{_pct(s['strict_correct'], n)}** ({s['strict_correct']}/{n}) "
        f"| Cited an expected file AND mentioned every required keyword |",
        f"| Citation hit rate | {_pct(s['citation_hits'], s['citation_questions'])} "
        f"({s['citation_hits']}/{s['citation_questions']}) | Cited at least one expected file |",
        f"| Hallucinated citations | **{s['hallucinated_citations']}** of {s['total_citations']} "
        f"(in {s['questions_with_hallucination']} questions) | Cited file doesn't exist or line range is impossible |",
    ]
    if "judge_correct" in s:
        out.append(f"| Judge/strict agreement | {_pct(s['judge_strict_agree'], n)} "
                   f"| How often the two accuracy measures agree |")

    out += ["", f"## Confidence vs correctness ({s['calibration_basis']})", "",
            "| Stated confidence | Answers | Correct | Accuracy |", "|---|---|---|---|"]
    for level in ["high", "medium", "low"]:
        c = s["calibration"].get(level, {"n": 0, "correct": 0})
        out.append(f"| {level} | {c['n']} | {c['correct']} | {_pct(c['correct'], c['n'])} |")
    out += ["", "A well-calibrated agent is right more often when it says \"high\" than when it says \"low\".", ""]

    out += ["## Per question", "",
            "| id | type | strict | judge | cites expected | hallucinations | confidence | missing keywords |",
            "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            out.append(f"| {r['id']} | {r['type']} | ERROR: {r['error'][:60]} | | | | | |")
            continue
        judge = r.get("judge", {}).get("verdict", "-")
        hit = {True: "yes", False: "no", None: "n/a"}[r["citation_hit"]]
        out.append(f"| {r['id']} | {r['type']} | {'✓' if r['strict_correct'] else '✗'} | {judge} | {hit} "
                   f"| {len(r['hallucinations'])} | {r['answer']['confidence']} "
                   f"| {', '.join(r['missing_keywords']) or '-'} |")
    return "\n".join(out) + "\n"
