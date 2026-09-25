# codeQandA: a codebase Q&A agent with an eval harness

Ask a question about a codebase in plain English ("How are price signals computed?") and get back a
**structured, cited answer**: which files and line ranges support it, plus how confident the agent is.
An **evaluation harness** then measures whether those answers are actually correct and whether any
citation was made up.

## The problem

LLMs are good at explaining code they can see, and bad at admitting what they can't see. Ask one about
a real repo and it will happily invent a plausible-sounding file name or function. For code Q&A to be
useful, answers have to be **grounded** (built from code the agent actually retrieved), **verifiable**
(citations you can click through) and **measured** (you know how often it's right, rather than trusting
a demo). This project does all three, and the eval is the part that makes the other two claims checkable.

## How it works

```
index (once)            question
   │                       │
   ▼                       ▼
repo ──► chunk ──► embed ──► chromadb ◄── search_code ─┐
         (AST: function/class,           read_file   ├── agent loop (LLM + tool calling, ≤ 8 steps)
          token-budgeted)                list_files ─┘           │
                                                                 ▼
                                             final JSON (validated with Pydantic)
                                             { answer, cited_files[{path, lines}], confidence }
                                                                 │
                                    trace: every LLM call + tool call ──► traces/*.jsonl
```

1. **Indexing** ([codeqa/indexer.py](codeqa/indexer.py)): walks the repo, chunks Python files by
   function/class with `ast` (other files by line windows), splits any chunk that would exceed the
   embedding model's 256-token window, embeds locally (all-MiniLM-L6-v2) and stores the chunks in
   chromadb with `path`, `start_line`, `end_line` and `symbol` metadata.
2. **Tools** ([codeqa/tools.py](codeqa/tools.py)): `search_code(query)` does vector search;
   `read_file(path, start, end)` and `list_files(dir)` are sandboxed to the repo root. Tool errors are
   returned to the model as data so it can recover (e.g. search again after guessing a wrong path).
3. **Agent loop** ([codeqa/agent.py](codeqa/agent.py)): a plain loop with no framework. The model calls
   tools until it's done, then a final formatting call (tools off, JSON Schema on) produces the answer,
   which is validated with Pydantic and retried once on failure. Every run ends in valid JSON, even one
   that hits the step limit.
4. **Tracing** ([codeqa/tracing.py](codeqa/tracing.py)): each question writes a JSONL trace with every
   LLM call (which model answered, tokens, latency, stated reasoning), every tool call (args, result
   summary, errors) and the final answer.
5. **Eval harness** ([evals/](evals/)): 20 hand-checked questions about a real repo, scored
   automatically (see below).

Works with any OpenAI-compatible API. It defaults to Google Gemini's free tier and falls back to other
models when one is rate-limited.

## Web app

**Live demo:** _add your Render link here_ (free instance: the first visit after a quiet spell takes about a minute to wake up)

A FastAPI backend streams the agent's steps to the browser as they happen (Server-Sent Events). The
page draws every file in the repo as a bar, then marks where the agent searched, what it read and which
lines it cited. Each citation is checked against the repo live, so a made-up file shows up as
"not in repo". A second tab shows the before/after eval results question by question.

Running it publicly on a free API key needed some guardrails ([web/guard.py](web/guard.py)):
- **Answer cache.** Repeat questions replay their recorded steps at speed: instant, and no API calls.
  It's pre-seeded with the 20 eval questions, so every example works even when the quota is gone.
- **Limits.** 5 new questions per visitor per hour, 60 per day overall, 2 running at once. Cached
  questions never count.
- **Fails clearly.** Without an API key, or once the quota is used, new questions are turned away with
  a clear message and the examples keep working.

```bash
.venv/bin/uvicorn web.app:app --reload        # then open http://127.0.0.1:8000
```

**Deploying** (free, on [Render](https://render.com)): [render.yaml](render.yaml) defines the service
and the [Dockerfile](Dockerfile) builds it. At build time the image clones the target repo pinned to
the commit the eval measured and builds the index, so the published numbers, the cached answers and
the live index all describe the same code. The API key is a Render secret, never part of the image
([.dockerignore](.dockerignore) keeps `.env` out). Every push to `main` redeploys. The free instance
has 512 MB RAM (the app peaks around 380 MB) and sleeps after 15 idle minutes, so the first visit
after a quiet spell takes about a minute.

## Evaluation

Target repo: [Signal-stockwatchlist](https://github.com/h4rshini/Signal-stockwatchlist), a FastAPI +
React stock-watchlist app (~1,200 lines of backend Python). The 20 questions in
[evals/questions.yaml](evals/questions.yaml) cover five types: *locate* ("where is X?"), *config*
("what's the threshold?"), *explain* ("how does X work?"), *cross-file* (the answer spans modules) and
*negative* (the thing asked about doesn't exist). Reference answers were written from the **code**, not
the docs. One question (q18) deliberately targets a spot where the repo's docs and code disagree.

### What the metrics mean

| Metric | How it's computed | Why it matters |
|---|---|---|
| **Accuracy (LLM judge)** | A second LLM call compares the answer to the reference answer: correct / partially correct / incorrect | Handles paraphrasing; the closest thing to "would a human mark this right?" |
| **Accuracy (strict)** | Cited at least one expected file **and** mentioned every required keyword | Deterministic and free, with no LLM in the loop, so the judge can't be fooled by confident-sounding text |
| **Citation hit rate** | Share of answers citing at least one expected file (exact path match) | Is the agent pointing at the right code? |
| **Hallucinated citations** | Cited files that don't exist, or line ranges that are impossible (past end of file, reversed, 0-indexed) | The failure this project is built to catch |
| **Confidence vs correctness** | Accuracy split by the agent's own `high` / `medium` / `low` label | A useful agent is right more often when it says "high" |

### Results

Both runs use the same 20 questions, the same pinned model (`gemini-3.5-flash-lite`, no fallbacks)
for agent and judge, and the same grading code. The only change between them is the chunking fix.

| Metric | Baseline | After chunking fix |
|---|---|---|
| Accuracy (strict) | 80% (16/20) | **95%** (19/20) |
| Accuracy (LLM judge) | 85% (17/20) | **90%** (18/20) |
| Citation hit rate | 95% (18/19) | **100%** (19/19) |
| Hallucinated citations | 0 of 30 | **0 of 33** |
| Chunks indexed | 205 | 328 |

Full reports: [baseline](evals/results/20260925-160805_report.md), [after](evals/results/20260925-162417_report.md).

### What the eval found

**1. A retrieval bug hidden inside the chunker.** In the baseline, the agent answered "what's the
threshold?" questions with things like *"exceeds the configured threshold (settings.volume_ratio_threshold)"*:
it found the setting's name but never its value. The traces showed repeated searches for `SPY` and
`index_symbol` that never returned `config.py`. The cause: the local embedding model reads only the first
**256 tokens** of a chunk and silently drops the rest, and **54 of 205 chunks (26%)** were longer than
that. In `config.py`, embedding stopped around line 23, so every threshold after it was invisible to
search. The fix splits any chunk over 220 tokens by lines, using the embedder's own tokenizer, with a
small overlap. It resolved all four questions that depended on those values (q07, q08, q10, q11).

**2. The first LLM judge was too lenient.** With the original prompt, the judge scored the baseline
**100%**, including answers that left out the exact number the question asked for. Strict scoring caught
it (70% with the original keywords). After tightening the judge prompt ("omitting a specific value from
the reference is partially correct"), the two measures agreed on 95% of baseline answers. Saved answers
are re-graded with `--rescore`, so both runs are graded under identical rules.

**3. The agent is overconfident.** It said `high` confidence on all 40 answers across both runs,
including the ones it got partly wrong, so its self-reported confidence carries no signal yet. That's the
next thing to fix (e.g. derive confidence from whether cited lines were actually read with `read_file`).

**Remaining failures are grading noise, not agent errors:** q01 gave a correct answer without the
keyword `hash_password`; q03 and q07 are correct but the stricter judge wanted extra implementation
detail the question didn't ask for.

**Caveats:** one run per condition on 20 questions, so each question is 5 points and LLM
nondeterminism can move a question either way between runs. The judge is the same model as the agent.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "GEMINI_API_KEY=your-key" > .env       # free key: aistudio.google.com (never commit .env)
# optional: CODEQA_MODEL, CODEQA_RPM (default 12 req/min; 0 = no pacing)

.venv/bin/python -m codeqa.indexer /path/to/repo                      # 1. index
.venv/bin/python -m codeqa.agent "How are passwords hashed?"          # 2. ask (prints trace + JSON)
.venv/bin/python -m codeqa.tracing traces/<file>.jsonl                # 3. inspect a past run
.venv/bin/python -m evals.run_eval --model gemini-3.5-flash-lite      # 4. evaluate (resumable)
.venv/bin/python -m evals.run_eval --rescore <run_id>                 #    re-grade saved answers
.venv/bin/python -m pytest tests                                      # offline tests, no API calls
```

To use a different provider, set `LLM_BASE_URL`, `LLM_API_KEY` and `CODEQA_MODEL` in `.env` (any
OpenAI-compatible endpoint works).

## Design decisions

- **Final formatting call instead of JSON on every turn.** The model explores freely with tools; one last
  call with a JSON Schema produces the answer. That costs one extra call per question, and in exchange
  every run ends in valid output.
- **Validation checks shape, not truth.** Pydantic rejects malformed JSON, but a citation to a file that
  doesn't exist is kept and counted by the eval. Rejecting it would hide the hallucination.
- **Two accuracy measures.** An LLM judge alone can be fooled by fluent text; keyword matching alone
  penalizes valid paraphrases. Reporting both, plus how often they agree, shows where each is wrong.
- **The eval is resumable, and answering is separate from grading.** Results are appended per question,
  so `--resume <run_id>` picks up after a quota stop, and `--rescore <run_id>` re-grades saved answers when
  the scoring rules change, without re-running the agent.
- **Rate limits are handled rather than hoped away.** Requests are paced under the free tier's 15/minute
  limit; a per-minute 429 waits the server's `retryDelay` and retries the same model, and only a daily
  quota triggers fallback. Eval runs pin one model with no fallbacks, so before/after numbers compare
  like with like.

## Limitations

- The hallucination check catches citations that *can't* be right (missing file, impossible lines), not
  ones that exist but don't support the claim. The judge and keyword checks partly cover that.
- 20 questions on one repo is a small sample: each question is 5 percentage points. The ground truth was
  drafted with AI help from the code and then reviewed by hand.
- The judge is from the same model family as the agent, which can bias it toward the agent's phrasing.
- `gemini-flash-latest` is an alias whose underlying model changes over time; traces record the exact
  model that answered each call.

## Layout

```
codeqa/   indexer.py  tools.py  agent.py  schemas.py  tracing.py  config.py
evals/    questions.yaml  run_eval.py  scoring.py  results/   (reports + per-question JSONL)
web/      app.py (API)  guard.py (cache + limits)  static/index.html  seed_cache.json
Dockerfile, render.yaml   (free deploy on Render)
tests/    offline tests with a fake LLM client
```
