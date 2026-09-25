"""Steps 2-4: the agent loop. The model sees the question + tool schemas, calls tools,
reads the results, and repeats; a final formatting call returns a validated AgentAnswer.
Every LLM call, tool call and the final answer is written to a JSONL trace.

Usage: python -m codeqa.agent "How are passwords hashed?"
"""
import json
import sys
import time

from openai import APIStatusError, OpenAI
from pydantic import ValidationError

from codeqa import config
from codeqa.schemas import RESPONSE_FORMAT, AgentAnswer
from codeqa.tools import TOOL_SCHEMAS, execute_tool
from codeqa.tracing import Tracer, print_trace, summarize_result

MAX_STEPS = 8

SYSTEM_PROMPT = """You answer questions about a software repository using the tools provided.
- Use search_code to find relevant code, then read_file to verify details before answering.
- Base every claim on code you actually retrieved. If you could not find it, say so.
- Paths are repo-relative, e.g. 'backend/app/main.py'.
- Before each tool call, state in one short sentence what you are looking for and why."""


def _client() -> OpenAI:
    # max_retries backs off on 429/503, which the free tier returns under load.
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY, max_retries=3)


_exhausted: set[str] = set()  # models that hit their quota (429) this process; skip them


def complete(client: OpenAI, **kwargs):
    """chat.completions.create with model fallback when a model is overloaded or out of quota."""
    models = [m for m in [config.LLM_MODEL] + config.FALLBACK_MODELS if m not in _exhausted]
    if not models:
        raise RuntimeError("All configured models are out of quota; try again later.")
    for i, model in enumerate(models):
        try:
            return client.chat.completions.create(model=model, **kwargs)
        except APIStatusError as e:
            if e.status_code not in (429, 503) or i == len(models) - 1:
                raise
            if e.status_code == 429:
                _exhausted.add(model)
            print(f"[{model} unavailable ({e.status_code}), falling back to {models[i + 1]}]", file=sys.stderr)


FINALIZE_PROMPT = """Now give your final answer as JSON.
- "answer": a direct, concise answer to the question.
- "cited_files": the files and line ranges that support the answer. Only cite ranges you saw
  in tool results; line numbers are 1-indexed and inclusive.
- "confidence": "high" if you verified the answer in the code, "medium" if partly verified
  or inferred, "low" if you could not find supporting code."""


class _Run:
    """Per-question state: the client, the tracer, and running token totals."""
    def __init__(self, client: OpenAI, tracer: Tracer):
        self.client, self.tracer, self.tokens = client, tracer, 0

    def llm(self, step, **kwargs):
        t0 = time.perf_counter()
        resp = complete(self.client, **kwargs)
        usage = getattr(resp, "usage", None)
        self.tokens += getattr(usage, "total_tokens", 0) or 0
        msg = resp.choices[0].message
        self.tracer.log(
            "llm_call", step=step, model=getattr(resp, "model", None),
            latency_ms=round((time.perf_counter() - t0) * 1000),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            # Text the model wrote alongside its tool calls = its stated reasoning for them.
            reasoning=msg.content if msg.tool_calls else None,
            n_tool_calls=len(msg.tool_calls or []))
        return msg


def _finalize(run: _Run, messages: list) -> AgentAnswer:
    """Formatting call: tools off, JSON schema on. Retries once with the validation error."""
    messages = messages + [{"role": "user", "content": FINALIZE_PROMPT}]
    for attempt in range(2):
        raw = run.llm(f"finalize-{attempt}", messages=messages, response_format=RESPONSE_FORMAT).content or ""
        try:
            return AgentAnswer.model_validate_json(raw)
        except ValidationError as e:
            run.tracer.log("validation_error", attempt=attempt, error=str(e), raw=raw)
            if attempt == 1:
                raise
            messages = messages + [{"role": "assistant", "content": raw},
                                   {"role": "user", "content": f"That JSON was invalid: {e}. Fix it."}]


def run_agent(question: str, client: OpenAI | None = None, tracer: Tracer | None = None) -> AgentAnswer:
    """Answer a question about the indexed repo. Always writes a trace (see tracer.path)."""
    tracer = tracer or Tracer(question)
    run = _Run(client or _client(), tracer)
    tracer.log("start", question=question, model=config.LLM_MODEL,
               fallbacks=config.FALLBACK_MODELS, max_steps=MAX_STEPS)
    try:
        answer, steps = _loop(run, question)
    except Exception as e:
        tracer.log("error", error=f"{type(e).__name__}: {e}")
        tracer.close()
        raise
    tracer.log("final", answer=answer.model_dump(), steps=steps, total_tokens=run.tokens)
    tracer.close()
    return answer


def _loop(run: _Run, question: str) -> tuple[AgentAnswer, int]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    step = 0
    for step in range(MAX_STEPS):
        msg = run.llm(step, messages=messages, tools=TOOL_SCHEMAS)
        # Append the assistant turn as-is (keeps provider extras such as Gemini's thought signatures).
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            break  # model is done exploring

        for call in msg.tool_calls:
            args, t0 = None, time.perf_counter()
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                result = {"error": f"invalid JSON arguments: {e}"}
            else:
                result = execute_tool(call.function.name, args)
            run.tracer.log("tool_call", step=step, name=call.function.name, args=args,
                           latency_ms=round((time.perf_counter() - t0) * 1000),
                           error=result.get("error"),
                           result_summary=summarize_result(call.function.name, result))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

    # Runs whether the model finished or hit the step limit, so there is always a structured answer.
    return _finalize(run, messages), step + 1


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What does this project do?"
    tracer = Tracer(q)
    try:
        run_agent(q, tracer=tracer)
    finally:
        print_trace(tracer.path)
        print(f"\ntrace: {tracer.path}")
