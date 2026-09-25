"""Steps 2-3: the agent loop. The model sees the question + tool schemas, calls tools,
reads the results, and repeats; a final formatting call returns a validated AgentAnswer.

Usage: python -m codeqa.agent "How are passwords hashed?"
"""
import json
import sys

from openai import APIStatusError, OpenAI
from pydantic import ValidationError

from codeqa import config
from codeqa.schemas import RESPONSE_FORMAT, AgentAnswer
from codeqa.tools import TOOL_SCHEMAS, execute_tool

MAX_STEPS = 8

SYSTEM_PROMPT = """You answer questions about a software repository using the tools provided.
- Use search_code to find relevant code, then read_file to verify details before answering.
- Base every claim on code you actually retrieved. If you could not find it, say so.
- Paths are repo-relative, e.g. 'backend/app/main.py'."""


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


def _finalize(client: OpenAI, messages: list) -> AgentAnswer:
    """Formatting call: tools off, JSON schema on. Retries once with the validation error."""
    messages = messages + [{"role": "user", "content": FINALIZE_PROMPT}]
    for attempt in range(2):
        resp = complete(client, messages=messages, response_format=RESPONSE_FORMAT)
        raw = resp.choices[0].message.content or ""
        try:
            return AgentAnswer.model_validate_json(raw)
        except ValidationError as e:
            if attempt == 1:
                raise
            messages = messages + [{"role": "assistant", "content": raw},
                                   {"role": "user", "content": f"That JSON was invalid: {e}. Fix it."}]


def run_agent(question: str, client: OpenAI | None = None, verbose: bool = False) -> AgentAnswer:
    client = client or _client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]

    for step in range(MAX_STEPS):
        resp = complete(client, messages=messages, tools=TOOL_SCHEMAS)
        msg = resp.choices[0].message
        # Append the assistant turn as-is (keeps provider extras such as Gemini's thought signatures).
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            break  # model is done exploring

        for call in msg.tool_calls:
            args = None
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                result = {"error": f"invalid JSON arguments: {e}"}
            else:
                result = execute_tool(call.function.name, args)
            if verbose:
                print(f"[step {step}] {call.function.name}({args}) -> "
                      f"{'ERROR ' + result['error'] if 'error' in result else 'ok'}", file=sys.stderr)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

    # Runs whether the model finished or hit the step limit, so there is always a structured answer.
    return _finalize(client, messages)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What does this project do?"
    print(run_agent(q, verbose=True).model_dump_json(indent=2))
