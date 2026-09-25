"""Step 2: the agent loop. The model sees the question + tool schemas, calls tools,
reads the results, and repeats until it produces a final answer.

Usage: python -m codeqa.agent "How are passwords hashed?"
"""
import json
import sys

from openai import APIStatusError, OpenAI

from codeqa import config
from codeqa.tools import TOOL_SCHEMAS, execute_tool

MAX_STEPS = 8

SYSTEM_PROMPT = """You answer questions about a software repository using the tools provided.
- Use search_code to find relevant code, then read_file to verify details before answering.
- Base every claim on code you actually retrieved. If you could not find it, say so.
- Paths are repo-relative, e.g. 'backend/app/main.py'."""


def _client() -> OpenAI:
    # max_retries backs off on 429/503, which the free tier returns under load.
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY, max_retries=3)


def complete(client: OpenAI, **kwargs):
    """chat.completions.create with model fallback when a model stays overloaded."""
    models = [config.LLM_MODEL] + config.FALLBACK_MODELS
    for i, model in enumerate(models):
        try:
            return client.chat.completions.create(model=model, **kwargs)
        except APIStatusError as e:
            if e.status_code not in (429, 503) or i == len(models) - 1:
                raise
            print(f"[{model} unavailable ({e.status_code}), falling back to {models[i + 1]}]", file=sys.stderr)


def run_agent(question: str, client: OpenAI | None = None, verbose: bool = False) -> str:
    client = client or _client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]

    for step in range(MAX_STEPS):
        resp = complete(client, messages=messages, tools=TOOL_SCHEMAS)
        msg = resp.choices[0].message
        # Append the assistant turn as-is (keeps provider extras such as Gemini's thought signatures).
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or ""

        for call in msg.tool_calls:
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

    return "Stopped: reached the step limit without a final answer."


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What does this project do?"
    print(run_agent(q, verbose=True))
