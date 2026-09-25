"""Offline tests for the agent loop: a scripted fake client stands in for the LLM."""
import json
from types import SimpleNamespace as NS

import pytest
from openai import APIStatusError

from codeqa import agent, config

FINAL = json.dumps({"answer": "done", "confidence": "high",
                    "cited_files": [{"path": "a.py", "lines": [1, 5]}]})


def _msg(content=None, tool_calls=None):
    m = NS(content=content, tool_calls=tool_calls)
    m.model_dump = lambda **_: {"role": "assistant", "content": content}
    return NS(choices=[NS(message=m)])


def _call(name, args, id="c1"):
    return NS(id=id, function=NS(name=name, arguments=args))


class FakeClient:
    """Returns scripted replies in order; an Exception in the script is raised instead."""
    def __init__(self, replies):
        self.replies, self.seen, self.models = list(replies), [], []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, model, messages, **_):
        self.seen.append(list(messages))
        self.models.append(model)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture(autouse=True)
def _reset_exhausted():
    agent._exhausted.clear()


def test_tool_result_is_fed_back_then_structured_answer(monkeypatch):
    monkeypatch.setattr(agent, "execute_tool", lambda name, args: {"echo": [name, args]})
    client = FakeClient([_msg(tool_calls=[_call("list_files", '{"directory": "."}')]),
                         _msg(content="I found it."), _msg(content=FINAL)])
    ans = agent.run_agent("q", client=client)
    assert ans.answer == "done" and ans.cited_files[0].lines == [1, 5]
    tool_msg = client.seen[1][-1]
    assert tool_msg["role"] == "tool" and "list_files" in tool_msg["content"]


def test_bad_json_arguments_become_an_error_not_a_crash():
    client = FakeClient([_msg(tool_calls=[_call("read_file", "{not json")]),
                         _msg(content="ok"), _msg(content=FINAL)])
    agent.run_agent("q", client=client, verbose=True)
    assert "invalid JSON" in client.seen[1][-1]["content"]


def test_step_limit_still_returns_structured_answer(monkeypatch):
    monkeypatch.setattr(agent, "execute_tool", lambda name, args: {})
    loop = [_msg(tool_calls=[_call("list_files", '{"directory": "."}')])] * agent.MAX_STEPS
    ans = agent.run_agent("q", client=FakeClient(loop + [_msg(content=FINAL)]))
    assert ans.answer == "done"


def test_invalid_final_json_is_retried_with_the_error():
    bad = json.dumps({"answer": "x", "cited_files": [{"path": "a.py", "lines": [1]}], "confidence": "sure"})
    client = FakeClient([_msg(content="done exploring"), _msg(content=bad), _msg(content=FINAL)])
    assert agent.run_agent("q", client=client).answer == "done"
    assert "invalid" in client.seen[-1][-1]["content"]


def _err(code):
    return APIStatusError("x", response=NS(status_code=code, headers={}, request=None), body=None)


def test_quota_exhausted_model_is_skipped_afterwards(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "main")
    monkeypatch.setattr(config, "FALLBACK_MODELS", ["backup"])
    client = FakeClient([_err(429), _msg(content="a"), _msg(content="b")])
    agent.complete(client, messages=[])
    agent.complete(client, messages=[])
    assert client.models == ["main", "backup", "backup"]
