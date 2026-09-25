"""Offline tests for the agent loop: a scripted fake client stands in for the LLM."""
from types import SimpleNamespace as NS

from codeqa import agent


def _msg(content=None, tool_calls=None):
    m = NS(content=content, tool_calls=tool_calls)
    m.model_dump = lambda **_: {"role": "assistant", "content": content}
    return NS(choices=[NS(message=m)])


def _call(name, args, id="c1"):
    return NS(id=id, function=NS(name=name, arguments=args))


class FakeClient:
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, model, messages, **_):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def test_tool_result_is_fed_back_then_final_answer(monkeypatch):
    monkeypatch.setattr(agent, "execute_tool", lambda name, args: {"echo": [name, args]})
    client = FakeClient([_msg(tool_calls=[_call("list_files", '{"directory": "."}')]),
                         _msg(content="done")])
    assert agent.run_agent("q", client=client) == "done"
    tool_msg = client.seen[1][-1]
    assert tool_msg["role"] == "tool" and "list_files" in tool_msg["content"]


def test_bad_json_arguments_become_an_error_not_a_crash(monkeypatch):
    client = FakeClient([_msg(tool_calls=[_call("read_file", "{not json")]), _msg(content="ok")])
    assert agent.run_agent("q", client=client) == "ok"
    assert "invalid JSON" in client.seen[1][-1]["content"]


def test_step_limit(monkeypatch):
    monkeypatch.setattr(agent, "execute_tool", lambda name, args: {})
    loop = [_msg(tool_calls=[_call("list_files", '{"directory": "."}')])] * agent.MAX_STEPS
    assert "step limit" in agent.run_agent("q", client=FakeClient(loop))
