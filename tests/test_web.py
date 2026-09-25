"""Offline tests for the web API: the agent is replaced with a fake that logs trace events."""
import json

import pytest
from fastapi.testclient import TestClient

from codeqa import config
from codeqa.agent import AllModelsExhausted
from codeqa.schemas import AgentAnswer
from web import app as web


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    return TestClient(web.app)


def _events(resp):
    out = []
    for block in resp.text.strip().split("\n\n"):
        kind = block.split("\n")[0].removeprefix("event: ")
        out.append((kind, json.loads(block.split("\n")[1].removeprefix("data: "))))
    return out


def test_ask_streams_trace_then_answer(client, monkeypatch):
    def fake_agent(question, tracer):
        tracer.log("tool_call", name="search_code", args={"query": question}, result_summary="1 hit")
        tracer.close()
        return AgentAnswer(answer="42", cited_files=[{"path": "a.py", "lines": [1, 2]}], confidence="high")
    monkeypatch.setattr(web, "run_agent", fake_agent)
    events = _events(client.post("/api/ask", json={"question": "what is it?"}))
    assert [k for k, _ in events] == ["trace", "answer"]
    assert events[0][1]["name"] == "search_code" and events[1][1]["answer"] == "42"


def test_errors_are_friendly_and_hide_internals(client, monkeypatch):
    def boom(question, tracer):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(web, "run_agent", boom)
    kind, data = _events(client.post("/api/ask", json={"question": "hello?"}))[-1]
    assert kind == "error" and "secret" not in data["message"]

    def exhausted(question, tracer):
        raise AllModelsExhausted("x")
    monkeypatch.setattr(web, "run_agent", exhausted)
    assert "quota" in _events(client.post("/api/ask", json={"question": "hello?"}))[-1][1]["message"]


def test_question_length_is_validated(client):
    assert client.post("/api/ask", json={"question": "x" * 301}).status_code == 422
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_source_is_sandboxed(client):
    assert client.get("/api/source", params={"path": "../../etc/passwd", "start": 1, "end": 3}).status_code == 404


def test_result_meta_is_structured_and_never_raises():
    from codeqa.tracing import result_meta
    assert result_meta("read_file", {"path": "a.py", "lines": [1, 5], "total_lines": 9, "content": ""}) == \
        {"path": "a.py", "lines": [1, 5], "total_lines": 9}
    assert result_meta("search_code", {"results": [{"path": "a.py", "lines": [1, 2], "score": 0.5}]}) == \
        {"hits": [{"path": "a.py", "lines": [1, 2], "score": 0.5}]}
    assert result_meta("read_file", {"error": "x"}) is None
    assert result_meta("read_file", {}) is None


def test_collection_opens_once_under_concurrency(monkeypatch):
    import threading
    from codeqa import tools
    calls = []
    monkeypatch.setattr(tools, "_col", None)
    monkeypatch.setattr(tools, "get_collection", lambda: calls.append(1) or object())
    threads = [threading.Thread(target=tools._collection) for _ in range(16)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(calls) == 1
