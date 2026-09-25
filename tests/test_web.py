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
