"""Tests for OpenAICompatibleJudge: a provider-neutral, stdlib-only judge.

Spins up a real local HTTP server in a thread (no API key, no network) so the
full request -> parse path is exercised, plus the fail-closed degradation cases.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tracegate.judge import OpenAICompatibleJudge
from tracegate.schema import Scenario, ToolCall, Trajectory


def scenario():
    return Scenario(
        id="sc",
        prompt="Cancel order 1234 for customer 42.",
        required_tools=["get_orders", "cancel_order"],
    )


def trace():
    return Trajectory(
        scenario_id="sc",
        calls=[ToolCall(index=i, name=n) for i, n in enumerate(["get_orders", "cancel_order"])],
        terminated=True,
        final_answer="order 1234 cancelled",
    )


def completion(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")


class FakeOpenAICompatibleServer:
    """Minimal OpenAI-compatible /chat/completions server, records requests."""

    def __init__(self, response_body: bytes, status: int = 200):
        self.response_body = response_body
        self.status = status
        self.requests: list[dict] = []
        self._httpd = HTTPServer(("127.0.0.1", 0), self._make_handler())
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                outer.requests.append(
                    {
                        "path": self.path,
                        "headers": {k: v for k, v in self.headers.items()},
                        "body": json.loads(raw),
                    }
                )
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(outer.response_body)

            def log_message(self, *args):  # silence
                pass

        return Handler

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()


def judge_url(server: FakeOpenAICompatibleServer) -> str:
    return f"http://127.0.0.1:{server.port}/v1"


def test_judge_end_to_end_via_http():
    content = '{"score": 0.9, "rationale": "goal met"}'
    with FakeOpenAICompatibleServer(completion(content)) as server:
        judge = OpenAICompatibleJudge(
            model="fake-model", api_key="sk-test", base_url=judge_url(server)
        )
        result = judge.judge(scenario(), trace())
    assert result.score == pytest.approx(0.9)
    assert result.label == "SATISFIED"
    assert result.rationale == "goal met"
    assert result.judge_model == "fake-model"


def test_judge_sends_auth_and_payload():
    with FakeOpenAICompatibleServer(completion('{"score": 0.9, "rationale": "ok"}')) as server:
        judge = OpenAICompatibleJudge(
            model="fake-model", api_key="sk-test", base_url=judge_url(server)
        )
        judge.judge(scenario(), trace())
    req = server.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"].get("Authorization") == "Bearer sk-test"
    assert req["body"]["model"] == "fake-model"
    assert req["body"]["temperature"] == 0.0
    assert "seed" not in req["body"]
    assert "You evaluate whether" in req["body"]["messages"][1]["content"]


def test_judge_omits_auth_header_when_no_key():
    with FakeOpenAICompatibleServer(completion('{"score": 1.0, "rationale": ""}')) as server:
        judge = OpenAICompatibleJudge(model="m", api_key=None, base_url=judge_url(server))
        judge.judge(scenario(), trace())
    assert "Authorization" not in server.requests[0]["headers"]


def test_judge_sends_seed_when_configured():
    with FakeOpenAICompatibleServer(completion('{"score": 1.0, "rationale": ""}')) as server:
        judge = OpenAICompatibleJudge(
            model="m", api_key=None, base_url=judge_url(server), seed=7
        )
        judge.judge(scenario(), trace())
    assert server.requests[0]["body"].get("seed") == 7


def test_judge_fails_closed_on_http_error():
    with FakeOpenAICompatibleServer(b'{"error": "boom"}', status=500) as server:
        judge = OpenAICompatibleJudge(model="m", api_key="k", base_url=judge_url(server))
        result = judge.judge(scenario(), trace())
    assert result.label == "UNAVAILABLE"
    assert result.score is None


def test_judge_fails_closed_on_garbage_response():
    with FakeOpenAICompatibleServer(b"not json at all") as server:
        judge = OpenAICompatibleJudge(model="m", api_key="k", base_url=judge_url(server))
        result = judge.judge(scenario(), trace())
    assert result.label == "UNAVAILABLE"
    assert result.score is None


def test_judge_fails_closed_on_connection_error():
    judge = OpenAICompatibleJudge(
        model="m", api_key="k", base_url="http://127.0.0.1:1/v1", timeout=1
    )
    result = judge.judge(scenario(), trace())
    assert result.label == "UNAVAILABLE"
    assert result.score is None


def test_judge_parses_10_point_scale():
    content = '{"score": 8, "rationale": "nearly there"}'
    with FakeOpenAICompatibleServer(completion(content)) as server:
        judge = OpenAICompatibleJudge(model="m", api_key="k", base_url=judge_url(server))
        result = judge.judge(scenario(), trace())
    assert result.score == pytest.approx(0.8)
    assert result.label == "SATISFIED"
