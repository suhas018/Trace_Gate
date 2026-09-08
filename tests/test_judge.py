import pytest

from tracegate.judge import (
    JudgeResult,
    _summarize_result,
    build_judge_prompt,
    label_for,
    parse_judge_response,
)
from tracegate.metrics import ComponentScores
from tracegate.runner import RunResult
from tracegate.schema import Scenario, ToolCall, Trajectory
from tracegate.suite import SuiteReport, apply_judge


def scenario():
    return Scenario(
        id="sc",
        prompt="Cancel order 1234 for customer 42.",
        required_tools=["get_orders", "cancel_order"],
    )


def trace(calls=("get_orders", "cancel_order"), answer="order 1234 cancelled"):
    return Trajectory(
        scenario_id="sc",
        calls=[ToolCall(index=i, name=n) for i, n in enumerate(calls)],
        terminated=True,
        final_answer=answer,
    )


class FakeJudge:
    def __init__(self, result: JudgeResult):
        self.result = result

    def judge(self, scenario, trace):
        return self.result.model_copy(update={"scenario_id": scenario.id})


def test_parse_score_0_to_1():
    score, rationale = parse_judge_response('{"score": 0.85, "rationale": "good"}')
    assert score == pytest.approx(0.85)
    assert rationale == "good"


def test_parse_score_0_to_10_normalized():
    score, _ = parse_judge_response('{"score": 7.5, "rationale": "ok"}')
    assert score == pytest.approx(0.75)


def test_parse_markdown_fenced_json():
    score, rationale = parse_judge_response('```json\n{"score": 0.9, "rationale": "yes"}\n```')
    assert score == pytest.approx(0.9)
    assert rationale == "yes"


def test_parse_garbage_returns_none():
    assert parse_judge_response("no json here at all") == (None, "")


def test_label_thresholds():
    assert label_for(0.9) == "SATISFIED"
    assert label_for(0.6) == "PARTIAL"
    assert label_for(0.2) == "FAILED"
    assert label_for(None) == "UNAVAILABLE"


def test_judge_prompt_contains_task_and_answer():
    prompt = build_judge_prompt(scenario(), trace())
    assert "Cancel order 1234" in prompt
    assert "order 1234 cancelled" in prompt
    assert "get_orders" in prompt


def test_judge_prompt_grounded_on_tool_results():
    tr = Trajectory(
        scenario_id="sc",
        calls=[
            ToolCall(index=0, name="get_price", arguments={"ticker": "AAPL"}, result={"price": 150}),
            ToolCall(index=1, name="is_available", arguments={"ticker": "AAPL"}, result={"found": False}),
        ],
        terminated=True,
        final_answer="AAPL is in stock and costs $150.",
    )
    prompt = build_judge_prompt(scenario(), tr)
    assert "{\"price\": 150}" in prompt
    assert '"found": false' in prompt or "{'found': False}" in prompt
    assert "call -> result" in prompt


def test_judge_prompt_truncates_oversized_result():
    big = {"rows": ["x" * 2000]}
    tr = Trajectory(
        scenario_id="sc",
        calls=[ToolCall(index=0, name="query", result=big)],
        terminated=True,
        final_answer="done",
    )
    prompt = build_judge_prompt(scenario(), tr)
    assert "chars truncated" in prompt
    assert len(prompt) < 1200


def test_summarize_result_variants():
    assert _summarize_result(None) == "(no result)"
    assert _summarize_result("plain") == "plain"
    assert _summarize_result(42) == "42"
    assert _summarize_result(True) == "true"


def test_apply_judge_attaches_results_and_summary():
    sc = scenario()
    report = SuiteReport(
        results=[
            RunResult(
                scenario_id="sc",
                trace=trace(),
                scores=ComponentScores(
                    sequence=1, required_coverage=1, forbidden=1, termination=1, length=1, overall=1
                ),
            )
        ],
        run_id="x",
    )
    judge = FakeJudge(
        JudgeResult(scenario_id="sc", score=0.9, label="SATISFIED", judge_model="fake")
    )
    apply_judge(report, [sc], judge)
    assert len(report.judges) == 1
    assert report.judges[0].scenario_id == "sc"
    assert report.judges[0].score == pytest.approx(0.9)
    assert report.summary["judge_satisfaction_mean"] == pytest.approx(0.9)


def test_apply_judge_missing_trace_is_unavailable():
    sc = Scenario(id="other", prompt="p", min_calls=0)
    report = SuiteReport()
    judge = FakeJudge(
        JudgeResult(scenario_id="other", score=None, label="UNAVAILABLE", judge_model="fake")
    )
    apply_judge(report, [sc], judge)
    assert report.judges[0].label == "UNAVAILABLE"
